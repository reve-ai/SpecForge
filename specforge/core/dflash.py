# coding=utf-8
"""DFlash Training Wrapper."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.dflash import DFlashDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None


def dflash_block_visible(q_idx, kv_idx, q_len, block_size, causal):
    """Single source of truth for the DFlash/JetSpec block-attention pattern.

    Keys are laid out as ``[context (0..q_len-1) ; noise (q_len..2*q_len-1)]``.
    Returns a boolean (tensor or scalar) — True iff query ``q_idx`` may attend to
    key ``kv_idx``:
      - a context key (its block ``b``) is visible to a query in block ``q`` iff
        ``b < q`` (strictly earlier blocks);
      - a noise key is visible iff it is in the same block as the query, and — when
        ``causal`` (JetSpec) — its intra-block offset is ``<=`` the query's offset.

    Works elementwise for torch tensors, so both the flex-attention mask_fn
    (scalar indices) and the dense path (index grids) call this identical logic.
    """
    is_ctx = kv_idx < q_len
    q_block = q_idx // block_size
    k_block_ctx = kv_idx // block_size
    k_block_noise = (kv_idx - q_len) // block_size
    ctx_visible = is_ctx & (k_block_ctx < q_block)
    noise_visible = (~is_ctx) & (k_block_noise == q_block)
    if causal:
        q_intra = q_idx % block_size
        k_intra_noise = (kv_idx - q_len) % block_size
        noise_visible = noise_visible & (k_intra_noise <= q_intra)
    return ctx_visible | noise_visible


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with block-wise CE loss."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        head_type: str = "bidirectional",
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        # head_type selects the within-block (noise) attention pattern:
        #   "bidirectional" -> DFlash: every noise position sees the whole block (default).
        #   "causal"        -> JetSpec: noise position i sees only noise positions j <= i,
        #                      restoring the target's autoregressive factorization.
        # The context (prefix) half of the mask is identical for both.
        if head_type not in ("bidirectional", "causal"):
            raise ValueError(
                f"head_type must be 'bidirectional' or 'causal', got {head_type!r}"
            )
        self.head_type = head_type
        # Soft-label distillation (temperature-scaled KL to the frozen target's next-token
        # distribution). distill_alpha weights the hard CE term: loss = alpha*CE + (1-alpha)*KL*T^2.
        self.distill = False
        self.distill_temp = 1.0
        self.distill_alpha = 0.0
        # Anchor scheme: "tiling" = contiguous blocks (DFlash default); "random" = JetSpec-style
        # random anchor sampling (up to num_anchors blocks placed at random positions/example).
        self.anchor_mode = "tiling"
        self.num_anchors = 128

        # Cache for BlockMask
        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None
        self._cached_num_heads: Optional[int] = None

    def _distill_loss(self, student_logits, teacher_hidden, chunk: int = 1024):
        """KL(student || teacher) at active positions, temperature-scaled. teacher_hidden is
        the target's FINAL hidden at the position that predicts each label (i.e. label_pos-1);
        the target's own lm_head turns it into the teacher distribution.

        Chunked over active positions: the teacher softmax over the full vocab is a big
        transient, so we materialize it `chunk` rows at a time (the no_grad teacher tensors
        free between chunks) to keep peak memory within budget at batch size 4.
        """
        T = self.distill_temp
        n = student_logits.shape[0]
        total = student_logits.new_zeros(())
        for i in range(0, n, chunk):
            s = student_logits[i : i + chunk]
            with torch.no_grad():
                tsoft = F.softmax(self.lm_head(teacher_hidden[i : i + chunk]).float() / T, dim=-1)
            slogp = F.log_softmax(s.float() / T, dim=-1)
            total = total + F.kl_div(slogp, tsoft, reduction="sum") * (T * T)
        return total / max(n, 1)

    def prepare_noise_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prepare noise input: first token of each block is real, rest are MASK."""
        seq_len = input_ids.shape[1]
        device = input_ids.device

        positions = torch.arange(seq_len, device=device)
        is_block_start = (positions % self.block_size) == 0

        noise_input_ids = torch.full_like(input_ids, self.mask_token_id)
        noise_input_ids[:, is_block_start] = input_ids[:, is_block_start]

        return noise_input_ids

    def _get_or_create_block_mask(
        self, bsz: int, num_heads: int, q_len: int, kv_len: int, device: torch.device
    ) -> "BlockMask":
        """Get cached BlockMask or create a new one."""
        if (
            self._cached_block_mask is not None
            and self._cached_seq_len == q_len
            and self._cached_bsz == bsz
            and self._cached_num_heads == num_heads
        ):
            return self._cached_block_mask

        block_size = self.block_size
        causal = self.head_type == "causal"

        def dflash_mask_fn(b, h, q_idx, kv_idx):
            return dflash_block_visible(q_idx, kv_idx, q_len, block_size, causal)

        block_mask = create_block_mask(
            dflash_mask_fn,
            B=bsz,
            H=num_heads,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

        self._cached_block_mask = block_mask
        self._cached_seq_len = q_len
        self._cached_bsz = bsz
        self._cached_num_heads = num_heads

        return block_mask

    def _create_parallel_attention_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create [L, 2L] attention mask for parallel training.
        - Left half (ctx): Q can see K_ctx if K's block < Q's block
        - Right half (noise): Q can see K_noise if same block. The within-block
          pattern depends on self.head_type:
            "bidirectional" (DFlash): full block visibility.
            "causal" (JetSpec): only K offsets <= Q offset within the block.
        """
        q_idx = torch.arange(seq_len, device=device).unsqueeze(1)  # [L, 1]
        kv_idx = torch.arange(2 * seq_len, device=device).unsqueeze(0)  # [1, 2L]
        full_mask_bool = dflash_block_visible(
            q_idx, kv_idx, seq_len, self.block_size, self.head_type == "causal"
        )  # [L, 2L]
        full_mask = torch.zeros_like(full_mask_bool, dtype=torch.float32)
        full_mask.masked_fill_(~full_mask_bool, torch.finfo(torch.float32).min)

        return full_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel block-wise training forward pass (dispatches on anchor_mode)."""
        if getattr(self, "anchor_mode", "tiling") == "random":
            return self.forward_random_anchors(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                num_anchors=self.num_anchors,
                causal=(self.head_type == "causal"),
                target_last_hidden=target_last_hidden,
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Truncate to multiple of block_size
        n_blocks = seq_len // self.block_size
        effective_len = n_blocks * self.block_size
        input_ids = input_ids[:, :effective_len]
        hidden_states = hidden_states[:, :effective_len, :]
        loss_mask = loss_mask[:, :effective_len]
        attention_mask = attention_mask[:, :effective_len]
        if target_last_hidden is not None:
            target_last_hidden = target_last_hidden[:, :effective_len, :]

        # Prepare inputs
        noise_input_ids = self.prepare_noise_input(input_ids)
        noise_embedding = self.embed_tokens(noise_input_ids)

        # Position IDs: [ctx_pos, noise_pos] both 0..L-1
        pos_seq = torch.arange(effective_len, device=device)
        position_ids = torch.cat([pos_seq, pos_seq], dim=0).unsqueeze(0).expand(bsz, -1)

        # Construct attention mask
        if (
            self.attention_backend == "flex_attention"
            and FLEX_ATTENTION_AVAILABLE
            and create_block_mask is not None
        ):
            num_heads = self.draft_model.config.num_attention_heads
            dflash_attn_mask = self._get_or_create_block_mask(
                bsz=bsz,
                num_heads=num_heads,
                q_len=effective_len,
                kv_len=effective_len * 2,
                device=device,
            )
        else:
            dflash_attn_mask = self._create_parallel_attention_mask(
                effective_len, device
            )
            dflash_attn_mask = dflash_attn_mask.to(dtype=hidden_states.dtype)
            dflash_attn_mask = (
                dflash_attn_mask.unsqueeze(0).unsqueeze(0).expand(bsz, -1, -1, -1)
            )

        # Forward pass
        hidden = self.draft_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )

        # Compute loss (skip block 0 and block starts)
        dflash_loss_mask_base = create_dflash_loss_mask(
            effective_len, self.block_size, device
        )
        combined_mask = loss_mask * dflash_loss_mask_base.unsqueeze(0)

        logits = self.lm_head(hidden)

        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = input_ids.reshape(-1)
        mask_flat = combined_mask.reshape(-1)

        active_indices = mask_flat > 0.5
        active_logits = logits_flat[active_indices]
        active_labels = labels_flat[active_indices]

        ce_loss = F.cross_entropy(active_logits, active_labels)
        loss = ce_loss
        if self.distill and target_last_hidden is not None:
            # teacher for label at flat position p is the target's dist from last_hidden[p-1].
            active_pos = active_indices.nonzero(as_tuple=True)[0]  # flat indices p
            tlh_flat = target_last_hidden.reshape(-1, target_last_hidden.size(-1))
            teacher_hidden = tlh_flat[active_pos - 1]
            distill = self._distill_loss(active_logits, teacher_hidden)
            loss = self.distill_alpha * ce_loss + (1.0 - self.distill_alpha) * distill

        with torch.no_grad():
            preds = active_logits.argmax(dim=-1)
            correct = (preds == active_labels).float().sum()
            total = active_labels.numel()
            accuracy = correct / total

        return loss, accuracy

    def _sample_anchors(self, loss_mask: torch.Tensor, num_anchors: int):
        """Sample up to `num_anchors` random anchor positions per example.

        A valid anchor `a` needs a non-empty prefix (a>=1), room for a full block
        (a+block_size-1 <= L-1), and at least one loss token to predict in
        (a, a+block_size). Returns anchors [B, K] and valid [B, K] (padded slots False).
        """
        B, L = loss_mask.shape
        bs = self.block_size
        device = loss_mask.device
        anchors = torch.zeros(B, num_anchors, dtype=torch.long, device=device)
        valid = torch.zeros(B, num_anchors, dtype=torch.bool, device=device)
        max_a = L - bs
        if max_a < 1:
            return anchors, valid
        for b in range(B):
            lm = loss_mask[b].float()
            csum = torch.cat([torch.zeros(1, device=device), lm.cumsum(0)])
            cand = torch.arange(1, max_a + 1, device=device)
            has_loss = (csum[cand + bs] - csum[cand + 1]) > 0  # loss in (a+1 .. a+bs-1)
            cand = cand[has_loss]
            if cand.numel() == 0:
                continue
            k = min(num_anchors, cand.numel())
            perm = torch.randperm(cand.numel(), device=device)[:k]
            anchors[b, :k] = cand[perm]
            valid[b, :k] = True
        return anchors, valid

    def forward_random_anchors(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        num_anchors: int,
        causal: bool = True,
        target_last_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """JetSpec-style training over K randomly-placed anchors (vs contiguous tiling).

        The full real sequence is the shared context K/V; each anchor contributes a
        block_size noise segment (anchor real, rest MASK) that attends only to ctx[0:a)
        and (causally) to earlier offsets in its own block. Matches inference, where
        blocks start at arbitrary accepted positions.
        """
        B, L = input_ids.shape
        device = input_ids.device
        bs = self.block_size
        dtype = hidden_states.dtype

        anchors, valid = self._sample_anchors(loss_mask, num_anchors)  # [B,K],[B,K]
        K = num_anchors
        Q = K * bs

        # per-query-row anchor-index (j) and intra-block offset (i)
        q_block = (torch.arange(Q, device=device) // bs)  # [Q]
        q_off = (torch.arange(Q, device=device) % bs)  # [Q]
        anchor_of_col = anchors.gather(1, q_block.unsqueeze(0).expand(B, -1))  # [B,Q]
        seq_pos = anchor_of_col + q_off.unsqueeze(0)  # [B,Q] sequence position of each noise slot

        # noise tokens: offset 0 = real anchor token, rest = MASK
        noise_ids = torch.full((B, Q), self.mask_token_id, dtype=torch.long, device=device)
        is_anchor = (q_off == 0).unsqueeze(0).expand(B, -1)
        noise_ids[is_anchor] = input_ids.gather(1, seq_pos)[is_anchor]
        noise_embedding = self.embed_tokens(noise_ids)  # [B,Q,H]

        # position ids: [ctx 0..L-1 ; noise seq positions]
        ctx_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # [B,L]
        position_ids = torch.cat([ctx_pos, seq_pos], dim=1)  # [B, L+Q]

        # visibility mask [B, Q, L+Q]
        ctx_cols = torch.arange(L, device=device)
        vis_ctx = ctx_cols.view(1, 1, L) < anchor_of_col.unsqueeze(-1)  # [B,Q,L]
        k_block = (torch.arange(Q, device=device) // bs)
        k_off = (torch.arange(Q, device=device) % bs)
        same_block = k_block.view(1, Q) == q_block.view(Q, 1)  # [Q,Q]
        if causal:
            within = k_off.view(1, Q) <= q_off.view(Q, 1)
            vis_noise = (same_block & within)
        else:
            vis_noise = same_block
        vis_noise = vis_noise.unsqueeze(0).expand(B, -1, -1)  # [B,Q,Q]
        visible = torch.cat([vis_ctx, vis_noise], dim=2)  # [B,Q,L+Q]
        # invalid (padded) rows: let them see ctx col 0 to avoid all-masked NaN rows
        invalid_row = ~valid.gather(1, q_block.unsqueeze(0).expand(B, -1))  # [B,Q]
        visible[:, :, 0] = visible[:, :, 0] | invalid_row
        attn_mask = torch.zeros(B, 1, Q, L + Q, dtype=dtype, device=device)
        attn_mask.masked_fill_(~visible.unsqueeze(1), torch.finfo(dtype).min)

        hidden = self.draft_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=attn_mask,
            is_causal=False,  # causality is fully encoded in attn_mask above
        )
        logits = self.lm_head(hidden)  # [B,Q,V]

        # labels/weights: predict offsets 1..bs-1 of valid blocks, weighted by loss_mask
        labels = input_ids.gather(1, seq_pos)  # [B,Q] token at each noise seq position
        weight = (
            loss_mask.gather(1, seq_pos).float()
            * (q_off >= 1).float().unsqueeze(0)
            * valid.gather(1, q_block.unsqueeze(0).expand(B, -1)).float()
        )  # [B,Q]

        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = labels.reshape(-1)
        active = weight.reshape(-1) > 0.5
        active_logits = logits_flat[active]
        active_labels = labels_flat[active]
        ce_loss = F.cross_entropy(active_logits, active_labels)
        loss = ce_loss
        if self.distill and target_last_hidden is not None:
            # teacher for label at sequence position p = target dist from last_hidden[p-1]
            batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, Q)
            teacher_pos = (seq_pos - 1).clamp(min=0)
            bidx = batch_idx.reshape(-1)[active]
            pidx = teacher_pos.reshape(-1)[active]
            teacher_hidden = target_last_hidden[bidx, pidx]  # [n_active, H]
            distill = self._distill_loss(active_logits, teacher_hidden)
            loss = self.distill_alpha * ce_loss + (1.0 - self.distill_alpha) * distill
        with torch.no_grad():
            accuracy = (active_logits.argmax(-1) == active_labels).float().mean()
        return loss, accuracy


def create_dflash_loss_mask(
    seq_len: int, block_size: int, device: torch.device
) -> torch.Tensor:
    """Create DFlash loss mask: excludes block 0 and first position of each block."""
    positions = torch.arange(seq_len, device=device)
    block_ids = positions // block_size

    is_block_0 = block_ids == 0
    is_block_start = (positions % block_size) == 0

    valid_mask = ~is_block_0 & ~is_block_start
    return valid_mask.float()
