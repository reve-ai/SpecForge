"""JetSpec tree drafting + ancestor-masked verification (HF/SDPA reference).

This is the in-repo port of the JetSpec inference path (the public repo's
`tree/_core` + `tree/baselines/accum_logp` + `core/model_runner`), implemented on
plain HF + SDPA so it runs without the Triton serving engine. It is a *correctness*
reference for measuring acceptance length / tokens-per-forward, not an optimized server.

Pipeline per decode step:
  1. Draft: one causal draft-head forward over a block (anchor + mask tokens) → per-depth
     logits `(1, D, V)`, D = block_size-1.
  2. Build tree (accum_logp): per-depth top-k (shared across parents) expanded best-first by
     cumulative log-prob under a node budget. Root = anchor; each root→leaf path is a
     candidate continuation.
  3. Verify: one target forward over [committed_prefix ++ tree_nodes] with a 4D ANCESTOR
     mask — each tree node attends to the whole prefix + its ancestors only (not sibling
     branches) — and position_ids = prefix_len-1 + depth. Greedily accept the longest path
     matching the target's argmax (lossless under greedy).

Everything is cache-free (recompute prefix each step): simple and correct-by-construction,
which is what we want for a measurement harness. tree_width=1, budget>=D reduces to the
linear single-chain decode (asserted in tests).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Optional

import torch

from specforge.modeling.draft.dflash import extract_context_feature


@dataclass
class DraftTree:
    token_ids: List[int]          # node -> token id (node 0 = root/anchor)
    parent: List[int]             # node -> parent index (root parent = -1)
    depth: List[int]              # node -> depth (root = 0)
    cum_logprob: List[float]      # node -> cumulative path log-prob (root = 0.0)

    @property
    def num_nodes(self) -> int:
        return len(self.token_ids)

    def children_of(self, node: int) -> List[int]:
        return [i for i, p in enumerate(self.parent) if p == node]

    def is_ancestor(self, anc: int, node: int) -> bool:
        """True if `anc` is on the path root->node (inclusive of anc, exclusive of node)."""
        cur = self.parent[node]
        while cur != -1:
            if cur == anc:
                return True
            cur = self.parent[cur]
        return False


def build_accum_logp_tree(
    draft_logits: torch.Tensor,  # (1, D, V)
    root_token: int,
    tree_width: int,
    budget: int,
) -> DraftTree:
    """V0 accum_logp tree: per-depth top-k (shared across parents), best-first by
    cumulative log-prob, bounded by `budget` nodes (including the root)."""
    if draft_logits.dim() != 3 or draft_logits.shape[0] != 1:
        raise ValueError(f"draft_logits must be (1, D, V); got {tuple(draft_logits.shape)}")
    D = draft_logits.shape[1]
    log_probs = torch.log_softmax(draft_logits[0].float(), dim=-1)  # (D, V)
    k = min(tree_width, log_probs.shape[-1])
    topk_lp, topk_tok = torch.topk(log_probs, k, dim=-1)  # (D, k)
    topk_lp = topk_lp.tolist()
    topk_tok = topk_tok.tolist()

    tokens = [int(root_token)]
    parent = [-1]
    depth = [0]
    cum = [0.0]
    # max-heap by cumulative logprob (negate for min-heap); tie-break by insertion order
    counter = 0
    heap: list[tuple[float, int, int]] = [(0.0, counter, 0)]
    while heap and len(tokens) < budget:
        neg_cum, _, node = heapq.heappop(heap)
        d = depth[node]
        if d >= D:
            continue
        add = min(k, budget - len(tokens))
        for j in range(add):
            tokens.append(int(topk_tok[d][j]))
            parent.append(node)
            depth.append(d + 1)
            child_cum = -neg_cum + topk_lp[d][j]
            cum.append(child_cum)
            counter += 1
            heapq.heappush(heap, (-child_cum, counter, len(tokens) - 1))
    return DraftTree(token_ids=tokens, parent=parent, depth=depth, cum_logprob=cum)


def build_tree_verify_mask(prefix_len: int, tree: DraftTree, device, dtype):
    """Assemble (position_ids, additive 4D mask) for verifying the tree appended after a
    causal prefix of length `prefix_len`. Non-root tree nodes are laid out in node-index
    order 1..N-1 right after the prefix.

    Layout (length T = prefix_len + (N-1)):
      cols/rows [0, prefix_len)            -> committed prefix (causal among itself)
      cols/rows [prefix_len, T)            -> non-root tree nodes (col p+ (i-1) = node i)
    Visibility:
      prefix row r       : attends to prefix cols 0..r (causal), no tree cols.
      tree row (node i)  : attends to ALL prefix cols + tree cols of its ancestors + self.
    position_ids: prefix = 0..prefix_len-1; node i = (prefix_len - 1) + depth(i).
    """
    N = tree.num_nodes
    M = N - 1  # non-root nodes
    T = prefix_len + M
    visible = torch.zeros((T, T), dtype=torch.bool, device=device)

    # prefix causal block
    idx = torch.arange(prefix_len, device=device)
    visible[:prefix_len, :prefix_len] = idx.unsqueeze(1) >= idx.unsqueeze(0)

    # map node index (1..N-1) -> column/row in the flattened sequence
    def col(node: int) -> int:
        return prefix_len + (node - 1)

    for i in range(1, N):
        ri = col(i)
        visible[ri, :prefix_len] = True  # full prefix (includes the root/anchor)
        visible[ri, ri] = True  # self
        anc = tree.parent[i]
        while anc != -1 and anc != 0:  # walk non-root ancestors
            visible[ri, col(anc)] = True
            anc = tree.parent[anc]

    additive = torch.zeros((1, 1, T, T), dtype=dtype, device=device)
    additive.masked_fill_(~visible.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)

    position_ids = torch.empty((1, T), dtype=torch.long, device=device)
    position_ids[0, :prefix_len] = torch.arange(prefix_len, device=device)
    for i in range(1, N):
        position_ids[0, col(i)] = (prefix_len - 1) + tree.depth[i]
    return position_ids, additive


def _greedy_accept(tree: DraftTree, node_pred_token: List[int]):
    """Walk root->down taking the child whose token == the target's argmax at the parent.
    `node_pred_token[i]` = token the target predicts to FOLLOW node i.
    Returns (accepted_path_including_root, bonus_token)."""
    path = [0]
    cur = 0
    while True:
        pred = node_pred_token[cur]
        match = next((c for c in tree.children_of(cur) if tree.token_ids[c] == pred), None)
        if match is None:
            break
        path.append(match)
        cur = match
    bonus = node_pred_token[cur]  # target's prediction after the last accepted node
    return path, bonus


@torch.inference_mode()
def tree_spec_generate(
    draft,
    target,
    input_ids: torch.LongTensor,
    mask_token_id: int,
    max_new_tokens: int,
    stop_token_ids: Optional[List[int]],
    tree_width: int,
    budget: int,
):
    """Greedy tree speculative decode. Returns (output_ids, stats) where stats has
    per-step accepted counts so callers can compute acceptance length and tokens/forward."""
    draft.eval()
    target.eval()
    device = input_ids.device
    dtype = next(target.parameters()).dtype
    block_size = draft.block_size
    D = block_size - 1
    layer_ids = draft.target_layer_ids
    embed = target.get_input_embeddings()

    # ---- prefill (cache-free) ----
    out = target(input_ids, output_hidden_states=True, use_cache=False)
    H = extract_context_feature(out.hidden_states, layer_ids)  # (1, prefix_len, DL)
    anchor = int(torch.argmax(out.logits[0, -1]).item())
    committed = input_ids[0].tolist() + [anchor]  # positions 0..L_c-1, anchor at L_c-1

    accepted_per_step: List[int] = []  # tokens committed per verify forward (>=1)
    num_input = input_ids.shape[1]
    stop_set = set(stop_token_ids or [])

    while len(committed) - num_input < max_new_tokens:
        anchor_pos = len(committed) - 1  # position of the anchor (last committed token)
        # ---- draft (cache-free, causal): ctx = features of positions [0, anchor_pos) ----
        ctx = H[:, :anchor_pos, :]
        ctx_len = ctx.shape[1]
        block_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        block_ids[0, 0] = anchor
        noise_embedding = embed(block_ids)
        position_ids = torch.arange(ctx_len + block_size, device=device).unsqueeze(0)
        draft_hidden = draft(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=ctx,
            attention_mask=None,
            use_cache=False,
            is_causal=draft.causal_head,
        )
        draft_logits = target.lm_head(draft_hidden[:, -D:, :])  # (1, D, V)

        # ---- build tree ----
        tree = build_accum_logp_tree(draft_logits, anchor, tree_width, budget)

        # ---- verify (cache-free) over [committed ++ non-root tree nodes] ----
        prefix_len = len(committed)
        nonroot_tokens = tree.token_ids[1:]
        full_ids = torch.tensor([committed + nonroot_tokens], dtype=torch.long, device=device)
        position_ids_v, mask4d = build_tree_verify_mask(prefix_len, tree, device, dtype)
        vout = target(
            full_ids, position_ids=position_ids_v, attention_mask=mask4d,
            output_hidden_states=True, use_cache=False,
        )
        logits = vout.logits[0]  # (T, V)
        vH = extract_context_feature(vout.hidden_states, layer_ids)[0]  # (T, DL)

        # node -> row in flattened seq; root prediction comes from the last prefix position
        def row(node: int) -> int:
            return prefix_len - 1 if node == 0 else prefix_len + (node - 1)

        node_pred = [int(torch.argmax(logits[row(n)]).item()) for n in range(tree.num_nodes)]
        path, bonus = _greedy_accept(tree, node_pred)

        accepted_nodes = path[1:]  # non-root nodes on the accepted path
        # append accepted tokens + their target features, then the bonus (next anchor)
        for n in accepted_nodes:
            committed.append(tree.token_ids[n])
            H = torch.cat([H, vH[row(n)].view(1, 1, -1)], dim=1)
        committed.append(bonus)
        # committed tokens this step = accepted_nodes + bonus
        accepted_per_step.append(len(accepted_nodes) + 1)
        anchor = bonus

        if stop_set and any(t in stop_set for t in committed[num_input:]):
            break

    output_ids = torch.tensor([committed], dtype=torch.long, device=device)
    stats = {
        "accepted_per_step": accepted_per_step,
        "total_committed": sum(accepted_per_step),
        "num_verifies": len(accepted_per_step),
    }
    return output_ids, stats
