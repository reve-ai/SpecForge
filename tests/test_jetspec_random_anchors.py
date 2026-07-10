"""Verify random-anchor training construction (mask / positions / labels) — no GPU.

Monkeypatches _sample_anchors to fixed anchors and captures the tensors passed to the
draft, checking them against an independent nested-loop reference. This exercises the real
`forward_random_anchors` construction code, not a copy.

Run: PYTHONPATH=. python tests/test_jetspec_random_anchors.py
"""
import torch

from specforge.core.dflash import OnlineDFlashModel


def _make_model(block_size, mask_token_id, H=8, V=50):
    m = object.__new__(OnlineDFlashModel)
    torch.nn.Module.__init__(m)  # set up _parameters/_modules so submodule assignment works
    m.block_size = block_size
    m.mask_token_id = mask_token_id
    emb = torch.nn.Embedding(V, H)
    m.embed_tokens = emb
    m.lm_head = torch.nn.Linear(H, V, bias=False)
    captured = {}

    def fake_draft(position_ids, noise_embedding, target_hidden, attention_mask, is_causal=None, **kw):
        captured["position_ids"] = position_ids
        captured["attention_mask"] = attention_mask
        captured["is_causal"] = is_causal
        B, Q, _ = noise_embedding.shape
        return torch.zeros(B, Q, H)

    m.draft_model = fake_draft
    return m, captured


def _ref(anchors, valid, L, bs, causal):
    B = len(anchors); K = len(anchors[0]); Q = K * bs
    vis = [[[False] * (L + Q) for _ in range(Q)] for _ in range(B)]
    pos = [[0] * (L + Q) for _ in range(B)]
    for b in range(B):
        for c in range(L):
            pos[b][c] = c
        for j in range(K):
            a = anchors[b][j]
            for i in range(bs):
                row = j * bs + i
                pos[b][L + row] = a + i
                for c in range(L):
                    if c < a:
                        vis[b][row][c] = True
                for jj in range(K):
                    for ii in range(bs):
                        n = jj * bs + ii
                        ok = (jj == j) and (ii <= i if causal else True)
                        vis[b][row][L + n] = ok
                if not valid[b][j]:
                    vis[b][row][0] = True
    return torch.tensor(vis), torch.tensor(pos)


def test_random_anchor_construction():
    torch.manual_seed(0)
    B, L, bs, K, V = 2, 20, 4, 3, 50
    mask_id = 7
    for causal in (True, False):
        m, cap = _make_model(bs, mask_id, V=V)
        anchors = [[2, 8, 14], [5, 11, 0]]
        valid = [[True, True, True], [True, True, False]]
        m._sample_anchors = lambda lm, k: (
            torch.tensor(anchors), torch.tensor(valid, dtype=torch.bool)
        )
        input_ids = torch.randint(0, V, (B, L))
        loss_mask = torch.ones(B, L)
        hidden = torch.zeros(B, L, 8)
        m.forward_random_anchors(input_ids, hidden, loss_mask, num_anchors=K, causal=causal)

        vis_ref, pos_ref = _ref(anchors, valid, L, bs, causal)
        got_vis = cap["attention_mask"][:, 0] == 0.0  # [B,Q,L+Q]
        assert torch.equal(got_vis, vis_ref), f"mask mismatch (causal={causal})"
        assert torch.equal(cap["position_ids"], pos_ref), f"position mismatch (causal={causal})"
        assert cap["is_causal"] is False, "must pass is_causal=False (mask encodes causality)"
    print("ok: random-anchor mask + positions match reference (causal & bidir)")


def test_no_all_masked_rows():
    """Padded/invalid anchor rows must still attend to >=1 key (no NaN rows)."""
    B, L, bs, K, V = 1, 16, 4, 4, 50
    m, cap = _make_model(bs, 7, V=V)
    # only 1 valid anchor, 3 padded
    m._sample_anchors = lambda lm, k: (
        torch.tensor([[3, 0, 0, 0]]), torch.tensor([[True, False, False, False]])
    )
    m.forward_random_anchors(torch.randint(0, V, (B, L)), torch.zeros(B, L, 8),
                             torch.ones(B, L), num_anchors=K, causal=True)
    vis = cap["attention_mask"][:, 0] == 0.0
    assert (vis.sum(dim=-1) >= 1).all(), "every row must see at least one key"
    print("ok: no all-masked rows")


def test_sample_anchors_validity():
    """Sampled anchors must have a non-empty prefix and a loss token in their block."""
    m = object.__new__(OnlineDFlashModel)
    m.block_size = 4
    L = 40
    loss_mask = torch.zeros(1, L)
    loss_mask[0, 10:30] = 1  # only positions 10..29 are loss tokens
    anchors, valid = m._sample_anchors(loss_mask, num_anchors=50)
    for j in range(anchors.shape[1]):
        if not valid[0, j]:
            continue
        a = int(anchors[0, j])
        assert a >= 1 and a + m.block_size - 1 <= L - 1
        assert loss_mask[0, a + 1:a + m.block_size].sum() > 0
    print(f"ok: sampled {int(valid.sum())} valid anchors, all satisfy constraints")


if __name__ == "__main__":
    test_random_anchor_construction()
    test_no_all_masked_rows()
    test_sample_anchors_validity()
    print("ALL PASS")
