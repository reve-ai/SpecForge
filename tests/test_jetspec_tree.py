"""Tests for JetSpec tree construction + ancestor verification mask (no GPU).

Validates: accum_logp tree shape (width-1 == linear chain; width-k branching; budget cap),
cumulative-logprob best-first ordering, and that the verify mask encodes exactly
"prefix + ancestors + self" (siblings/cousins masked out).

Run: PYTHONPATH=. python tests/test_jetspec_tree.py
"""
import torch

from specforge.modeling.draft.jetspec_tree import (
    DraftTree,
    build_accum_logp_tree,
    build_tree_verify_mask,
)


def _logits_with_known_topk(D, V):
    """Construct (1, D, V) logits whose per-depth ordering is token v=0 best, then 1, ..."""
    base = torch.zeros(1, D, V)
    for d in range(D):
        for v in range(V):
            base[0, d, v] = (V - v)  # token 0 highest logit, descending
    return base


def test_width1_is_linear_chain():
    D, V = 4, 6
    logits = _logits_with_known_topk(D, V)
    tree = build_accum_logp_tree(logits, root_token=99, tree_width=1, budget=1000)
    # chain: root + one node per depth
    assert tree.num_nodes == D + 1, tree.num_nodes
    assert tree.token_ids[0] == 99
    for i in range(1, tree.num_nodes):
        assert tree.parent[i] == i - 1
        assert tree.depth[i] == i
        assert tree.token_ids[i] == 0  # argmax token each depth
    # exactly one leaf
    leaves = [n for n in range(tree.num_nodes) if not tree.children_of(n)]
    assert leaves == [D]


def test_width2_branching_and_budget():
    D, V = 3, 6
    logits = _logits_with_known_topk(D, V)
    # unbounded-ish budget: root has 2 children, each has 2, etc. -> 1+2+4+8 = 15
    tree = build_accum_logp_tree(logits, root_token=7, tree_width=2, budget=1000)
    assert tree.num_nodes == 1 + 2 + 4 + 8, tree.num_nodes
    # depth-1 children are the top-2 tokens {0,1}
    d1 = tree.children_of(0)
    assert sorted(tree.token_ids[c] for c in d1) == [0, 1]
    # budget cap honored
    capped = build_accum_logp_tree(logits, root_token=7, tree_width=2, budget=5)
    assert capped.num_nodes == 5


def test_best_first_cumulative_logprob():
    D, V = 3, 6
    logits = _logits_with_known_topk(D, V)
    tree = build_accum_logp_tree(logits, root_token=7, tree_width=2, budget=6)
    # children cum_logprob must be <= parent cum_logprob (logprobs are negative additions)
    for i in range(1, tree.num_nodes):
        assert tree.cum_logprob[i] <= tree.cum_logprob[tree.parent[i]] + 1e-6


def test_is_ancestor():
    # root(0) -> 1 -> 3 ; root -> 2
    tree = DraftTree(token_ids=[0, 10, 20, 30], parent=[-1, 0, 0, 1],
                     depth=[0, 1, 1, 2], cum_logprob=[0, -1, -2, -3])
    assert tree.is_ancestor(0, 3) and tree.is_ancestor(1, 3)
    assert not tree.is_ancestor(2, 3)  # sibling branch
    assert not tree.is_ancestor(3, 1)


def test_verify_mask_prefix_plus_ancestors_only():
    # root(0) -> 1, 2 (depth1); 1 -> 3 (depth2)
    tree = DraftTree(token_ids=[0, 11, 12, 13], parent=[-1, 0, 0, 1],
                     depth=[0, 1, 1, 2], cum_logprob=[0, -1, -1, -2])
    prefix_len = 2
    pos, mask = build_tree_verify_mask(prefix_len, tree, torch.device("cpu"), torch.float32)
    vis = mask[0, 0] == 0.0  # True where attention is allowed
    T = prefix_len + (tree.num_nodes - 1)
    assert vis.shape == (T, T)

    # prefix rows: causal among prefix, nothing into tree cols
    assert vis[0, 0] and not vis[0, 1]
    assert vis[1, 0] and vis[1, 1]
    assert not vis[0, 2:].any() and not vis[1, 2:].any()

    # column index of each non-root node
    c1, c2, c3 = prefix_len + 0, prefix_len + 1, prefix_len + 2
    # node1 (row c1): prefix + self, no other tree nodes
    assert vis[c1, 0] and vis[c1, 1] and vis[c1, c1]
    assert not vis[c1, c2] and not vis[c1, c3]
    # node2 (row c2): prefix + self only (sibling of node1)
    assert vis[c2, 0] and vis[c2, 1] and vis[c2, c2]
    assert not vis[c2, c1] and not vis[c2, c3]
    # node3 (row c3): prefix + ancestor node1 + self; NOT sibling-branch node2
    assert vis[c3, 0] and vis[c3, 1] and vis[c3, c1] and vis[c3, c3]
    assert not vis[c3, c2]

    # position ids: prefix 0,1 ; node1,node2 at depth1 -> prefix_len-1+1 = 2 ; node3 -> 3
    assert pos[0].tolist() == [0, 1, 2, 2, 3]


if __name__ == "__main__":
    test_width1_is_linear_chain(); print("ok: width-1 == linear chain")
    test_width2_branching_and_budget(); print("ok: width-2 branching + budget cap")
    test_best_first_cumulative_logprob(); print("ok: best-first cumulative logprob")
    test_is_ancestor(); print("ok: is_ancestor")
    test_verify_mask_prefix_plus_ancestors_only(); print("ok: verify mask = prefix + ancestors + self")
    print("ALL PASS")
