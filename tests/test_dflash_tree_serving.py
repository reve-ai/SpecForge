# coding=utf-8
"""Parity tests: serving-side tree/mask/retrieve arrays == jetspec_tree reference."""
import torch

from specforge.modeling.draft.jetspec_tree import build_accum_logp_tree
from specforge.serving.dflash_tree import (
    ancestors_inclusive,
    build_accum_logp_tree_arrays,
    build_ancestor_allow_mask,
    build_retrieve_arrays,
    pad_tree,
)

FILLER = 999999


def _topk(logits, k):
    lp = torch.log_softmax(logits[0].float(), dim=-1)
    tlp, ttok = torch.topk(lp, k, dim=-1)
    return ttok.tolist(), tlp.tolist()


def _cases():
    out = []
    for seed, D, V, w, budget, root in [
        (0, 15, 200, 4, 24, 42), (1, 15, 500, 8, 40, 7),
        (2, 7, 128, 3, 20, 100), (3, 15, 300, 1, 16, 5),  # width=1 -> linear
    ]:
        torch.manual_seed(seed)
        out.append((torch.randn(1, D, V), w, budget, root, D))
    return out


def test_tree_topology_matches_reference():
    for logits, w, budget, root, D in _cases():
        ref = build_accum_logp_tree(logits, root_token=root, tree_width=w, budget=budget)
        ttok, tlp = _topk(logits, min(w, logits.shape[-1]))
        tok, par, dep = build_accum_logp_tree_arrays(ttok, tlp, root, budget)
        assert tok == ref.token_ids, (tok, ref.token_ids)
        assert par == ref.parent
        assert dep == ref.depth


def test_width1_reduces_to_linear_chain():
    logits = _cases()[3][0]  # width=1 case
    ttok, tlp = _topk(logits, 1)
    tok, par, dep = build_accum_logp_tree_arrays(ttok, tlp, 5, budget=16)
    n = len(tok)
    assert par == [-1] + list(range(n - 1))          # 0<-1, 1<-0, 2<-1, ...
    assert dep == list(range(n))
    ri, nt, ns = build_retrieve_arrays(par, n, budget=16)
    assert nt[: n - 1] == list(range(1, n)) and nt[n - 1] == -1
    assert all(s == -1 for s in ns[:n])              # linear chain: no siblings


def test_ancestor_mask_matches_reference_semantics():
    for logits, w, budget, root, D in _cases():
        ref = build_accum_logp_tree(logits, root_token=root, tree_width=w, budget=budget)
        ttok, tlp = _topk(logits, min(w, logits.shape[-1]))
        tok, par, dep = build_accum_logp_tree_arrays(ttok, tlp, root, budget)
        n = len(tok)
        prefix_len = 11
        allow = build_ancestor_allow_mask(prefix_len, par, n, budget)
        for q in range(n):
            # full prefix visible
            assert all(allow[q][k] for k in range(prefix_len))
            # tree columns == ancestors-inclusive (root always included for non-root q)
            want = set(ancestors_inclusive(par, q))
            got = {j for j in range(n) if allow[q][prefix_len + j]}
            assert got == want, (q, got, want)
            # cross-check against the reference DraftTree.is_ancestor
            for j in range(1, n):
                if j != q:
                    assert allow[q][prefix_len + j] == ref.is_ancestor(j, q)
        # filler rows: self only
        for q in range(n, budget):
            assert sum(allow[q]) == 1 and allow[q][prefix_len + q]


def test_retrieve_arrays_reconstruct_tree():
    for logits, w, budget, root, D in _cases():
        ttok, tlp = _topk(logits, min(w, logits.shape[-1]))
        tok, par, dep = build_accum_logp_tree_arrays(ttok, tlp, root, budget)
        n = len(tok)
        _, nt, ns = build_retrieve_arrays(par, n, budget)
        # walk children via next_token then next_sibling; recover parent map
        recovered = {0: -1}
        for i in range(n):
            c = nt[i]
            while c != -1:
                recovered[c] = i
                c = ns[c]
        assert recovered == {i: par[i] for i in range(n)}


def test_pad_tree_shapes_and_filler():
    ttok, tlp = _topk(_cases()[0][0], 4)
    tok, par, dep = build_accum_logp_tree_arrays(ttok, tlp, 42, budget=24)
    n = len(tok)
    t, p, d = pad_tree(tok, par, dep, budget=24, filler_token=FILLER)
    assert len(t) == len(p) == len(d) == 24
    assert all(t[i] == FILLER and p[i] == i for i in range(n, 24))  # self-parented filler
