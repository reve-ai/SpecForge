# coding=utf-8
"""Build sglang tree-verify structures for DFlash/JetSpec accum-logp tree drafting.

This is the serving-side port of `specforge/modeling/draft/jetspec_tree.py`'s tree build +
ancestor mask, reshaped into the flat, fixed-size arrays sglang's DFLASH worker needs:

  * token_ids / parent / depth            — the accum-logp tree, padded to `budget` nodes
  * retrieve_next_token / _next_sibling   — sglang/EAGLE greedy-accept traversal encoding
  * ancestor allow-mask                   — per-node visibility over [prefix ++ tree nodes]

Layout convention (matches sglang, differs from the HF reference by design): the anchor/current
token is tree node 0 (a verify query row), NOT folded into the prefix. So there are exactly
`budget` query rows and node 0 attends to the prefix + itself only. Padding beyond the real tree
uses self-masked filler nodes (never accepted, never seen by real nodes).

`heapq`/list logic mirrors `build_accum_logp_tree` exactly so the topology is byte-identical.
"""
from __future__ import annotations

import heapq
from typing import List, Sequence, Tuple


def build_accum_logp_tree_arrays(
    topk_tok: Sequence[Sequence[int]],   # (D, k) shared per-depth top-k token ids
    topk_lp: Sequence[Sequence[float]],  # (D, k) their log-probs
    root_token: int,
    budget: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Best-first, budget-bounded accum-logp tree from precomputed per-depth top-k.

    Returns (token_ids, parent, depth) for the REAL nodes only (node 0 = root). Identical
    topology to jetspec_tree.build_accum_logp_tree given the same top-k inputs.
    """
    D = len(topk_tok)
    k = len(topk_tok[0]) if D else 0

    tokens = [int(root_token)]
    parent = [-1]
    depth = [0]
    counter = 0
    heap: List[Tuple[float, int, int]] = [(0.0, counter, 0)]  # (-cum_logprob, tiebreak, node)
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
            child_cum = -neg_cum + float(topk_lp[d][j])
            counter += 1
            heapq.heappush(heap, (-child_cum, counter, len(tokens) - 1))
    return tokens, parent, depth


def build_retrieve_arrays(
    parent: Sequence[int], num_nodes: int, budget: int
) -> Tuple[List[int], List[int], List[int]]:
    """Encode the tree for sglang's greedy tree-accept traversal (padded to `budget`).

    retrieve_index[i]        = i                (per-request flat slot; worker adds b*budget)
    retrieve_next_token[i]   = first child of i in node order, else -1
    retrieve_next_sibling[i] = next same-parent node after i in node order, else -1
    filler slots (i >= num_nodes) are self-parented, token=filler, all -1.
    """
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    for i in range(1, num_nodes):
        children[parent[i]].append(i)

    next_token = [-1] * budget
    next_sibling = [-1] * budget
    retrieve_index = list(range(budget))
    for i in range(num_nodes):
        if children[i]:
            next_token[i] = children[i][0]
        p = parent[i]
        if p != -1:
            sibs = children[p]
            pos = sibs.index(i)
            if pos + 1 < len(sibs):
                next_sibling[i] = sibs[pos + 1]
    return retrieve_index, next_token, next_sibling


def ancestors_inclusive(parent: Sequence[int], node: int) -> List[int]:
    """{node} ∪ all proper ancestors up to the root (node 0)."""
    out = [node]
    cur = parent[node]
    while cur != -1:
        out.append(cur)
        cur = parent[cur]
    return out


def build_ancestor_allow_mask(
    prefix_len: int, parent: Sequence[int], num_nodes: int, budget: int
) -> List[List[bool]]:
    """(budget, prefix_len + budget) boolean allow-mask.

    Real node q: sees the full prefix (all k < prefix_len) + tree columns of its ancestors
    (inclusive of self and root node 0). Filler rows see only themselves (avoids empty-softmax).
    """
    kv = prefix_len + budget
    allow = [[False] * kv for _ in range(budget)]
    for q in range(budget):
        if q < num_nodes:
            for k in range(prefix_len):
                allow[q][k] = True
            for anc in ancestors_inclusive(parent, q):
                allow[q][prefix_len + anc] = True
        else:
            allow[q][prefix_len + q] = True  # self-masked filler
    return allow


def pad_tree(
    tokens: List[int], parent: List[int], depth: List[int], budget: int, filler_token: int
) -> Tuple[List[int], List[int], List[int]]:
    """Pad real-node arrays out to exactly `budget` (fixed shape for cuda graphs)."""
    n = len(tokens)
    t = list(tokens) + [int(filler_token)] * (budget - n)
    p = list(parent) + [i for i in range(n, budget)]      # filler self-parented
    d = list(depth) + [0] * (budget - n)
    return t, p, d
