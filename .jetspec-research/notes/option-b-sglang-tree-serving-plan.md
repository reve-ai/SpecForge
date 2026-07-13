# Option B — tree drafting for the sglang DFlash worker (bidir head)

Goal: serve DFlash with the accum-logp TREE + ancestor-masked verify (HF ref ~9.3 accept len)
instead of the linear block (~7.2). Uses the bidirectional head (causal mask proven irrelevant).

## Approach: extend the builtin DFLASH worker with a tree mode (NOT a plugin)
DFLASH has special-casing across the stack (`is_dflash`, `supports_target_verify_for_draft`) a
`CustomSpecAlgo` plugin would lose. Gate tree mode on `tree_width > 1`; `tree_width==1` keeps the
exact current linear path (regression guard — the reference reduces to linear at width 1).

## Reference to port: specforge/modeling/draft/jetspec_tree.py
- build_accum_logp_tree(draft_logits, root, tree_width, budget)  → best-first, budget-bounded tree
- build_tree_verify_mask(prefix_len, tree)  → 4D ancestor-additive mask; pos_ids = prefix_len-1+depth
- _greedy_accept  → longest root→leaf path + bonus

## sglang files to modify (venv: /mnt/home/sglang-013-venv/.../sglang/srt/speculative/)
1. dflash_worker.py — draft forward `_prepare_for_speculative_decoding` (:525); argmax head
   `_greedy_sample_from_vocab_parallel_head` (:701) → ADD `_topk_from_vocab_parallel_head` (per-depth
   top-k tok+logprob; tp==1 fast path first, tp>1 needs global logsumexp). Replace linear candidate
   build (:674-691) with tree build.
2. dflash_info.py — DFlashVerifyInput: `prepare_for_verify` mask (:227-252) → ancestor allow-mask;
   `verify` accept (:313) → tree greedy; ADD retrieve_index/next_token/next_sibling fields (:150).
3. eagle_utils.py — REUSE `verify_tree_greedy_func` (:217) for longest-path accept.
4. arg_groups/speculative_hook.py — `_handle_dflash` (:133): tree_width/budget normalize; set
   speculative_num_draft_tokens = budget; force triton target backend in tree mode.
5. server_args.py — fields near :609, CLI near :5871: --speculative-dflash-tree-width / -tree-budget.

## Key structural facts
- Anchor/current token = verify block position 0; seq_lens excludes anchor → tree root=node0,
  positions[node]=seq_lens+depth (ref's prefix_len-1+depth, equivalent, +1 offset).
- draft block stays block_size (D=block_size-1 depth logits); VERIFY window becomes budget.
  speculative_num_draft_tokens must = budget (drives KV reserve, cuda-graph, adjust coeff).
- custom_mask allow[q,k]: k<prefix_len True; else j=k-prefix_len, True iff j==q or j ancestor of q.
  (flattened per-req, reuse generate_attn_arg_prefill as-is). Force build_custom_mask=True.
- retrieve arrays: retrieve_index[b,i]=b*budget+i; next_token=first child; next_sibling=next sibling
  (linear chain = next_token[1,2,..,-1], next_sibling all -1 — confirms semantics).
- Pad tree to exactly budget nodes each step (cuda-graph fixed shape); filler = self-masked mask_token.

## Accept + KV compaction (sharpest divergence)
- verify_tree_greedy_func(predict, accept_index, accept_token_num, candidates, retrieve_*, target_predict).
- Accepted nodes are SCATTERED (not a prefix): gather out_cache_loc/req_to_token/next_target_hidden at
  accepted node slots (EAGLE-style, eagle_info.py:457-463), free non-accepted. page_size==1 for v1.

## Validation
1. Unit (no GPU): ported tree arrays byte-identical to jetspec_tree tree; custom_mask == ~(mask4d<0);
   tree_width=1 reduces to linear.
2. Single-request greedy parity: sglang tree worker == tree_spec_generate (same tokens + accepted/step).
3. Accept length ≈ 9.3 on eval corpus; tree_width=1 served run reproduces ~7.2.
4. triton target-verify logits vs HF SDPA additive-mask (argmax agreement) on a few steps.

## Risks
- Custom mask ignored on flashinfer/fa (resolve_dflash_verify_mask_policy) → require triton for v1.
- verify_len (budget) decoupled from block_size — audit everywhere assuming equal.
- cuda-graph fixed shapes → pad to budget. Scattered KV compaction (page_size==1 v1). TP>1 top-k logsumexp.
- overlap already disabled for DFLASH (fine). greedy only for v1 (matches ref).

## Build order (incremental, test-first)
A. [testable now, no GPU] Port tree build + ancestor mask + retrieve arrays into a repo module;
   unit-test parity vs jetspec_tree. ← START HERE
B. Add server args + speculative_hook normalization (tree_width/budget, num_draft_tokens=budget, triton).
C. Add top-k head to worker; wire tree build into _prepare_for_speculative_decoding.
D. Rewrite DFlashVerifyInput mask + verify (tree greedy accept, scattered KV compaction).
E. End-to-end: serve, single-req greedy parity vs HF, accept length ≈ 9.3, throughput vs linear.
