# JetSpec in SpecForge — Verified Design & Experiment Plan

Status: living document. Everything below marked **[verified]** was confirmed by reading
real code (this fork and/or the JetSpec repo) or scraped sources; **[hypothesis]** is to be
validated empirically; **[assumption]** is a default we'll revisit.

Goal (per user): **add JetSpec to SpecForge as a first-class spec-decoding algorithm**
alongside eagle3/dflash — not merely reproduce paper numbers. Validate with experiments;
claim only what we can measure.

---

## 1. What JetSpec is, precisely

EAGLE3 = autoregressive draft → path-conditioned, high acceptance, but cost grows with tree
depth (sequential passes). DFlash = bidirectional block-diffusion draft → one forward for a
whole block, but positions are **branch-agnostic marginals**, so deep blocks form
individually-plausible-but-mutually-inconsistent trees. JetSpec keeps DFlash's one-pass
efficiency but trains the head with a **block-causal mask** so each in-block position is
predicted only from the prefix + earlier in-block positions → per-depth marginals align with
the target's autoregressive factorization. Frozen target verifies the whole token tree in one
masked pass → lossless. **[verified: paper abstract + blog + methods excerpt]**

### Key realization that de-risks the whole project **[verified]**
The default inference tree algorithm `accum_logp` builds the tree from **independent per-depth
top-k marginals** (same candidate set at depth d regardless of parent), scored by cumulative
log-prob; the target verifies under an ancestor mask. True path-conditioned re-drafting
(`path_conditional_refresh`, design id A6) is **PLANNED / not implemented** and the authors
note "raw path-conditioning doesn't beat accum_logp at our budgets" and that it "needs a vLLM
draft-side tree-attention hook."
→ Therefore **branch-wise causal conditioning lives entirely in how the head is TRAINED**
(block-causal mask), and the full JetSpec — including tree-decoding eval — is implementable
inside SpecForge on HF/SDPA. We do **not** need their vLLM/Triton engine to demonstrate the
acceptance-length gain. (Source: `.jetspec-research/sources/jetspec-code/jetspec_tree_baselines_accum_logp.py`,
`..._path_conditional_refresh.py`, `..._model_runner.py`.)

---

## 2. The exact code-level delta vs this fork's DFlash **[verified]**

This fork's DFlash (`specforge/core/dflash.py`, `specforge/modeling/draft/dflash.py`):
- Head: N Qwen3 decoder layers. Target features from selected layers concatenated → `fc`
  (bias-free) → `hidden_norm` (RMSNorm) → injected as **contextual K/V** (`k_ctx`/`v_ctx`)
  alongside noise-token K/V. KV layout length = `ctx_len + q_len`; mask shape `[L, 2L]`.
- Noise input: block-start positions keep the real token (anchor); all other positions become
  `mask_token_id` embeddings.
- Loss: **uniform CE** on positions that are (loss_mask=1) AND (not block-0) AND (not
  block-start). **No depth weighting** — matches the paper's main causal-vs-bidirectional
  comparison. (Issue #465's "Loss Decay Gamma 7.0" is z-lab DFlash, not this code.)
- Within-block mask is **bidirectional**:
  - dense `_create_parallel_attention_mask`: `noise_mask = (q_block_ids == k_block_ids)`
  - flex `dflash_mask_fn`: `noise_visible = (~is_ctx) & (k_block_noise == q_block)`
  - context half (both): `k_block < q_block` (strictly earlier blocks).

**JetSpec change = make the within-block (noise) mask causal**: a noise position at
intra-block offset `i` may attend to noise positions at offset `j <= i` in the same block
(context half unchanged). One condition flips:
`noise_visible = (~is_ctx) & (k_block_noise == q_block) & (k_intra <= q_intra)`.

The JetSpec public repo confirms this is the *only* structural difference — its
`draft_head.py` IS the DFlash head plus a `dflash_config.causal_head` flag; when set it builds
a causal mask over the running `[ctx ; noise]` sequence. The training-time block-causal mask
lives in their separate vendored "reference" (PTD/causal_parallel_drafting); the public repo
is engine/inference-only.
(Source: `.jetspec-research/sources/jetspec-code/jetspec_models_draft_head.py`, esp.
`_build_dflash_causal_attention_mask` and `dflash_config.causal_head`.)

### Architecture parity notes (paper vs this fork) **[verified]**
- Paper Qwen3-8B head: 5 layers, 32 heads, 8 KV heads, head_dim 128, MLP 12288, target layers
  `{1,9,17,25,33}` of 36, concat 5·d → bias-free linear → RMSNorm. This fork's
  `configs/qwen3-8b-dflash.json` matches (5 layers, block_size 16, etc.) EXCEPT target layers:
  this fork uses `build_target_layer_ids(36, 5)` ≈ `[1, 9, 17, 25, 32]` (last is 32 not 33).
  Near-identical; we'll make target layers configurable and test the paper's exact set as an
  ablation.

---

## 3. Inference / eval mechanics to port **[verified from JetSpec repo]**

Per decode step, given accepted prefix and its tapped target hidden features `target_hidden`:
1. **Draft (one forward):** run the (causal) head over a block: position 0 = anchor (last
   accepted token), positions 1..block_size-1 = mask embeddings → `draft_logits (1, D, V)`,
   `D = block_size - 1`. (Apply target `lm_head`.)
2. **Build tree:** `accum_logp` — per-depth `log_softmax` → top-k tokens/logprobs → best-first
   heap expansion keyed by cumulative log-prob, bounded by `budget`; per-depth top-k shared
   across parents. Produces `DraftTree(token_ids, parent_indices, depth, cum_logprob,
   ancestor, child_maps)`.
3. **Verify (one forward):** flatten tree to a sequence; target forward with a **4D additive
   ancestor mask** (each node attends to prefix + its ancestors only) via SDPA; accept the
   longest path matching the target's argmax/sample at each node (lossless). Re-tap
   `target_hidden` for the accepted nodes; advance.
Linear-chain decode (existing DFlash `spec_generate`) is the `tree_width=1` / no-tree special
case — useful intermediate baseline.

Reference files to port: `tree/_core/{base,ancestor,accept,topk_build}.py`,
`tree/baselines/accum_logp.py`, `core/model_runner.py` (ancestor-mask-through-SDPA seam).

---

## 4. SpecForge integration shape (mirror eagle3/dflash conventions)

Reuse shared infra (data `build_eagle3_dataset`/`prepare_dp_dataloaders`, `BF16Optimizer`,
`tracker`, FSDP, target wrappers). Minimize divergence from validated DFlash so the
experiment isolates the mask. Candidate layout (final names TBD during impl):
- `specforge/modeling/draft/jetspec.py` — reuse DFlash head; add `causal_head` →
  block-causal mask. (Possibly just extend dflash.py with a flag rather than fork the file,
  to stay DRY; decide during impl. The JetSpec repo uses a flag on one class — strong
  precedent.)
- `specforge/core/jetspec.py` — `OnlineJetSpecModel`: mirrors `OnlineDFlashModel` but emits the
  causal within-block mask (flex + dense). Optional: distillation loss, random anchors.
- `scripts/train_jetspec.py` — mirrors `train_dflash.py`.
- `configs/qwen3-8b-jetspec.json` — DFlash config + `dflash_config: {causal_head: true,
  target_layer_ids: [1,9,17,25,33]}` (ablatable).
- Eval: extend the acceptance harness (`/mnt/home/dflash-smoke/eval_accept.py` is the existing
  reference) with a tree-decode path.
- Registry wiring (`specforge/modeling/auto.py`) so the architecture name resolves.

---

## 5. Experiment plan (experiment-driven, controlled)

- **Phase 0 — Control.** Env: train on `/mnt/home/dflash-venv`, serve/eval on
  `/mnt/home/sglang-013-venv`. Establish a DFlash baseline acceptance length on Qwen3-8B
  non-thinking (reuse existing `/mnt/home/dflash-run` numbers if directly comparable; else a
  short smoke train on `sharegpt_train_2k.jsonl`). Record exact invocations + numbers.
  *(Phase-0 specifics filled from on-disk recon — see §6.)*
- **Phase 1 — H1: causal mask alone helps.** Implement block-causal mask; unit-test
  (dense≡flex; attend-pattern matches the intended triangular-within-block; numerical parity
  vs a hand-built reference mask). Train JetSpec at strict parity with the DFlash control
  (same data, lr, block_size=16, layers, steps). Compare acceptance length on a **linear-chain
  decode** (apples-to-apples with DFlash `spec_generate`). H1 passes if JetSpec ≥ DFlash.
- **Phase 2 — refinements as single-variable ablations** (acceptance length each): random
  anchors ≤512/example vs contiguous tiling; soft-label distillation vs hard CE; target layers
  `{1,9,17,25,33}` vs auto.
- **Phase 3 — tree decoding.** Port tree-build + ancestor-masked verify; sweep
  (budget, depth, width). Report acceptance length & tokens-per-forward for DFlash-bidirectional
  vs JetSpec-causal under the SAME tree algorithm — this is where the causal head's marginal
  alignment should convert budget into longer accepted prefixes.

### Metrics & honesty
In-repo verifiable metric = **acceptance length** (and tokens-per-forward in tree decode).
Wall-clock end-to-end speedup (paper's 9.64×) requires the optimized serving engine
(Triton tree-attention + CUDA graphs / vLLM) and is **out of SpecForge scope**; we won't claim
it from SpecForge alone. We *can* optionally sanity-check serving via the sglang path the
DFlash run already used (§6).

---

## 6. On-disk assets (Phase-0 inputs) — filled from recon **[verified]**

Decision update: first experiment stays on **Qwen3-8B-Instruct** (user's choice) — it is
fully supported on disk, so the choice is cheap AND reuses the DFlash infra.

Environments (use these, NOT /opt/uv/venv):
- Train: `/mnt/home/dflash-venv/bin/{python,torchrun}` (torch 2.11+cu129, transformers 5.6.0)
- Serve/eval (sglang 0.5.13): `/mnt/home/sglang-013-venv/bin/python`

Qwen3-8B assets:
- Target: `/mnt/data/shared-checkpoints/Qwen3-8B-Instruct`
- Draft config: `configs/qwen3-8b-dflash.json` (5 layers, block 16, num_target_layers 36,
  hidden 4096, vocab 151936, target_layer_ids None → build_target_layer_ids(36,5)).
- Regen (no-think) train data: `/mnt/home/dflash-run/perfectblend_regen_nothink.jsonl` (526M),
  `/mnt/home/dflash-run/sharegpt_train_regen_nothink.jsonl` (534M),
  `..._clean.jsonl` variant. Regen recipe used `--no-think`, temp 0.8.
- **DFlash control checkpoint (pinned):** `/mnt/home/dflash-run/qwen3-8b-perfectblend/epoch_3_step_18483`
  (lr 6e-4, block 16). Eval no-think greedy on MATH-500 (100, `math500_prompts.jsonl`) and
  held-out PerfectBlend (50). Measured DFlash accept_len ≈ 6–8 range (re-measure exactly in P0).
- Eval harness: `/mnt/home/dflash-smoke/eval_accept.py` — sglang-based; accept_len =
  completion_tokens / spec_verify_ct (cap block_size+1). Env-var driven (EVAL_NOTHINK,
  EVAL_PROMPTS_FILE, EVAL_N, EVAL_MODEL_PATH). Serving uses
  `--speculative-algorithm DFLASH --speculative-dflash-block-size 16`.

CRITICAL eval constraint **[verified]**: sglang ships `DFLASH` but **not** a JETSPEC algorithm.
So JetSpec acceptance must be measured via an **HF/SDPA in-repo harness** (port of
`spec_generate` + tree decode). Plan: (1) build the HF acceptance harness, (2) validate it by
reproducing DFlash's sglang accept_len on the same control checkpoint/data (HF≈sglang sanity),
(3) then run JetSpec vs DFlash head-to-head *inside the HF harness* (engine held constant).
Serving JetSpec in sglang (a JETSPEC algorithm) is a separate, larger effort, out of scope.

### Integration-shape decision **[decided]**
Implement the causal head as a **one-flag variant of the shared DFlash code**, not a forked
module tree — because the train-time delta is exactly the within-block mask, and upstream
JetSpec itself uses a `causal_head` flag on the same class. Concretely:
- `core/dflash.py`: add causal within-block mask (flex + dense), selected by `head_type`
  (`bidirectional` = DFlash default, unchanged; `causal` = JetSpec). Training needs ONLY this.
- `modeling/draft/dflash.py`: carry a `causal_head` flag for the INFERENCE KV-cache mask
  (Phase 3); training is mask-only so the head is untouched for P1.
- First-class UX: `configs/qwen3-8b-jetspec.json` + `scripts/train_jetspec.py` (thin) and/or a
  `--head-type` flag, plus registry naming, so JetSpec is selectable as its own algorithm
  while sharing the validated backbone. Keeps H1 a literally-one-variable diff.

### Resource-gating note
Code + unit tests need no GPU (safe to proceed on branch). Training runs and sglang servers
consume the shared 8×H100 node — pause for user go-ahead before launching those.

---

## 6b. Implementation status (Phase 1 training) — DONE, no GPU **[verified by unit test]**
- `specforge/core/dflash.py`: added `dflash_block_visible()` (single source of truth for the
  block mask, used by BOTH flex + dense paths); `OnlineDFlashModel(head_type=...)`; causal
  within-block branch. Bidirectional path semantics unchanged (proven against reference).
- `tests/test_dflash_jetspec_mask.py`: 4 checks PASS — dense≡reference (both modes), ctx half
  identical / only noise differs, causal noise = block-lower-triangular, flex calling-convention
  ≡ reference.
- `configs/qwen3-8b-jetspec.json` (head_type="causal"); `scripts/train_dflash.py` gains
  `--head-type {auto,bidirectional,causal}` (auto reads config).
- Note: `build_target_layer_ids(36,5) = [1,9,17,25,33]` == paper's exact 8B taps. No divergence.

### Refined H1 protocol (rigorous control)
Train BOTH heads with the **current code**, identical data/seed/steps/lr/block, differing ONLY
in `--head-type` (bidirectional control vs causal treatment). The existing
`qwen3-8b-perfectblend/epoch_3_step_18483` DFlash ckpt is a sanity anchor, not the head-to-head
control (it predates the refactor). Eval both in the SAME HF acceptance harness on the SAME
held-out prompts (math500 + perfectblend-heldout), block 16, greedy no-think.
Eval needs causal inference-mask support in `spec_generate` (small Phase-3 prereq) since sglang
has no JETSPEC algorithm.

## 6c. sglang DFLASH internals & JETSPEC feasibility **[verified from installed sglang 0.5.13]**
- `SpeculativeAlgorithm.DFLASH` (spec_info.py) → `DFlashWorker`. Draft forward runs with
  `AttentionType.ENCODER_ONLY` = **bidirectional** (models/dflash.py:122, intentional for
  DFlash). Verify uses a linear causal mask (`k_idx <= prefix_len + q_idx`). `topk=1`, **no
  tree**. No `causal_head`/`head_type` in its dflash_config parser.
- ⇒ Serving our causal-trained head via the existing `--speculative-algorithm DFLASH` would be
  a **train/inference mask mismatch** (causal head run bidirectionally) → expect under-report.
  Don't use sglang DFLASH to judge JetSpec.
- Adding JETSPEC to sglang: minimal *linear* variant = make draft `attn_type` configurable +
  set a causal draft mask (~50-100 LOC) → serve causal head correctly, reuse eval_accept.py.
- *Tree* variant — REFINED (investigation #ae9): EAGLE's tree-verify is **drafter-agnostic and
  fully reusable**. `EagleVerifyInput` + `build_tree_kernel_efficient()` + `verify_tree_greedy_func()`
  (sgl_kernel) need only (draft_token, custom_mask, positions, retrieve_index, retrieve_next_token,
  retrieve_next_sibling) — NO autoregressive drafting. JetSpec worker: one causal draft forward →
  accum_logp tree → feed parent_list+tokens to `build_tree_kernel_efficient` (set topk=tree_width>1
  to enable the tree path) → reuse verify unchanged. **~850-1200 LOC** (down from 2.5-3.5k): new
  jetspec_worker.py (~300-400), draft head (~150-250), eagle_worker dispatch (~50); verify = 0 LOC.
  Backend must be **Triton** (or hybrid_linear_attn for the 35B) — flashinfer/FA3/TRTLLM force
  topk=1 (no tree). Risks: single-block draft KV-cache layout; depth→RoPE position mapping;
  `spec_steps` likely = tree depth (D=block_size-1), NOT 1 — verify in build_tree_kernel_efficient.
  Sequence AFTER the HF JetSpec gain is validated. HF in-repo harness remains the primary eval.

## 6d. HF eval harness design **[decided]**
Measure acceptance length = total accepted tokens / number of target verify forwards (matches
`eval_accept.py`'s `completion_tokens / spec_verify_ct`), engine-agnostic so comparable to the
sglang DFlash anchor. Use a **cache-free, whole-prefix** linear decode (correct-by-construction,
mirrors training; slower but this is measurement not serving) rather than reusing DFlash's
incremental-cache spec_generate, whose causal-correctness with KV cache is unverified:
  per step: ctx = target features of positions [0, s); noise block = [anchor@s, MASK×(bs-1)] at
  positions [s, s+bs); position_ids = arange(s+bs); is_causal from head. Draft predicts
  s+1..s+bs-1 in one forward; target verifies; accept longest greedy-matching prefix; re-tap
  features for accepted tokens; advance. Same harness serves both heads (causal_head toggles the
  mask) and the DFlash-bidirectional control → apples-to-apples. Load draft with
  attn_implementation="sdpa" (explicit-mask path). Validate by reproducing the existing DFlash
  baseline (~6-8) on `qwen3-8b-perfectblend/epoch_3_step_18483` before trusting JetSpec numbers.
Tree decode (Phase 3) builds on this: per-depth top-k → accum_logp tree → flatten → target
verify under 4D ancestor mask → accept longest path.

## 6e. RESULTS LOG

### R1 — OOD eval (MATH-500, 30 prompts, no-think, greedy, max_new 256) [2026-06-30]
Draft trained on **general PerfectBlend** (perfectblend_regen_nothink_clean, 200K), evaluated
**OOD on MATH-500**. HF harness (accept_len = committed/verifies). Both checkpoints = same
data/recipe/seed (lr 6e-4, 3 epochs, block16); only --head-type differs.

| config | OVERALL | per-prompt |
|---|---|---|
| DFlash-linear (bidir) | 4.85 | 5.39 |
| JetSpec-linear (causal) | 4.73 | 5.22 |
| DFlash-tree w8/b64 | 5.53 | 5.73 |
| JetSpec-tree w8/b64 | 5.34 | 5.58 |

**NULL RESULT**: causal ≈ bidirectional (nominally bidir slightly ahead) in both modes; gaps
~1 SE (std~1.1/30 → SE~0.2). Verified NOT a bug: draft.causal_head=True in causal runs; toggling
is_causal changes draft hidden by max|Δ|=10.56 (mask is applied). Tree helps both ~equally (+0.6).
JetSpec advantage NOT reproduced here.

**Confounds to eliminate before concluding** (ranked): (1) eval domain mismatch — trained
general, evaluated math (OOD); paper trains+evals matched domain; use in-distribution
perfectblend_test (R2). (2) budget too small — paper's causal gain grows with budget (128-256),
we used 64; sweep budget. (3) power — 30 prompts. (4) training regime — paper uses 780K math/code;
we used 200K general.

### R2 — in-distribution eval (perfectblend_test, 50 prompts, max_new 512) [2026-06-30]
| config | OVERALL | per-prompt |
|---|---|---|
| DFlash-linear | 3.39 | 4.72 |
| JetSpec-linear | 3.27 | 4.67 |
| DFlash-tree w8/b64 | 3.88 | 4.93 |
| JetSpec-tree w8/b64 | 3.75 | 4.78 |
NULL RESULT HOLDS in-distribution too (bidir nominally ahead). In-distribution accept < MATH-500
(math is more structured/draftable). ⇒ domain mismatch was NOT masking a JetSpec advantage.
Prior DFlash refs (sglang, eval_accept.py): 35B on layout ~5.1 (step15k); 35B on MATH-500 ~7.2
(v013 rope/src patches; 5.5→7 from the patches). No prior causal/JetSpec numbers — ours are first.

### R3 — MATH-500 budget sweep, both heads (50 prompts, max_new512, width8) [2026-06-30]
| head | b64 | b128 | b256 |
|---|---|---|---|
| DFlash (bidir) | 5.69 | 6.81 | 7.23 |
| JetSpec (causal) | 5.57 | 6.56 | 7.02 |
Budget scales acceptance hard for BOTH (+1.5 from 64→256), but bidir is ahead at EVERY budget;
the gap does not close/reverse. **Budget hypothesis refuted.** (Our b256 ~7.2–7.8 matches the prior
35B DFlash MATH-500 ~7.2 → harness well-calibrated.) Null result robust across linear/tree,
in-dist/OOD, budgets 64–256, same LR 6e-4.

### Direction after R3
Ruled out: causal-mask-alone, domain, budget, LR (controlled). Remaining JetSpec-specific training
ingredients: random anchors (SHELVED — user notes their DFlash trained fine with contiguous tiling,
so unlikely the blocker; implemented + unit-tested, not run) and **soft-label distillation (RUNNING)**.
Distill impl verified: pure-distill loss = 0 exactly when teacher==student (KL + p-1 teacher-shift
correct); chunked to fit batch-4 memory.

### R4 — causal + soft-label distillation (T=1, alpha=0, else identical recipe) — RUNNING (train_causal_distill)
Compare to bidir baseline (7.23 @ b256) and causal-no-distill (7.02 @ b256).

## 6f. First-class SpecForge integration — DONE [2026-07-01]
SpecForge = *training* (produces the draft checkpoint); sglang = *serving* (loads + runs it).
JetSpec is now a first-class SpecForge *training* method (serving = separate sglang task #6):
- `specforge/modeling/draft/jetspec.py`: `JetSpecDraftModel(DFlashDraftModel)` forces the causal
  head; re-exports the tree utils. Exported in `draft/__init__.py`.
- `scripts/train_dflash.py` `build_models`: selects draft class by config architecture
  (JetSpecDraftModel vs DFlashDraftModel).
- `configs/qwen3-8b-jetspec.json`: `architectures=["JetSpecDraftModel"]`, `head_type=causal`.
- `scripts/train_jetspec.py`: thin entrypoint (defaults `--head-type causal`, reuses train_dflash).
- `examples/run_qwen3_8b_jetspec_online.sh`.
- Inference/eval: `jetspec_tree.py` (accum_logp tree + ancestor-mask verify) + `eval_accept_hf.py --mode tree`.
- Tests pass: mask (dense≡flex≡ref), tree (ancestor mask, width-1≡chain), random-anchor, JetSpec-forces-causal.
Note: our eval loads via `DFlashDraftModel.from_pretrained` + reads `head_type` (works for both
DFlash- and JetSpec-arch checkpoints); trust_remote_code/auto_map portability of JetSpec ckpts is a
follow-up (save_checkpoint copies modeling_dflash.py only).

### R5 — matched math/code data (DeepMath 103K + opc 147K, ~250K unique, regen no-think), both heads + distill, lr 6e-4, 3 epochs [2026-07-01]
Trajectory (MATH-500, tree w8, causal−bidir; N=50 intermediates, N=150 final):
| budget | step8k | step16k | step20k | FINAL(23484,N150) |
|---|---|---|---|---|
| 32 | −0.25 | −0.01 | +0.04 | −0.01 |
| 64 | −0.49 | −0.11 | −0.03 | −0.09 |
| 128 | −0.07 | −0.06 | −0.09 | −0.08 |
| 256 | +0.02 | −0.03 | −0.00 | −0.08 |
FINAL absolute b256: causal 7.94 / bidir 8.03 (per-prompt 8.33±1.68 vs 8.41±1.70; SE~±0.14).
**Definitive: causal ≈ bidirectional on MATH-500, converged — TIE within noise (bidir nominally
+0.08 everywhere). No causal advantage.** Loss trajectory: causal & bidir tracked each other
near-identically the whole run (mask barely changes what's learned on content-free MASK tokens);
epoch-3 gains small → nearing a plateau on the 250K unique set (more *unique* data, not epochs, is
the lever). Paper b256 MATH-500 = JetSpec 10.76 / DDTree 9.81 (+0.95); we're ~8.0 and not separated.

Conclusion so far: across general + matched data, ±distillation, budgets 32–256, full training —
the causal mask does NOT beat bidirectional in our regime. Prime suspect = unique-data
diversity/volume (250K vs paper's 780K incl. gated Nemotron STEM/chat) + lr 3e-4.

### R6 — code benchmarks (HumanEval + MBPP), final ckpts, budgets 64/256, N150 [2026-07-01]
| bench | b64 causal/bidir | b256 causal/bidir |
|---|---|---|
| HumanEval | 5.85 / 6.04 | 7.74 / 7.69 |
| MBPP | 4.76 / 4.82 | 5.81 / 5.89 |
All within ~1–1.5 SE; nominal leads flip-flop by bench → **no signal**. COMPREHENSIVE PARITY:
causal ≈ bidir on MATH + HumanEval + MBPP, converged, every budget. Paper's +0.95 not reproduced.
Deviations left (audit): (1) DATA — 250K unique DeepMath+opc vs paper 780K incl. Nemotron STEM/chat;
(2) RANDOM ANCHORS — we used tiling (paper uses random ≤512/ex). LR 6e-4 = paper-comparable (not a
deviation); forward-KL distill MATCHES; no depth-weighting MATCHES; Triton kernel = speed-only.

### R7 — RANDOM-ANCHOR ablation (matched data, distill, sdpa, num_anchors=128) — DONE [2026-07-02]
Both runs completed (final ckpt epoch_3_step_23484; each hit the benign teardown exit-1, ckpts intact).
Swept with eval_randanchor_math.sh / eval_randanchor_code.sh (randanchor-dir variants of the sweep
scripts; distinct log prefixes randanchor_math_* / randanchor_code_*), N=150, tree width 8.

FULL 2x2 GRID — accept length, causal / bidir (tiling numbers from R5/R6):
                 tiling (causal/bidir)      random-anchor (causal/bidir)
  MATH-500 b256:   7.94 / 8.03                8.24 / 8.22
  HumanEval b256:  7.74 / 7.69                8.61 / 8.57
  MBPP    b256:    5.81 / 5.89                6.18 / 6.22
  (random MATH b32/64/128 = 4.33/6.21/7.65 causal, 4.36/6.24/7.71 bidir; code b64: HE 6.55/6.57, MBPP 5.07/5.08)

TWO findings:
1. causal ≈ bidir parity HOLDS under random anchors too, every budget, math+code (within ~1 SE).
   → the paper's +0.95 causal advantage does NOT reproduce; robust across {tiling,random}x{±distill}x{budget}.
2. Random anchors is a real ABSOLUTE acceptance win for BOTH heads (not a causal-specific unlock):
   +0.2–0.3 on MATH, +0.3–0.4 on MBPP, +0.87/0.88 on HumanEval, vs tiling on the SAME 250K data.
   → anchor sampling diversity matters for acceptance; the causal mask itself does not.

Remaining untested lever vs paper = DATA (Nemotron volume/mix: STEM+chat, 780K vs our 250K). Token now
available → next: pull Nemotron v2, FILTER OUT StackOverflow (CC-BY-SA, copyleft) + WildChat (ODC-BY)
by the per-sample license field, keep math/code(/stem) categories, retrain the random-anchor pair.

## 7. Open items / risks
- [resolved] How tree gets branch-conditioning at inference → from causal *training*, tree from
  marginals (§1).
- flex_attention block-causal mask correctness → unit test vs dense reference before trusting.
- RoPE: reuse DFlash's exact rotary setup (Q uses last q_len, K full); user flagged
  train/serve RoPE mismatch as historically critical — keep draft rotary config aligned with
  target; verify in eval.
- Random-anchor data pipeline differs from DFlash's contiguous tiling; treat as Phase-2
  ablation, not Phase-1, to keep H1 clean.
- Tokenizer mask token: DFlash adds `<|MASK|>` if absent; JetSpec must use the identical
  mask-token convention as training for inference.
</content>
