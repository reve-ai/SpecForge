# JetSpec Research — Reference Material

Scraped reference material for implementing **JetSpec** (causal parallel tree drafting)
in our SpecForge fork. Captured 2026-06-30.

## Sources (verbatim scrapes)

| File | Source URL | Notes |
|---|---|---|
| `sources/jetspec-paper-arxiv.md` | https://arxiv.org/html/2606.18394 (v3) | Full paper text, HTML→markdown. 113KB. |
| `sources/jetspec-blog-parallel-tree-decoding.md` | https://haoailab.com/blogs/parallel-tree-decoding/ | Full blog post. |
| `sources/jetspec-github-readme.md` | github.com/hao-ai-lab/JetSpec (master) | Raw README. |
| `sources/jetspec-github-filetree.txt` | github.com/hao-ai-lab/JetSpec (master) | Recursive file listing. |
| `sources/specforge-issue-465.md` | github.com/sgl-project/SpecForge/issues/465 | DFlash Qwen3-8B training writeup + all 22 comments. |
| `notes/paper-methods-excerpt.md` | (provided by user) | Key methods paragraph the user pasted. |

## One-paragraph problem statement

EAGLE3 = autoregressive draft → high acceptance per path but cost grows with tree depth
(can't go deep). DFlash = bidirectional block-diffusion draft → one forward pass for a whole
block, but positions are scored as branch-agnostic marginals, so a block of tokens isn't a
valid autoregressive continuation (mutually inconsistent). **JetSpec** keeps DFlash's
one-pass efficiency but swaps the within-block bidirectional mask for a **tree-causal mask**:
each drafted position attends only to the prefix + earlier positions in its block, so the
draft distribution follows the target's autoregressive order. The frozen target verifies the
whole token tree in one masked forward pass → lossless.

## The single most important code-level delta vs DFlash (in our fork)

DFlash's within-block mask is **bidirectional**:
`specforge/core/dflash.py` — `noise_mask = (q_block_ids == k_block_ids)` (dense path) and
`noise_visible = (~is_ctx) & (k_block_noise == q_block)` (flex path).

JetSpec makes the within-block (noise) attention **causal**: a noise position at intra-block
offset `i` may attend to noise positions at offset `j <= i` in the same block (plus all prior
blocks' context), instead of all positions in the block. That is the conceptual core; tree
drafting at inference time builds on top of this causal head.

See `../.jetspec-research/notes/plan.md` (to be written) for the full implementation plan.
