#!/usr/bin/env python3
# coding=utf-8
"""In-repo (HF) acceptance-length eval for DFlash / JetSpec draft heads.

Why this exists: sglang ships a DFLASH speculative algorithm but no JETSPEC, and its
DFLASH draft forward is hardcoded *bidirectional* — so a causal-trained (JetSpec) head
cannot be measured fairly there. This script runs the draft + target verify loop in pure
HF/SDPA, where we control the draft attention mask (bidirectional vs block-causal), and
reports the same acceptance metric the sglang harness does:

    accept_len = (total tokens committed) / (number of target verify forwards)

so an HF DFlash number is directly comparable to the sglang `eval_accept.py` baseline
(use that equivalence to validate this harness before trusting JetSpec numbers).

This is a LINEAR (single-chain) decode — the apples-to-apples comparison between the
bidirectional DFlash head and the causal JetSpec head. Tree decoding is a separate path.

Example (DFlash control, should reproduce the sglang ~6-8 number):
  /mnt/home/dflash-venv/bin/python scripts/eval_accept_hf.py \
    --target-model-path /mnt/data/shared-checkpoints/Qwen3-8B-Instruct \
    --draft-path /mnt/home/dflash-run/qwen3-8b-perfectblend/epoch_3_step_18483 \
    --head-type bidirectional \
    --prompts-file /mnt/home/dflash-run/math500_prompts.jsonl --num-prompts 50 \
    --nothink --max-new-tokens 512

JetSpec (causal) head: same command with --head-type causal and the causal checkpoint.
"""
import argparse
import json
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.jetspec_tree import tree_spec_generate


def parse_args():
    p = argparse.ArgumentParser(description="HF acceptance-length eval for DFlash/JetSpec")
    p.add_argument("--target-model-path", required=True)
    p.add_argument("--draft-path", required=True)
    p.add_argument(
        "--head-type",
        default="auto",
        choices=["auto", "bidirectional", "causal"],
        help="auto reads head_type from the draft config (default bidirectional). "
        "Override for checkpoints whose config predates head_type persistence.",
    )
    p.add_argument("--prompts-file", default=None, help="jsonl with conversations; first user turn used")
    p.add_argument("--num-prompts", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (deterministic)")
    p.add_argument("--nothink", action="store_true", help="disable Qwen3 thinking in the chat template")
    p.add_argument("--mask-token-id", type=int, default=None, help="default: tokenizer.mask_token_id")
    p.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"],
                   help="draft+target attn impl. sdpa/eager required for the explicit causal mask.")
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--mode", default="linear", choices=["linear", "tree"],
                   help="linear = single-chain block decode (apples-to-apples DFlash vs JetSpec); "
                   "tree = accum_logp tree drafting + ancestor-masked verify (JetSpec's payoff).")
    p.add_argument("--tree-width", type=int, default=4, help="per-depth top-k (tree mode)")
    p.add_argument("--budget", type=int, default=64, help="max tree nodes (tree mode)")
    return p.parse_args()


DEFAULT_PROMPTS = [
    "Explain how a transformer neural network works, step by step.",
    "Write a Python function that returns the n-th Fibonacci number using memoization.",
    "Summarize the main causes of World War I.",
    "Describe the process of photosynthesis in detail.",
]


def load_prompts(path, n):
    if not path:
        return DEFAULT_PROMPTS[:n]
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line).get("conversations", [])
            except Exception:
                continue
            first_user = next(
                (m.get("content") for m in conv
                 if m.get("role") == "user" and (m.get("content") or "").strip()),
                None,
            )
            if first_user:
                out.append(first_user)
            if len(out) >= n:
                break
    return out


def main():
    args = parse_args()
    device = "cuda"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    mask_token_id = args.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("No mask_token_id: pass --mask-token-id (must match training).")
    print(f"mask_token_id={mask_token_id}")

    print(f"Loading target: {args.target_model_path}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model_path, torch_dtype=dtype, attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(device).eval()

    print(f"Loading draft: {args.draft_path}")
    draft = DFlashDraftModel.from_pretrained(
        args.draft_path, torch_dtype=dtype, attn_implementation=args.attn_implementation,
    ).to(device).eval()

    if args.head_type != "auto":
        draft.causal_head = args.head_type == "causal"
    print(f"draft.causal_head={draft.causal_head}  block_size={draft.block_size}  "
          f"target_layer_ids={draft.target_layer_ids}")

    stop_ids = [tokenizer.eos_token_id]
    prompts = load_prompts(args.prompts_file, args.num_prompts)
    mode_desc = (f"tree(width={args.tree_width}, budget={args.budget})"
                 if args.mode == "tree" else "linear")
    print(f"prompts={len(prompts)}  nothink={args.nothink}  temp={args.temperature}  mode={mode_desc}")
    print("-" * 60)

    total_committed, total_verifies = 0, 0
    per_prompt_means = []
    for i, prompt in enumerate(prompts):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=not args.nothink,
        )
        input_ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
        if input_ids.shape[1] > args.max_prompt_tokens:
            input_ids = input_ids[:, : args.max_prompt_tokens]
        if args.mode == "tree":
            _, stats = tree_spec_generate(
                draft=draft, target=target, input_ids=input_ids, mask_token_id=mask_token_id,
                max_new_tokens=args.max_new_tokens, stop_token_ids=stop_ids,
                tree_width=args.tree_width, budget=args.budget,
            )
            committed, verifies = stats["total_committed"], stats["num_verifies"]
        else:
            _, accept_lengths = draft.spec_generate(
                target=target, input_ids=input_ids, mask_token_id=mask_token_id,
                max_new_tokens=args.max_new_tokens, stop_token_ids=stop_ids,
                temperature=args.temperature, return_acceptance=True,
            )
            committed, verifies = sum(accept_lengths), len(accept_lengths)
        if not verifies:
            print(f"[{i}] no verify steps (skipped)")
            continue
        total_committed += committed
        total_verifies += verifies
        mean_i = committed / verifies
        per_prompt_means.append(mean_i)
        print(f"[{i}] committed={committed:4d} verifies={verifies:3d} accept_len={mean_i:.2f}")

    print("-" * 60)
    if total_verifies:
        overall = total_committed / total_verifies
        print(f"OVERALL ACCEPT LENGTH = {overall:.3f}  "
              f"(committed={total_committed}, verifies={total_verifies}, cap={draft.block_size + 1})")
        if len(per_prompt_means) > 1:
            print(f"per-prompt mean={statistics.mean(per_prompt_means):.3f} "
                  f"std={statistics.stdev(per_prompt_means):.3f} "
                  f"min={min(per_prompt_means):.2f} max={max(per_prompt_means):.2f}")
    else:
        print("No verify steps recorded.")


if __name__ == "__main__":
    main()
