#!/usr/bin/env python3
# coding=utf-8
"""Build HumanEval + MBPP eval-prompt files for the acceptance harness.

Writes {"conversations": [{"role": "user", "content": <prompt>}]} jsonl (the format
scripts/eval_accept_hf.py consumes). These are HELD-OUT code benchmarks (not in training).
"""
import argparse
import json
import os


def w(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({"conversations": [{"role": "user", "content": r}]},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/mnt/home/jetspec-run/data")
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()
    from datasets import load_dataset

    # HumanEval: complete the given function stub.
    he = load_dataset("openai/openai_humaneval", split="test")
    he_rows = [f"Complete the following Python function. Return only the completed code.\n\n{r['prompt']}"
               for r in list(he)[: args.n]]
    w(os.path.join(args.out_dir, "humaneval_prompts.jsonl"), he_rows)

    # MBPP: natural-language task + the first test as the spec (standard MBPP prompting).
    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="test")
    mb_rows = []
    for r in list(mbpp)[: args.n]:
        tests = r.get("test_list") or []
        spec = f"\nYour code should pass this test: {tests[0]}" if tests else ""
        mb_rows.append(f"{r['text']}{spec}")
    w(os.path.join(args.out_dir, "mbpp_prompts.jsonl"), mb_rows)


if __name__ == "__main__":
    main()
