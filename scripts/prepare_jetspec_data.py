#!/usr/bin/env python3
# coding=utf-8
"""Build a matched math+code prompt set (regen-ready) for JetSpec training.

Emits a jsonl of {"conversations": [{"role": "user", "content": <prompt>}], "source": ...}
— the first-user-turn format that scripts/regenerate_train_data.py consumes (it applies the
target's chat template and continues generation with Qwen3-8B in no-think mode).

Sources (all public, no token):
  - DeepMath-103K   -> math prompts (the `question` field)
  - opc-sft-stage1  -> code prompts (the `instruction` field), streamed + subsampled
Optional (only if a HF token with accepted terms is available):
  - nvidia/Nemotron-Post-Training-Dataset-v2  (gated) -> math/code/STEM/chat

Example:
  python scripts/prepare_jetspec_data.py \
    --deepmath-n 103000 --opc-n 150000 --opc-subset largescale_diverse_instruct \
    --seed 42 --out ./cache/dataset/jetspec_math_code_prompts.jsonl
"""
import argparse
import itertools
import json
import os
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--deepmath-n", type=int, default=103000)
    p.add_argument("--opc-n", type=int, default=150000)
    p.add_argument("--opc-subset", type=str, default="largescale_diverse_instruct")
    p.add_argument("--nemotron-n", type=int, default=0,
                   help="Global cap on total Nemotron rows kept (0 = no global cap; use with --nemotron-categories).")
    p.add_argument("--nemotron-categories", type=str, default="math,code",
                   help="Comma-separated Nemotron v2 categories to pull (files data/<cat>-*.parquet). "
                        "Each may carry a per-category cap as 'cat:N' (e.g. 'math,code,stem:115043'). "
                        "Available: math,code,stem,chat,multilingual*. Paper regime = math,code(,stem).")
    p.add_argument("--nemotron-per-category", type=int, default=0,
                   help="Default cap per category when not given inline as 'cat:N' (0 = all).")
    p.add_argument("--min-chars", type=int, default=8, help="Skip trivially short prompts.")
    p.add_argument("--max-chars", type=int, default=8000, help="Skip pathologically long prompts.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def _emit(prompt, source):
    prompt = (prompt or "").strip()
    return {"conversations": [{"role": "user", "content": prompt}], "source": source}


def _license_blocked(lic):
    """True for licenses we exclude from training: StackOverflow-style copyleft
    (CC BY-SA / ShareAlike) and Open Data Commons (WildChat ODC-BY / ODbL).
    Everything else (notably CC BY 4.0, the whole English Nemotron-v2 corpus) is kept."""
    s = "".join(ch for ch in (lic or "").lower() if ch.isalnum())
    return ("bysa" in s or "sharealike" in s
            or "odc" in s or "odbl" in s or "opendatacommons" in s)


def _nemotron_user_prompt(row):
    msgs = row.get("messages") or row.get("conversations")
    if isinstance(msgs, list) and msgs:
        p = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
        if p:
            return p
    return row.get("input") or row.get("prompt") or row.get("question")


def main():
    args = parse_args()
    from datasets import load_dataset

    rows, counts = [], {}

    def add(prompt, source):
        if not prompt:
            return
        n = len(prompt)
        if n < args.min_chars or n > args.max_chars:
            return
        rows.append(_emit(prompt, source))
        counts[source] = counts.get(source, 0) + 1

    # --- DeepMath (math): small, full load ---
    if args.deepmath_n > 0:
        print(f"[deepmath] loading zwhe99/DeepMath-103K ...", flush=True)
        ds = load_dataset("zwhe99/DeepMath-103K", split="train")
        for r in itertools.islice(ds, args.deepmath_n):
            add(r.get("question"), "deepmath")
        print(f"[deepmath] kept {counts.get('deepmath', 0)}", flush=True)

    # --- opc (code): stream + subsample to avoid full download ---
    if args.opc_n > 0:
        print(f"[opc] streaming OpenCoder-LLM/opc-sft-stage1:{args.opc_subset} ...", flush=True)
        ds = load_dataset("OpenCoder-LLM/opc-sft-stage1", args.opc_subset,
                          split="train", streaming=True)
        for r in itertools.islice(ds, args.opc_n):
            add(r.get("instruction"), "opc")
        print(f"[opc] kept {counts.get('opc', 0)}", flush=True)

    # --- Nemotron v2 (gated): category-targeted, license-filtered ---
    has_token = bool(os.environ.get("HF_TOKEN") or os.path.exists(
        os.path.expanduser("~/.cache/huggingface/token")))
    # Parse "cat" or "cat:N" entries into (name, cap) pairs (cap 0 => use default/all).
    cats = []
    for tok in args.nemotron_categories.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            name, cap = tok.split(":", 1)
            cats.append((name.strip(), int(cap)))
        else:
            cats.append((tok, args.nemotron_per_category))
    # Treat nemotron as requested only if the user explicitly asks.
    nemotron_requested = (args.nemotron_n > 0 or args.nemotron_per_category > 0
                          or any(cap > 0 for _, cap in cats))
    if nemotron_requested and has_token and cats:
        REPO = "nvidia/Nemotron-Post-Training-Dataset-v2"
        lic_kept, lic_dropped = {}, {}     # license-string -> count (empirical audit)
        global_cap = args.nemotron_n if args.nemotron_n > 0 else None
        n_before_nemotron = len(rows)
        for cat, cap in cats:
            print(f"[nemotron] streaming category '{cat}' (data/{cat}-*.parquet)"
                  f"{f' cap={cap}' if cap else ''} ...", flush=True)
            kept_cat = 0
            try:
                ds = load_dataset(REPO, data_files=f"data/{cat}-*.parquet",
                                  split="train", streaming=True)
                for r in ds:
                    if global_cap is not None and \
                       (len(rows) - n_before_nemotron) >= global_cap:
                        break
                    if cap and kept_cat >= cap:
                        break
                    lic = r.get("license")
                    if _license_blocked(lic):
                        lic_dropped[lic] = lic_dropped.get(lic, 0) + 1
                        continue
                    prompt = _nemotron_user_prompt(r)
                    before = len(rows)
                    add(prompt, f"nemotron_{cat}")
                    if len(rows) > before:
                        kept_cat += 1
                        lic_kept[lic] = lic_kept.get(lic, 0) + 1
                print(f"[nemotron] '{cat}': kept {kept_cat}", flush=True)
            except Exception as e:
                print(f"[nemotron] '{cat}' skipped: {type(e).__name__}: {str(e)[:160]}", flush=True)
        print(f"[nemotron] license kept: {lic_kept}", flush=True)
        print(f"[nemotron] license DROPPED (BY-SA/ODC): {lic_dropped or '{}'}", flush=True)
    elif nemotron_requested and not has_token:
        print("[nemotron] requested but no HF token found — skipping (accept terms + set HF_TOKEN).",
              flush=True)

    random.Random(args.seed).shuffle(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWROTE {len(rows)} prompts -> {args.out}")
    print("by source:", counts)


if __name__ == "__main__":
    main()
