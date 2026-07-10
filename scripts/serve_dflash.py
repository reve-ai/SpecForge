#!/usr/bin/env python3
# coding=utf-8
"""Serve a trained DFlash draft head in sglang (native DFLASH speculative decoding).

A trained DFlash checkpoint is serve-ready as-is EXCEPT that sglang's DFLASH parser needs
the mask token id declared under a `dflash_config` block (our training used a reserved id,
e.g. 151669, that has no string form so sglang's default `mask_token` string can't recover
it). This script builds a lightweight "serving dir" (symlinks to the weights + a patched
config.json) and launches sglang with the right speculative args.

Validated on Qwen3-8B-Instruct + the 780K bidir DFlash checkpoint:
  * served DFlash-linear accept length = 7.46 on MATH-500 == HF reference 7.23 (parity)
  * single-stream throughput = 4.36x vs no-speculation baseline

Everything else lines up automatically: sglang derives target_layer_ids via the same formula
as training ([1,9,17,25,33] for a 36-layer target), and loads the weights (fusing q/k/v and
gate/up, stripping the `model.` prefix) without conversion.

Usage:
  # prepare a serving dir + print the launch command
  python scripts/serve_dflash.py --draft-ckpt /path/to/epoch_3_step_XXXX \
      --target-model-path /mnt/data/shared-checkpoints/Qwen3-8B-Instruct

  # prepare AND launch the server
  python scripts/serve_dflash.py --draft-ckpt ... --target-model-path ... --launch --port 30012
"""
import argparse
import json
import os
import struct
import subprocess
import sys

DEFAULT_MASK_TOKEN_ID = 151669  # the reserved mask id our train/eval used (no string form)


def _safetensors_shape(path, key):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr.get(key, {}).get("shape")


def prepare_serving_dir(draft_ckpt, served_dir, mask_token_id, target_num_layers):
    os.makedirs(served_dir, exist_ok=True)
    cfg = json.load(open(os.path.join(draft_ckpt, "config.json")))

    # --- sanity checks so failures are loud here, not deep in sglang ---
    archs = cfg.get("architectures") or []
    if "DFlashDraftModel" not in archs and "JetSpecDraftModel" not in archs:
        sys.exit(f"unexpected architectures={archs}; expected a DFlash/JetSpec draft checkpoint")
    hidden = cfg["hidden_size"]
    ndraft = cfg["num_hidden_layers"]
    fc_shape = _safetensors_shape(os.path.join(draft_ckpt, "model.safetensors"), "fc.weight")
    if fc_shape is not None:
        n_ctx = fc_shape[1] // hidden
        if fc_shape[1] % hidden or n_ctx != ndraft:
            sys.exit(f"fc.weight {fc_shape} implies {fc_shape[1]/hidden} context features but "
                     f"num_hidden_layers={ndraft}; sglang would reject this. Check the checkpoint.")
        print(f"  fc.weight {fc_shape} -> {n_ctx} context features (matches {ndraft} draft layers) OK")

    # --- the one required patch: declare the mask id sglang must use directly ---
    cfg.setdefault("dflash_config", {})["mask_token_id"] = int(mask_token_id)

    # symlink the big files (no multi-GB copy)
    for fn in ("model.safetensors", "modeling_dflash.py"):
        src = os.path.abspath(os.path.join(draft_ckpt, fn))
        dst = os.path.join(served_dir, fn)
        if os.path.exists(src):
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
    json.dump(cfg, open(os.path.join(served_dir, "config.json"), "w"), indent=2)
    print(f"  serving dir ready: {served_dir}")
    print(f"  patched dflash_config.mask_token_id = {mask_token_id}")
    print(f"  block_size={cfg.get('block_size')}  num_target_layers={cfg.get('num_target_layers', target_num_layers)}")
    return cfg


def build_launch_cmd(args, cfg):
    block_size = cfg.get("block_size") or 16
    return [
        args.sglang_python, "-m", "sglang.launch_server",
        "--model-path", args.target_model_path,
        "--speculative-algorithm", "DFLASH",
        "--speculative-draft-model-path", os.path.abspath(args.served_dir),
        "--speculative-num-draft-tokens", str(args.num_draft_tokens or block_size),
        "--dtype", "bfloat16",
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--tp-size", str(args.tp_size),
        "--host", args.host, "--port", str(args.port),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draft-ckpt", required=True, help="trained DFlash checkpoint dir (epoch_*_step_*)")
    p.add_argument("--target-model-path", required=True)
    p.add_argument("--served-dir", default=None, help="output serving dir (default: <draft-ckpt>_served)")
    p.add_argument("--mask-token-id", type=int, default=DEFAULT_MASK_TOKEN_ID)
    p.add_argument("--num-draft-tokens", type=int, default=None, help="default: draft block_size")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--mem-fraction-static", type=float, default=0.85)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30012)
    p.add_argument("--sglang-python", default="/mnt/home/sglang-013-venv/bin/python")
    p.add_argument("--launch", action="store_true", help="launch the server (otherwise just print the command)")
    args = p.parse_args()

    if args.served_dir is None:
        args.served_dir = args.draft_ckpt.rstrip("/") + "_served"

    print("preparing DFlash serving dir...")
    cfg = prepare_serving_dir(args.draft_ckpt, args.served_dir, args.mask_token_id,
                              target_num_layers=None)
    cmd = build_launch_cmd(args, cfg)
    print("\nlaunch command:\n  " + " ".join(cmd) + "\n")
    if args.launch:
        os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
