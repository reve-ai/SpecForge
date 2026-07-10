#!/usr/bin/env python3
# coding=utf-8
"""Train a JetSpec draft head (causal parallel tree drafting).

JetSpec = the DFlash head trained with a **causal** within-block mask (each draft position
attends only to the prefix + earlier positions in its block), so the per-depth marginals follow
the target's autoregressive factorization and form coherent speculation trees.

This is a thin entrypoint over ``scripts/train_dflash.py`` that defaults ``--head-type`` to
``causal`` — so you train a JetSpec head with no extra flags. All ``train_dflash.py`` arguments
apply (soft-label distillation via ``--distill`` is recommended, per the paper). Pass a JetSpec
draft config (``configs/qwen3-8b-jetspec.json``) or any DFlash config; the causal head is forced.

Inference / acceptance eval (tree drafting): ``scripts/eval_accept_hf.py --mode tree``.

Example:
  torchrun --standalone --nproc_per_node 8 scripts/train_jetspec.py \
    --target-model-path /path/to/Qwen3-8B-Instruct \
    --draft-config-path configs/qwen3-8b-jetspec.json \
    --train-data-path ./cache/dataset/regen.jsonl \
    --output-dir ./outputs/qwen3-8b-jetspec \
    --num-epochs 3 --batch-size 4 --max-length 3072 \
    --learning-rate 6e-4 --warmup-ratio 0.04 --distill
"""
import os
import sys

# Reuse the full DFlash training pipeline (JetSpec differs only by the causal head + distill).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default to the causal head unless the caller overrides it explicitly.
if "--head-type" not in sys.argv:
    sys.argv += ["--head-type", "causal"]

import train_dflash  # noqa: E402

if __name__ == "__main__":
    train_dflash.main()
