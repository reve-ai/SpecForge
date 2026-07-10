#!/bin/bash
# Train a JetSpec draft head (causal parallel tree drafting) for Qwen3-8B.
# JetSpec = DFlash head + causal within-block mask + soft-label distillation. Evaluate the
# trained head with tree decoding via scripts/eval_accept_hf.py --mode tree.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32
NUM_GPUS=${1:-8}
ATTENTION_BACKEND=${2:-flex_attention}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_jetspec.py \
    --target-model-path Qwen/Qwen3-8B \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-jetspec.json \
    --train-data-path $ROOT_DIR/cache/dataset/regen_math_code.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-jetspec \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-length 3072 \
    --chat-template qwen \
    --distill --distill-temp 1.0 --distill-alpha 0.0 \
    --attention-backend $ATTENTION_BACKEND \
    --log-interval 50 \
    --save-interval 1000 \
    --report-to wandb \
    --wandb-project specforge-qwen3-8b-jetspec \
    --wandb-name qwen3-8b-jetspec

# Evaluate acceptance length with tree drafting (after training), e.g.:
#   python $ROOT_DIR/scripts/eval_accept_hf.py \
#     --target-model-path Qwen/Qwen3-8B \
#     --draft-path $ROOT_DIR/outputs/qwen3-8b-jetspec/epoch_3_step_XXXX \
#     --head-type causal --mask-token-id <id> \
#     --prompts-file <math500_prompts.jsonl> --num-prompts 50 --nothink \
#     --mode tree --tree-width 8 --budget 256
