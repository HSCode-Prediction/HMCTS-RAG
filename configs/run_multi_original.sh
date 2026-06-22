#!/bin/bash
# MCTS-RAG多进程并行版 - 充分利用A800 80G
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "MCTS-RAG HSCode分类系统 (多进程并行版)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 3个进程，每个约20GB显存，总共60GB
NUM_PROCS=${NUM_PROCS:-3}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_DIR/outputs/v2_20251223_095528"}

echo ""
echo "配置: NUM_PROCS=$NUM_PROCS"
echo "输出目录: $OUTPUT_DIR"
echo ""

python code/run_multi_gpu.py \
    --test_file "$PROJECT_DIR/data/MCTS_hard.json" \
    --output_dir "$OUTPUT_DIR" \
    --num_procs $NUM_PROCS \
    --num_rollouts 10 \
    --max_depth 6

echo "完成!"
