#!/bin/bash
# MCTS-RAG HSCode分类系统 - 多线程并行版
# 充分利用A800 80G GPU

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "MCTS-RAG HSCode分类系统 (多线程并行版)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv
echo ""

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

NUM_THREADS=${NUM_THREADS:-4}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-12}
MAX_DEPTH=${MAX_DEPTH:-6}
RESUME_DIR=${RESUME_DIR:-""}

# 检查检索库
if [ ! -f "$PROJECT_DIR/RAG_db/unified_bm25.pkl" ]; then
    echo "[Step 1] 构建统一检索库..."
    python code/build_unified_db.py
else
    echo "[Step 1] 统一检索库已存在"
fi

echo ""
echo "配置: THREADS=$NUM_THREADS, ROLLOUTS=$NUM_ROLLOUTS"
echo ""

RESUME_ARG=""
if [ -n "$RESUME_DIR" ]; then
    RESUME_ARG="--resume $RESUME_DIR"
fi

python code/run_parallel.py \
    --test_file "$PROJECT_DIR/data/MCTS_hard.json" \
    --rag_db "$PROJECT_DIR/RAG_db" \
    --model_ckpt "$PROJECT_DIR/models/Qwen3-8B" \
    --bert_model "$PROJECT_DIR/models/bert-base-chinese" \
    --bert_data_prefix "$PROJECT_DIR/data/suiji/data" \
    --bert_ckpt "$PROJECT_DIR/models/best_f1_macro.pt" \
    --output_dir "$PROJECT_DIR/outputs" \
    --num_threads $NUM_THREADS \
    --num_rollouts $NUM_ROLLOUTS \
    --max_depth $MAX_DEPTH \
    --save_interval 50 \
    $RESUME_ARG

echo "完成!"
