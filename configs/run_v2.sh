#!/bin/bash
# MCTS-RAG HSCode分类系统 - 优化版V2运行脚本
# 针对A800 80G GPU优化

set -e

# 项目目录
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "MCTS-RAG HSCode分类系统 (优化版V2)"
echo "============================================================"
echo "项目目录: $PROJECT_DIR"
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# 环境变量优化
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8

# 参数配置
NUM_ROLLOUTS=${NUM_ROLLOUTS:-12}
MAX_DEPTH=${MAX_DEPTH:-6}
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:--1}

# 检查统一检索库
if [ ! -f "$PROJECT_DIR/RAG_db/unified_bm25.pkl" ]; then
    echo "[Step 1] 构建统一检索库..."
    python code/build_unified_db.py
    echo ""
else
    echo "[Step 1] 统一检索库已存在，跳过构建"
fi

# 显示配置
echo ""
echo "============================================================"
echo "运行配置"
echo "============================================================"
echo "  NUM_ROLLOUTS: $NUM_ROLLOUTS"
echo "  MAX_DEPTH: $MAX_DEPTH"
echo "  START_IDX: $START_IDX"
echo "  END_IDX: $END_IDX"
echo ""

# 运行推理
echo "[Step 2] 开始推理..."
echo ""

python code/run_optimized_v2.py \
    --test_file "$PROJECT_DIR/data/MCTS_hard.json" \
    --rag_db "$PROJECT_DIR/RAG_db" \
    --model_ckpt "$PROJECT_DIR/models/Qwen3-8B" \
    --bert_model "$PROJECT_DIR/models/bert-base-chinese" \
    --bert_data_prefix "$PROJECT_DIR/data/suiji/data" \
    --bert_ckpt "$PROJECT_DIR/models/best_f1_macro.pt" \
    --embedding_model "$PROJECT_DIR/models/bge-base-zh-v1.5" \
    --output_dir "$PROJECT_DIR/outputs" \
    --num_rollouts $NUM_ROLLOUTS \
    --max_depth $MAX_DEPTH \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --save_interval 100

echo ""
echo "============================================================"
echo "运行完成!"
echo "============================================================"

# 显示最新结果
LATEST_OUTPUT=$(ls -td "$PROJECT_DIR/outputs/v2_"* 2>/dev/null | head -1)
if [ -n "$LATEST_OUTPUT" ]; then
    echo "结果目录: $LATEST_OUTPUT"
    echo ""
    if [ -f "$LATEST_OUTPUT/metrics_summary.json" ]; then
        echo "指标摘要:"
        cat "$LATEST_OUTPUT/metrics_summary.json"
    fi
fi
