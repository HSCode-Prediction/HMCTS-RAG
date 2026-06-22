
import os
import sys
import json
import time
import argparse
from tqdm import tqdm
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcts_optimized_v2 import (
    UnifiedRetriever, BertPredictor, LLMInterface,
    run_mcts_inference, MCTSNode
)


def parse_args():
    parser = argparse.ArgumentParser()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 路径
    parser.add_argument('--test_file', type=str,
                        default=os.path.join(base_dir, 'data', 'MCTS_hard.json'))
    parser.add_argument('--rag_db', type=str,
                        default=os.path.join(base_dir, 'RAG_db'))
    parser.add_argument('--model_ckpt', type=str,
                        default=os.path.join(base_dir, 'models', 'Qwen3-8B'))
    parser.add_argument('--bert_model', type=str,
                        default=os.path.join(base_dir, 'models', 'bert-base-chinese'))
    parser.add_argument('--bert_data_prefix', type=str,
                        default=os.path.join(base_dir, 'data', 'suiji', 'data'))
    parser.add_argument('--bert_ckpt', type=str,
                        default=os.path.join(base_dir, 'models', 'best_f1_macro.pt'))
    parser.add_argument('--embedding_model', type=str,
                        default=os.path.join(base_dir, 'models', 'bge-base-zh-v1.5'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(base_dir, 'outputs'))
    
    # MCTS参数
    parser.add_argument('--num_rollouts', type=int, default=12)
    parser.add_argument('--max_depth', type=int, default=6)
    
    # 运行参数
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_interval', type=int, default=100)
    
    return parser.parse_args()


def compute_metrics(y_true, y_pred):
    """计算多分类指标"""
    # 过滤空预测
    valid_pairs = [(gt, pred) for gt, pred in zip(y_true, y_pred) if pred]
    if not valid_pairs:
        return {}
    
    y_true_valid = [p[0] for p in valid_pairs]
    y_pred_valid = [p[1] for p in valid_pairs]
    
    # 编码标签
    all_labels = list(set(y_true_valid + y_pred_valid))
    le = LabelEncoder()
    le.fit(all_labels)
    
    y_true_enc = le.transform(y_true_valid)
    y_pred_enc = le.transform(y_pred_valid)
    
    # Accuracy
    acc = accuracy_score(y_true_enc, y_pred_enc)
    
    # Macro 指标
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true_enc, y_pred_enc, average='macro', zero_division=0
    )
    
    # Weighted 指标
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true_enc, y_pred_enc, average='weighted', zero_division=0
    )
    
    return {
        'Acc': round(acc, 4),
        'MP': round(mp, 4),
        'MR': round(mr, 4),
        'MF1': round(mf1, 4),
        'WP': round(wp, 4),
        'WR': round(wr, 4),
        'WF1': round(wf1, 4),
        'valid_samples': len(valid_pairs),
        'total_classes': len(all_labels)
    }


def compute_hierarchical_metrics(results):
    """计算各层级的指标"""
    metrics = {}
    
    for level, prefix_len in [('L2', 2), ('L4', 4), ('L6', 6), ('L8', 8), ('L10', 10)]:
        y_true = []
        y_pred = []
        
        for r in results:
            if 'error' in r:
                continue
            gt = r.get('gt', '')
            pred = r.get('pred', '')
            
            if len(gt) >= prefix_len and len(pred) >= prefix_len:
                y_true.append(gt[:prefix_len])
                y_pred.append(pred[:prefix_len])
        
        if y_true:
            level_metrics = compute_metrics(y_true, y_pred)
            metrics[level] = level_metrics
    
    return metrics


def load_llm(model_path: str):
    """加载LLM模型 - 优化为A800 80G"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"[LLM] 加载模型: {model_path}")
    print(f"[LLM] GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 优化加载配置 for A800 80G
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # A800支持bf16更高效
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # 启用torch.compile加速 (可选)
    # model = torch.compile(model, mode="reduce-overhead")
    
    print(f"[LLM]  模型加载完成")
    print(f"[LLM] 已用显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    return tokenizer, model


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MCTS-RAG HSCode分类系统 (优化版V2)")
    print("=" * 60)
    
    # 创建输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"v2_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "solutions"), exist_ok=True)
    
    # 加载测试数据
    print("\n[1/4] 加载测试数据...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"       样本数: {len(test_data)}")
    
    # 加载检索器
    print("\n[2/4] 加载混合检索器...")
    embedding_path = args.embedding_model if os.path.exists(args.embedding_model) else None
    retriever = UnifiedRetriever(args.rag_db, embedding_path)
    
    # 加载BERT
    print("\n[3/4] 加载BERT预测器...")
    predictor = BertPredictor(
        args.bert_model,
        args.bert_data_prefix,
        args.bert_ckpt
    )
    
    # 加载LLM
    print("\n[4/4] 加载LLM...")
    tokenizer, model = load_llm(args.model_ckpt)
    llm = LLMInterface(tokenizer, model, retriever)
    
    # 确定处理范围
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(test_data)
    
    print("\n" + "=" * 60)
    print(f"开始推理: {start_idx} -> {end_idx}")
    print(f"Rollouts: {args.num_rollouts}, MaxDepth: {args.max_depth}")
    print("=" * 60)
    
    # 统计
    results = []
    correct = 0
    correct_4 = 0  # 4位准确率
    correct_6 = 0  # 6位准确率
    total = 0
    start_time = time.time()
    
    # 重置节点计数
    MCTSNode._id_counter = 0
    
    for i in tqdm(range(start_idx, min(end_idx, len(test_data))), desc="Processing"):
        item = test_data[i]
        problem = item.get('problem', '')
        gt_code = item.get('solution', '')
        
        item_start = time.time()
        
        try:
            # MCTS推理
            solutions = run_mcts_inference(
                problem=problem,
                predictor=predictor,
                llm=llm,
                retriever=retriever,
                num_rollouts=args.num_rollouts,
                max_depth=args.max_depth,
                verbose=args.verbose
            )
            
            # 获取预测结果
            pred_code = solutions[0]['hs'] if solutions else ""
            
            is_correct = (pred_code == gt_code)
            is_correct_4 = (pred_code[:4] == gt_code[:4]) if len(pred_code) >= 4 and len(gt_code) >= 4 else False
            is_correct_6 = (pred_code[:6] == gt_code[:6]) if len(pred_code) >= 6 and len(gt_code) >= 6 else False
            
            if is_correct:
                correct += 1
            if is_correct_4:
                correct_4 += 1
            if is_correct_6:
                correct_6 += 1
            total += 1
            
            result = {
                'id': i,
                'gt': gt_code,
                'pred': pred_code,
                'correct': is_correct,
                'correct_4': is_correct_4,
                'correct_6': is_correct_6,
                'time': round(time.time() - item_start, 2),
                'solutions': solutions[:5]
            }
            results.append(result)
            
            # 保存单个结果
            sol_path = os.path.join(output_dir, "solutions", f"Q{i:04d}.json")
            with open(sol_path, 'w', encoding='utf-8') as f:
                json.dump(solutions, f, ensure_ascii=False, indent=2)
            
            # 定期保存checkpoint
            if (i + 1) % args.save_interval == 0:
                checkpoint = {
                    'last_idx': i,
                    'results': results,
                    'correct': correct,
                    'correct_4': correct_4,
                    'correct_6': correct_6,
                    'total': total,
                    'llm_calls': llm.call_count
                }
                with open(os.path.join(output_dir, "checkpoint.json"), 'w') as f:
                    json.dump(checkpoint, f)
                print(f"\n[Checkpoint] 已保存到Q{i}")
            
            # 定期打印
            if args.verbose or i % 20 == 0:
                acc = correct / total if total > 0 else 0
                acc4 = correct_4 / total if total > 0 else 0
                print(f"\n[Q{i}] GT:{gt_code} Pred:{pred_code} {'✓' if is_correct else '✗'}")
                print(f"       Acc10:{acc:.2%} Acc4:{acc4:.2%} LLM:{llm.call_count}")
                if solutions:
                    print(f"       Action: {solutions[0].get('action', 'N/A')}")
            
        except Exception as e:
            import traceback
            print(f"\n[Q{i}] Error: {e}")
            traceback.print_exc()
            results.append({'id': i, 'error': str(e)})
            total += 1
    
    # 最终统计
    elapsed = time.time() - start_time
    
    # 计算多分类指标
    print("\n" + "=" * 60)
    print("计算多分类指标...")
    print("=" * 60)
    
    y_true_10 = [r['gt'] for r in results if 'error' not in r]
    y_pred_10 = [r['pred'] for r in results if 'error' not in r]
    
    # 10位指标
    metrics_10 = compute_metrics(y_true_10, y_pred_10)
    
    # 各层级指标
    hier_metrics = compute_hierarchical_metrics(results)
    
    print("\n" + "=" * 60)
    print("最终结果 - 多分类指标")
    print("=" * 60)
    print(f"总样本: {total}")
    print(f"有效样本: {metrics_10.get('valid_samples', 0)}")
    print(f"类别数: {metrics_10.get('total_classes', 0)}")
    print(f"\n10位HSCode指标:")
    print(f"  Acc  = {metrics_10.get('Acc', 0):.4f}")
    print(f"  MP   = {metrics_10.get('MP', 0):.4f}")
    print(f"  MR   = {metrics_10.get('MR', 0):.4f}")
    print(f"  MF1  = {metrics_10.get('MF1', 0):.4f}")
    print(f"  WP   = {metrics_10.get('WP', 0):.4f}")
    print(f"  WR   = {metrics_10.get('WR', 0):.4f}")
    print(f"  WF1  = {metrics_10.get('WF1', 0):.4f}")
    
    print(f"\n各层级指标:")
    for level in ['L2', 'L4', 'L6', 'L8', 'L10']:
        if level in hier_metrics:
            m = hier_metrics[level]
            print(f"  {level}: Acc={m.get('Acc',0):.4f} MF1={m.get('MF1',0):.4f} WF1={m.get('WF1',0):.4f}")
    
    print(f"\n运行统计:")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  平均每样本: {elapsed/max(total,1):.2f}s")
    print(f"  LLM调用次数: {llm.call_count}")
    print(f"  GPU显存: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB (peak)")
    
    # 保存最终结果
    final = {
        'total': total,
        'valid_samples': metrics_10.get('valid_samples', 0),
        'total_classes': metrics_10.get('total_classes', 0),
        'metrics_10': metrics_10,
        'hierarchical_metrics': hier_metrics,
        'elapsed': elapsed,
        'avg_time': elapsed / max(total, 1),
        'llm_calls': llm.call_count,
        'gpu_memory_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
        'results': results
    }
    
    with open(os.path.join(output_dir, "final_result.json"), 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    
    # 保存指标摘要
    summary = {
        'L10': metrics_10,
        'hierarchical': hier_metrics,
        'runtime': {
            'total_time': elapsed,
            'avg_time': elapsed / max(total, 1),
            'llm_calls': llm.call_count
        }
    }
    with open(os.path.join(output_dir, "metrics_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果保存: {output_dir}")
    print(f"  - final_result.json")
    print(f"  - metrics_summary.json")
    print(f"  - solutions/Q*.json")


if __name__ == "__main__":
    main()
