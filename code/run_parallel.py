import os
import sys
import json
import time
import argparse
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
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
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(base_dir, 'outputs'))
    
    # MCTS参数
    parser.add_argument('--num_rollouts', type=int, default=12)
    parser.add_argument('--max_depth', type=int, default=6)
    
    # 并行参数
    parser.add_argument('--num_threads', type=int, default=4,
                        help='并行线程数')
    
    # 运行参数
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--resume', type=str, default='',
                        help='断点续传目录')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_interval', type=int, default=50)
    
    return parser.parse_args()


def compute_metrics(y_true, y_pred):
    valid_pairs = [(gt, pred) for gt, pred in zip(y_true, y_pred) if pred]
    if not valid_pairs:
        return {}
    
    y_true_valid = [p[0] for p in valid_pairs]
    y_pred_valid = [p[1] for p in valid_pairs]
    
    all_labels = list(set(y_true_valid + y_pred_valid))
    le = LabelEncoder()
    le.fit(all_labels)
    
    y_true_enc = le.transform(y_true_valid)
    y_pred_enc = le.transform(y_pred_valid)
    
    acc = accuracy_score(y_true_enc, y_pred_enc)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true_enc, y_pred_enc, average='macro', zero_division=0
    )
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
    metrics = {}
    for level, prefix_len in [('L2', 2), ('L4', 4), ('L6', 6), ('L8', 8), ('L10', 10)]:
        y_true, y_pred = [], []
        for r in results:
            if 'error' in r:
                continue
            gt, pred = r.get('gt', ''), r.get('pred', '')
            if len(gt) >= prefix_len and len(pred) >= prefix_len:
                y_true.append(gt[:prefix_len])
                y_pred.append(pred[:prefix_len])
        if y_true:
            metrics[level] = compute_metrics(y_true, y_pred)
    return metrics


def get_completed_indices(output_dir):
    solutions_dir = os.path.join(output_dir, "solutions")
    if not os.path.exists(solutions_dir):
        return set()
    completed = set()
    for f in os.listdir(solutions_dir):
        if f.startswith("Q") and f.endswith(".json"):
            try:
                idx = int(f[1:5])
                completed.add(idx)
            except:
                pass
    return completed


class ParallelMCTSRunner:
    """并行MCTS运行器"""
    
    def __init__(self, predictor, llm, retriever, num_rollouts, max_depth, output_dir):
        self.predictor = predictor
        self.llm = llm
        self.retriever = retriever
        self.num_rollouts = num_rollouts
        self.max_depth = max_depth
        self.output_dir = output_dir
        
        # 线程锁
        self.llm_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.file_lock = threading.Lock()
        
        # 统计
        self.results = []
        self.correct = 0
        self.correct_4 = 0
        self.correct_6 = 0
        self.processed = 0
        
    def process_item(self, idx, problem, gt_code):
        """处理单个样本（线程安全）"""
        from mcts_optimized_v2 import run_mcts_inference, MCTSNode
        
        item_start = time.time()
        
        try:
            # 创建线程局部的节点计数器
            MCTSNode._id_counter = idx * 10000
            
            # 运行MCTS（LLM调用内部会使用锁）
            solutions = run_mcts_inference(
                problem=problem,
                predictor=self.predictor,
                llm=self.llm,
                retriever=self.retriever,
                num_rollouts=self.num_rollouts,
                max_depth=self.max_depth,
                verbose=False
            )
            
            pred_code = solutions[0]['hs'] if solutions else ""
            
            is_correct = (pred_code == gt_code)
            is_correct_4 = (pred_code[:4] == gt_code[:4]) if len(pred_code) >= 4 and len(gt_code) >= 4 else False
            is_correct_6 = (pred_code[:6] == gt_code[:6]) if len(pred_code) >= 6 and len(gt_code) >= 6 else False
            
            result = {
                'id': idx,
                'gt': gt_code,
                'pred': pred_code,
                'correct': is_correct,
                'correct_4': is_correct_4,
                'correct_6': is_correct_6,
                'time': round(time.time() - item_start, 2),
                'solutions': solutions[:5]
            }
            
            # 保存结果（线程安全）
            with self.file_lock:
                sol_path = os.path.join(self.output_dir, "solutions", f"Q{idx:04d}.json")
                with open(sol_path, 'w', encoding='utf-8') as f:
                    json.dump(solutions, f, ensure_ascii=False, indent=2)
            
            # 更新统计
            with self.result_lock:
                self.results.append(result)
                if is_correct:
                    self.correct += 1
                if is_correct_4:
                    self.correct_4 += 1
                if is_correct_6:
                    self.correct_6 += 1
                self.processed += 1
            
            return result
            
        except Exception as e:
            import traceback
            error_result = {'id': idx, 'gt': gt_code, 'pred': '', 'error': str(e)}
            with self.result_lock:
                self.results.append(error_result)
                self.processed += 1
            return error_result
    
    def save_checkpoint(self):
        """保存检查点"""
        with self.result_lock:
            checkpoint = {
                'processed': self.processed,
                'correct': self.correct,
                'correct_4': self.correct_4,
                'correct_6': self.correct_6,
                'results': self.results,
                'llm_calls': self.llm.call_count
            }
        
        with self.file_lock:
            with open(os.path.join(self.output_dir, "checkpoint.json"), 'w') as f:
                json.dump(checkpoint, f)


class ThreadSafeLLMInterface:
    """线程安全的LLM接口包装器"""
    
    def __init__(self, llm, lock):
        self._llm = llm
        self._lock = lock
        self.call_count = 0
        
    def action_A4_arbitrate(self, desc, candidates):
        with self._lock:
            result = self._llm.action_A4_arbitrate(desc, candidates)
            self.call_count = self._llm.call_count
            return result
    
    def action_A5_choose(self, desc, prefix, candidates, target_len):
        with self._lock:
            result = self._llm.action_A5_choose(desc, prefix, candidates, target_len)
            self.call_count = self._llm.call_count
            return result
    
    def action_A6_verify(self, desc, code):
        with self._lock:
            result = self._llm.action_A6_verify(desc, code)
            self.call_count = self._llm.call_count
            return result


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MCTS-RAG HSCode分类系统 (多线程并行版)")
    print("=" * 60)
    
    # 加载测试数据
    print("\n[1/3] 加载测试数据...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"       样本数: {len(test_data)}")
    
    # 输出目录
    if args.resume:
        output_dir = args.resume
        print(f"\n[断点续传] 从 {output_dir} 继续")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"v4_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "solutions"), exist_ok=True)
    
    # 获取已完成样本
    completed_indices = get_completed_indices(output_dir)
    print(f"       已完成: {len(completed_indices)} 样本")
    
    # 待处理样本
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(test_data)
    
    pending_items = []
    for i in range(start_idx, min(end_idx, len(test_data))):
        if i not in completed_indices:
            item = test_data[i]
            pending_items.append((i, item.get('problem', ''), item.get('solution', '')))
    
    print(f"       待处理: {len(pending_items)} 样本")
    
    if not pending_items:
        print("\n所有样本已处理完成!")
        return
    
    # 加载模型
    print("\n[2/3] 加载模型...")
    from mcts_optimized_v2 import UnifiedRetriever, BertPredictor, LLMInterface
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    retriever = UnifiedRetriever(args.rag_db)
    predictor = BertPredictor(args.bert_model, args.bert_data_prefix, args.bert_ckpt)
    
    print(f"[LLM] 加载模型: {args.model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    llm_base = LLMInterface(tokenizer, model, retriever)
    llm_lock = threading.Lock()
    llm = ThreadSafeLLMInterface(llm_base, llm_lock)
    
    print(f"[LLM] 已用显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    
    # 创建并行运行器
    runner = ParallelMCTSRunner(
        predictor=predictor,
        llm=llm,
        retriever=retriever,
        num_rollouts=args.num_rollouts,
        max_depth=args.max_depth,
        output_dir=output_dir
    )
    
    # 加载已有结果
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            ckpt = json.load(f)
            runner.results = ckpt.get('results', [])
            runner.correct = ckpt.get('correct', 0)
            runner.correct_4 = ckpt.get('correct_4', 0)
            runner.correct_6 = ckpt.get('correct_6', 0)
            runner.processed = ckpt.get('processed', 0)
    
    print("\n[3/3] 并行推理...")
    print("=" * 60)
    print(f"线程数: {args.num_threads}")
    print(f"Rollouts: {args.num_rollouts}, MaxDepth: {args.max_depth}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 使用ThreadPoolExecutor并行处理
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        for idx, problem, gt_code in pending_items:
            future = executor.submit(runner.process_item, idx, problem, gt_code)
            futures[future] = idx
        
        # 进度条
        with tqdm(total=len(pending_items), desc="Processing") as pbar:
            checkpoint_counter = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    pbar.update(1)
                    
                    # 更新进度条描述
                    with runner.result_lock:
                        acc = runner.correct / max(runner.processed, 1)
                        pbar.set_postfix({
                            'Acc': f'{acc:.2%}',
                            'LLM': llm.call_count
                        })
                    
                    checkpoint_counter += 1
                    if checkpoint_counter % args.save_interval == 0:
                        runner.save_checkpoint()
                        
                except Exception as e:
                    print(f"\n[Q{idx}] Error: {e}")
                    pbar.update(1)
    
    # 最终保存
    runner.save_checkpoint()
    
    elapsed = time.time() - start_time
    
    # 计算指标
    print("\n" + "=" * 60)
    print("计算多分类指标...")
    print("=" * 60)
    
    y_true_10 = [r['gt'] for r in runner.results if 'error' not in r]
    y_pred_10 = [r['pred'] for r in runner.results if 'error' not in r]
    
    metrics_10 = compute_metrics(y_true_10, y_pred_10)
    hier_metrics = compute_hierarchical_metrics(runner.results)
    
    print("\n" + "=" * 60)
    print("最终结果 - 多分类指标")
    print("=" * 60)
    print(f"总样本: {runner.processed}")
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
    print(f"  平均每样本: {elapsed/max(len(pending_items),1):.2f}s")
    print(f"  吞吐量: {len(pending_items)/elapsed*60:.1f} samples/min")
    print(f"  LLM调用: {llm.call_count}")
    print(f"  GPU显存: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB (peak)")
    
    # 保存结果
    final = {
        'total': runner.processed,
        'metrics_10': metrics_10,
        'hierarchical_metrics': hier_metrics,
        'elapsed': elapsed,
        'throughput': len(pending_items)/elapsed*60,
        'llm_calls': llm.call_count,
        'gpu_memory_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
        'results': runner.results
    }
    
    with open(os.path.join(output_dir, "final_result.json"), 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, "metrics_summary.json"), 'w', encoding='utf-8') as f:
        json.dump({'L10': metrics_10, 'hierarchical': hier_metrics}, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果保存: {output_dir}")


if __name__ == "__main__":
    main()
