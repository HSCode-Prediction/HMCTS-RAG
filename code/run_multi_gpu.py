
import os
import sys
import json
import time
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser.add_argument('--test_file', type=str,
                        default='/data/A/data/test_hard.json')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(base_dir, 'outputs', 'v2_20251223_095528'))
    parser.add_argument('--num_procs', type=int, default=10,
                        help='并行进程数，2xA800 160GB建议10个进程')
    parser.add_argument('--num_gpus', type=int, default=2,
                        help='GPU数量')
    parser.add_argument('--num_rollouts', type=int, default=10)
    parser.add_argument('--max_depth', type=int, default=6)
    
    return parser.parse_args()


def get_completed_indices(output_dir):
    """获取已完成的样本索引 - 支持新格式"""
    if not os.path.exists(output_dir):
        return set()
    completed = set()
    for f in os.listdir(output_dir):
        # 新格式: Question_XXXX_solutions.json
        if f.startswith("Question_") and f.endswith("_solutions.json"):
            try:
                idx = int(f.split('_')[1])
                completed.add(idx)
            except:
                pass
    return completed


def run_worker(worker_id, indices, args_dict):
    """运行单个worker进程 - 支持2xA800 80GB"""
    import torch
    from tqdm import tqdm
    
    # 设置环境
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 根据worker_id分配GPU
    num_gpus = args_dict.get('num_gpus', 2)
    gpu_id = worker_id % num_gpus
    device = f"cuda:{gpu_id}"
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mcts_optimized_v2 import (
        UnifiedRetriever, BertPredictor, LLMInterface,
        run_mcts_inference, MCTSNode
    )
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"[Worker {worker_id}] 启动，分配GPU {gpu_id}，处理 {len(indices)} 样本")
    
    # 加载测试数据
    with open(args_dict['test_file'], 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 加载模型 - 指定设备
    retriever = UnifiedRetriever(os.path.join(base_dir, 'RAG_db'))
    predictor = BertPredictor(
        '/data/A/models/bert-base-chinese',
        '/data/AAA/data/suiji/data',
        '/data/A/models/best_f1_macro.pt',
        device=device
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        '/data/A/models/Meta-Llama-3.1-8B-Instruct',
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载LLM到指定GPU - Meta-Llama-3.1-8B-Instruct
    model = AutoModelForCausalLM.from_pretrained(
        '/data/A/models/Meta-Llama-3.1-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map={"": gpu_id},
        trust_remote_code=True,
    )
    model.eval()
    llm = LLMInterface(tokenizer, model, retriever)
    
    print(f"[Worker {worker_id}] 模型加载完成，显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # 处理样本 - 实时ACC监控
    results = []
    output_dir = args_dict['output_dir']
    correct_count = 0
    topk_correct = 0
    processed = 0
    
    for idx in tqdm(indices, desc=f"W{worker_id}|GPU{gpu_id}", position=worker_id):
        item = test_data[idx]
        problem = item.get('problem', '')
        gt_code = item.get('gold_solution', item.get('solution', ''))
        
        try:
            MCTSNode._id_counter = idx * 10000
            solutions = run_mcts_inference(
                problem=problem,
                predictor=predictor,
                llm=llm,
                retriever=retriever,
                num_rollouts=args_dict['num_rollouts'],
                max_depth=args_dict['max_depth'],
                verbose=False
            )
            
            pred_code = solutions[0]['hs'] if solutions else ""
            
            # === 后处理：添加BERT Top-K候选以提高覆盖率 ===
            seen_hs = set(s.get('hs', '') for s in solutions)
            bert_preds = predictor.predict(problem, 10, top_k=10)
            for code, conf in bert_preds:
                if len(code) >= 10:
                    hs_code = code[:10]
                    if hs_code not in seen_hs:
                        seen_hs.add(hs_code)
                        solutions.append({
                            'hs': hs_code,
                            'reward': round(conf * 0.5, 4),
                            'confidence': round(conf, 4),
                            'action': 'BERT_TopK',
                            'source': 'bert'
                        })
            
            # 实时统计
            is_correct = pred_code == gt_code
            all_hs = [s.get('hs', '') for s in solutions]
            is_topk = gt_code in all_hs
            
            if is_correct:
                correct_count += 1
            if is_topk:
                topk_correct += 1
            processed += 1
            
            # 每10个样本输出一次实时ACC
            if processed % 10 == 0:
                acc = correct_count / processed * 100
                topk_acc = topk_correct / processed * 100
                tqdm.write(f"[W{worker_id}] ACC: {acc:.1f}% | TopK: {topk_acc:.1f}% ({processed}/{len(indices)})")
            
            result = {
                'id': idx,
                'gt': gt_code,
                'pred': pred_code,
                'correct': is_correct,
                'correct_4': (pred_code[:4] == gt_code[:4]) if len(pred_code) >= 4 and len(gt_code) >= 4 else False,
                'correct_6': (pred_code[:6] == gt_code[:6]) if len(pred_code) >= 6 and len(gt_code) >= 6 else False,
                'topk_correct': is_topk,
                'num_solutions': len(solutions),
                'solutions': solutions[:5]
            }
            results.append(result)
            
            # 保存结果 - 与answer_sheets格式一致
            # 1. Answer文件
            answer_data = {
                'id': item.get('id', idx),
                'problem': problem,
                'gold_solution': gt_code,
                'gold_answer': gt_code
            }
            ans_path = os.path.join(output_dir, f"Question_{idx:04d}_Answer.json")
            with open(ans_path, 'w', encoding='utf-8') as f:
                json.dump(answer_data, f, ensure_ascii=False, indent=2)
            
            # 2. Solutions文件 - 格式与answer_sheets一致
            formatted_solutions = []
            for sol in solutions:
                formatted_sol = {
                    'hs': sol.get('hs', ''),
                    'reward': sol.get('reward', sol.get('confidence', 0)),
                    'conf': sol.get('confidence', sol.get('reward', 0)),
                    'trace': {
                        '0': {
                            'q': problem,
                            'path': [f"[{sol.get('action', 'Direct')}] -> {sol.get('hs', '')} (conf:{sol.get('confidence', 0):.4f})"]
                        }
                    }
                }
                formatted_solutions.append(formatted_sol)
            
            sol_path = os.path.join(output_dir, f"Question_{idx:04d}_solutions.json")
            with open(sol_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_solutions, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            results.append({'id': idx, 'gt': gt_code, 'pred': '', 'error': str(e)})
    
    # 最终统计
    final_acc = correct_count / max(1, processed) * 100
    final_topk = topk_correct / max(1, processed) * 100
    print(f"[Worker {worker_id}] 完成 {len(results)} 样本 | Top1: {final_acc:.2f}% | TopK: {final_topk:.2f}%")
    return results


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MCTS-RAG HSCode分类系统 (多进程并行版)")
    print("=" * 60)
    
    # 加载测试数据
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"总样本数: {len(test_data)}")
    
    # 获取已完成样本
    os.makedirs(args.output_dir, exist_ok=True)
    completed = get_completed_indices(args.output_dir)
    print(f"已完成: {len(completed)} 样本")
    
    # 待处理样本
    pending = [i for i in range(len(test_data)) if i not in completed]
    print(f"待处理: {len(pending)} 样本")
    
    if not pending:
        print("所有样本已处理完成!")
        return
    
    # 分配给多个进程
    num_procs = min(args.num_procs, len(pending))
    chunk_size = len(pending) // num_procs
    chunks = []
    for i in range(num_procs):
        start = i * chunk_size
        end = start + chunk_size if i < num_procs - 1 else len(pending)
        chunks.append(pending[start:end])
    
    print(f"\n并行进程数: {num_procs}")
    for i, chunk in enumerate(chunks):
        print(f"  Worker {i}: {len(chunk)} 样本 ({chunk[0]} - {chunk[-1]})")
    
    args_dict = {
        'test_file': args.test_file,
        'output_dir': args.output_dir,
        'num_rollouts': args.num_rollouts,
        'max_depth': args.max_depth,
        'num_gpus': args.num_gpus
    }
    
    print("\n开始并行处理...")
    start_time = time.time()
    
    # 使用spawn方式启动多进程
    mp.set_start_method('spawn', force=True)
    
    all_results = []
    with ProcessPoolExecutor(max_workers=num_procs) as executor:
        futures = {
            executor.submit(run_worker, i, chunks[i], args_dict): i 
            for i in range(num_procs)
        }
        
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f"[Worker {worker_id}] 返回 {len(results)} 结果")
            except Exception as e:
                print(f"[Worker {worker_id}] 错误: {e}")
    
    elapsed = time.time() - start_time
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("汇总结果...")
    print("=" * 60)
    
    # 重新加载所有结果 - 新格式
    final_results = []
    top1_correct = 0
    topk_correct = 0
    
    for i in range(len(test_data)):
        sol_path = os.path.join(args.output_dir, f"Question_{i:04d}_solutions.json")
        if os.path.exists(sol_path):
            with open(sol_path, 'r') as f:
                solutions = json.load(f)
            pred = solutions[0]['hs'] if solutions else ""
            gt = test_data[i].get('gold_solution', test_data[i].get('solution', ''))
            
            is_correct = pred == gt
            if is_correct:
                top1_correct += 1
            
            # Top-K
            all_hs = [s.get('hs', '') for s in solutions]
            if gt in all_hs:
                topk_correct += 1
            
            final_results.append({
                'id': i,
                'gt': gt,
                'pred': pred,
                'correct': is_correct,
                'correct_4': (pred[:4] == gt[:4]) if len(pred) >= 4 and len(gt) >= 4 else False,
                'correct_6': (pred[:6] == gt[:6]) if len(pred) >= 6 and len(gt) >= 6 else False
            })
    
    print(f"\nTop-1 Accuracy: {top1_correct}/{len(final_results)} = {top1_correct/max(1,len(final_results))*100:.2f}%")
    print(f"Top-K Accuracy: {topk_correct}/{len(final_results)} = {topk_correct/max(1,len(final_results))*100:.2f}%")
    
    # 计算指标
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from sklearn.preprocessing import LabelEncoder
    
    def compute_metrics(y_true, y_pred):
        valid = [(gt, pred) for gt, pred in zip(y_true, y_pred) if pred]
        if not valid:
            return {}
        y_true_v = [p[0] for p in valid]
        y_pred_v = [p[1] for p in valid]
        
        all_labels = list(set(y_true_v + y_pred_v))
        le = LabelEncoder()
        le.fit(all_labels)
        y_true_e = le.transform(y_true_v)
        y_pred_e = le.transform(y_pred_v)
        
        acc = accuracy_score(y_true_e, y_pred_e)
        mp, mr, mf1, _ = precision_recall_fscore_support(y_true_e, y_pred_e, average='macro', zero_division=0)
        wp, wr, wf1, _ = precision_recall_fscore_support(y_true_e, y_pred_e, average='weighted', zero_division=0)
        
        return {
            'Acc': round(acc, 4), 'MP': round(mp, 4), 'MR': round(mr, 4), 'MF1': round(mf1, 4),
            'WP': round(wp, 4), 'WR': round(wr, 4), 'WF1': round(wf1, 4),
            'valid_samples': len(valid), 'total_classes': len(all_labels)
        }
    
    # 10位指标
    y_true = [r['gt'] for r in final_results]
    y_pred = [r['pred'] for r in final_results]
    metrics_10 = compute_metrics(y_true, y_pred)
    
    # 各层级指标
    hier_metrics = {}
    for level, plen in [('L2', 2), ('L4', 4), ('L6', 6), ('L8', 8), ('L10', 10)]:
        yt = [r['gt'][:plen] for r in final_results if len(r['gt']) >= plen and len(r['pred']) >= plen]
        yp = [r['pred'][:plen] for r in final_results if len(r['gt']) >= plen and len(r['pred']) >= plen]
        if yt:
            hier_metrics[level] = compute_metrics(yt, yp)
    
    # 输出结果
    print(f"\n总样本: {len(final_results)}")
    print(f"有效样本: {metrics_10.get('valid_samples', 0)}")
    print(f"类别数: {metrics_10.get('total_classes', 0)}")
    
    print(f"\n{'='*60}")
    print("10位HSCode指标:")
    print(f"{'='*60}")
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
    print(f"  总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  吞吐量: {len(pending)/elapsed*60:.1f} samples/min")
    
    # 保存结果
    final = {
        'total': len(final_results),
        'metrics_10': metrics_10,
        'hierarchical_metrics': hier_metrics,
        'elapsed': elapsed,
        'results': final_results
    }
    
    with open(os.path.join(args.output_dir, "final_result.json"), 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    
    # 保存指标摘要
    summary = {
        'L10': metrics_10,
        'hierarchical': hier_metrics,
        'runtime': {'elapsed': elapsed, 'throughput': len(pending)/elapsed*60}
    }
    with open(os.path.join(args.output_dir, "metrics_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果保存: {args.output_dir}")
    print(f"  - final_result.json")
    print(f"  - metrics_summary.json")


if __name__ == "__main__":
    main()
