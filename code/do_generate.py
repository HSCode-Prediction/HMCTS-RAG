
import os
import sys
import json
import time
import argparse
from tqdm import tqdm

# 设置环境
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bert_agent import BertAgent
from rag_retriever import RAGRetriever
from mcts_system import Generator, search_for_answers


def parse_args():
    parser = argparse.ArgumentParser(description='MCTS-RAG HSCode Classification')
    
    # 数据路径
    parser.add_argument('--dataset_name', type=str, default='HSCode')
    parser.add_argument('--test_json_filename', type=str, default='MCTS_hard')
    parser.add_argument('--data_root', type=str, 
                        default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    parser.add_argument('--prompts_root', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'prompts'))
    
    # 模型路径
    parser.add_argument('--model_ckpt', type=str, required=True,
                        help='LLM模型路径')
    parser.add_argument('--bert_model_name', type=str, required=True,
                        help='BERT模型路径')
    parser.add_argument('--bert_data_prefix', type=str, required=True,
                        help='BERT编码器前缀')
    parser.add_argument('--bert_classifier_ckpt', type=str, required=True,
                        help='BERT分类器checkpoint')
    parser.add_argument('--embedding_model', type=str, default='',
                        help='Embedding模型路径（可选）')
    
    # RAG配置
    parser.add_argument('--rag_db', type=str, 
                        default=os.path.join(os.path.dirname(__file__), '..', 'RAG_db'))
    parser.add_argument('--disable_rag', action='store_true')
    
    # MCTS配置
    parser.add_argument('--num_rollouts', type=int, default=10)
    parser.add_argument('--max_depth_allowed', type=int, default=5)
    parser.add_argument('--mcts_exploration_weight', type=float, default=1.0)
    parser.add_argument('--num_votes', type=int, default=5)
    
    # 运行配置
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--api', type=str, default='huggingface',
                        choices=['huggingface', 'vllm'])
    parser.add_argument('--half_precision', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--note', type=str, default='mcts_run')
    
    # LLM生成配置
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.9)
    
    # 输出配置
    parser.add_argument('--run_outputs_dir', type=str, default='')
    parser.add_argument('--answer_sheets_dir', type=str, default='')
    
    return parser.parse_args()


def setup_output_dirs(args):
    """设置输出目录"""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    base_dir = os.path.join(
        os.path.dirname(__file__), '..', 'outputs',
        args.dataset_name, args.note, timestamp
    )
    
    args.run_outputs_dir = base_dir
    args.answer_sheets_dir = os.path.join(base_dir, 'answer_sheets')
    
    os.makedirs(args.run_outputs_dir, exist_ok=True)
    os.makedirs(args.answer_sheets_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(base_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    return args


def load_test_data(args):
    """加载测试数据"""
    test_file = os.path.join(
        args.data_root, args.dataset_name, 
        args.test_json_filename + '.json'
    )
    
    if not os.path.exists(test_file):
        test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename)
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试文件不存在: {test_file}")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def load_llm(args):
    """加载LLM模型"""
    print(f"\n[LLM] 加载模型: {args.model_ckpt}")
    
    if args.api == 'huggingface':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_ckpt, 
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_ckpt,
            torch_dtype=torch.float16 if args.half_precision else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        print(f"[LLM] ✅ 模型加载完成")
        return tokenizer, model
    
    elif args.api == 'vllm':
        from vllm import LLM, SamplingParams
        
        model = LLM(model=args.model_ckpt, trust_remote_code=True)
        tokenizer = model.get_tokenizer()
        
        print(f"vLLM模型加载完成")
        return tokenizer, model
    
    return None, None


class DummyEvaluator:
    def extract_answer_from_gold_solution(self, solution):
        return solution


def main():
    args = parse_args()
    args = setup_output_dirs(args)
    
    print("=" * 60)
    print("MCTS-RAG HSCode Classification System")
    print("=" * 60)
    print(f"Config: {json.dumps(vars(args), indent=2, ensure_ascii=False)[:500]}...")
    print("=" * 60)
    
    # 加载测试数据
    print("\n[1/5] 加载测试数据...")
    test_data = load_test_data(args)
    print(f"       总样本数: {len(test_data)}")
    
    # 加载BERT智能体
    print("\n[2/5] 加载BERT智能体...")
    bert_agent = BertAgent(
        bert_model_name=args.bert_model_name,
        data_prefix=args.bert_data_prefix,
        classifier_ckpt=args.bert_classifier_ckpt
    )
    
    # 加载RAG检索器
    print("\n[3/5] 加载RAG检索器...")
    retriever = RAGRetriever(
        db_path=args.rag_db,
        embedding_model_path=args.embedding_model if args.embedding_model else None,
        use_reranker=True
    )
    
    # 加载LLM
    print("\n[4/5] 加载LLM模型...")
    tokenizer, model = load_llm(args)
    
    # 创建Generator
    print("\n[5/5] 初始化Generator...")
    generator = Generator(
        args=args,
        tokenizer=tokenizer,
        model=model,
        bert_agent=bert_agent,
        retriever=retriever
    )
    
    print("\n" + "=" * 60)
    print("开始MCTS推理")
    print("=" * 60)
    
    # 确定处理范围
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(test_data)
    
    results = []
    correct = 0
    total = 0
    start_time = time.time()
    
    for i in tqdm(range(start_idx, min(end_idx, len(test_data))), desc="Processing"):
        item = test_data[i]
        problem = item.get('problem', '')
        gt_code = item.get('solution', '')
        
        item_start = time.time()
        
        try:
            # 执行MCTS搜索
            solutions = search_for_answers(
                args=args,
                user_question=problem,
                question_id=i,
                gt_answer=gt_code,
                generator=generator
            )
            
            # 记录结果
            pred_code = solutions[0]['hs'] if solutions else ""
            is_correct = (pred_code == gt_code)
            
            if is_correct:
                correct += 1
            total += 1
            
            result = {
                "id": i,
                "gt": gt_code,
                "pred": pred_code,
                "correct": is_correct,
                "time": round(time.time() - item_start, 2),
                "num_solutions": len(solutions),
                "top_reward": solutions[0]['reward'] if solutions else 0
            }
            results.append(result)
            
            if args.verbose or i % 10 == 0:
                acc = correct / total if total > 0 else 0
                print(f"\n[Q{i}] GT: {gt_code} | Pred: {pred_code} | "
                      f"{'✓' if is_correct else '✗'} | Acc: {acc:.2%}")
            
        except Exception as e:
            print(f"\n[Q{i}] Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({"id": i, "error": str(e)})
            total += 1
    
    # 统计
    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"总样本: {total}")
    print(f"正确: {correct}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"平均每样本: {elapsed/max(total,1):.2f}s")
    
    stats = generator.get_stats()
    print(f"\n调用统计:")
    print(f"  BERT: {stats.get('bert_calls', 0)}")
    print(f"  LLM: {stats.get('llm_calls', 0)}")
    print(f"  RAG: {stats.get('rag_calls', 0)}")
    
    # 保存最终结果
    final_result = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed": elapsed,
        "stats": stats,
        "results": results
    }
    
    with open(os.path.join(args.run_outputs_dir, "final_result.json"), 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(args.run_outputs_dir, "final_result.txt"), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n")
        f.write(f"Total Time: {elapsed:.1f}s\n")
        f.write(f"Avg Time: {elapsed/max(total,1):.2f}s\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    
    print(f"\n结果已保存到: {args.run_outputs_dir}")


if __name__ == "__main__":
    main()
