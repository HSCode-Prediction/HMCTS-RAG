import os
import json
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 文件路径
FULL_TEST_FILE = os.path.join(BASE_DIR, 'data', 'LLM_suiji_Test.json')
HARD_TEST_FILE = os.path.join(BASE_DIR, 'data', 'MCTS_hard.json')
MCTS_OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'v2_20251223_095528')
OUTPUT_FILE = os.path.join(BASE_DIR, 'outputs', 'final_all_metrics.json')


def compute_metrics(y_true, y_pred):
    """计算多分类指标"""
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
        'Acc': round(acc, 4),
        'MP': round(mp, 4),
        'MR': round(mr, 4),
        'MF1': round(mf1, 4),
        'WP': round(wp, 4),
        'WR': round(wr, 4),
        'WF1': round(wf1, 4),
        'valid_samples': len(valid),
        'total_classes': len(all_labels)
    }


def main():
    print("=" * 60)
    print("计算最终指标（合并BERT + MCTS结果）")
    print("=" * 60)
    
    # 1. 加载完整测试数据
    print("\n[1] 加载完整测试数据...")
    with open(FULL_TEST_FILE, 'r', encoding='utf-8') as f:
        full_test = json.load(f)
    print(f"    总样本数: {len(full_test)}")
    
    # 建立ID到样本的映射
    full_test_map = {item['id']: item for item in full_test}
    
    # 2. 加载困难样本
    print("\n[2] 加载困难样本...")
    with open(HARD_TEST_FILE, 'r', encoding='utf-8') as f:
        hard_test = json.load(f)
    print(f"    困难样本数: {len(hard_test)}")
    
    # 困难样本ID集合
    hard_ids = set()
    for i, item in enumerate(hard_test):
        if 'id' in item:
            hard_ids.add(item['id'])
        else:
            # 如果没有id，用索引标记
            hard_ids.add(f"hard_{i}")
    
    # 3. 加载MCTS预测结果
    print("\n[3] 加载MCTS预测结果...")
    solutions_dir = os.path.join(MCTS_OUTPUT_DIR, 'solutions')
    
    mcts_results = {}  # idx -> {pred, candidates}
    candidate_hit_count = 0  # 候选答案命中数
    total_candidates = 0
    
    for i in range(len(hard_test)):
        sol_path = os.path.join(solutions_dir, f"Q{i:04d}.json")
        if os.path.exists(sol_path):
            with open(sol_path, 'r', encoding='utf-8') as f:
                solutions = json.load(f)
            
            gt_code = hard_test[i].get('solution', '')
            pred = solutions[0]['hs'] if solutions else ""
            
            # 所有候选答案
            candidates = [s['hs'] for s in solutions if s.get('hs')]
            
            mcts_results[i] = {
                'pred': pred,
                'candidates': candidates,
                'gt': gt_code
            }
            
            # 检查候选答案中是否包含正确答案
            if gt_code in candidates:
                candidate_hit_count += 1
            total_candidates += 1
    
    print(f"    MCTS结果数: {len(mcts_results)}")
    
    # 4. 合并结果
    print("\n[4] 合并BERT全对样本和MCTS困难样本...")
    
    # 简单样本数 = 总样本 - 困难样本
    easy_count = len(full_test) - len(hard_test)
    print(f"    简单样本（BERT全对）: {easy_count}")
    print(f"    困难样本（MCTS处理）: {len(hard_test)}")
    
    # 构建完整预测结果
    all_results = []
    
    # 简单样本：pred = gt（全对）
    easy_samples = []
    for item in full_test:
        item_id = item['id']
        gt = item['solution']
        
        # 检查是否是困难样本
        is_hard = False
        for i, h in enumerate(hard_test):
            if h.get('id') == item_id or h.get('solution') == gt and h.get('problem') == item.get('problem'):
                is_hard = True
                break
        
        if not is_hard:
            easy_samples.append({'gt': gt, 'pred': gt, 'correct': True})
    
    print(f"    实际简单样本: {len(easy_samples)}")
    
    # 困难样本：使用MCTS预测
    hard_results = []
    for i, item in enumerate(hard_test):
        gt = item.get('solution', '')
        if i in mcts_results:
            pred = mcts_results[i]['pred']
        else:
            pred = ''
        hard_results.append({
            'gt': gt,
            'pred': pred,
            'correct': pred == gt
        })
    
    # 统计困难样本准确率
    hard_correct = sum(1 for r in hard_results if r['correct'])
    print(f"    困难样本正确: {hard_correct}/{len(hard_results)} ({hard_correct/len(hard_results)*100:.2f}%)")
    
    # 合并
    all_results = easy_samples + hard_results
    print(f"    总样本: {len(all_results)}")
    
    # 5. 计算整体指标
    print("\n[5] 计算整体指标...")
    
    y_true_all = [r['gt'] for r in all_results]
    y_pred_all = [r['pred'] for r in all_results]
    
    # 10位指标
    metrics_10 = compute_metrics(y_true_all, y_pred_all)
    
    # 各层级指标
    hier_metrics = {}
    for level, plen in [('L2', 2), ('L4', 4), ('L6', 6), ('L8', 8), ('L10', 10)]:
        yt = [r['gt'][:plen] for r in all_results if len(r['gt']) >= plen and len(r['pred']) >= plen]
        yp = [r['pred'][:plen] for r in all_results if len(r['gt']) >= plen and len(r['pred']) >= plen]
        if yt:
            hier_metrics[level] = compute_metrics(yt, yp)
    
    # 6. 计算候选答案命中率（仅困难样本）
    print("\n[6] 计算候选答案命中率...")
    candidate_hit_rate = candidate_hit_count / total_candidates if total_candidates > 0 else 0
    print(f"    候选答案命中: {candidate_hit_count}/{total_candidates} ({candidate_hit_rate*100:.2f}%)")
    
    max_correct_hard = candidate_hit_count
    max_correct_total = len(easy_samples) + max_correct_hard
    max_acc = max_correct_total / len(all_results)
    
    print(f"    潜在最大ACC（假设验证器完美）: {max_acc*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    
    print(f"\n总体统计:")
    print(f"  总样本: {len(all_results)}")
    print(f"  简单样本（BERT全对）: {len(easy_samples)}")
    print(f"  困难样本（MCTS处理）: {len(hard_results)}")
    print(f"  困难样本正确: {hard_correct} ({hard_correct/len(hard_results)*100:.2f}%)")
    
    print(f"\n10位HSCode整体指标:")
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
    
    print(f"\n候选答案统计（仅困难样本）:")
    print(f"  候选命中率: {candidate_hit_rate*100:.2f}%")
    print(f"  Top-1 ACC: {hard_correct/len(hard_results)*100:.2f}%")
    print(f"  潜在最大ACC（候选中有正确答案）: {candidate_hit_rate*100:.2f}%")
    
    print(f"\n整体潜在最大ACC（假设验证器完美）:")
    print(f"  {max_acc*100:.2f}% ({max_correct_total}/{len(all_results)})")
    
    # 8. 保存结果
    final_output = {
        'summary': {
            'total_samples': len(all_results),
            'easy_samples_bert_correct': len(easy_samples),
            'hard_samples_mcts': len(hard_results),
            'hard_samples_correct': hard_correct,
            'hard_samples_acc': round(hard_correct/len(hard_results), 4)
        },
        'overall_metrics_L10': metrics_10,
        'hierarchical_metrics': hier_metrics,
        'candidate_analysis': {
            'candidate_hit_count': candidate_hit_count,
            'candidate_total': total_candidates,
            'candidate_hit_rate': round(candidate_hit_rate, 4),
            'top1_acc_hard': round(hard_correct/len(hard_results), 4),
            'potential_max_acc_hard': round(candidate_hit_rate, 4),
            'potential_max_acc_overall': round(max_acc, 4)
        },
        'hard_only_metrics': {
            'L10': compute_metrics([r['gt'] for r in hard_results], [r['pred'] for r in hard_results])
        }
    }
    
    hard_hier = {}
    for level, plen in [('L2', 2), ('L4', 4), ('L6', 6), ('L8', 8), ('L10', 10)]:
        yt = [r['gt'][:plen] for r in hard_results if len(r['gt']) >= plen and len(r['pred']) >= plen]
        yp = [r['pred'][:plen] for r in hard_results if len(r['gt']) >= plen and len(r['pred']) >= plen]
        if yt:
            hard_hier[level] = compute_metrics(yt, yp)
    final_output['hard_only_metrics']['hierarchical'] = hard_hier
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
