
import torch
from torch.utils.data import DataLoader
from bert_get_data import HMBertClassifier, GenerateData 
import joblib # 用于加载编码器
from tqdm import tqdm
import os
import json
import numpy as np
from torch.amp import autocast # 导入 autocast
from sklearn.metrics import classification_report # 导入 scikit-learn


model_save_path = '/data/HMCTS/models/suiji' 
data_prefix = '/data/Data/suiji/data' # 数据和编码器的前缀
batch_size = 2048 
dataloader_num_workers = 16
dataloader_pin_memory = True 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_json_path = os.path.join(model_save_path, 'test_results.json')
output_report_path = os.path.join(model_save_path, 'test_classification_report.txt')

print("加载标签编码器...")
encoders = {}
num_classes = {}
label_names = {} # 存储标签名，用于分类报告
task_names = ['A1', 'A2_L1', 'A2_L2', 'A2_L3', 'A2_L4']
try:
    for task_name in task_names:
        encoder_path = f'{data_prefix}_{task_name}_encoder.pkl'
        encoders[task_name] = joblib.load(encoder_path)
        num_classes[task_name] = len(encoders[task_name].classes_)
        label_names[task_name] = encoders[task_name].classes_ 
        print(f"已加载 {task_name} 编码器，包含 {num_classes[task_name]} 个类别。")
except FileNotFoundError as e:
    print(f"加载编码器时出错: {e}。无法继续测试。")
    exit()
except Exception as e:
    print(f"加载编码器时发生错误: {e}")
    exit()


model_path = os.path.join(model_save_path, 'best_f1_macro.pt')

print(f"从 {model_path} 加载 Checkpoint...") # 打印正确的路径

try:
    model = HMBertClassifier(
        num_a1=num_classes['A1'],
        num_a2_l1=num_classes['A2_L1'],
        num_a2_l2=num_classes['A2_L2'],
        num_a2_l3=num_classes['A2_L3'],
        num_a2_l4=num_classes['A2_L4']
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
        consume_prefix_in_state_dict_if_present(state_dict, "_orig_mod.")
        
    model.load_state_dict(state_dict)
    
    best_metric = checkpoint.get('best_dev_f1_macro_a1', checkpoint.get('best_dev_acc_a1', 'N/A'))
    try:
        best_metric = f"{best_metric:.4f}"
    except:
        pass # 保持 'N/A'
    
    print(f"成功从第 {checkpoint.get('epoch', 'N/A')} 轮加载模型 (Best Metric: {best_metric})")

    model = model.to(device)
    
    try:
        model = torch.compile(model)
        print("模型已通过 torch.compile() 加速 (评估模式)！")
    except Exception as e:
        print(f"torch.compile() 失败 (可能是 PyTorch 版本过低): {e}")

    model.eval()

except FileNotFoundError:
    print(f"错误：在 {model_path} 中未找到模型文件。")
    exit()
except Exception as e:
    print(f"加载模型时出错: {e}")
    exit()

print("加载测试数据集...")
test_file_path = f'{data_prefix}_test_encoded.jsonl'
try:
    print(f"正在从 {test_file_path} 加载数据到 DataFrame...")
    test_dataset = GenerateData(mode='test', data_prefix=data_prefix)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory
    )
    print(f"已加载 {len(test_dataset)} 条样本。")
    print(f"测试集大小: {len(test_dataset)}")
except FileNotFoundError as e:
    print(f"加载测试数据集时出错: {e}。请确保 '{test_file_path}' 文件存在。")
    exit()
except Exception as e:
    print(f"加载测试数据集时发生错误: {e}")
    exit()

# 评估函数
def evaluate_on_test(model, test_loader, device, task_names):
    correct_preds_test = {task: 0 for task in task_names}
    valid_samples_test = {task: 0 for task in task_names} 
    correct_preds_test_top5_A1 = 0
    
    all_preds_for_report = {task: [] for task in task_names}
    all_labels_for_report = {task: [] for task in task_names}

    print("\n开始在测试集上评估...")
    with torch.no_grad(): 
        for batch in tqdm(test_loader, desc="测试中"):
            input_ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = {
                'A1': batch['A1_labels_idx'].to(device),
                'A2_L1': batch['A2_L1_labels_idx'].to(device),
                'A2_L2': batch['A2_L2_labels_idx'].to(device),
                'A2_L3': batch['A2_L3_labels_idx'].to(device),
                'A2_L4': batch['A2_L4_labels_idx'].to(device)
            }
            
            with autocast(device_type='cuda'):
                logits_list = model(input_ids, masks)
                logits = dict(zip(task_names, logits_list))

            for task in task_names:
                valid_mask = labels[task] != -1
                num_valid_in_batch = valid_mask.sum().item()
                
                if num_valid_in_batch > 0:
                    task_labels = labels[task][valid_mask]
                    task_logits = logits[task][valid_mask]
                    
                    task_preds = task_logits.argmax(dim=1)
                    
                    correct_preds_test[task] += (task_preds == task_labels).sum().item()
                    valid_samples_test[task] += num_valid_in_batch

                    all_preds_for_report[task].append(task_preds.cpu().numpy())
                    all_labels_for_report[task].append(task_labels.cpu().numpy())

                    if task == 'A1':
                        _, top5_indices = torch.topk(task_logits, 5, dim=1)
                        task_labels_expanded = task_labels.unsqueeze(1)
                        correct_top5 = (top5_indices == task_labels_expanded).sum().item()
                        correct_preds_test_top5_A1 += correct_top5

    for task in task_names:
        if all_labels_for_report[task]:
            all_preds_for_report[task] = np.concatenate(all_preds_for_report[task])
            all_labels_for_report[task] = np.concatenate(all_labels_for_report[task])
        else:
            all_preds_for_report[task] = np.array([], dtype=int)
            all_labels_for_report[task] = np.array([], dtype=int)

    return correct_preds_test, valid_samples_test, correct_preds_test_top5_A1, all_preds_for_report, all_labels_for_report

correct_counts, valid_counts, top5_a1_correct, preds_dict, labels_dict = evaluate_on_test(model, test_loader, device, task_names)

results = {
    "model_path": model_path,
    "data_prefix": data_prefix,
    "task_results": {}
}

print("\n--- 最终测试集结果 (Top-1 准确率) ---")
for task in task_names:
    acc = correct_counts[task] / valid_counts[task] if valid_counts[task] > 0 else 0
    results["task_results"][task] = {
        "top1_accuracy": acc,
        "correct_predictions": correct_counts[task],
        "valid_samples": valid_counts[task]
    }
    print(f"  {task}: {acc:.6f} ({correct_counts[task]}/{valid_counts[task]})")

top5_a1_acc = top5_a1_correct / valid_counts['A1'] if valid_counts['A1'] > 0 else 0
results["task_results"]["A1"]["top5_accuracy"] = top5_a1_acc
print(f"  A1 (Top-5): {top5_a1_acc:.6f} ({top5_a1_correct}/{valid_counts['A1']})")
print("-" * 40)

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print(f"简易评估结果已保存到: {output_json_path}")

print(f"正在生成详细分类报告 (F1, Precision, Recall)...")
try:
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("=== 最终测试集评估报告 ===\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {data_prefix}_test_encoded.jsonl\n")
        f.write("=" * 40 + "\n\n")

        for task in task_names:
            preds = preds_dict[task]
            labels = labels_dict[task]
            
            f.write(f"--- 任务: {task} ---\n")
            if len(labels) == 0:
                f.write("  (此任务在测试集中没有有效标签)\n\n")
                continue

            report_dict = classification_report(
                labels, 
                preds, 
                #labels=np.arange(len(label_names[task])), 
                #target_names=None, 
                output_dict=True, 
                zero_division=0
            )
            
            f.write(f"  Top-1 准确率:  {report_dict['accuracy']:.6f}\n")
            f.write(f"  Macro Avg (F1):    {report_dict['macro avg']['f1-score']:.6f}\n")
            f.write(f"  Weighted Avg (F1): {report_dict['weighted avg']['f1-score']:.6f}\n\n")
            
            f.write("  --- 详细平均值 ---\n")
            f.write(f"  Macro Avg:    {report_dict['macro avg']}\n")
            f.write(f"  Weighted Avg: {report_dict['weighted avg']}\n\n")

    print(f"详细分类报告已保存到: {output_report_path}")

except Exception as e:
    print(f"生成详细分类报告时出错: {e}")
    print("确保 scikit-learn 已安装: pip install scikit-learn")

print("\n评估完成！")