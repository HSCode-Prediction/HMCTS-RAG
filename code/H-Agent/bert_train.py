import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os
from torch.utils.data import DataLoader
from bert_get_data import HMBertClassifier, GenerateData 
import joblib # 用于加载编码器
from torch.cuda.amp import GradScaler 
from torch.amp import autocast
from transformers import get_linear_schedule_with_warmup # 导入 Scheduler
from sklearn.metrics import f1_score 

class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, alpha=0.25, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)

    def forward(self, logits, labels):
        ce = self.ce_loss(logits, labels)
        
        valid_mask = (labels != self.ignore_index)
        ce_valid = ce[valid_mask]
        
        if ce_valid.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        pt = torch.exp(-ce_valid)
        
        focal_term = (1.0 - pt) ** self.gamma
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_term = self.alpha
            else:
                alpha_term = 0.25 # 默认值
        else:
            alpha_term = 1.0

        focal_loss = alpha_term * focal_term * ce_valid

        return focal_loss.mean()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(save_name, model, optimizer, scheduler, scaler, epoch, best_acc, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model_to_save = model.module if hasattr(model, 'module') else model

    state_dict = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_dev_acc_a1': best_acc
    }
    torch.save(state_dict, os.path.join(save_path, save_name))

# --- 配置 ---
epoch = 5
batch_size = 1536 
lr = 5e-5       # 基础学习率
weight_decay = 1e-2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 20251010
save_path = '/data/HMCTS/models/changweibuqi' 
#data_prefix = '/data/Data/changweibuqi/data_expert_model' 
data_prefix = '/data/Data/test/data' 

# DataLoader 优化
dataloader_num_workers = 16 
dataloader_pin_memory = True 


max_grad_norm = 1.0       # 梯度裁剪阈值
warmup_ratio = 0.03       


focal_loss_gamma = 2.0
focal_loss_alpha = 0.25 # 设为 None 也可以

loss_weights = {
    'A1': 0.3,      # L10
    'A2_L1': 0.1,   # L2
    'A2_L2': 0.1,   # L4
    'A2_L3': 0.2,   # L6
    'A2_L4': 0.3    # L10
}


setup_seed(random_seed)

# 加载编码器以获取类别数量
print("加载标签编码器...")
encoders = {}
num_classes = {}
task_names = ['A1', 'A2_L1', 'A2_L2', 'A2_L3', 'A2_L4']
try:
    for task_name in task_names:
        encoder_path = f'{data_prefix}_{task_name}_encoder.pkl'
        encoders[task_name] = joblib.load(encoder_path)
        num_classes[task_name] = len(encoders[task_name].classes_)
        print(f"已加载 {task_name} 编码器，包含 {num_classes[task_name]} 个类别，来自 {encoder_path}")
except FileNotFoundError as e:
    print(f"加载编码器时出错: {e}。请确保编码器文件存在。")
    exit()
except Exception as e:
    print(f"加载编码器时发生错误: {e}")
    exit()

print("初始化模型...")
try:
    model = HMBertClassifier(
        num_a1=num_classes['A1'],
        num_a2_l1=num_classes['A2_L1'],
        num_a2_l2=num_classes['A2_L2'],
        num_a2_l3=num_classes['A2_L3'],
        num_a2_l4=num_classes['A2_L4']
    )
except Exception as e: 
    print(f"初始化 HMBertClassifier 时出错: {e}")
    exit()

try:
    model = torch.compile(model)
    print("模型已通过 torch.compile() 加速！")
except Exception as e:
    print(f"torch.compile() 失败 (可能是 PyTorch 版本过低或模型不支持): {e}")


# 定义损失函数和优化器
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# 将模型和损失函数移到 GPU
model = model.to(device)


print("加载数据集...")
try:
    train_dataset = GenerateData(mode='train', data_prefix=data_prefix)
    dev_dataset = GenerateData(mode='val', data_prefix=data_prefix)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        persistent_workers=True if dataloader_num_workers > 0 else False 
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        persistent_workers=True if dataloader_num_workers > 0 else False
    )

except FileNotFoundError as e:
    print(f"加载数据集时出错: {e}。请确保数据文件存在。")
    exit()
except Exception as e:
    print(f"加载数据集时发生错误: {e}")
    exit()

criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)

criterion_focal = FocalLoss(
    gamma=focal_loss_gamma, 
    alpha=focal_loss_alpha, 
    ignore_index=-1
).to(device)

print("!!! 已启用 Focal Loss (用于 A1 和 A2_L4) [模仿论文] !!!")
print("!!! 已启用 层级损失加权 (用于所有任务) [模仿论文] !!!")


# 初始化 AMP 梯度缩放器
scaler = GradScaler()

# 初始化 Scheduler
# 计算总训练步数
num_total_training_steps = len(train_loader) * epoch
num_warmup_steps = int(num_total_training_steps * warmup_ratio) 

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_total_training_steps 
)

# 打印训练信息
print("=" * 60)
print("HM-BERT 多任务训练开始 (适配 5 个任务)")
print(f"⚡️ 已为 A800 80G 优化 (AMP + 大 Batch + 高速 IO) ⚡️")
print(f"已启用健壮性训练 (Scheduler + Grad Clip + Checkpointing) ")
print("=" * 60)
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(dev_dataset)}")
# 打印类别数量
for task_name in task_names:
    print(f"类别数量 ({task_name}): {num_classes[task_name]}")
print(f"批次大小 (Batch Size): {batch_size}")
print(f"学习率 (Learning Rate): {lr} (带 {warmup_ratio*100}% 预热, 共 {num_warmup_steps} 步)") 
print(f"权重衰减 (Weight Decay): {weight_decay}")
print(f"梯度裁剪 (Max Grad Norm): {max_grad_norm}")
print(f"设备 (Device): {device}")
print(f"DataLoader Workers: {dataloader_num_workers}")
print(f"自动混合精度 (AMP): 启用")
print(f"损失策略: Focal Loss (G={focal_loss_gamma}, A={focal_loss_alpha}) + 层级权重")
print(f"损失权重: {loss_weights}")
print("=" * 60)

# 训练循环
best_dev_acc_a1 = 0 
best_dev_f1_macro_a1 = 0
start_epoch = 0 

for epoch_num in range(start_epoch, epoch):
    model.train() 
    total_loss_train = 0
    total_samples_train = 0
    correct_preds_train = {task: 0 for task in task_names}
    valid_samples_train = {task: 0 for task in task_names} 

    progress_bar = tqdm(train_loader, desc=f'轮次 {epoch_num+1}/{epoch}')

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        labels = {
            'A1': batch['A1_labels_idx'].to(device),
            'A2_L1': batch['A2_L1_labels_idx'].to(device),
            'A2_L2': batch['A2_L2_labels_idx'].to(device),
            'A2_L3': batch['A2_L3_labels_idx'].to(device),
            'A2_L4': batch['A2_L4_labels_idx'].to(device)
        }
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            logits_list = model(input_ids, masks)
            logits = dict(zip(task_names, logits_list)) 
            
            # 对不平衡的、关键的任务使用 Focal Loss
            loss_a1 = criterion_focal(logits['A1'], labels['A1'])
            loss_a2_l4 = criterion_focal(logits['A2_L4'], labels['A2_L4'])
            
            # 其他层级任务使用普通损失
            loss_a2_l1 = criterion(logits['A2_L1'], labels['A2_L1'])
            loss_a2_l2 = criterion(logits['A2_L2'], labels['A2_L2'])
            loss_a2_l3 = criterion(logits['A2_L3'], labels['A2_L3'])

            # 使用层级权重加和 (模仿论文 Eq.4)
            batch_loss = (loss_weights['A1'] * loss_a1 + 
                          loss_weights['A2_L1'] * loss_a2_l1 + 
                          loss_weights['A2_L2'] * loss_a2_l2 + 
                          loss_weights['A2_L3'] * loss_a2_l3 + 
                          loss_weights['A2_L4'] * loss_a2_l4)
            
        
        scaler.scale(batch_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() # 每个 step 更新一次学习率
        
        
        # 更新指标
        total_loss_train += batch_loss.item()
        current_batch_size = input_ids.size(0)
        total_samples_train += current_batch_size

        for task in task_names:
            valid_mask = labels[task] != -1
            num_valid_in_batch = valid_mask.sum().item()
            if num_valid_in_batch > 0:
                correct_preds_train[task] += (logits[task].argmax(dim=1)[valid_mask] == labels[task][valid_mask]).sum().item()
                valid_samples_train[task] += num_valid_in_batch 

        # 更新进度条显示
        avg_loss = total_loss_train / (batch_idx + 1)
        avg_acc = {}
        for task in task_names:
            avg_acc[task] = correct_preds_train[task] / valid_samples_train[task] if valid_samples_train[task] > 0 else 0

        # 在进度条中显示当前学习率
        current_lr = scheduler.get_last_lr()[0]
        
        
        progress_bar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'A1_Acc': f"{avg_acc['A1']:.3f}", # 区分 Acc 和 F1
            'L1_Acc': f"{avg_acc['A2_L1']:.3f}",
            'L2_Acc': f"{avg_acc['A2_L2']:.3f}",
            'L3_Acc': f"{avg_acc['A2_L3']:.3f}",
            'L4_Acc': f"{avg_acc['A2_L4']:.3f}",
            'LR': f"{current_lr:.1e}" # 显示学习率
        })
        
        

    # --- 验证 ---
    model.eval() 
    total_loss_val = 0
    correct_preds_val = {task: 0 for task in task_names}
    valid_samples_val = {task: 0 for task in task_names}
    correct_preds_val_top5_A1 = 0 
    
    all_labels_val = {task: [] for task in task_names}
    all_preds_val = {task: [] for task in task_names}

    print("\n正在验证...")
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="验证中"):
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
                
                loss_a1_val = criterion_focal(logits['A1'], labels['A1'])
                loss_a2_l4_val = criterion_focal(logits['A2_L4'], labels['A2_L4'])
                
                loss_a2_l1_val = criterion(logits['A2_L1'], labels['A2_L1'])
                loss_a2_l2_val = criterion(logits['A2_L2'], labels['A2_L2'])
                loss_a2_l3_val = criterion(logits['A2_L3'], labels['A2_L3'])
                
                batch_loss_val = (loss_weights['A1'] * loss_a1_val + 
                                  loss_weights['A2_L1'] * loss_a2_l1_val + 
                                  loss_weights['A2_L2'] * loss_a2_l2_val + 
                                  loss_weights['A2_L3'] * loss_a2_l3_val + 
                                  loss_weights['A2_L4'] * loss_a2_l4_val)
                
            
            total_loss_val += batch_loss_val.item()

            # 精确计算验证准确率
            for task in task_names:
                valid_mask = labels[task] != -1
                num_valid_in_batch = valid_mask.sum().item()
                if num_valid_in_batch > 0:
                    task_labels = labels[task][valid_mask]
                    task_logits = logits[task][valid_mask]
                    task_preds = task_logits.argmax(dim=1)
                    
                    # Top-1 准确率
                    correct_preds_val[task] += (task_preds == task_labels).sum().item()
                    valid_samples_val[task] += num_valid_in_batch
                    
                    all_labels_val[task].extend(task_labels.cpu().numpy())
                    all_preds_val[task].extend(task_preds.cpu().numpy())
                    
                    if task == 'A1':
                        # Top-5 准确率
                        _, top5_indices = torch.topk(task_logits, 5, dim=1)
                        task_labels_expanded = task_labels.unsqueeze(1)
                        correct_top5 = (top5_indices == task_labels_expanded).sum().item()
                        correct_preds_val_top5_A1 += correct_top5

    
    # 计算该轮次的平均准确率和损失
    print(f"\n--- 轮次 {epoch_num + 1}/{epoch} 总结 ---")
    
    print("[训练集]")
    print(f"  平均损失: {total_loss_train / len(train_loader):.4f}")
    for task in task_names:
        acc = correct_preds_train[task] / valid_samples_train[task] if valid_samples_train[task] > 0 else 0
        print(f"  Top-1 准确率 ({task}): {acc:.4f} ({correct_preds_train[task]}/{valid_samples_train[task]})")

    print("[验证集]")
    print(f"  平均损失: {total_loss_val / len(dev_loader):.4f}")
    val_accuracies = {}
    val_f1_macro = {}
    val_f1_weighted = {}

    for task in task_names:
        # Top-1 准确率
        acc = correct_preds_val[task] / valid_samples_val[task] if valid_samples_val[task] > 0 else 0
        val_accuracies[task] = acc
        print(f"  Top-1 准确率 ({task}): {acc:.4f} ({correct_preds_val[task]}/{valid_samples_val[task]})")
        
        if valid_samples_val[task] > 0:
            f1_macro = f1_score(all_labels_val[task], all_preds_val[task], average='macro', zero_division=0)
            val_f1_macro[task] = f1_macro
            f1_weighted = f1_score(all_labels_val[task], all_preds_val[task], average='weighted', zero_division=0)
            val_f1_weighted[task] = f1_weighted
            
            if task == 'A1' or task == 'A2_L1':
                print(f"  F1-Macro ({task}):   {f1_macro:.4f}  <-- (评估稀有类别的关键指标)")
                print(f"  F1-Weighted ({task}): {f1_weighted:.4f}  <-- (论文  指标)")

    acc_top5_a1 = correct_preds_val_top5_A1 / valid_samples_val['A1'] if valid_samples_val['A1'] > 0 else 0
    print(f"  Top-5 准确率 (A1): {acc_top5_a1:.4f} ({correct_preds_val_top5_A1}/{valid_samples_val['A1']})")
    print(f"  (注意: A2_L4 Top-1 应与 A1 Top-1 相同)")

    current_dev_acc_a1 = val_accuracies['A1']
    current_dev_f1_macro_a1 = val_f1_macro.get('A1', 0) # 获取 A1 的 Macro F1

    if current_dev_acc_a1 > best_dev_acc_a1:
        best_dev_acc_a1 = current_dev_acc_a1
        print(f"★ 新的最佳模型 (按Acc)！验证 Top-1 (A1): {best_dev_acc_a1:.4f}")
        save_checkpoint(
            save_name='best_acc.pt',
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch_num,
            best_acc=best_dev_f1_macro_a1,
            save_path=save_path
        )
    
    if current_dev_f1_macro_a1 > best_dev_f1_macro_a1:
        best_dev_f1_macro_a1 = current_dev_f1_macro_a1
        print(f"★ 新的最佳模型 (按F1-Macro)！验证 F1-Macro (A1): {best_dev_f1_macro_a1:.4f}")
        save_checkpoint(
            save_name='best_f1_macro.pt', # 保存为不同的文件名
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch_num,
            best_acc=best_dev_acc_a1,
            save_path=save_path
        )
    print("-" * 60)

# 保存最后一个轮次的模型
print(f"\n 训练完成！正在保存最终 Checkpoint...")
save_checkpoint(
    save_name='last.pt',
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    epoch=epoch - 1, # epoch 索引从 0 开始, 所以最后一轮是 epoch-1
    best_acc=best_dev_acc_a1,
    save_path=save_path
)
print(f"最佳验证 Top-1 准确率 (A1): {best_dev_acc_a1:.4f}")
print(f"最佳验证 F1-Macro (A1): {best_dev_f1_macro_a1:.4f}")
print(f"模型保存在: {save_path}")
print("=" * 60)