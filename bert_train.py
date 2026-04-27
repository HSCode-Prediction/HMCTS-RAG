

# import torch
# from torch import nn
# from torch.optim import AdamW
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# import random
# import os
# from torch.utils.data import DataLoader
# from bert_get_data import HMBertClassifier, GenerateData # 导入修改后的类
# import joblib # 用于加载编码器
# from torch.cuda.amp import GradScaler 
# from torch.amp import autocast
# from transformers import get_linear_schedule_with_warmup # 导入 Scheduler
# from sklearn.utils.class_weight import compute_class_weight # 权重计算

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# def save_checkpoint(save_name, model, optimizer, scheduler, scaler, epoch, best_acc, save_path):
#     """保存完整的训练状态以便断点续训"""
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
    
#     # 包装为 torch.compile 兼容模式 
#     model_to_save = model.module if hasattr(model, 'module') else model

#     state_dict = {
#         'epoch': epoch,
#         'model_state_dict': model_to_save.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#         'scaler_state_dict': scaler.state_dict(),
#         'best_dev_acc_a1': best_acc
#     }
#     torch.save(state_dict, os.path.join(save_path, save_name))

# # --- 配置 ---
# epoch = 5
# batch_size = 1536 
# lr = 5e-5      # 基础学习率
# weight_decay = 1e-3
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# random_seed = 20251010
# save_path = '/root/HMCTS/models/HBert1' # 训练好的模型保存路径
# data_prefix = '/root/agent/bert_mtl_all_data/bert_mtl_all_data' 

# # DataLoader 优化
# dataloader_num_workers = 16 
# dataloader_pin_memory = True 


# max_grad_norm = 1.0      # 梯度裁剪阈值
# warmup_ratio = 0.03      


# setup_seed(random_seed)

# # 加载编码器以获取类别数量
# print("加载标签编码器...")
# encoders = {}
# num_classes = {}
# task_names = ['A1', 'A2_L1', 'A2_L2', 'A2_L3', 'A2_L4']
# try:
#     for task_name in task_names:
#         encoder_path = f'{data_prefix}_{task_name}_encoder.pkl'
#         encoders[task_name] = joblib.load(encoder_path)
#         num_classes[task_name] = len(encoders[task_name].classes_)
#         print(f"已加载 {task_name} 编码器，包含 {num_classes[task_name]} 个类别，来自 {encoder_path}")
# except FileNotFoundError as e:
#     print(f"加载编码器时出错: {e}。请确保编码器文件存在。")
#     exit()
# except Exception as e:
#     print(f"加载编码器时发生错误: {e}")
#     exit()

# # 使用每个头的正确类别数量定义模型
# print("初始化模型...")
# try:
#     model = HMBertClassifier(
#         num_a1=num_classes['A1'],
#         num_a2_l1=num_classes['A2_L1'],
#         num_a2_l2=num_classes['A2_L2'],
#         num_a2_l3=num_classes['A2_L3'],
#         num_a2_l4=num_classes['A2_L4']
#     )
# except Exception as e: 
#     print(f"初始化 HMBertClassifier 时出错: {e}")
#     exit()

# # PyTorch 2.0+ JIT 编译
# try:
#     model = torch.compile(model)
#     print("模型已通过 torch.compile() 加速！")
# except Exception as e:
#     print(f"torch.compile() 失败 (可能是 PyTorch 版本过低或模型不支持): {e}")


# # 定义损失函数和优化器
# # (注意: 这里的 criterion 将被下面的代码修改)
# optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# # 将模型和损失函数移到 GPU
# model = model.to(device)


# # --- (2) MODIFICATION START: 加载数据 & 计算权重 ---
# # 使用修改后的 GenerateData 函数构建数据集
# print("加载数据集...")
# try:
#     train_dataset = GenerateData(mode='train', data_prefix=data_prefix)
#     dev_dataset = GenerateData(mode='val', data_prefix=data_prefix)
    
#     # --- 新增：计算类别权重 ---
#     # 假设 train_dataset 有一个 .df 属性
#     try:
#         train_df = train_dataset.df
#         if train_df is None:
#             raise AttributeError
#     except AttributeError:
#         print("错误: train_dataset 没有 .df 属性。")
#         print("请确保您的 GenerateData 类在加载数据后，将 DataFrame 存储在 self.df 中。")
#         exit()

#     # 获取 A1/A2_L4 的编码器 (假设 A1 和 A2_L4 相同)
#     encoder_l4 = encoders['A2_L4']
    
#     # 过滤出在编码器中注册过的标签
#     valid_labels_mask = train_df['label_L4_Final'].isin(encoder_l4.classes_)
#     valid_train_labels = train_df[valid_labels_mask]['label_L4_Final']

#     valid_labels_mask = train_df['label_L4_Final'].isin(encoder_l4.classes_)
#     valid_train_labels = train_df[valid_labels_mask]['label_L4_Final']

#     print("正在计算类别权重 (用于 A1 和 A2_L4)...")


#     y_for_weights_calc = pd.concat([
#         valid_train_labels, 
#         pd.Series(encoder_l4.classes_)
#     ])
    
#     # 4. 计算权重 (使用修改后的 y)
#     class_weights_l4 = compute_class_weight(
#         class_weight='balanced',
#         classes=encoder_l4.classes_, # 必须传入编码器的所有类
#         y=y_for_weights_calc          # <--- 使用修正后的 y
#     )

#     # 转换为 PyTorch Tensor 并移到 GPU
#     class_weights_l4_tensor = torch.tensor(class_weights_l4, dtype=torch.float).to(device)
#     print(f"已为 A1/A2_L4 (共 {len(encoder_l4.classes_)} 类) 计算了类别权重。")
#     # --- 新增结束 ---

    
#     # 优化 DataLoader
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True,
#         num_workers=dataloader_num_workers,
#         pin_memory=dataloader_pin_memory,
#         persistent_workers=True if dataloader_num_workers > 0 else False 
#     )
#     dev_loader = DataLoader(
#         dev_dataset, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         num_workers=dataloader_num_workers,
#         pin_memory=dataloader_pin_memory,
#         persistent_workers=True if dataloader_num_workers > 0 else False
#     )

# except FileNotFoundError as e:
#     print(f"加载数据集时出错: {e}。请确保数据文件存在。")
#     exit()
# except Exception as e:
#      print(f"加载数据集时发生错误: {e}")
#      exit()
# # --- (2) MODIFICATION END ---


# # --- (3) MODIFICATION START: 修改损失函数定义 ---


# # 2. 定义两个损失函数：一个不加权，一个加权
# # 不加权的 (用于 L1, L2, L3)
# criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)

# # 加权的 (用于 A1, A2_L4), 使用我们刚刚计算的权重
# criterion_weighted = nn.CrossEntropyLoss(
#     weight=class_weights_l4_tensor, 
#     ignore_index=-1
# ).to(device)

# print("!!! 已启用类别加权 (Class Weighting) 损失 (用于 A1 和 A2_L4) !!!")

# # --- (3) MODIFICATION END ---


# # 初始化 AMP 梯度缩放器
# scaler = GradScaler()

# # 初始化 Scheduler
# # 计算总训练步数
# num_total_training_steps = len(train_loader) * epoch
# num_warmup_steps = int(num_total_training_steps * warmup_ratio) 

# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=num_warmup_steps,
#     num_training_steps=num_total_training_steps 
# )

# # 打印训练信息
# print("=" * 60)
# print("HM-BERT 多任务训练开始 (适配 5 个任务)")
# print(f"⚡️ 已为 A800 80G 优化 (AMP + 大 Batch + 高速 IO) ⚡️")
# print(f"已启用健壮性训练 (Scheduler + Grad Clip + Checkpointing) ")
# print("=" * 60)
# print(f"训练集大小: {len(train_dataset)}")
# print(f"验证集大小: {len(dev_dataset)}")
# # ... (类别数量打印) ...
# print(f"批次大小 (Batch Size): {batch_size}")
# # <--- 调整 1: 这里的打印输出会动态反映你的 1% 设置 --->
# print(f"学习率 (Learning Rate): {lr} (带 {warmup_ratio*100}% 预热, 共 {num_warmup_steps} 步)") 
# print(f"权重衰减 (Weight Decay): {weight_decay}")
# print(f"梯度裁剪 (Max Grad Norm): {max_grad_norm}")
# print(f"设备 (Device): {device}")
# print(f"DataLoader Workers: {dataloader_num_workers}")
# print(f"自动混合精度 (AMP): 启用")
# print("=" * 60)

# # 训练循环
# best_dev_acc_a1 = 0 
# start_epoch = 0 

# for epoch_num in range(start_epoch, epoch):
#     model.train() 
#     total_loss_train = 0
#     total_samples_train = 0
#     correct_preds_train = {task: 0 for task in task_names}
#     valid_samples_train = {task: 0 for task in task_names} 

#     progress_bar = tqdm(train_loader, desc=f'轮次 {epoch_num+1}/{epoch}')

#     for batch_idx, batch in enumerate(progress_bar):
#         input_ids = batch['input_ids'].to(device)
#         masks = batch['attention_mask'].to(device)
#         labels = {
#             'A1': batch['A1_labels_idx'].to(device),
#             'A2_L1': batch['A2_L1_labels_idx'].to(device),
#             'A2_L2': batch['A2_L2_labels_idx'].to(device),
#             'A2_L3': batch['A2_L3_labels_idx'].to(device),
#             'A2_L4': batch['A2_L4_labels_idx'].to(device)
#         }
        
#         optimizer.zero_grad()
        
#         with autocast(device_type='cuda'):
#             logits_list = model(input_ids, masks)
#             logits = dict(zip(task_names, logits_list)) 

#             # --- (4) MODIFICATION START: 修改训练损失计算 ---
            
#             # 对不平衡的、关键的任务使用加权损失
#             loss_a1 = criterion_weighted(logits['A1'], labels['A1'])
#             loss_a2_l4 = criterion_weighted(logits['A2_L4'], labels['A2_L4'])
            
#             # 其他层级任务使用普通损失
#             loss_a2_l1 = criterion(logits['A2_L1'], labels['A2_L1'])
#             loss_a2_l2 = criterion(logits['A2_L2'], labels['A2_L2'])
#             loss_a2_l3 = criterion(logits['A2_L3'], labels['A2_L3'])

#             # 简单相加，不再使用 loss_weights
#             batch_loss = (loss_a1 + 
#                           loss_a2_l1 + 
#                           loss_a2_l2 + 
#                           loss_a2_l3 + 
#                           loss_a2_l4)
            
#             # --- (4) MODIFICATION END ---
        
#         # 健壮的反向传播步骤
#         scaler.scale(batch_loss).backward()
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#         scaler.step(optimizer)
#         scaler.update()
#         scheduler.step() # 每个 step 更新一次学习率
        
        
#         # 更新指标
#         total_loss_train += batch_loss.item()
#         current_batch_size = input_ids.size(0)
#         total_samples_train += current_batch_size

#         for task in task_names:
#             valid_mask = labels[task] != -1
#             num_valid_in_batch = valid_mask.sum().item()
#             if num_valid_in_batch > 0:
#                   correct_preds_train[task] += (logits[task].argmax(dim=1)[valid_mask] == labels[task][valid_mask]).sum().item()
#                   valid_samples_train[task] += num_valid_in_batch 

#         # 更新进度条显示
#         avg_loss = total_loss_train / (batch_idx + 1)
#         avg_acc = {}
#         for task in task_names:
#             avg_acc[task] = correct_preds_train[task] / valid_samples_train[task] if valid_samples_train[task] > 0 else 0

#         # 在进度条中显示当前学习率
#         current_lr = scheduler.get_last_lr()[0]
        
#         # ========================================================
        
#         progress_bar.set_postfix({
#             'Loss': f'{avg_loss:.4f}',
#             'A1': f"{avg_acc['A1']:.3f}",
#             'L1': f"{avg_acc['A2_L1']:.3f}",
#             'L2': f"{avg_acc['A2_L2']:.3f}",
#             'L3': f"{avg_acc['A2_L3']:.3f}",
#             'L4': f"{avg_acc['A2_L4']:.3f}",
#             'LR': f"{current_lr:.1e}" # 显示学习率
#         })
        
        

#     # --- 验证 ---
#     model.eval() 
#     total_loss_val = 0
#     correct_preds_val = {task: 0 for task in task_names}
#     valid_samples_val = {task: 0 for task in task_names}
#     correct_preds_val_top5_A1 = 0 

#     print("\n正在验证...")
#     with torch.no_grad():
#         for batch in tqdm(dev_loader, desc="验证中"):
#             input_ids = batch['input_ids'].to(device)
#             masks = batch['attention_mask'].to(device)
#             labels = {
#                 'A1': batch['A1_labels_idx'].to(device),
#                 'A2_L1': batch['A2_L1_labels_idx'].to(device),
#                 'A2_L2': batch['A2_L2_labels_idx'].to(device),
#                 'A2_L3': batch['A2_L3_labels_idx'].to(device),
#                 'A2_L4': batch['A2_L4_labels_idx'].to(device)
#             }

#             with autocast(device_type='cuda'):
#                 logits_list = model(input_ids, masks)
#                 logits = dict(zip(task_names, logits_list))

#                 # --- (5) MODIFICATION START: 修改验证损失计算 ---
                
#                 loss_a1_val = criterion_weighted(logits['A1'], labels['A1'])
#                 loss_a2_l4_val = criterion_weighted(logits['A2_L4'], labels['A2_L4'])
                
#                 loss_a2_l1_val = criterion(logits['A2_L1'], labels['A2_L1'])
#                 loss_a2_l2_val = criterion(logits['A2_L2'], labels['A2_L2'])
#                 loss_a2_l3_val = criterion(logits['A2_L3'], labels['A2_L3'])
                
#                 # 简单相加，不再使用 loss_weights
#                 batch_loss_val = (loss_a1_val + 
#                                   loss_a2_l1_val + 
#                                   loss_a2_l2_val + 
#                                   loss_a2_l3_val + 
#                                   loss_a2_l4_val)
                
#                 # --- (5) MODIFICATION END ---
            
#             total_loss_val += batch_loss_val.item()

#             # 精确计算验证准确率 (这部分逻辑不变)
#             for task in task_names:
#                 valid_mask = labels[task] != -1
#                 num_valid_in_batch = valid_mask.sum().item()
#                 if num_valid_in_batch > 0:
#                     task_labels = labels[task][valid_mask]
#                     task_logits = logits[task][valid_mask]
                    
#                     # Top-1
#                     correct_preds_val[task] += (task_logits.argmax(dim=1) == task_labels).sum().item()
#                     valid_samples_val[task] += num_valid_in_batch
                    
#                     if task == 'A1':
#                         # Top-5
#                         _, top5_indices = torch.topk(task_logits, 5, dim=1)
#                         task_labels_expanded = task_labels.unsqueeze(1)
#                         correct_top5 = (top5_indices == task_labels_expanded).sum().item()
#                         correct_preds_val_top5_A1 += correct_top5

#     # --- (6) 评估和保存逻辑 (保持不变) ---
    
#     # 计算该轮次的平均准确率和损失
#     print(f"\n--- 轮次 {epoch_num + 1}/{epoch} 总结 ---")
    
#     print("[训练集]")
#     print(f"  平均损失: {total_loss_train / len(train_loader):.4f}")
#     for task in task_names:
#          acc = correct_preds_train[task] / valid_samples_train[task] if valid_samples_train[task] > 0 else 0
#          print(f"  准确率 ({task}): {acc:.4f} ({correct_preds_train[task]}/{valid_samples_train[task]})")

#     print("[验证集]")
#     print(f"  平均损失: {total_loss_val / len(dev_loader):.4f}")
#     val_accuracies = {}
#     for task in task_names:
#          acc = correct_preds_val[task] / valid_samples_val[task] if valid_samples_val[task] > 0 else 0
#          val_accuracies[task] = acc
#          print(f"  Top-1 准确率 ({task}): {acc:.4f} ({correct_preds_val[task]}/{valid_samples_val[task]})")

#     acc_top5_a1 = correct_preds_val_top5_A1 / valid_samples_val['A1'] if valid_samples_val['A1'] > 0 else 0
#     print(f"  Top-5 准确率 (A1): {acc_top5_a1:.4f} ({correct_preds_val_top5_A1}/{valid_samples_val['A1']})")
#     print(f"  (注意: A2_L4 Top-1 应与 A1 Top-1 相同)")

#     # 基于 A1 验证准确率保存最佳模型
#     current_dev_acc_a1 = val_accuracies['A1']
#     if current_dev_acc_a1 > best_dev_acc_a1:
#         best_dev_acc_a1 = current_dev_acc_a1
#         print(f" 新的最佳模型已保存！验证 Top-1 (A1): {best_dev_acc_a1:.4f}")
#         save_checkpoint(
#             save_name='best.pt',
#             model=model,
#             optimizer=optimizer,
#             scheduler=scheduler,
#             scaler=scaler,
#             epoch=epoch_num,
#             best_acc=best_dev_acc_a1,
#             save_path=save_path
#         )
#     print("-" * 60)

# # 保存最后一个轮次的模型
# print(f"\n 训练完成！正在保存最终 Checkpoint...")
# save_checkpoint(
#     save_name='last.pt',
#     model=model,
#     optimizer=optimizer,
#     scheduler=scheduler,
#     scaler=scaler,
#     epoch=epoch - 1, # epoch 索引从 0 开始, 所以最后一轮是 epoch-1
#     best_acc=best_dev_acc_a1,
#     save_path=save_path
# )
# print(f"最佳验证 Top-1 准确率 (A1): {best_dev_acc_a1:.4f}")
# print(f"模型保存在: {save_path}")
# print("=" * 60)

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os
from torch.utils.data import DataLoader
from bert_get_data import HMBertClassifier, GenerateData # 导入修改后的类
import joblib # 用于加载编码器
from torch.cuda.amp import GradScaler 
from torch.amp import autocast
from transformers import get_linear_schedule_with_warmup # 导入 Scheduler
# 【新增】: 评估指标
from sklearn.metrics import f1_score 
# 【删除】: 不再需要
# from sklearn.utils.class_weight import compute_class_weight 

# --- 【新增】Focal Loss 的标准实现 (论文的思想 [cite: 404-405, 421]) ---
class FocalLoss(nn.Module):
    """
    一个更健壮、更正确的 Focal Loss 实现。
    alpha (float): 论文 [38] (cite: 618) 中建议的标量平衡因子, e.g., 0.25
    gamma (float): 论文 [38] (cite: 618) 中建议的专注参数, e.g., 2.0
    """
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        # reduction='none' 让我们能手动应用所有逻辑
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)

    def forward(self, logits, labels):
        # 1. 计算标准的交叉熵损失 (但尚未 reduction)
        ce = self.ce_loss(logits, labels)
        
        # 2. 获取有效（非 ignore_index）的损失
        valid_mask = (labels != self.ignore_index)
        ce_valid = ce[valid_mask]
        
        if ce_valid.numel() == 0:
            # 如果这个批次所有样本都被忽略了
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # 3. 计算 pt (模型对正确类别的置信度)
        pt = torch.exp(-ce_valid)
        
        # 4. 计算 Focal Loss 核心: (1-pt)^gamma * ce
        focal_term = (1.0 - pt) ** self.gamma
        
        # 5. 应用 Alpha (如果提供了)
        # 论文 [38] 的 alpha 是一个标量
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_term = self.alpha
            else:
                alpha_term = 0.25 # 默认值
        else:
            alpha_term = 1.0

        # 最终损失 = alpha * (1-pt)^gamma * ce
        focal_loss = alpha_term * focal_term * ce_valid

        return focal_loss.mean()
# --- 结束 Focal Loss ---


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(save_name, model, optimizer, scheduler, scaler, epoch, best_acc, save_path):
    """保存完整的训练状态以便断点续训"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 包装为 torch.compile 兼容模式 
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
save_path = '/data/HMCTS/models/changweibuqi' # 【修改】: 建议为新实验设置新路径
#data_prefix = '/data/Data/changweibuqi/data_expert_model' 
data_prefix = '/data/Data/test/data' 

# DataLoader 优化
dataloader_num_workers = 16 
dataloader_pin_memory = True 


max_grad_norm = 1.0       # 梯度裁剪阈值
warmup_ratio = 0.03       

# --- 【新增】: 论文策略的超参数 ---
# 1. Focal Loss 参数 (gamma=2, alpha=0.25 是论文 [38] 的标准设置) [cite: 618]
focal_loss_gamma = 2.0
focal_loss_alpha = 0.25 # 设为 None 也可以

# 2. 层级损失权重 (模仿论文 Eq.4 的 beta 和 1-beta) 
loss_weights = {
    'A1': 0.3,      # L10
    'A2_L1': 0.1,   # L2
    'A2_L2': 0.1,   # L4
    'A2_L3': 0.2,   # L6
    'A2_L4': 0.3    # L10
}
# --- 结束 ---


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

# 使用每个头的正确类别数量定义模型
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

# PyTorch 2.0+ JIT 编译
try:
    model = torch.compile(model)
    print("模型已通过 torch.compile() 加速！")
except Exception as e:
    print(f"torch.compile() 失败 (可能是 PyTorch 版本过低或模型不支持): {e}")


# 定义损失函数和优化器
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# 将模型和损失函数移到 GPU
model = model.to(device)


# --- (2) 【修改】: 加载数据 (删除类别加权计算) ---
# 使用修改后的 GenerateData 函数构建数据集
print("加载数据集...")
try:
    train_dataset = GenerateData(mode='train', data_prefix=data_prefix)
    dev_dataset = GenerateData(mode='val', data_prefix=data_prefix)
    
    # --- 【删除】: 开始 ---
    # (删除了所有 'train_df', 'compute_class_weight', 'class_weights_l4_tensor' 相关的代码)
    # --- 【删除】: 结束 ---
    
    # 优化 DataLoader
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
# --- (2) 结束 ---


# --- (3) 【修改】: 定义 Focal Loss 和标准 CE Loss ---

# 1. 不加权的 (用于 L1, L2, L3)
criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)

# 2. Focal Loss (用于 A1, A2_L4), 模仿论文 [cite: 404-405, 421]
criterion_focal = FocalLoss(
    gamma=focal_loss_gamma, 
    alpha=focal_loss_alpha, # 论文 [38] 建议 0.25 [cite: 618]
    ignore_index=-1
).to(device)

print("!!! 已启用 Focal Loss (用于 A1 和 A2_L4) [模仿论文] !!!")
print("!!! 已启用 层级损失加权 (用于所有任务) [模仿论文] !!!")
# --- (3) 结束 ---


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
best_dev_acc_a1 = 0 # 我们仍使用 A1 Acc 保存模型, 您可以改为 f1_macro_a1
best_dev_f1_macro_a1 = 0 # 【新增】: 追踪最佳 F1 Macro
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

            # --- (4) 【修改】: 论文策略的损失计算  ---
            
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
            
            # --- (4) 结束 ---
        
        # 健壮的反向传播步骤
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
        
        # ========================================================
        
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
    
    # 【新增】: F1 分数计算的存储桶
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

                # --- (5) 【修改】: 论文策略的损失计算  ---
                
                loss_a1_val = criterion_focal(logits['A1'], labels['A1'])
                loss_a2_l4_val = criterion_focal(logits['A2_L4'], labels['A2_L4'])
                
                loss_a2_l1_val = criterion(logits['A2_L1'], labels['A2_L1'])
                loss_a2_l2_val = criterion(logits['A2_L2'], labels['A2_L2'])
                loss_a2_l3_val = criterion(logits['A2_L3'], labels['A2_L3'])
                
                # 使用层级权重加和 (模仿论文 Eq.4)
                batch_loss_val = (loss_weights['A1'] * loss_a1_val + 
                                  loss_weights['A2_L1'] * loss_a2_l1_val + 
                                  loss_weights['A2_L2'] * loss_a2_l2_val + 
                                  loss_weights['A2_L3'] * loss_a2_l3_val + 
                                  loss_weights['A2_L4'] * loss_a2_l4_val)
                
                # --- (5) 结束 ---
            
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
                    
                    # 【新增】: 为 F1 分数收集数据
                    all_labels_val[task].extend(task_labels.cpu().numpy())
                    all_preds_val[task].extend(task_preds.cpu().numpy())
                    
                    if task == 'A1':
                        # Top-5 准确率
                        _, top5_indices = torch.topk(task_logits, 5, dim=1)
                        task_labels_expanded = task_labels.unsqueeze(1)
                        correct_top5 = (top5_indices == task_labels_expanded).sum().item()
                        correct_preds_val_top5_A1 += correct_top5

    # --- (6) 【修改】: 评估和保存逻辑 (已更新 F1) ---
    
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
        
        # 【新增】: 计算 F1 分数
        if valid_samples_val[task] > 0:
            # 宏F1: 平等对待所有类别 (评估稀有类别的关键指标)
            f1_macro = f1_score(all_labels_val[task], all_preds_val[task], average='macro', zero_division=0)
            val_f1_macro[task] = f1_macro
            # 加权F1: 按样本量加权 (论文  使用的指标)
            f1_weighted = f1_score(all_labels_val[task], all_preds_val[task], average='weighted', zero_division=0)
            val_f1_weighted[task] = f1_weighted
            
            # 只打印 A1 (L10) 和 A2_L1 (L2) 的 F1 分数
            if task == 'A1' or task == 'A2_L1':
                print(f"  F1-Macro ({task}):   {f1_macro:.4f}  <-- (评估稀有类别的关键指标)")
                print(f"  F1-Weighted ({task}): {f1_weighted:.4f}  <-- (论文  指标)")

    acc_top5_a1 = correct_preds_val_top5_A1 / valid_samples_val['A1'] if valid_samples_val['A1'] > 0 else 0
    print(f"  Top-5 准确率 (A1): {acc_top5_a1:.4f} ({correct_preds_val_top5_A1}/{valid_samples_val['A1']})")
    print(f"  (注意: A2_L4 Top-1 应与 A1 Top-1 相同)")

    # 【修改】: 现在我们同时追踪 Acc 和 Macro F1
    current_dev_acc_a1 = val_accuracies['A1']
    current_dev_f1_macro_a1 = val_f1_macro.get('A1', 0) # 获取 A1 的 Macro F1

    # 您可以选择是按 Acc 保存还是按 F1 保存
    # 这里我们仍按 Acc (A1) 保存
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
    
    # 【新增】: 按 Macro F1 保存最佳模型
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
            best_acc=best_dev_acc_a1, # 您可以自定义保存内容
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