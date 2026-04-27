# import json
# import joblib # 使用 joblib 加载 fenceng.py 保存的编码器
# from torch.utils.data import Dataset
# from transformers import BertTokenizer
# from torch import nn
# from transformers import BertModel
# import os # 导入 os 模块

# # 确保 bert_name 指向你下载的模型目录
# bert_name = '/root/HMCTS/models/bert_chinese'
# try:
#     tokenizer = BertTokenizer.from_pretrained(bert_name)
# except Exception as e:
#     print(f"从 {bert_name} 加载 tokenizer 时出错: {e}")
#     print("请确保路径正确且模型文件存在。")
#     exit()

# class HMDataset(Dataset):
#     """ 读取预处理的 .jsonl 数据并使用预先训练好的编码器 """
#     def __init__(self, data_file, encoder_paths, max_length=128):
#         self.data = []
#         # 从 .jsonl 文件逐行加载数据
#         try:
#             with open(data_file, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     try:
#                         self.data.append(json.loads(line.strip()))
#                     except json.JSONDecodeError:
#                         print(f"警告：跳过 {data_file} 中的无效 JSON 行: {line.strip()}")
#         except FileNotFoundError:
#             print(f"错误：在 {data_file} 未找到数据文件")
#             raise # 重新抛出异常，让调用者知道

#         self.max_length = max_length
#         self.encoder_paths = encoder_paths # 存储路径，如果只使用 idx 可能不需要

#         # 如果我们直接使用 _idx 字段，这里不需要加载编码器
#         # 如果你需要在数据集中转换 *原始* HS 编码，你会加载它们：
#         # self.encoders = {}
#         # for task_name, path in encoder_paths.items():
#         #     try:
#         #         self.encoders[task_name] = joblib.load(path)
#         #     except FileNotFoundError:
#         #          print(f"错误：在 {path} 未找到编码器文件")
#         #          raise

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         text = item.get('product_description', '') # 使用正确的键名

#         # Tokenize 文本
#         encoding = tokenizer(
#             text,
#             padding='max_length',
#             max_length=self.max_length,
#             truncation=True,
#             return_tensors="pt"
#         )

#         # 直接从数据文件中获取预编码的整数标签
#         # 使用 .get 并提供默认值 (-1 或其他指示符) 以增强鲁棒性
#         a1_label_idx = item.get('A1_idx', -1)
#         a2_l1_label_idx = item.get('A2_L1_idx', -1)
#         a2_l2_label_idx = item.get('A2_L2_idx', -1)
#         a2_l3_label_idx = item.get('A2_L3_idx', -1)
#         a2_l4_label_idx = item.get('A2_L4_idx', -1) # 与 A1_idx 相同，但为了保持一致性

#         # 检查是否有无效标签
#         if -1 in [a1_label_idx, a2_l1_label_idx, a2_l2_label_idx, a2_l3_label_idx, a2_l4_label_idx]:
#              print(f"警告：索引 {idx} (ID: {item.get('id', 'N/A')}) 的项目包含缺失或无效的编码标签。")

#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'A1_labels_idx': a1_label_idx,         # A1 任务的标签索引
#             'A2_L1_labels_idx': a2_l1_label_idx,   # A2 L1 任务的标签索引
#             'A2_L2_labels_idx': a2_l2_label_idx,   # A2 L2 任务的标签索引
#             'A2_L3_labels_idx': a2_l3_label_idx,   # A2 L3 任务的标签索引
#             'A2_L4_labels_idx': a2_l4_label_idx    # A2 L4 任务的标签索引
#         }

#     def __len__(self):
#         return len(self.data)

# class HMBertClassifier(nn.Module):
#     """ 修改为 5 个任务：A1, A2_L1, A2_L2, A2_L3, A2_L4 """
#     def __init__(self, num_a1, num_a2_l1, num_a2_l2, num_a2_l3, num_a2_l4):
#         super(HMBertClassifier, self).__init__()
#         # ===== BERT 编码层 =====
#         try:
#             self.bert = BertModel.from_pretrained(bert_name)
#         except Exception as e:
#             print(f"从 {bert_name} 加载 BERT 模型时出错: {e}")
#             print("请确保路径正确且模型文件存在。")
#             raise # 重新抛出异常

#         # ===== Dropout 层 =====
#         self.dropout = nn.Dropout(0.3)

#         # ===== 五个分类头直接连接 BERT 输出 =====
#         # 如果性能不佳，可以添加中间层
#         self.classifier_a1 = nn.Linear(768, num_a1)         # 直接预测 10 位码
#         self.classifier_a2_l1 = nn.Linear(768, num_a2_l1)   # 预测章 (2 位)
#         self.classifier_a2_l2 = nn.Linear(768, num_a2_l2)   # 预测目 (4 位)
#         self.classifier_a2_l3 = nn.Linear(768, num_a2_l3)   # 预测子目 (6 位)
#         self.classifier_a2_l4 = nn.Linear(768, num_a2_l4)   # 为分层路径预测最终码 (10 位)

#     def forward(self, input_ids, attention_mask):
#         # ===== BERT 编码层 =====
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output # 使用 [CLS] token 的表示

#         # ===== Dropout 层 =====
#         dropout_output = self.dropout(pooled_output)

#         # ===== 直接分类 =====
#         a1_logits = self.classifier_a1(dropout_output)
#         a2_l1_logits = self.classifier_a2_l1(dropout_output)
#         a2_l2_logits = self.classifier_a2_l2(dropout_output)
#         a2_l3_logits = self.classifier_a2_l3(dropout_output)
#         a2_l4_logits = self.classifier_a2_l4(dropout_output)

#         return a1_logits, a2_l1_logits, a2_l2_logits, a2_l3_logits, a2_l4_logits

# def GenerateData(mode, data_prefix='/root/agent/bert_mtl_all_data'):
#     """ 使用预处理文件生成 HM-BERT 数据集 """
#     # 使用指定模式的 _encoded.jsonl 文件
#     data_file = f'{data_prefix}_{mode}_encoded.jsonl'

#     # 定义预先训练好的编码器 .pkl 文件的路径
#     encoder_paths = {
#         'A1': f'{data_prefix}_A1_encoder.pkl',
#         'A2_L1': f'{data_prefix}_A2_L1_encoder.pkl',
#         'A2_L2': f'{data_prefix}_A2_L2_encoder.pkl',
#         'A2_L3': f'{data_prefix}_A2_L3_encoder.pkl',
#         'A2_L4': f'{data_prefix}_A2_L4_encoder.pkl'
#     }

#     # 检查数据文件是否存在
#     if not os.path.exists(data_file):
#          raise FileNotFoundError(f"模式 '{mode}' 的数据文件未找到于: {data_file}")

#     # 检查所有编码器文件是否存在（可选，但良好的实践）
#     for task, path in encoder_paths.items():
#         if not os.path.exists(path):
#             print(f"警告：任务 '{task}' 的编码器文件未找到于: {path}。推理/评估可能会失败。")

#     # 创建数据集实例
#     dataset = HMDataset(data_file, encoder_paths)
#     return dataset


import json
import joblib 
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
import os 
import pandas as pd # <-- 1. 导入 PANDAS

# 确保 bert_name 指向你下载的模型目录
bert_name = '/data/models/bert-base-chinese'
try:
    tokenizer = BertTokenizer.from_pretrained(bert_name)
except Exception as e:
    print(f"从 {bert_name} 加载 tokenizer 时出错: {e}")
    print("请确保路径正确且模型文件存在。")
    exit()

class HMDataset(Dataset):
    """ 读取预处理的 .jsonl 数据并使用预先训练好的编码器 """
    def __init__(self, data_file, encoder_paths, max_length=128):
        
        # --- 2. 修改：使用 pandas 一次性加载所有数据 ---
        try:
            print(f"正在从 {data_file} 加载数据到 DataFrame...")
            # 使用 pandas 读取 .jsonl 文件
            self.df = pd.read_json(data_file, lines=True)
            print(f"已加载 {len(self.df)} 条样本。")
            
        except FileNotFoundError:
            print(f"错误：在 {data_file} 未找到数据文件")
            raise # 重新抛出异常，让调用者知道
        except Exception as e:
            print(f"使用 pandas 加载 {data_file} 时发生错误: {e}")
            raise

        self.max_length = max_length
        self.encoder_paths = encoder_paths # 存储路径

    def __getitem__(self, idx):
        # --- 3. 修改：从 self.df 中按索引获取数据 ---
        item = self.df.iloc[idx] # 使用 .iloc G根据整数索引高效获取行
        
        text = item.get('product_description', '') # 使用正确的键名

        # Tokenize 文本
        encoding = tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        # 直接从数据文件中获取预编码的整数标签
        a1_label_idx = item.get('A1_idx', -1)
        a2_l1_label_idx = item.get('A2_L1_idx', -1)
        a2_l2_label_idx = item.get('A2_L2_idx', -1)
        a2_l3_label_idx = item.get('A2_L3_idx', -1)
        a2_l4_label_idx = item.get('A2_L4_idx', -1) 

        # 检查是否有无效标签 (在训练中可以跳过打印以提高速度)
        if -1 in [a1_label_idx, a2_l1_label_idx, a2_l2_label_idx, a2_l3_label_idx, a2_l4_label_idx]:
             pass # 暂时跳过警告

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'A1_labels_idx': a1_label_idx,      # A1 任务的标签索引
            'A2_L1_labels_idx': a2_l1_label_idx,   # A2 L1 任务的标签索引
            'A2_L2_labels_idx': a2_l2_label_idx,   # A2 L2 任务的标签索引
            'A2_L3_labels_idx': a2_l3_label_idx,   # A2 L3 任务的标签索引
            'A2_L4_labels_idx': a2_l4_label_idx    # A2 L4 任务的标签索引
        }

    def __len__(self):
        # --- 4. 修改：返回 self.df 的长度 ---
        return len(self.df)

class HMBertClassifier(nn.Module):
    """ 修改为 5 个任务：A1, A2_L1, A2_L2, A2_L3, A2_L4 """
    def __init__(self, num_a1, num_a2_l1, num_a2_l2, num_a2_l3, num_a2_l4):
        super(HMBertClassifier, self).__init__()
        # ===== BERT 编码层 =====
        try:
            self.bert = BertModel.from_pretrained(bert_name)
        except Exception as e:
            print(f"从 {bert_name} 加载 BERT 模型时出错: {e}")
            print("请确保路径正确且模型文件存在。")
            raise # 重新抛出异常

        # ===== Dropout 层 =====
        self.dropout = nn.Dropout(0.3)

        # ===== 五个分类头直接连接 BERT 输出 =====
        self.classifier_a1 = nn.Linear(768, num_a1)       # 直接预测 10 位码
        self.classifier_a2_l1 = nn.Linear(768, num_a2_l1)   # 预测章 (2 位)
        self.classifier_a2_l2 = nn.Linear(768, num_a2_l2)   # 预测目 (4 位)
        self.classifier_a2_l3 = nn.Linear(768, num_a2_l3)   # 预测子目 (6 位)
        self.classifier_a2_l4 = nn.Linear(768, num_a2_l4)   # 为分层路径预测最终码 (10 位)

    def forward(self, input_ids, attention_mask):
        # ===== BERT 编码层 =====
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output # 使用 [CLS] token 的表示

        # ===== Dropout 层 =====
        dropout_output = self.dropout(pooled_output)

        # ===== 直接分类 =====
        a1_logits = self.classifier_a1(dropout_output)
        a2_l1_logits = self.classifier_a2_l1(dropout_output)
        a2_l2_logits = self.classifier_a2_l2(dropout_output)
        a2_l3_logits = self.classifier_a2_l3(dropout_output)
        a2_l4_logits = self.classifier_a2_l4(dropout_output)

        return a1_logits, a2_l1_logits, a2_l2_logits, a2_l3_logits, a2_l4_logits

def GenerateData(mode, data_prefix='/data/Data/test/data'):
    """ 使用预处理文件生成 HM-BERT 数据集 """
    # 使用指定模式的 _encoded.jsonl 文件
    data_file = f'{data_prefix}_{mode}_encoded.jsonl'

    # 定义预先训练好的编码器 .pkl 文件的路径
    encoder_paths = {
        'A1': f'{data_prefix}_A1_encoder.pkl',
        'A2_L1': f'{data_prefix}_A2_L1_encoder.pkl',
        'A2_L2': f'{data_prefix}_A2_L2_encoder.pkl',
        'A2_L3': f'{data_prefix}_A2_L3_encoder.pkl',
        'A2_L4': f'{data_prefix}_A2_L4_encoder.pkl'
    }

    # 检查数据文件是否存在
    if not os.path.exists(data_file):
         raise FileNotFoundError(f"模式 '{mode}' 的数据文件未找到于: {data_file}")

    # 检查所有编码器文件是否存在（可选，但良好的实践）
    for task, path in encoder_paths.items():
        if not os.path.exists(path):
            print(f"警告：任务 '{task}' 的编码器文件未找到于: {path}。推理/评估可能会失败。")

    # 创建数据集实例
    dataset = HMDataset(data_file, encoder_paths)
    return dataset