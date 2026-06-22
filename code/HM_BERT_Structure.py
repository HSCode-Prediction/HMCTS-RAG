# 文件: run_src/HM_BERT_Structure.py

import json
import joblib
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
import os

class HMDataset(Dataset):
    def __init__(self, data_file, encoder_paths, tokenizer, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        # 从 .jsonl 文件逐行加载数据
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        self.data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"警告：跳过 {data_file} 中的无效 JSON 行: {line.strip()}")
        except FileNotFoundError:
            print(f"错误：在 {data_file} 未找到数据文件")
            raise # 重新抛出异常

        self.max_length = max_length
        self.encoder_paths = encoder_paths # 存储路径，如果只使用 idx 可能不需要

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('product_description', '') # 使用正确的键名

        # Tokenize 文本
        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        a1_label_idx = item.get('A1_idx', -1)
        a2_l1_label_idx = item.get('A2_L1_idx', -1)
        a2_l2_label_idx = item.get('A2_L2_idx', -1)
        a2_l3_label_idx = item.get('A2_L3_idx', -1)
        a2_l4_label_idx = item.get('A2_L4_idx', -1) 

        if -1 in [a1_label_idx, a2_l1_label_idx, a2_l2_label_idx, a2_l3_label_idx, a2_l4_label_idx]:
             print(f"警告：索引 {idx} (ID: {item.get('id', 'N/A')}) 的项目包含缺失或无效的编码标签。")

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'A1_labels_idx': a1_label_idx,      
            'A2_L1_labels_idx': a2_l1_label_idx,  
            'A2_L2_labels_idx': a2_l2_label_idx,  
            'A2_L3_labels_idx': a2_l3_label_idx,  
            'A2_L4_labels_idx': a2_l4_label_idx    
        }

    def __len__(self):
        return len(self.data)

class HMBertClassifier(nn.Module):
    """ 修改为 5 个任务：A1, A2_L1, A2_L2, A2_L3, A2_L4 """
    def __init__(self, bert_model_name, num_a1, num_a2_l1, num_a2_l2, num_a2_l3, num_a2_l4):
        super(HMBertClassifier, self).__init__()
        # ===== BERT 编码层 =====
        try:
            self.bert = BertModel.from_pretrained(bert_model_name)
        except Exception as e:
            print(f"从 {bert_model_name} 加载 BERT 模型时出错: {e}")
            print("请确保路径正确且模型文件存在。")
            raise # 重新抛出异常

        # ===== Dropout 层 =====
        self.dropout = nn.Dropout(0.3)

        # ===== 五个分类头直接连接 BERT 输出 =====
        self.classifier_a1 = nn.Linear(768, num_a1)        # 直接预测 10 位码
        self.classifier_a2_l1 = nn.Linear(768, num_a2_l1)  # 预测章 (2 位)
        self.classifier_a2_l2 = nn.Linear(768, num_a2_l2)  # 预测目 (4 位)
        self.classifier_a2_l3 = nn.Linear(768, num_a2_l3)  # 预测子目 (6 位)
        self.classifier_a2_l4 = nn.Linear(768, num_a2_l4)  # 为分层路径预测最终码 (10 位)


    def forward(self, input_ids, attention_mask, token_type_ids=None): # 添加 token_type_ids
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids # 将 token_type_ids 传递给 self.bert
        )
        
        pooled_output = outputs.pooler_output # 使用 [CLS] token 的表示

        dropout_output = self.dropout(pooled_output)

        a1_logits = self.classifier_a1(dropout_output)
        a2_l1_logits = self.classifier_a2_l1(dropout_output)
        a2_l2_logits = self.classifier_a2_l2(dropout_output)
        a2_l3_logits = self.classifier_a2_l3(dropout_output)
        a2_l4_logits = self.classifier_a2_l4(dropout_output)

        return a1_logits, a2_l1_logits, a2_l2_logits, a2_l3_logits, a2_l4_logits