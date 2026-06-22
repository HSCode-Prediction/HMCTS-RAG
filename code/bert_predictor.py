import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
from transformers import BertModel, BertTokenizer
import joblib


class MultiTaskBertClassifier(nn.Module):
    
    def __init__(self, bert_model_name: str, num_classes: Dict[str, int]):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        
        # 每个层级一个分类头
        self.classifiers = nn.ModuleDict()
        for name, num_class in num_classes.items():
            self.classifiers[name] = nn.Linear(hidden_size, num_class)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled = outputs.pooler_output
        
        logits = {}
        for name, classifier in self.classifiers.items():
            logits[name] = classifier(pooled)
        
        return logits


class BertPredictor:
    """BERT预测器"""
    
    def __init__(self, 
                 bert_model_name: str,
                 data_prefix: str,
                 classifier_ckpt: str,
                 device: str = 'cuda'):

        self.device = device
        self.tokenizer = None
        self.model = None
        self.encoders = {}
        self.num_classes = {}
        
        self._load_encoders(data_prefix)
        self._load_model(bert_model_name, classifier_ckpt)
        
        # 缓存
        self.cache = {}
        self.cache_hits = 0
    
    def _load_encoders(self, data_prefix: str):
        """加载标签编码器"""
        level_map = {
            2: ['L1', 'L2', '2'],
            4: ['L2', 'L4', '4'],
            6: ['L3', 'L6', '6'],
            8: ['L4', 'L8', '8'],
            10: ['A1', 'L10', '10', 'full']
        }
        
        for level, suffixes in level_map.items():
            for suffix in suffixes:
                # 尝试不同的文件名格式
                candidates = [
                    f"{data_prefix}_L{level}_encoder.pkl",
                    f"{data_prefix}_A2_{suffix}_encoder.pkl",
                    f"{data_prefix}_{suffix}_encoder.pkl",
                ]
                
                for path in candidates:
                    if os.path.exists(path):
                        try:
                            self.encoders[level] = joblib.load(path)
                            self.num_classes[f'L{level}'] = len(self.encoders[level].classes_)
                            print(f"[BertPredictor] ✅ 加载编码器 L{level}: {path}")
                            break
                        except Exception as e:
                            print(f"[BertPredictor] ⚠️ 加载失败 {path}: {e}")
        
        print(f"[BertPredictor] 编码器: {list(self.encoders.keys())}")
    
    def _load_model(self, bert_model_name: str, classifier_ckpt: str):
        """加载模型"""
        print(f"[BertPredictor] 加载BERT: {bert_model_name}")
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        if self.num_classes:
            self.model = MultiTaskBertClassifier(bert_model_name, self.num_classes)
            
            if os.path.exists(classifier_ckpt):
                print(f"[BertPredictor] 加载checkpoint: {classifier_ckpt}")
                state_dict = torch.load(classifier_ckpt, map_location='cpu')
                
                # 处理可能的key前缀问题
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                self.model.load_state_dict(state_dict, strict=False)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 半精度
            self.model = self.model.half()
            
            print(f"[BertPredictor] ✅ 模型加载完成")
        else:
            print(f"[BertPredictor] ⚠️ 无编码器，模型未加载")
    
    def predict(self, text: str, level: int, top_k: int = 5) -> List[Tuple[str, float]]:
        # 检查缓存
        cache_key = f"{text[:100]}:{level}:{top_k}"
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        if self.model is None or level not in self.encoders:
            return []
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=256,
                padding='max_length'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                logits = self.model(**inputs)
            
            task_key = f'L{level}'
            if task_key not in logits:
                return []
            
            probs = torch.softmax(logits[task_key], dim=-1)[0]
            
            # Top-k
            k = min(top_k, probs.numel())
            top_probs, top_indices = torch.topk(probs, k)
            
            # 解码
            codes = self.encoders[level].inverse_transform(top_indices.cpu().numpy())
            
            results = []
            for code, prob in zip(codes, top_probs.cpu().tolist()):
                results.append((str(code), float(prob)))
            
            # 缓存
            self.cache[cache_key] = results
            
            return results
            
        except Exception as e:
            print(f"[BertPredictor] Error: {e}")
            return []
    
    def predict_all_levels(self, text: str, top_k: int = 3) -> Dict[int, List[Tuple[str, float]]]:
        """预测所有层级"""
        results = {}
        for level in [2, 4, 6, 8, 10]:
            if level in self.encoders:
                results[level] = self.predict(text, level, top_k)
        return results
    
    def predict_hierarchical(self, text: str, prefix: str = "", top_k: int = 5) -> List[Tuple[str, float]]:
 
        prefix_len = len(prefix)
        
        # 确定目标层级
        if prefix_len == 0:
            target_level = 2
        elif prefix_len == 2:
            target_level = 4
        elif prefix_len == 4:
            target_level = 6
        elif prefix_len == 6:
            target_level = 8
        elif prefix_len == 8:
            target_level = 10
        else:
            return []
        
        # 预测
        all_preds = self.predict(text, target_level, top_k * 3)
        
        # 过滤：只保留与前缀匹配的
        if prefix:
            filtered = [(code, prob) for code, prob in all_preds if code.startswith(prefix)]
        else:
            filtered = all_preds
        
        return filtered[:top_k]
    
    def get_uncertainty(self, text: str, level: int = 2) -> float:
        """
        计算预测不确定性（基于熵）
        """
        if self.model is None or level not in self.encoders:
            return 0.5
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=256,
                padding='max_length'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.model(**inputs)
            
            task_key = f'L{level}'
            if task_key not in logits:
                return 0.5
            
            probs = torch.softmax(logits[task_key], dim=-1)[0]
            
            # 计算熵
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            
            # 归一化（假设最大熵约为log(num_classes)）
            max_entropy = np.log(probs.numel())
            normalized = entropy / max_entropy
            
            return min(normalized, 1.0)
            
        except Exception as e:
            return 0.5


if __name__ == "__main__":
    # 测试
    predictor = BertPredictor(
        bert_model_name="/media/lzw/新加卷/LZW/Data_A800/A/models/bert_chinese",
        data_prefix="/media/lzw/新加卷/LZW/Data_A800/A/data/encoder",
        classifier_ckpt="/media/lzw/新加卷/LZW/Data_A800/A/models/bert_classifier.pt"
    )
    
    text = "不锈钢螺丝，用于机械设备"
    print(f"\n测试: {text}")
    
    for level in [2, 4, 6, 8, 10]:
        preds = predictor.predict(text, level, top_k=3)
        print(f"L{level}: {preds}")
