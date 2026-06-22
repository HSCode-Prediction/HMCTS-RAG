import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from transformers import BertTokenizer


try:
    from HM_BERT_Structure import HMBertClassifier
except ImportError:
    HMBertClassifier = None


class ActionType(Enum):
    """动作类型"""
    A2_DIRECT = "A2_Direct"      
    A3_HIERARCHICAL = "A3_Hierarchical"  
    NEED_A4 = "Need_A4"          
    NEED_A5 = "Need_A5"          


@dataclass
class BertPrediction:
    code: str
    confidence: float
    level: int  # 2/4/6/8/10


@dataclass
class BertDecision:
    action: ActionType
    predictions: List[BertPrediction]
    uncertainty: float
    reasoning: str


class BertAgent:
    def __init__(self,
                 bert_model_name: str,
                 data_prefix: str,
                 classifier_ckpt: str,
                 device: str = 'cuda'):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.encoders = {}
        self.num_classes = {}
        
        # 统计
        self.call_count = 0
        self.cache = {}
        self.cache_hits = 0
        
        # 加载模型
        self._load_model(bert_model_name, data_prefix, classifier_ckpt)
    
    def _load_model(self, bert_model_name: str, data_prefix: str, classifier_ckpt: str):
        """加载BERT模型和编码器"""
        print(f"[BertAgent] 加载编码器: {data_prefix}")
        
        # 加载编码器
        encoder_names = {
            'A1': f'{data_prefix}_A1_encoder.pkl',
            'A2_L1': f'{data_prefix}_A2_L1_encoder.pkl',
            'A2_L2': f'{data_prefix}_A2_L2_encoder.pkl',
            'A2_L3': f'{data_prefix}_A2_L3_encoder.pkl',
            'A2_L4': f'{data_prefix}_A2_L4_encoder.pkl',
        }
        
        for name, path in encoder_names.items():
            if os.path.exists(path):
                try:
                    self.encoders[name] = joblib.load(path)
                    self.num_classes[name] = len(self.encoders[name].classes_)
                    print(f"   {name}: {self.num_classes[name]} classes")
                except Exception as e:
                    print(f"   {name}: {e}")
        
        if not self.encoders:
            print("[BertAgent]  无可用编码器")
            return
        
        # 加载Tokenizer
        print(f"[BertAgent] 加载BERT: {bert_model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # 加载模型
        if HMBertClassifier is None:
            print("[BertAgent] HMBertClassifier未找到")
            return
        
        self.model = HMBertClassifier(
            bert_model_name=bert_model_name,
            num_a1=self.num_classes.get('A1', 100),
            num_a2_l1=self.num_classes.get('A2_L1', 100),
            num_a2_l2=self.num_classes.get('A2_L2', 100),
            num_a2_l3=self.num_classes.get('A2_L3', 100),
            num_a2_l4=self.num_classes.get('A2_L4', 100)
        )
        
        # 加载权重
        if os.path.exists(classifier_ckpt):
            print(f"[BertAgent] 加载checkpoint: {classifier_ckpt}")
            ckpt = torch.load(classifier_ckpt, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt)
            
            # 处理key前缀
            from collections import OrderedDict
            new_state = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('_orig_mod.', '').replace('module.', '')
                new_state[name] = v
            
            self.model.load_state_dict(new_state, strict=False)
        
        self.model.to(self.device).eval()
        self.model.half()  # FP16加速
        
        print("[BertAgent]  模型加载完成")
    
    def _get_task_key(self, level) -> str:
        """获取任务key"""
        if level == 'A1' or level == 10:
            return 'A1'
        elif level == 0 or level == 2:
            return 'A2_L1'
        elif level == 2 or level == 4:
            return 'A2_L2'
        elif level == 4 or level == 6:
            return 'A2_L3'
        elif level == 6 or level == 8:
            return 'A2_L4'
        return 'A1'
    
    def _predict_raw(self, text: str, task_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is None or self.tokenizer is None:
            return None, None
        
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding='max_length'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        idx_map = {'A1': 0, 'A2_L1': 1, 'A2_L2': 2, 'A2_L3': 3, 'A2_L4': 4}
        logits = outputs[idx_map[task_key]]
        probs = torch.softmax(logits, dim=-1)[0]
        
        return logits, probs
    
    def predict(self, text: str, level, top_k: int = 5) -> List[BertPrediction]:
        # 缓存检查
        cache_key = f"{text[:100]}:{level}:{top_k}"
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.call_count += 1
        
        task_key = self._get_task_key(level)
        if task_key not in self.encoders:
            return []
        
        try:
            _, probs = self._predict_raw(text, task_key)
            if probs is None:
                return []
            
            k = min(top_k, probs.numel())
            top_probs, top_indices = torch.topk(probs, k)
            
            codes = self.encoders[task_key].inverse_transform(top_indices.cpu().numpy())
            
            # 确定输出层级
            if level == 'A1' or level == 10:
                out_level = 10
            elif level == 0:
                out_level = 2
            elif level == 2:
                out_level = 4
            elif level == 4:
                out_level = 6
            elif level == 6:
                out_level = 8
            else:
                out_level = 10
            
            results = []
            for code, prob in zip(codes, top_probs.cpu().tolist()):
                results.append(BertPrediction(
                    code=str(code),
                    confidence=float(prob),
                    level=out_level
                ))
            
            self.cache[cache_key] = results
            return results
            
        except Exception as e:
            print(f"[BertAgent] Predict Error: {e}")
            return []
    
    def get_distribution(self, text: str, task_key: str = 'A2_L1') -> Optional[torch.Tensor]:
        """获取概率分布（用于计算不确定性）"""
        if self.model is None:
            return None
        
        try:
            _, probs = self._predict_raw(text, task_key)
            return probs
        except:
            return None
    
    def get_uncertainty(self, text: str) -> float:
        probs = self.get_distribution(text, 'A2_L1')
        if probs is None:
            return 0.5
        
        # 计算熵
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        # 归一化
        max_entropy = np.log(probs.numel())
        normalized = entropy / max_entropy
        
        return min(normalized, 1.0)
    
    def action_A2_direct(self, text: str, top_k: int = 5) -> List[BertPrediction]:
        """
        A2动作: 直接预测10位HSCode
        """
        return self.predict(text, 'A1', top_k)
    
    def action_A3_hierarchical(self, text: str, current_level: int, 
                                prefix: str = "", top_k: int = 5) -> List[BertPrediction]:
        preds = self.predict(text, current_level, top_k * 2)
        
        if prefix:
            preds = [p for p in preds if p.code.startswith(prefix)]
        
        return preds[:top_k]
    
    def decide(self, text: str, current_prefix: str = "") -> BertDecision:
       
        prefix_len = len(current_prefix)
        
        # 获取不确定性
        uncertainty = self.get_uncertainty(text)
        
        # 根据层级决定动作
        if prefix_len == 0:
            # 根节点：判断是直接预测还是分层
            direct_preds = self.action_A2_direct(text, top_k=3)
            hier_preds = self.action_A3_hierarchical(text, 0, "", top_k=3)
            
            if direct_preds and direct_preds[0].confidence > 0.8:
                return BertDecision(
                    action=ActionType.A2_DIRECT,
                    predictions=direct_preds,
                    uncertainty=uncertainty,
                    reasoning="高置信度直接预测"
                )
            else:
                return BertDecision(
                    action=ActionType.A3_HIERARCHICAL,
                    predictions=hier_preds,
                    uncertainty=uncertainty,
                    reasoning="进行分层预测"
                )
        else:
            # 中间节点：分层预测
            preds = self.action_A3_hierarchical(text, prefix_len, current_prefix, top_k=5)
            
            if not preds:
                return BertDecision(
                    action=ActionType.NEED_A5,
                    predictions=[],
                    uncertainty=1.0,
                    reasoning="无匹配候选，需要RAG增强"
                )
            
            # 检查是否需要仲裁
            if len(preds) >= 2:
                top_conf = preds[0].confidence
                second_conf = preds[1].confidence
                
                # 品目级别冲突
                if prefix_len == 2 and preds[0].code[:4] != preds[1].code[:4]:
                    if top_conf - second_conf < 0.15:
                        return BertDecision(
                            action=ActionType.NEED_A4,
                            predictions=preds,
                            uncertainty=uncertainty,
                            reasoning="品目冲突，需要LLM仲裁"
                        )
            
            # 低置信度
            if preds[0].confidence < 0.4:
                return BertDecision(
                    action=ActionType.NEED_A5,
                    predictions=preds,
                    uncertainty=uncertainty,
                    reasoning="置信度低，需要RAG增强"
                )
            
            return BertDecision(
                action=ActionType.A3_HIERARCHICAL,
                predictions=preds,
                uncertainty=uncertainty,
                reasoning="正常分层预测"
            )
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "call_count": self.call_count,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.cache)
        }


if __name__ == "__main__":
    # 测试
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agent = BertAgent(
        bert_model_name=os.path.join(base_dir, 'models', 'bert-base-chinese'),
        data_prefix=os.path.join(base_dir, 'data', 'suiji', 'data'),
        classifier_ckpt=os.path.join(base_dir, 'models', 'best_f1_macro.pt')
    )
    
    text = "不锈钢螺丝，用于机械设备"
    
    print("\nA2直接预测:")
    for p in agent.action_A2_direct(text, top_k=3):
        print(f"  {p.code} ({p.confidence:.4f})")
    
    print("\nA3分层预测 (L0):")
    for p in agent.action_A3_hierarchical(text, 0, "", top_k=3):
        print(f"  {p.code} ({p.confidence:.4f})")
    
    print("\n智能决策:")
    decision = agent.decide(text, "")
    print(f"  动作: {decision.action}")
    print(f"  不确定性: {decision.uncertainty:.4f}")
    print(f"  理由: {decision.reasoning}")
