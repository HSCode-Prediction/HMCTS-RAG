import os
import sys
import json
import math
import torch
import pickle
import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class UnifiedRetriever:
    
    def __init__(self, db_path: str, embedding_model_path: str = None):
        self.db_path = db_path
        
        # BM25索引
        self.bm25 = None
        self.bm25_texts = []
        self.bm25_codes = []
        self.bm25_sources = []
        
        # FAISS索引
        self.faiss_index = None
        self.faiss_texts = []
        self.faiss_codes = []
        self.embed_model = None
        
        # Few-shot库
        self.fewshot_db = {}
        
        # 层级结构
        self.hierarchy = {}
        
        # 缓存
        self.cache = {}
        
        self._load(embedding_model_path)
    
    def _load(self, embedding_model_path: str = None):
        """加载索引"""
        import jieba
        self.jieba = jieba
        
        # 1. 优先加载统一索引
        unified_path = os.path.join(self.db_path, "unified_bm25.pkl")
        if os.path.exists(unified_path):
            with open(unified_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data.get('bm25')
                self.bm25_texts = data.get('texts', [])
                self.bm25_codes = data.get('codes', [])
                self.bm25_sources = data.get('sources', [])
            print(f"[Retriever] ✅ 统一BM25: {len(self.bm25_texts)} docs")
        else:
            # 回退旧索引
            old_path = os.path.join(self.db_path, "heading_bm25.pkl")
            if os.path.exists(old_path):
                with open(old_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data.get('bm25')
                    self.bm25_texts = data.get('texts', [])
                    self.bm25_codes = data.get('codes', [])
                    self.bm25_sources = ['legacy'] * len(self.bm25_texts)
                print(f"[Retriever] ✅ 旧BM25: {len(self.bm25_texts)} docs")
        
        # 2. FAISS索引
        faiss_path = os.path.join(self.db_path, "unified_faiss.index")
        meta_path = os.path.join(self.db_path, "unified_faiss_meta.pkl")
        if os.path.exists(faiss_path) and embedding_model_path:
            try:
                import faiss
                from sentence_transformers import SentenceTransformer
                
                self.faiss_index = faiss.read_index(faiss_path)
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    self.faiss_texts = meta.get('texts', [])
                    self.faiss_codes = meta.get('codes', [])
                
                if os.path.exists(embedding_model_path):
                    self.embed_model = SentenceTransformer(embedding_model_path, device='cuda')
                    print(f"[Retriever] ✅ FAISS: {self.faiss_index.ntotal} vectors")
            except Exception as e:
                print(f"[Retriever] ⚠️ FAISS加载失败: {e}")
        
        # 3. Few-shot库
        fs_path = os.path.join(self.db_path, "fewshot_examples.json")
        if os.path.exists(fs_path):
            with open(fs_path, 'r', encoding='utf-8') as f:
                self.fewshot_db = json.load(f)
            print(f"[Retriever] ✅ Few-shot: {len(self.fewshot_db)} codes")
        
        # 4. 层级结构
        hier_path = os.path.join(self.db_path, "hierarchy_enhanced.pkl")
        if os.path.exists(hier_path):
            with open(hier_path, 'rb') as f:
                self.hierarchy = pickle.load(f)
            print(f"[Retriever]  层级结构已加载")
    
    def search_bm25(self, query: str, top_k: int = 10, prefix: str = "") -> List[Tuple[str, str, float, str]]:
        """BM25检索 -> [(code, text, score, source)]"""
        if self.bm25 is None:
            return []
        
        tokens = list(self.jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in indices:
            if len(results) >= top_k:
                break
            if scores[idx] <= 0:
                continue
            code = self.bm25_codes[idx] if idx < len(self.bm25_codes) else ""
            if prefix and code and not code.startswith(prefix):
                continue
            source = self.bm25_sources[idx] if idx < len(self.bm25_sources) else "bm25"
            results.append((code, self.bm25_texts[idx], float(scores[idx]), source))
        
        return results
    
    def search_faiss(self, query: str, top_k: int = 10, prefix: str = "") -> List[Tuple[str, str, float, str]]:
        """向量检索"""
        if self.faiss_index is None or self.embed_model is None:
            return []
        
        query_vec = self.embed_model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec).astype('float32')
        
        search_k = top_k * 3 if prefix else top_k
        scores, indices = self.faiss_index.search(query_vec, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if len(results) >= top_k:
                break
            if idx < 0 or idx >= len(self.faiss_texts):
                continue
            code = self.faiss_codes[idx] if idx < len(self.faiss_codes) else ""
            if prefix and code and not code.startswith(prefix):
                continue
            results.append((code, self.faiss_texts[idx], float(score), "faiss"))
        
        return results
    
    def search_hybrid(self, query: str, top_k: int = 8, prefix: str = "") -> List[Tuple[str, str, float, str]]:
        """混合检索 (RRF融合)"""
        cache_key = f"{query[:60]}:{top_k}:{prefix}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        bm25_res = self.search_bm25(query, top_k * 2, prefix)
        faiss_res = self.search_faiss(query, top_k * 2, prefix)
        
        # RRF融合
        k = 60
        doc_scores = {}
        
        for i, (code, text, score, src) in enumerate(bm25_res):
            key = (text[:80], code)
            rrf = 0.4 / (k + i + 1)
            if key not in doc_scores:
                doc_scores[key] = {'data': (code, text, 0, src), 'score': 0}
            doc_scores[key]['score'] += rrf
        
        for i, (code, text, score, src) in enumerate(faiss_res):
            key = (text[:80], code)
            rrf = 0.6 / (k + i + 1)
            if key not in doc_scores:
                doc_scores[key] = {'data': (code, text, 0, src), 'score': 0}
            doc_scores[key]['score'] += rrf
        
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
        results = []
        for d in sorted_docs[:top_k]:
            code, text, _, src = d['data']
            results.append((code, text, d['score'], src))
        
        self.cache[cache_key] = results
        return results
    
    def get_fewshot(self, code: str, max_examples: int = 3) -> str:
        """获取精确匹配few-shot"""
        if code not in self.fewshot_db:
            return ""
        examples = self.fewshot_db[code][:max_examples]
        return "\n".join([f"示例: {ex[:200]}" for ex in examples])
    
    def get_fewshot_by_prefix(self, prefix: str, query: str, max_examples: int = 3) -> str:
        """获取前缀匹配few-shot"""
        matching = [(c, exs) for c, exs in self.fewshot_db.items() if c.startswith(prefix)]
        if not matching:
            return ""
        
        # 简单取前几个
        results = []
        for code, exs in matching[:max_examples]:
            if exs:
                results.append(f"[{code}] {exs[0][:150]}")
        return "\n".join(results)
    
    def get_candidates(self, prefix: str) -> List[str]:
        """获取下一级候选"""
        pl = len(prefix)
        if pl == 0:
            return list(self.hierarchy.get('L2', {}).keys())
        elif pl == 2:
            return self.hierarchy.get('L2', {}).get(prefix, [])
        elif pl == 4:
            return self.hierarchy.get('L4', {}).get(prefix, [])
        elif pl == 6:
            return self.hierarchy.get('L6', {}).get(prefix, [])
        elif pl == 8:
            return self.hierarchy.get('L8', {}).get(prefix, [])
        return []
    
    def get_context_for_llm(self, query: str, prefix: str = "", top_k: int = 4) -> str:
        """生成LLM上下文"""
        results = self.search_hybrid(query, top_k, prefix)
        if not results:
            return ""
        
        lines = []
        for i, (code, text, score, src) in enumerate(results):
            snippet = text[:120].replace('\n', ' ')
            lines.append(f"{i+1}. [{code}] {snippet}")
        return "\n".join(lines)



class BertPredictor:    
    def __init__(self, bert_model_name: str, data_prefix: str, classifier_ckpt: str, device: str = None):
        from transformers import BertTokenizer
        import joblib
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载编码器
        self.encoders = {}
        self.num_classes = {}
        encoder_names = {
            'A1': f'{data_prefix}_A1_encoder.pkl',
            'A2_L1': f'{data_prefix}_A2_L1_encoder.pkl',
            'A2_L2': f'{data_prefix}_A2_L2_encoder.pkl',
            'A2_L3': f'{data_prefix}_A2_L3_encoder.pkl',
            'A2_L4': f'{data_prefix}_A2_L4_encoder.pkl',
        }
        
        for name, path in encoder_names.items():
            if os.path.exists(path):
                self.encoders[name] = joblib.load(path)
                self.num_classes[name] = len(self.encoders[name].classes_)
        
        print(f"[BERT] 编码器: {list(self.num_classes.keys())}")
        
        # 加载Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # 加载模型
        from HM_BERT_Structure import HMBertClassifier
        self.model = HMBertClassifier(
            bert_model_name=bert_model_name,
            num_a1=self.num_classes.get('A1', 100),
            num_a2_l1=self.num_classes.get('A2_L1', 100),
            num_a2_l2=self.num_classes.get('A2_L2', 100),
            num_a2_l3=self.num_classes.get('A2_L3', 100),
            num_a2_l4=self.num_classes.get('A2_L4', 100)
        )
        
        # 加载权重
        ckpt = torch.load(classifier_ckpt, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('_orig_mod.', '').replace('module.', '')
            new_state[name] = v
        self.model.load_state_dict(new_state, strict=False)
        self.model.to(self.device).eval().half()
        
        self.cache = {}
        print(f"模型加载完成")
    
    def predict(self, text: str, level: int, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        预测指定层级
        level: 0->2位, 2->4位, 4->6位, 6->10位, 10->直接10位
        """
        cache_key = f"{text[:100]}:{level}:{top_k}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if level == 10 or level == 'A1':
            task_key = 'A1'
        elif level == 0:
            task_key = 'A2_L1'
        elif level == 2:
            task_key = 'A2_L2'
        elif level == 4:
            task_key = 'A2_L3'
        elif level == 6:
            task_key = 'A2_L4'
        else:
            task_key = 'A1'
        
        if task_key not in self.encoders:
            return []
        
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                max_length=256, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        idx_map = {'A1': 0, 'A2_L1': 1, 'A2_L2': 2, 'A2_L3': 3, 'A2_L4': 4}
        logits = outputs[idx_map[task_key]]
        probs = torch.softmax(logits, dim=-1)[0]
        
        k = min(top_k, probs.numel())
        top_probs, top_indices = torch.topk(probs, k)
        codes = self.encoders[task_key].inverse_transform(top_indices.cpu().numpy())
        
        results = [(str(c), float(p)) for c, p in zip(codes, top_probs.cpu().tolist())]
        self.cache[cache_key] = results
        return results
    
    def get_uncertainty(self, text: str) -> float:
        """计算不确定性"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                max_length=256, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs[1], dim=-1)[0]
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = np.log(probs.numel())
        
        return min(entropy / max_entropy, 1.0)


class LLMInterface:    
    def __init__(self, tokenizer, model, retriever: UnifiedRetriever):
        self.tokenizer = tokenizer
        self.model = model
        self.retriever = retriever
        self.call_count = 0
    
    def generate(self, prompt: str, max_tokens: int = 64) -> str:
        """生成回复"""
        self.call_count += 1
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = outputs[0][len(input_ids[0]):]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
    
    def action_A4_arbitrate(self, desc: str, candidates: List[Tuple[str, float]]) -> str:
        """A4: 仲裁冲突候选"""
        # 获取上下文
        context = self.retriever.get_context_for_llm(desc, prefix=candidates[0][0][:4] if candidates else "")
        
        # 候选few-shot
        cand_fs = []
        for c, _ in candidates[:3]:
            fs = self.retriever.get_fewshot(c, 2)
            if fs:
                cand_fs.append(f"[{c}]:\n{fs}")
        cand_fs_text = "\n".join(cand_fs)
        
        cands = "\n".join([f"{i+1}. {c[0]} (置信度:{c[1]:.2f})" for i, c in enumerate(candidates[:5])])
        
        prompt = f"""你是海关商品归类专家。从候选HSCode中选择最合适的10位编码。

商品描述:
{desc[:350]}

检索到的相似案例:
{context}

候选码历史归类:
{cand_fs_text}

候选编码:
{cands}

要求: 只输出10位数字HSCode，不要解释。
答案:"""
        
        result = self.generate(prompt, max_tokens=20)
        codes = re.findall(r'\d{10}', result)
        if codes:
            return codes[0]
        return candidates[0][0] if candidates else ""
    
    def action_A5_choose(self, desc: str, prefix: str, candidates: List[str], target_len: int) -> str:
        """A5: LLM选择下一级编码"""
        if not candidates:
            return ""
        
        # 过滤并去重
        candidates = list(set(c[:target_len] for c in candidates if len(c) >= target_len and c.startswith(prefix)))[:12]
        if not candidates:
            return ""
        
        # few-shot
        fs = self.retriever.get_fewshot_by_prefix(prefix, desc, 3)
        
        cand_lines = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
        
        prompt = f"""你是海关商品归类专家。选择最合理的{target_len}位HSCode编码。

商品描述:
{desc[:300]}

当前前缀: {prefix}

相似案例:
{fs}

候选:
{cand_lines}

要求: 只输出{target_len}位数字，不要解释。
答案:"""
        
        result = self.generate(prompt, max_tokens=16)
        m = re.findall(r'\d+', result)
        if m:
            code = m[0][:target_len]
            if len(code) == target_len and code.startswith(prefix):
                return code
        return ""
    
    def action_A6_verify(self, desc: str, code: str) -> float:
        """A6: 验证并计算奖励"""
        # 检查few-shot匹配
        fs = self.retriever.get_fewshot(code, 2)
        if fs:
            return 0.92
        
        # 检查前缀匹配
        similar = self.retriever.search_hybrid(desc, top_k=5, prefix=code[:4])
        for c, text, score, src in similar:
            if c == code:
                return 0.95
            if c[:6] == code[:6]:
                return 0.75
            if c[:4] == code[:4]:
                return 0.55
        
        return 0.35


class MCTSNode:
    """MCTS节点"""
    _id_counter = 0
    
    def __init__(self, parent, depth: int, hs_prefix: str, confidence: float,
                 action_type: str = "A3", **kwargs):
        MCTSNode._id_counter += 1
        self.id = MCTSNode._id_counter
        
        self.parent = parent
        self.depth = depth
        self.hs_prefix = hs_prefix
        self.confidence = confidence
        self.action_type = action_type
        self.children = []
        
        if parent is None:
            self.predictor = kwargs.get('predictor')
            self.llm = kwargs.get('llm')
            self.retriever = kwargs.get('retriever')
            self.user_question = kwargs.get('user_question')
            self.max_depth = kwargs.get('max_depth', 6)
            self.uncertainty = kwargs.get('uncertainty', 0.5)
        else:
            self.predictor = parent.predictor
            self.llm = parent.llm
            self.retriever = parent.retriever
            self.user_question = parent.user_question
            self.max_depth = parent.max_depth
            self.uncertainty = parent.uncertainty
    
    def is_terminal(self) -> bool:
        if self.hs_prefix == "FAIL":
            return True
        if len(self.hs_prefix) >= 10:
            return True
        if self.depth >= self.max_depth:
            return True
        return False
    
    def get_action_trace(self) -> List[str]:
        """获取动作链"""
        trace = []
        node = self
        while node is not None and node.parent is not None:
            trace.append(node.action_type)
            node = node.parent
        trace.reverse()
        return trace
    
    def calculate_reward(self) -> float:
        if self.hs_prefix == "FAIL":
            return -1.0
        if not self.is_terminal():
            return 0.0
        return self.llm.action_A6_verify(self.user_question, self.hs_prefix)
    
    def expand(self) -> List['MCTSNode']:
        """扩展子节点"""
        if self.children:
            return self.children
        
        if self.is_terminal():
            return []
        
        cur_len = len(self.hs_prefix)
        
        if self.depth == 0:
            # A2: 直接预测10位 (降低阈值确保有结果)
            direct_preds = self.predictor.predict(self.user_question, 10, top_k=5)
            for code, conf in direct_preds:
                if len(code) >= 10 and conf > 0.15:
                    self.children.append(MCTSNode(
                        self, 1, code[:10], conf, action_type="A2_Direct"
                    ))
            
            # A3: 分层预测2位 (始终添加，确保有层级路径)
            layer1_preds = self.predictor.predict(self.user_question, 0, top_k=8)
            added_prefixes = set()
            for code, conf in layer1_preds:
                if len(code) >= 2:
                    p2 = code[:2]
                    if p2 not in added_prefixes:
                        added_prefixes.add(p2)
                        self.children.append(MCTSNode(
                            self, 1, p2, conf, action_type="A3_L2"
                        ))
            
            # 从RAG检索补充更多2位前缀
            similar = self.retriever.search_hybrid(self.user_question, top_k=8)
            for code, text, score, src in similar:
                if len(code) >= 2:
                    p2 = code[:2]
                    if p2 not in added_prefixes:
                        added_prefixes.add(p2)
                        self.children.append(MCTSNode(
                            self, 1, p2, 0.35, action_type="A5_RAG_L2"
                        ))
            
            return self.children
        
        
        # 2位/4位 -> 下一级
        if cur_len in [2, 4]:
            bert_preds = self.predictor.predict(self.user_question, cur_len, top_k=12)
            filtered = [(c, s) for c, s in bert_preds if c.startswith(self.hs_prefix)]
            
            next_len = cur_len + 2
            added_prefixes = set()
            
            # A3: BERT预测 (先添加)
            for code, conf in filtered[:8]:
                child_code = code[:next_len] if len(code) >= next_len else code
                if child_code not in added_prefixes:
                    added_prefixes.add(child_code)
                    self.children.append(MCTSNode(
                        self, self.depth + 1, child_code, conf, action_type=f"A3_L{next_len}"
                    ))
            
            # 从层级结构补充候选 (重要: 确保覆盖更多可能性)
            hier_candidates = self.retriever.get_candidates(self.hs_prefix)
            for hc in hier_candidates[:15]:
                if hc not in added_prefixes:
                    added_prefixes.add(hc)
                    self.children.append(MCTSNode(
                        self, self.depth + 1, hc, 0.2, action_type=f"A3_L{next_len}_Hier"
                    ))
            
            # A5 RAG: 从检索结果补充
            similar = self.retriever.search_hybrid(self.user_question, top_k=6, prefix=self.hs_prefix)
            for code, text, score, src in similar:
                if len(code) >= next_len:
                    child_code = code[:next_len]
                    if child_code not in added_prefixes:
                        added_prefixes.add(child_code)
                        self.children.append(MCTSNode(
                            self, self.depth + 1, child_code, 0.4, action_type="A5_RAG"
                        ))
            
            # A4仲裁: 品目冲突 (低置信度时LLM介入)
            if cur_len == 2 and len(filtered) >= 2:
                if filtered[0][1] < 0.5 or (filtered[0][1] - filtered[1][1] < 0.1):
                    best = self.llm.action_A4_arbitrate(self.user_question, filtered[:5])
                    if best and len(best) >= next_len:
                        child_code = best[:next_len]
                        if child_code not in added_prefixes:
                            self.children.append(MCTSNode(
                                self, self.depth + 1, child_code, 0.65, action_type="A4_Arb"
                            ))
        
        # 6位 -> 8位
        elif cur_len == 6:
            bert_preds = self.predictor.predict(self.user_question, 6, top_k=12)
            filtered = [(c, s) for c, s in bert_preds if c.startswith(self.hs_prefix) and len(c) >= 8]
            
            # 聚合8位
            p8_conf = {}
            for code, conf in filtered:
                p8 = code[:8]
                p8_conf[p8] = max(p8_conf.get(p8, 0), conf)
            
            for p8, conf in sorted(p8_conf.items(), key=lambda x: x[1], reverse=True)[:6]:
                self.children.append(MCTSNode(
                    self, self.depth + 1, p8, conf, action_type="A3_L8"
                ))
            
            # 层级补充
            candidates8 = self.retriever.get_candidates(self.hs_prefix)
            for c8 in candidates8[:6]:
                if c8 not in [c.hs_prefix for c in self.children]:
                    self.children.append(MCTSNode(
                        self, self.depth + 1, c8, 0.22, action_type="A3_L8"
                    ))
            
            if not self.children:
                choice = self.llm.action_A5_choose(
                    self.user_question, self.hs_prefix, candidates8, 8
                )
                if choice:
                    self.children.append(MCTSNode(
                        self, self.depth + 1, choice, 0.4, action_type="A5_LLM_L8"
                    ))
        
        # 8位 -> 10位
        elif cur_len == 8:
            bert_preds = self.predictor.predict(self.user_question, 6, top_k=15)
            filtered10 = [(c, s) for c, s in bert_preds if c.startswith(self.hs_prefix) and len(c) >= 10]
            
            for code, conf in filtered10[:6]:
                self.children.append(MCTSNode(
                    self, self.depth + 1, code[:10], conf, action_type="A3_L10"
                ))
            
            # 层级补充
            candidates10 = self.retriever.get_candidates(self.hs_prefix)
            for c10 in candidates10[:6]:
                if c10 not in [c.hs_prefix for c in self.children]:
                    self.children.append(MCTSNode(
                        self, self.depth + 1, c10, 0.2, action_type="A3_L10"
                    ))
            
            # RAG补充
            if not self.children:
                similar = self.retriever.search_hybrid(self.user_question, top_k=6, prefix=self.hs_prefix)
                for code, text, score, src in similar:
                    if len(code) >= 10:
                        self.children.append(MCTSNode(
                            self, self.depth + 1, code[:10], 0.3, action_type="A5_RAG"
                        ))
            
            # LLM兜底
            if not self.children:
                choice = self.llm.action_A5_choose(
                    self.user_question, self.hs_prefix, candidates10, 10
                )
                if choice:
                    self.children.append(MCTSNode(
                        self, self.depth + 1, choice, 0.35, action_type="A5_LLM_L10"
                    ))
        
        # 无法扩展时的兜底策略
        if not self.children:
            # 尝试从RAG检索任何相关结果
            similar = self.retriever.search_hybrid(self.user_question, top_k=8, prefix=self.hs_prefix)
            for code, text, score, src in similar:
                if len(code) >= cur_len + 2:
                    child_code = code[:cur_len + 2]
                    if child_code.startswith(self.hs_prefix):
                        self.children.append(MCTSNode(
                            self, self.depth + 1, child_code, 0.25,
                            action_type=f"A5_RAG_L{cur_len + 2}"
                        ))
                        break
            
            # 如果还是没有，尝试直接到10位
            if not self.children and cur_len >= 6:
                direct = self.predictor.predict(self.user_question, 10, top_k=5)
                for code, conf in direct:
                    if code.startswith(self.hs_prefix) and len(code) >= 10:
                        self.children.append(MCTSNode(
                            self, self.depth + 1, code[:10], conf * 0.7,
                            action_type="A2_Direct_Fallback"
                        ))
                        break
            
            # 最终兜底
            if not self.children:
                self.children.append(MCTSNode(
                    self, self.depth + 1, "FAIL", 0.0, action_type="FAIL"
                ))
        
        return self.children


class MCTSSearcher:
    """MCTS搜索器 - UP-Q + 动态PUCT"""
    
    def __init__(self, exploration_weight: float = 1.0,
                 alpha_threshold: float = 0.9, beta: float = 0.5):
        self.Q: Dict[int, float] = defaultdict(float)
        self.N: Dict[int, int] = defaultdict(int)
        self.children: Dict[int, List[MCTSNode]] = {}
        self.explored = set()
        
        self.c = exploration_weight
        self.alpha = alpha_threshold
        self.beta = beta
    
    def search(self, root: MCTSNode, num_rollouts: int, verbose: bool = False):
        """执行MCTS搜索"""
        for i in range(num_rollouts):
            path = self._select(root)
            leaf = path[-1]
            
            if leaf.id not in self.explored:
                children = leaf.expand()
                self.children[leaf.id] = children
            
            self._backpropagate(path)
    
    def _select(self, node: MCTSNode) -> List[MCTSNode]:
        """选择路径"""
        path = [node]
        
        while True:
            if node.id not in self.children:
                return path
            
            children = self.children[node.id]
            if not children:
                return path
            
            # 优先未访问
            unexplored = [c for c in children if c.id not in self.explored]
            if unexplored:
                unexplored.sort(key=lambda x: x.confidence, reverse=True)
                node = unexplored[0]
                path.append(node)
                return path
            
            if node.is_terminal():
                return path
            
            node = self._uct_select(node)
            path.append(node)
        
        return path
    
    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        """UCT选择"""
        children = self.children.get(node.id, [])
        if not children:
            return node
        
        best_child = None
        best_score = -float('inf')
        
        uncertainty = node.uncertainty
        c_dynamic = self.c * (1.0 + 1.5 * uncertainty)
        
        for child in children:
            if self.N[child.id] == 0:
                return child
            
            exploitation = self.Q[child.id]
            exploration = c_dynamic * child.confidence * \
                         (math.sqrt(self.N[node.id]) / (1 + self.N[child.id]))
            
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child or children[0]
    
    def _backpropagate(self, path: List[MCTSNode]):
        """UP-Q反向传播"""
        if not path:
            return
        
        leaf = path[-1]
        if leaf.depth == 0:
            return
        
        v = leaf.calculate_reward()
        
        for node in reversed(path):
            self.N[node.id] += 1
            self.explored.add(node.id)
            
            if self.N[node.id] == 1:
                self.Q[node.id] = v
            else:
                q_old = self.Q[node.id]
                
                if v >= self.alpha:
                    self.Q[node.id] = v
                elif v > q_old:
                    self.Q[node.id] = q_old + self.beta * (v - q_old)
                else:
                    self.Q[node.id] = (q_old * (self.N[node.id] - 1) + v) / self.N[node.id]
            
            v *= 0.99
    
    def get_best_solutions(self, root: MCTSNode, top_k: int = 5) -> List[Dict]:
        solutions = []
        partial_solutions = []  
        seen_hs = set()  
        
        def collect(node: MCTSNode):
            if node.is_terminal() and node.hs_prefix != "FAIL":
                if len(node.hs_prefix) >= 10:
                    hs_code = node.hs_prefix[:10]
                    if hs_code not in seen_hs:
                        seen_hs.add(hs_code)
                        action_trace = node.get_action_trace()
                        solutions.append({
                            'hs': hs_code,
                            'reward': round(self.Q[node.id], 4),
                            'confidence': round(node.confidence, 4),
                            'visits': self.N[node.id],
                            'action': ">".join(action_trace) if action_trace else node.action_type,
                            'depth': node.depth,
                            'source': 'mcts'
                        })
                elif len(node.hs_prefix) >= 6:
                    partial_solutions.append({
                        'prefix': node.hs_prefix,
                        'reward': self.Q[node.id],
                        'node': node
                    })
            
            for child in self.children.get(node.id, []):
                collect(child)
        
        collect(root)
        
        if not solutions and partial_solutions:
            partial_solutions.sort(key=lambda x: x['reward'], reverse=True)
            for partial in partial_solutions[:5]:
                node = partial['node']
                prefix = partial['prefix']
                candidates = node.retriever.get_candidates(prefix)
                if candidates:
                    for c in candidates[:3]:
                        if len(c) >= 10:
                            solutions.append({
                                'hs': c[:10],
                                'reward': round(partial['reward'] * 0.8, 4),
                                'confidence': 0.3,
                                'visits': 1,
                                'action': f"{node.action_type}>Fallback_L10",
                                'depth': node.depth + 1
                            })
        
        if not solutions:
            for child in self.children.get(root.id, []):
                if child.action_type == 'A2_Direct' and len(child.hs_prefix) >= 10:
                    solutions.append({
                        'hs': child.hs_prefix[:10],
                        'reward': round(self.Q[child.id] if self.Q[child.id] > 0 else 0.3, 4),
                        'confidence': round(child.confidence, 4),
                        'visits': max(1, self.N[child.id]),
                        'action': 'A2_Direct_Fallback',
                        'depth': child.depth
                    })
            
            # 从检索结果中获取
            if not solutions and hasattr(root, 'retriever') and hasattr(root, 'user_question'):
                similar = root.retriever.search_hybrid(root.user_question, top_k=5)
                for code, text, score, src in similar:
                    if len(code) >= 10:
                        solutions.append({
                            'hs': code[:10],
                            'reward': 0.25,
                            'confidence': 0.2,
                            'visits': 1,
                            'action': 'RAG_Fallback',
                            'depth': 1
                        })
                        break
        
        solutions.sort(key=lambda x: x['reward'], reverse=True)
        return solutions[:top_k]


def run_mcts_inference(
    problem: str,
    predictor: BertPredictor,
    llm: LLMInterface,
    retriever: UnifiedRetriever,
    num_rollouts: int = 8,
    max_depth: int = 6,
    verbose: bool = False
) -> List[Dict]:
    """执行MCTS推理"""
    
    # 计算不确定性
    uncertainty = predictor.get_uncertainty(problem)
    
    # 创建根节点
    root = MCTSNode(
        parent=None,
        depth=0,
        hs_prefix="",
        confidence=1.0,
        action_type="ROOT",
        predictor=predictor,
        llm=llm,
        retriever=retriever,
        user_question=problem,
        max_depth=max_depth,
        uncertainty=uncertainty
    )
    
    # MCTS搜索
    searcher = MCTSSearcher(exploration_weight=1.0)
    searcher.search(root, num_rollouts, verbose)
    
    # 获取结果
    solutions = searcher.get_best_solutions(root, top_k=10)
    
    return solutions



if __name__ == "__main__":
    print("MCTS-RAG HSCode分类系统 (优化版V2)")
    print("请使用 run_optimized_v2.py 运行完整推理")
