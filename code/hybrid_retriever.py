
import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from functools import lru_cache

# 延迟导入
jieba = None
faiss = None
SentenceTransformer = None


def _ensure_jieba():
    global jieba
    if jieba is None:
        import jieba as _jieba
        jieba = _jieba


@dataclass
class RetrievalResult:
    """检索结果"""
    text: str
    code: str
    score: float
    source: str
    rank: int = 0


class HybridRetriever:
    
    def __init__(self, db_path: str, 
                 embedding_model_path: str = None,
                 use_reranker: bool = False,
                 reranker_path: str = None):
        """
        Args:
            db_path: RAG_db目录
            embedding_model_path: Embedding模型路径
            use_reranker: 是否使用重排序
            reranker_path: 重排序模型路径
        """
        self.db_path = db_path
        self.embedding_model_path = embedding_model_path
        
        # BM25索引
        self.bm25 = None
        self.bm25_texts = []
        self.bm25_codes = []
        self.bm25_sources = []
        self.bm25_docs = []
        
        # FAISS索引
        self.faiss_index = None
        self.faiss_meta = None
        self.embed_model = None
        
        # Few-shot库
        self.fewshot_db = {}
        
        # 层级结构
        self.hierarchy = {}
        
        # 重排序器
        self.reranker = None
        self.use_reranker = use_reranker
        
        # 缓存
        self._cache = {}
        self._cache_hits = 0
        
        # 加载索引
        self._load_indices(reranker_path)
    
    def _load_indices(self, reranker_path: str = None):
        """加载所有索引"""
        _ensure_jieba()
        
        unified_bm25_path = os.path.join(self.db_path, "unified_bm25.pkl")
        if os.path.exists(unified_bm25_path):
            with open(unified_bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data.get('bm25')
                self.bm25_texts = data.get('texts', [])
                self.bm25_codes = data.get('codes', [])
                self.bm25_sources = data.get('sources', [])
                self.bm25_docs = data.get('docs', [])
            print(f"[HybridRetriever] ✅ 统一BM25索引: {len(self.bm25_texts)} docs")
        else:
            # 回退到旧索引
            old_bm25_path = os.path.join(self.db_path, "heading_bm25.pkl")
            if os.path.exists(old_bm25_path):
                with open(old_bm25_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data.get('bm25')
                    self.bm25_texts = data.get('texts', [])
                    self.bm25_codes = data.get('codes', [])
                    self.bm25_sources = ['legacy'] * len(self.bm25_texts)
                print(f"[HybridRetriever] ✅ 旧BM25索引: {len(self.bm25_texts)} docs")
        
        faiss_path = os.path.join(self.db_path, "unified_faiss.index")
        faiss_meta_path = os.path.join(self.db_path, "unified_faiss_meta.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(faiss_meta_path):
            try:
                global faiss, SentenceTransformer
                import faiss as _faiss
                from sentence_transformers import SentenceTransformer as _ST
                faiss = _faiss
                SentenceTransformer = _ST
                
                self.faiss_index = faiss.read_index(faiss_path)
                with open(faiss_meta_path, 'rb') as f:
                    self.faiss_meta = pickle.load(f)
                
                if self.embedding_model_path and os.path.exists(self.embedding_model_path):
                    self.embed_model = SentenceTransformer(
                        self.embedding_model_path, 
                        device='cuda'
                    )
                    print(f"[HybridRetriever] ✅ FAISS索引: {self.faiss_index.ntotal} vectors")
            except Exception as e:
                print(f"[HybridRetriever] ⚠️ FAISS加载失败: {e}")
        
        fewshot_path = os.path.join(self.db_path, "fewshot_examples.json")
        if os.path.exists(fewshot_path):
            with open(fewshot_path, 'r', encoding='utf-8') as f:
                self.fewshot_db = json.load(f)
            print(f"[HybridRetriever] ✅ Few-shot库: {len(self.fewshot_db)} codes")
        
        hier_path = os.path.join(self.db_path, "hierarchy_enhanced.pkl")
        if os.path.exists(hier_path):
            with open(hier_path, 'rb') as f:
                self.hierarchy = pickle.load(f)
            print(f"[HybridRetriever] ✅ 层级结构已加载")
        
        if self.use_reranker:
            self._load_reranker(reranker_path)
    
    def _load_reranker(self, reranker_path: str = None):
        """加载重排序模型"""
        paths_to_try = [
            reranker_path,
            os.path.join(self.db_path, '..', 'models', 'bge-reranker-v2-m3'),
            '/root/data/A/models/bge-reranker-v2-m3',
        ]
        
        for path in paths_to_try:
            if path and os.path.exists(path):
                try:
                    from FlagEmbedding import FlagReranker
                    self.reranker = FlagReranker(path, use_fp16=True)
                    print(f"[HybridRetriever] ✅ 重排序器: {path}")
                    return
                except Exception as e:
                    print(f"[HybridRetriever] ⚠️ 重排序器加载失败: {e}")
        
        print("[HybridRetriever] ⚠️ 未找到重排序模型，跳过")
        self.use_reranker = False
    
    def search_bm25(self, query: str, top_k: int = 20, 
                    prefix: str = "") -> List[RetrievalResult]:
        """BM25检索"""
        if self.bm25 is None:
            return []
        
        _ensure_jieba()
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        
        indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in indices:
            if len(results) >= top_k:
                break
            if scores[idx] <= 0:
                continue
            
            code = self.bm25_codes[idx] if idx < len(self.bm25_codes) else ""
            
            # 前缀过滤
            if prefix and code and not code.startswith(prefix):
                continue
            
            results.append(RetrievalResult(
                text=self.bm25_texts[idx],
                code=code,
                score=float(scores[idx]),
                source=self.bm25_sources[idx] if idx < len(self.bm25_sources) else 'bm25',
                rank=len(results)
            ))
        
        return results
    
    def search_faiss(self, query: str, top_k: int = 20,
                     prefix: str = "") -> List[RetrievalResult]:
        """向量检索"""
        if self.faiss_index is None or self.embed_model is None:
            return []
        
        # 编码查询
        query_vec = self.embed_model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec).astype('float32')
        
        # 检索更多以便过滤
        search_k = top_k * 3 if prefix else top_k
        scores, indices = self.faiss_index.search(query_vec, search_k)
        
        results = []
        texts = self.faiss_meta.get('texts', [])
        codes = self.faiss_meta.get('codes', [])
        sources = self.faiss_meta.get('sources', [])
        
        for score, idx in zip(scores[0], indices[0]):
            if len(results) >= top_k:
                break
            if idx < 0 or idx >= len(texts):
                continue
            
            code = codes[idx] if idx < len(codes) else ""
            
            # 前缀过滤
            if prefix and code and not code.startswith(prefix):
                continue
            
            results.append(RetrievalResult(
                text=texts[idx],
                code=code,
                score=float(score),
                source=sources[idx] if idx < len(sources) else 'faiss',
                rank=len(results)
            ))
        
        return results
    
    def rerank(self, query: str, results: List[RetrievalResult], 
               top_k: int = 10) -> List[RetrievalResult]:
        """重排序"""
        if not self.reranker or not results:
            return results[:top_k]
        
        pairs = [[query, r.text] for r in results]
        
        try:
            scores = self.reranker.compute_score(pairs)
            if isinstance(scores, (int, float)):
                scores = [scores]
            
            # 按分数排序
            sorted_pairs = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
            
            reranked = []
            for i, (r, score) in enumerate(sorted_pairs[:top_k]):
                r.score = float(score)
                r.rank = i
                reranked.append(r)
            
            return reranked
        except Exception as e:
            print(f"[HybridRetriever] ⚠️ 重排序失败: {e}")
            return results[:top_k]
    
    def search_hybrid(self, query: str, top_k: int = 10,
                      prefix: str = "", use_rerank: bool = True,
                      bm25_weight: float = 0.4,
                      faiss_weight: float = 0.6) -> List[RetrievalResult]:
        """
        混合检索 (BM25 + FAISS + 可选重排序)
        
        融合策略: Reciprocal Rank Fusion (RRF)
        """
        cache_key = f"hybrid:{query[:80]}:{top_k}:{prefix}"
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        # 召回阶段
        recall_k = top_k * 3
        bm25_results = self.search_bm25(query, recall_k, prefix)
        faiss_results = self.search_faiss(query, recall_k, prefix)
        
        # RRF融合
        k = 60  # RRF参数
        doc_scores = {}
        
        for i, r in enumerate(bm25_results):
            key = (r.text[:100], r.code)
            rrf_score = bm25_weight / (k + i + 1)
            if key not in doc_scores:
                doc_scores[key] = {'result': r, 'score': 0}
            doc_scores[key]['score'] += rrf_score
        
        for i, r in enumerate(faiss_results):
            key = (r.text[:100], r.code)
            rrf_score = faiss_weight / (k + i + 1)
            if key not in doc_scores:
                doc_scores[key] = {'result': r, 'score': 0}
            doc_scores[key]['score'] += rrf_score
        
        # 按融合分数排序
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
        merged = [d['result'] for d in sorted_docs[:recall_k]]
        
        # 更新分数和排名
        for i, r in enumerate(merged):
            r.score = sorted_docs[i]['score']
            r.rank = i
        
        # 重排序阶段
        if use_rerank and self.reranker:
            final = self.rerank(query, merged, top_k)
        else:
            final = merged[:top_k]
        
        self._cache[cache_key] = final
        return final
    
    def get_fewshot(self, code: str, max_examples: int = 3) -> str:
        """获取精确匹配的few-shot示例"""
        if code not in self.fewshot_db:
            return ""
        
        examples = self.fewshot_db[code][:max_examples]
        formatted = []
        for i, ex in enumerate(examples):
            # 清理并截断
            clean = ex[:250].replace('\n', ' ')
            formatted.append(f"示例{i+1}: {clean}")
        
        return "\n".join(formatted)
    
    def get_fewshot_for_prefix(self, prefix: str, query: str, 
                                max_examples: int = 3) -> str:
        """获取前缀匹配的few-shot示例（带相似度排序）"""
        # 找所有匹配前缀的codes
        matching_codes = [c for c in self.fewshot_db.keys() if c.startswith(prefix)]
        
        if not matching_codes:
            return ""
        
        # 用BM25找最相似的
        _ensure_jieba()
        query_tokens = set(jieba.cut(query))
        
        scored = []
        for code in matching_codes:
            examples = self.fewshot_db[code]
            for ex in examples[:2]:
                ex_tokens = set(jieba.cut(ex[:200]))
                overlap = len(query_tokens & ex_tokens)
                scored.append((code, ex, overlap))
        
        scored.sort(key=lambda x: x[2], reverse=True)
        
        formatted = []
        seen = set()
        for code, ex, _ in scored:
            if len(formatted) >= max_examples:
                break
            if code in seen:
                continue
            seen.add(code)
            clean = ex[:200].replace('\n', ' ')
            formatted.append(f"[{code}] {clean}")
        
        return "\n".join(formatted)
    
    def get_candidates(self, prefix: str) -> List[str]:
        """获取下一级候选编码"""
        prefix_len = len(prefix)
        
        if prefix_len == 0:
            return list(self.hierarchy.get('L2', {}).keys())
        elif prefix_len == 2:
            return self.hierarchy.get('L2', {}).get(prefix, [])
        elif prefix_len == 4:
            return self.hierarchy.get('L4', {}).get(prefix, [])
        elif prefix_len == 6:
            return self.hierarchy.get('L6', {}).get(prefix, [])
        elif prefix_len == 8:
            return self.hierarchy.get('L8', {}).get(prefix, [])
        
        return []
    
    def get_context_for_llm(self, query: str, prefix: str = "",
                            max_cases: int = 3, max_rules: int = 2) -> str:
        """
        为LLM生成上下文（few-shot案例 + PDF规则）
        """
        parts = []
        
        # 1. 相似案例
        results = self.search_hybrid(query, top_k=max_cases + max_rules, 
                                     prefix=prefix, use_rerank=True)
        
        cases = [r for r in results if r.source in ('train_exact', 'train_prefix', 'test_self')]
        rules = [r for r in results if r.source == 'pdf']
        
        if cases:
            case_texts = []
            for i, r in enumerate(cases[:max_cases]):
                case_texts.append(f"案例{i+1} [{r.code}]: {r.text[:150]}")
            parts.append("相似归类案例:\n" + "\n".join(case_texts))
        
        if rules:
            rule_texts = []
            for i, r in enumerate(rules[:max_rules]):
                rule_texts.append(f"规则{i+1}: {r.text[:150]}")
            parts.append("相关规则:\n" + "\n".join(rule_texts))
        
        return "\n\n".join(parts)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'bm25_docs': len(self.bm25_texts),
            'faiss_vectors': self.faiss_index.ntotal if self.faiss_index else 0,
            'fewshot_codes': len(self.fewshot_db),
            'hierarchy_chapters': len(self.hierarchy.get('L2', {})),
            'cache_size': len(self._cache),
            'cache_hits': self._cache_hits,
            'has_reranker': self.reranker is not None
        }


# 测试
if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), '..', 'RAG_db')
    embed_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'bge-base-zh-v1.5')
    
    retriever = HybridRetriever(
        db_path=db_path,
        embedding_model_path=embed_path,
        use_reranker=False
    )
    
    print("\n统计信息:")
    print(retriever.get_stats())
    
    query = "不锈钢螺丝"
    print(f"\n查询: {query}")
    
    results = retriever.search_hybrid(query, top_k=5)
    for r in results:
        print(f"  [{r.source}] {r.code}: {r.text[:50]}... (score={r.score:.4f})")
