
import os
import json
import pickle
import jieba
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """检索结果"""
    text: str
    code: str
    score: float
    source: str  # history/pdf


class RAGRetriever:

    def __init__(self, 
                 db_path: str,
                 embedding_model_path: str = None,
                 use_reranker: bool = True):
        """
        Args:
            db_path: RAG_db目录路径
            embedding_model_path: embedding模型路径
            use_reranker: 是否使用重排序
        """
        self.db_path = db_path
        self.use_reranker = use_reranker
        
        # 历史案例索引
        self.history_bm25 = None
        self.history_texts = []
        self.history_codes = []
        
        # PDF规则索引（如果存在）
        self.pdf_bm25 = None
        self.pdf_texts = []
        
        # 向量索引
        self.faiss_index = None
        self.embed_model = None
        
        # 层级结构
        self.hierarchy = None
        self.code_examples = None
        
        # 重排序模型
        self.reranker = None
        
        # 缓存
        self.cache = {}
        self.cache_hits = 0
        
        # 加载索引
        self._load_indices(embedding_model_path)
    
    def _load_indices(self, embedding_model_path: str = None):
        """加载所有索引"""
        # 加载历史BM25
        bm25_path = os.path.join(self.db_path, "history_bm25.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.history_bm25 = data['bm25']
                self.history_texts = data['texts']
                self.history_codes = data['codes']
            print(f"[RAG] 历史BM25: {len(self.history_texts)}条")
        
        # 加载层级结构
        hier_path = os.path.join(self.db_path, "hierarchy.pkl")
        if os.path.exists(hier_path):
            with open(hier_path, 'rb') as f:
                self.hierarchy = pickle.load(f)
            print(f"[RAG] 层级结构已加载")
        
        # 加载代码示例
        examples_path = os.path.join(self.db_path, "code_examples.json")
        if os.path.exists(examples_path):
            with open(examples_path, 'r', encoding='utf-8') as f:
                self.code_examples = json.load(f)
            print(f"[RAG] 代码示例: {len(self.code_examples)}个")
        
        # 加载FAISS（可选）
        faiss_path = os.path.join(self.db_path, "history.faiss")
        if os.path.exists(faiss_path) and embedding_model_path:
            try:
                import faiss
                from sentence_transformers import SentenceTransformer
                
                self.faiss_index = faiss.read_index(faiss_path)
                self.embed_model = SentenceTransformer(embedding_model_path, device='cuda')
                print(f"[RAG] FAISS索引: {self.faiss_index.ntotal}向量")
            except Exception as e:
                print(f"[RAG] ⚠️ FAISS加载失败: {e}")
        
        # 加载重排序模型（可选）
        if self.use_reranker:
            try:
                from FlagEmbedding import FlagReranker
                # 尝试多个可能的路径
                reranker_paths = [
                    os.path.join(self.db_path, '..', 'models', 'bge-reranker-v2-m3'),
                    '/root/data/A/models/bge-reranker-v2-m3',
                    '/root/models/bge-reranker-v2-m3',
                ]
                for reranker_path in reranker_paths:
                    if os.path.exists(reranker_path):
                        self.reranker = FlagReranker(reranker_path, use_fp16=True)
                        print(f"[RAG] 重排序模型已加载: {reranker_path}")
                        break
                if self.reranker is None:
                    print(f"[RAG] ⚠️ 重排序模型未找到，跳过")
            except Exception as e:
                print(f"[RAG] ⚠️ 重排序模型加载失败: {e}")
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """BM25检索"""
        if self.history_bm25 is None:
            return []
        
        tokens = list(jieba.cut(query))
        scores = self.history_bm25.get_scores(tokens)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(RetrievalResult(
                    text=self.history_texts[idx],
                    code=self.history_codes[idx],
                    score=float(scores[idx]),
                    source='history_bm25'
                ))
        
        return results
    
    def search_faiss(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """向量检索"""
        if self.faiss_index is None or self.embed_model is None:
            return []
        
        query_vec = self.embed_model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec).astype('float32')
        
        scores, indices = self.faiss_index.search(query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.history_texts):
                results.append(RetrievalResult(
                    text=self.history_texts[idx],
                    code=self.history_codes[idx],
                    score=float(score),
                    source='history_faiss'
                ))
        
        return results
    
    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        """重排序"""
        if self.reranker is None or not results:
            return results[:top_k]
        
        pairs = [[query, r.text] for r in results]
        
        try:
            scores = self.reranker.compute_score(pairs)
            if isinstance(scores, float):
                scores = [scores]
            
            # 按分数排序
            sorted_results = sorted(
                zip(results, scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return [r for r, s in sorted_results[:top_k]]
        except:
            return results[:top_k]
    
    def retrieve(self, query: str, top_k: int = 5, 
                 prefix: str = "", use_rerank: bool = True) -> List[RetrievalResult]:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            prefix: HSCode前缀过滤
            use_rerank: 是否重排序
        """
        cache_key = f"{query[:100]}:{top_k}:{prefix}"
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # BM25检索
        bm25_results = self.search_bm25(query, top_k * 3)
        
        # FAISS检索
        faiss_results = self.search_faiss(query, top_k * 3)
        
        # 合并去重
        seen = set()
        merged = []
        
        for r in bm25_results + faiss_results:
            key = r.code
            if key not in seen:
                seen.add(key)
                merged.append(r)
        
        # 前缀过滤
        if prefix:
            merged = [r for r in merged if r.code.startswith(prefix)]
        
        # 重排序
        if use_rerank and self.reranker:
            merged = self.rerank(query, merged, top_k)
        else:
            merged = merged[:top_k]
        
        self.cache[cache_key] = merged
        return merged
    
    def get_similar_cases(self, query: str, top_k: int = 3) -> str:
        """获取格式化的相似案例（用于Few-shot）"""
        results = self.retrieve(query, top_k)
        
        if not results:
            return "未找到相似案例"
        
        cases = []
        for i, r in enumerate(results):
            cases.append(f"案例{i+1}: {r.text} -> HSCode: {r.code}")
        
        return "\n".join(cases)
    
    def get_fewshot_examples(self, query: str, top_k: int = 3) -> str:
        """获取Few-shot示例（带层级路径）"""
        results = self.retrieve(query, top_k)
        
        if not results:
            return ""
        
        examples = []
        for i, r in enumerate(results):
            code = r.code
            if len(code) >= 10:
                path = f"{code[:2]} → {code[:4]} → {code[:6]} → {code}"
            else:
                path = code
            
            # 提取商品描述
            desc = r.text.split('\n')[0].replace('商品: ', '')
            
            examples.append(f"""示例{i+1}:
商品: {desc[:100]}
归类路径: {path}
HSCode: {code}""")
        
        return "\n\n".join(examples)
    
    def get_code_examples(self, code: str, max_examples: int = 2) -> List[str]:
        """获取某个HSCode的历史示例"""
        if self.code_examples is None:
            return []
        return self.code_examples.get(code, [])[:max_examples]
    
    def get_candidates_by_prefix(self, prefix: str) -> List[str]:
        """根据前缀获取下一级候选"""
        if self.hierarchy is None:
            return []
        
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
    
    def retrieve_for_A5(self, product_desc: str, prefix: str, 
                        confidence: float) -> Tuple[str, List[str]]:
        """
        A5动作专用检索
        
        Returns:
            (上下文文本, 建议的编码列表)
        """
        # 构建查询
        query = f"{product_desc} HS编码 {prefix}"
        
        results = self.retrieve(query, top_k=5, prefix=prefix)
        
        if not results:
            return "", []
        
        # 格式化上下文
        context_parts = []
        suggested_codes = []
        
        for r in results:
            context_parts.append(f"- {r.text} (HSCode: {r.code})")
            if r.code not in suggested_codes:
                suggested_codes.append(r.code)
        
        context = "相似历史案例:\n" + "\n".join(context_parts)
        
        return context, suggested_codes
    
    def retrieve_for_A6(self, product_desc: str, final_code: str) -> str:
        """
        A6动作专用检索
        检索品目注释和排他条款
        """
        code_4 = final_code[:4] if len(final_code) >= 4 else ""
        
        # 构建查询
        query = f"HS编码 {code_4} 品目注释 归类说明"
        
        results = self.retrieve(query, top_k=3, prefix=final_code[:2])
        
        if not results:
            return f"HSCode {final_code} 相关信息未找到"
        
        context_parts = [f"HSCode {final_code} 相关案例:"]
        for r in results:
            context_parts.append(f"- {r.text}")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "history_docs": len(self.history_texts),
            "cache_hits": self.cache_hits,
            "cache_size": len(self.cache),
            "has_faiss": self.faiss_index is not None,
            "has_reranker": self.reranker is not None
        }


if __name__ == "__main__":
    # 测试
    retriever = RAGRetriever(
        db_path=os.path.join(os.path.dirname(__file__), '..', 'RAG_db'),
        use_reranker=False
    )
    
    query = "不锈钢螺丝"
    print(f"\n查询: {query}")
    
    results = retriever.retrieve(query, top_k=5)
    for r in results:
        print(f"  [{r.source}] {r.code}: {r.text[:50]}...")
    
    print("\nFew-shot示例:")
    print(retriever.get_fewshot_examples(query, top_k=2))
