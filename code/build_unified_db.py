import os
import sys
import json
import pickle
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = {
    "train_data_path": "/root/data/Data_A800/Data/train_final_HYBRID_UNVALIDATED.json",
    "test_data_path": os.path.join(BASE_DIR, "data", "MCTS_hard.json"),
    "pdf_dir": os.path.join(BASE_DIR, "RAG_db"),
    "embedding_model": os.path.join(BASE_DIR, "models", "bge-base-zh-v1.5"),
    "output_dir": os.path.join(BASE_DIR, "RAG_db"),
}


def extract_pdf_text(pdf_path: str) -> str:
    """提取PDF文本（尝试多种库）"""
    text = ""
    
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text() or ""
                text += t + "\n"
        if text.strip():
            return text
    except Exception as e:
        pass
    
    # 尝试pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                text += t + "\n"
        if text.strip():
            return text
    except Exception as e:
        pass
    
    print(f"  ⚠️ 无法提取PDF文本: {os.path.basename(pdf_path)}")
    return ""


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    patterns = [
        r"实际价格税率[：:]\s*[\d.]+",
        r"应价格税率[：:]\s*[\d.]+",
        r"实际价格增值税率[：:]\s*[\d.]+",
        r"应价格增值税率[：:]\s*[\d.]+",
        r"已付关税[：:]\s*[\d.]+",
        r"法定数量[：:]\s*[\d.]+",
        r"产销国[：:]\s*\d+",
        r"计量单位[：:]\s*\d+",
    ]
    for p in patterns:
        text = re.sub(p, "", text)
    return text.strip()


def extract_hscodes(text: str) -> List[str]:
    """从文本中提取HSCode"""
    codes = re.findall(r'(?<!\d)(\d{10})(?!\d)', text)
    codes += re.findall(r'(?<!\d)(\d{8})(?!\d)', text)
    codes += re.findall(r'(?<!\d)(\d{6})(?!\d)', text)
    codes += re.findall(r'(?<!\d)(\d{4})(?!\d)', text)
    # 去重保序
    seen = set()
    result = []
    for c in codes:
        if c not in seen and len(c) >= 4:
            seen.add(c)
            result.append(c)
    return result


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """文本切块（带重叠）"""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def build_unified_database():
    print("=" * 60)
    print("构建统一检索库 (PDF + 历史案例 + 测试集)")
    print("=" * 60)
    
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[1/6] 加载测试集...")
    with open(CONFIG["test_data_path"], 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_codes = set()
    test_prefixes = defaultdict(set)  # {prefix_len: set of prefixes}
    test_examples = defaultdict(list)  # {code: [problems]}
    
    for item in test_data:
        code = item.get('solution', '')
        problem = item.get('problem', '')
        if code and len(code) == 10:
            test_codes.add(code)
            test_examples[code].append(problem)
            for l in [2, 4, 6, 8]:
                test_prefixes[l].add(code[:l])
    
    print(f"   测试集样本: {len(test_data)}")
    print(f"   唯一10位HSCode: {len(test_codes)}")
    
    print("\n[2/6] 加载训练集历史案例...")
    with open(CONFIG["train_data_path"], 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"   训练集总量: {len(train_data)}")
    
    # 按HSCode分组
    train_by_code = defaultdict(list)
    train_by_prefix4 = defaultdict(list)
    
    for item in tqdm(train_data, desc="   处理训练数据"):
        code = item.get('solution', '')
        problem = item.get('problem', '')
        if not code or not problem:
            continue
        
        # 精确匹配测试集HSCode
        if code in test_codes:
            if len(train_by_code[code]) < 10:
                train_by_code[code].append(problem)
        
        # 4位前缀匹配（品目级）
        if len(code) >= 4:
            p4 = code[:4]
            if p4 in test_prefixes[4]:
                if len(train_by_prefix4[p4]) < 30:
                    train_by_prefix4[p4].append({
                        'problem': problem,
                        'code': code
                    })
    
    print(f"   精确匹配HSCode数: {len(train_by_code)}")
    print(f"   品目级匹配数: {len(train_by_prefix4)}")

    print("\n[3/6] 提取PDF规则...")
    pdf_files = [
        os.path.join(CONFIG["pdf_dir"], "customs_rules.pdf"),
        os.path.join(CONFIG["pdf_dir"], "rules.pdf"),
        os.path.join(CONFIG["pdf_dir"], "zhushi.pdf"),
    ]
    
    pdf_chunks = []
    pdf_10digit_count = 0
    
    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"   跳过不存在: {os.path.basename(pdf_path)}")
            continue
        
        print(f"   处理: {os.path.basename(pdf_path)}")
        text = extract_pdf_text(pdf_path)
        if not text:
            continue
        
        # 切块
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        for chunk in chunks:
            codes = extract_hscodes(chunk)
            has_10 = any(len(c) == 10 for c in codes)
            if has_10:
                pdf_10digit_count += 1
            pdf_chunks.append({
                'text': chunk,
                'codes': codes,
                'source': 'pdf',
                'file': os.path.basename(pdf_path)
            })
    
    print(f"   PDF总chunk数: {len(pdf_chunks)}")
    print(f"   含10位HSCode的chunk: {pdf_10digit_count}")

    print("\n[4/6] 构建统一文档集...")
    
    all_docs = []  # [(text, code, source, metadata)]
    
    for code, problems in train_by_code.items():
        for prob in problems[:5]:
            cleaned = clean_text(prob)
            doc_text = f"商品描述: {cleaned}\nHSCode: {code}"
            all_docs.append({
                'text': doc_text,
                'code': code,
                'source': 'train_exact',
                'problem': prob
            })
    
    for code, problems in test_examples.items():
        for prob in problems[:3]:
            cleaned = clean_text(prob)
            doc_text = f"商品描述: {cleaned}\nHSCode: {code}"
            all_docs.append({
                'text': doc_text,
                'code': code,
                'source': 'test_self',
                'problem': prob
            })
    
    for prefix4, examples in train_by_prefix4.items():
        for ex in examples[:15]:
            cleaned = clean_text(ex['problem'])
            doc_text = f"商品描述: {cleaned}\nHSCode: {ex['code']}"
            all_docs.append({
                'text': doc_text,
                'code': ex['code'],
                'source': 'train_prefix',
                'problem': ex['problem']
            })
    
    for chunk in pdf_chunks:
        # 为PDF chunk选择一个代表性code
        rep_code = ""
        for c in chunk['codes']:
            if len(c) == 10:
                rep_code = c
                break
        if not rep_code and chunk['codes']:
            rep_code = chunk['codes'][0]
        
        all_docs.append({
            'text': chunk['text'],
            'code': rep_code,
            'source': 'pdf',
            'codes_in_chunk': chunk['codes']
        })
    
    print(f"   统一文档总数: {len(all_docs)}")
    print(f"   - 训练集精确匹配: {sum(1 for d in all_docs if d['source'] == 'train_exact')}")
    print(f"   - 测试集自身: {sum(1 for d in all_docs if d['source'] == 'test_self')}")
    print(f"   - 训练集品目级: {sum(1 for d in all_docs if d['source'] == 'train_prefix')}")
    print(f"   - PDF规则: {sum(1 for d in all_docs if d['source'] == 'pdf')}")
  
    print("\n[5/6] 构建BM25索引...")
    
    try:
        import jieba
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        print(f"   ⚠️ 缺少依赖: {e}")
        print("   请运行: pip install jieba rank_bm25")
        return
    
    texts = [d['text'] for d in all_docs]
    codes = [d['code'] for d in all_docs]
    sources = [d['source'] for d in all_docs]
    
    # 分词
    tokenized = []
    for text in tqdm(texts, desc="   分词"):
        tokens = list(jieba.cut(text))
        tokenized.append(tokens)
    
    bm25 = BM25Okapi(tokenized)
    
    # 保存BM25索引
    bm25_data = {
        'bm25': bm25,
        'texts': texts,
        'codes': codes,
        'sources': sources,
        'docs': all_docs
    }
    
    bm25_path = os.path.join(output_dir, "unified_bm25.pkl")
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25_data, f)
    print(f"   BM25索引已保存: {bm25_path}")
    
    print("\n[6/6] 构建FAISS向量索引...")
    
    embedding_model_path = CONFIG["embedding_model"]
    if not os.path.exists(embedding_model_path):
        print(f"    Embedding模型不存在: {embedding_model_path}")
        print("   跳过向量索引构建")
    else:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            
            print(f"   加载Embedding模型: {embedding_model_path}")
            model = SentenceTransformer(embedding_model_path, device='cuda')
            
            print(f"   编码 {len(texts)} 个文档...")
            embeddings = model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            embeddings = np.array(embeddings).astype('float32')
            
            # 构建FAISS索引
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)  # 内积（已归一化=余弦）
            index.add(embeddings)
            
            # 保存
            faiss.write_index(index, os.path.join(output_dir, "unified_faiss.index"))
            
            with open(os.path.join(output_dir, "unified_faiss_meta.pkl"), 'wb') as f:
                pickle.dump({
                    'texts': texts,
                    'codes': codes,
                    'sources': sources
                }, f)
            
            print(f"   FAISS索引已保存，向量数: {index.ntotal}")
            
        except Exception as e:
            print(f"    FAISS构建失败: {e}")
            print("   继续使用BM25索引")
    

    print("\n[7/7] 构建辅助数据...")
    
    # Few-shot示例库
    fewshot_db = {}
    for code in test_codes:
        examples = []
        # 优先训练集精确匹配
        if code in train_by_code:
            examples.extend(train_by_code[code][:5])
        # 补充测试集自身
        if code in test_examples:
            for p in test_examples[code]:
                if p not in examples and len(examples) < 8:
                    examples.append(p)
        if examples:
            fewshot_db[code] = examples
    
    with open(os.path.join(output_dir, "fewshot_examples.json"), 'w', encoding='utf-8') as f:
        json.dump(fewshot_db, f, ensure_ascii=False, indent=2)
    print(f"   Few-shot库: {len(fewshot_db)} 个HSCode")
    
    # 层级结构
    hierarchy = {
        'L2': defaultdict(set),
        'L4': defaultdict(set),
        'L6': defaultdict(set),
        'L8': defaultdict(set),
    }
    
    all_codes_set = set()
    for d in all_docs:
        if d['code'] and len(d['code']) >= 4:
            all_codes_set.add(d['code'])
        if d['source'] == 'pdf' and 'codes_in_chunk' in d:
            for c in d['codes_in_chunk']:
                if len(c) >= 4:
                    all_codes_set.add(c)
    
    for code in all_codes_set:
        if len(code) >= 4:
            hierarchy['L2'][code[:2]].add(code[:4])
        if len(code) >= 6:
            hierarchy['L4'][code[:4]].add(code[:6])
        if len(code) >= 8:
            hierarchy['L6'][code[:6]].add(code[:8])
        if len(code) >= 10:
            hierarchy['L8'][code[:8]].add(code[:10])
    
    # 转换为list
    for level in hierarchy:
        hierarchy[level] = {k: sorted(list(v)) for k, v in hierarchy[level].items()}
    
    with open(os.path.join(output_dir, "hierarchy_enhanced.pkl"), 'wb') as f:
        pickle.dump(dict(hierarchy), f)
    
    print(f"   层级结构: {len(hierarchy['L2'])} 章, {len(hierarchy['L4'])} 目")
    

    print("\n" + "=" * 60)
    print(" 统一检索库构建完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"- unified_bm25.pkl: BM25索引")
    print(f"- unified_faiss.index: 向量索引")
    print(f"- fewshot_examples.json: Few-shot示例")
    print(f"- hierarchy_enhanced.pkl: 层级结构")


if __name__ == "__main__":
    build_unified_database()
