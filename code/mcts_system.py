
import os
import sys
import json
import math
import random
import torch
import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from abc import ABC, abstractmethod
from tqdm import trange

# 导入模块
from bert_agent import BertAgent, ActionType
from rag_retriever import RAGRetriever
from IO_System import IO_System


node_cnt = 0


class MCTS_Node_Base(ABC):
    
    def __init__(self):
        global node_cnt
        self.id = node_cnt
        node_cnt += 1
        self.rollout_id = None
        self.uncertainty_score = 0.0
        self.confidence = 0.0
    
    def set_rollout_id(self, rid: int):
        self.rollout_id = rid
    
    @abstractmethod
    def find_children(self, rollout_id: int): raise NotImplementedError
    @abstractmethod
    def is_terminal(self): raise NotImplementedError
    @abstractmethod
    def calculate_reward(self): raise NotImplementedError
    @abstractmethod
    def skip_backprop(self): raise NotImplementedError


class MCTS_Searcher:

    def __init__(self,
                 exploration_weight: float = 1.0,
                 num_rollouts: int = 10,
                 discount: float = 1.0,
                 verbose: bool = False,
                 alpha_threshold: float = 0.95,  # UP-Q强信号阈值
                 beta: float = 0.5):             # UP-Q激进更新率
        
        self.Q: Dict[MCTS_Node_Base, float] = defaultdict(lambda: 0.0)
        self.N: Dict[MCTS_Node_Base, int] = defaultdict(lambda: 0)
        self.parent2children: Dict[MCTS_Node_Base, List[MCTS_Node_Base]] = dict()
        self.explored_nodes = set()
        
        self.exploration_weight = exploration_weight
        self.num_rollouts = num_rollouts
        self.discount = discount
        self.verbose = verbose
        
        self.alpha_threshold = alpha_threshold
        self.beta = beta
        
        global node_cnt
        node_cnt = 0
    
    def do_rollout(self, root_node: MCTS_Node_Base, rollout_id: int):
        """执行一次rollout"""
        path = self._select(root_node, rollout_id)
        leaf = path[-1]
        self._expand(leaf, rollout_id)
        self._backpropagate(path)
    
    def _select(self, node: MCTS_Node_Base, rollout_id: int) -> List[MCTS_Node_Base]:
        """选择节点"""
        path = []
        while True:
            path.append(node)
            if node not in self.parent2children:
                return path
            
            # 优先探索未访问节点（按置信度排序）
            unexplored = [n for n in self.parent2children[node] if n not in self.explored_nodes]
            if unexplored:
                unexplored.sort(key=lambda n: getattr(n, 'confidence', 0), reverse=True)
                n = unexplored[0] if random.random() > 0.3 else random.choice(unexplored)
                path.append(n)
                return path
            
            if node.is_terminal():
                return path
            
            node = self._uct_select(node, rollout_id)
        
        return path
    
    def _expand(self, node: MCTS_Node_Base, rollout_id: int):
        """展开节点"""
        if node in self.explored_nodes or node.is_terminal():
            return
        children = node.find_children(rollout_id)
        self.parent2children[node] = children
    
    def _backpropagate(self, path: List[MCTS_Node_Base]):
        """
        UP-Q反向传播
        核心算法：强信号覆盖 + 激进/保守更新
        """
        leaf = path[-1]
        if leaf.skip_backprop():
            return
        
        # 获取A6计算的分层奖励
        v = leaf.calculate_reward()
        
        for node in reversed(path):
            self.N[node] += 1
            
            if self.N[node] == 1:
                self.Q[node] = v
            else:
                q_old = self.Q[node]
                
                # 1. 强信号覆盖
                if v >= self.alpha_threshold:
                    self.Q[node] = v
                # 2. 激进更新（新路径更好）
                elif v > q_old:
                    self.Q[node] = q_old + self.beta * (v - q_old)
                # 3. 保守更新（新路径更差）
                else:
                    self.Q[node] = (q_old * (self.N[node] - 1) + v) / self.N[node]
            
            self.explored_nodes.add(node)
            v *= self.discount
    
    def _uct_select(self, node: MCTS_Node_Base, rollout_id: int) -> MCTS_Node_Base:
        """UCT选择"""
        children = self.parent2children.get(node)
        if not children:
            return node
        return max(children, key=lambda n: self._compute_uct(node, n))
    
    def _compute_uct(self, parent: MCTS_Node_Base, child: MCTS_Node_Base) -> float:
        if self.N[child] == 0:
            return float('inf')
        
        # 获取不确定性
        uncertainty = getattr(parent, 'uncertainty_score', 0.0)
        
        # 动态探索系数
        c_dynamic = self.exploration_weight * (1.0 + 1.5 * uncertainty)
        
        # BERT先验
        prior = getattr(child, 'confidence', 0.5)
        
        # AlphaGo风格PUCT
        exploitation = self.Q[child]
        exploration = c_dynamic * prior * (math.sqrt(self.N[parent]) / (1 + self.N[child]))
        
        return exploitation + exploration



class Generator:
    
    def __init__(self, args, tokenizer, model, bert_agent: BertAgent, retriever: RAGRetriever):
        self.args = args
        self.io = IO_System(args, tokenizer, model)
        self.bert_agent = bert_agent
        self.retriever = retriever
        
        # 缓存
        self.cache_A1 = {}
        self.cache_A6 = {}
        
        # 统计
        self.cnt_bert = 0
        self.cnt_llm = 0
        self.cnt_rag = 0
        
        # 加载Prompt模板
        self.prompts = {}
        self._load_prompts()
    
    def _load_prompts(self):
        """加载Prompt模板"""
        prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'HSCode')
        if not os.path.exists(prompt_dir):
            prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
        
        for name in ["A1_systematic_analysis", "A4_self_refinement", 
                     "A5_context_enhancement", "A6_constraint_check"]:
            path = os.path.join(prompt_dir, f"{name}.txt")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.prompts[name] = f.read()
    
    
    def get_bert_distribution(self, text: str, task_key: str = 'A2_L1'):
        """获取BERT概率分布（用于计算不确定性）"""
        if self.bert_agent is None:
            return None
        return self.bert_agent.get_distribution(text, task_key)
    
    def action_A2_direct(self, desc: str) -> List[Tuple[str, float]]:
        """A2: BERT直接预测10位"""
        if self.bert_agent is None:
            return []
        self.cnt_bert += 1
        preds = self.bert_agent.action_A2_direct(desc, top_k=self.args.num_votes)
        return [(p.code, p.confidence) for p in preds]
    
    def action_A3_hierarchical(self, desc: str, level: int) -> List[Tuple[str, float]]:
        """A3: BERT分层预测"""
        if self.bert_agent is None:
            return []
        self.cnt_bert += 1
        preds = self.bert_agent.action_A3_hierarchical(desc, level, "", self.args.num_votes)
        return [(p.code, p.confidence) for p in preds]
    
    
    def action_A1_analysis(self, product_desc: str) -> str:
        """A1: LLM系统分析（审题）"""
        if product_desc in self.cache_A1:
            return self.cache_A1[product_desc]
        
        # 获取Few-shot示例
        self.cnt_rag += 1
        fewshot = self.retriever.get_fewshot_examples(product_desc, top_k=3)
        
        template = self.prompts.get("A1_systematic_analysis", "")
        if not template:
            template = """请分析以下商品的归类要点：

商品描述：{product_desc}

参考案例：
{fewshot}

请简要分析：
1. 核心功能/用途
2. 主要材质
3. 可能的HS章节范围"""
        
        prompt = template.format(product_desc=product_desc, fewshot=fewshot)
        
        try:
            self.cnt_llm += 1
            res = self.io.generate(prompt, max_tokens=256, num_return=1)[0]
            self.cache_A1[product_desc] = res
            return res
        except Exception as e:
            print(f"[A1 Error] {e}")
            return ""
    
    def action_A4_refinement(self, product_desc: str, current_path: str,
                              candidates: List[Tuple[str, float]], 
                              analysis_ctx: str) -> List[Tuple[str, float]]:
        """A4: LLM自我完善（仲裁）"""
        if len(candidates) < 2:
            return candidates
        
        cands_fmt = "\n".join([f"选项{i+1}: {c[0]} (置信度: {c[1]:.2f})" 
                               for i, c in enumerate(candidates[:3])])
        
        # RAG检索差异信息
        self.cnt_rag += 1
        query = f"区别 {' '.join([c[0][:4] for c in candidates[:2]])} {product_desc[:100]}"
        context = self.retriever.get_similar_cases(query, top_k=2)
        
        template = self.prompts.get("A4_self_refinement", "")
        if not template:
            template = """请从以下候选中选择最合适的HSCode：

商品描述：{product_desc}
分析：{analysis_ctx}
参考案例：{context}

候选选项：
{cands_str}

请直接输出：选项X"""
        
        prompt = template.format(
            product_desc=product_desc,
            analysis_ctx=analysis_ctx,
            context=context,
            cands_str=cands_fmt
        )
        
        try:
            self.cnt_llm += 1
            res = self.io.generate(prompt, max_tokens=128, num_return=1)[0]
            
            # 解析选择
            for i in range(min(3, len(candidates))):
                if f"选项{i+1}" in res or f"选项 {i+1}" in res:
                    return [(candidates[i][0], 0.95)]
            
            return candidates[:1]
        except:
            return candidates[:1]
    
    def action_A5_context(self, product_desc: str, current_path: str,
                          analysis_ctx: str) -> List[Tuple[str, float]]:
        """A5: LLM+RAG上下文增强"""
        self.cnt_rag += 1
        
        # 获取低置信度时的BERT预测
        bert_preds = self.action_A3_hierarchical(product_desc, len(current_path))
        top_conf = bert_preds[0][1] if bert_preds else 0.3
        
        # RAG检索
        context, suggested_codes = self.retriever.retrieve_for_A5(
            product_desc, current_path, top_conf
        )
        
        if not context:
            return []
        
        template = self.prompts.get("A5_context_enhancement", "")
        if not template:
            # 使用默认模板
            return [(code, 0.6) for code in suggested_codes[:3] if code.startswith(current_path)]
        
        prompt = template.format(
            product_desc=product_desc,
            analysis_ctx=analysis_ctx,
            current_path=current_path,
            context=context
        )
        
        try:
            self.cnt_llm += 1
            res = self.io.generate(prompt, max_tokens=128, num_return=1)[0]
            
            # 提取编码
            import re
            codes = re.findall(r'\b\d{4,10}\b', res)
            
            results = []
            for code in codes[:3]:
                if code.startswith(current_path):
                    results.append((code, 0.7))
            
            if not results and suggested_codes:
                results = [(code, 0.6) for code in suggested_codes[:2] 
                          if code.startswith(current_path)]
            
            return results
        except:
            return [(code, 0.6) for code in suggested_codes[:2]]
    
    def action_A6_audit(self, product_desc: str, final_code: str) -> float:
        cache_key = f"{product_desc[:100]}_{final_code}"
        if cache_key in self.cache_A6:
            return self.cache_A6[cache_key]
        
        # RAG检索品目注释
        self.cnt_rag += 1
        context = self.retriever.retrieve_for_A6(product_desc, final_code)
        
        code_2 = final_code[:2] if len(final_code) >= 2 else ""
        code_4 = final_code[:4] if len(final_code) >= 4 else ""
        code_6 = final_code[:6] if len(final_code) >= 6 else ""
        
        template = self.prompts.get("A6_constraint_check", "")
        if not template:
            template = """请验证以下HSCode归类：

商品描述：{product_desc}
预测编码：{final_code}
参考信息：{context}

请判断（Yes/No）：
1. 第{code_2}章是否正确？
2. 品目{code_4}是否正确？
3. 子目{code_6}是否正确？
4. 全码{final_code}是否正确？

输出JSON格式：
{{"check_chapter": true/false, "check_heading": true/false, "check_subheading": true/false, "check_full": true/false}}"""
        
        prompt = template.format(
            product_desc=product_desc,
            final_code=final_code,
            code_2=code_2,
            code_4=code_4,
            code_6=code_6,
            context=context
        )
        
        try:
            self.cnt_llm += 1
            res = self.io.generate(prompt, max_tokens=128, num_return=1)[0]
            
            # 解析JSON
            start_idx = res.find('{')
            end_idx = res.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = res[start_idx:end_idx + 1]
                json_str = json_str.replace("True", "true").replace("False", "false")
                
                try:
                    data = json.loads(json_str)
                except:
                    return 0.3
            else:
                return 0.3
            
            def is_true(val):
                return str(val).lower() in ['true', 'yes', '1', 't']
            
            checks = [
                is_true(data.get('check_chapter', False)),
                is_true(data.get('check_heading', False)),
                is_true(data.get('check_subheading', False)),
                is_true(data.get('check_full', False))
            ]
            
            # 分层计分
            score = 0.0
            if checks[0]:  # 章正确 +0.2
                score += 0.2
                if checks[1]:  # 目正确 +0.3
                    score += 0.3
                    if checks[2]:  # 子目正确 +0.2
                        score += 0.2
                        if checks[3]:  # 全码正确 +0.3
                            score += 0.3
            
            final_score = score if checks[0] else 0.0
            self.cache_A6[cache_key] = final_score
            return final_score
            
        except Exception as e:
            print(f"[A6 Error] {e}")
            return 0.2
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            "bert_calls": self.cnt_bert,
            "llm_calls": self.cnt_llm,
            "rag_calls": self.cnt_rag
        }
        if self.bert_agent:
            stats.update(self.bert_agent.get_stats())
        return stats



class HSCode_MCTS_Node(MCTS_Node_Base):
    """HSCode MCTS节点"""
    
    def __init__(self, parent, depth, hs_prefix, confidence, 
                 analysis_context="", **kwargs):
        super().__init__()
        self.parent = parent
        self.depth = depth
        self.hs_prefix = hs_prefix
        self.confidence = confidence
        self.children = []
        
        # 继承参数
        if parent is None:
            self.generator = kwargs.get('generator')
            self.user_question = kwargs.get('user_question')
            self.max_depth = kwargs.get('max_depth_allowed', 5)
            self.analysis_context = ""
            
            # 计算根节点不确定性
            if self.generator and self.generator.bert_agent:
                self.uncertainty_score = self.generator.bert_agent.get_uncertainty(self.user_question)
            else:
                self.uncertainty_score = 0.5
        else:
            self.generator = parent.generator
            self.user_question = parent.user_question
            self.max_depth = parent.max_depth
            self.analysis_context = analysis_context or parent.analysis_context
            self.uncertainty_score = parent.uncertainty_score
        
        self.is_direct_prediction = kwargs.get('is_direct_prediction', False)
        self.action_source = kwargs.get('action_source', 'BERT')
        
        # 解决方案追踪
        if parent is None:
            self.solution_trace = {"path": []}
        else:
            self.solution_trace = deepcopy(parent.solution_trace)
            tag = "A2" if self.is_direct_prediction else f"A3_L{len(hs_prefix)}"
            self.solution_trace["path"].append(f"[{tag}] {hs_prefix} ({confidence:.2f})")
    
    def skip_backprop(self):
        return self.depth == 0
    
    def is_terminal(self):
        if self.hs_prefix == "FAIL":
            return True
        if self.is_direct_prediction:
            return True
        if len(self.hs_prefix) >= 10:
            return True
        if self.depth >= self.max_depth:
            return True
        return False
    
    def calculate_reward(self):
        if not self.is_terminal():
            return 0.0
        if self.hs_prefix == "FAIL":
            return -1.0
        
        # 调用A6进行分级校验
        score = self.generator.action_A6_audit(self.user_question, self.hs_prefix)
        
        if score <= 0.1:
            return -1.0
        return score
    
    def find_children(self, rid):
        if self.children:
            for c in self.children:
                c.set_rollout_id(rid)
            return self.children
        
        return self._create_children()
    
    def _create_children(self):
        """创建子节点"""
        
        if self.depth == 0:
            # A1: 系统分析
            if not self.analysis_context:
                self.analysis_context = self.generator.action_A1_analysis(self.user_question)
            
            # [Fast Path] A2直接预测10位
            direct_cands = self.generator.action_A2_direct(self.user_question)
            for c, s in direct_cands[:3]:
                if len(str(c)) >= 8:
                    self.children.append(HSCode_MCTS_Node(
                        self, self.depth + 1, str(c), s,
                        self.analysis_context,
                        is_direct_prediction=True,
                        action_source="A2_Direct"
                    ))
            
            # [Slow Path] A3分层预测（从2位开始）
            layer1_cands = self.generator.action_A3_hierarchical(self.user_question, 0)
            for c, s in layer1_cands:
                self.children.append(HSCode_MCTS_Node(
                    self, self.depth + 1, str(c), s,
                    self.analysis_context,
                    is_direct_prediction=False,
                    action_source="A3_L2"
                ))
            
            return self.children
        
        cur_len = len(self.hs_prefix)
        
        # BERT分层预测
        bert_cands = []
        if cur_len in [2, 4, 6]:
            bert_cands = self.generator.action_A3_hierarchical(self.user_question, cur_len)
        else:
            return []
        
        final_cands = bert_cands
        trigger_A4 = False
        
        # A4触发逻辑：品目(4位)级别冲突
        if cur_len == 2 and len(bert_cands) >= 2:
            c1, c2 = str(bert_cands[0][0]), str(bert_cands[1][0])
            if c1[:4] != c2[:4]:  # 品目不同
                trigger_A4 = True
        
        # 置信度检查
        if bert_cands and bert_cands[0][1] < 0.4:
            # A5上下文增强
            rag_cands = self.generator.action_A5_context(
                self.user_question, self.hs_prefix, self.analysis_context
            )
            if rag_cands:
                final_cands = rag_cands
        elif trigger_A4:
            # A4仲裁
            llm_choice = self.generator.action_A4_refinement(
                self.user_question, self.hs_prefix,
                bert_cands[:3], self.analysis_context
            )
            if llm_choice:
                final_cands = llm_choice
        
        # 创建子节点
        seen = set()
        for c, s in final_cands:
            c_str = str(c)
            if c_str in seen:
                continue
            if not c_str.startswith(self.hs_prefix):
                continue
            seen.add(c_str)
            
            action_src = "A4" if trigger_A4 else f"A3_L{len(c_str)}"
            self.children.append(HSCode_MCTS_Node(
                self, self.depth + 1, c_str, s,
                self.analysis_context,
                is_direct_prediction=False,
                action_source=action_src
            ))
        
        if not self.children:
            self.children.append(HSCode_MCTS_Node(
                self, self.depth + 1, "FAIL", 0.0,
                action_source="FAIL"
            ))
        
        return self.children


def search_for_answers(args, user_question, question_id, gt_answer, generator):
    """执行MCTS搜索"""
    
    mcts = MCTS_Searcher(
        exploration_weight=getattr(args, 'mcts_exploration_weight', 1.0),
        num_rollouts=args.num_rollouts,
        verbose=args.verbose,
        alpha_threshold=0.90,
        beta=0.5
    )
    
    root = HSCode_MCTS_Node(
        None, 0, "", 1.0,
        generator=generator,
        user_question=user_question,
        max_depth_allowed=args.max_depth_allowed
    )
    
    # 执行MCTS搜索
    for i in trange(args.num_rollouts, disable=not args.verbose, 
                    desc=f"MCTS Q{question_id}"):
        mcts.do_rollout(root, i)
    
    # 收集结果
    valid_solutions = []
    
    def collect_leaves(node):
        if node.is_terminal() and node.hs_prefix != "FAIL":
            valid_solutions.append(node)
        elif node in mcts.parent2children:
            for child in mcts.parent2children[node]:
                collect_leaves(child)
    
    collect_leaves(root)
    
    # 按Q值排序
    final_data = []
    if valid_solutions:
        valid_solutions.sort(key=lambda n: mcts.Q[n], reverse=True)
        for n in valid_solutions:
            final_data.append({
                "hs": n.hs_prefix,
                "reward": round(mcts.Q[n], 4),
                "bert_conf": round(n.confidence, 4),
                "visits": mcts.N[n],
                "source": n.action_source,
                "depth": n.depth,
                "trace": n.solution_trace
            })
    
    # 保存结果
    res_path = os.path.join(args.answer_sheets_dir, f"Question_{question_id:04d}_solutions.json")
    with open(res_path, "w", encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    return final_data
