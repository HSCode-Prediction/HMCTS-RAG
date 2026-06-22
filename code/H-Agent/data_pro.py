
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os
import numpy as np

class HM_BERT_DataProcessor:
    def __init__(self):
        self.hs4_categories = set()
        self.hs6_categories = set()
        self.hs10_categories = set()
        
    def clean_problem_text(self, problem_text):
        """清理问题文本，按照论文方式提取商品信息"""
        cleaned = problem_text.replace(
            "您是一位海关商品分类专家。请根据以下商品信息，分析其材质、功能、用途等特征，预测其HS编码（商品编号）。请只输出HS编码，不需要解释：\n\n", 
            ""
        )
        
        lines = cleaned.split('\n')
        g_name = ""
        g_model = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('商品名称:'):
                g_name = line.replace('商品名称:', '').strip()
            elif line.startswith('规格型号:'):
                g_model = line.replace('规格型号:', '').strip()
        
        if g_name and g_model:
            return f"{g_name} {g_model}"
        elif g_name:
            return g_name
        else:
            return g_model
    
    def extract_hs_hierarchy(self, hs10_code):
        """从10位HS编码提取4位和6位编码"""
        if len(hs10_code) != 10:
            raise ValueError(f"HS编码长度不正确: {hs10_code}")
        
        hs4 = hs10_code[:4]  # 前4位
        hs6 = hs10_code[:6]  # 前6位
        
        # 收集所有类别
        self.hs4_categories.add(hs4)
        self.hs6_categories.add(hs6)
        self.hs10_categories.add(hs10_code)
        
        return hs4, hs6, hs10_code

    def robust_data_split(self, df, test_size=0.1, val_size=0.1, random_state=42):
        """按照论文0.81:0.09:0.10的比例划分数据，使用HS10分层"""
        print("使用HS10分层数据划分策略...")
        
        # 统计每个HS10类别的样本数
        hs10_counts = df['hs10'].value_counts()
        
        print(f"HS10类别总数: {len(hs10_counts)}")
        print(f"总样本数: {len(df)}")
        
        rare_hs10 = hs10_counts[hs10_counts == 1].index.tolist()
        normal_df = df[~df['hs10'].isin(rare_hs10)]
        rare_df = df[df['hs10'].isin(rare_hs10)]
        
        print(f"正常类别数据: {len(normal_df)} 条")
        print(f"稀有类别数据: {len(rare_df)} 条")
        
        # 对正常类别数据进行分层抽样
        if len(normal_df) > 0:
            try:
                train_normal, temp_normal = train_test_split(
                    normal_df, 
                    test_size=test_size + val_size, 
                    stratify=normal_df['hs10'],
                    random_state=random_state
                )
                
                if len(temp_normal) > 0:
                    val_ratio = val_size / (test_size + val_size)
                    val_normal, test_normal = train_test_split(
                        temp_normal,
                        test_size=1 - val_ratio,  # 0.10 / 0.19 ≈ 0.526
                        stratify=temp_normal['hs10'],
                        random_state=random_state
                    )
                else:
                    val_normal = pd.DataFrame()
                    test_normal = pd.DataFrame()
            except Exception as e:
                print(f"分层抽样失败: {e}，使用随机抽样")
                train_normal, temp_normal = train_test_split(
                    normal_df, 
                    test_size=test_size + val_size, 
                    random_state=random_state
                )
                val_ratio = val_size / (test_size + val_size)
                val_normal, test_normal = train_test_split(
                    temp_normal,
                    test_size=1 - val_ratio,
                    random_state=random_state
                )
        else:
            train_normal = pd.DataFrame()
            val_normal = pd.DataFrame()
            test_normal = pd.DataFrame()
        
        train_rare = rare_df
        
        train_df = pd.concat([train_normal, train_rare], ignore_index=True)
        
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        print(f"最终划分结果:")
        print(f"训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
        print(f"验证集: {len(val_normal)} 条 ({len(val_normal)/len(df)*100:.1f}%)")  
        print(f"测试集: {len(test_normal)} 条 ({len(test_normal)/len(df)*100:.1f}%)")
        
        return train_df, val_normal, test_normal

    def process_to_hmbert_format(self, json_file_path, output_dir='/root/HSCode-MCTS-RAG/fintuning/hmbert_data'):
        """处理为HM-BERT训练所需的JSON格式"""
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"数据将保存到: {output_dir}")
        
        # 读取原始数据
        with open(json_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        processed_samples = []
        
        for item in raw_data:
            try:
                text = self.clean_problem_text(item['problem'])
                
                hs4, hs6, hs10 = self.extract_hs_hierarchy(item['solution'])
                
                sample = {
                    "text": text,
                    "hs4": hs4,    # 4位HS编码
                    "hs6": hs6,    # 6位HS编码
                    "hs10": hs10   # 10位HS编码
                }
                processed_samples.append(sample)
                
            except Exception as e:
                print(f"处理数据出错 (ID: {item.get('id', 'unknown')}): {e}")
                continue
        
        print(f"成功处理 {len(processed_samples)} 条数据")
        
        # 转换为DataFrame
        df = pd.DataFrame(processed_samples)
        
        # 数据划分
        print("开始数据集划分...")
        train_df, val_df, test_df = self.robust_data_split(df)
        
        # 检查数据分布
        self._check_data_distribution(train_df, val_df, test_df)
        
        # 转换为JSON格式
        def df_to_json_format(df_split):
            return df_split.to_dict('records')
        
        # 保存各数据集
        train_json = df_to_json_format(train_df)
        val_json = df_to_json_format(val_df)
        test_json = df_to_json_format(test_df)
        
        # 保存JSON文件
        train_path = f'{output_dir}/train.json'
        val_path = f'{output_dir}/val.json'
        test_path = f'{output_dir}/test.json'
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_json, f, ensure_ascii=False, indent=2)
        
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_json, f, ensure_ascii=False, indent=2)
            
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_json, f, ensure_ascii=False, indent=2)
        
        # 保存类别信息
        label_info = {
            "hs4_categories": sorted(list(self.hs4_categories)),
            "hs6_categories": sorted(list(self.hs6_categories)),
            "hs10_categories": sorted(list(self.hs10_categories)),
            "hs4_num_classes": len(self.hs4_categories),
            "hs6_num_classes": len(self.hs6_categories),
            "hs10_num_classes": len(self.hs10_categories)
        }
        
        label_info_path = f'{output_dir}/label_info.json'
        with open(label_info_path, 'w', encoding='utf-8') as f:
            json.dump(label_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 数据保存完成 ===")
        print(f"训练集: {train_path} ({len(train_df)} 条)")
        print(f"验证集: {val_path} ({len(val_df)} 条)")
        print(f"测试集: {test_path} ({len(test_df)} 条)")
        print(f"标签信息: {label_info_path}")
        
        # 保存划分统计信息
        split_info = {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "split_ratio": "0.81:0.09:0.10",
            "split_method": "stratified_split_hs10"
        }
        
        with open(f'{output_dir}/split_info.json', 'w', encoding='utf-8') as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)
        
        return {
            "train": train_json,
            "val": val_json,
            "test": test_json,
            "label_info": label_info
        }
    
    def _check_data_distribution(self, train_df, val_df, test_df):
        print("\n=== 数据集分布检查 ===")
        
        for name, df in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
            if len(df) == 0:
                print(f"{name}: 无数据")
                continue
                
            hs4_counts = df['hs4'].value_counts()
            hs6_counts = df['hs6'].value_counts()
            hs10_counts = df['hs10'].value_counts()
            
            print(f"{name}:")
            print(f"  - 总样本数: {len(df)}")
            print(f"  - HS4类别数: {len(hs4_counts)}")
            print(f"  - HS6类别数: {len(hs6_counts)}")
            print(f"  - HS10类别数: {len(hs10_counts)}")
            print()

# 使用示例
if __name__ == "__main__":
    processor = HM_BERT_DataProcessor()
    
    json_file_path = "/root/MCTS-RAG-main/data/data.json"
    
    try:
        result = processor.process_to_hmbert_format(json_file_path)
        print("数据处理完成！")
        
        # 显示样例
        print("\n训练集样例:")
        print(json.dumps(result["train"][0], ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()