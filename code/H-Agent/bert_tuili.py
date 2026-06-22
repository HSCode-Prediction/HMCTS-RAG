"""
HM-BERT 多任务推理脚本 (适配 5 个任务)
"""

import os
from transformers import BertTokenizer
import torch
from bert_get_data import HMBertClassifier # 导入修改后的模型
import joblib # 用于加载编码器

# --- 配置 ---
bert_name = '/root/HMCTS/models/bert_chinese'
model_save_path = '/root/HMCTS/models/HS-BERT' # 训练好的模型保存路径
data_prefix = '/root/agent/bert_mtl_all_data/bert_mtl_all_data' # 编码器的前缀
max_length = 128
# -------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载 Tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained(bert_name)
except Exception as e:
    print(f"加载 tokenizer 时出错: {e}")
    exit()

# 加载编码器
print("加载标签编码器...")
encoders = {}
num_classes = {}
task_names = ['A1', 'A2_L1', 'A2_L2', 'A2_L3', 'A2_L4']
try:
    for task_name in task_names:
        encoder_path = f'{data_prefix}_{task_name}_encoder.pkl'
        encoders[task_name] = joblib.load(encoder_path)
        num_classes[task_name] = len(encoders[task_name].classes_)
        print(f"已加载 {task_name} 编码器，包含 {num_classes[task_name]} 个类别。")
except FileNotFoundError as e:
    print(f"加载编码器时出错: {e}。没有编码器无法进行推理。")
    exit()
except Exception as e:
    print(f"加载编码器时发生错误: {e}")
    exit()

# 初始化并加载模型
print(f"从 {os.path.join(model_save_path, 'best.pt')} 加载模型...")
try:
    model = HMBertClassifier(
        num_a1=num_classes['A1'],
        num_a2_l1=num_classes['A2_L1'],
        num_a2_l2=num_classes['A2_L2'],
        num_a2_l3=num_classes['A2_L3'],
        num_a2_l4=num_classes['A2_L4']
    )
    # 加载状态字典到正确的设备
    model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pt'), map_location=device))
    model = model.to(device)
    model.eval() # 设置为评估模式
except FileNotFoundError:
    print(f"错误：在 {model_save_path} 中未找到模型文件 'best.pt'")
    exit()
except Exception as e:
    print(f"加载模型时出错: {e}")
    exit()

print("\nHM-BERT 推理系统准备就绪。")
print("请输入商品描述以进行 HS 编码预测。")
print("输入 'quit' 退出。")
print("-" * 50)

while True:
    text = input('请输入商品描述: ')
    if text.lower() == 'quit':
        break
    if not text.strip(): # 处理空输入
        print("请输入描述。")
        continue

    # Tokenize 输入
    bert_input = tokenizer(text, padding='max_length',
                           max_length=max_length,
                           truncation=True,
                           return_tensors="pt")
    input_ids = bert_input['input_ids'].to(device)
    masks = bert_input['attention_mask'].to(device)

    # 执行推理
    with torch.no_grad():
        logits_list = model(input_ids, masks)
        logits = dict(zip(task_names, logits_list))

    # 获取预测的索引
    predictions_idx = {}
    for task in task_names:
        predictions_idx[task] = logits[task].argmax(dim=1).cpu().item()

    # 将索引解码回 HS 编码
    predictions_code = {}
    try:
        for task in task_names:
            # 使用对应的编码器进行反向转换
            predictions_code[task] = encoders[task].inverse_transform([predictions_idx[task]])[0]
    except ValueError as e:
         print(f"解码过程中出错: {e}。请检查模型输出索引对于编码器是否有效。")
         # 处理模型预测的索引超出编码器已知类别范围的潜在问题
         for task in task_names:
              predictions_code[task] = f"解码索引 {predictions_idx[task]} 时出错"

    # 打印结果
    print("\n--- 预测结果 ---")
    print(f"直接预测 (A1):      {predictions_code['A1']}")
    print("分层路径预测 (A2):")
    print(f"  章 (L1):          {predictions_code['A2_L1']}")
    print(f"  目 (L2):          {predictions_code['A2_L2']}")
    print(f"  子目 (L3):        {predictions_code['A2_L3']}")
    print(f"  最终编码 (L4):    {predictions_code['A2_L4']}") # 这个理想情况下应与 A1 匹配
    print("-" * 50)