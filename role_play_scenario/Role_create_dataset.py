import pandas as pd
import os
import json
import random


output_dir = 'role_play_scenario/dataset'
os.makedirs(output_dir, exist_ok=True)

with open('role_play_scenario/data.json','r') as f:
    data = json.load(f)

print(len(data))

# 随机抽样 64 个作为测试集
test_size = 60
test_data = random.sample(data, test_size)

# 剩余的作为训练集
train_data = [item for item in data if item not in test_data]

# 保存为 JSON 文件
with open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(os.path.join(output_dir, 'test.json'), 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

# 打印样本数量
print(f"训练集样本数: {len(train_data)}")
print(f"测试集样本数: {len(test_data)}")

