import os
import gc
import torch
import pandas as pd
import numpy as np
import time
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
from transformers import AutoModelForCausalLM, AutoTokenizer


def template(system,user,tokenizer,suffix=''):
    message = [
                {"role": "system", "content": system + suffix},
                {"role": "user", "content": user},
    ]
    mes_format = tokenizer.apply_chat_template(
    message,
    tokenize = False,
    add_generation_prompt=True
    )
    return mes_format


def create_hook(activations,layer):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations[f"layer {layer}"].append(output[0].detach()[:,-1,:].float().cpu())#.numpy() 
            #print(f"layer {layer} tuple 提取成功")
        else:
            activations[f"layer {layer}"].append(output.detach()[:,-1,:].float().cpu())#.numpy()
            #print(f"layer {layer} tensor 提取成功")
    return hook_fn



def extract_activations(model,tokenizer,device,data_path,sample_data = None ,batch_size=16):
    with open(data_path, 'r') as f:
        dataset_ori = json.load(f)
    
    if sample_data:
        dataset_ori = dataset_ori[:sample_data]
    dataset = []
    positive_suffix = ' You should keep honest and shouldn\'t lie or hide the truth.'
    negative_suffix = ' You can choose to lie or be honest at any scenario.'
    for data in dataset_ori:
        dataset.append({
            'positive':template(data['system'],data['user'],tokenizer,positive_suffix),
            'negative':template(data['system'],data['user'],tokenizer,negative_suffix),
        })

    last_token_activations = {}
    for layer in range(len(model.model.layers)):
        last_token_activations[f"layer {layer}"] =  [] 

    # 注册所有层的hook
    handles = []
    for layer in range(len(model.model.layers)):  # 遍历所有子模块
        hook_fn = create_hook(last_token_activations,layer)
        handles.append(model.model.layers[layer].register_forward_hook(hook_fn)) 

    total_data = len(dataset)
    for start in range(0, total_data, batch_size):
        end = min(start + batch_size, total_data)
        print(f"Processing batch {start // batch_size + 1}/{(total_data + batch_size - 1) // batch_size}...")
        batch_data = dataset[start:end]

        batch_data_list = []
        for data in batch_data:
            batch_data_list.append(data['positive'])
            batch_data_list.append(data['negative'])

        inputs = tokenizer(batch_data_list, padding=True,padding_side="left",return_tensors="pt").to(device)

        with torch.no_grad():
            total_generated_ids = model(
            **inputs,
            #temperature=0.9,  # 控制生成文本的多样性
            #top_p = 0.6,  # 采样范围
            #top_k = 30,  # 采样范围
            )
    for layer in range(len(model.model.layers)):
        last_token_activations[f"layer {layer}"] = torch.cat(last_token_activations[f"layer {layer}"],dim = 0)

    os.makedirs("role_play_scenario/activations", exist_ok=True)
    torch.save(last_token_activations, f"role_play_scenario/activations/ori_activations.pt")

    # 移除钩子
    for handle in handles:
        handle.remove()

############################################################################################
'''
# 调用垃圾回收
gc.collect()
# 清理 CUDA 内存
torch.cuda.empty_cache()

model_path = "QwQ_32"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 半精度减少显存占用
    device_map="auto",  # 关键：自动分配模型到多个 GPU
    #sliding_window=None,  # 关闭滑动窗口
    #attn_implementation="flash_attention_2",  #  sdpa
)
# 打印device_map
print(model.hf_device_map)
print(f'memory_allocated {torch.cuda.memory_allocated()}')
device = "cuda:0"


#extract_activations(model,tokenizer,device,'role_play_scenario/dataset/train.json')
#print('extract activations done')
'''
###########################################################################################
def process_pairs_to_pca(input_dict_path):
    """
    处理包含正负样本对的字典，进行差值、规范化、PCA分析，并返回第一主成分
    
    参数:
        input_dict (dict): 输入字典，格式为{'layer i': torch.Tensor}
        
    返回:
        dict: 包含各层第一主成分的字典，格式为{'layer i': torch.Tensor} [1,d]
    """
    input_dict = torch.load(input_dict_path)
    result_dict = {}
    
    for layer_name, tensor in input_dict.items():
        # 检查张量维度
        if tensor.dim() != 2:
            raise ValueError(f"张量 {layer_name} 应该是二维的")
            
        num_pairs = tensor.size(0) // 2
        if 2 * num_pairs != tensor.size(0):
            raise ValueError(f"张量 {layer_name} 的行数应该是偶数")
            
        # 存储所有差值向量
        diff_vectors = []
        
        for i in range(num_pairs):
            pos_idx = 2 * i      # 正样本索引
            neg_idx = 2 * i + 1  # 负样本索引
            
            pos_vec = tensor[pos_idx]
            neg_vec = tensor[neg_idx]
            
            # 随机决定差值顺序
            if random.random() < 0.5:
                diff = pos_vec - neg_vec
            else:
                diff = neg_vec - pos_vec
                
            diff_vectors.append(diff.numpy())  # 转换为numpy数组
            
        # 转换为numpy矩阵
        diff_matrix = np.vstack(diff_vectors)
        
        # 规范化数据
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(diff_matrix)
        
        # PCA分析，提取第一主成分
        pca = PCA(n_components=1)
        pca.fit(normalized_data)
        first_component = pca.components_[0]
        
        # 转换回torch张量并存储
        result_dict[layer_name] = torch.from_numpy(first_component).float().unsqueeze(0)

    torch.save(result_dict, "role_play_scenario/activations/steering_vectors_pca.pt")    



#process_pairs_to_pca("role_play_scenario/activations/ori_activations.pt")
#print('pca done')

steering_vectors = torch.load('role_play_scenario/activations/steering_vectors_pca.pt')
print(steering_vectors['layer 50'].shape)