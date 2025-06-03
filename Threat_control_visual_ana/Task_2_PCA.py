import os
import gc
import torch
import pandas as pd
import numpy as np
import time
import json
from umap import UMAP
from sklearn.decomposition import PCA

dir_path = 'data/test_data'
dataset_list = ['animals','cities','companies','elements','facts','inventions']
dfs = []
for type in dataset_list:
    dataset_path = os.path.join(dir_path, type + '.csv')
    df = pd.read_csv(dataset_path)
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)  # `ignore_index=True` 重置索引
labels = final_df['label'].tolist() 


liar_activations_path = 'results/run_p_0_activations_liar.pt'
baseline_activations_path = 'results/run_p_0_activations_baseline.pt'

liar_activations = torch.load(liar_activations_path)
baseline_activations = torch.load(baseline_activations_path)

labeled_activations = {}
for key in liar_activations:
    mask_0 = np.array(labels) == 0
    mask_1 = np.array(labels) == 1
    liar_true_activations = liar_activations[key][mask_1]
    liar_false_activations = liar_activations[key][mask_0]

    baseline_true_activations = baseline_activations[key][mask_1]
    baseline_false_activations = baseline_activations[key][mask_0]

    labeled_activations[key] = (liar_true_activations,liar_false_activations,baseline_true_activations,baseline_false_activations)

torch.save(labeled_activations, "results/labeled_activations.pt")

###############################################################################
#labeled_activations = torch.load("results/labeled_activations.pt")

def apply_pca_to_tuple(tensors, n_components=2):
    """
    对元组中的四个张量合并后做PCA，并分别降维
    Args:
        tensors: 包含四个2D张量的元组 (tensor1, tensor2, tensor3, tensor4)
        n_components: 目标维度（默认2）
    Returns:
        降维后的四个张量的元组 (pca_tensor1, pca_tensor2, pca_tensor3, pca_tensor4)
    """
    # 合并四个张量（沿行方向拼接）

    combined = torch.cat(tensors, dim=0)  # shape: (sum(N_i), D)
    # 转换为 NumPy（sklearn PCA 需要）
    combined_np = combined.numpy()
    
    # 拟合 PCA
    pca = PCA(n_components=n_components)
    pca.fit(combined_np)  # 在合并数据上计算PCA
    
    # 对每个张量分别降维
    pca_results = []
    for tensor in tensors:
        tensor_np = tensor.numpy()
        transformed = pca.transform(tensor_np)  # shape: (N_i, 2)
        pca_results.append(torch.from_numpy(transformed))
    
    return tuple(pca_results)

#labeled_activations_pca = {}
#for key in labeled_activations:
#    labeled_activations_pca[key] = apply_pca_to_tuple(labeled_activations[key], n_components=2)

#torch.save(labeled_activations_pca ,"results/labeled_activations_pca.pt")

#####################################################################################


def apply_umap_to_tuple(tensors, n_components=2):
    """
    对元组中的四个张量合并后做UMAP，并分别降维
    Args:
        tensors: 包含四个2D张量的元组 (tensor1, tensor2, tensor3, tensor4)
        n_components: 目标维度（默认2）
    Returns:
        降维后的四个张量的元组 (umap_tensor1, umap_tensor2, umap_tensor3, umap_tensor4)
    """
    # 合并四个张量（沿行方向拼接）
    combined = torch.cat(tensors, dim=0)  # shape: (sum(N_i), D)
    # 转换为 NumPy（UMAP 需要）
    combined_np = combined.numpy()
    
    # 拟合 UMAP
    umap = UMAP(n_components=n_components)
    umap.fit(combined_np)  # 在合并数据上计算UMAP
    
    # 对每个张量分别降维
    umap_results = []
    for tensor in tensors:
        tensor_np = tensor.numpy()
        transformed = umap.transform(tensor_np)  # shape: (N_i, 2)
        umap_results.append(torch.from_numpy(transformed))
    
    return tuple(umap_results)

labeled_activations = torch.load("results/labeled_activations.pt")
labeled_activations_umap = {}
for key in labeled_activations:
    labeled_activations_umap[key] = apply_umap_to_tuple(labeled_activations[key], n_components=2)

torch.save(labeled_activations_umap, "results/labeled_activations_umap.pt")










