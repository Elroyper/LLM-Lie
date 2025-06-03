import os
import gc
import torch
import pandas as pd
import numpy as np
import time
import json

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

labeled_activations_pca = torch.load("results/labeled_activations_pca.pt")

import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_activations(layer_key, activations_tuple, n_samples=10, save_dir="direct_visualizations"):
    """
    可视化一层中四类激活向量中各N个样本

    参数:
        layer_key: 层的名称
        activations_tuple: 四元组 (liar_true, liar_false, baseline_true, baseline_false)
        n_samples: 每类选择的样本数量
        save_dir: 保存图像的目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 解包四元组
    liar_true, liar_false, baseline_true, baseline_false = activations_tuple

    # 为每类选择n个样本（如果样本数量不足，则使用全部样本）
    liar_true_samples = liar_true[:min(n_samples, liar_true.shape[0])]
    liar_false_samples = liar_false[:min(n_samples, liar_false.shape[0])]
    baseline_true_samples = baseline_true[:min(n_samples, baseline_true.shape[0])]
    baseline_false_samples = baseline_false[:min(n_samples, baseline_false.shape[0])]

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 绘制四类点
    plt.scatter(liar_true_samples[:, 0], liar_true_samples[:, 1], color='red', label='Liar True', marker='o')
    plt.scatter(liar_false_samples[:, 0], liar_false_samples[:, 1], color='red', label='Liar False', marker='*')
    plt.scatter(baseline_true_samples[:, 0], baseline_true_samples[:, 1], color='blue', label='Baseline True', marker='o')
    plt.scatter(baseline_false_samples[:, 0], baseline_false_samples[:, 1], color='blue', label='Baseline False', marker='*')

    # 添加图例和标签
    plt.legend()
    plt.title(f'Layer {layer_key} Activations')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    save_path = os.path.join(save_dir, f'layer_{layer_key}_activations.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已保存层 {layer_key} 的可视化到 {save_path}")

# 遍历字典中的每一层并进行可视化
def visualize_all_layers(labeled_activations_pca, n_samples=10, save_dir="results/visualizations"):
    """
    对字典中所有层的激活向量进行可视化

    参数:
        labeled_activations_pca: 字典，键为层名，值为四元组激活向量
        n_samples: 每类选择的样本数量
        save_dir: 保存可视化结果的目录
    """
    for layer_key, activations_tuple in labeled_activations_pca.items():
        visualize_activations(layer_key, activations_tuple, n_samples, save_dir)

    print(f"已完成所有层的可视化，结果保存在 {save_dir}")


visualize_all_layers(labeled_activations_pca, 50,"results/visualizations_50")