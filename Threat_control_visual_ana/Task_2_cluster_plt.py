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
random_seed = 42

activations_cluster = {}
for key in labeled_activations_pca:
    tempt = []
    datas = labeled_activations_pca[key]

    for data in datas:
        data_np = data.numpy()

        kmeans = KMeans(n_clusters=3, random_state=random_seed)
        kmeans.fit(data_np) 

        cluster_centers_np = kmeans.cluster_centers_
        cluster_centers = torch.from_numpy(cluster_centers_np)  

        tempt.append(cluster_centers)
    
    tempt = tuple(tempt)
    activations_cluster[key] = tempt
#torch.save(activations_cluster,"results/activations_cluster.pt")
#######################################################################


def visualize_cluster_centers(layer_dict, output_dir="cluster_visualizations", figsize=(12, 8)):
    """
    聚类中心可视化，明确展示四种类别
    
    参数:
        layer_dict: 字典，key是layer名称，value是四元元组(每个元素是聚类中心张量)
        output_dir: 输出目录路径
        figsize: 图像大小
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义四种不同的样式（颜色+标记形状）
    styles = [
        {'color': 'red', 'marker': 'o', 'label': 'liar_true'},
        {'color': 'blue', 'marker': 's', 'label': 'liar_false'},
        {'color': 'green', 'marker': '^', 'label': 'baseline_true'},
        {'color': 'purple', 'marker': 'D', 'label': 'baseline_false'}
    ]
    
    for layer_name, clusters in layer_dict.items():
        plt.figure(figsize=figsize)
        
        # 绘制每个类别的聚类中心
        for i, (cluster_tensor, style) in enumerate(zip(clusters, styles)):
            if cluster_tensor is None or len(cluster_tensor) == 0:
                continue
                
            centers = cluster_tensor.numpy()
            
            # 绘制该类别的所有中心点
            plt.scatter(centers[:, 0], centers[:, 1],
                        c=style['color'],
                        marker=style['marker'],
                        s=80,  # 点大小
                        edgecolors='k',  # 黑色边框
                        linewidths=0.5,  # 边框宽度
                        alpha=0.8,
                        label=style['label'])
            
            # 绘制该类别的平均中心（用更大的星形表示）
            if len(centers) > 0:
                mean_center = np.mean(centers, axis=0)
                plt.scatter(mean_center[0], mean_center[1],
                          c=style['color'],
                          marker='*',  # 星形表示均值
                          s=200,  # 更大的尺寸
                          edgecolors='k',
                          linewidths=1,
                          label=f'{style["label"]} Mean')
        
        # 添加图表元素
        plt.title(f'Cluster Centers Visualization - {layer_name}\n(Showing 4 Types with Distinct Markers)', pad=20)
        plt.xlabel('Feature Dimension 1', labelpad=10)
        plt.ylabel('Feature Dimension 2', labelpad=10)
        
        # 添加说明文字
        plt.annotate('Four Distinct Cluster Types:',
                    xy=(0.02, 0.95),
                    xycoords='axes fraction',
                    fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 调整图例位置和样式
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 调整布局防止标签被截断
        plt.tight_layout()
        
        # 保存图像
        output_path = os.path.join(output_dir, f"{layer_name}_clusters.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved enhanced visualization for {layer_name} to {output_path}")
        
        plt.close()

#activations_cluster = torch.load("results/activations_cluster.pt")
#visualize_cluster_centers(activations_cluster, output_dir="cluster_visualizations", figsize=(12, 8))
