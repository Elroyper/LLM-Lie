import os
import gc
import torch
import pandas as pd
import numpy as np
import random
import time
import pickle
import json
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_curve
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

path = 'results/predic_information/ana_info_combined_threshold_0.5.json'
with open(path, 'r', encoding='utf-8') as f:
    data_dict = json.load(f)

def visualize_metrics(data_dict, metric, output_folder, by_key_plot_type='heatmap'):
    """
    可视化不同分类下的评估指标
    
    参数:
        data_dict (dict): 包含by_key, by_layer, by_type三个子字典的大字典
        metric (str): 要可视化的指标名称('Acc', 'Precision', 'Recall', 'F1')
        output_folder (str): 保存图像的文件夹路径
        by_key_plot_type (str): by_key的可视化类型，'heatmap'或'line'，默认为'heatmap'
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 检查metric是否有效
    valid_metrics = ['Acc', 'Precision', 'Recall', 'F1']
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric. Must be one of {valid_metrics}")
    
    # 1. 可视化by_layer - 折线图 (优化横轴显示)
    by_layer_data = data_dict['by_layer']
    layers = sorted(by_layer_data.keys(), key=lambda x: int(x.split()[1]))
    values = [by_layer_data[layer][metric] for layer in layers]

    plt.figure(figsize=(12, 6))  # 增加图形宽度

    # 方法1: 每隔N层显示一个刻度
    step = max(1, len(layers) // 10)  # 自动计算步长，最多显示10个刻度
    plt.xticks(range(0, len(layers), step), [layers[i] for i in range(0, len(layers), step)])

    # 方法2: 旋转刻度标签(45度)并调整对齐方式
    plt.xticks(range(len(layers)), layers, rotation=45, ha='right', rotation_mode='anchor')

    # 方法3: 如果层数太多，可以只显示部分标签
    if len(layers) > 15:
        # 只显示奇数层或每5层显示一次
        show_layers = [layers[i] for i in range(len(layers)) if i % 5 == 0 or i == len(layers)-1]
        plt.xticks([layers.index(l) for l in show_layers], show_layers, rotation=45)

    # 绘制折线图
    plt.plot(layers, values, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.title(f'{metric} by Layer', fontsize=14)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 调整布局防止标签被截断
    plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(output_folder, f'{metric}_by_layer.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 可视化by_type - 柱状图
    by_type_data = data_dict['by_type']
    types = ['animals','cities','companies','elements','facts','inventions']
    values = [by_type_data[t][metric] for t in types]
    
    plt.figure(figsize=(10, 6))
    plt.bar(types, values)
    plt.title(f'{metric} by Type')
    plt.xlabel('Type')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{metric}_by_type.png'))
    plt.close()
    
    # 3. 可视化by_key - 热力图或折线图
    by_key_data = data_dict['by_key']
    
    # 提取所有layer和type
    layers = sorted(list(set([k.split()[1] for k in by_key_data.keys()])), key=lambda x: int(x))
    types = sorted(list(set([k.split()[2] for k in by_key_data.keys()])))
    
    # 创建DataFrame用于可视化
    df = pd.DataFrame(index=types, columns=layers)
    
    for key, metrics in by_key_data.items():
        _, layer, type_ = key.split()
        df.loc[type_, layer] = metrics[metric]
    
    df = df.astype(float)
    
    if by_key_plot_type == 'heatmap':
        plt.figure(figsize=(14, 8))  # 进一步增大图形宽度
        
        # 设置全局字体大小
        sns.set(font_scale=1.1)
        
        # 自动计算横轴刻度间隔（最多显示10个层标签）
        n_layers = len(df.columns)
        step = max(1, n_layers // 10)  # 如果层数>10，每隔step层显示一个
        
        # 生成热力图 - 移除了annot参数，使用从蓝到红的颜色映射
        ax = sns.heatmap(
            df,
            annot=False,  # 不显示数值
            cmap="coolwarm",  # 从蓝到红的颜色映射
            vmin=0,  # 设置最小值
            vmax=1,  # 设置最大值，使1对应最红色
            linewidths=0.5,
            linecolor='white',
            cbar_kws={"shrink": 0.8, "label": metric}
        )
        
        # 优化标题
        plt.title(f'{metric} by Layer and Type', fontsize=16, pad=20)
        
        # 调整X轴（Layer）刻度（仅显示部分层）
        x_labels = df.columns
        if n_layers > 10:
            # 仅显示部分层标签（如1, 5, 10, 15...）
            x_ticks = range(0, n_layers, step)
            x_labels = [df.columns[i] if i in x_ticks else '' for i in range(n_layers)]
        
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
        # 调整Y轴（Type）标签
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # 水平显示Type
        
        # 优化颜色条
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(metric, fontsize=12)
        
        # 调整布局，防止标签被截断
        plt.tight_layout()
        
        # 保存高质量图像
        plt.savefig(
            os.path.join(output_folder, f'{metric}_by_key_heatmap.png'),
            dpi=300,
            bbox_inches='tight',
            transparent=False
        )
        plt.close()
    elif by_key_plot_type == 'line':
        plt.figure(figsize=(12, 7))  # 增大图形尺寸
        
        # 绘制每条线（每个Type）
        for type_ in types:
            plt.plot(
                layers, 
                df.loc[type_], 
                marker='o', 
                markersize=8, 
                linewidth=2, 
                label=type_
            )
        
        # 优化标题和坐标轴标签
        plt.title(f'{metric} by Layer for Each Type', fontsize=14, pad=15)
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        
        # 自动调整X轴刻度（避免过多标签）
        n_layers = len(layers)
        if n_layers > 10:
            step = max(1, n_layers // 10)  # 最多显示10个刻度
            plt.xticks(
                range(0, n_layers, step),
                [layers[i] for i in range(0, n_layers, step)],
                rotation=45,
                ha='right'
            )
        else:
            plt.xticks(rotation=45, ha='right')  # 层数少时仅旋转标签
        
        # 优化图例（避免遮挡数据）
        plt.legend(
            bbox_to_anchor=(1.05, 1),  # 移到图外右上角
            loc='upper left',
            frameon=False,
            fontsize=10
        )
        
        # 添加网格线（提高可读性）
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 调整布局（防止标签被截断）
        plt.tight_layout()
        
        # 保存高质量图像
        plt.savefig(
            os.path.join(output_folder, f'{metric}_by_key_line.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
    else:
        raise ValueError("by_key_plot_type must be either 'heatmap' or 'line'")

# 可视化不同分类下的评估指标
visualize_metrics(data_dict, 'Acc', 'results/predic_information/visualization', by_key_plot_type='heatmap')
visualize_metrics(data_dict, 'Precision', 'results/predic_information/visualization', by_key_plot_type='heatmap')
visualize_metrics(data_dict, 'Recall', 'results/predic_information/visualization', by_key_plot_type='heatmap')
visualize_metrics(data_dict, 'F1', 'results/predic_information/visualization', by_key_plot_type='heatmap')

# 可视化不同分类下的评估指标
visualize_metrics(data_dict, 'Acc', 'results/predic_information/visualization', by_key_plot_type='line')
visualize_metrics(data_dict, 'Precision', 'results/predic_information/visualization', by_key_plot_type='line')
visualize_metrics(data_dict, 'Recall', 'results/predic_information/visualization', by_key_plot_type='line')
visualize_metrics(data_dict, 'F1', 'results/predic_information/visualization', by_key_plot_type='line')
