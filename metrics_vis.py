# visualization/metrics_vis.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC曲线
    if 'val_auc' in history:
        axes[1].plot(history['val_auc'], label='Val AUC', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Validation AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.show()

def plot_model_comparison(results, save_path=None):
    """绘制模型比较图"""
    models = list(results.keys())
    auc_scores = [results[m]['test_auc'] for m in models]
    ap_scores = [results[m]['test_ap'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # AUC柱状图
    x = range(len(models))
    axes[0].bar(x, auc_scores, color='skyblue', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel('AUC Score')
    axes[0].set_title('模型AUC对比')
    axes[0].set_ylim(0.5, 1.0)
    
    # 添加数值标签
    for i, v in enumerate(auc_scores):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # AP柱状图
    axes[1].bar(x, ap_scores, color='lightcoral', edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_ylabel('Average Precision')
    axes[1].set_title('模型AP对比')
    axes[1].set_ylim(0.5, 1.0)
    
    for i, v in enumerate(ap_scores):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()