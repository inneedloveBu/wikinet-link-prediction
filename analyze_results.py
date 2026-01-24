# 创建 analyze_results.py

"""
分析训练结果和模型表现
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import os

def plot_comparison_results():
    """绘制模型比较结果"""
    results_path = "models/saved_models/experiment_results.csv"
    
    if not os.path.exists(results_path):
        print("实验结果文件不存在，请先运行比较实验")
        return
    
    df = pd.read_csv(results_path)
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # AUC比较
    ax = axes[0]
    x = range(len(df))
    auc_values = df['测试AUC'].astype(float)
    bars = ax.bar(x, auc_values, color=sns.color_palette("husl", len(x)))
    ax.set_xlabel('实验')
    ax.set_ylabel('AUC')
    ax.set_title('不同模型的AUC比较')
    ax.set_xticks(x)
    ax.set_xticklabels(df['实验名称'], rotation=45, ha='right')
    
    # 在柱子上添加数值
    for bar, value in zip(bars, auc_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # AP比较
    ax = axes[1]
    ap_values = df['测试AP'].astype(float)
    bars = ax.bar(x, ap_values, color=sns.color_palette("husl", len(x)))
    ax.set_xlabel('实验')
    ax.set_ylabel('Average Precision')
    ax.set_title('不同模型的AP比较')
    ax.set_xticks(x)
    ax.set_xticklabels(df['实验名称'], rotation=45, ha='right')
    
    # 在柱子上添加数值
    for bar, value in zip(bars, ap_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('models/saved_models/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("实验结果:")
    print(df.to_string(index=False))

def visualize_embeddings(model_path, data_path):
    """可视化学习到的节点嵌入"""
    
    # 加载模型
    from models.link_prediction import LinkPredictionModel
    
    checkpoint = torch.load(model_path, weights_only=False)
    
    # 需要知道模型的参数
    # 这里我们假设一个模型结构
    model = LinkPredictionModel(
        in_channels=5,  # 结构特征维度
        hidden_channels=128,
        out_channels=64,
        model_type='gcn',
        dropout=0.3
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载数据
    dataset = torch.load(data_path, weights_only=False)
    from torch_geometric.data import Data
    data = Data(
        x=dataset['data_dict']['x'],
        edge_index=dataset['data_dict']['edge_index'],
        num_nodes=dataset['data_dict']['num_nodes']
    )
    
    # 获取嵌入
    device = torch.device('cpu')
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        z = model.encode(x, edge_index)
        z = z.cpu().numpy()
    
    # 使用t-SNE降维
    print(f"嵌入维度: {z.shape}")
    
    if z.shape[0] > 1000:
        # 采样部分节点
        indices = np.random.choice(z.shape[0], 1000, replace=False)
        z_sampled = z[indices]
    else:
        z_sampled = z
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(z_sampled)
    
    # 绘制嵌入
    plt.figure(figsize=(10, 8))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=10, alpha=0.6, c=np.arange(len(z_2d)), cmap='viridis')
    plt.title('学习到的节点嵌入 (t-SNE可视化)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar(label='节点索引')
    plt.grid(True, alpha=0.3)
    plt.savefig('models/saved_models/learned_embeddings.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"可视化 {len(z_2d)} 个节点的嵌入")

def analyze_training_history():
    """分析训练历史"""
    history_paths = [
        ('随机特征', 'models/saved_models/training_history.json'),
        ('改进特征', 'models/saved_models/training_history_improved.json')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (label, path) in enumerate(history_paths):
        if os.path.exists(path):
            with open(path, 'r') as f:
                history = json.load(f)
            
            # 训练损失
            ax = axes[idx, 0]
            ax.plot(history['train_loss'], label='训练损失')
            ax.set_title(f'{label} - 训练损失')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 验证AUC
            ax = axes[idx, 1]
            if 'val_auc' in history:
                ax.plot(history['val_auc'], label='验证AUC', color='green')
                if 'test_auc' in history:
                    ax.axhline(y=history['test_auc'], color='red', linestyle='--', 
                              label=f"最终测试AUC: {history['test_auc']:.4f}")
                ax.set_title(f'{label} - 验证AUC')
                ax.set_xlabel('Epoch (每5轮)')
                ax.set_ylabel('AUC')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/saved_models/training_history_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("分析训练结果...")
    
    # 1. 绘制模型比较结果
    plot_comparison_results()
    
    # 2. 分析训练历史
    analyze_training_history()
    
    # 3. 可视化嵌入（如果模型和数据可用）
    model_path = "models/saved_models/best_model_improved.pt"
    data_path = "data/processed/improved_features/small_improved.pt"
    
    if os.path.exists(model_path) and os.path.exists(data_path):
        visualize_embeddings(model_path, data_path)
    else:
        print("模型或数据文件不存在，跳过嵌入可视化")

if __name__ == "__main__":
    main()
