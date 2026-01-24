# cat > visualize_results.py << 'EOF'
"""
可视化训练结果和模型预测
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import networkx as nx
from sklearn.manifold import TSNE

def plot_training_history(history_path):
    """绘制训练历史"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 训练损失
    axes[0].plot(history['train_loss'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # 验证AUC
    if 'val_auc' in history:
        axes[1].plot(history['val_auc'])
        axes[1].set_title('Validation AUC')
        axes[1].set_xlabel('Epoch (每5轮)')
        axes[1].set_ylabel('AUC')
        axes[1].grid(True, alpha=0.3)
    
    # 验证AP
    if 'val_ap' in history:
        axes[2].plot(history['val_ap'])
        axes[2].set_title('Validation AP')
        axes[2].set_xlabel('Epoch (每5轮)')
        axes[2].set_ylabel('Average Precision')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/saved_models/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印最终结果
    print("训练结果摘要:")
    if 'best_val_auc' in history:
        print(f"最佳验证AUC: {history['best_val_auc']:.4f} (第{history['best_epoch']}轮)")
    if 'test_auc' in history:
        print(f"测试集AUC: {history['test_auc']:.4f}")
    if 'test_ap' in history:
        print(f"测试集AP: {history['test_ap']:.4f}")

def visualize_node_embeddings(model, data, num_nodes=1000):
    """可视化节点嵌入"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        # 获取节点嵌入
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        z = model.encode(x, edge_index)
        
        # 采样部分节点
        if num_nodes < z.shape[0]:
            indices = torch.randperm(z.shape[0])[:num_nodes]
            z = z[indices].cpu().numpy()
        else:
            z = z.cpu().numpy()
        
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        z_2d = tsne.fit_transform(z)
        
        # 绘制
        plt.figure(figsize=(10, 8))
        plt.scatter(z_2d[:, 0], z_2d[:, 1], s=10, alpha=0.6)
        plt.title('Node Embeddings (t-SNE)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.savefig('models/saved_models/node_embeddings.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"可视化 {len(z_2d)} 个节点的嵌入")

if __name__ == "__main__":
    # 绘制训练历史
    history_path = "models/saved_models/training_history.json"
    if os.path.exists(history_path):
        plot_training_history(history_path)
    else:
        print(f"训练历史文件不存在: {history_path}")
