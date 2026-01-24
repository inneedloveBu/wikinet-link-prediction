# simple_train.py
"""
简化版GNN训练脚本
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_simple_dataset(num_nodes=1000, num_edges=5000):
    """创建简单的合成数据集"""
    print("创建合成数据集...")
    
    # 使用BA模型创建更真实的图
    graph = nx.barabasi_albert_graph(num_nodes, 5)
    
    # 提取简单特征
    features = []
    for node in range(num_nodes):
        # 基本特征：度、邻居数、聚类系数
        degree = graph.degree(node)
        neighbors = list(graph.neighbors(node))
        
        # 简单特征向量
        feature = [
            degree / (num_nodes - 1),  # 归一化度
            len(neighbors) / (num_nodes - 1),  # 归一化邻居数
            1 if degree > np.mean([graph.degree(n) for n in graph.nodes()]) else 0,  # 是否高于平均度
            node / num_nodes,  # 节点ID归一化
            np.random.random(),  # 随机特征
        ]
        features.append(feature)
    
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    # 创建边索引
    edge_list = []
    for src, dst in graph.edges():
        edge_list.append([src, dst])
        edge_list.append([dst, src])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # 创建数据对象
    data = Data(x=features_tensor, edge_index=edge_index, num_nodes=num_nodes)
    
    # 划分数据集
    edges = list(graph.edges())
    edge_set = set()
    for src, dst in edges:
        if src < dst:
            edge_set.add((src, dst))
    
    edges = list(edge_set)
    
    train_edges, test_edges = train_test_split(edges, test_size=0.2, random_state=42)
    train_edges, val_edges = train_test_split(train_edges, test_size=0.125, random_state=42)
    
    train_edge_list = []
    for src, dst in train_edges:
        train_edge_list.append([src, dst])
        train_edge_list.append([dst, src])
    
    train_edge_index = torch.tensor(train_edge_list, dtype=torch.long).t().contiguous()
    
    split_data = {
        'train': {'edges': train_edges, 'edge_index': train_edge_index},
        'val': {'edges': val_edges},
        'test': {'edges': test_edges},
        'all_edges': edges
    }
    
    print(f"数据集: {num_nodes}节点, {len(edges)}边, {features_tensor.shape[1]}维特征")
    return data, split_data

class SimpleGraphSAGE(torch.nn.Module):
    """简单GraphSAGE模型"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
        # 边预测头
        self.lin = torch.nn.Linear(out_channels * 2, 1)
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        src, dst = edge_index
        edge_features = torch.cat([z[src], z[dst]], dim=-1)
        return torch.sigmoid(self.lin(edge_features)).squeeze()

def train_model(data, split_data, epochs=50):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = SimpleGraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=64,
        out_channels=32
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    x = data.x.to(device)
    train_edge_index = split_data['train']['edge_index'].to(device)
    train_edges = split_data['train']['edges']
    
    best_val_auc = 0
    history = {'train_loss': [], 'val_auc': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # 正样本
        pos_edge_index = torch.tensor(train_edges, dtype=torch.long).t().to(device)
        
        # 负采样
        num_neg = len(train_edges)
        neg_edges = []
        while len(neg_edges) < num_neg:
            src = torch.randint(0, data.num_nodes, (num_neg * 2,))
            dst = torch.randint(0, data.num_nodes, (num_neg * 2,))
            for s, d in zip(src, dst):
                if len(neg_edges) >= num_neg:
                    break
                if (s.item(), d.item()) not in train_edges:
                    neg_edges.append((s.item(), d.item()))
        
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().to(device)
        
        # 前向传播
        z = model.encode(x, train_edge_index)
        pos_scores = model.decode(z, pos_edge_index)
        neg_scores = model.decode(z, neg_edge_index)
        
        # 计算损失
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        history['train_loss'].append(loss.item())
        
        # 验证
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_edges = split_data['val']['edges']
                pos_edge_index = torch.tensor(val_edges, dtype=torch.long).t().to(device)
                
                # 生成验证负样本
                num_neg_val = len(val_edges)
                neg_edges_val = []
                while len(neg_edges_val) < num_neg_val:
                    src = torch.randint(0, data.num_nodes, (num_neg_val * 2,))
                    dst = torch.randint(0, data.num_nodes, (num_neg_val * 2,))
                    for s, d in zip(src, dst):
                        if len(neg_edges_val) >= num_neg_val:
                            break
                        if (s.item(), d.item()) not in val_edges:
                            neg_edges_val.append((s.item(), d.item()))
                
                neg_edge_index_val = torch.tensor(neg_edges_val, dtype=torch.long).t().to(device)
                
                # 计算分数
                pos_scores_val = model.decode(z, pos_edge_index)
                neg_scores_val = model.decode(z, neg_edge_index_val)
                
                # 计算指标
                y_true = torch.cat([torch.ones_like(pos_scores_val), torch.zeros_like(neg_scores_val)]).cpu().numpy()
                y_pred = torch.cat([pos_scores_val, neg_scores_val]).cpu().numpy()
                
                val_auc = roc_auc_score(y_true, y_pred)
                history['val_auc'].append(val_auc)
                
                print(f'Epoch {epoch:03d}: Loss={loss:.4f}, Val AUC={val_auc:.4f}')
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), 'simple_best_model.pt')
    
    # 测试
    model.load_state_dict(torch.load('simple_best_model.pt', map_location=device))
    model.eval()
    with torch.no_grad():
        test_edges = split_data['test']['edges']
        pos_edge_index = torch.tensor(test_edges, dtype=torch.long).t().to(device)
        
        # 生成测试负样本
        num_neg_test = len(test_edges)
        neg_edges_test = []
        while len(neg_edges_test) < num_neg_test:
            src = torch.randint(0, data.num_nodes, (num_neg_test * 2,))
            dst = torch.randint(0, data.num_nodes, (num_neg_test * 2,))
            for s, d in zip(src, dst):
                if len(neg_edges_test) >= num_neg_test:
                    break
                if (s.item(), d.item()) not in test_edges:
                    neg_edges_test.append((s.item(), d.item()))
        
        neg_edge_index_test = torch.tensor(neg_edges_test, dtype=torch.long).t().to(device)
        
        pos_scores_test = model.decode(z, pos_edge_index)
        neg_scores_test = model.decode(z, neg_edge_index_test)
        
        y_true = torch.cat([torch.ones_like(pos_scores_test), torch.zeros_like(neg_scores_test)]).cpu().numpy()
        y_pred = torch.cat([pos_scores_test, neg_scores_test]).cpu().numpy()
        
        test_auc = roc_auc_score(y_true, y_pred)
        test_ap = average_precision_score(y_true, y_pred)
        
        print(f"\n测试结果: AUC={test_auc:.4f}, AP={test_ap:.4f}")
    
    return history, test_auc, test_ap

def plot_results(history, test_auc, test_ap):
    """绘制结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], 'b-', label='训练损失')
    axes[0].set_xlabel('轮次')
    axes[0].set_ylabel('损失')
    axes[0].set_title('训练损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC曲线
    if 'val_auc' in history:
        x_vals = list(range(10, len(history['val_auc']) * 10 + 1, 10))
        axes[1].plot(x_vals, history['val_auc'], 'r-o', label='验证AUC')
        axes[1].set_xlabel('轮次')
        axes[1].set_ylabel('AUC')
        axes[1].set_title(f'验证AUC曲线 (测试AUC: {test_auc:.3f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=" * 60)
    print("简化版GNN训练系统")
    print("=" * 60)
    
    # 创建数据集
    data, split_data = create_simple_dataset(num_nodes=2000, num_edges=8000)
    
    # 训练模型
    history, test_auc, test_ap = train_model(data, split_data, epochs=50)
    
    # 绘制结果
    plot_results(history, test_auc, test_ap)
    
    print(f"\n最终性能: AUC={test_auc:.4f}, AP={test_ap:.4f}")

if __name__ == "__main__":
    main()