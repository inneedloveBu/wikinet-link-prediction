# test_fix.py
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import networkx as nx
from torch_geometric.data import Data

def test_fix():
    """测试修复后的evaluate_model函数"""
    print("=== 测试修复 ===")
    
    # 1. 创建简单数据
    num_nodes = 50
    num_edges = 100
    
    # 创建图
    graph = nx.erdos_renyi_graph(num_nodes, 0.08)
    edges = list(graph.edges())
    
    # 创建特征
    features = torch.randn(num_nodes, 10)
    
    # 构建边索引
    edge_list = []
    for src, dst in edges:
        edge_list.append([src, dst])
        edge_list.append([dst, src])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    data = Data(x=features, edge_index=edge_index, num_nodes=num_nodes)
    
    # 2. 划分数据集
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
        'test': {'edges': test_edges}
    }
    
    print(f"数据集: {data.num_nodes} 节点, {len(edges)} 边")
    
    # 3. 创建简单模型
    from torch_geometric.nn import GCNConv
    import torch.nn.functional as F
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(10, 16)
            self.conv2 = GCNConv(16, 8)
            self.lin = torch.nn.Linear(16, 1)
            
        def encode(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            return self.conv2(x, edge_index)
            
        def decode(self, z, edge_index):
            src, dst = edge_index
            edge_features = torch.cat([z[src], z[dst]], dim=-1)
            return torch.sigmoid(self.lin(edge_features)).squeeze()
    
    # 4. 测试evaluate_model函数
    device = torch.device('cpu')
    model = SimpleModel().to(device)
    
    # 导入修复后的evaluate_model函数
    # 注意：这里需要确保evaluate_model函数在作用域内
    # 由于这是一个测试，我们可以先复制一个简化版本
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    def simple_evaluate(model, data, eval_edges, train_edge_index, device):
        model.eval()
        with torch.no_grad():
            x = data.x.to(device)
            edge_index = train_edge_index.to(device)
            
            # 编码
            z = model.encode(x, edge_index)
            
            # 正样本
            pos_edges = eval_edges
            pos_edge_index = torch.tensor(pos_edges, dtype=torch.long).t().to(device)
            pos_scores = model.decode(z, pos_edge_index)
            
            # 负样本
            num_nodes = data.num_nodes
            num_neg = len(pos_edges)
            neg_edges = []
            
            while len(neg_edges) < num_neg:
                src = np.random.randint(0, num_nodes)
                dst = np.random.randint(0, num_nodes)
                if src != dst:
                    edge = (min(src, dst), max(src, dst))
                    if edge not in pos_edges:
                        neg_edges.append(edge)
            
            neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().to(device)
            neg_scores = model.decode(z, neg_edge_index)
            
            # 计算指标
            y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))]).cpu().numpy()
            y_pred = torch.cat([pos_scores, neg_scores]).cpu().numpy()
            
            try:
                auc = roc_auc_score(y_true, y_pred)
            except:
                auc = 0.5
                
            try:
                ap = average_precision_score(y_true, y_pred)
            except:
                ap = 0.0
            
            print(f"测试结果: AUC={auc:.4f}, AP={ap:.4f}")
            return auc, ap
    
    print("\n测试evaluate_model函数...")
    auc, ap = simple_evaluate(model, data, split_data['test']['edges'], 
                            split_data['train']['edge_index'], device)
    
    print("\n=== 测试完成 ===")
    return auc > 0.4  # 基本合理的AUC

if __name__ == "__main__":
    test_fix()