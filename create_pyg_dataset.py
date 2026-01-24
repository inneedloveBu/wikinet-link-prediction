# create_pyg_dataset.py
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
import networkx as nx

class WikiGraphDataset(Dataset):
    """维基百科图数据集"""
    
    def __init__(self, root, transform=None, pre_transform=None, sample_size=50000):
        self.sample_size = sample_size
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return ['wiki_topcats.csv']
    
    @property
    def processed_file_names(self):
        return ['data.pt', 'train_test_split.pt']
    
    def download(self):
        # 数据已手动下载
        pass
    
    def process(self):
        """处理原始数据"""
        print("处理原始数据...")
        
        # 读取数据
        df = pd.read_csv(self.raw_paths[0], sep='\t', header=None,
                        names=['source', 'target'], nrows=1000000)
        
        # 采样
        if len(df) > self.sample_size:
            df = df.sample(self.sample_size, random_state=42)
        
        # 重新编码节点
        all_nodes = pd.concat([df['source'], df['target']]).unique()
        node_mapping = {node: i for i, node in enumerate(all_nodes)}
        
        # 创建边索引
        edges = []
        for _, row in df.iterrows():
            src = node_mapping[row['source']]
            dst = node_mapping[row['target']]
            edges.append([src, dst])
            edges.append([dst, src])  # 无向图
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # 创建节点特征（使用随机特征或基于度的特征）
        num_nodes = len(all_nodes)
        
        # 方法1: 随机特征（简单）
        x = torch.randn(num_nodes, 128)
        
        # 方法2: 基于度的特征
        # degree = torch.zeros(num_nodes, 1)
        # for src, dst in edges:
        #     degree[src] += 1
        # x = degree
        
        # 创建Data对象
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        
        # 为链接预测任务创建训练/测试分割
        self.create_train_test_split(data)
        
        # 保存处理后的数据
        torch.save(data, self.processed_paths[0])
        print(f"数据集已保存: {self.processed_paths[0]}")
    
    def create_train_test_split(self, data):
        """为链接预测创建训练/测试分割"""
        print("创建训练/测试分割...")
        
        # 获取所有边
        edges = data.edge_index.t().numpy()
        
        # 分割正样本边
        train_edges, test_edges = train_test_split(
            edges, test_size=0.2, random_state=42
        )
        
        # 创建负样本边
        num_nodes = data.num_nodes
        num_neg_samples = len(test_edges)
        
        neg_edges = []
        while len(neg_edges) < num_neg_samples:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            # 确保不是自环且不是已有边
            if src != dst and not self.edge_exists(edges, src, dst):
                neg_edges.append([src, dst])
        
        neg_edges = np.array(neg_edges)
        
        # 创建标签
        train_labels = torch.ones(len(train_edges), dtype=torch.float)
        test_pos_labels = torch.ones(len(test_edges), dtype=torch.float)
        test_neg_labels = torch.zeros(len(neg_edges), dtype=torch.float)
        
        # 合并测试集
        test_edges_all = np.vstack([test_edges, neg_edges])
        test_labels_all = torch.cat([test_pos_labels, test_neg_labels])
        
        # 保存分割
        split_data = {
            'train_edges': torch.tensor(train_edges, dtype=torch.long),
            'train_labels': train_labels,
            'test_edges': torch.tensor(test_edges_all, dtype=torch.long),
            'test_labels': test_labels_all,
            'num_nodes': num_nodes
        }
        
        torch.save(split_data, self.processed_paths[1])
        print(f"训练测试分割已保存: {self.processed_paths[1]}")
    
    def edge_exists(self, edges, src, dst):
        """检查边是否存在"""
        # 简单线性搜索（对于大数据集需要优化）
        for e in edges:
            if (e[0] == src and e[1] == dst) or (e[0] == dst and e[1] == src):
                return True
        return False
    
    def len(self):
        return 1  # 我们只有一个图
    
    def get(self, idx):
        data = torch.load(self.processed_paths[0])
        split_data = torch.load(self.processed_paths[1])
        
        # 添加分割信息到data对象
        data.train_edges = split_data['train_edges']
        data.train_labels = split_data['train_labels']
        data.test_edges = split_data['test_edges']
        data.test_labels = split_data['test_labels']
        
        return data

def create_simple_dataset():
    """创建一个小型测试数据集"""
    print("创建小型测试数据集...")
    
    # 创建一个简单的图
    num_nodes = 1000
    num_edges = 5000
    
    # 随机边
    edges = []
    for _ in range(num_edges):
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst:
            edges.append([src, dst])
            edges.append([dst, src])  # 无向图
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # 节点特征
    x = torch.randn(num_nodes, 64)
    
    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    
    # 保存
    torch.save(data, "data/processed/test_graph.pt")
    print(f"测试图已保存: data/processed/test_graph.pt")
    print(f"  节点数: {data.num_nodes}")
    print(f"  边数: {data.num_edges}")
    
    return data

if __name__ == "__main__":
    # 首先尝试处理真实数据
    try:
        dataset = WikiGraphDataset(root='data', sample_size=50000)
        data = dataset.get(0)
        print("真实数据加载成功!")
        print(f"节点特征维度: {data.x.shape}")
        print(f"边索引形状: {data.edge_index.shape}")
        print(f"训练边数: {len(data.train_edges)}")
        print(f"测试边数: {len(data.test_edges)}")
    except Exception as e:
        print(f"处理真实数据失败: {e}")
        print("创建测试数据集...")
        create_simple_dataset()