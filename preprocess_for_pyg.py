# 首先，让我们修改preprocess_for_pyg.py中的保存方式
# cat > preprocess_for_pyg_fixed.py << 'EOF'
"""
将维基百科图数据转换为PyTorch Geometric格式（修复版本）
"""
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

class WikiGraphDataset:
    def __init__(self, edges_path, use_largest_component=True):
        """
        初始化数据集
        
        Args:
            edges_path: 边列表文件路径
            use_largest_component: 是否只使用最大连通分量
        """
        self.edges_path = edges_path
        self.use_largest_component = use_largest_component
        self.data = None
        
    def load_edges(self):
        """加载边数据"""
        print(f"加载边数据: {self.edges_path}")
        
        if self.edges_path.endswith('.parquet'):
            df = pd.read_parquet(self.edges_path)
        else:
            df = pd.read_csv(self.edges_path)
            
        print(f"加载了 {len(df)} 条边")
        return df
    
    def create_node_mapping(self, edges):
        """创建节点ID到连续索引的映射"""
        # 获取所有唯一节点
        all_nodes = pd.concat([edges['source'], edges['target']]).unique()
        
        # 创建映射
        node_to_idx = {int(node): int(idx) for idx, node in enumerate(all_nodes)}
        idx_to_node = {int(idx): int(node) for node, idx in node_to_idx.items()}
        
        print(f"创建了 {len(node_to_idx)} 个节点的映射")
        return node_to_idx, idx_to_node
    
    def prepare_pyg_data(self, edges, node_to_idx, add_features=True):
        """
        准备PyG数据
        
        Args:
            edges: 边DataFrame
            node_to_idx: 节点映射
            add_features: 是否添加节点特征
        """
        print("准备PyTorch Geometric数据...")
        
        # 转换边为索引
        edge_index = []
        for _, row in tqdm(edges.iterrows(), total=len(edges), desc="处理边"):
            src = node_to_idx[int(row['source'])]
            dst = node_to_idx[int(row['target'])]
            edge_index.append([src, dst])
            edge_index.append([dst, src])  # 无向图，添加双向边
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # 节点特征（这里先使用简单特征，后续可以改进）
        num_nodes = len(node_to_idx)
        if add_features:
            # 使用随机特征（实际项目中应该使用更好的特征）
            x = torch.randn(num_nodes, 128)  # 128维随机特征
            print(f"创建了 {num_nodes} 个节点的 {x.shape[1]} 维特征")
        else:
            x = torch.ones(num_nodes, 1)  # 占位符特征
        
        # 创建PyG数据对象
        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = num_nodes
        
        return data
    
    def split_edges_for_link_prediction(self, data, test_ratio=0.1, val_ratio=0.1):
        """
        为链路预测划分边
        
        Args:
            data: PyG数据对象
            test_ratio: 测试集比例
            val_ratio: 验证集比例
        """
        print("划分边用于链路预测...")
        
        # 获取所有边（无向图，每个边只有一次）
        edge_index = data.edge_index.t().tolist()
        
        # 转换为元组并去重
        edges_set = set()
        for src, dst in edge_index:
            if src < dst:  # 确保每个边只出现一次
                edges_set.add((int(src), int(dst)))
        
        edges = list(edges_set)
        print(f"有 {len(edges)} 条无向边")
        
        # 划分训练/验证/测试集
        train_edges, test_edges = train_test_split(
            edges, test_size=test_ratio, random_state=42
        )
        
        train_edges, val_edges = train_test_split(
            train_edges, test_size=val_ratio/(1-test_ratio), random_state=42
        )
        
        print(f"训练集: {len(train_edges)} 条边")
        print(f"验证集: {len(val_edges)} 条边")
        print(f"测试集: {len(test_edges)} 条边")
        
        # 将训练边转换为edge_index格式
        train_edge_index = []
        for src, dst in train_edges:
            train_edge_index.append([src, dst])
            train_edge_index.append([dst, src])
        
        train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).t().contiguous()
        
        # 保存划分结果
        return {
            'train': {'edges': train_edges, 'edge_index': train_edge_index},
            'val': {'edges': val_edges},
            'test': {'edges': test_edges},
            'all_edges': edges
        }
    
    def save_data_safely(self, data, split_data, node_to_idx, idx_to_node, output_dir):
        """安全保存数据（兼容weights_only=True）"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存PyG数据（使用torch.save，但保存为可序列化的格式）
        # 将Data对象转换为字典
        data_dict = {
            'x': data.x,
            'edge_index': data.edge_index,
            'num_nodes': data.num_nodes
        }
        torch.save(data_dict, os.path.join(output_dir, "wiki_graph_data.pt"))
        
        # 2. 保存节点映射（转换为Python基本类型）
        node_mappings = {
            'node_to_idx': {int(k): int(v) for k, v in node_to_idx.items()},
            'idx_to_node': {int(k): int(v) for k, v in idx_to_node.items()}
        }
        torch.save(node_mappings, os.path.join(output_dir, "node_mappings.pt"))
        
        # 3. 保存边划分（转换为Python基本类型）
        split_data_safe = {
            'train': {
                'edges': [(int(src), int(dst)) for src, dst in split_data['train']['edges']],
                'edge_index': split_data['train']['edge_index']
            },
            'val': {
                'edges': [(int(src), int(dst)) for src, dst in split_data['val']['edges']]
            },
            'test': {
                'edges': [(int(src), int(dst)) for src, dst in split_data['test']['edges']]
            },
            'all_edges': [(int(src), int(dst)) for src, dst in split_data['all_edges']]
        }
        torch.save(split_data_safe, os.path.join(output_dir, "edge_splits.pt"))
        
        # 4. 保存统计信息
        stats = {
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.shape[1] // 2,  # 无向边数
            'feature_dim': data.x.shape[1],
            'train_edges': len(split_data['train']['edges']),
            'val_edges': len(split_data['val']['edges']),
            'test_edges': len(split_data['test']['edges'])
        }
        
        with open(os.path.join(output_dir, "dataset_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
    
    def process(self):
        """处理数据并创建数据集"""
        print("=" * 50)
        print("开始处理维基百科图数据")
        print("=" * 50)
        
        # 1. 加载边
        edges = self.load_edges()
        
        # 2. 创建节点映射
        node_to_idx, idx_to_node = self.create_node_mapping(edges)
        
        # 3. 准备PyG数据
        data = self.prepare_pyg_data(edges, node_to_idx)
        
        # 4. 划分数据集
        split_data = self.split_edges_for_link_prediction(data)
        
        # 5. 保存所有数据
        output_dir = "data/processed/pyg_format"
        
        self.save_data_safely(data, split_data, node_to_idx, idx_to_node, output_dir)
        
        print("\n" + "=" * 50)
        print("数据处理完成!")
        print(f"保存到: {output_dir}")
        print(f"节点数: {data.num_nodes}")
        print(f"总边数: {data.edge_index.shape[1] // 2}")
        print(f"特征维度: {data.x.shape[1]}")
        print("=" * 50)
        
        return data, split_data, node_to_idx, idx_to_node

def main():
    # 使用最大连通分量的边
    edges_path = "data/processed/largest_component_edges.parquet"
    
    if not os.path.exists(edges_path):
        print(f"警告: {edges_path} 不存在，使用完整图")
        edges_path = "data/processed/wiki_graph_edges.parquet"
    
    # 创建数据集
    dataset = WikiGraphDataset(edges_path, use_largest_component=True)
    data, split_data, node_to_idx, idx_to_node = dataset.process()

if __name__ == "__main__":
    main()