# 创建 feature_engineering.py

"""
改进节点特征工程
"""
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from node2vec import Node2Vec  # 需要安装: pip install node2vec

class GraphFeatureExtractor:
    """图特征提取器"""
    
    def __init__(self, edges_df):
        """
        Args:
            edges_df: 包含'source'和'target'列的边DataFrame
        """
        self.edges_df = edges_df
        self.G = None
        self.node_features = {}
        
    def build_graph(self):
        """构建NetworkX图"""
        print("构建图...")
        self.G = nx.Graph()
        
        # 添加边
        edges = list(zip(self.edges_df['source'], self.edges_df['target']))
        self.G.add_edges_from(edges)
        
        print(f"图构建完成: {self.G.number_of_nodes()} 个节点, {self.G.number_of_edges()} 条边")
        return self.G
    
    def extract_structure_features(self):
        """提取图结构特征"""
        print("提取图结构特征...")
        
        if self.G is None:
            self.build_graph()
        
        features = {}
        
        # 1. 节点度
        degrees = dict(self.G.degree())
        
        # 2. 聚类系数
        clustering = nx.clustering(self.G)
        
        # 3. PageRank
        pagerank = nx.pagerank(self.G, alpha=0.85)
        
        # 4. 中心性度量
        try:
            # 接近中心性（计算较慢，可以抽样）
            closeness = nx.closeness_centrality(self.G)
        except:
            closeness = {node: 0 for node in self.G.nodes()}
        
        # 5. 邻域大小
        neighborhood_sizes = {node: len(list(self.G.neighbors(node))) for node in self.G.nodes()}
        
        # 归一化特征
        max_degree = max(degrees.values()) if degrees.values() else 1
        max_clustering = max(clustering.values()) if clustering.values() else 1
        max_pagerank = max(pagerank.values()) if pagerank.values() else 1
        max_closeness = max(closeness.values()) if closeness.values() else 1
        max_neighborhood = max(neighborhood_sizes.values()) if neighborhood_sizes.values() else 1
        
        for node in self.G.nodes():
            features[node] = np.array([
                degrees.get(node, 0) / max_degree,
                clustering.get(node, 0) / max_clustering,
                pagerank.get(node, 0) / max_pagerank,
                closeness.get(node, 0) / max_closeness,
                neighborhood_sizes.get(node, 0) / max_neighborhood
            ])
        
        print(f"提取了 {len(features)} 个节点的结构特征")
        return features
    
    def extract_embedding_features(self, dimensions=64, walk_length=30, num_walks=200):
        """使用Node2Vec提取嵌入特征"""
        print("使用Node2Vec提取嵌入特征...")
        
        if self.G is None:
            self.build_graph()
        
        try:
            # 创建Node2Vec模型
            node2vec = Node2Vec(
                self.G,
                dimensions=dimensions,
                walk_length=walk_length,
                num_walks=num_walks,
                workers=4,
                quiet=True
            )
            
            # 训练模型
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            
            # 获取嵌入
            embeddings = {}
            for node in self.G.nodes():
                if str(node) in model.wv:
                    embeddings[node] = model.wv[str(node)]
                else:
                    embeddings[node] = np.zeros(dimensions)
            
            print(f"Node2Vec嵌入完成，维度: {dimensions}")
            return embeddings
            
        except Exception as e:
            print(f"Node2Vec失败: {e}")
            # 返回零向量作为备用
            embeddings = {node: np.zeros(dimensions) for node in self.G.nodes()}
            return embeddings
    
    def extract_all_features(self, use_node2vec=False):
        """提取所有特征"""
        print("开始特征工程...")
        
        # 结构特征
        struct_features = self.extract_structure_features()
        
        # 嵌入特征（可选）
        embedding_features = None
        if use_node2vec:
            embedding_features = self.extract_embedding_features(dimensions=32)
        
        # 组合特征
        all_features = {}
        
        if embedding_features is not None:
            # 组合结构特征和嵌入特征
            for node in self.G.nodes():
                struct_feat = struct_features.get(node, np.zeros(5))
                embed_feat = embedding_features.get(node, np.zeros(32))
                combined = np.concatenate([struct_feat, embed_feat])
                all_features[node] = combined
            print(f"组合特征维度: {len(combined)} (结构:5 + 嵌入:32)")
        else:
            # 只使用结构特征
            all_features = struct_features
            print(f"结构特征维度: 5")
        
        # 转换为PyTorch张量
        nodes = sorted(all_features.keys())
        features_list = [all_features[node] for node in nodes]
        features_tensor = torch.tensor(features_list, dtype=torch.float32)
        
        print(f"特征张量形状: {features_tensor.shape}")
        
        return features_tensor, nodes

def create_improved_dataset(use_small=True, use_node2vec=False):
    """创建改进特征的数据集"""
    
    if use_small:
        # 创建小型数据集（用于快速实验）
        print("创建改进特征的小型数据集...")
        num_nodes = 1000
        num_edges = 5000
        
        # 随机生成边
        edges = set()
        while len(edges) < num_edges:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst and (src, dst) not in edges and (dst, src) not in edges:
                edges.add((src, dst))
        
        edges = list(edges)
        edges_df = pd.DataFrame(edges, columns=['source', 'target'])
        
    else:
        # 使用完整数据集
        print("加载完整数据集...")
        edges_path = "data/processed/largest_component_edges.parquet"
        edges_df = pd.read_parquet(edges_path)
        
        # 为了快速实验，采样部分数据
        if len(edges_df) > 50000:
            edges_df = edges_df.sample(n=50000, random_state=42)
            print(f"采样50000条边用于特征工程")
    
    # 提取特征
    extractor = GraphFeatureExtractor(edges_df)
    features_tensor, nodes = extractor.extract_all_features(use_node2vec=use_node2vec)
    
    # 创建节点ID到索引的映射
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # 创建边索引
    edge_list = []
    for _, row in edges_df.iterrows():
        src = row['source']
        dst = row['target']
        if src in node_to_idx and dst in node_to_idx:
            src_idx = node_to_idx[src]
            dst_idx = node_to_idx[dst]
            edge_list.append([src_idx, dst_idx])
            edge_list.append([dst_idx, src_idx])  # 无向图
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # 创建数据对象
    from torch_geometric.data import Data
    data = Data(x=features_tensor, edge_index=edge_index, num_nodes=len(nodes))
    
    print(f"数据集创建完成:")
    print(f"  节点数: {data.num_nodes}")
    print(f"  边数: {data.edge_index.shape[1] // 2}")
    print(f"  特征维度: {data.x.shape[1]}")
    
    # 划分数据集
    from sklearn.model_selection import train_test_split
    
    # 获取所有无向边
    edges_set = set()
    for i in range(0, edge_index.shape[1], 2):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        if src < dst:
            edges_set.add((src, dst))
    
    edges = list(edges_set)
    
    train_edges, test_edges = train_test_split(edges, test_size=0.2, random_state=42)
    train_edges, val_edges = train_test_split(train_edges, test_size=0.125, random_state=42)
    
    # 创建训练边索引
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
    
    print(f"  训练边: {len(train_edges)}")
    print(f"  验证边: {len(val_edges)}")
    print(f"  测试边: {len(test_edges)}")
    
    return data, split_data, node_to_idx

if __name__ == "__main__":
    # 测试特征提取
    data, split_data, node_to_idx = create_improved_dataset(use_small=True, use_node2vec=False)
    
    # 保存数据集
    import os
    output_dir = "data/processed/improved_features"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = {
        'data_dict': {
            'x': data.x,
            'edge_index': data.edge_index,
            'num_nodes': data.num_nodes
        },
        'split_data': split_data,
        'node_to_idx': node_to_idx
    }
    
    torch.save(dataset, os.path.join(output_dir, "small_improved.pt"))
    print(f"\n数据集已保存到: {output_dir}")
