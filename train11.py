"""
改进版GNN模型训练脚本 - 深度优化 (完整修复版)
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score
from sklearn.model_selection import train_test_split
import os
import json
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib
import warnings
import gzip
from tqdm import tqdm
from collections import Counter
warnings.filterwarnings('ignore')
import re
import random

# 设置中文字体
def setup_chinese_font():
    """设置中文字体显示"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        import matplotlib.font_manager as fm
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
        for font in chinese_fonts:
            if any(font.lower() in f.lower() for f in available_fonts):
                plt.rcParams['font.sans-serif'] = [font]
                print(f"✓ 使用中文字体: {font}")
                break
    except Exception as e:
        print(f"字体设置失败: {e}")
        print("使用默认字体")
    
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    return True

setup_chinese_font()

# ==================== 改进的数据加载器 ====================

class ImprovedWikiDataLoader:
    """改进版维基百科数据加载器"""
    
    @staticmethod
    def load_and_filter_data(target_nodes=150, target_edges=700, max_lines=50000, save_cleaned_data=True):
        """加载并过滤数据，直接构建目标大小的图"""
        print(f"加载WikiLinks数据，目标: {target_nodes} 节点, {target_edges} 边")
        
        data_path = "data/raw/enwiki.wikilink_graph.2018-03-01.csv.gz"
        
        if not os.path.exists(data_path):
            print(f"错误: 数据文件不存在: {data_path}")
            print("请确保已下载数据集并放在正确位置")
            return None
        
        try:
            # 读取所有边
            edges = []
            with gzip.open(data_path, 'rt', encoding='utf-8') as f:
                next(f)  # 跳过标题
                
                # 读取足够多的边以便采样
                for i, line in enumerate(tqdm(f, desc="读取边", total=max_lines)):
                    if i >= max_lines:
                        break
                    
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        src = parts[0].strip('"')
                        dst = parts[1].strip('"')
                        
                        if src != dst and src and dst:  # 确保不是自环且非空
                            edges.append((src, dst))
            
            print(f"读取完成: {len(edges):,} 条原始边")
            
            # 检查原始数据
            print("\n=== 原始数据检查 ===")
            print(f"原始边数: {len(edges)}")
            
            # 直接从边构建图
            G = nx.Graph()
            G.add_edges_from(edges)
            
            print(f"\n原始图统计:")
            print(f"  节点数: {G.number_of_nodes()}")
            print(f"  边数: {G.number_of_edges()}")
            
            # 检查孤立节点
            isolated_nodes = [node for node, deg in G.degree() if deg == 0]
            print(f"  孤立节点数: {len(isolated_nodes)}")
            
            # 提取最大连通分量
            if not nx.is_connected(G):
                print("\n提取最大连通分量...")
                components = list(nx.connected_components(G))
                print(f"  连通分量数: {len(components)}")
                component_sizes = [len(c) for c in components]
                print(f"  连通分量大小: {component_sizes[:10]}{'...' if len(component_sizes) > 10 else ''}")
                
                largest_component = max(components, key=len)
                G = G.subgraph(largest_component).copy()
                print(f"  最大连通分量: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
            
            # 如果图太大，使用BFS采样得到目标大小的子图
            if G.number_of_nodes() > target_nodes:
                print(f"\n采样 {target_nodes} 个节点的子图...")
                G = ImprovedWikiDataLoader._bfs_sampling(G, target_nodes)
            
            print(f"\n采样后图统计:")
            print(f"  节点数: {G.number_of_nodes()}")
            print(f"  边数: {G.number_of_edges()}")
            
            # 检查采样后的节点度分布
            degrees = [d for _, d in G.degree()]
            print(f"  平均度: {np.mean(degrees):.2f}")
            print(f"  最大度: {max(degrees)}")
            print(f"  最小度: {min(degrees)}")
            
            # 如果边数太少，添加一些随机边（数据增强）
            if G.number_of_edges() < target_edges * 0.8:
                print(f"\n边数不足 ({G.number_of_edges()}/{target_edges})，进行数据增强...")
                G = ImprovedWikiDataLoader._add_random_edges(G, target_edges)
            
            # 转换为DataFrame格式
            edges_df = pd.DataFrame(list(G.edges()), columns=['source', 'target'])
            
            # 保存清洗后的数据到文件
            if save_cleaned_data:
                ImprovedWikiDataLoader._save_cleaned_data(edges_df, G)
            
            return edges_df
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def _bfs_sampling(graph, target_nodes):
        """使用BFS采样子图"""
        nodes = list(graph.nodes())
        
        # 选择度最高的节点作为起点
        degrees = dict(graph.degree())
        if not degrees:
            # 如果图为空，随机选择起点
            start_node = random.choice(nodes) if nodes else None
        else:
            start_node = max(degrees.items(), key=lambda x: x[1])[0]
        
        if start_node is None:
            return graph
        
        # BFS遍历
        visited = set([start_node])
        queue = [start_node]
        
        while queue and len(visited) < target_nodes:
            current = queue.pop(0)
            neighbors = list(graph.neighbors(current))
            
            # 随机打乱邻居，增加多样性
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in visited and len(visited) < target_nodes:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        print(f"  BFS采样完成: {len(visited)} 个节点")
        
        # 创建子图
        subgraph = graph.subgraph(visited).copy()
        return subgraph
    
    @staticmethod
    def _add_random_edges(graph, target_edges):
        """添加随机边（数据增强）"""
        current_edges = graph.number_of_edges()
        nodes = list(graph.nodes())
        
        # 计算需要添加的边数
        edges_to_add = max(0, target_edges - current_edges)
        if edges_to_add == 0:
            return graph
        
        print(f"  添加 {edges_to_add} 条随机边...")
        
        added_edges = 0
        attempts = 0
        max_attempts = edges_to_add * 10
        
        existing_edges = set(graph.edges())
        
        # 获取节点度用于概率采样
        degrees = dict(graph.degree())
        total_degree = sum(degrees.values())
        
        while added_edges < edges_to_add and attempts < max_attempts:
            if total_degree > 0 and len(degrees) > 0:
                # 按节点度概率选择节点
                node_probs = {node: deg/total_degree for node, deg in degrees.items()}
                nodes_list = list(node_probs.keys())
                probs = list(node_probs.values())
                
                if len(nodes_list) < 2:
                    break
                    
                src = np.random.choice(nodes_list, p=probs)
                dst = np.random.choice(nodes_list, p=probs)
            else:
                if len(nodes) < 2:
                    break
                src = random.choice(nodes)
                dst = random.choice(nodes)
            
            if src != dst and (src, dst) not in existing_edges and (dst, src) not in existing_edges:
                graph.add_edge(src, dst)
                existing_edges.add((src, dst))
                existing_edges.add((dst, src))  # 无向图
                added_edges += 1
                
                # 更新度
                degrees[src] = degrees.get(src, 0) + 1
                degrees[dst] = degrees.get(dst, 0) + 1
                total_degree += 2
            
            attempts += 1
        
        print(f"  成功添加 {added_edges} 条随机边")
        return graph
    
    @staticmethod
    def _save_cleaned_data(edges_df, graph):
        """保存清洗后的数据到文件"""
        print("\n=== 保存清洗后的数据 ===")
        
        # 创建输出目录
        output_dir = "data/cleaned"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存边数据
        edges_path = os.path.join(output_dir, "cleaned_edges.txt")
        with open(edges_path, 'w', encoding='utf-8') as f:
            f.write("清洗后的边数据 (共{}条边)\n".format(len(edges_df)))
            f.write("=" * 80 + "\n")
            for i, row in edges_df.iterrows():
                f.write(f"{i+1}. {row['source']} <-> {row['target']}\n")
        print(f"✓ 边数据已保存到: {edges_path}")
        
        # 2. 保存节点数据
        nodes_path = os.path.join(output_dir, "cleaned_nodes.txt")
        all_nodes = pd.concat([edges_df['source'], edges_df['target']]).unique()
        with open(nodes_path, 'w', encoding='utf-8') as f:
            f.write("清洗后的节点数据 (共{}个节点)\n".format(len(all_nodes)))
            f.write("=" * 80 + "\n")
            for i, node in enumerate(all_nodes):
                f.write(f"{i+1}. {node}\n")
        print(f"✓ 节点数据已保存到: {nodes_path}")
        
        # 3. 保存图统计信息
        stats_path = os.path.join(output_dir, "graph_stats.json")
        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1) // 2),
            "avg_degree": np.mean([d for _, d in graph.degree()]),
            "max_degree": max([d for _, d in graph.degree()]),
            "min_degree": min([d for _, d in graph.degree()]),
            "is_connected": nx.is_connected(graph)
        }
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"✓ 图统计信息已保存到: {stats_path}")
        
        # 4. 保存为CSV格式以便后续使用
        edges_csv_path = os.path.join(output_dir, "cleaned_edges.csv")
        edges_df.to_csv(edges_csv_path, index=False, encoding='utf-8')
        print(f"✓ CSV格式边数据已保存到: {edges_csv_path}")
    
    @staticmethod
    def create_synthetic_dataset(num_nodes=150, num_edges=733):
        """创建合成数据集（当真实数据不可用时）"""
        print(f"创建合成数据集: {num_nodes} 节点, {num_edges} 边")
        
        # 使用BA模型（无标度网络）
        m = max(1, num_edges // num_nodes)
        graph = nx.barabasi_albert_graph(num_nodes, m)
        
        # 确保边数接近目标
        current_edges = graph.number_of_edges()
        if current_edges < num_edges:
            # 添加随机边
            nodes = list(graph.nodes())
            existing_edges = set(graph.edges())
            
            for _ in range(num_edges - current_edges):
                src, dst = random.sample(nodes, 2)
                if (src, dst) not in existing_edges and (dst, src) not in existing_edges:
                    graph.add_edge(src, dst)
                    existing_edges.add((src, dst))
                    existing_edges.add((dst, src))
        
        # 转换为DataFrame格式，节点名改为字符串
        edges = list(graph.edges())
        edges_df = pd.DataFrame(edges, columns=['source', 'target'])
        edges_df['source'] = edges_df['source'].astype(str)
        edges_df['target'] = edges_df['target'].astype(str)
        
        print("合成数据集创建完成")
        
        # 保存合成数据
        ImprovedWikiDataLoader._save_cleaned_data(edges_df, graph)
        
        return edges_df

# ==================== 改进的特征提取器 ====================

class ImprovedFeatureExtractor:
    """改进版特征提取器"""
    
    def __init__(self, graph):
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
    
    def extract_structural_features(self):
        """提取图结构特征"""
        features = {}
        
        # 基础结构特征
        print("计算结构特征...")
        degrees = dict(self.graph.degree())
        
        # 计算聚类系数
        try:
            clustering = nx.clustering(self.graph)
        except:
            clustering = {node: 0 for node in self.graph.nodes()}
        
        # 只对较小的图计算这些中心性
        if self.num_nodes < 500:
            try:
                betweenness = nx.betweenness_centrality(self.graph, k=min(100, self.num_nodes))
                pagerank = nx.pagerank(self.graph, alpha=0.85)
            except:
                betweenness = {node: 0 for node in self.graph.nodes()}
                pagerank = {node: 1/self.num_nodes for node in self.graph.nodes()}
        else:
            betweenness = {node: 0 for node in self.graph.nodes()}
            pagerank = {node: 1/self.num_nodes for node in self.graph.nodes()}
        
        # 邻居统计
        neighbor_stats = {}
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            neighbor_degrees = [degrees.get(n, 0) for n in neighbors]
            neighbor_stats[node] = {
                'mean_deg': np.mean(neighbor_degrees) if neighbor_degrees else 0,
                'std_deg': np.std(neighbor_degrees) if len(neighbor_degrees) > 1 else 0,
                'max_deg': max(neighbor_degrees) if neighbor_degrees else 0,
                'min_deg': min(neighbor_degrees) if neighbor_degrees else 0,
                'num_neighbors': len(neighbors)
            }
        
        # 计算最大值用于归一化
        max_degree = max(degrees.values()) if degrees else 1
        max_neighbor_mean = max([stats['mean_deg'] for stats in neighbor_stats.values()]) if neighbor_stats else 1
        
        # 构建特征向量 (10维)
        for node in self.graph.nodes():
            deg = degrees.get(node, 0)
            clust = clustering.get(node, 0)
            bet = betweenness.get(node, 0)
            pr = pagerank.get(node, 0)
            stats = neighbor_stats.get(node, {})
            
            feature_vector = np.array([
                deg / max_degree if max_degree > 0 else 0,  # 归一化度
                clust,  # 聚类系数
                bet,  # 介数中心性
                pr,  # PageRank
                stats.get('mean_deg', 0) / max(max_neighbor_mean, 1),  # 平均邻居度
                stats.get('std_deg', 0) / 100,  # 邻居度标准差
                stats.get('max_deg', 0) / 100,  # 最大邻居度
                stats.get('min_deg', 0) / 100,  # 最小邻居度
                stats.get('num_neighbors', 0) / self.num_nodes,  # 邻居比例
                1 if deg > np.mean(list(degrees.values())) else 0  # 是否高于平均度
            ])
            
            features[node] = feature_vector
        
        return features
    
    def extract_content_features(self):
        """提取基于节点内容的特征"""
        features = {}
        
        # 从节点字符串中提取特征
        for node in self.graph.nodes():
            node_str = str(node)
            
            # 字符串长度特征
            length = len(node_str)
            
            # 数字特征
            numbers = re.findall(r'\d+', node_str)
            num_count = len(numbers)
            has_numbers = 1 if numbers else 0
            
            # 大写字母比例
            upper_ratio = sum(1 for c in node_str if c.isupper()) / max(1, length)
            
            # 特殊字符特征
            special_chars = sum(1 for c in node_str if not c.isalnum() and c != ' ')
            special_ratio = special_chars / max(1, length)
            
            # 单词特征
            words = node_str.split()
            word_count = len(words)
            
            feature_vector = np.array([
                length / 100,  # 归一化长度
                num_count / 10,  # 数字数量
                has_numbers,  # 是否有数字
                upper_ratio,  # 大写字母比例
                special_ratio,  # 特殊字符比例
                word_count / 5,  # 单词数量
                1 if 'http' in node_str.lower() else 0,  # 是否URL
                1 if 'wiki' in node_str.lower() else 0,  # 是否wiki相关
                np.random.random() * 0.1,  # 少量随机噪声
                (hash(node_str) % 100) / 100  # 哈希特征
            ])
            
            features[node] = feature_vector
        
        return features
    
    def extract_all_features(self):
        """提取所有特征"""
        print("提取改进特征...")
        
        # 提取结构特征
        structural_features = self.extract_structural_features()
        
        # 提取内容特征
        content_features = self.extract_content_features()
        
        # 合并特征
        combined_features = {}
        for node in self.graph.nodes():
            struct = structural_features.get(node, np.zeros(10))
            content = content_features.get(node, np.zeros(10))
            combined = np.concatenate([struct, content])  # 总共20维
            
            # 标准化
            combined = (combined - np.mean(combined)) / (np.std(combined) + 1e-8)
            combined_features[node] = combined
        
        print(f"特征提取完成: {len(combined_features)} 节点, {combined.shape[0]} 维特征")
        
        # 显示特征统计
        all_features = np.array(list(combined_features.values()))
        print(f"特征统计: 均值={np.mean(all_features):.3f}, 标准差={np.std(all_features):.3f}")
        print(f"特征范围: [{np.min(all_features):.3f}, {np.max(all_features):.3f}]")
        
        return combined_features

# ==================== 改进的模型 ====================

class SimpleLinkPredictionModel(torch.nn.Module):
    """简化但有效的链路预测模型"""
    
    def __init__(self, in_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
        # 编码器
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim * 2),
            torch.nn.BatchNorm1d(hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            
            torch.nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 边预测器 - 多种交互方式
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear((hidden_dim // 2) * 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def encode(self, x):
        """编码节点特征"""
        return self.encoder(x)
    
    def decode(self, u_hidden, v_hidden):
        """解码边存在概率"""
        # 多种交互方式
        diff = torch.abs(u_hidden - v_hidden)
        mul = u_hidden * v_hidden
        concat = torch.cat([u_hidden, v_hidden, diff, mul], dim=1)
        return self.edge_predictor(concat).squeeze()  # 返回logits，不经过sigmoid
    
    def forward(self, x, edge_index):
        """前向传播"""
        z = self.encode(x)
        src, dst = edge_index
        u_hidden = z[src]
        v_hidden = z[dst]
        logits = self.decode(u_hidden, v_hidden)
        return torch.sigmoid(logits)  # 返回概率

# ==================== 改进的训练策略 ====================

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=15, min_delta=0.001, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            if self.restore_best:
                self.best_state = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if self.restore_best:
                self.best_state = model.state_dict().copy()
        
        return self.early_stop
    
    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)

def generate_hard_negatives(graph, positive_edges, num_samples, node_to_idx, hardness_level='medium'):
    """生成困难负样本"""
    num_nodes = graph.number_of_nodes()
    negative_edges = []
    positive_set = set(positive_edges)
    
    # 获取节点度
    degrees = dict(graph.degree())
    node_list = list(graph.nodes())
    
    if hardness_level == 'easy':
        # 随机负采样
        attempts = 0
        while len(negative_edges) < num_samples and attempts < num_samples * 10:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src == dst:
                attempts += 1
                continue
            
            edge = (min(src, dst), max(src, dst))
            if edge not in positive_set:
                negative_edges.append(edge)
                positive_set.add(edge)
            attempts += 1
    
    elif hardness_level == 'medium':
        # 基于共同邻居的困难负采样
        adj_list = {i: set() for i in range(num_nodes)}
        for src, dst in positive_edges:
            adj_list[src].add(dst)
            adj_list[dst].add(src)
        
        attempts = 0
        while len(negative_edges) < num_samples and attempts < num_samples * 20:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src == dst:
                attempts += 1
                continue
            
            # 有共同邻居但不是现有边
            common_neighbors = len(adj_list[src] & adj_list[dst])
            if common_neighbors > 0 and common_neighbors < 3:  # 适度困难
                edge = (min(src, dst), max(src, dst))
                if edge not in positive_set:
                    negative_edges.append(edge)
                    positive_set.add(edge)
            
            attempts += 1
    
    elif hardness_level == 'hard':
        # 基于度的相似性
        degree_probs = np.array([degrees.get(node_list[i], 1) for i in range(num_nodes)])
        degree_probs = degree_probs / degree_probs.sum()
        
        attempts = 0
        while len(negative_edges) < num_samples and attempts < num_samples * 30:
            # 按度分布采样
            src = np.random.choice(num_nodes, p=degree_probs)
            dst = np.random.choice(num_nodes, p=degree_probs)
            if src == dst:
                attempts += 1
                continue
            
            src_deg = degrees.get(node_list[src], 1)
            dst_deg = degrees.get(node_list[dst], 1)
            
            # 度相似但不是现有边
            if abs(src_deg - dst_deg) < max(src_deg, dst_deg) * 0.3:
                edge = (min(src, dst), max(src, dst))
                if edge not in positive_set:
                    negative_edges.append(edge)
                    positive_set.add(edge)
            
            attempts += 1
    
    # 补充随机负样本
    if len(negative_edges) < num_samples:
        additional = num_samples - len(negative_edges)
        additional_edges = generate_hard_negatives(
            graph, positive_edges, additional, node_to_idx, hardness_level='easy'
        )
        negative_edges.extend(additional_edges)
    
    return negative_edges[:num_samples]

def improved_train_link_prediction(model, data, split_data, optimizer, device, 
                                  num_epochs=300, num_neg_samples=3, 
                                  hardness_level='medium'):
    """改进的训练函数"""
    # 准备数据
    x = data.x.to(device)
    train_edge_index = split_data['train']['edge_index'].to(device)
    train_pos_edges = split_data['train']['edges']
    val_edges = split_data['val']['edges']
    test_edges = split_data['test']['edges']
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_auc': [],
        'val_ap': [],
        'best_val_auc': 0,
        'best_epoch': 0
    }
    
    # 早停机制
    early_stopping = EarlyStopping(patience=20, min_delta=0.001, restore_best=True)
    
    # 学习率调度器 - 修复verbose参数问题
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
    except TypeError:
        # 某些PyTorch版本不支持verbose参数
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
    
    print("\n开始改进训练...")
    print(f"训练边: {len(train_pos_edges)}")
    print(f"验证边: {len(val_edges)}")
    print(f"测试边: {len(test_edges)}")
    
    # 临时修改：添加一个辅助函数来获取logits
    def get_logits(model, x, edge_index):
        """获取模型的logits输出"""
        z = model.encode(x)
        src, dst = edge_index
        u_hidden = z[src]
        v_hidden = z[dst]
        
        # 多种交互方式
        diff = torch.abs(u_hidden - v_hidden)
        mul = u_hidden * v_hidden
        concat = torch.cat([u_hidden, v_hidden, diff, mul], dim=1)
        return model.edge_predictor(concat).squeeze()
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        
        # 生成负样本
        num_neg_edges = len(train_pos_edges) * num_neg_samples
        neg_edges = generate_hard_negatives(
            data.graph, train_pos_edges, num_neg_edges, 
            data.node_to_idx, hardness_level
        )
        
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().to(device)
        pos_edge_index = torch.tensor(train_pos_edges, dtype=torch.long).t().to(device)
        
        # 合并正负样本
        all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        labels = torch.cat([
            torch.ones(pos_edge_index.size(1), device=device),
            torch.zeros(neg_edge_index.size(1), device=device)
        ])
        
        # 获取logits
        logits = get_logits(model, x, all_edge_index)
        
        # 计算损失 - 使用带权重的二值交叉熵
        pos_weight = torch.tensor([num_neg_edges / max(1, len(train_pos_edges))], device=device)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        history['train_loss'].append(loss.item())
        
        # 验证
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # 验证集评估
                val_pos_edge_index = torch.tensor(val_edges, dtype=torch.long).t().to(device)
                
                # 生成验证负样本
                num_val_neg = len(val_edges)
                val_neg_edges = generate_hard_negatives(
                    data.graph, list(set(train_pos_edges) | set(val_edges)), 
                    num_val_neg, data.node_to_idx, 'easy'
                )
                val_neg_edge_index = torch.tensor(val_neg_edges, dtype=torch.long).t().to(device)
                
                # 计算分数（使用模型的forward方法，返回概率）
                val_pos_scores = model(x, val_pos_edge_index)
                val_neg_scores = model(x, val_neg_edge_index)
                
                # 计算指标
                y_true = torch.cat([torch.ones_like(val_pos_scores), 
                                  torch.zeros_like(val_neg_scores)]).cpu().numpy()
                y_pred = torch.cat([val_pos_scores, val_neg_scores]).cpu().numpy()
                
                try:
                    val_auc = roc_auc_score(y_true, y_pred)
                except:
                    val_auc = 0.5
                
                try:
                    val_ap = average_precision_score(y_true, y_pred)
                except:
                    val_ap = 0.0
                
                history['val_auc'].append(val_auc)
                history['val_ap'].append(val_ap)
                
                print(f'Epoch {epoch:03d}: Loss={loss:.4f}, Val AUC={val_auc:.4f}, Val AP={val_ap:.4f}')
                
                # 更新学习率
                scheduler.step(val_auc)
                
                # 早停检查
                if val_auc > history['best_val_auc']:
                    history['best_val_auc'] = val_auc
                    history['best_epoch'] = epoch
                    
                    # 保存最佳模型
                    os.makedirs('models', exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_auc': val_auc,
                        'val_ap': val_ap,
                    }, 'models/best_improved_model.pt')
                
                # 早停
                if early_stopping(val_auc, model):
                    print(f"早停在第 {epoch} 轮")
                    early_stopping.restore(model)
                    break
    
    # 最终测试
    print("\n=== 最终测试 ===")
    test_results = comprehensive_evaluation(model, data, test_edges, 
                                          split_data['train']['edge_index'], device)
    
    history.update(test_results)
    
    # 保存训练历史（修复JSON序列化问题）
    save_training_history(history)
    
    return model, history

def comprehensive_evaluation(model, data, test_pos_edges, train_edge_index, device):
    """综合评估模型性能"""
    model.eval()
    
    with torch.no_grad():
        x = data.x.to(device)
        
        # 准备正样本
        pos_edge_index = torch.tensor(test_pos_edges, dtype=torch.long).t().to(device)
        pos_scores = model(x, pos_edge_index)
        
        # 生成负样本
        num_neg = len(test_pos_edges)
        train_edges_set = set()
        train_edges_np = train_edge_index.t().cpu().numpy()
        for src, dst in train_edges_np:
            train_edges_set.add((min(src, dst), max(src, dst)))
        
        neg_edges = []
        attempts = 0
        while len(neg_edges) < num_neg and attempts < num_neg * 10:
            src = np.random.randint(0, data.num_nodes)
            dst = np.random.randint(0, data.num_nodes)
            if src == dst:
                attempts += 1
                continue
            
            edge = (min(src, dst), max(src, dst))
            if edge not in test_pos_edges and edge not in train_edges_set:
                neg_edges.append(edge)
            attempts += 1
        
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().to(device)
        neg_scores = model(x, neg_edge_index)
        
        # 计算所有指标
        y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu().numpy()
        y_pred = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        
        # AUC
        try:
            auc_score = roc_auc_score(y_true, y_pred)
        except:
            auc_score = 0.5
        
        # AP
        try:
            ap_score = average_precision_score(y_true, y_pred)
        except:
            ap_score = 0.0
        
        # 计算最佳阈值
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # 排除最后一个元素
        best_threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5
        best_f1 = f1_scores[best_idx]
        
        # PR-AUC
        pr_auc = auc(recall, precision)
        
        # 准确率
        predictions = (y_pred > best_threshold).astype(int)
        accuracy = np.mean(predictions == y_true)
        
        print(f"测试结果:")
        print(f"  AUC: {auc_score:.4f}")
        print(f"  AP: {ap_score:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")
        print(f"  F1分数: {best_f1:.4f} (阈值={best_threshold:.3f})")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  正样本分数: [{pos_scores.min():.3f}, {pos_scores.max():.3f}], 均值={pos_scores.mean():.3f}")
        print(f"  负样本分数: [{neg_scores.min():.3f}, {neg_scores.max():.3f}], 均值={neg_scores.mean():.3f}")
        
        # 转换为Python原生类型
        return {
            'test_auc': float(auc_score),
            'test_ap': float(ap_score),
            'test_pr_auc': float(pr_auc),
            'test_f1': float(best_f1),
            'test_accuracy': float(accuracy),
            'best_threshold': float(best_threshold),
            'pos_scores_stats': {
                'min': float(pos_scores.min().item()),
                'max': float(pos_scores.max().item()),
                'mean': float(pos_scores.mean().item())
            },
            'neg_scores_stats': {
                'min': float(neg_scores.min().item()),
                'max': float(neg_scores.max().item()),
                'mean': float(neg_scores.mean().item())
            }
        }

def save_training_history(history):
    """保存训练历史，处理序列化问题"""
    # 转换所有值为可序列化的格式
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    serializable_history = convert_to_serializable(history)
    
    os.makedirs('models', exist_ok=True)
    with open('models/improved_training_history.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_history, f, indent=2, ensure_ascii=False)
    
    print("✓ 训练历史已保存到: models/improved_training_history.json")

# ==================== 深度诊断函数 ====================

def deep_diagnosis(graph, features_tensor, edges_idx, node_list):
    """深度诊断图数据和特征"""
    print("\n=== 深度诊断 ===")
    
    # 1. 图结构分析
    print("1. 图结构分析:")
    print(f"   节点数: {graph.number_of_nodes()}")
    print(f"   边数: {graph.number_of_edges()}")
    
    # 边密度
    possible_edges = graph.number_of_nodes() * (graph.number_of_nodes() - 1) // 2
    density = graph.number_of_edges() / possible_edges if possible_edges > 0 else 0
    print(f"   边密度: {density:.6f} ({density*100:.4f}%)")
    
    # 聚类系数
    try:
        clustering_coeff = nx.average_clustering(graph)
        print(f"   平均聚类系数: {clustering_coeff:.4f}")
    except:
        print("   平均聚类系数: 无法计算")
    
    # 同配性系数
    try:
        assortativity = nx.degree_assortativity_coefficient(graph)
        print(f"   同配性系数: {assortativity:.4f}")
    except:
        print("   同配性系数: 无法计算")
    
    # 度分布
    degrees = [d for _, d in graph.degree()]
    if degrees:
        print(f"   度分布: 均值={np.mean(degrees):.2f}, 标准差={np.std(degrees):.2f}")
        print(f"   最大度: {max(degrees)}, 最小度: {min(degrees)}")
    else:
        print("   度分布: 无数据")
    
    # 2. 特征分析
    print("\n2. 特征分析:")
    print(f"   特征形状: {features_tensor.shape}")
    print(f"   特征范围: [{features_tensor.min():.3f}, {features_tensor.max():.3f}]")
    print(f"   特征均值: {features_tensor.mean():.3f}")
    print(f"   特征标准差: {features_tensor.std():.3f}")
    
    # 检查特征维度
    if features_tensor.shape[1] < 10:
        print("   警告: 特征维度较低，考虑增加特征")
    
    # 3. 链路预测难度分析
    print("\n3. 链路预测难度分析:")
    
    # 计算正样本边的特征相似度
    if len(edges_idx) > 0 and len(node_list) > 0:
        edge_features = []
        sample_edges = edges_idx[:min(100, len(edges_idx))]
        
        for src, dst in sample_edges:
            if src < len(node_list) and dst < len(node_list):
                src_feat = features_tensor[src]
                dst_feat = features_tensor[dst]
                similarity = F.cosine_similarity(src_feat.unsqueeze(0), dst_feat.unsqueeze(0))
                edge_features.append(similarity.item())
        
        if edge_features:
            print(f"   正样本边特征相似度均值: {np.mean(edge_features):.3f}")
            
            # 随机负样本的相似度
            random_similarities = []
            for _ in range(100):
                src_idx = np.random.randint(0, len(node_list))
                dst_idx = np.random.randint(0, len(node_list))
                if src_idx != dst_idx:
                    src_feat = features_tensor[src_idx]
                    dst_feat = features_tensor[dst_idx]
                    similarity = F.cosine_similarity(src_feat.unsqueeze(0), dst_feat.unsqueeze(0))
                    random_similarities.append(similarity.item())
            
            if random_similarities:
                print(f"   随机负样本特征相似度均值: {np.mean(random_similarities):.3f}")
                
                # 难度估计
                diff = abs(np.mean(edge_features) - np.mean(random_similarities))
                if diff < 0.1:
                    print("   ⚠️ 警告: 正负样本特征相似度差异小，任务难度高")
                else:
                    print(f"   ✓ 正负样本特征相似度差异: {diff:.3f}")
    
    # 4. 改进建议
    print("\n4. 改进建议:")
    
    if density < 0.01:
        print("   - 图较稀疏，考虑使用基于路径的特征")
    
    try:
        if nx.average_clustering(graph) < 0.1:
            print("   - 聚类系数低，社区结构不明显")
    except:
        pass
    
    if features_tensor.std() < 0.1:
        print("   - 特征方差小，考虑特征工程")
    
    if len(edges_idx) < 1000:
        print("   - 边数较少，考虑增加采样或使用数据增强")
    
    return {
        'graph_stats': {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': float(density),
            'avg_degree': float(np.mean(degrees)) if degrees else 0,
            'std_degree': float(np.std(degrees)) if degrees else 0
        },
        'feature_stats': {
            'shape': list(features_tensor.shape),
            'range': [float(features_tensor.min()), float(features_tensor.max())],
            'mean': float(features_tensor.mean()),
            'std': float(features_tensor.std())
        }
    }

# ==================== 改进的实验函数 ====================

def improved_experiment():
    """改进的实验主函数"""
    print("=" * 70)
    print("改进实验 - 深度优化")
    print("=" * 70)
    
    # 1. 尝试加载真实数据
    print("1. 加载数据...")
    edges_df = ImprovedWikiDataLoader.load_and_filter_data(
        target_nodes=150,
        target_edges=700,
        max_lines=50000,
        save_cleaned_data=True  # 保存清洗后的数据
    )
    
    # 如果加载失败或数据不足，使用合成数据
    if edges_df is None or len(edges_df) < 100:
        print("\n真实数据不足，使用合成数据集...")
        edges_df = ImprovedWikiDataLoader.create_synthetic_dataset(
            num_nodes=150,
            num_edges=733
        )
    
    print(f"\n数据加载完成: {len(edges_df)} 条边")
    
    # 2. 构建图
    print("\n2. 构建图...")
    G = nx.Graph()
    G.add_edges_from(list(zip(edges_df['source'], edges_df['target'])))
    
    print(f"图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 3. 提取改进特征
    print("\n3. 提取改进特征...")
    extractor = ImprovedFeatureExtractor(G)
    features_dict = extractor.extract_all_features()
    
    # 4. 创建数据集
    print("\n4. 创建数据集...")
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    # 特征矩阵
    features_list = [features_dict[node] for node in node_list]
    features_tensor = torch.tensor(features_list, dtype=torch.float32)
    
    # 边索引
    edge_list = []
    edges = list(G.edges())
    for src, dst in edges:
        src_idx = node_to_idx[src]
        dst_idx = node_to_idx[dst]
        edge_list.append([src_idx, dst_idx])
        edge_list.append([dst_idx, src_idx])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # 创建数据对象
    data = Data(x=features_tensor, edge_index=edge_index, num_nodes=len(node_list))
    data.graph = G
    data.node_to_idx = node_to_idx
    data.node_list = node_list
    
    # 5. 深度诊断
    print("\n5. 深度诊断...")
    
    # 准备边列表（去重）
    edge_set = set()
    for src, dst in edges:
        src_idx = node_to_idx[src]
        dst_idx = node_to_idx[dst]
        if src_idx < dst_idx:
            edge_set.add((src_idx, dst_idx))
        else:
            edge_set.add((dst_idx, src_idx))
    
    edges_idx = list(edge_set)
    
    diagnosis = deep_diagnosis(G, features_tensor, edges_idx, node_list)
    
    # 6. 划分数据集
    print("\n6. 划分数据集...")
    
    # 分层划分
    train_edges, temp_edges = train_test_split(edges_idx, test_size=0.4, random_state=42)
    val_edges, test_edges = train_test_split(temp_edges, test_size=0.5, random_state=42)
    
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
        'all_edges': edges_idx
    }
    
    print(f"数据集: {data.num_nodes} 节点, {len(edges_idx)} 边")
    print(f"特征维度: {data.x.shape[1]}")
    print(f"划分: 训练{len(train_edges)}边, 验证{len(val_edges)}边, 测试{len(test_edges)}边")
    
    # 7. 训练改进模型
    print("\n7. 训练改进模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建简化模型
    model = SimpleLinkPredictionModel(
        in_dim=data.x.shape[1],
        hidden_dim=64,
        dropout=0.4
    ).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总共 {total_params:,}, 可训练 {trainable_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # 训练
    model, history = improved_train_link_prediction(
        model, data, split_data, optimizer, device,
        num_epochs=300,
        num_neg_samples=3,
        hardness_level='medium'
    )
    
    # 8. 可视化
    print("\n8. 可视化结果...")
    plot_improved_results(history, diagnosis)
    
    # 9. 实验总结
    print("\n" + "=" * 70)
    print("实验总结:")
    print("=" * 70)
    
    if history.get('test_auc', 0.5) > 0.6:
        print("✓ 实验成功，模型学习到了有效模式")
    elif history.get('test_auc', 0.5) > 0.55:
        print("~ 实验部分成功，需要进一步优化")
    else:
        print("✗ 实验失败，模型没有学习到有效模式")
    
    print(f"测试集 AUC: {history.get('test_auc', 0):.4f}")
    print(f"测试集 AP: {history.get('test_ap', 0):.4f}")
    print(f"测试集 F1: {history.get('test_f1', 0):.4f}")
    print(f"测试集准确率: {history.get('test_accuracy', 0):.4f}")
    
    return history

def plot_improved_results(history, diagnosis):
    """绘制改进实验结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 训练损失
    if 'train_loss' in history and len(history['train_loss']) > 0:
        axes[0, 0].plot(history['train_loss'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('训练轮次')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].set_title('训练损失曲线')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 验证AUC
    if 'val_auc' in history and len(history['val_auc']) > 0:
        x_vals = list(range(10, len(history['val_auc']) * 10 + 1, 10))
        if len(x_vals) > len(history['val_auc']):
            x_vals = x_vals[:len(history['val_auc'])]
        axes[0, 1].plot(x_vals, history['val_auc'], 'r-o', linewidth=2, markersize=4)
        axes[0, 1].set_xlabel('训练轮次')
        axes[0, 1].set_ylabel('AUC分数')
        axes[0, 1].set_title('验证AUC曲线')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 验证AP
    if 'val_ap' in history and len(history['val_ap']) > 0:
        x_vals = list(range(10, len(history['val_ap']) * 10 + 1, 10))
        if len(x_vals) > len(history['val_ap']):
            x_vals = x_vals[:len(history['val_ap'])]
        axes[0, 2].plot(x_vals, history['val_ap'], 'g-s', linewidth=2, markersize=4)
        axes[0, 2].set_xlabel('训练轮次')
        axes[0, 2].set_ylabel('AP分数')
        axes[0, 2].set_title('验证AP曲线')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 图结构分析
    if diagnosis and 'graph_stats' in diagnosis:
        stats = diagnosis['graph_stats']
        labels = ['节点数', '边数', '边密度\n(x100)', '平均度']
        values = [
            stats['num_nodes'],
            stats['num_edges'],
            stats['density'] * 100,
            stats['avg_degree']
        ]
        
        axes[1, 0].bar(range(len(labels)), values, color='skyblue')
        axes[1, 0].set_xticks(range(len(labels)))
        axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, 0].set_title('图结构统计')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. 特征分析
    if diagnosis and 'feature_stats' in diagnosis:
        stats = diagnosis['feature_stats']
        labels = ['最小值', '最大值', '均值', '标准差']
        values = [
            stats['range'][0],
            stats['range'][1],
            stats['mean'],
            stats['std']
        ]
        
        axes[1, 1].bar(range(len(labels)), values, color='lightcoral')
        axes[1, 1].set_xticks(range(len(labels)))
        axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, 1].set_title('特征统计')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. 测试结果总结
    test_metrics = ['AUC', 'AP', 'F1', '准确率']
    test_values = [
        history.get('test_auc', 0),
        history.get('test_ap', 0),
        history.get('test_f1', 0),
        history.get('test_accuracy', 0)
    ]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    axes[1, 2].bar(test_metrics, test_values, color=colors)
    axes[1, 2].set_ylim(0, 1.0)
    axes[1, 2].set_title('测试集性能指标')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值
    for i, (metric, value) in enumerate(zip(test_metrics, test_values)):
        axes[1, 2].text(i, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f'改进实验 - 测试AUC: {history.get("test_auc", 0):.4f}, 测试F1: {history.get("test_f1", 0):.4f}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('models', exist_ok=True)
    filename = "models/improved_experiment_results.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {filename}")
    
    plt.show()

# ==================== 主函数 ====================

def main():
    """主函数"""
    print("=" * 70)
    print("GNN链路预测改进实验 (完整修复版)")
    print("=" * 70)
    
    # 运行改进实验
    history = improved_experiment()
    
    if history:
        print("\n实验完成!")
        if history.get('test_auc', 0) > 0.55:
            print("✓ 改进有效!")
        else:
            print("⚠️ 需要进一步优化")
    else:
        print("实验失败")

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    main()