"""
GNN模型训练脚本 - 使用真实WikiLinks数据集
"""
from sched import scheduler
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import os
import json
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
import matplotlib.pyplot as plt
import matplotlib
import warnings
import gzip
from tqdm import tqdm
from collections import Counter
warnings.filterwarnings('ignore')
import re
import random

# 尝试导入社区检测库，如果没有就使用备用方案
try:
    import community as community_louvain
    HAS_COMMUNITY = True
except:
    HAS_COMMUNITY = False
    print("警告: 未安装python-louvain库，将使用备用社区检测方法")

# ==================== 中文字符显示解决方案 ====================

def setup_chinese_font():
    """设置中文字体显示"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 测试字体
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
    
    # 设置图表样式
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    return True

setup_chinese_font()

# ==================== 数据加载与处理 ====================

class WikiDataLoader:
    """维基百科数据加载器"""
    
    @staticmethod
    def load_wikilinks_data(sample_size=300000, max_nodes=2000, min_degree=1):
        """加载真实WikiLinks数据集"""
        print(f"加载真实WikiLinks数据集...")
        print(f"采样: {sample_size:,} 条边")
        print(f"最大节点数: {max_nodes:,}")
        print(f"最小节点度: {min_degree}")
        
        data_path = "data/raw/enwiki.wikilink_graph.2018-03-01.csv.gz"
        
        if not os.path.exists(data_path):
            print(f"错误: 数据文件不存在: {data_path}")
            print("请确保已下载数据集并放在正确位置")
            return None
        
        try:
            # 读取压缩的CSV文件
            print("读取CSV文件...")
            edges = []
            node_counter = Counter()
            
            # 由于文件很大，我们使用更高效的方法
            with gzip.open(data_path, 'rt', encoding='utf-8') as f:
                # 跳过第一行（可能是标题）
                next(f)
                
                for i, line in enumerate(tqdm(f, total=sample_size)):
                    if i >= sample_size:
                        break
                    
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        src = parts[0].strip('"')
                        dst = parts[1].strip('"')
                        
                        # 过滤掉自环
                        if src != dst:
                            edges.append((src, dst))
                            node_counter.update([src, dst])
            
            print(f"读取完成: {len(edges):,} 条边, {len(node_counter):,} 个节点")
            
            # 构建图以计算节点度
            print("构建临时图计算节点度...")
            G_temp = nx.Graph()
          
            
            # 修改后：使用更多边或全部边
            if len(edges) > 100000:
                # 对大图进行抽样，但保证足够样本
                sample_size = min(100000, len(edges))
                sample_edges = random.sample(edges, sample_size)
                G_temp.add_edges_from(sample_edges)
            else:
                # 对小图使用全部边
                G_temp.add_edges_from(edges)


            # 计算节点度
            degrees = dict(G_temp.degree())
            
            # 选择出现频率高且度大的节点
            print("选择高频高连接度节点...")
            top_nodes = []

            for node, count in node_counter.most_common(max_nodes * 3):  # 多看一些
                if node in G_temp.nodes() and G_temp.degree(node) >= min_degree:
                    top_nodes.append(node)
                if len(top_nodes) >= max_nodes:
                    break
            
            # 如果节点太少，放宽条件
            if len(top_nodes) < max_nodes // 2:
                print(f"节点不足({len(top_nodes)})，放宽选择条件...")
                top_nodes = []
                for node, count in node_counter.most_common(max_nodes * 5):
                    if node in G_temp.nodes():
                        top_nodes.append(node)
                    if len(top_nodes) >= max_nodes:
                        break

            node_set = set(top_nodes)
            
            # 过滤边，只保留两个节点都在节点集中的边
            print("过滤边...")
            filtered_edges = []
            for src, dst in tqdm(edges):
                if src in node_set and dst in node_set:
                    filtered_edges.append((src, dst))
            
            print(f"过滤后: {len(filtered_edges):,} 条边")
            
            # 创建DataFrame
            edges_df = pd.DataFrame(filtered_edges, columns=['source', 'target'])
            
            # 在过滤边后添加
            print(f"原始边数: {len(edges):,}")
            print(f"节点计数器大小: {len(node_counter):,}")
            print(f"临时图节点数: {G_temp.number_of_nodes():,}")
            print(f"临时图边数: {G_temp.number_of_edges():,}")
            print(f"选择节点数: {len(top_nodes):,}")
            print(f"最终过滤边数: {len(filtered_edges):,}")

            # 分析节点度分布
            if G_temp.number_of_nodes() > 0:
                degrees = [d for _, d in G_temp.degree()]
                print(f"临时图节点度统计: 平均={np.mean(degrees):.1f}, 最大={max(degrees)}, 最小={min(degrees)}")

            return edges_df
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def preprocess_data(edges_df, max_component_size=20000):
        """预处理数据：构建图并提取最大连通分量"""
        print("构建图...")
        G = nx.Graph()
        G.add_edges_from(list(zip(edges_df['source'], edges_df['target'])))
        
        print(f"原始图: {G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")
        
        # 提取最大连通分量
        if nx.is_connected(G):
            largest_cc = G
        else:
            print("提取最大连通分量...")
            connected_components = list(nx.connected_components(G))
            largest_cc_nodes = max(connected_components, key=len)
            largest_cc = G.subgraph(largest_cc_nodes).copy()
        
        print(f"最大连通分量: {largest_cc.number_of_nodes():,} 节点, {largest_cc.number_of_edges():,} 边")
        
        # 如果连通分量太大，采样子图
        if largest_cc.number_of_nodes() > max_component_size:
            print(f"采样 {max_component_size:,} 个节点的子图...")
            # 随机选择种子节点，然后采样邻居
            seed_node = np.random.choice(list(largest_cc.nodes()))
            subgraph_nodes = set([seed_node])
            
            # BFS采样
            for _ in range(3):  # 3跳邻居
                new_nodes = set()
                for node in subgraph_nodes:
                    new_nodes.update(largest_cc.neighbors(node))
                subgraph_nodes.update(new_nodes)
                if len(subgraph_nodes) >= max_component_size:
                    break
            
            # 限制大小
            if len(subgraph_nodes) > max_component_size:
                subgraph_nodes = set(list(subgraph_nodes)[:max_component_size])
            
            largest_cc = largest_cc.subgraph(subgraph_nodes).copy()
            print(f"采样后: {largest_cc.number_of_nodes():,} 节点, {largest_cc.number_of_edges():,} 边")
        
        return largest_cc

    @staticmethod
    def analyze_data_quality(edges_df):
        """分析数据质量"""
        if edges_df is None or len(edges_df) == 0:
            print("无数据可分析")
            return
        
        print("\n=== 数据质量分析 ===")
        
        # 构建图
        G = nx.Graph()
        G.add_edges_from(list(zip(edges_df['source'], edges_df['target'])))
        
        # 基本统计
        print(f"节点数: {G.number_of_nodes()}")
        print(f"边数: {G.number_of_edges()}")
        
        # 连通性
        if nx.is_connected(G):
            print("图是连通的")
        else:
            components = list(nx.connected_components(G))
            print(f"连通分量数: {len(components)}")
            print(f"最大连通分量大小: {len(max(components, key=len))}")
        
        # 度分布
        degrees = [d for _, d in G.degree()]
        print(f"平均度: {np.mean(degrees):.2f}")
        print(f"最大度: {max(degrees)}")
        print(f"最小度: {min(degrees)}")
        
        # 节点示例
        print("\n节点示例:")
        for i, node in enumerate(list(G.nodes())[:3]):
            print(f"  {i+1}. '{node}'")
        
        return G

# ==================== 特征工程 ====================

class EnhancedFeatureExtractor:
    """增强特征提取器"""
    
    def __init__(self, graph):
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        
    def extract_all_features(self):
        """提取所有特征"""
        print("开始提取特征...")
        
        try:
            # 1. 基础结构特征
            print("  1. 提取基础结构特征...")
            structural_features = self._extract_structural_features()
            
            # 2. 随机游走特征
            print("  2. 提取随机游走特征...")
            walk_features = self._extract_random_walk_features()
            
            # 3. 社区特征
            print("  3. 提取社区特征...")
            community_features = self._extract_community_features()
            
            # 4. 节点嵌入特征（Node2Vec简化版）
            print("  4. 提取节点嵌入特征...")
            embedding_features = self._extract_simple_embedding_features()
            
            # 合并特征
            print("合并所有特征...")
            features = {}
            
            for node in self.graph.nodes():
                # 获取各种特征
                struct_feat = structural_features.get(node, np.zeros(7))
                walk_feat = walk_features.get(node, np.zeros(10))
                comm_feat = community_features.get(node, np.zeros(5))
                embed_feat = embedding_features.get(node, np.zeros(8))
                
                # 合并（总共30维特征）
                combined = np.concatenate([struct_feat, walk_feat, comm_feat, embed_feat])
                features[node] = combined
            
            print(f"特征提取完成: {len(features)} 个节点, {combined.shape[0]} 维特征")
            return features
            
        except Exception as e:
            print(f"复杂特征提取失败: {e}")
            print("使用简单特征作为备用...")
            return self.extract_simple_features()
    
    def _extract_structural_features(self):
        """提取图结构特征"""
        features = {}
        
        # 计算各种中心性
        print("    计算度中心性...")
        degrees = dict(self.graph.degree())
        max_degree = max(degrees.values()) if degrees.values() else 1
        
        print("    计算聚类系数...")
        clustering = nx.clustering(self.graph)
        max_clustering = max(clustering.values()) if clustering.values() else 1
        # 防止除以0
        if max_clustering == 0:
            max_clustering = 1
        
        print("    计算PageRank...")
        pagerank = nx.pagerank(self.graph, alpha=0.85)
        max_pagerank = max(pagerank.values()) if pagerank.values() else 1
        if max_pagerank == 0:
            max_pagerank = 1
        
        print("    计算接近中心性（小图）...")
        if self.num_nodes < 10000:
            try:
                closeness = nx.closeness_centrality(self.graph)
                max_closeness = max(closeness.values()) if closeness.values() else 1
                if max_closeness == 0:
                    max_closeness = 1
            except:
                closeness = {node: 0 for node in self.graph.nodes()}
                max_closeness = 1
        else:
            closeness = {node: 0 for node in self.graph.nodes()}
            max_closeness = 1
        
        # 邻居数量分布
        neighbor_counts = {node: len(list(self.graph.neighbors(node))) for node in self.graph.nodes()}
        max_neighbors = max(neighbor_counts.values()) if neighbor_counts.values() else 1
        
        # 计算平均度用于二值特征
        degree_values = list(degrees.values())
        avg_degree = np.mean(degree_values) if degree_values else 0
        
        # 构建特征向量
        for node in self.graph.nodes():
            # 使用安全除法
            deg_norm = degrees.get(node, 0) / max_degree if max_degree > 0 else 0
            clust_norm = clustering.get(node, 0) / max_clustering if max_clustering > 0 else 0
            pr_norm = pagerank.get(node, 0) / max_pagerank if max_pagerank > 0 else 0
            close_norm = closeness.get(node, 0) / max_closeness if max_closeness > 0 else 0
            neighbor_norm = neighbor_counts.get(node, 0) / max_neighbors if max_neighbors > 0 else 0
            
            # 对数变换的特征
            log_deg = np.log1p(degrees.get(node, 0)) / np.log1p(max_degree) if max_degree > 0 else 0
            
            # 二值特征：节点度是否高于平均值
            above_avg = 1 if degrees.get(node, 0) > avg_degree else 0
            
            features[node] = np.array([
                deg_norm,
                clust_norm,
                pr_norm,
                close_norm,
                neighbor_norm,
                log_deg,
                above_avg
            ])
        
        return features
    
    def _extract_random_walk_features(self):
        """提取随机游走特征（简化版）"""
        features = {}
        
        # 由于计算量大，对小图才计算
        if self.num_nodes > 10000:
            print("    跳过随机游走特征（图太大）...")
            return {node: np.zeros(10) for node in self.graph.nodes()}
        
        print("    计算随机游走特征...")
        for node in self.graph.nodes():
            # 执行多次随机游走
            walk_counts = Counter()
            for _ in range(5):  # 5次随机游走
                current = node
                for step in range(10):  # 游走长度10
                    neighbors = list(self.graph.neighbors(current))
                    if not neighbors:
                        break
                    current = np.random.choice(neighbors)
                    walk_counts[current] += 1
            
            # 构建特征向量（访问频率最高的10个节点）
            total_steps = sum(walk_counts.values())
            if total_steps > 0:
                # 获取访问频率最高的10个节点（不包括自己）
                top_nodes = [n for n, _ in walk_counts.most_common(11) if n != node][:10]
                feature_vector = [walk_counts[n] / total_steps for n in top_nodes]
                # 补零
                while len(feature_vector) < 10:
                    feature_vector.append(0)
            else:
                feature_vector = [0] * 10
            
            features[node] = np.array(feature_vector)
        
        return features
    
    def _extract_community_features(self):
        """提取社区特征"""
        features = {}
        
        try:
            if HAS_COMMUNITY:
                print("    检测社区...")
                partition = community_louvain.best_partition(self.graph)
                
                # 统计社区大小
                community_sizes = Counter(partition.values())
                
                for node in self.graph.nodes():
                    comm_id = partition.get(node, 0)
                    comm_size = community_sizes.get(comm_id, 0)
                    
                    # 构建特征向量
                    features[node] = np.array([
                        comm_id % 5,  # 社区ID取模（限制范围）
                        comm_size / len(self.graph.nodes()),  # 社区相对大小
                        1 if comm_size > 10 else 0,  # 是否是大社区
                        len([n for n in self.graph.neighbors(node) if partition.get(n, 0) == comm_id]) / max(1, len(list(self.graph.neighbors(node)))),  # 邻居在同一社区的比例
                        1 if comm_id < 5 else 0  # 是否属于前5个社区
                    ])
            else:
                # 使用简单的基于度的"社区"
                for node in self.graph.nodes():
                    degree = self.graph.degree(node)
                    features[node] = np.array([
                        degree % 5,
                        1 if degree > 5 else 0,
                        1 if degree > 10 else 0,
                        np.random.random(),  # 随机值替代
                        0
                    ])
        except Exception as e:
            print(f"    社区检测失败: {e}")
            for node in self.graph.nodes():
                features[node] = np.zeros(5)
        
        return features
    
    def _extract_simple_embedding_features(self):
        """提取简单的节点嵌入特征"""
        features = {}
        
        # 使用节点的结构特性作为简单嵌入
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            neighbor_degrees = [self.graph.degree(n) for n in neighbors]
            
            features[node] = np.array([
                len(neighbors) / max(1, self.num_nodes),
                np.mean(neighbor_degrees) / 100 if neighbor_degrees else 0,
                np.std(neighbor_degrees) / 100 if len(neighbor_degrees) > 1 else 0,
                max(neighbor_degrees) / 100 if neighbor_degrees else 0,
                min(neighbor_degrees) / 100 if neighbor_degrees else 0,
                1 if len(neighbors) > np.mean([self.graph.degree(n) for n in self.graph.nodes()]) / 2 else 0,
                1 if any(self.graph.degree(n) > 100 for n in neighbors) else 0,
                len([n for n in neighbors if self.graph.degree(n) > 10]) / max(1, len(neighbors))
            ])
        
        return features

    def extract_node_id_from_wiki_string(node_str):
            """从维基百科格式的节点字符串中提取数字ID"""
            # 格式: "State   number   description"
            parts = node_str.split('\t')
            
            # 方法1：查找包含纯数字的字段
            for part in parts:
                stripped = part.strip()
                if stripped and stripped.isdigit():
                    return int(stripped)
            
            # 方法2：使用正则表达式从任意字段提取数字
            import re
            for part in parts:
                numbers = re.findall(r'\d+', part)
                if numbers:
                    return int(numbers[0])
            
            # 方法3：使用字符串哈希作为替代（确保一致性）
            return abs(hash(node_str)) % 10000  

    def extract_simple_features(self):
        """提取简单特征（避免复杂的图计算）"""
        print("提取简单特征...")
        
        # 添加辅助函数
        import re
        
        def extract_node_id(node_str):
            """从节点字符串中提取数字ID"""
            parts = node_str.split('\t')
            
            # 查找数字字段
            for part in parts:
                stripped = part.strip()
                if stripped and stripped.isdigit():
                    return int(stripped)
            
            # 如果没有找到纯数字，尝试提取数字
            for part in parts:
                numbers = re.findall(r'\d+', part)
                if numbers:
                    return int(numbers[0])
            
            # 使用哈希作为替代
            return abs(hash(node_str)) % 10000
        
        features_dict = {}
        
        # 只使用度特征和简单的统计特征
        degrees = dict(self.graph.degree())
        max_degree = max(degrees.values()) if degrees.values() else 1
        degree_values = list(degrees.values())
        avg_degree = np.mean(degree_values) if degree_values else 0
        
        # 邻居计数
        neighbor_counts = {node: len(list(self.graph.neighbors(node))) for node in self.graph.nodes()}
        max_neighbors = max(neighbor_counts.values()) if neighbor_counts.values() else 1
        
        # 减少调试输出
        debug_count = 0
        for node in self.graph.nodes():
            # 基础特征
            deg = degrees.get(node, 0)
            neighbors = neighbor_counts.get(node, 0)
            
            # 邻居的度统计
            neighbor_nodes = list(self.graph.neighbors(node))
            neighbor_degrees = [degrees.get(n, 0) for n in neighbor_nodes]
            
            # 提取节点ID
            node_id = extract_node_id(node)
            
            # 只在第一次遇到ID=0时警告
            if node_id == 0 and debug_count < 3:
                print(f"警告: 节点 '{node[:50]}...' 的ID格式异常，已使用默认ID 0")
                debug_count += 1
            
            # 构建特征向量 (20维)
            feature_vector = np.array([
                deg / max_degree if max_degree > 0 else 0,  # 归一化度
                np.log1p(deg) / np.log1p(max_degree) if max_degree > 0 else 0,  # 对数度
                neighbors / max_neighbors if max_neighbors > 0 else 0,  # 邻居数
                1 if deg > avg_degree else 0,  # 是否高于平均度
                len(neighbor_nodes) / self.num_nodes if self.num_nodes > 0 else 0,  # 邻居比例
                
                # 邻居统计特征
                np.mean(neighbor_degrees) / 100 if neighbor_degrees else 0,  # 平均邻居度
                np.std(neighbor_degrees) / 100 if len(neighbor_degrees) > 1 else 0,  # 邻居度标准差
                max(neighbor_degrees) / 100 if neighbor_degrees else 0,  # 最大邻居度
                min(neighbor_degrees) / 100 if neighbor_degrees else 0,  # 最小邻居度
                
                # 随机特征（占位符）
                np.random.random() * 0.1,  # 随机噪声1
                np.random.random() * 0.1,  # 随机噪声2
                
                # 基于节点ID的特征
                (node_id % 10) / 10,
                (node_id % 100) / 100,
                (node_id % 1000) / 1000,
                
                # 二值特征
                1 if deg % 2 == 0 else 0,
                1 if neighbors % 2 == 0 else 0,
                1 if len(neighbor_nodes) > 0 else 0,
                1 if any(degrees.get(n, 0) > 10 for n in neighbor_nodes) else 0,
                1 if deg > 5 else 0,
                1 if neighbors > 3 else 0
            ])
        
            # 直接将特征向量存入 features_dict
            features_dict[node] = feature_vector
        
        return features_dict
    
  

# ==================== 模型定义 ====================

class BaseGNN(torch.nn.Module):
    """基础GNN模型"""
    def __init__(self, in_channels, hidden_channels, out_channels, model_type='gcn'):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'gcn':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif model_type == 'gat':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
            self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False)
        elif model_type == 'sage':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        
        # 边预测头
        self.lin1 = torch.nn.Linear(out_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
        
    def encode(self, x, edge_index):
        """编码节点特征"""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        """解码边存在概率"""
        src, dst = edge_index
        edge_features = torch.cat([z[src], z[dst]], dim=-1)
        edge_features = self.lin1(edge_features)
        edge_features = F.relu(edge_features)
        edge_features = F.dropout(edge_features, p=0.3, training=self.training)
        return torch.sigmoid(self.lin2(edge_features)).squeeze()
    
    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        """完整前向传播"""
        z = self.encode(x, edge_index)
        pos_scores = self.decode(z, pos_edge_index)
        neg_scores = self.decode(z, neg_edge_index)
        return z, pos_scores, neg_scores

class EnhancedGraphSAGE(torch.nn.Module):
    """增强版GraphSAGE"""
    
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3, use_residual=True):
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropout = dropout
        self.use_residual = use_residual
        
        # 输入层
        self.layers.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
            
        # 输出层
        self.layers.append(SAGEConv(hidden_channels, out_channels))
        
        # 边预测头
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(out_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1)
        )
        
    def encode(self, x, edge_index):
        """编码节点特征"""
        x_all = x
        
        for i, (conv, norm) in enumerate(zip(self.layers[:-1], self.norms)):
            x_res = x_all if self.use_residual else None
            
            x_all = conv(x_all, edge_index)
            x_all = norm(x_all)
            x_all = F.relu(x_all)
            x_all = F.dropout(x_all, p=self.dropout, training=self.training)
            
            if self.use_residual and x_res is not None and x_res.size(1) == x_all.size(1):
                x_all = x_all + x_res
        
        # 最后一层
        x_all = self.layers[-1](x_all, edge_index)
        return x_all
    
    def decode(self, z, edge_index):
        """解码边存在概率"""
        src, dst = edge_index
        edge_features = torch.cat([z[src], z[dst]], dim=-1)
        return torch.sigmoid(self.edge_predictor(edge_features)).squeeze()
    
    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        """完整前向传播"""
        z = self.encode(x, edge_index)
        pos_scores = self.decode(z, pos_edge_index)
        neg_scores = self.decode(z, neg_edge_index)
        return z, pos_scores, neg_scores
    



# ==================== 训练与评估函数 ====================
def train_link_prediction(model, data, split_data, optimizer, device, 
                         num_neg_samples=2, epochs=100, patience=15,
                         scheduler=None, difficulty_level='medium'):
    """训练链路预测模型(使用困难负采样）"""
    # 将数据移到设备
    x = data.x.to(device)
    train_edge_index = split_data['train']['edge_index'].to(device)
    # 准备训练边（正样本）
    train_pos_edges = split_data['train']['edges']
    
    # 训练历史
    train_losses = []
    val_aucs = []
    val_aps = []
    best_val_auc = 0
    best_epoch = 0
    
    print("开始训练...")
    print(f"设备: {device}")
    print(f"训练正样本数: {len(train_pos_edges)}")
    print(f"验证集边数: {len(split_data['val']['edges'])}")
    print(f"测试集边数: {len(split_data['test']['edges'])}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
         # 生成负样本
        num_nodes = data.num_nodes
        
        num_pos_edges = len(train_pos_edges)  # 定义这个变量
        num_neg_edges = num_pos_edges * num_neg_samples
        
        
        # 随机采样负样本
        neg_edges = []
        while len(neg_edges) < num_neg_edges:
            src = torch.randint(0, num_nodes, (num_neg_edges * 2,))
            dst = torch.randint(0, num_nodes, (num_neg_edges * 2,))
            
            for s, d in zip(src, dst):
                if len(neg_edges) >= num_neg_edges:
                    break
                # 检查是否是正样本
                if (s.item(), d.item()) not in train_pos_edges and (d.item(), s.item()) not in train_pos_edges:
                    neg_edges.append((s.item(), d.item()))
        
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().to(device)
        
        # 准备正样本边索引
        pos_edge_index = torch.tensor(train_pos_edges, dtype=torch.long).t().to(device)
        
        # 前向传播
        if isinstance(model, EnhancedGraphSAGE):
            z = model.encode(x, train_edge_index)
            pos_scores = model.decode(z, pos_edge_index)
            neg_scores = model.decode(z, neg_edge_index)
        else:
            z, pos_scores, neg_scores = model(x, train_edge_index, pos_edge_index, neg_edge_index)
        
        # 计算损失 - 使用带权重的损失
        pos_weight = torch.tensor([2.0]).to(device) # 正样本权重更高
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores), pos_weight=pos_weight)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # 验证
        if epoch % 5 == 0:
            val_auc, val_ap = evaluate_model(model, data, split_data['val']['edges'], 
                                           split_data['train']['edge_index'], device, 
                                           verbose=False)
            val_aucs.append(val_auc)
            val_aps.append(val_ap)
            
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}')
            
            # 早停检查
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                
                # 保存最佳模型
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_ap': val_ap,
                }, 'models/best_model.pt')
                
                print(f"  保存最佳模型 (Val AUC: {val_auc:.4f})")
            
            # 早停
            if epoch - best_epoch > patience:
                print(f"早停在第 {epoch} 轮")
                break
        else:
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    # 加载最佳模型
    try:
        checkpoint = torch.load('models/best_model.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print("无法加载最佳模型，使用最后训练的模型")
    
    print("\n=== 训练完成，开始最终评估 ===")
    test_auc, test_ap = evaluate_model(model, data, split_data['test']['edges'], 
                                     split_data['train']['edge_index'], device, 
                                     verbose=True)
    print(f"\n最终测试结果:")
    print(f"测试集 AUC: {test_auc:.4f}")
    print(f"测试集 AP: {test_ap:.4f}")
    # 保存训练历史
    history = {
        'train_loss': train_losses,
        'val_auc': val_aucs,
        'val_ap': val_aps,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'test_auc': test_auc,
        'test_ap': test_ap
    }
    os.makedirs('models', exist_ok=True)
    with open('models/training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    return model, history

def evaluate_model(model, data, eval_edges, train_edge_index, device, num_neg_samples=1, verbose=True):
    """评估模型性能"""
    model.eval()
    split_data=[]
    
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = train_edge_index.to(device)
        
        # 编码节点
        if isinstance(model, EnhancedGraphSAGE):
            z = model.encode(x, edge_index)
        else:
            z = model.encode(x, edge_index)
        
        # 准备正样本
        pos_edges = eval_edges
        pos_edge_index = torch.tensor(pos_edges, dtype=torch.long).t().to(device)
        
        # 生成负样本
        num_nodes = data.num_nodes
        num_pos = len(pos_edges)  # 使用 len(pos_edges)
        num_neg = num_pos * num_neg_samples
        
        # 采样负样本
        all_train_edges = set(split_data['train']['edges'])
        if 'val' in split_data:
            all_train_edges.update(split_data['val']['edges'])
        neg_edges = []
        attempts = 0
        max_attempts = num_neg * 10
        while len(neg_edges) < num_neg and attempts < max_attempts:
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            
            if src == dst:
                continue
                
            # 确保不是正样本，也不是训练/验证集的边
            edge = (min(src, dst), max(src, dst))
            if (edge not in pos_edges and 
                edge not in all_train_edges and
                (dst, src) not in pos_edges and
                (dst, src) not in all_train_edges):
                neg_edges.append(edge)
            
            attempts += 1
        
        if len(neg_edges) < num_neg:
            print(f"警告: 只能生成 {len(neg_edges)}/{num_neg} 个合格的负样本")
        
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().to(device)
        
        # 计算分数
        if isinstance(model, EnhancedGraphSAGE):
            pos_scores = model.decode(z, pos_edge_index)
            neg_scores = model.decode(z, neg_edge_index)
        else:
            pos_scores = model.decode(z, pos_edge_index)
            neg_scores = model.decode(z, neg_edge_index)
        
        # 计算指标
        y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))]).cpu().numpy()
        y_pred = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        
        if verbose:
            print(f"评估结果: AUC={auc:.4f}, AP={ap:.4f}")
            # 添加分数分布信息
            print(f"  正样本分数: [{pos_scores.min():.3f}, {pos_scores.max():.3f}], 均值={pos_scores.mean():.3f}")
            print(f"  负样本分数: [{neg_scores.min():.3f}, {neg_scores.max():.3f}], 均值={neg_scores.mean():.3f}")
    
    return auc, ap

def generate_difficult_negative_samples(graph, node_list, node_to_idx, positive_edges, num_samples, difficulty_level='hard'):
    """生成困难负样本"""
    num_nodes = len(node_list)
    negative_edges = []
    positive_set = set(positive_edges)
    
    # 计算节点度
    degrees = dict(graph.degree())
    degree_values = [degrees.get(node, 1) for node in node_list]
    
    if difficulty_level == 'easy':
        # 随机负采样（当前使用的）
        attempts = 0
        while len(negative_edges) < num_samples and attempts < num_samples * 10:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src == dst:
                continue
            edge = (min(src, dst), max(src, dst))
            if edge not in positive_set:
                negative_edges.append(edge)
                positive_set.add(edge)
            attempts += 1
    
    elif difficulty_level == 'medium':
        # 基于度的负采样：按节点度概率选择
        degree_probs = np.array(degree_values) / sum(degree_values)
        
        attempts = 0
        while len(negative_edges) < num_samples and attempts < num_samples * 5:
            # 按节点度概率选择节点
            src = np.random.choice(num_nodes, p=degree_probs)
            dst = np.random.choice(num_nodes, p=degree_probs)
            if src == dst:
                continue
            edge = (min(src, dst), max(src, dst))
            if edge not in positive_set:
                negative_edges.append(edge)
                positive_set.add(edge)
            attempts += 1
    
    elif difficulty_level == 'hard':
        # 更困难的负采样：选择共同邻居少的节点对
        # 先构建邻接表
        adj_list = {i: set() for i in range(num_nodes)}
        for src, dst in positive_edges:
            adj_list[src].add(dst)
            adj_list[dst].add(src)
        
        attempts = 0
        while len(negative_edges) < num_samples and attempts < num_samples * 20:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src == dst:
                continue
            
            # 检查是否有共同邻居（有共同邻居的边更可能是正样本）
            common_neighbors = len(adj_list[src] & adj_list[dst])
            if common_neighbors > 1:  # 如果有多个共同邻居，很可能是正样本
                continue
                
            edge = (min(src, dst), max(src, dst))
            if edge not in positive_set:
                negative_edges.append(edge)
                positive_set.add(edge)
            attempts += 1
    
    # 如果采样不够，用随机采样补充
    if len(negative_edges) < num_samples:
        additional = num_samples - len(negative_edges)
        additional_edges = generate_difficult_negative_samples(
            graph, node_list, node_to_idx, positive_edges, additional, difficulty_level='easy'
        )
        negative_edges.extend(additional_edges)
    
    return negative_edges[:num_samples]

def generate_difficult_negatives(num_nodes, positive_edges, existing_edges_set, num_samples, graph=None, node_list=None):
    """生成困难的负样本"""
    neg_edges = []
    
    # 如果提供了图信息，使用基于图的困难负采样
    if graph is not None and node_list is not None:
        # 计算节点度
        degrees = dict(graph.degree())
        
        # 策略1：选择度相似的节点对
        for src, dst in positive_edges[:num_samples//2]:
            src_deg = degrees[node_list[src]]
            dst_deg = degrees[node_list[dst]]
            
            # 寻找与src度相似的节点
            candidates = []
            for node in range(num_nodes):
                if node != src and node != dst:
                    cand_deg = degrees[node_list[node]]
                    if abs(cand_deg - dst_deg) < max(dst_deg * 0.5, 5):
                        edge = (min(src, node), max(src, node))
                        if edge not in existing_edges_set:
                            candidates.append(node)
            
            if candidates:
                chosen = np.random.choice(candidates)
                edge = (min(src, chosen), max(src, chosen))
                neg_edges.append(edge)
                existing_edges_set.add(edge)
    
    # 策略2：随机补充
    while len(neg_edges) < num_samples:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src == dst:
            continue
        edge = (min(src, dst), max(src, dst))
        if edge not in existing_edges_set:
            neg_edges.append(edge)
            existing_edges_set.add(edge)
    
    return neg_edges[:num_samples]


def evaluate_task_difficulty(graph, edges, node_list, node_to_idx):
    """评估链路预测任务的难度"""
    print("\n=== 任务难度评估 ===")
    
    num_nodes = len(node_list)
    num_edges = len(edges)
    
    # 1. 图的基本统计
    degrees = dict(graph.degree())
    degree_values = list(degrees.values())
    
    print(f"节点数: {num_nodes}")
    print(f"边数: {num_edges}")
    print(f"平均度: {np.mean(degree_values):.2f}")
    print(f"最大度: {max(degree_values)}")
    print(f"最小度: {min(degree_values)}")
    
    # 2. 边密度
    possible_edges = num_nodes * (num_nodes - 1) // 2
    density = num_edges / possible_edges
    print(f"边密度: {density:.6f} ({density*100:.4f}%)")
    
    # 3. 负样本的"自然难度"
    # 随机采样负样本，计算它们的特征与正样本的差异
    positive_features = []
    negative_features = []
    
    # 计算节点度特征
    for src_idx, dst_idx in edges[:100]:  # 采样100条正样本边
        src_deg = degrees[node_list[src_idx]]
        dst_deg = degrees[node_list[dst_idx]]
        positive_features.append([src_deg, dst_deg, abs(src_deg - dst_deg)])
    
    # 随机生成100条负样本边
    neg_samples = 0
    while neg_samples < 100:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src == dst:
            continue
        edge = (min(src, dst), max(src, dst))
        if edge not in edges:
            src_deg = degrees[node_list[src]]
            dst_deg = degrees[node_list[dst]]
            negative_features.append([src_deg, dst_deg, abs(src_deg - dst_deg)])
            neg_samples += 1
    
    # 计算正负样本特征的差异
    pos_mean = np.mean(positive_features, axis=0)
    neg_mean = np.mean(negative_features, axis=0)
    
    print(f"\n正样本平均特征: 源节点度={pos_mean[0]:.2f}, 目标节点度={pos_mean[1]:.2f}, 度差={pos_mean[2]:.2f}")
    print(f"负样本平均特征: 源节点度={neg_mean[0]:.2f}, 目标节点度={neg_mean[1]:.2f}, 度差={neg_mean[2]:.2f}")
    
    # 计算可分性指标
    from scipy.spatial.distance import mahalanobis
    
    # 简单差异度量
    degree_diff = abs(pos_mean[0] - neg_mean[0]) / max(pos_mean[0], neg_mean[0])
    print(f"节点度差异: {degree_diff:.3f} ({'高' if degree_diff > 0.5 else '中等' if degree_diff > 0.2 else '低'}难度)")
    
    # 4. 社区结构检测（如果图不太大）
    if num_nodes < 1000:
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(graph)
            
            # 计算社区内的边比例
            intra_community_edges = 0
            for src, dst in edges:
                if partition[node_list[src]] == partition[node_list[dst]]:
                    intra_community_edges += 1
            
            intra_ratio = intra_community_edges / len(edges)
            print(f"社区内边比例: {intra_ratio:.3f} ({'强' if intra_ratio > 0.8 else '中等' if intra_ratio > 0.6 else '弱'}社区结构)")
        except:
            print("无法检测社区结构")
    
    return {
        'density': density,
        'degree_difference': degree_diff,
        'positive_features': pos_mean,
        'negative_features': neg_mean
    }

# ==================== 快速实验函数 ====================

def quick_real_data_experiment():
    """快速真实数据实验"""
    print("=" * 70)
    print("快速真实数据实验")
    print("=" * 70)
    
    # 1. 加载数据
    print("1. 加载数据...")
    edges_df = WikiDataLoader.load_wikilinks_data(
        sample_size=50000, 
        max_nodes=1000, 
        min_degree=1)
    
    if edges_df is None or len(edges_df) == 0:
        print("数据加载失败，尝试更小参数...")
        edges_df = WikiDataLoader.load_wikilinks_data(
            sample_size=20000,
            max_nodes=500,
            min_degree=1
        )
   
        
    # 2. 如果数据仍然很小，警告用户
    if edges_df is not None and len(edges_df) < 100:
        G_analysis = WikiDataLoader.analyze_data_quality(edges_df)
        print(f"警告: 数据量非常小 ({len(edges_df)}条边)")
        print("可能需要调整数据加载参数或检查原始数据文件")
    
    # 2. 构建图
    print("2. 构建图...")
    G = nx.Graph()
    G.add_edges_from(list(zip(edges_df['source'], edges_df['target'])))
    
    # 提取最大连通分量
    if nx.is_connected(G):
        largest_cc = G
    else:
        connected_components = list(nx.connected_components(G))
        largest_cc_nodes = max(connected_components, key=len)
        largest_cc = G.subgraph(largest_cc_nodes).copy()
    
    print(f"图: {largest_cc.number_of_nodes()} 节点, {largest_cc.number_of_edges()} 边")
    
    # 3. 提取简单特征
    print("3. 提取特征...")
    extractor = EnhancedFeatureExtractor(largest_cc)
    features_dict = extractor.extract_simple_features()  # 使用简单特征
    
    
    # 4. 创建数据集
    print("4. 创建数据集...")
    node_list = list(largest_cc.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    # 构建特征矩阵
    features_list = [features_dict[node] for node in node_list]
    features_tensor = torch.tensor(features_list, dtype=torch.float32)

    # print(f"节点数量: {len(node_list)}")
    # print(f"特征字典键数量: {len(features_dict)}")
    # print(f"特征列表长度: {len(features_list)}")
    # print(f"特征向量维度: {features_list[0].shape if features_list else '无'}")
    
    # 构建边索引
    edge_list = []
    for src, dst in largest_cc.edges():
        src_idx = node_to_idx[src]
        dst_idx = node_to_idx[dst]
        edge_list.append([src_idx, dst_idx])
        edge_list.append([dst_idx, src_idx])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    data = Data(x=features_tensor, edge_index=edge_index, num_nodes=len(node_list))
    
    # 5. 划分数据集
    print("5. 划分数据集...")
    edges = list(largest_cc.edges())
    edge_indices = [(node_to_idx[src], node_to_idx[dst]) for src, dst in edges]
    
    # 去重（无向图）
    edge_set = set()
    for src, dst in edge_indices:
        if src < dst:
            edge_set.add((src, dst))
        else:
            edge_set.add((dst, src))
    
    edges = list(edge_set)
    
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
    
    print(f"数据集: {data.num_nodes} 节点, {len(edges)} 边")
    print(f"特征维度: {data.x.shape[1]}")
    
    # 6. 训练模型
    print("6. 训练模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 使用增强版GraphSAGE
    model = EnhancedGraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=256,
        out_channels=128,
        num_layers=3,
        dropout=0.4,
        use_residual=True
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=0.005, 
                                 weight_decay=1e-4)
    
    # 添加学习率调度器（位置A3）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )


    # 训练
    model, history = train_link_prediction(
        model, data, split_data, optimizer, device,
        num_neg_samples=3, 
        epochs=200, 
        patience=20,
        scheduler=scheduler  # 传入调度器
    )
    
    print(f"最终结果: AUC={history['test_auc']:.4f}, AP={history['test_ap']:.4f}")
    
    # ===== 添加分析函数调用（位置C1）=====
    print("\n=== 开始结果分析 ===")
    analyze_model_performance(model, data, split_data, largest_cc, node_list, device)



    # 7. 可视化
    print("7. 可视化结果...")
    plot_training_history(history, "快速实验")
    
    return history['test_auc'], history['test_ap']

def create_synthetic_dataset():
    # """创建合成数据集（后备方案）"""
    print("创建合成数据集...")
    num_nodes = 5000
    num_edges = 20000
    
    # 使用BA模型创建更真实的图
    graph = nx.barabasi_albert_graph(num_nodes, 4)
    
    # 提取特征
    extractor = EnhancedFeatureExtractor(graph)
    features_dict = extractor.extract_simple_features()
    
    # 构建特征矩阵
    features_list = [features_dict[i] for i in range(num_nodes)]
    features_tensor = torch.tensor(features_list, dtype=torch.float32)
    
    # 构建边索引
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
    
    return data, split_data

# ==================== 可视化函数 ====================

def plot_training_history(history, model_name="模型"):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], 'b-', linewidth=2, label='训练损失')
    axes[0].set_xlabel('训练轮次')
    axes[0].set_ylabel('损失值')
    axes[0].set_title('训练损失曲线')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # AUC曲线
    if 'val_auc' in history and len(history['val_auc']) > 0:
        x_vals = list(range(5, len(history['val_auc']) * 5 + 1, 5))
        axes[1].plot(x_vals, history['val_auc'], 'r-o', linewidth=2, 
                    markersize=6, label='验证AUC')
        axes[1].set_xlabel('训练轮次')
        axes[1].set_ylabel('AUC分数')
        axes[1].set_title('验证AUC曲线')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # 标记最佳AUC
        best_auc = max(history['val_auc'])
        best_idx = history['val_auc'].index(best_auc)
        axes[1].scatter(x_vals[best_idx], best_auc, color='red', s=100, 
                       zorder=5, label=f'最佳AUC: {best_auc:.3f}')
        axes[1].legend()
    
    # AP曲线
    if 'val_ap' in history and len(history['val_ap']) > 0:
        x_vals = list(range(5, len(history['val_ap']) * 5 + 1, 5))
        axes[2].plot(x_vals, history['val_ap'], 'g-s', linewidth=2, 
                    markersize=6, label='验证AP')
        axes[2].set_xlabel('训练轮次')
        axes[2].set_ylabel('AP分数')
        axes[2].set_title('验证AP曲线')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # 标记最佳AP
        best_ap = max(history['val_ap'])
        best_idx = history['val_ap'].index(best_ap)
        axes[2].scatter(x_vals[best_idx], best_ap, color='red', s=100, 
                       zorder=5, label=f'最佳AP: {best_ap:.3f}')
        axes[2].legend()
    
    plt.suptitle(f'{model_name} - 测试AUC: {history["test_auc"]:.4f}, 测试AP: {history["test_ap"]:.4f}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    filename = f"models/{model_name}_训练过程.png"
    os.makedirs('models', exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {filename}")
    
    plt.show()

def full_real_data_experiment():
    """完整真实数据实验（使用更多数据和复杂特征）"""
    print("=" * 70)
    print("完整真实数据实验")
    print("=" * 70)
    
    # 1. 加载更多数据
    print("1. 加载完整数据...")
    edges_df = WikiDataLoader.load_wikilinks_data(
        sample_size=500000,      # 50万条边
        max_nodes=10000,         # 1万个节点
        min_degree=2             # 最小度要求
    )
    
    if edges_df is None or len(edges_df) == 0:
        print("无法加载完整数据，回退到快速实验")
        return quick_real_data_experiment()
    
    # 2. 构建图
    print("2. 构建图...")
    G = nx.Graph()
    G.add_edges_from(list(zip(edges_df['source'], edges_df['target'])))
    
    # 提取最大连通分量
    if nx.is_connected(G):
        largest_cc = G
    else:
        connected_components = list(nx.connected_components(G))
        largest_cc_nodes = max(connected_components, key=len)
        largest_cc = G.subgraph(largest_cc_nodes).copy()
    
    # 如果图太大，采样子图
    if largest_cc.number_of_nodes() > 5000:
        print(f"图太大 ({largest_cc.number_of_nodes()} 节点)，采样5000个节点的子图...")
        # 使用随机游走采样
        sampled_nodes = set()
        start_node = np.random.choice(list(largest_cc.nodes()))
        sampled_nodes.add(start_node)
        
        # 多轮随机游走采样
        for _ in range(50):
            current = np.random.choice(list(sampled_nodes))
            neighbors = list(largest_cc.neighbors(current))
            if neighbors:
                sampled_nodes.add(np.random.choice(neighbors))
            if len(sampled_nodes) >= 5000:
                break
        
        largest_cc = largest_cc.subgraph(sampled_nodes).copy()
    
    print(f"图: {largest_cc.number_of_nodes()} 节点, {largest_cc.number_of_edges()} 边")
    
    # 3. 提取完整特征
    print("3. 提取完整特征...")
    extractor = EnhancedFeatureExtractor(largest_cc)
    features_dict = extractor.extract_all_features()  # 使用完整特征
    
    # 4. 创建数据集
    print("4. 创建数据集...")
    node_list = list(largest_cc.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    # 构建特征矩阵
    features_list = [features_dict[node] for node in node_list]
    features_tensor = torch.tensor(features_list, dtype=torch.float32)
    
    # 构建边索引
    edge_list = []
    for src, dst in largest_cc.edges():
        src_idx = node_to_idx[src]
        dst_idx = node_to_idx[dst]
        edge_list.append([src_idx, dst_idx])
        edge_list.append([dst_idx, src_idx])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    data = Data(x=features_tensor, edge_index=edge_index, num_nodes=len(node_list))
    
    # 5. 划分数据集（更多测试数据）
    print("5. 划分数据集...")
    edges = list(largest_cc.edges())
    edge_indices = [(node_to_idx[src], node_to_idx[dst]) for src, dst in edges]
    
    # 去重（无向图）
    edge_set = set()
    for src, dst in edge_indices:
        if src < dst:
            edge_set.add((src, dst))
        else:
            edge_set.add((dst, src))
    
    edges = list(edge_set)
    
    train_edges, test_edges = train_test_split(edges, test_size=0.3, random_state=42)  # 30%测试
    train_edges, val_edges = train_test_split(train_edges, test_size=0.15, random_state=42)  # 15%验证
    
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
    
    print(f"数据集: {data.num_nodes} 节点, {len(edges)} 边")
    print(f"特征维度: {data.x.shape[1]}")
    print(f"划分: 训练{len(train_edges)}边, 验证{len(val_edges)}边, 测试{len(test_edges)}边")
    
    # 5.5 评估任务难度
    task_info = evaluate_task_difficulty(largest_cc, edges, node_list, node_to_idx)
    # 根据任务难度调整负采样难度
    if task_info['degree_difference'] > 0.5:
        difficulty = 'hard'
    elif task_info['degree_difference'] > 0.2:
        difficulty = 'medium'
    else:
        difficulty = 'easy'
    
    print(f"\n根据任务难度，使用 {difficulty} 级别负采样")



    # 6. 训练更强大的模型
    print("6. 训练增强模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = EnhancedGraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=256,
        out_channels=128,
        num_layers=4,
        dropout=0.5,
        use_residual=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # 训练
    model, history = train_link_prediction(
        model, data, split_data, optimizer, device,
        num_neg_samples=3,
        epochs=300,
        patience=25,
        scheduler=scheduler,
        difficulty_level=difficulty  # 新增参数
    )
    
    print(f"\n最终结果: AUC={history['test_auc']:.4f}, AP={history['test_ap']:.4f}")
    
    # 7. 详细分析
    print("\n=== 详细分析 ===")
    analyze_model_performance(model, data, split_data, largest_cc, node_list, device)
    
    # 8. 可视化
    print("7. 可视化结果...")
    plot_training_history(history, "完整实验")
    
    return history

def compare_experiments():
    """比较不同模型的实验"""
    print("=" * 70)
    print("模型比较实验")
    print("=" * 70)
    
    # 1. 加载数据（使用中等规模）
    print("1. 加载数据...")
    edges_df = WikiDataLoader.load_wikilinks_data(
        sample_size=200000,
        max_nodes=3000,
        min_degree=1
    )
    
    if edges_df is None or len(edges_df) == 0:
        print("无法加载数据，使用合成数据")
        data, split_data = create_synthetic_dataset()
        largest_cc = nx.Graph()
        largest_cc.add_edges_from(split_data['all_edges'])
        node_list = list(range(data.num_nodes))
    else:
        # 构建图
        print("2. 构建图...")
        G = nx.Graph()
        G.add_edges_from(list(zip(edges_df['source'], edges_df['target'])))
        
        # 提取最大连通分量
        if nx.is_connected(G):
            largest_cc = G
        else:
            connected_components = list(nx.connected_components(G))
            largest_cc_nodes = max(connected_components, key=len)
            largest_cc = G.subgraph(largest_cc_nodes).copy()
        
        # 限制大小以便快速比较
        if largest_cc.number_of_nodes() > 1000:
            print(f"采样1000个节点的子图...")
            nodes = list(largest_cc.nodes())[:1000]
            largest_cc = largest_cc.subgraph(nodes).copy()
        
        print(f"图: {largest_cc.number_of_nodes()} 节点, {largest_cc.number_of_edges()} 边")
        
        # 3. 提取特征
        print("3. 提取特征...")
        extractor = EnhancedFeatureExtractor(largest_cc)
        features_dict = extractor.extract_simple_features()
        
        # 4. 创建数据集
        print("4. 创建数据集...")
        node_list = list(largest_cc.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        features_list = [features_dict[node] for node in node_list]
        features_tensor = torch.tensor(features_list, dtype=torch.float32)
        
        edge_list = []
        for src, dst in largest_cc.edges():
            src_idx = node_to_idx[src]
            dst_idx = node_to_idx[dst]
            edge_list.append([src_idx, dst_idx])
            edge_list.append([dst_idx, src_idx])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        data = Data(x=features_tensor, edge_index=edge_index, num_nodes=len(node_list))
        
        # 5. 划分数据集
        print("5. 划分数据集...")
        edges = list(largest_cc.edges())
        edge_indices = [(node_to_idx[src], node_to_idx[dst]) for src, dst in edges]
        
        edge_set = set()
        for src, dst in edge_indices:
            if src < dst:
                edge_set.add((src, dst))
            else:
                edge_set.add((dst, src))
        
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
    
    print(f"数据集: {data.num_nodes} 节点, {len(edges)} 边")
    print(f"特征维度: {data.x.shape[1]}")
    
    # 6. 定义要比较的模型
    print("\n6. 训练并比较不同模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def compare_experiments():
        """改进的模型比较实验"""
        print("=" * 70)
        print("改进的模型比较实验 - 公平评估版")
        print("=" * 70)
        
        # ... 数据加载和预处理 ...
        
        # 在训练每个模型前，重置随机种子确保公平
        import random
        
        models_config = [
            {
                'name': 'GraphSAGE',
                'class': EnhancedGraphSAGE,
                'params': {
                    'in_channels': data.x.shape[1],
                    'hidden_channels': 128,
                    'out_channels': 64,
                    'num_layers': 2,
                    'dropout': 0.3,  # 添加dropout
                    'use_residual': True
                },
                'lr': 0.01
            },
            {
                'name': 'GCN',
                'class': BaseGNN,
                'params': {
                    'in_channels': data.x.shape[1],
                    'hidden_channels': 128,
                    'out_channels': 64,
                    'model_type': 'gcn'
                },
                'lr': 0.01
            },
            {
                'name': 'GAT',
                'class': BaseGNN,
                'params': {
                    'in_channels': data.x.shape[1],
                    'hidden_channels': 128,
                    'out_channels': 64,
                    'model_type': 'gat'
                },
                'lr': 0.005  # GAT通常需要更小的学习率
            },
            {
                'name': 'Enhanced GraphSAGE',
                'class': EnhancedGraphSAGE,
                'params': {
                    'in_channels': data.x.shape[1],
                    'hidden_channels': 256,
                    'out_channels': 128,
                    'num_layers': 3,
                    'dropout': 0.4,  # 更强的正则化
                    'use_residual': True
                },
                'lr': 0.005
            }
        ]
        
        results = {}
        histories = {}

        for config in models_config:
            print(f"\n{'='*60}")
            print(f"训练模型: {config['name']}")
            print(f"{'='*60}")
            
            # 重置随机种子，确保公平比较
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            
            # 创建模型
            model = config['class'](**config['params']).to(device)
            
            # 计算模型参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"模型参数: 总共 {total_params:,}, 可训练 {trainable_params:,}")
            
            # 优化器
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=5e-4)
            
            # 训练（使用改进的训练函数）
            model, history = train_link_prediction(
                model, data, split_data, optimizer, device,
                num_neg_samples=2,
                epochs=100,
                patience=20
            )
            
            # 记录结果
            results[config['name']] = {
                'test_auc': history['test_auc'],
                'test_ap': history['test_ap'],
                'best_val_auc': history['best_val_auc'],
                'best_epoch': history['best_epoch'],
                'params': trainable_params,
                'final_loss': history['train_loss'][-1] if history['train_loss'] else None
            }
            
            # 详细分析该模型的表现
            print(f"\n{config['name']} 详细分析:")
            print(f"  最终测试 AUC: {history['test_auc']:.4f}")
            print(f"  最终测试 AP:  {history['test_ap']:.4f}")
            print(f"  最佳验证 AUC: {history['best_val_auc']:.4f} (第 {history['best_epoch']} 轮)")
            
            if len(history['train_loss']) > 0:
                final_loss = history['train_loss'][-1]
                min_loss = min(history['train_loss'])
                print(f"  最终训练损失: {final_loss:.4f}, 最小损失: {min_loss:.4f}")
        
        # ... 结果比较和可视化 ...
            histories[config['name']] = history
        
        print(f"{config['name']} 完成: AUC={history['test_auc']:.4f}, AP={history['test_ap']:.4f}")
        # 7. 比较结果
        print("\n" + "="*70)
        print("模型比较结果")
        print("="*70)
        
        print("\n性能对比表:")
        print("-"*70)
        print(f"{'模型':<25} {'测试AUC':<10} {'测试AP':<10} {'最佳验证AUC':<15} {'最佳轮次':<10}")
        print("-"*70)
        
        for model_name, result in sorted(results.items(), key=lambda x: x[1]['test_auc'], reverse=True):
            print(f"{model_name:<25} {result['test_auc']:<10.4f} {result['test_ap']:<10.4f} "
                f"{result['best_val_auc']:<15.4f} {result['best_epoch']:<10}")
        
        print("-"*70)
        
        # 8. 可视化比较结果
        print("\n7. 生成比较图表...")
        plot_comparison_results(results, histories)

        # 保存比较结果
        os.makedirs('models', exist_ok=True)
        with open('models/comparison_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n比较结果已保存到: models/comparison_results.json")
        
        return results, histories

def plot_comparison_results(results, histories):
    """绘制模型比较结果"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 性能对比条形图
    model_names = list(results.keys())
    test_aucs = [results[name]['test_auc'] for name in model_names]
    test_aps = [results[name]['test_ap'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0].bar(x - width/2, test_aucs, width, label='测试AUC', color='skyblue')
    axes[0].bar(x + width/2, test_aps, width, label='测试AP', color='lightcoral')
    axes[0].set_xlabel('模型')
    axes[0].set_ylabel('分数')
    axes[0].set_title('模型性能对比')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 在每个条形上添加数值
    for i, (auc, ap) in enumerate(zip(test_aucs, test_aps)):
        axes[0].text(i - width/2, auc + 0.01, f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, ap + 0.01, f'{ap:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 训练损失对比
    axes[1].set_title('训练损失对比')
    for model_name, history in histories.items():
        if 'train_loss' in history:
            axes[1].plot(history['train_loss'], label=model_name, linewidth=2)
    axes[1].set_xlabel('训练轮次')
    axes[1].set_ylabel('损失值')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 验证AUC对比
    axes[2].set_title('验证AUC对比')
    for model_name, history in histories.items():
        if 'val_auc' in history and len(history['val_auc']) > 0:
            x_vals = list(range(5, len(history['val_auc']) * 5 + 1, 5))
            axes[2].plot(x_vals, history['val_auc'], 'o-', label=model_name, linewidth=2, markersize=4)
    axes[2].set_xlabel('训练轮次')
    axes[2].set_ylabel('验证AUC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('models', exist_ok=True)
    filename = "models/模型比较结果.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"比较图表已保存: {filename}")
    
    plt.show()

def compute_baseline_performance(graph, edges, test_edges, num_neg_samples=1):
    """计算基线模型性能"""
    print("\n=== 基线模型性能 ===")
    
    # 1. 随机猜测
    print("1. 随机猜测:")
    print(f"   AUC ≈ 0.5000")
    print(f"   AP ≈ {len(test_edges)/(len(test_edges)*(1+num_neg_samples)):.4f}")
    
    # 2. 基于节点度的启发式方法
    degrees = dict(graph.degree())
    
    # 计算训练集的度特征
    train_nodes = set()
    for src, dst in edges:
        train_nodes.add(src)
        train_nodes.add(dst)
    
    # 简单的度乘积启发式
    test_scores = []
    test_labels = []
    
    for src, dst in test_edges:
        # 正样本
        src_deg = degrees.get(src, 1)
        dst_deg = degrees.get(dst, 1)
        score = (src_deg * dst_deg) / (max(degrees.values()) ** 2)
        test_scores.append(score)
        test_labels.append(1)
        
        # 负样本（随机生成）
        for _ in range(num_neg_samples):
            while True:
                neg_src = np.random.choice(list(train_nodes))
                neg_dst = np.random.choice(list(train_nodes))
                if neg_src != neg_dst and (neg_src, neg_dst) not in edges and (neg_dst, neg_src) not in edges:
                    break
            src_deg = degrees.get(neg_src, 1)
            dst_deg = degrees.get(neg_dst, 1)
            score = (src_deg * dst_deg) / (max(degrees.values()) ** 2)
            test_scores.append(score)
            test_labels.append(0)
    
    # 计算AUC和AP
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(test_labels, test_scores)
    ap = average_precision_score(test_labels, test_scores)
    
    print(f"2. 度乘积启发式:")
    print(f"   AUC = {auc:.4f}")
    print(f"   AP = {ap:.4f}")
    
    return {
        'random_auc': 0.5,
        'random_ap': len(test_edges)/(len(test_edges)*(1+num_neg_samples)),
        'degree_heuristic_auc': auc,
        'degree_heuristic_ap': ap
    }

# =======================分析函数定义======================
def analyze_model_performance(model, data, split_data, graph, node_list, device):
    """深入分析模型表现"""
    model.eval()
    
    # 1. 准备数据
    degrees = dict(graph.degree())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    # 按节点度分类
    results_by_degree = {'低度(1-2)': [], '中度(3-5)': [], '高度(6+)': []}
    
    # 2. 分析测试集边
    test_edges = split_data['test']['edges']
    
    with torch.no_grad():
        # 编码所有节点
        z = model.encode(data.x.to(device), split_data['train']['edge_index'].to(device))
        
        for src_idx, dst_idx in test_edges:
            src_node = node_list[src_idx]
            dst_node = node_list[dst_idx]
            
            avg_deg = (degrees.get(src_node, 0) + degrees.get(dst_node, 0)) / 2
            
            # 获取预测分数
            edge_tensor = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long).to(device)
            score = model.decode(z, edge_tensor).item()
            
            # 分类
            if avg_deg <= 2:
                results_by_degree['低度(1-2)'].append(score)
            elif avg_deg <= 5:
                results_by_degree['中度(3-5)'].append(score)
            else:
                results_by_degree['高度(6+)'].append(score)
    
    # 3. 打印分析结果
    print("\n=== 按节点度分析测试集边 ===")
    for deg_type, scores in results_by_degree.items():
        if scores:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"{deg_type}: 平均分数={avg_score:.3f}±{std_score:.3f}, 数量={len(scores)}")
    
    # 4. 分析训练-测试重叠
    train_nodes = set()
    for src, dst in split_data['train']['edges']:
        train_nodes.add(src)
        train_nodes.add(dst)
    
    test_nodes = set()
    for src, dst in test_edges:
        test_nodes.add(src)
        test_nodes.add(dst)
    
    overlap = len(test_nodes & train_nodes)
    print(f"\n=== 数据划分分析 ===")
    print(f"训练集节点数: {len(train_nodes)}")
    print(f"测试集节点数: {len(test_nodes)}")
    print(f"重叠节点数: {overlap} ({overlap/len(test_nodes)*100:.1f}%)")
    
    return results_by_degree

# ==================== 主函数 ====================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练GNN链路预测模型')
    parser.add_argument('--mode', type=str, default='quick', 
                       choices=['quick', 'real', 'compare', 'single', 'visualize'],
                       help='运行模式: quick(快速实验), real(真实数据), compare(比较实验), single(单个模型), visualize(可视化)')
    parser.add_argument('--model', type=str, default='enhanced_sage', 
                       choices=['gcn', 'gat', 'sage', 'enhanced_sage'],
                       help='模型类型')
    parser.add_argument('--config', type=str, default='baseline',
                       help='实验配置')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        # 快速实验
        quick_real_data_experiment()
        
    elif args.mode == 'real':
        # 完整真实数据实验
        full_real_data_experiment()
        
    elif args.mode == 'compare':
        # 比较实验
        compare_experiments()
        
    elif args.mode == 'single':
        # 训练单个模型
        print(f"训练单个模型: {args.model}")
        # 这里可以扩展
        print("使用完整实验模式代替...")
        full_real_data_experiment()
        
    elif args.mode == 'visualize':
        # 可视化结果
        print("可视化结果...")
        try:
            with open('models/training_history.json', 'r', encoding='utf-8') as f:
                history = json.load(f)
            plot_training_history(history, "最佳模型")
        except:
            print("未找到训练历史文件")
            # 尝试加载比较结果
            try:
                with open('models/comparison_results.json', 'r', encoding='utf-8') as f:
                    results = json.load(f)
                plot_comparison_results(results, {})
            except:
                print("也未找到比较结果文件")
        
    else:  # 默认快速实验
        print("使用快速实验模式...")
        quick_real_data_experiment()


if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    main()