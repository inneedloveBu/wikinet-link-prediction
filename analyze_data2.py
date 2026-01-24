# analyze_data.py
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def analyze_data_problems():
    """分析数据问题"""
    print("=== 数据问题分析 ===")
    
    # 1. 重新加载数据
    from train import WikiDataLoader
    
    print("1. 加载数据...")
    edges_df = WikiDataLoader.load_wikilinks_data(
        sample_size=200000,
        max_nodes=3000,
        min_degree=1
    )
    
    if edges_df is None:
        print("无法加载数据")
        return
    
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
    
    # 3. 分析图结构
    print("\n3. 图结构分析:")
    
    # 度分布
    degrees = dict(largest_cc.degree())
    degree_values = list(degrees.values())
    
    print(f"  平均度: {np.mean(degree_values):.2f}")
    print(f"  最大度: {max(degree_values)}")
    print(f"  最小度: {min(degree_values)}")
    
    # 检查是否有星型结构
    max_degree_node = max(degrees.items(), key=lambda x: x[1])
    print(f"  最高度节点: {max_degree_node[0]} (度={max_degree_node[1]})")
    
    # 统计度分布
    degree_counts = Counter(degree_values)
    print(f"  度分布: {sorted(degree_counts.items())[:10]}...")
    
    # 4. 检查连通性
    print("\n4. 连通性分析:")
    if nx.is_connected(largest_cc):
        print("  图是连通的")
    else:
        print(f"  图有 {nx.number_connected_components(largest_cc)} 个连通分量")
    
    # 5. 检查负样本问题
    print("\n5. 负样本分析:")
    
    # 创建节点映射
    node_list = list(largest_cc.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    # 随机采样一些负样本
    edges = list(largest_cc.edges())
    edge_set = set()
    for src, dst in edges:
        src_idx = node_to_idx[src]
        dst_idx = node_to_idx[dst]
        edge_set.add((min(src_idx, dst_idx), max(src_idx, dst_idx)))
    
    edges = list(edge_set)
    
    # 采样负样本
    num_nodes = len(node_list)
    num_neg = 100
    neg_edges = []
    
    for _ in range(num_neg * 10):  # 多次尝试
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src == dst:
            continue
        
        edge = (min(src, dst), max(src, dst))
        if edge not in edge_set:
            neg_edges.append(edge)
        
        if len(neg_edges) >= num_neg:
            break
    
    # 分析负样本的特征
    print(f"  采样了 {len(neg_edges)} 个负样本")
    
    # 计算负样本节点的平均度
    neg_node_degrees = []
    for src_idx, dst_idx in neg_edges[:20]:  # 只看前20个
        src_node = node_list[src_idx]
        dst_node = node_list[dst_idx]
        neg_node_degrees.append(degrees.get(src_node, 0))
        neg_node_degrees.append(degrees.get(dst_node, 0))
    
    if neg_node_degrees:
        print(f"  负样本节点的平均度: {np.mean(neg_node_degrees):.2f}")
        print(f"  负样本节点的最大度: {max(neg_node_degrees)}")
        print(f"  负样本节点的最小度: {min(neg_node_degrees)}")
    
    # 6. 可视化度分布
    print("\n6. 生成可视化...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 度分布直方图
    axes[0].hist(degree_values, bins=30, alpha=0.7, color='blue')
    axes[0].set_xlabel('节点度')
    axes[0].set_ylabel('频数')
    axes[0].set_title('节点度分布')
    axes[0].grid(True, alpha=0.3)
    
    # 累积度分布
    sorted_degrees = sorted(degree_values, reverse=True)
    cumulative = np.cumsum(sorted_degrees) / np.sum(sorted_degrees)
    
    axes[1].plot(range(1, len(cumulative) + 1), cumulative, 'r-', linewidth=2)
    axes[1].set_xlabel('节点排名')
    axes[1].set_ylabel('累积度比例')
    axes[1].set_title('累积度分布')
    axes[1].grid(True, alpha=0.3)
    
    # 标记80%的点
    idx_80 = np.where(cumulative >= 0.8)[0][0]
    axes[1].axvline(x=idx_80, color='green', linestyle='--', alpha=0.7, label=f'80% at {idx_80}')
    axes[1].axhline(y=0.8, color='green', linestyle='--', alpha=0.7)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  图表已保存: data_analysis.png")
    
    plt.show()
    
    # 7. 建议
    print("\n7. 改进建议:")
    if max_degree_node[1] > num_nodes * 0.5:
        print("  ⚠️ 警告: 存在极高度节点（可能为中心节点）")
        print("    建议: 过滤掉度太大的节点，或使用更平衡的图")
    
    if np.mean(degree_values) < 2:
        print("  ⚠️ 警告: 平均度过低，图可能过于稀疏")
        print("    建议: 增加最小度过滤条件")
    
    return largest_cc, degrees

if __name__ == "__main__":
    analyze_data_problems()