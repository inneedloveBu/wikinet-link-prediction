# 创建修复版本 analyze_data_fixed.py
import pandas as pd
import numpy as np
import os
import networkx as nx  # 添加这行
import json
from pathlib import Path

def analyze_wiki_topcats(file_path, nrows=1000000):
    """分析wiki_topcats数据"""
    print(f"分析 {file_path}...")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    # 先查看文件内容结构
    print("查看文件前几行原始内容:")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i in range(10):
            line = f.readline().strip()
            if line:
                print(f"  {i+1}: {line}")
    
    # 尝试不同的分隔符读取
    try:
        # 尝试空格分隔
        df = pd.read_csv(file_path, sep=' ', header=None, nrows=nrows)
        print(f"成功用空格分隔读取，形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"前5行数据:\n{df.head()}")
        
        # 重命名列
        df.columns = ['source', 'target']
        
        # 统计信息
        print(f"\n数据统计:")
        print(f"节点数（唯一ID数）: {pd.concat([df['source'], df['target']]).nunique()}")
        print(f"边数: {len(df)}")
        print(f"最大节点ID: {pd.concat([df['source'], df['target']]).max()}")
        print(f"最小节点ID: {pd.concat([df['source'], df['target']]).min()}")
        
        # 检查是否有重复边
        duplicates = df.duplicated().sum()
        print(f"重复边数: {duplicates}")
        
        return df
        
    except Exception as e:
        print(f"读取失败: {e}")
        return None

def create_graph_from_data(df):
    """从数据框创建NetworkX图"""
    print("\n创建图...")
    G = nx.Graph()
    
    # 添加边
    edges = list(zip(df['source'], df['target']))
    G.add_edges_from(edges)
    
    print(f"图信息:")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    print(f"  连通分量数: {nx.number_connected_components(G)}")
    
    # 计算连通分量大小
    components = list(nx.connected_components(G))
    component_sizes = [len(c) for c in components]
    
    if component_sizes:
        print(f"  最大连通分量大小: {max(component_sizes)}")
        print(f"  最小连通分量大小: {min(component_sizes)}")
    
    return G

def main():
    # 确保数据目录存在
    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)
    
    # 分析 wiki_topcats.csv
    topcats_path = os.path.join(data_dir, "wiki_topcats.csv")
    if os.path.exists(topcats_path):
        df = analyze_wiki_topcats(topcats_path, nrows=1000000)  # 读取100万行
        
        if df is not None:
            # 创建并保存图
            G = create_graph_from_data(df)
            
            # 保存处理后的数据
            processed_dir = "data/processed"
            os.makedirs(processed_dir, exist_ok=True)
            
            # 保存为Parquet格式（更小更快）
            output_path = os.path.join(processed_dir, "wiki_graph_edges.parquet")
            df.to_parquet(output_path, index=False)
            print(f"\n已保存处理后的边列表到: {output_path}")
            
            # 保存图的统计信息
            stats = {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'num_components': nx.number_connected_components(G),
                'max_component_size': len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0
            }
            
            stats_path = os.path.join(processed_dir, "graph_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"已保存图统计信息到: {stats_path}")
            
            # 保存最大连通分量
            largest_cc = max(nx.connected_components(G), key=len)
            print(f"\n最大连通分量有 {len(largest_cc)} 个节点")
            
            # 创建最大连通分量子图
            subgraph = G.subgraph(largest_cc).copy()
            
            # 保存子图的边
            subgraph_edges = list(subgraph.edges())
            subgraph_df = pd.DataFrame(subgraph_edges, columns=['source', 'target'])
            subgraph_path = os.path.join(processed_dir, "largest_component_edges.parquet")
            subgraph_df.to_parquet(subgraph_path, index=False)
            print(f"已保存最大连通分量边列表到: {subgraph_path}")
            
    else:
        print(f"文件不存在: {topcats_path}")

if __name__ == "__main__":
    main()