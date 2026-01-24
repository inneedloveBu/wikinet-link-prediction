# debug_data.py
from train import WikiDataLoader
import networkx as nx

print("=== 数据加载诊断 ===")

# 测试不同参数
test_params = [
    {"sample_size": 20000, "max_nodes": 300, "min_degree": 1},
    {"sample_size": 50000, "max_nodes": 500, "min_degree": 1},
    {"sample_size": 100000, "max_nodes": 800, "min_degree": 1},
]

for params in test_params:
    print(f"\n测试参数: {params}")
    edges_df = WikiDataLoader.load_wikilinks_data(**params)
    
    if edges_df is not None:
        G = nx.Graph()
        G.add_edges_from(list(zip(edges_df['source'], edges_df['target'])))
        print(f"结果: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")