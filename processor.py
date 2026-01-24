# data/processor.py
import pandas as pd
import numpy as np
import networkx as nx
from typing import Tuple, Dict
import torch
from torch_geometric.data import Data

class WikiGraphProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = config.get('data_dir', 'data/processed')
        
    def load_and_process(self, size='small') -> Data:
        """加载并处理数据"""
        if size == 'small':
            return self._create_small_dataset()
        else:
            return self._load_full_dataset()
    
    def _load_full_dataset(self) -> Data:
        """加载完整数据集"""
        edges_path = f"{self.data_dir}/wikilinks.parquet"
        
        if not os.path.exists(edges_path):
            raise FileNotFoundError(f"数据文件不存在: {edges_path}")
        
        # 加载边数据
        edges_df = pd.read_parquet(edges_path)
        
        # 构建图
        print(f"构建图中... ({len(edges_df)} 条边)")
        G = nx.from_pandas_edgelist(edges_df, 'source', 'target')
        
        # 提取最大连通分量
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
        print(f"最大连通分量: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        
        # 转换为PyG格式
        data = self._graph_to_pyg(G)
        
        return data