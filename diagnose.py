# diagnose.py
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def diagnose_problem():
    """诊断程序卡住的问题"""
    print("=== 开始诊断 ===")
    
    # 1. 检查PyTorch
    print("1. 检查PyTorch...")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   当前设备: {torch.cuda.current_device()}")
        print(f"   设备名称: {torch.cuda.get_device_name(0)}")
    
    # 2. 检查内存
    print("\n2. 检查内存...")
    import psutil
    memory = psutil.virtual_memory()
    print(f"   总内存: {memory.total / 1e9:.2f} GB")
    print(f"   可用内存: {memory.available / 1e9:.2f} GB")
    print(f"   使用率: {memory.percent}%")
    
    # 3. 检查数据文件
    print("\n3. 检查数据文件...")
    import os
    data_path = "data/raw/enwiki.wikilink_graph.2018-03-01.csv.gz"
    if os.path.exists(data_path):
        size = os.path.getsize(data_path) / 1e9
        print(f"   数据文件存在: {data_path}")
        print(f"   文件大小: {size:.2f} GB")
    else:
        print(f"   数据文件不存在: {data_path}")
        print("   请下载数据集并放在正确位置")
    
    # 4. 创建简单测试
    print("\n4. 创建简单测试图...")
    G = nx.erdos_renyi_graph(100, 0.1)
    print(f"   测试图: {G.number_of_nodes()} 节点, {G.number_of_nodes()} 边")
    
    # 5. 测试小规模模型
    print("\n5. 测试小规模模型...")
    from torch_geometric.data import Data
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    
    # 创建简单数据
    num_nodes = 100
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    data = Data(x=x, edge_index=edge_index)
    
    # 简单模型
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(16, 32)
            self.conv2 = GCNConv(32, 16)
            self.lin = torch.nn.Linear(32, 1)
            
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 简单训练循环
    print("   开始简单训练测试（5轮）...")
    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        print(f"     轮次 {epoch+1}, 损失: {loss.item():.4f}")
    
    print("\n=== 诊断完成 ===")
    print("如果上述测试都能通过，问题可能出在数据加载或模型训练部分")
    
    return True

if __name__ == "__main__":
    diagnose_problem()