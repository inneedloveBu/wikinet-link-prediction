# models/enhanced_models.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv

class EnhancedGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3, use_residual=True):
        super().__init__()
        # 简化的增强版实现
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else out_channels
            
            conv = SAGEConv(in_dim, out_dim)
            self.convs.append(conv)
            
            if i < num_layers - 1:
                self.bns.append(torch.nn.BatchNorm1d(out_dim))
        
        self.dropout = dropout
        self.use_residual = use_residual
        
    def forward(self, x, edge_index):
        x_all = x
        
        for i, conv in enumerate(self.convs[:-1]):
            x_res = x_all
            
            x_all = conv(x_all, edge_index)
            x_all = self.bns[i](x_all)
            x_all = F.relu(x_all)
            x_all = F.dropout(x_all, p=self.dropout, training=self.training)
            
            if self.use_residual and x_res.size(1) == x_all.size(1):
                x_all = x_all + x_res
        
        x_all = self.convs[-1](x_all, edge_index)
        return x_all