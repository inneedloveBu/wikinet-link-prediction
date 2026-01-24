# 创建 train_improved.py

"""
改进的GNN训练脚本（使用更好的特征）
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import json
from tqdm import tqdm

# 安全加载函数
def safe_torch_load(path):
    """安全加载torch文件"""
    try:
        return torch.load(path, weights_only=True)
    except:
        print(f"使用weights_only=False加载 {path}")
        return torch.load(path, weights_only=False)

def train_link_prediction_improved(model, data, split_data, optimizer, device, 
                                  num_neg_samples=1, epochs=100, patience=10):
    """改进的训练函数"""
    
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
    print(f"特征维度: {x.shape[1]}")
    print(f"训练正样本数: {len(train_pos_edges)}")
    print(f"验证集边数: {len(split_data['val']['edges'])}")
    print(f"测试集边数: {len(split_data['test']['edges'])}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # 生成负样本（更智能的负采样）
        num_nodes = data.num_nodes
        num_pos_edges = len(train_pos_edges)
        num_neg_edges = num_pos_edges * num_neg_samples
        
        # 方法1: 完全随机负采样
        neg_edges = []
        attempts = 0
        max_attempts = num_neg_edges * 10
        
        while len(neg_edges) < num_neg_edges and attempts < max_attempts:
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            
            # 过滤条件
            if src != dst and (src, dst) not in train_pos_edges and (dst, src) not in train_pos_edges:
                neg_edges.append((src, dst))
            
            attempts += 1
        
        # 如果负样本不足，补充随机负样本
        while len(neg_edges) < num_neg_edges:
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            if src != dst:
                neg_edges.append((src, dst))
        
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().to(device)
        
        # 准备正样本边索引
        pos_edge_index = torch.tensor(train_pos_edges, dtype=torch.long).t().to(device)
        
        # 前向传播
        _, pos_scores, neg_scores = model(x, train_edge_index, pos_edge_index, neg_edge_index)
        
        # 计算损失
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        
        # 添加L2正则化
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # 验证
        if epoch % 5 == 0:
            val_auc, val_ap = evaluate_improved(model, data, split_data['val']['edges'], 
                                               split_data['train']['edge_index'], device)
            val_aucs.append(val_auc)
            val_aps.append(val_ap)
            
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}')
            
            # 学习率调度
            if epoch > 20 and val_auc < best_val_auc:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95
            
            # 早停检查
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                
                # 保存最佳模型
                os.makedirs('models/saved_models', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_ap': val_ap,
                }, 'models/saved_models/best_model_improved.pt')
                
                print(f"  保存最佳模型 (Val AUC: {val_auc:.4f})")
            
            # 早停
            if epoch - best_epoch > patience:
                print(f"早停在第 {epoch} 轮")
                break
        else:
            if epoch % 1 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    # 加载最佳模型
    checkpoint_path = 'models/saved_models/best_model_improved.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = safe_torch_load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("警告: 未找到最佳模型检查点")
    
    # 最终测试
    test_auc, test_ap = evaluate_improved(model, data, split_data['test']['edges'], 
                                         split_data['train']['edge_index'], device)
    
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
    
    with open('models/saved_models/training_history_improved.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history

def evaluate_improved(model, data, eval_edges, train_edge_index, device, num_neg_samples=1):
    """改进的评估函数"""
    model.eval()
    
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = train_edge_index.to(device)
        
        # 编码节点
        z = model.encode(x, edge_index)
        
        # 准备正样本
        pos_edges = eval_edges
        pos_edge_index = torch.tensor(pos_edges, dtype=torch.long).t().to(device)
        
        # 生成负样本（与正样本相同数量）
        num_nodes = data.num_nodes
        num_pos = len(pos_edges)
        num_neg = num_pos * num_neg_samples
        
        # 采样负样本
        neg_edges = []
        while len(neg_edges) < num_neg:
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            
            # 确保不是正样本
            if src != dst and (src, dst) not in eval_edges and (dst, src) not in eval_edges:
                neg_edges.append((src, dst))
        
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().to(device)
        
        # 计算分数
        pos_scores = model.decode(z, pos_edge_index)
        neg_scores = model.decode(z, neg_edge_index)
        
        # 组合标签和预测
        y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))]).cpu().numpy()
        y_pred = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        
        # 计算指标
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
    
    return auc, ap

def train_with_improved_features(dataset_type='small', model_type='gcn'):
    """使用改进特征进行训练"""
    
    print(f"使用改进特征的{dataset_type.upper()}数据集训练")
    
    # 加载数据集
    if dataset_type == 'small':
        dataset_path = "data/processed/improved_features/small_improved.pt"
        if not os.path.exists(dataset_path):
            print("数据集不存在，先运行特征工程...")
            from feature_engineering import create_improved_dataset
            create_improved_dataset(use_small=True, use_node2vec=False)
            dataset = safe_torch_load(dataset_path)
        else:
            dataset = safe_torch_load(dataset_path)
        
        # 重建数据
        from torch_geometric.data import Data
        data = Data(
            x=dataset['data_dict']['x'],
            edge_index=dataset['data_dict']['edge_index'],
            num_nodes=dataset['data_dict']['num_nodes']
        )
        split_data = dataset['split_data']
        
        # 模型参数
        in_channels = data.x.shape[1]
        hidden_channels = 128
        out_channels = 64
        epochs = 50
        
    else:
        # 完整数据集（这里先使用小样本进行实验）
        print("注意: 完整数据集的特征工程可能需要较长时间")
        print("先使用采样数据进行实验...")
        
        from feature_engineering import create_improved_dataset
        data, split_data, _ = create_improved_dataset(use_small=False, use_node2vec=False)
        
        # 模型参数
        in_channels = data.x.shape[1]
        hidden_channels = 256
        out_channels = 128
        epochs = 30
    
    # 创建模型
    from models.link_prediction import LinkPredictionModel
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = LinkPredictionModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        model_type=model_type,
        dropout=0.3
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # 训练
    model, history = train_link_prediction_improved(
        model, data, split_data, optimizer, device,
        num_neg_samples=1, epochs=epochs, patience=10
    )
    
    return model, history, data, split_data

def compare_models():
    """比较不同模型和特征的性能"""
    print("=" * 60)
    print("模型性能比较实验")
    print("=" * 60)
    
    experiments = [
        {'name': 'GCN随机特征', 'dataset': 'small', 'model': 'gcn', 'features': 'random'},
        {'name': 'GCN改进特征', 'dataset': 'small', 'model': 'gcn', 'features': 'improved'},
        {'name': 'GAT改进特征', 'dataset': 'small', 'model': 'gat', 'features': 'improved'},
        {'name': 'GraphSAGE改进特征', 'dataset': 'small', 'model': 'sage', 'features': 'improved'},
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\n实验: {exp['name']}")
        print("-" * 40)
        
        if exp['features'] == 'random':
            # 使用原始训练脚本
            from train_fix import train_on_small_dataset
            model, history, data, split_data = train_on_small_dataset()
            test_auc = history['test_auc']
            test_ap = history['test_ap']
        else:
            # 使用改进特征
            model, history, data, split_data = train_with_improved_features(
                dataset_type=exp['dataset'], 
                model_type=exp['model']
            )
            test_auc = history['test_auc']
            test_ap = history['test_ap']
        
        results.append({
            '实验名称': exp['name'],
            '测试AUC': f"{test_auc:.4f}",
            '测试AP': f"{test_ap:.4f}",
            '特征维度': data.x.shape[1] if hasattr(data, 'x') else 'N/A',
            '模型类型': exp['model']
        })
        
        print(f"结果: AUC={test_auc:.4f}, AP={test_ap:.4f}")
    
    # 显示比较结果
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    
    for result in results:
        print(f"{result['实验名称']:20} | AUC: {result['测试AUC']:6} | AP: {result['测试AP']:6} | "
              f"特征: {result['特征维度']:3} | 模型: {result['模型类型']}")
    
    # 保存结果
    import pandas as pd
    df_results = pd.DataFrame(results)
    df_results.to_csv('models/saved_models/experiment_results.csv', index=False)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练改进的GNN模型')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'compare', 'full'],
                       help='运行模式: train(训练), compare(比较实验), full(完整数据集)')
    parser.add_argument('--dataset', type=str, default='small', 
                       choices=['small', 'full'],
                       help='数据集类型')
    parser.add_argument('--model', type=str, default='gcn', 
                       choices=['gcn', 'gat', 'sage'],
                       help='模型类型')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_with_improved_features(dataset_type=args.dataset, model_type=args.model)
    elif args.mode == 'compare':
        compare_models()
    elif args.mode == 'full':
        print("在完整数据集上训练（需要较长时间）...")
        train_with_improved_features(dataset_type='full', model_type=args.model)
