# training/trainer.py
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

class GNNTrainer:
    def __init__(self, model, optimizer, device, config):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
    def train(self, dataset):
        """训练循环"""
        self.model.train()
        history = {'train_loss': [], 'val_auc': [], 'val_ap': []}
        
        best_auc = 0
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # 训练步骤
            loss = self._train_step(dataset)
            history['train_loss'].append(loss)
            
            # 验证
            if epoch % self.config['eval_every'] == 0:
                auc, ap = self.evaluate(dataset, split='val')
                history['val_auc'].append(auc)
                history['val_ap'].append(ap)
                
                # 早停逻辑
                if auc > best_auc:
                    best_auc = auc
                    patience_counter = 0
                    self._save_checkpoint(epoch, auc)
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config['patience']:
                    print(f"早停在epoch {epoch}")
                    break
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
        
        return history
    
    def _train_step(self, dataset):
        """单次训练步骤"""
        self.optimizer.zero_grad()
        
        # 正负采样
        pos_edges, neg_edges = self._sample_edges(dataset)
        
        # 前向传播
        embeddings = self.model(dataset.x, dataset.edge_index)
        
        # 计算损失
        loss = self._compute_loss(embeddings, pos_edges, neg_edges)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()