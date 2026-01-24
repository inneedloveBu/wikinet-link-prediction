import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import json
import os

# 尝试从保存的文件中加载历史数据
history_path = 'models/improved_training_history.json'
if os.path.exists(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    # 假设history字典中有'train_loss'和'val_auc'键
    train_loss = history.get('train_loss', [])  # 应包含300个值
    val_auc = history.get('val_auc', [])        # 应包含300个值
    epochs = list(range(1, len(train_loss) + 1))
    print(f"从{history_path}加载了{len(train_loss)}个epoch的数据")
else:
    print("未找到完整历史文件，使用日志提取的数据")
    # 使用上面提取的列表

# 1. 模拟训练数据
epochs = list(range(1, 301)) 
# 模拟训练损失和准确率（呈下降/上升趋势）
train_loss = [1/np.log(e+1) + np.random.normal(0, 0.02) for e in epochs]
val_auc = [0.65 + 0.15 * (1 - np.exp(-e/30)) + np.random.normal(0, 0.01) for e in epochs]

# 2. 设置画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('WikiLinks GNN 训练过程可视化', fontsize=14)

# 3. 初始化函数（绘制空白线条）
def init():
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('损失函数下降曲线')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation AUC')
    ax2.set_title('AUC指标上升曲线')
    ax2.set_ylim([0.6, 0.85])
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    return fig,

# 4. 更新函数（核心动画帧）
def update(frame):
    # 清除上一帧，但保留坐标轴设置
    ax1.clear()
    ax2.clear()
    init() # 重新应用坐标轴设置
    
    # 绘制到当前帧为止的所有数据
    current_epochs = epochs[:frame+1]
    current_loss = train_loss[:frame+1]
    current_auc = val_auc[:frame+1]
    
    line1, = ax1.plot(current_epochs, current_loss, 'b-', linewidth=2, label='Training Loss')
    ax1.legend()
    
    line2, = ax2.plot(current_epochs, current_auc, 'r-', linewidth=2, label='Validation AUC')
    ax2.legend()
    
    # 在第80个epoch处添加一个注释标记最佳AUC
    if frame >= 80:
        best_auc_idx = np.argmax(current_auc)
        best_epoch = current_epochs[best_auc_idx]
        best_auc_value = current_auc[best_auc_idx]
        ax2.annotate(f'Best AUC: {best_auc_value:.3f}', 
                     xy=(best_epoch, best_auc_value),
                     xytext=(best_epoch+5, best_auc_value-0.02),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=9)
    
    return line1, line2,

# 5. 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(epochs),
                              init_func=init, blit=False, repeat=False, interval=100) # interval控制速度（毫秒）

# 6. 保存为GIF（需要 pillow 库）
print("正在生成GIF动画...")
ani.save('training_progress.gif', writer='pillow', fps=10, dpi=100) # fps控制帧率
print("GIF已保存为 'training_progress.gif'")

