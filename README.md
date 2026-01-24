# WikiLinks图神经网络链路预测
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![AUC](https://img.shields.io/badge/AUC-0.889-orange)
<img width="1735" height="900" alt="training_progress_chinese_202601241601_final" src="https://github.com/user-attachments/assets/85d2b794-b492-4c60-aea4-2fca9d658d7e" />
<img width="1735" height="900" alt="training_progress_english_202601241600_final" src="https://github.com/user-attachments/assets/b227e98d-7cb2-43de-920c-4a1a4f3f4dc8" />


## 📊 实验结果与可视化
下图展示了模型在训练过程中损失下降和AUC指标上升的趋势：
<img src="https://raw.githubusercontent.com/inneedloveBu/wikinet-link-prediction/main/animations/training_progress_english_202601241600_final.gif" alt="训练过程动画" style="max-width: 100%; border: 1px solid #ddd;" />
https://github.com/inneedloveBu/wikinet-link-prediction/tree/main/animations/training_progress_english_202601241600_final.gif
![训练进度动图](https://raw.githubusercontent.com/inneedloveBu/wikinet-link-prediction/main/animations/training_progress_english_202601241600_final.gif)
<img src="https://raw.githubusercontent.com/inneedloveBu/wikinet-link-prediction/main/animations/training_progress_chinese_202601241601_final.gif" width="50%" />

### 训练过程动态展示

<div align="center">
  <img src="https://raw.githubusercontent.com/inneedloveBu/wikinet-link-prediction/main/animations/training_progress_english_202601241600_final.gif" width="90%" alt="GNN训练进度">
</div>

一个基于PyTorch Geometric的图神经网络项目，用于维基百科链接图的链路预测任务。

## 📊 项目概述

本项目实现了对维基百科链接图的链路预测，使用改进的图神经网络模型和特征工程方法，取得了显著的效果提升。

### 主要成果
- **测试集AUC**: 0.7976
- **测试集AP**: 0.7841  
- **测试集F1分数**: 0.7627
- **准确率**: 0.6964

## 🏗️ 项目结构
```bash
wikinet/
├── data/ # 数据目录
│ ├── raw/ # 原始数据（需自行下载）
│ └── cleaned/ # 清洗后的数据
├── models/ # 模型文件
├── train11.py # 主训练脚本
├── requirements.txt # 依赖包列表
├── README.md # 项目说明
└── .gitignore # Git忽略文件
```


## 🚀 快速开始

### 1. 环境安装

# 克隆项目
```bash
git clone https://github.com/inneedloveBu/wikinet-link-prediction.git
cd wikinet-link-prediction
```
# 创建虚拟环境（可选）
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```
2. 数据准备
下载WikiLinks数据集：

访问：https://zenodo.org/record/1193740

下载 enwiki.wikilink_graph.2018-03-01.csv.gz

将文件放置在 data/raw/ 目录下

3. 运行训练
`python train11.py`

🔬 技术特点
数据预处理
连通分量提取：自动提取最大连通分量

数据增强：智能添加随机边以解决稀疏问题

特征工程：结合结构特征和内容特征

模型架构
简化但有效的模型设计：21,729个参数

多种特征交互方式：拼接、差值、乘积

正则化策略：Dropout + BatchNorm

训练策略
困难负采样：按不同难度级别生成负样本

早停机制：自动保存最佳模型

学习率调度：动态调整学习率


关键指标
指标	数值	说明
测试集AUC	0.7976	分类器性能优秀
测试集AP	0.7841	精度-召回平衡良好
F1分数	0.7627	综合性能指标
准确率	0.6964	基础分类准确度
图结构分析
节点数: 114

边数: 700

边密度: 10.87%

平均度: 12.28

聚类系数: 0.4368

📂 文件说明
主要脚本
train11.py：主训练脚本，包含数据加载、特征提取、模型训练和评估

输出文件
data/cleaned/：清洗后的数据文件

cleaned_edges.txt：清洗后的边数据

cleaned_nodes.txt：清洗后的节点数据

graph_stats.json：图统计信息

models/：模型和结果文件

best_improved_model.pt：最佳模型权重

improved_training_history.json：训练历史

improved_experiment_results.png：可视化图表

🛠️ 自定义配置
你可以通过修改以下参数来调整实验：

python
# 在train11.py的main()函数中修改
target_nodes = 150      # 目标节点数
target_edges = 700      # 目标边数
num_epochs = 300        # 训练轮数
hidden_dim = 64         # 隐藏层维度
learning_rate = 0.01    # 学习率
🤝 贡献指南
欢迎贡献！请遵循以下步骤：

Fork 本仓库

创建功能分支 `git checkout -b feature/AmazingFeature`

提交更改 `git commit -m 'Add some AmazingFeature'`

推送到分支 `git push origin feature/AmazingFeature)`

开启 `Pull Request`

📄 许可证
本项目采用 MIT 许可证 - 查看 LICENSE 文件了解详情

🙏 致谢
数据来源：维基百科WikiLinks数据集

框架：PyTorch Geometric, NetworkX, scikit-learn

## 📚 参考文献与致谢

本项目在实现过程中参考或基于以下优秀的研究工作，在此向原作者致谢：

1.  **图卷积网络 (GCN) 的奠基工作**：
    ```bibtex
    @article{kipf2016semi,
      title={Semi-Supervised Classification with Graph Convolutional Networks},
      author={Kipf, Thomas N. and Welling, Max},
      journal={arXiv preprint arXiv:1609.02907},
      year={2016}
    }
    ```
2.  **大规模图表示学习**：
    ```bibtex
    @inproceedings{hamilton2017inductive,
      title={Inductive Representation Learning on Large Graphs},
      author={Hamilton, Will and Ying, Rex and Leskovec, Jure},
      booktitle={Advances in Neural Information Processing Systems},
      pages={1024--1034},
      year={2017}
    }
    ```
3.  **链路预测的经典方法**：
    - Liben-Nowell, D., & Kleinberg, J. (2007). The link-prediction problem for social networks. *Journal of the American Society for Information Science and Technology*.

**如果本项目的代码或思路对您的研究有帮助，请考虑引用上述相关文献。**


📞 联系方式
如有问题或建议，请通过以下方式联系：

项目issue：https://github.com/inneedoveBu/wikinet-link-prediction/issues

邮件：indeedlove@foxmail.com

⭐ 如果这个项目对你有帮助，请给个Star！
