# WikiLinks图神经网络链路预测

一个基于PyTorch Geometric的图神经网络项目，用于维基百科链接图的链路预测任务。

## 📊 项目概述

本项目实现了对维基百科链接图的链路预测，使用改进的图神经网络模型和特征工程方法，取得了显著的效果提升。

### 主要成果
- **测试集AUC**: 0.7976
- **测试集AP**: 0.7841  
- **测试集F1分数**: 0.7627
- **准确率**: 0.6964

## 🏗️ 项目结构
wikinet/
├── data/ # 数据目录
│ ├── raw/ # 原始数据（需自行下载）
│ └── cleaned/ # 清洗后的数据
├── models/ # 模型文件
├── train11.py # 主训练脚本
├── requirements.txt # 依赖包列表
├── README.md # 项目说明
└── .gitignore # Git忽略文件

text

## 🚀 快速开始

### 1. 环境安装
```bash
# 克隆项目
git clone https://github.com/inneedloveBu/wikinet-link-prediction.git
cd wikinet-link-prediction

# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
2. 数据准备
下载WikiLinks数据集：

访问：https://zenodo.org/record/1193740

下载 enwiki.wikilink_graph.2018-03-01.csv.gz

将文件放置在 data/raw/ 目录下

3. 运行训练
bash
python train11.py
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

📈 实验结果
训练曲线
https://models/improved_experiment_results.png

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

创建功能分支 (git checkout -b feature/AmazingFeature)

提交更改 (git commit -m 'Add some AmazingFeature')

推送到分支 (git push origin feature/AmazingFeature)

开启 Pull Request

📄 许可证
本项目采用 MIT 许可证 - 查看 LICENSE 文件了解详情

🙏 致谢
数据来源：维基百科WikiLinks数据集

框架：PyTorch Geometric, NetworkX, scikit-learn

感谢所有开源社区的贡献者

📞 联系方式
如有问题或建议，请通过以下方式联系：

项目issue：https://github.com/inneedoveBu/wikinet-link-prediction/issues

邮件：indeedlove@foxmail.com

⭐ 如果这个项目对你有帮助，请给个Star！