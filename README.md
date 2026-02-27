# WikiLinks图神经网络链路预测 WikiLinks Graph Neural Network Link Prediction
[![bilibili](https://img.shields.io/badge/🎥-Video%20on%20Bilibili-red)](https://www.bilibili.com/video/BV1j4zkBVEgu/?p=5&share_source=copy_web&vd_source=56cdc7ef44ed1ee2c9b9515febf8e9ce&t=0)

[![githubio](https://img.shields.io/badge/🤗-github.io-blue)](https://inneedlovebu.github.io/wikinet-link-prediction/)
[![GitHub](https://img.shields.io/badge/📂-GitHub-black)](https://github.com/inneedloveBu/wikinet-link-prediction)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inneedloveBu/wikinet-link-prediction/blob/main/notebooks/WikiLinks_Demo.ipynb)


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![AUC](https://img.shields.io/badge/AUC-0.889-orange)
<img width="1735" height="900" alt="training_progress_chinese_202601241601_final" src="https://github.com/user-attachments/assets/85d2b794-b492-4c60-aea4-2fca9d658d7e" />
<img width="1735" height="900" alt="training_progress_english_202601241600_final" src="https://github.com/user-attachments/assets/b227e98d-7cb2-43de-920c-4a1a4f3f4dc8" />

## 📊 Experimental Results & Visualization  
The following figure shows the trend of loss decrease and AUC increase during model training:  
<img src="https://raw.githubusercontent.com/inneedloveBu/wikinet-link-prediction/main/animations/training_progress_english_20260124_160006.gif" alt="Training progress animation" style="max-width: 100%; border: 1px solid #ddd;" />  
https://github.com/inneedloveBu/wikinet-link-prediction/animations/training_progress_english_20260124_160006.gif  
![Training progress gif](https://raw.githubusercontent.com/inneedloveBu/wikinet-link-prediction/main/animations/training_progress_chinese_20260124_160116.gif)  


下面是按照你要求：

* 一级标题 `#`
* 二级标题 `##`
* 三级标题 `###`
* 所有带点条目统一改为 `- **xxx**` 规范格式
* 结构完全统一
* 删除重复版本
* 数值统一为最终实验版本
* AUC 统一为 Test AUC 0.798

整理后的**完整终版 README**如下：

---

# WikiNet: Hardness-Aware Link Prediction with Graph Neural Networks

[![bilibili](https://img.shields.io/badge/🎥-Video%20on%20Bilibili-red)](https://www.bilibili.com/video/BV1j4zkBVEgu/?p=5)
[![githubio](https://img.shields.io/badge/🤗-github.io-blue)](https://inneedlovebu.github.io/wikinet-link-prediction/)
[![GitHub](https://img.shields.io/badge/📂-GitHub-black)](https://github.com/inneedloveBu/wikinet-link-prediction)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inneedloveBu/wikinet-link-prediction/blob/main/notebooks/WikiLinks_Demo.ipynb)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![AUC](https://img.shields.io/badge/Test%20AUC-0.798-orange)

---

# Overview

WikiNet is a graph representation learning project designed to perform link prediction on a Wikipedia hyperlink network.

The task is formulated as a binary classification problem over node pairs.

The project focuses on:

* **Structural feature engineering**
* **Hardness-aware negative sampling**
* **Multi-interaction edge decoding**
* **Robust training stabilization techniques**

---

# Dataset

The graph is constructed from the WikiLinks dataset.

## Experimental Configuration

* **Number of nodes: 1000**
* **Number of edges: 5000**
* **Node feature dimension: 20 (structural + content features)**

## Data Split

* **Training edges: 1771**
* **Validation edges: 253**
* **Test edges: 507**

---

# Feature Engineering

## Structural Features (via NetworkX)

* **Normalized degree**
* **Clustering coefficient**
* **Betweenness centrality**
* **PageRank score**
* **Neighbor statistics (mean / std / max / min degree)**

## Content-Based Features

* **Title length**
* **Word count**
* **Numeric token count**
* **Uppercase ratio**
* **Special character ratio**
* **Hash-based embedding component**

All features are standardized before training.

---

# Model Architecture

## Node Encoder

A 3-layer MLP encoder:

* **Linear → BatchNorm → ReLU → Dropout**

* **Linear → BatchNorm → ReLU → Dropout**

* **Linear projection layer**

* **Hidden dimension: 64**

* **Total trainable parameters: 21,729**

---

# Edge Interaction Mechanism

Instead of naive concatenation, the decoder integrates multiple interaction patterns:

* **Node embedding u**
* **Node embedding v**
* **Absolute difference |u − v|**
* **Element-wise product u ⊙ v**

These representations are concatenated and passed through an MLP classifier.

This design increases expressive power for link prediction tasks.

---

# Hardness-Aware Negative Sampling

Three difficulty levels were implemented:

* **Easy — Random negative sampling**
* **Medium — Common-neighbor-based sampling**
* **Hard — Degree-similarity-based sampling**

Final training uses:

* **Medium hardness sampling**

This generates more informative negative edges and improves generalization.

---

# Training Strategy

* **Binary Cross Entropy with logits**

* **Class-weight balancing**

* **Gradient clipping**

* **Early stopping (patience = 20)**

* **ReduceLROnPlateau learning rate scheduler**

* **Best model checkpointing**

* **Training epochs: 300**

* **Best validation AUC: 0.897 (epoch 110)**

---

# Results

## Test Set Performance

* **ROC-AUC: 0.798**
* **Average Precision (AP): 0.784**
* **F1-score: 0.763**
* **Accuracy: 0.696**

The model demonstrates stable generalization performance under controlled negative sampling.

---

# Technical Stack

* **PyTorch 2.0+**
* **PyTorch Geometric**
* **NetworkX**
* **NumPy**
* **Pandas**
* **scikit-learn**
* **Matplotlib**

---

# Project Structure

```bash
wikinet-link-prediction/
├── data/
│   ├── raw/
│   └── cleaned/
├── models/
├── notebooks/
├── train11.py
├── requirements.txt
└── README.md
```

---

# Quick Start

## 1. Clone Repository

```bash
git clone https://github.com/inneedloveBu/wikinet-link-prediction.git
cd wikinet-link-prediction
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run Training

```bash
python train11.py
```

---

# Key Contributions

* **Designed hardness-controlled negative sampling mechanism**
* **Implemented multi-interaction edge decoder**
* **Combined structural and lightweight content features**
* **Applied full training stabilization pipeline**
* **Conducted systematic multi-metric evaluation**

---

# Future Improvements

* **Replace MLP encoder with GCN / GAT**
* **Neighbor sampling for scalability**
* **Larger-scale graph experiments**
* **Embedding visualization**
* **Interactive web deployment**

---

# License

* **MIT License**

---

# Contact

* **GitHub Issues:** [https://github.com/inneedloveBu/wikinet-link-prediction/issues](https://github.com/inneedloveBu/wikinet-link-prediction/issues)
* **Email:** [indeedlove@foxmail.com](mailto:indeedlove@foxmail.com)

---


