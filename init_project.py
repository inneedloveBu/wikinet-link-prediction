# init_project.py
import sys
import subprocess
import platform
import os
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("错误: Python 3.8或更高版本是必需的")
        return False
    print(f"Python版本: {sys.version}")
    return True

def check_cuda():
    """检查CUDA可用性"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA可用, 版本: {torch.version.cuda}")
            return True, torch.version.cuda
        else:
            print("CUDA不可用，将使用CPU")
            return False, None
    except ImportError:
        print("PyTorch未安装")
        return False, None

def create_directory_structure():
    """创建项目目录结构"""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "notebooks",
        "app",
        "experiments/configs",
        "experiments/results",
        "src/utils",
        "src/models"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")

def create_requirements_file():
    """创建requirements.txt文件"""
    requirements = """# PyTorch - 根据你的系统选择合适的版本
# CPU版本:
# torch==2.1.0
# torchvision==0.16.0
# torchaudio==2.1.0
# 下载链接: https://download.pytorch.org/whl/cpu

# PyTorch Geometric 依赖
# 注意: 需要根据PyTorch和CUDA版本选择正确的wheel
# torch-scatter==2.1.2
# torch-sparse==0.6.18
# torch-cluster==1.6.3
# torch-spline-conv==1.2.2
# torch-geometric==2.4.0

# 基础数据科学包
pandas==2.1.1
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.3

# 图处理
networkx==3.1

# 可视化
matplotlib==3.8.0
seaborn==0.13.0
plotly==5.17.0

# Web应用
streamlit==1.28.0

# 工具
tqdm==4.66.1
jupyter==1.0.0
requests==2.31.0
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("创建: requirements.txt")

def install_dependencies():
    """安装依赖包（不带PyTorch）"""
    print("安装基础依赖...")
    
    # 基础依赖列表（不包含PyTorch和PyG）
    base_deps = [
        "pandas==2.1.1",
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "networkx==3.1",
        "matplotlib==3.8.0",
        "seaborn==0.13.0",
        "plotly==5.17.0",
        "streamlit==1.28.0",
        "tqdm==4.66.1",
        "jupyter==1.0.0",
        "requests==2.31.0",
        "beautifulsoup4==4.12.2",
        "lxml==4.9.3"
    ]
    
    for dep in base_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✓ 安装成功: {dep.split('==')[0]}")
        except subprocess.CalledProcessError:
            print(f"✗ 安装失败: {dep.split('==')[0]}")
    
    print("\n基础依赖安装完成！")
    print("下一步：请根据你的系统手动安装PyTorch和PyTorch Geometric")

def main():
    """主函数"""
    print("=" * 50)
    print("WikiNet项目初始化")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        return
    
    # 创建目录结构
    print("\n1. 创建项目目录结构...")
    create_directory_structure()
    
    # 创建requirements.txt
    print("\n2. 创建配置文件...")
    create_requirements_file()
    
    # 创建基础脚本
    print("\n3. 创建基础脚本...")
    create_basic_scripts()
    
    # 询问是否安装依赖
    print("\n4. 安装依赖...")
    response = input("是否安装基础依赖包？(y/n): ").lower()
    if response == 'y':
        install_dependencies()
    else:
        print("跳过依赖安装")
    
    print("\n" + "=" * 50)
    print("初始化完成！")
    print("=" * 50)
    print("\n下一步：")
    print("1. 根据你的系统安装PyTorch:")
    print("   CPU版本: pip install torch torchvision torchaudio")
    print("   CUDA版本: 请访问 https://pytorch.org/get-started/locally/")
    print("\n2. 安装PyTorch Geometric:")
    print("   参考: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    print("\n3. 运行项目: python train.py")

def create_basic_scripts():
    """创建基础脚本文件"""
    
    # 创建配置文件
    config = """{
    "data": {
        "raw_path": "data/raw",
        "processed_path": "data/processed",
        "sample_size": 50000
    },
    "model": {
        "gnn_type": "gcn",
        "hidden_channels": 128,
        "out_channels": 64,
        "num_layers": 2,
        "dropout": 0.3
    },
    "training": {
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "epochs": 100,
        "batch_size": 32
    }
}
"""
    
    with open("config.json", "w") as f:
        f.write(config)
    
    # 创建.gitignore
    gitignore = """# 数据文件
data/raw/
data/processed/
*.csv
*.csv.gz
*.pt
*.pth

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/

# 虚拟环境
venv*/
env*/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统
.DS_Store
Thumbs.db

# 输出
models/*.png
experiments/results/
logs/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore)
    
    print("创建: config.json, .gitignore")

if __name__ == "__main__":
    main()