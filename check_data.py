# 创建 check_data.py 文件
"""
检查数据集并创建用于训练的数据
"""
import os
import gzip
import pandas as pd
from tqdm import tqdm
import numpy as np

def find_data_file():
    """查找数据文件"""
    possible_paths = [
        "data/raw/enwiki.wikilink_graph.2018-03-01.csv.gz",
        "data/raw/wikilinks.csv.gz",
        "data/raw/wikilinks.txt.gz",
        "data/wikilinks.csv.gz",
        "wikilinks.csv.gz",
        "enwiki.wikilink_graph.2018-03-01.csv.gz"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ 找到数据集: {path}")
            return path
    
    print("✗ 未找到数据集文件")
    print("请将数据集放在以下位置之一:")
    for path in possible_paths:
        print(f"  - {path}")
    
    # 列出当前目录和data/raw目录
    print("\n当前目录中的文件:")
    for f in os.listdir('.'):
        if f.endswith(('.gz', '.csv', '.txt')):
            print(f"  - {f}")
    
    if os.path.exists('data/raw'):
        print("\ndata/raw目录中的文件:")
        for f in os.listdir('data/raw'):
            print(f"  - {f}")
    
    return None

def create_small_dataset_for_training(data_path, output_path="data/processed/wikilinks_small.parquet", 
                                      sample_size=10000):
    """创建小型数据集用于训练"""
    print(f"创建小型数据集 ({sample_size}条边)...")
    
    try:
        # 读取部分数据
        edges = []
        with gzip.open(data_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, total=sample_size)):
                if i >= sample_size:
                    break
                
                # 尝试不同的分隔符
                if ',' in line:
                    parts = line.strip().split(',')
                elif '\t' in line:
                    parts = line.strip().split('\t')
                else:
                    parts = line.strip().split()
                
                if len(parts) >= 2:
                    src = parts[0].strip().strip('"')
                    dst = parts[1].strip().strip('"')
                    
                    if src and dst and src != dst:
                        edges.append((src, dst))
        
        # 创建DataFrame
        df = pd.DataFrame(edges, columns=['source', 'target'])
        
        # 保存为Parquet格式
        os.makedirs('data/processed', exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        print(f"✓ 数据集已保存: {output_path}")
        print(f"  边数: {len(df):,}")
        print(f"  唯一节点数: {pd.concat([df['source'], df['target']]).nunique():,}")
        
        return output_path
        
    except Exception as e:
        print(f"✗ 创建数据集失败: {e}")
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("数据集检查工具")
    print("=" * 60)
    
    # 1. 查找数据文件
    data_path = find_data_file()
    if data_path is None:
        return
    
    # 2. 创建小型数据集
    print("\n" + "-" * 60)
    print("创建小型数据集用于快速训练...")
    create_small_dataset_for_training(data_path, sample_size=20000)

if __name__ == "__main__":
    main()