# unzip_data.py
import gzip
import shutil
from pathlib import Path

def decompress_gz_files():
    """解压所有.gz文件"""
    raw_dir = Path("data/raw")
    
    for gz_file in raw_dir.glob("*.gz"):
        output_file = gz_file.with_suffix('')  # 移除.gz后缀
        
        print(f"解压 {gz_file.name} -> {output_file.name}")
        
        try:
            with gzip.open(gz_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"  成功！大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"  失败: {e}")

if __name__ == "__main__":
    decompress_gz_files()