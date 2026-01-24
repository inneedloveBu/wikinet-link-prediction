# scripts/download_data.py
import os
import requests
import gzip
import pandas as pd
from tqdm import tqdm
import shutil
from pathlib import Path
import logging
import tarfile
import zipfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDownloader:
    """多源数据下载器"""
    
    DATA_SOURCES = {
        'clickstream': {
            'urls': [
                # 尝试不同的月份
                #'https://dumps.wikimedia.org/other/clickstream/2024-01/clickstream-enwiki-2024-01.tsv.gz',
                #'https://dumps.wikimedia.org/other/clickstream/2023-12/clickstream-enwiki-2023-12.tsv.gz',
                #'https://dumps.wikimedia.org/other/clickstream/2023-11/clickstream-enwiki-2023-11.tsv.gz',
            ],
            'description': 'Wikipedia Clickstream数据（用户点击流）'
        },
        'wikilinks_snap': {
            'urls': [
                'https://snap.stanford.edu/data/wiki-Links.txt.gz',
                'https://snap.stanford.edu/data/wiki-topcats.txt.gz',
            ],
            'description': 'Stanford SNAP Wikipedia链接数据'
        },
        'test_graph': {
            'urls': [
                'https://snap.stanford.edu/data/email-Eu-core.txt.gz',
                'https://snap.stanford.edu/data/ca-AstroPh.txt.gz',
            ],
            'description': '测试图数据集'
        }
    }
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_with_fallback(self, source_key='clickstream'):
        """使用回退机制下载数据"""
        if source_key not in self.DATA_SOURCES:
            raise ValueError(f"未知数据源: {source_key}")
        
        urls = self.DATA_SOURCES[source_key]['urls']
        downloaded_files = []
        
        for url in urls:
            try:
                filename = self._download_file(url)
                if filename:
                    downloaded_files.append(filename)
                    logger.info(f"成功下载: {filename}")
                    break  # 成功下载一个就停止
            except Exception as e:
                logger.warning(f"下载失败 {url}: {e}")
                continue
        
        if not downloaded_files:
            logger.error(f"所有数据源都失败了，尝试生成模拟数据...")
            return self._generate_mock_data()
        
        return downloaded_files
    
    def _download_file(self, url):
        """下载单个文件"""
        filename = self.data_dir / url.split('/')[-1]
        
        # 如果文件已存在且大小合理，跳过下载
        if filename.exists() and filename.stat().st_size > 1000:
            logger.info(f"文件已存在: {filename}")
            return filename
        
        logger.info(f"下载: {url}")
        
        try:
            # 设置请求头
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # 流式下载
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # 下载并显示进度条
            with open(filename, 'wb') as f, tqdm(
                desc=filename.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # 如果是gzip文件，解压
            if filename.suffix == '.gz':
                extracted = self._extract_gz(filename)
                return extracted
            elif filename.suffix == '.zip':
                extracted = self._extract_zip(filename)
                return extracted
                
            return filename
            
        except requests.exceptions.RequestException as e:
            logger.error(f"下载失败: {e}")
            return None
    
    def _extract_gz(self, gz_path):
        """解压.gz文件"""
        extracted_path = gz_path.with_suffix('')  # 移除.gz后缀
        
        if extracted_path.exists():
            return extracted_path
        
        logger.info(f"解压: {gz_path}")
        
        with gzip.open(gz_path, 'rb') as f_in:
            with open(extracted_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return extracted_path
    
    def _extract_zip(self, zip_path):
        """解压.zip文件"""
        extracted_dir = zip_path.with_suffix('')
        extracted_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
        
        return extracted_dir
    
    def _generate_mock_data(self):
        """生成模拟数据作为后备"""
        logger.info("生成模拟Wikipedia链接数据...")
        
        # 创建模拟数据
        np = __import__('numpy')
        pd = __import__('pandas')
        
        # 模拟1000个页面
        num_pages = 1000
        edges = []
        
        # 创建一些链接关系
        for i in range(num_pages):
            # 每个页面链接到2-5个其他页面
            num_links = np.random.randint(2, 6)
            targets = np.random.choice(num_pages, num_links, replace=False)
            targets = targets[targets != i]  # 移除自链接
            
            for target in targets:
                edges.append({
                    'source': f'Page_{i}',
                    'target': f'Page_{target}',
                    'weight': np.random.randint(1, 100)
                })
        
        # 保存为Parquet
        df = pd.DataFrame(edges)
        output_path = self.data_dir / 'mock_wikipedia_links.parquet'
        df.to_parquet(output_path, index=False)
        
        logger.info(f"模拟数据已保存: {output_path}")
        logger.info(f"数据统计: {len(df)} 条边, {df['source'].nunique()} 个源节点")
        
        return [output_path]
    
    def process_clickstream_data(self, data_path):
        """处理Clickstream数据"""
        logger.info("处理Clickstream数据...")
        
        if str(data_path).endswith('.gz'):
            # 解压
            data_path = self._extract_gz(data_path)
        
        # 读取TSV文件
        df = pd.read_csv(data_path, sep='\t', 
                        names=['prev', 'curr', 'type', 'n'])
        
        logger.info(f"原始数据: {len(df)} 行")
        
        # 过滤：只保留内部链接，且点击次数大于阈值
        df = df[df['type'] == 'link']
        df = df[df['n'] >= 10]  # 只保留点击10次以上的链接
        
        logger.info(f"过滤后: {len(df)} 行")
        
        # 重命名列
        df = df.rename(columns={'prev': 'source', 'curr': 'target', 'n': 'weight'})
        df = df[['source', 'target', 'weight']]
        
        # 保存为Parquet
        output_path = self.data_dir / 'wikipedia_clickstream_processed.parquet'
        df.to_parquet(output_path, index=False)
        
        return output_path
    
    def process_wikilinks_data(self, data_path):
        """处理WikiLinks数据"""
        logger.info("处理WikiLinks数据...")
        
        if str(data_path).endswith('.gz'):
            data_path = self._extract_gz(data_path)
        
        # 读取数据（格式：源节点 目标节点 类别）
        df = pd.read_csv(data_path, sep='\t', 
                        header=None, 
                        names=['source', 'target', 'category'])
        
        logger.info(f"数据统计: {len(df)} 条边")
        
        # 保存为Parquet
        output_path = self.data_dir / 'wikipedia_links_processed.parquet'
        df.to_parquet(output_path, index=False)
        
        return output_path

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='下载Wikipedia网络数据')
    parser.add_argument('--source', type=str, default='clickstream',
                       choices=['clickstream', 'wikilinks_snap', 'test_graph'],
                       help='数据源')
    parser.add_argument('--process', action='store_true',
                       help='是否立即处理数据')
    
    args = parser.parse_args()
    
    downloader = DataDownloader()
    
    # 下载数据
    downloaded = downloader.download_with_fallback(args.source)
    
    if not downloaded:
        logger.error("下载失败，退出")
        return
    
    # 处理数据
    if args.process:
        for file_path in downloaded:
            if 'clickstream' in str(file_path).lower():
                processed = downloader.process_clickstream_data(file_path)
                logger.info(f"已处理Clickstream数据: {processed}")
            elif 'wiki-links' in str(file_path).lower():
                processed = downloader.process_wikilinks_data(file_path)
                logger.info(f"已处理WikiLinks数据: {processed}")

if __name__ == "__main__":
    main()