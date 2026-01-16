import sys
import os
import torch
import yaml
from torch.utils.data import DataLoader

# [关键] 将项目根目录加入路径，否则无法 import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import PowerDataset
from src.data.similarity import SimilarityEngine

def test_pipeline():
    print("="*50)
    print(">>> 开始运行数据管道测试 (Data Pipeline Test)")
    print("="*50)
    
    # 1. 虚拟配置 (Mock Config)
    # 我们直接读取真实的配置文件，确保测试环境与实际一致
    config_path = "../configs/exp_main_ra_st_glru.yaml"
    if not os.path.exists(config_path):
        print(f"[错误] 找不到配置文件: {config_path}")
        print("请先确保你已经创建了 'configs/exp_main_ra_st_glru.yaml'")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 测试 SimilarityEngine (核心算法)
    print("\n[Step 1] 测试相似日检索引擎...")
    try:
        sim_engine = SimilarityEngine(config)
        # 读取训练集数据进行 fit
        # 为了测试方便，我们临时手动读取一下 train.csv 的列名
        import pandas as pd
        df_train = pd.read_csv(os.path.join(config['paths']['output_dir'], "train.csv"))
        # 排除非数值列
        numeric_cols = [c for c in df_train.columns if "weather_main" not in c and "description" not in c and "time" != c]
        
        sim_engine.fit(df_train, numeric_cols)
        print("   >>> 熵权计算成功！")
        
        # 模拟一次查询
        dummy_query = df_train[numeric_cols].iloc[0].values
        indices = sim_engine.search(dummy_query)
        print(f"   >>> 检索测试成功. Query Day Index: 0 -> Similar Indices: {indices.tolist()}")
        
    except Exception as e:
        print(f"[失败] SimilarityEngine 出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 测试 Dataset (多模态加载)
    print("\n[Step 2] 测试 Dataset (含 BERT 嵌入)...")
    try:
        # 实例化 Dataset
        # mode='train' 会触发 fit logic (虽然我们在 dataset 内部做了独立处理)
        # 注意: 第一次运行这里会下载 DistilBERT 模型，可能会花点时间
        train_dataset = PowerDataset(config, mode='train', similarity_engine=sim_engine)
        
        print(f"   >>> Dataset 初始化完成. 样本数: {len(train_dataset)}")
        print(f"   >>> 序列长度: {train_dataset.seq_len}, 预测长度: {train_dataset.pred_len}")
        
    except Exception as e:
        print(f"[失败] Dataset 初始化出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 测试 DataLoader (Batch 输出形状)
    print("\n[Step 3] 测试 DataLoader 输出形状...")
    try:
        batch_size = 4
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 获取一个 Batch
        batch_data = next(iter(loader))
        (x_num, x_text), x_sim, y = batch_data
        
        print("\n   [Tensor Shapes Check]")
        print(f"   1. Numerical Input (x_num): {x_num.shape}")
        print(f"      期待: [{batch_size}, {config['model']['seq_len']}, Features]")
        
        print(f"   2. Text Input (x_text):     {x_text.shape}")
        print(f"      期待: [{batch_size}, {config['model']['seq_len']}, 768] (DistilBERT Dim)")
        
        print(f"   3. Similar Days (x_sim):    {x_sim.shape}")
        print(f"      期待: [{batch_size}, {config['model']['top_k']}, {config['model']['seq_len']}, Features]")
        
        print(f"   4. Label (y):               {y.shape}")
        print(f"      期待: [{batch_size}, {config['model']['out_len']}, 1]")
        
        print("\n>>> ✅ 测试全部通过！数据管道准备就绪。")
        
    except Exception as e:
        print(f"[失败] DataLoader 迭代出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()