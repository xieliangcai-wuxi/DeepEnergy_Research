import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader

# 1. 路径 Hack (确保能导入 src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import PowerDataset
from src.data.similarity import SimilarityEngine
from src.models.ra_st_glru import RA_ST_GLRU
from src.utils.trainer import StandardTrainer

def run():
    # --- 配置与准备 ---
    EXP_NAME = "exp_main_ra_st_glru"
    CONFIG_PATH = "../configs/exp_main_ra_st_glru.yaml"
    
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"配置文件未找到: {CONFIG_PATH}")
        
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    print(f">>> 启动实验: {EXP_NAME}")
    
    # --- 步骤 1: 准备 Similarity Engine ---
    # 如果启用检索，先要在训练集上 fit 权重
    sim_engine = None
    if config['model']['use_retrieval']:
        print("[1/4] 初始化相似日检索引擎...")
        sim_engine = SimilarityEngine(config)
        # 临时读取 train.csv 获取列名用于 fit
        import pandas as pd
        train_df = pd.read_csv(os.path.join(config['paths']['output_dir'], "train.csv"))
        # 排除非数值列
        numeric_cols = [c for c in train_df.columns if "weather_main" not in c and "description" not in c and "time" != c]
        sim_engine.fit(train_df, numeric_cols)
    
    # --- 步骤 2: 准备 Datasets ---
    print("[2/4] 加载数据集 (Train/Val/Test)...")
    train_ds = PowerDataset(config, mode='train', similarity_engine=sim_engine)
    val_ds = PowerDataset(config, mode='val', similarity_engine=sim_engine)
    test_ds = PowerDataset(config, mode='test', similarity_engine=sim_engine)
    
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['train']['batch_size'], shuffle=False)
    
    # --- 步骤 3: 初始化模型 ---
    print("[3/4] 构建 RA-ST-GLRU 模型...")
    
    # [关键] 动态获取输入维度，防止配置错误
    # 从 Dataset 中获取一个样本，查看 numerical input 的 shape
    sample_input, _, _ = train_ds[0]
    sample_num, _ = sample_input
    real_in_features = sample_num.shape[-1]
    
    print(f"   >>> 检测到输入特征维度: {real_in_features}")
    
    model = RA_ST_GLRU(
        num_nodes=len(config['preprocessing']['cities']), # 5
        in_features=real_in_features,                     # 自动检测
        d_model=config['model']['d_model'],
        layers=config['model']['layers'],
        out_len=config['model']['out_len'],
        top_k=config['model']['top_k'],
        use_retrieval=config['model']['use_retrieval'],
        dropout=config['model'].get('dropout', 0.1) # 读取配置
    )
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   >>> 模型参数量: {total_params / 1e6:.2f} M")
    
    # --- 步骤 4: 训练 ---
    print("[4/4] 开始训练...")
    trainer = StandardTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        experiment_name=EXP_NAME
    )
    
    trainer.fit()
    print(">>> 实验结束.")

if __name__ == "__main__":
    run()