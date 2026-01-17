import sys
import os
import pandas as pd
import numpy as np
import yaml

# 添加路径以使用项目中的 metrics 工具
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.metrics import calculate_metrics

def simple_baseline_eval():
    print(">>> 正在计算官方预测指标 (处理时区偏移)...")
    
    # 1. 读取配置
    config_path = './configs/exp_main_ra_st_glru.yaml'
    if not os.path.exists(config_path): config_path = '../configs/exp_main_ra_st_glru.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    
    # 2. 读取数据
    # [Test Set]: 我们的测试集，已经统一过时间 (通常是 UTC)
    test_path = os.path.join(config['paths']['output_dir'], "test.csv")
    df_test = pd.read_csv(test_path)
    # 关键点：强制转为 UTC，消除歧义
    df_test['time'] = pd.to_datetime(df_test['time'], utc=True)
    
    # [Raw Data]: 原始数据，格式如 "2015-01-01 00:00:00+01:00"
    raw_path = config['paths']['raw_energy']
    df_raw = pd.read_csv(raw_path)
    
    # 关键点：这里使用 utc=True，pandas 会自动识别 "+01:00" 并将其转为 UTC 标准时间
    # 这样 "00:00:00+01:00" 就会和测试集中的 "23:00:00+00:00" 对上
    df_raw['time'] = pd.to_datetime(df_raw['time'], utc=True)
    
    # 3. 数据对齐与提取
    # 我们只关心测试集里有的那些时间点
    # 使用 merge (inner join) 确保时间戳完全一致
    df_merged = pd.merge(df_test[['time', 'total load actual']], 
                         df_raw[['time', 'total load forecast']], 
                         on='time', 
                         how='inner')
    
    # 去除空值 (以防官方预测在某些点缺失)
    df_final = df_merged.dropna()
    
    print(f"对齐后样本数: {len(df_final)} (原始测试集: {len(df_test)})")
    
    # 4. 计算指标
    y_true = df_final['total load actual'].values
    y_pred_official = df_final['total load forecast'].values
    
    metrics = calculate_metrics(y_pred_official, y_true)
    
    print("\n" + "="*30)
    print("官方预测 (TSO Baseline) 结果")
    print("="*30)
    print(f"MAE:  {metrics['mae']:.2f} MW")
    print(f"RMSE: {metrics['rmse']:.2f} MW")
    print(f"MAPE: {metrics['mape']:.2f} %")
    print("="*30)

if __name__ == "__main__":
    simple_baseline_eval()