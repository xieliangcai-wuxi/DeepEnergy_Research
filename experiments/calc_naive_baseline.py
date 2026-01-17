import sys
import os
import pandas as pd
import numpy as np
import yaml

# 添加项目路径以使用 metrics 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.metrics import calculate_metrics

def calc_naive_baseline():
    print(">>> 正在计算 Naive Baseline (24h Persistence)...")
    
    # 1. 读取配置
    config_path = './configs/exp_main_ra_st_glru.yaml'
    # 兼容路径查找
    if not os.path.exists(config_path): 
        config_path = '../configs/exp_main_ra_st_glru.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f: 
        config = yaml.safe_load(f)
    
    # 2. 读取测试集 (作为 Ground Truth 和时间基准)
    test_path = os.path.join(config['paths']['output_dir'], "test.csv")
    if not os.path.exists(test_path):
        print(f"Error: 测试集文件不存在 {test_path}")
        return

    df_test = pd.read_csv(test_path)
    # 强制转为 UTC，确保它是标准时间
    df_test['time'] = pd.to_datetime(df_test['time'], utc=True)
    
    # 3. 读取原始数据 (作为历史数据源)
    raw_path = config['paths']['raw_energy']
    df_raw = pd.read_csv(raw_path)
    # 关键：处理原始数据中的 "+01:00" 等偏移，统一转为 UTC
    df_raw['time'] = pd.to_datetime(df_raw['time'], utc=True)
    
    # 4. 构造朴素预测 (Persistence)
    # 逻辑: 预测值(t) = 真实值(t - 24h)
    # 实现: 将原始时间的 Total Load 向后"推" 24小时
    df_naive = df_raw[['time', 'total load actual']].copy()
    df_naive['time_pred'] = df_naive['time'] + pd.Timedelta(hours=24)
    df_naive = df_naive.rename(columns={'total load actual': 'naive_pred'})
    
    # 5. 严格对齐 (Inner Join)
    # 左边: 测试集 (Ground Truth)
    # 右边: 朴素预测 (Prediction)
    # 匹配条件: 测试集时间 == 预测时间 (即原来的时间+24h)
    df_eval = pd.merge(df_test[['time', 'total load actual']], 
                       df_naive[['time_pred', 'naive_pred']], 
                       left_on='time', 
                       right_on='time_pred', 
                       how='inner')
    
    # 6. 计算并输出指标
    y_true = df_eval['total load actual'].values
    y_pred = df_eval['naive_pred'].values
    
    metrics = calculate_metrics(y_pred, y_true)
    
    print("-" * 30)
    print("Naive Baseline (Persistence) 结果")
    print("-" * 30)
    print(f"MAE:  {metrics['mae']:.2f} MW")
    print(f"RMSE: {metrics['rmse']:.2f} MW")
    print(f"MAPE: {metrics['mape']:.2f} %")
    print("-" * 30)
    print(f"对比样本数: {len(df_eval)}")

if __name__ == "__main__":
    calc_naive_baseline()