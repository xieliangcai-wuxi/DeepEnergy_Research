import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import holidays

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.similarity import SimilarityEngine

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_day_status(date_obj, holiday_obj):
    is_weekend = date_obj.weekday() >= 5
    is_holiday = date_obj in holiday_obj
    if is_holiday: return "🔴节日"
    if is_weekend: return "🟠周末"
    return "🔵工作日"

def preprocess_runtime_simulation(df, target_col):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], utc=True)
    
    if target_col in df.columns:
        df['lag_24'] = df[target_col].shift(24).bfill()
        df['lag_168'] = df[target_col].shift(168).bfill()
    
    # 动态计算季节性特征
    doy = df['time'].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * doy / 365.0)
    df['cos_doy'] = np.cos(2 * np.pi * doy / 365.0)
    
    if 'is_holiday_int' not in df.columns:
        years = df['time'].dt.year.unique()
        es_holidays = holidays.Spain(years=years)
        def check_holiday(d):
            return 1.0 if (d in es_holidays or d.weekday() >= 5) else 0.0
        df['is_holiday_int'] = df['time'].dt.date.map({d: check_holiday(d) for d in df['time'].dt.date.unique()})
        
    years = df['time'].dt.year.unique()
    es_holidays = holidays.Spain(years=years)
    return df, es_holidays

def run_verification_report():
    out_dir = './test/similarity_check_ratio' # 改个文件夹名
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    print("\n" + "="*80)
    print("   🕵️‍♀️ 相似日逻辑验证 (比率修正版)   ")
    print("   核心思想: 不抄绝对值，只抄变化率 (Ratio)")
    print("="*80)
    
    config_path = './configs/exp_main_ra_st_glru.yaml'
    if not os.path.exists(config_path): config_path = '../configs/exp_main_ra_st_glru.yaml'
    with open(config_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    target_col = config['preprocessing']['target_col']
    
    print(">>> [Step 1] 加载数据...")
    train_path = os.path.join(config['paths']['output_dir'], "train.csv")
    test_path = os.path.join(config['paths']['output_dir'], "test.csv")
    
    df_train, es_holidays = preprocess_runtime_simulation(pd.read_csv(train_path), target_col)
    df_test, _ = preprocess_runtime_simulation(pd.read_csv(test_path), target_col)
    
    # 定义搜索特征 (含季节性)
    all_cols = df_train.columns
    search_features = ['lag_24', 'lag_168', 'price actual', 'is_holiday_int', 'sin_doy', 'cos_doy']
    search_features.extend([c for c in all_cols if '_temp' in c])
    if target_col in search_features: search_features.remove(target_col)
    
    print(f"    ✅ 搜索特征: {search_features}")
    
    print(f"\n>>> [Step 2] 训练引擎...")
    sim_engine = SimilarityEngine(config)
    sim_engine.fit(df_train, search_features)
    
    # 案例分析 (还是那几个)
    target_dates = ['2018-08-12 12:00', '2018-08-15 12:00', '2018-08-07 12:00']
    
    print(f"\n>>> [Step 3] 开始案例分析 (应用比率修正)...")
    
    for t_str in target_dates:
        target_ts = pd.to_datetime(t_str).tz_localize('UTC')
        idx = (df_test['time'] - target_ts).abs().idxmin()
        query_row = df_test.iloc[idx]
        query_time = query_row['time']
        
        # 搜索
        query_vals = query_row[search_features].values.reshape(1, -1)
        query_norm = sim_engine.scaler.transform(query_vals)
        indices = sim_engine.search(torch.tensor(query_norm, dtype=torch.float32), training_mode=False).numpy()[0][:3]
        
        # 基础信息
        q_status = get_day_status(query_time.date(), es_holidays)
        q_load = query_row[target_col]
        q_lag = query_row['lag_24'] # 这是锚点！
        
        print("\n" + "-"*80)
        print(f"📅 目标日: {query_time.date()} ({q_status})")
        print("-" * 80)
        print(f"【锚点】 昨天负荷 (Lag24): {q_lag:.0f} MW")
        print(f"【真值】 {q_load:.0f} MW")
        
        plt.figure(figsize=(12, 6))
        
        # 画真值
        q_start = query_time.normalize()
        q_data = df_test[(df_test['time'] >= q_start) & (df_test['time'] < q_start + pd.Timedelta(days=1))]
        plt.plot(q_data['time'].dt.hour, q_data[target_col], 'k-', linewidth=3, label=f'真值', zorder=10)
        
        colors = ['#E63946', '#F4A261', '#2A9D8F']
        
        for rank, sim_idx in enumerate(indices):
            sim_row = df_train.iloc[sim_idx]
            sim_date = sim_row['time'].date()
            sim_status = get_day_status(sim_date, es_holidays)
            
            # --- 核心修改：比率法 (Ratio Method) ---
            sim_lag = sim_row['lag_24']
            sim_actual = sim_row[target_col]
            
            # 计算历史那一天的变化率 (Ratio)
            # 防止分母为0 (虽然不太可能)
            ratio = sim_actual / (sim_lag + 1e-5)
            
            # 用今天的 Lag * 历史的 Ratio
            pred_load_ratio = q_lag * ratio
            
            # 计算直接拷贝的偏差 (Old Way)
            err_direct = abs(sim_actual - q_load) / q_load * 100
            # 计算比率法的偏差 (New Way)
            err_ratio = abs(pred_load_ratio - q_load) / q_load * 100
            
            print(f"   🏆 Rank {rank+1}: {sim_date} ({sim_status})")
            print(f"      -> 绝对值偏差: {err_direct:.1f}% (直接抄: {sim_actual:.0f})")
            print(f"      -> 比率法偏差: {err_ratio:.1f}% (修正后: {pred_load_ratio:.0f}) {'✅ 改善' if err_ratio < err_direct else '⚠️ 恶化'}")
            
            # 绘图：画出修正后的曲线
            # 我们需要把整条曲线都乘上 (q_lag / sim_lag) 这个缩放系数
            s_start = sim_row['time'].normalize()
            s_data = df_train[(df_train['time'] >= s_start) & (df_train['time'] < s_start + pd.Timedelta(days=1))]
            
            if len(s_data) > 0:
                # 计算全天的缩放系数 (基于 lag_24)
                # 注意: 这里简化了，全天都用同一个 scaling factor。
                # 实际上模型会在每个 timestep 动态调整。
                scale_factor = q_lag / (sim_lag + 1e-5)
                scaled_curve = s_data[target_col] * scale_factor
                
                plt.plot(s_data['time'].dt.hour, scaled_curve, 
                         linestyle='--', color=colors[rank], alpha=0.8,
                         label=f'Top-{rank+1} (修正版): {sim_date}')

        plt.title(f"相似日 (比率修正版): {query_time.date()}", fontsize=14)
        plt.xlabel("Hour")
        plt.ylabel("Load (MW)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_name = os.path.join(out_dir, f"report_{query_time.date()}.png")
        plt.savefig(save_name)
        plt.close()
        print(f"   📊 图表: {save_name}")

if __name__ == "__main__":
    run_verification_report()