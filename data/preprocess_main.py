import os
import yaml
import pandas as pd
import numpy as np
import holidays  # <--- 必须安装: pip install holidays
from typing import List, Dict, Union, Tuple
# 仅添加这4行代码，放在所有 import 之前 ✅ 合规写法、无侵入、毕设可用
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import setup_logger

class SocialModalityInjector:
    """
    [新增组件] 社会语义模态注入器 (Social Semantic Modality Injector)
    
    功能:
    将简单的时间戳转换为丰富的社会语义描述。
    不只是生成 0/1，而是生成 BERT 可以理解的自然语言描述。
    """
    def __init__(self, country='ES', years: List[int] = None):
        # 加载西班牙节日
        self.holidays_map = holidays.Spain(years=years)
        
    def analyze_date(self, date_obj) -> Tuple[str, str, int]:
        """
        分析单个日期，返回三元组:
        1. social_text (str): 用于 BERT 的自然语言描述 (e.g., "Christmas Day", "a regular workday")
        2. day_type (str): 用于 Embedding 的类别 (Holiday, Weekend, Workday)
        3. is_holiday_int (int): 用于数值模型的 0/1 标记
        """
        # 1. 优先判断是否为法定节假日
        holiday_name = self.holidays_map.get(date_obj)
        
        if holiday_name:
            # 是节日
            social_text = holiday_name  # e.g., "Epiphany"
            day_type = 'Holiday'
            is_holiday_int = 1
        elif date_obj.weekday() >= 5: # 5=Sat, 6=Sun
            # 是周末
            social_text = "Weekend"
            day_type = 'Weekend'
            is_holiday_int = 1 # 在电力预测中，周末的工况通常接近节日（低负荷）
        else:
            # 是工作日
            social_text = "Workday"
            day_type = 'Workday'
            is_holiday_int = 0
            
        return social_text, day_type, is_holiday_int

class DataPreprocessor:
    """
    科研级数据预处理流水线 (Research-Grade Data Preprocessing Pipeline)
    
    升级:
    集成 SocialModalityInjector，实现“气象-社会”文本级融合。
    """
    
    def __init__(self, config_path: str):
        # 1. 加载配置
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # 2. 初始化日志
        self.logger = setup_logger(self.config['paths']['log_dir'], "Preprocess")
        self.logger.info(">>> 初始化数据预处理器 (DataPreprocessor) [with Social Semantics]...")
        
    def load_raw_data(self) -> None:
        """加载原始 CSV 数据并进行初步的时间解析"""
        try:
            energy_path = self.config['paths']['raw_energy']
            weather_path = self.config['paths']['raw_weather']
            
            self.logger.info(f"正在读取电力数据: {energy_path}")
            self.df_energy = pd.read_csv(energy_path)
            
            self.logger.info(f"正在读取天气数据: {weather_path}")
            self.df_weather = pd.read_csv(weather_path)
            
            # 统一转换为 UTC 时间
            self.df_energy['time'] = pd.to_datetime(self.df_energy['time'], utc=True, errors='coerce')
            self.df_weather['dt_iso'] = pd.to_datetime(self.df_weather['dt_iso'], utc=True, errors='coerce')
            
            self.df_energy.dropna(subset=['time'], inplace=True)
            self.df_weather.dropna(subset=['dt_iso'], inplace=True)
            
            self.logger.info(f"数据加载成功. Energy Shape: {self.df_energy.shape}, Weather Shape: {self.df_weather.shape}")
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise e

    def process_weather(self) -> pd.DataFrame:
        """处理天气数据：透视变换"""
        self.logger.info("正在处理天气数据 (Pivot & Impute)...")
        
        target_cities = self.config['preprocessing']['cities']
        
        # 1. 筛选与去重
        df_w = self.df_weather[self.df_weather['city_name'].isin(target_cities)].copy()
        df_w = df_w.drop_duplicates(subset=['dt_iso', 'city_name'], keep='last')
        
        # 2. 定义特征
        numeric_feats = ['temp', 'humidity', 'wind_speed', 'pressure', 'rain_1h', 'snow_3h', 'clouds_all']
        text_feats = ['weather_main', 'weather_description']
        
        # 3. 透视变换 (Pivot)
        df_num = df_w.pivot(index='dt_iso', columns='city_name', values=numeric_feats)
        df_num = df_num.interpolate(method='linear', limit_direction='both')
        
        df_text = df_w.pivot(index='dt_iso', columns='city_name', values=text_feats)
        df_text = df_text.fillna(method='ffill').fillna(method='bfill')
        
        # 4. 合并与重命名
        processed_data = pd.concat([df_num, df_text], axis=1)
        
        new_columns = []
        for feature_name, city_name in processed_data.columns:
            new_columns.append(f"{city_name}_{feature_name}")
            
        processed_data.columns = new_columns
        processed_data.index.name = 'time'
        
        return processed_data

    def process_energy(self) -> pd.DataFrame:
        """处理电力数据"""
        self.logger.info("正在处理电力数据...")
        target_col = self.config['preprocessing']['target_col']
        
        # 1. 筛选列
        cols = ['time', target_col, 'price actual']
        df_e = self.df_energy[cols].copy()
        
        # 2. 索引与去重
        df_e = df_e.set_index('time')
        df_e = df_e[~df_e.index.duplicated(keep='first')]
        
        # 3. 插值
        df_e = df_e.interpolate(method='linear', limit_direction='both')
        
        return df_e

    def augment_with_social_semantics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [核心升级] 注入社会语义模态
        1. 生成 is_holiday_int (数值)
        2. 生成 social_text (文本)
        3. 执行文本融合: Weather Text + Social Text
        """
        self.logger.info(">>> 正在注入社会语义模态 (Social Semantic Modality)...")
        
        df = df.copy()
        # 提取年份用于初始化 holidays 库
        years = df.index.year.unique().tolist()
        injector = SocialModalityInjector(country='ES', years=years)
        
        # 1. 分析每一天
        # 使用 map/apply 可能会慢，但在预处理阶段可以接受
        results = df.index.to_series().apply(lambda x: injector.analyze_date(x))
        
        # 2. 分配新列
        # results 包含 (text, type, int)
        df['social_text'] = [r[0] for r in results]
        df['day_type'] = [r[1] for r in results]
        df['is_holiday_int'] = [r[2] for r in results]
        
        self.logger.info(f"社会特征已生成. 示例:\n{df[['social_text', 'day_type']].head(3)}")
        
        # 3. [Text Fusion Strategy] 文本级融合
        # 找到所有包含 'weather' 的文本列 (例如 Madrid_weather_description)
        # 将社会文本拼接到它们后面，形成复合语义
        
        text_cols = [c for c in df.columns if 'weather_' in c]
        self.logger.info(f"正在执行文本融合 (Weather + Social) 到 {len(text_cols)} 个特征列...")
        
        for col in text_cols:
            # 融合格式: "sky is clear. Today is Christmas Day."
            # 这种格式 BERT 非常容易理解
            df[col] = df[col].astype(str) + ". Today is " + df['social_text'].astype(str) + "."
            
        return df

    def merge_and_split(self, df_weather_wide: pd.DataFrame, df_energy_clean: pd.DataFrame):
        """归并、增强并切分"""
        self.logger.info("正在合并数据...")
        
        # 1. 合并 (Inner Join)
        df_final = pd.merge(df_energy_clean, df_weather_wide, left_index=True, right_index=True, how='inner')
        df_final = df_final.sort_index()
        
        # 2. [新增步骤] 注入社会语义
        # 必须在合并后做，因为我们需要完整的时间索引，且需要修改 weather 列
        df_final = self.augment_with_social_semantics(df_final)
        
        # 3. 切分
        n_samples = len(df_final)
        ratios = self.config['preprocessing']['split_ratio']
        
        train_idx = int(n_samples * ratios[0])
        val_idx = int(n_samples * (ratios[0] + ratios[1]))
        
        train_set = df_final.iloc[:train_idx]
        val_set = df_final.iloc[train_idx:val_idx]
        test_set = df_final.iloc[val_idx:]
        
        self.logger.info(f"数据集切分完成: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
        
        # 4. 保存
        out_dir = self.config['paths']['output_dir']
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        train_set.to_csv(os.path.join(out_dir, "train.csv"))
        val_set.to_csv(os.path.join(out_dir, "val.csv"))
        test_set.to_csv(os.path.join(out_dir, "test.csv"))
        
        self.logger.info(f"预处理全部完成! 增强后的数据已保存至: {out_dir}")

    def run(self):
        """执行流水线"""
        self.load_raw_data()
        df_weather_wide = self.process_weather()
        df_energy_clean = self.process_energy()
        self.merge_and_split(df_weather_wide, df_energy_clean)

if __name__ == "__main__":
    # 假设 config 路径
    CONFIG_PATH = "../configs/data_config.yaml"
    # 如果本地测试，可以兼容一下路径
    
        
    processor = DataPreprocessor(CONFIG_PATH)
    processor.run()