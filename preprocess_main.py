import os
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Union
from src.utils.logger import setup_logger

class DataPreprocessor:
    """
    科研级数据预处理流水线 (Research-Grade Data Preprocessing Pipeline)
    
    功能:
    1. 清洗 (Cleaning): 去重、去噪、处理缺失值。
    2. 对齐 (Alignment): 将不同频率或格式的多源数据对齐到同一时间索引。
    3. 变换 (Transformation): 针对多城市数据进行 Pivot 透视变换，保留空间维度信息。
    4. 切分 (Splitting): 严格按照时序进行 Train/Val/Test 切分。
    
    Attributes:
        config (dict): 配置字典，包含路径和超参数。
        logger (logging.Logger): 日志记录器。
    """
    
    def __init__(self, config_path: str):
        # 1. 加载配置
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # 2. 初始化日志
        # 日志将保存在 log_dir 下
        self.logger = setup_logger(self.config['paths']['log_dir'], "Preprocess")
        self.logger.info(">>> 初始化数据预处理器 (DataPreprocessor)...")
        
    def load_raw_data(self) -> None:
        """加载原始 CSV 数据并进行初步的时间解析"""
        try:
            energy_path = self.config['paths']['raw_energy']
            weather_path = self.config['paths']['raw_weather']
            
            self.logger.info(f"正在读取电力数据: {energy_path}")
            self.df_energy = pd.read_csv(energy_path)
            
            self.logger.info(f"正在读取天气数据: {weather_path}")
            self.df_weather = pd.read_csv(weather_path)
            
            # 统一转换为 UTC 时间，防止时区混乱
            # 'coerce' 模式会将无法解析的时间设为 NaT，方便后续剔除
            self.df_energy['time'] = pd.to_datetime(self.df_energy['time'], utc=True, errors='coerce')
            self.df_weather['dt_iso'] = pd.to_datetime(self.df_weather['dt_iso'], utc=True, errors='coerce')
            
            # 剔除时间解析失败的行
            self.df_energy.dropna(subset=['time'], inplace=True)
            self.df_weather.dropna(subset=['dt_iso'], inplace=True)
            
            self.logger.info(f"数据加载成功. Energy Shape: {self.df_energy.shape}, Weather Shape: {self.df_weather.shape}")
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise e

    def process_weather(self) -> pd.DataFrame:
        """
        [核心逻辑] 处理天气数据：透视变换与多模态处理
        
        Logic:
        1. 筛选目标城市。
        2. 将 Long Format (Time, City, Feat) 转换为 Wide Format (Time, City_Feat)。
        3. 分离数值特征和文本特征，分别采用不同的填充策略。
        """
        self.logger.info("正在处理天气数据 (Pivot & Impute)...")
        
        target_cities = self.config['preprocessing']['cities']
        
        # 1. 筛选 5 个主要城市
        # 使用 loc 进行显式拷贝，避免 SettingWithCopyWarning
        df_w = self.df_weather[self.df_weather['city_name'].isin(target_cities)].copy()
        
        # 2. 去除完全重复的行 (保留最后一次更新的数据)
        df_w = df_w.drop_duplicates(subset=['dt_iso', 'city_name'], keep='last')
        
        # 3. 定义特征类型
        # 数值型特征 (用于 GNN/GLRU 输入)
        numeric_feats = ['temp', 'humidity', 'wind_speed', 'pressure', 'rain_1h', 'snow_3h', 'clouds_all']
        # 文本型特征 (用于 DistilBERT 语义嵌入)
        text_feats = ['weather_main', 'weather_description']
        
        # 4. 透视变换 (Pivot)
        # 我们需要分别 pivot，因为 pandas 的插值函数对混合类型支持不好
        
        # --- 处理数值特征 ---
        df_num = df_w.pivot(index='dt_iso', columns='city_name', values=numeric_feats)
        # 插值策略: 线性插值 (符合物理连续性)
        df_num = df_num.interpolate(method='linear', limit_direction='both')
        
        # --- 处理文本特征 ---
        df_text = df_w.pivot(index='dt_iso', columns='city_name', values=text_feats)
        # 插值策略: 前向填充 (假设天气状态具有持续性)
        df_text = df_text.fillna(method='ffill').fillna(method='bfill')
        
        # 5. 合并并重命名列
        # 此时 df_num 的列是 MultiIndex: (Feature, City) -> e.g., ('temp', 'Madrid')
        # 我们将其展平为规范名称: 'Madrid_temp'
        
        processed_data = pd.concat([df_num, df_text], axis=1)
        
        new_columns = []
        for feature_name, city_name in processed_data.columns:
            # 命名规范: City_Feature (例如 Madrid_temp)
            # 这种命名有利于后续 Dataset 类通过 split('_') 自动还原空间结构
            new_columns.append(f"{city_name}_{feature_name}")
            
        processed_data.columns = new_columns
        processed_data.index.name = 'time'
        
        self.logger.info(f"天气数据处理完成. 结果特征数: {len(new_columns)}")
        return processed_data

    def process_energy(self) -> pd.DataFrame:
        """
        处理电力数据：筛选、去重与插值
        """
        self.logger.info("正在处理电力数据...")
        
        target_col = self.config['preprocessing']['target_col'] # total load actual
        
        # 1. 必要的列 (Target + Price)
        # 注意: 我们可以保留 solar/wind generation 作为强协变量
        cols = ['time', target_col, 'price actual']
        
        df_e = self.df_energy[cols].copy()
        
        # 2. 设置索引并去重
        df_e = df_e.set_index('time')
        df_e = df_e[~df_e.index.duplicated(keep='first')]
        
        # 3. 缺失值处理
        # 检查缺失情况
        nan_count = df_e[target_col].isna().sum()
        if nan_count > 0:
            self.logger.warning(f"检测到负荷数据中有 {nan_count} 个缺失值，正在进行线性插值...")
            
        df_e = df_e.interpolate(method='linear', limit_direction='both')
        
        return df_e

    def merge_and_split(self, df_weather_wide: pd.DataFrame, df_energy_clean: pd.DataFrame):
        """
        [关键步骤] 归并数据集并严格切分
        """
        self.logger.info("正在合并数据并进行切分...")
        
        # 1. Inner Join (确保时间完全对齐)
        # df_weather_wide 和 df_energy_clean 索引都是 'time'
        df_final = pd.merge(df_energy_clean, df_weather_wide, left_index=True, right_index=True, how='inner')
        
        # 2. 再次按时间排序 (Double Check)
        df_final = df_final.sort_index()
        
        # 3. 按照 8:1:1 计算切分点
        n_samples = len(df_final)
        ratios = self.config['preprocessing']['split_ratio'] # [0.8, 0.1, 0.1]
        
        train_idx = int(n_samples * ratios[0])
        val_idx = int(n_samples * (ratios[0] + ratios[1]))
        
        train_set = df_final.iloc[:train_idx]
        val_set = df_final.iloc[train_idx:val_idx]
        test_set = df_final.iloc[val_idx:]
        
        self.logger.info(f"数据集切分详情:")
        self.logger.info(f"  [Total] {n_samples} 样本")
        self.logger.info(f"  [Train] {len(train_set)} 样本 ({ratios[0]*100}%) | 时间范围: {train_set.index[0]} -> {train_set.index[-1]}")
        self.logger.info(f"  [Val  ] {len(val_set)} 样本 ({ratios[1]*100}%) | 时间范围: {val_set.index[0]} -> {val_set.index[-1]}")
        self.logger.info(f"  [Test ] {len(test_set)} 样本 ({ratios[2]*100}%) | 时间范围: {test_set.index[0]} -> {test_set.index[-1]}")
        
        # 4. 保存文件
        out_dir = self.config['paths']['output_dir']
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        # 保存时保留 index (time)，方便后续 Dataset 调试查看
        train_set.to_csv(os.path.join(out_dir, "train.csv"))
        val_set.to_csv(os.path.join(out_dir, "val.csv"))
        test_set.to_csv(os.path.join(out_dir, "test.csv"))
        
        self.logger.info(f"预处理完成! 文件已保存至: {out_dir}")

    def run(self):
        """执行流水线"""
        self.load_raw_data()
        df_weather_wide = self.process_weather()
        df_energy_clean = self.process_energy()
        self.merge_and_split(df_weather_wide, df_energy_clean)

if __name__ == "__main__":
    # 测试代码入口
    # 假设 config 在上级目录
    CONFIG_PATH = "./configs/data_config.yaml"
    processor = DataPreprocessor(CONFIG_PATH)
    processor.run()