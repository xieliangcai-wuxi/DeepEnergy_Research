import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
from ..utils.logger import setup_logger
from .similarity import SimilarityEngine

class PowerDataset(Dataset):
    """
    [极简多模态版] PowerDataset
    
    Features:
    1. 保留 BERT (文本模态)
    2. 保留 Similarity Search (检索模态)
    3. 数值特征极简 (只留 Temp + Load + Price + Lags)
    4. 自动注入 Lag Features (物理外挂)
    """
    def __init__(self, config: dict, mode: str = 'train', similarity_engine: SimilarityEngine = None):
        super().__init__()
        self.config = config
        self.mode = mode
        self.logger = setup_logger(config['paths']['log_dir'], f"Dataset_{mode}")
        
        # ===========================
        # 1. 加载当前模式的数据
        # ===========================
        file_path = os.path.join(config['paths']['output_dir'], f"{mode}.csv")
        self.df = pd.read_csv(file_path)
        self.df['time'] = pd.to_datetime(self.df['time'])
        
        # ===========================
        # 2. [核心] 数据预处理 (注入Lag + 瘦身)
        # ===========================
        # 我们定义一个处理函数，稍后也要用它处理 train_data_ref
        self.df = self._preprocess_table(self.df, is_ref=False)
        
        # 重新定义列 (经过瘦身后的)
        all_cols = self.df.columns
        # 包含 'weather_' 的是文本列
        self.text_cols = [c for c in all_cols if 'weather_' in c]
        self.target_col = config['preprocessing']['target_col']
        # 剩下的就是数值列 (Load, Price, Temp, Lags)
        self.numeric_cols = [c for c in all_cols if c not in self.text_cols and c != 'time']
        
        self.logger.info(f"[{mode}] 数值特征维度: {len(self.numeric_cols)} (Temp, Load, Price, Lags)")
        self.logger.info(f"[{mode}] 文本特征维度: {len(self.text_cols)} (用于 BERT)")
        
        # ===========================
        # 3. 准备标准化器 & 训练集引用
        # ===========================
        self.scaler = StandardScaler()
        train_path = os.path.join(config['paths']['output_dir'], "train.csv")
        df_train_raw = pd.read_csv(train_path)
        df_train_raw['time'] = pd.to_datetime(df_train_raw['time'])
        
        # [关键] 训练集引用也必须经过完全相同的 注入Lag + 瘦身 处理！
        # 否则 scaler.fit 会报错，且检索的数据维度会对不上
        df_train_raw = self._preprocess_table(df_train_raw, is_ref=True)
        
        # Fit Scaler on Train
        self.scaler.fit(df_train_raw[self.numeric_cols].values)
        
        # 准备 train_data_ref (用于检索)
        train_numeric_numpy = self.scaler.transform(df_train_raw[self.numeric_cols].values)
        self.train_data_ref = torch.tensor(train_numeric_numpy, dtype=torch.float32)
        self.train_data_ref = self._inject_time_features(df_train_raw, self.train_data_ref)
        
        # 处理当前数据 (self.data_numeric)
        curr_numeric_numpy = self.scaler.transform(self.df[self.numeric_cols].values)
        self.data_numeric = torch.tensor(curr_numeric_numpy, dtype=torch.float32)
        self.data_numeric = self._inject_time_features(self.df, self.data_numeric)

        # ===========================
        # 4. 文本嵌入
        # ===========================
        self.data_text = self._load_or_compute_embeddings()
        
        # ===========================
        # 5. 相似日逻辑
        # ===========================
        self.similarity_engine = similarity_engine
        
        if self.mode == 'train' and self.similarity_engine is not None:
            # 训练 Similarity Engine (使用瘦身后的特征计算距离，更准！)
            self.similarity_engine.fit(df_train_raw, self.numeric_cols)
            
        if self.config['model'].get('use_retrieval', True):
            self.retrieved_indices = self._precompute_similarity()
        else:
            self.retrieved_indices = None

        self.seq_len = config['model']['seq_len']
        self.pred_len = config['model']['out_len']

    def _preprocess_table(self, df: pd.DataFrame, is_ref: bool = False) -> pd.DataFrame:
        """
        [特征工程核心]
        1. 注入 Lag Features
        2. 执行白名单筛选 (瘦身)
        """
        # 1. 注入 Lag (物理外挂)
        # 确保目标列存在
        target = self.config['preprocessing']['target_col']
        if target in df.columns:
            # Shift 会产生 NaN，我们用 bfill (向后填充) 处理
            df['lag_24'] = df[target].shift(24).fillna(method='bfill')
            df['lag_168'] = df[target].shift(168).fillna(method='bfill')
            if not is_ref and self.mode == 'train':
                self.logger.info(">>> 已注入物理外挂: Lag-24, Lag-168")
        
        # 2. 特征瘦身 (筛选列)
        cols_to_keep = []
        
        # A. 基础核心 (时间, 负荷, 价格, Lags)
        # 注意: 如果某些列不存在(比如 price), 代码需兼容
        base_candidates = ['time', target, 'price actual', 'lag_24', 'lag_168']
        for col in base_candidates:
            if col in df.columns:
                cols_to_keep.append(col)
        
        # B. 只保留"温度" (Temperature Only)
        # 剔除 humidity, pressure, wind, clouds, rain...
        for col in df.columns:
            if '_temp' in col: 
                cols_to_keep.append(col)
                
        # C. 必须保留文本列 (给 BERT)
        for col in df.columns:
            if 'weather_' in col: # 假设文本列包含 weather_
                cols_to_keep.append(col)
                
        # D. 执行筛选
        df_filtered = df[cols_to_keep].copy()
        
        if not is_ref and self.mode == 'train':
            self.logger.info(f"特征瘦身完成! 从 {len(df.columns)} 列 -> {len(df_filtered.columns)} 列")
            
        return df_filtered

    def _inject_time_features(self, df_source, tensor_source):
        """辅助函数：为指定数据注入时间特征"""
        timestamps = pd.to_datetime(df_source['time']).dt
        hour_feat = timestamps.hour.values / 23.0 
        day_feat = timestamps.dayofweek.values / 6.0
        month_feat = (timestamps.month.values - 1) / 11.0
        time_feats = np.stack([hour_feat, day_feat, month_feat], axis=1)
        time_feats_tensor = torch.tensor(time_feats, dtype=torch.float32)
        return torch.cat([tensor_source, time_feats_tensor], dim=1)

    def _load_or_compute_embeddings(self):
        """加载文本向量"""
        # 注意：文件名加上 slim 标记，防止读取旧的缓存
        cache_path = os.path.join(self.config['paths']['output_dir'], f"{self.mode}_text_emb_slim.npy")
        if os.path.exists(cache_path):
            emb = np.load(cache_path)
            # 校验长度
            if len(emb) == len(self.df):
                return torch.tensor(emb, dtype=torch.float32)
        
        self.logger.info("重新计算文本嵌入 (BERT)...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        text_inputs = []
        for i in range(len(self.df)):
            row_str = []
            for col in self.text_cols:
                # 提取城市名简写 (Madrid_weather_main -> Madrid weather_main)
                # 简单拼凑即可，BERT 能理解
                row_str.append(f"{col} {self.df.iloc[i][col]}")
            text_inputs.append("; ".join(row_str))
            
        batch_size = 256
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(text_inputs), batch_size), desc="Encoding"):
                batch = text_inputs[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
                outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        
        full_embeddings = np.concatenate(embeddings, axis=0)
        np.save(cache_path, full_embeddings)
        return torch.tensor(full_embeddings, dtype=torch.float32)

    def _precompute_similarity(self):
        """严谨的全量相似日计算"""
        if self.similarity_engine is None: return None
        cache_path = os.path.join(self.config['paths']['output_dir'], f"{self.mode}_sim_indices_slim.npy")
        
        if os.path.exists(cache_path):
            indices = np.load(cache_path)
            if len(indices) == len(self.df): return indices

        self.logger.info("计算相似日索引...")
        # 1. 准备 Query (去除最后3列时间特征)
        num_phys = self.data_numeric.shape[1] - 3
        raw_data = self.scaler.inverse_transform(self.data_numeric[:, :num_phys].numpy())
        query_norm = self.similarity_engine.scaler.transform(raw_data)
        query_tensor = torch.tensor(query_norm, dtype=torch.float32)
        
        # 2. 批量搜索
        all_indices = []
        bs = 256
        for i in tqdm(range(0, len(query_tensor), bs), desc="Search"):
            batch_res = self.similarity_engine.search(query_tensor[i:i+bs])
            all_indices.append(batch_res.numpy())
            
        full = np.concatenate(all_indices, axis=0)
        np.save(cache_path, full)
        return full

    def __getitem__(self, idx):
        # 切片定义
        s_end = idx + self.seq_len
        r_end = s_end + self.pred_len
        
        # 1. 当前输入
        x_num = self.data_numeric[idx:s_end]
        x_text = self.data_text[idx:s_end]
        
        # 2. 当前目标
        target_idx = self.numeric_cols.index(self.target_col)
        y = self.data_numeric[s_end:r_end, target_idx].unsqueeze(-1)
        
        # 3. 检索增强
        x_sim = torch.zeros(1)
        if self.config['model'].get('use_retrieval', True):
            if self.retrieved_indices is None: raise RuntimeError("Indices missing")
            
            anchor_idx = s_end - 1
            if anchor_idx >= len(self.retrieved_indices): anchor_idx = len(self.retrieved_indices) - 1
            
            sim_indices = self.retrieved_indices[anchor_idx]
            sim_seqs = []
            for sim_idx in sim_indices:
                sim_idx = int(sim_idx)
                end = sim_idx + 1
                start = end - self.seq_len
                if start < 0: start = 0; end = self.seq_len
                
                # 从训练集引用中读取
                sim_seq = self.train_data_ref[start:end]
                
                if len(sim_seq) != self.seq_len:
                    pad = torch.zeros(self.seq_len - len(sim_seq), sim_seq.shape[1])
                    sim_seq = torch.cat([pad, sim_seq], dim=0)
                sim_seqs.append(sim_seq)
            
            x_sim = torch.stack(sim_seqs)
            
        return (x_num, x_text), x_sim, y
    
    def __len__(self):
        return len(self.df) - self.seq_len - self.pred_len + 1