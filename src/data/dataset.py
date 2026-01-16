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
    [最终修正版] PowerDataset
    
    Critical Fix:
    - [Index Alignment]: 修复了测试集检索时，索引指向训练集但数据从测试集读取的严重Bug。
      现在无论 Dataset 处于什么模式，x_sim 永远从训练集历史库 (train_data_ref) 中读取。
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
        # 2. 特征处理
        # ===========================
        all_cols = self.df.columns
        self.text_cols = [c for c in all_cols if 'weather_main' in c or 'description' in c]
        self.target_col = config['preprocessing']['target_col']
        self.numeric_cols = [c for c in all_cols if c not in self.text_cols and c != 'time']
        
        # ===========================
        # 3. 准备标准化器 & 训练集引用 (核心修复)
        # ===========================
        self.scaler = StandardScaler()
        train_path = os.path.join(config['paths']['output_dir'], "train.csv")
        df_train_raw = pd.read_csv(train_path)
        
        # Fit Scaler on Train
        self.scaler.fit(df_train_raw[self.numeric_cols].values)
        
        # [核心修复] 
        # 我们需要保留一份"标准化后的训练集数据"，专门用于检索切片。
        # 无论当前 dataset 是 val 还是 test，检索回来的索引 sim_idx 永远指向 Train Set。
        # 因此 x_sim 必须从 Train Set 里切出来。
        train_numeric_numpy = self.scaler.transform(df_train_raw[self.numeric_cols].values)
        self.train_data_ref = torch.tensor(train_numeric_numpy, dtype=torch.float32)
        
        # 同样，训练集也需要注入时间特征，否则维度不匹配 (30 vs 33)
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
            # 训练 Similarity Engine (只用原始物理特征)
            self.similarity_engine.fit(df_train_raw, self.numeric_cols)
            
        if self.config['model'].get('use_retrieval', True):
            self.retrieved_indices = self._precompute_similarity()
        else:
            self.retrieved_indices = None

        self.seq_len = config['model']['seq_len']
        self.pred_len = config['model']['out_len']

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
        cache_path = os.path.join(self.config['paths']['output_dir'], f"{self.mode}_text_emb.npy")
        if os.path.exists(cache_path):
            emb = np.load(cache_path)
            if len(emb) == len(self.df):
                return torch.tensor(emb, dtype=torch.float32)
        
        # 计算逻辑 (简略版，因为之前已经提供过完整版，逻辑不变)
        self.logger.info("重新计算文本嵌入...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        text_inputs = []
        for i in range(len(self.df)):
            row_str = []
            for col in self.text_cols:
                row_str.append(f"{col.split('_')[0]} {self.df.iloc[i][col]}")
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
        cache_path = os.path.join(self.config['paths']['output_dir'], f"{self.mode}_sim_indices.npy")
        
        if os.path.exists(cache_path):
            indices = np.load(cache_path)
            if len(indices) == len(self.df): return indices

        self.logger.info("计算相似日索引...")
        # 1. 准备 Query (去除最后3列时间特征)
        num_phys = self.data_numeric.shape[1] - 3
        # 必须先反归一化 (Standard Inverse) -> 再归一化 (MinMax Transform)
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
        """
        [Final Logic Check]
        - Input/Target: 来自当前数据集 (Train/Val/Test)
        - Retrieval: 索引指向 Train Set，数据也必须从 Train Set 读取！
        """
        # 切片定义
        s_end = idx + self.seq_len
        r_end = s_end + self.pred_len
        
        # 1. 当前输入 (Input)
        x_num = self.data_numeric[idx:s_end]
        x_text = self.data_text[idx:s_end]
        
        # 2. 当前目标 (Target)
        target_idx = self.numeric_cols.index(self.target_col)
        y = self.data_numeric[s_end:r_end, target_idx].unsqueeze(-1)
        
        # 3. 检索增强 (Retrieval)
        x_sim = torch.zeros(1)
        if self.config['model'].get('use_retrieval', True):
            if self.retrieved_indices is None: raise RuntimeError("Indices missing")
            
            # 使用输入窗口的"最后一步"作为锚点
            anchor_idx = s_end - 1
            if anchor_idx >= len(self.retrieved_indices): anchor_idx = len(self.retrieved_indices) - 1
            
            sim_indices = self.retrieved_indices[anchor_idx]
            
            sim_seqs = []
            for sim_idx in sim_indices:
                # sim_idx 指向的是【训练集】中的某一行
                sim_idx = int(sim_idx)
                end = sim_idx + 1
                start = end - self.seq_len
                
                if start < 0: start = 0; end = self.seq_len
                
                # [核心修正点] 
                # 必须从 self.train_data_ref (训练集) 中读取！
                # 绝对不能从 self.data_numeric (可能是测试集) 中读取！
                sim_seq = self.train_data_ref[start:end]
                
                if len(sim_seq) != self.seq_len:
                    pad = torch.zeros(self.seq_len - len(sim_seq), sim_seq.shape[1])
                    sim_seq = torch.cat([pad, sim_seq], dim=0)
                sim_seqs.append(sim_seq)
            
            x_sim = torch.stack(sim_seqs)
            
        return (x_num, x_text), x_sim, y
    
    def __len__(self):
        return len(self.df) - self.seq_len - self.pred_len + 1