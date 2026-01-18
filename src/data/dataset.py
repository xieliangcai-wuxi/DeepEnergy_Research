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
    [Dataset] S2-RA-ST-GLRU Data Pipeline
    Includes strict sorting and leakage prevention.
    """
    def __init__(self, config: dict, mode: str = 'train', similarity_engine: SimilarityEngine = None):
        super().__init__()
        self.config = config
        self.mode = mode
        self.logger = setup_logger(config['paths']['log_dir'], f"Dataset_{mode}")
        self.target_col = config['preprocessing']['target_col']
        
        # 1. Load Data
        file_path = os.path.join(config['paths']['output_dir'], f"{mode}.csv")
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Data file not found: {file_path}")
        self.df = pd.read_csv(file_path)
        self.df['time'] = pd.to_datetime(self.df['time'])
        
        # 2. Dynamic Feature Engineering
        self.df = self._preprocess_table(self.df, is_ref=False)
        
        # 3. Column Definition (Strict Sorting)
        all_cols = self.df.columns.tolist()
        self.text_cols = sorted([c for c in all_cols if 'weather_' in c])
        raw_numeric = [c for c in all_cols if c not in self.text_cols and c != 'time']
        self.numeric_cols = sorted(raw_numeric) # [CRITICAL] Sort ensures consistency
        
        # 4. Search Columns (Exclude Target)
        self.search_cols = sorted([c for c in self.numeric_cols if c != self.target_col])
        self.search_col_indices = [self.numeric_cols.index(c) for c in self.search_cols]
        
        # [自检] 验证 Target 是否在 Numeric Cols 里
        if self.target_col not in self.numeric_cols:
             raise ValueError(f"Target '{self.target_col}' not found in numeric columns!")
        
        # 5. Prepare Scaler & Reference Data
        self.scaler = StandardScaler()
        train_path = os.path.join(config['paths']['output_dir'], "train.csv")
        df_train_raw = pd.read_csv(train_path)
        df_train_raw['time'] = pd.to_datetime(df_train_raw['time'])
        df_train_raw = self._preprocess_table(df_train_raw, is_ref=True)
        
        # Fit Scaler
        self.scaler.fit(df_train_raw[self.numeric_cols].values)
        
        # Transform & Tensorize
        train_num = self.scaler.transform(df_train_raw[self.numeric_cols].values)
        self.train_data_ref = self._to_tensor(df_train_raw, train_num)
        
        curr_num = self.scaler.transform(self.df[self.numeric_cols].values)
        self.data_numeric = self._to_tensor(self.df, curr_num)

        # 6. BERT & Similarity
        self.data_text = self._load_embeddings()
        
        self.similarity_engine = similarity_engine
        if self.mode == 'train' and self.similarity_engine:
            self.similarity_engine.fit(df_train_raw, self.search_cols)
            
        if self.config['model'].get('use_retrieval', True):
            extra = 1 if mode == 'train' else 0
            self.retrieved_indices = self._precompute_similarity(extra)
        else:
            self.retrieved_indices = None

        self.seq_len = config['model']['seq_len']
        self.pred_len = config['model']['out_len']
        
        # [日志] 输出关键信息
        if self.mode == 'train':
            self.logger.info(f"Numeric Columns ({len(self.numeric_cols)}): {self.numeric_cols}")
            self.logger.info(f"Target Column Index: {self.numeric_cols.index(self.target_col)}")

    def get_target_idx(self):
        """Helper to safely get target index"""
        return self.numeric_cols.index(self.target_col)

    def _preprocess_table(self, df, is_ref=False):
        df = df.copy()
        if self.target_col in df.columns:
            df['lag_24'] = df[self.target_col].shift(24).bfill()
            df['lag_168'] = df[self.target_col].shift(168).bfill()
        
        if 'time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                 df['time'] = pd.to_datetime(df['time'], utc=True)
            doy = df['time'].dt.dayofyear
            df['sin_doy'] = np.sin(2 * np.pi * doy / 365.0)
            df['cos_doy'] = np.cos(2 * np.pi * doy / 365.0)
        
        # Whitelist & Sort
        cols_to_keep = set()
        base = ['time', self.target_col, 'price actual', 'lag_24', 'lag_168', 'is_holiday_int', 'sin_doy', 'cos_doy']
        for col in base:
            if col in df.columns: cols_to_keep.add(col)
        for col in df.columns:
            if '_temp' in col: cols_to_keep.add(col)
        for col in df.columns:
            if 'weather_' in col: cols_to_keep.add(col)
        
        # FillNa is critical
        return df[sorted(list(cols_to_keep))].fillna(0)

    def _to_tensor(self, df_source, tensor_source):
        timestamps = pd.to_datetime(df_source['time']).dt
        hour_feat = timestamps.hour.values / 23.0 
        day_feat = timestamps.dayofweek.values / 6.0
        month_feat = (timestamps.month.values - 1) / 11.0
        time_feats = np.stack([hour_feat, day_feat, month_feat], axis=1)
        return torch.cat([torch.tensor(tensor_source, dtype=torch.float32), 
                          torch.tensor(time_feats, dtype=torch.float32)], dim=1)

    def _load_embeddings(self):
        cache_path = os.path.join(self.config['paths']['output_dir'], f"{self.mode}_text_emb_social_fixed.npy")
        if os.path.exists(cache_path):
            emb = np.load(cache_path)
            if len(emb) == len(self.df): return torch.tensor(emb, dtype=torch.float32)
        
        # BERT Encoding Logic (省略详细代码以节省篇幅，保持之前的一样即可)
        # 如果需要，我可以补充完整，但这里假设你已经有这部分逻辑
        self.logger.info("Computing BERT (This might take a while)...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        text_inputs = ["; ".join([f"{col} {self.df.iloc[i][col]}" for col in self.text_cols]) for i in range(len(self.df))]
        
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(text_inputs), 256), desc="BERT"):
                inputs = tokenizer(text_inputs[i:i+256], return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
                outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        
        full = np.concatenate(embeddings, axis=0)
        np.save(cache_path, full)
        return torch.tensor(full, dtype=torch.float32)

    def _precompute_similarity(self, extra_k=0):
        cache_path = os.path.join(self.config['paths']['output_dir'], f"{self.mode}_sim_indices_social_fixed.npy")
        if os.path.exists(cache_path): return np.load(cache_path)

        self.logger.info("Computing Similarity...")
        if not hasattr(self.similarity_engine, 'weights_tensor'): 
             raise RuntimeError("SimilarityEngine Not Fitted")

        num_phys = len(self.numeric_cols)
        raw_phys = self.scaler.inverse_transform(self.data_numeric[:, :num_phys].numpy())
        search_data = raw_phys[:, self.search_col_indices]
        query_norm = self.similarity_engine.scaler.transform(search_data)
        
        orig_k = self.similarity_engine.top_k
        self.similarity_engine.top_k += extra_k 
        
        all_indices = []
        query_tensor = torch.tensor(query_norm, dtype=torch.float32)
        for i in tqdm(range(0, len(query_tensor), 256), desc="Retrieving"):
            batch_res = self.similarity_engine.search(query_tensor[i:i+256], training_mode=(self.mode=='train'))
            all_indices.append(batch_res.numpy())
            
        self.similarity_engine.top_k = orig_k
        full = np.concatenate(all_indices, axis=0)
        np.save(cache_path, full)
        return full

    def __getitem__(self, idx):
        s_end = idx + self.seq_len
        r_end = s_end + self.pred_len
        
        x_num = self.data_numeric[idx:s_end]
        x_text = self.data_text[idx:s_end]
        target_idx = self.numeric_cols.index(self.target_col)
        y = self.data_numeric[s_end:r_end, target_idx].unsqueeze(-1)
        
        x_sim = torch.zeros(1)
        if self.config['model'].get('use_retrieval', True):
            if self.retrieved_indices is None: raise RuntimeError("Indices missing")
            
            anchor_idx = min(s_end - 1, len(self.retrieved_indices) - 1)
            raw_indices = self.retrieved_indices[anchor_idx]
            
            # Anti-Leakage: Skip Self
            final_indices = [idx for idx in raw_indices if not (self.mode == 'train' and int(idx) == anchor_idx)]
            final_indices = final_indices[:self.config['model']['top_k']]
            while len(final_indices) < self.config['model']['top_k']:
                final_indices.append(final_indices[-1] if final_indices else 0)
            
            sim_seqs = []
            for sim_idx in final_indices:
                end = int(sim_idx) + 1
                start = end - self.seq_len
                sim_seq = self.train_data_ref[max(0, start):end]
                if len(sim_seq) != self.seq_len:
                    pad = torch.zeros(self.seq_len - len(sim_seq), sim_seq.shape[1])
                    sim_seq = torch.cat([pad, sim_seq], dim=0)
                sim_seqs.append(sim_seq)
            x_sim = torch.stack(sim_seqs)
            
        return (x_num, x_text), x_sim, y
    
    def __len__(self):
        return len(self.df) - self.seq_len - self.pred_len + 1