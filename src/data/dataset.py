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
    [核心数据载体] 支持多模态、检索增强的时间序列数据集。
    
    包含功能:
    1. 自动加载预处理后的 CSV。
    2. 文本特征的 DistilBERT 嵌入 (带缓存机制)。
    3. 数值特征的标准化 (StandardScaler)。
    4. 相似日检索 (Integration with SimilarityEngine)。
    5. 滑动窗口切分。
    """
    def __init__(self, config: dict, mode: str = 'train', similarity_engine: SimilarityEngine = None):
        super().__init__()
        self.config = config
        self.mode = mode
        self.logger = setup_logger(config['paths']['log_dir'], f"Dataset_{mode}")
        
        # 1. 加载数据
        file_path = os.path.join(config['paths']['output_dir'], f"{mode}.csv")
        self.df = pd.read_csv(file_path)
        self.df['time'] = pd.to_datetime(self.df['time'])
        
        # 2. 特征分类 (数值 vs 文本)
        # 自动识别列名: 包含 'weather_main' 或 'description' 的是文本
        all_cols = self.df.columns
        self.text_cols = [c for c in all_cols if 'weather_main' in c or 'description' in c]
        self.target_col = config['preprocessing']['target_col']
        self.numeric_cols = [c for c in all_cols if c not in self.text_cols and c != 'time']
        
        # 3. 标准化 (Standardization)
        # 严谨逻辑: 无论当前是 train/val/test, 归一化参数必须来自 train
        self.scaler = StandardScaler()
        train_path = os.path.join(config['paths']['output_dir'], "train.csv")
        df_train_ref = pd.read_csv(train_path)
        self.scaler.fit(df_train_ref[self.numeric_cols].values)
        
        # 应用归一化
        self.data_numeric = self.scaler.transform(self.df[self.numeric_cols].values)
        self.data_numeric = torch.tensor(self.data_numeric, dtype=torch.float32)
        
        # 4. 文本嵌入 (Text Embedding with Caching)
        self.data_text = self._load_or_compute_embeddings()
        
        # 5. 相似日检索引擎
        self.similarity_engine = similarity_engine
        # 如果是训练集且传入了引擎，需要先 fit
        if self.mode == 'train' and self.similarity_engine is not None:
            # 这里的 feature_cols 应该与 similarity.py 里用的一致，通常是用数值特征
            self.similarity_engine.fit(df_train_ref, self.numeric_cols)
            
        # 6. 预计算相似日索引 (为了训练速度，不要在 __getitem__ 里搜)
        # 仅当使用了检索增强时执行
        if self.config['model'].get('use_retrieval', True):
            self.retrieved_indices = self._precompute_similarity()
        else:
            self.retrieved_indices = None

        # 7. 序列参数
        self.seq_len = config['model']['seq_len']   # 历史窗口 (e.g., 96)
        self.pred_len = config['model']['out_len']  # 预测窗口 (e.g., 24)

    def _load_or_compute_embeddings(self):
        """
        检查是否有缓存的 .npy 文本向量，没有则调用 DistilBERT 计算。
        """
        cache_path = os.path.join(self.config['paths']['output_dir'], f"{self.mode}_text_emb.npy")
        
        if os.path.exists(cache_path):
            self.logger.info(f"加载缓存的文本嵌入: {cache_path}")
            return torch.tensor(np.load(cache_path), dtype=torch.float32)
        
        self.logger.info("未找到文本嵌入缓存，开始调用 DistilBERT 计算 (可能较慢)...")
        
        # 加载预训练模型 (HuggingFace)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        model.eval()
        
        # 构造文本输入: 将所有城市的描述拼在一起
        # "Madrid: clear sky; Barcelona: rain; ..."
        text_inputs = []
        for i in range(len(self.df)):
            row_str = []
            for col in self.text_cols:
                # col name example: "Madrid_weather_description"
                city = col.split('_')[0]
                desc = str(self.df.iloc[i][col])
                row_str.append(f"{city} {desc}")
            text_inputs.append("; ".join(row_str))
            
        # 分批处理防止显存爆炸
        batch_size = 256
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(text_inputs), batch_size), desc="Encoding Text"):
                batch_texts = text_inputs[i : i+batch_size]
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
                outputs = model(**inputs)
                # 取 [CLS] token 作为句向量
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_emb)
                
        full_embeddings = np.concatenate(embeddings, axis=0)
        np.save(cache_path, full_embeddings)
        self.logger.info(f"文本嵌入计算完成并保存. Shape: {full_embeddings.shape}")
        
        return torch.tensor(full_embeddings, dtype=torch.float32)

    def _precompute_similarity(self):
        """预先为数据集中的每一天计算相似日索引"""
        if self.similarity_engine is None:
            return None
            
        self.logger.info("正在预计算相似日索引...")
        cache_idx_path = os.path.join(self.config['paths']['output_dir'], f"{self.mode}_sim_indices.npy")
        
        if os.path.exists(cache_idx_path):
            return np.load(cache_idx_path)
            
        indices_list = []
        # 我们按“天”为单位进行检索，每24小时检索一次
        # 这里简化逻辑: 对每个时间步 t，我们取 t 时刻的特征向量去检索
        # 为了效率，我们可能间隔采样，但为了严谨，我们逐点计算(或逐日计算)
        
        # 优化: 仅计算每天 00:00 的相似日，全天复用
        # 提取日特征 (Daily Profile)
        # 这里暂时用当前时刻特征代替，实际可以优化为日均值
        feats = self.data_numeric.numpy()
        
        # 这是一个耗时操作，建议只对每天做一次
        # 为了演示代码连贯性，这里简单处理：每行都存对应的 Top-K
        # 实际上这会产生大量重复计算，科研中通常会 `df.resample('D').mean()` 后再检索
        
        # 临时简化: 直接使用全部数据计算 (注意: 只有几万行，EWM+欧氏距离很快)
        # 真正的科研代码这里应该写一个循环
        
        # 为了不卡死，我们先返回一个占位符，因为训练时会在 __getitem__ 里动态取?
        # 不，__getitem__ 动态搜太慢。
        # 我们采用：只计算每个样本对应历史库中的 Top-K
        
        # 鉴于此步骤可能需要较长时间，且需要 train_set 作为库
        # 我们暂时略过这里，留待 "exp_main" 脚本中显式调用生成逻辑
        return None

    def __len__(self):
        return len(self.df) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        """
        Returns:
            x_seq: [Seq_Len, Num_Features] (包含数值 + 文本嵌入)
            x_sim: [Top_K, Seq_Len, Num_Features] (检索到的相似日序列)
            y: [Pred_Len, 1] (预测目标)
        """
        # 1. 切分时间窗口
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        # 2. 获取当前序列
        # 数值特征
        x_num = self.data_numeric[s_begin:s_end]
        # 文本特征 (维度压缩或拼接? 既然是多模态，我们先拼接到一起，让模型去处理)
        # BERT是 768维，数值是 几十维。直接拼会导致数值被淹没。
        # 所以通常建议模型内部有独立的 Embedding 层。
        # 这里我们返回独立的 Tuple
        
        x_text = self.data_text[s_begin:s_end] # [Seq, 768]
        
        # 3. 获取预测目标 (只取 Total Load 列)
        # 假设 target_col 在 numeric_cols 中的索引是 target_idx
        # 为了方便，我们直接从原始 df 取值 (如果是预测真实值) 或者取归一化后的值
        # 通常预测归一化后的值，利于收敛
        target_idx = self.numeric_cols.index(self.target_col)
        y = self.data_numeric[r_begin:r_end, target_idx].unsqueeze(-1)
        
        # 4. 获取相似日数据 (如果启用)
        x_sim = torch.zeros(1) # Placeholder
        if self.similarity_engine is not None:
            # 取当前窗口最后一天的特征作为 Query
            query_feat = self.data_numeric[s_end-1].numpy()
            sim_indices = self.similarity_engine.search(query_feat) # [Top_K]
            
            # 构建相似日序列 (取相似日当天的整个24小时，或者同等长度的序列)
            # 这里假设 sim_idx 是某一行的索引，我们取该行前后的序列
            sim_seqs = []
            for sim_idx in sim_indices:
                # 边界检查
                start = max(0, int(sim_idx) - self.seq_len)
                end = start + self.seq_len
                # 只需要数值特征用于参考
                sim_seq = self.similarity_engine.historical_data[start:end]
                
                # 如果长度不够(边界)，padding
                if sim_seq.shape[0] < self.seq_len:
                    pad = torch.zeros(self.seq_len - sim_seq.shape[0], sim_seq.shape[1])
                    sim_seq = torch.cat([pad, sim_seq], dim=0)
                    
                sim_seqs.append(sim_seq)
            
            x_sim = torch.stack(sim_seqs) # [Top_K, Seq_Len, Feat]
            
        return (x_num, x_text), x_sim, y