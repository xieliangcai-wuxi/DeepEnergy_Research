import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import torch
from ..utils.logger import setup_logger

class SimilarityEngine:
    """
    [核心创新模块] 基于熵权法(Entropy Weighting)的相似日检索引擎
    
    Update: 
    - 支持 Batch 矩阵运算，移除所有 for 循环，确保全量计算在几秒内完成。
    """
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger(config['paths']['log_dir'], "SimilarityEngine")
        self.top_k = config['model']['top_k']
        self.feature_weights = None
        self.historical_data = None 
        self.scaler = MinMaxScaler()
        # 将计算设备设为 GPU (如果可用)，加速距离计算
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, df_train: pd.DataFrame, feature_cols: List[str]):
        """
        计算熵权并缓存历史数据库。
        """
        self.logger.info("正在计算特征熵权 (Entropy Weights)...")
        
        # 1. 提取数值
        X = df_train[feature_cols].values
        
        # 2. 归一化 (熵权法要求)
        X_scaled = self.scaler.fit_transform(X)
        
        # 存入 GPU 显存，作为"历史记忆库"
        self.historical_data = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # 3. 熵权计算
        P = X_scaled / (X_scaled.sum(axis=0, keepdims=True) + 1e-9)
        k = 1.0 / np.log(X_scaled.shape[0])
        entropy = -k * np.sum(P * np.log(P + 1e-9), axis=0)
        divergence = 1 - entropy
        self.feature_weights = divergence / divergence.sum()
        
        # 权重转为 Tensor 并存入 GPU
        self.weights_tensor = torch.tensor(self.feature_weights, dtype=torch.float32).to(self.device)
        
        # 打印分析
        weight_dict = dict(zip(feature_cols, self.feature_weights))
        sorted_weights = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        self.logger.info(f"Top-5 熵权特征: {sorted_weights}")

    def search(self, query_batch: torch.Tensor) -> torch.Tensor:
        """
        全量矩阵搜索 Top-K。
        
        Args:
            query_batch: [Batch_Size, Features] 当前的一批查询数据
        Returns:
            top_k_indices: [Batch_Size, Top_K]
        """
        # 确保 query 在 GPU 上
        if query_batch.device != self.device:
            query_batch = query_batch.to(self.device)
            
        # 维度检查: 历史库 [History_Len, Feats], 查询 [Batch, Feats]
        # 我们需要计算两两之间的距离。
        # Distance矩阵形状应为 [Batch, History_Len]
        
        # 扩展维度以利用广播
        # History: [1, H, F]
        # Query:   [B, 1, F]
        # Diff:    [B, H, F]
        # 警告: 如果 History 和 Batch 都很大，显存会爆。
        # 因此我们在 Dataset 里控制 Batch Size，这里只管计算。
        
        history = self.historical_data.unsqueeze(0) # [1, H, F]
        query = query_batch.unsqueeze(1)            # [B, 1, F]
        
        # 加权欧氏距离: sum( w * (x - y)^2 )
        # 权重广播: [1, 1, F]
        w = self.weights_tensor.view(1, 1, -1)
        
        # 计算差的平方
        diff_sq = (history - query) ** 2
        
        # 加权求和 -> 开根号
        # [B, H, F] -> [B, H]
        dist_matrix = torch.sqrt(torch.sum(w * diff_sq, dim=2))
        
        # 获取 Top-K (最小距离)
        # indices: [B, K]
        _, indices = torch.topk(dist_matrix, k=self.top_k, dim=1, largest=False)
        
        return indices.cpu() # 转回 CPU 节省显存