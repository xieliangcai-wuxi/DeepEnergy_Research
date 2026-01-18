import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional
import torch
from ..utils.logger import setup_logger

class SimilarityEngine:
    """
    [核心模块] 基于熵权法的相似日检索引擎 (Pro Version)
    
    Optimizations:
    1. Memory Efficiency: 使用矩阵乘法 (a-b)^2 = a^2 + b^2 - 2ab 替代广播，防止 OOM。
    2. Anti-Leakage: 训练模式下自动屏蔽"自身"，防止模型偷懒。
    3. Entropy Weighting: 自动计算特征重要性。
    """
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger(config['paths']['log_dir'], "SimilarityEngine")
        self.top_k = config['model']['top_k']
        
        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.historical_data = None 
        self.weights_tensor = None
        
    def fit(self, df_train: pd.DataFrame, feature_cols: List[str]):
        """计算熵权并建立索引库"""
        self.logger.info(">>> [Similarity] 初始化索引库 & 计算熵权...")
        
        # 1. 准备数据
        # ⚠️ 注意: feature_cols 绝对不能包含 'total load actual' (Target)
        X = df_train[feature_cols].values
        
        # 归一化是必须的，否则欧氏距离会被大数值特征主导
        X_scaled = self.scaler.fit_transform(X)
        
        # 存入 GPU (作为 Reference/Key)
        # Shape: [History_Len, Features]
        self.historical_data = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # 2. 熵权法计算 (保持你的逻辑)
        # 添加极小值防止 log(0)
        X_safe = X_scaled + 1e-9 
        P = X_safe / X_safe.sum(axis=0, keepdims=True)
        k = 1.0 / np.log(X_scaled.shape[0])
        entropy = -k * np.sum(P * np.log(P), axis=0)
        divergence = 1 - entropy
        weights = divergence / divergence.sum()
        
        # 3. 存入权重
        # Shape: [1, Features] -> 方便广播
        self.weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device).view(1, -1)
        
        # 打印 Top 权重，用于人工检查合理性
        weight_dict = dict(zip(feature_cols, weights))
        sorted_w = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        self.logger.info(f"    Top-5 熵权特征: {sorted_w}")

    def search(self, query_batch: torch.Tensor, training_mode: bool = False) -> torch.Tensor:
        """
        高性能全量搜索
        
        Args:
            query_batch: [Batch, Features]
            training_mode: 如果为 True，会屏蔽掉距离极小的样本（防止搜到自己）
        """
        # 1. 数据准备
        if query_batch.device != self.device:
            query_batch = query_batch.to(self.device)
            
        # 2. 加权欧氏距离优化计算
        # 公式: dist^2 = sum(w * (x - y)^2) 
        # 展开: sum(w*x^2) + sum(w*y^2) - 2 * sum(w*x*y)
        
        # Term 1: History squared (加权)
        # [History_Len, 1]
        H_sq = torch.sum(self.weights_tensor * (self.historical_data ** 2), dim=1, keepdim=True)
        
        # Term 2: Query squared (加权)
        # [Batch, 1]
        Q_sq = torch.sum(self.weights_tensor * (query_batch ** 2), dim=1, keepdim=True)
        
        # Term 3: Cross term (2xy)
        # Matrix Multiplication: [Batch, Feat] @ [Feat, History] -> [Batch, History]
        # 注意要乘权重: Q @ (W * H.T)
        weighted_history_T = (self.historical_data * self.weights_tensor).t()
        cross_term = torch.mm(query_batch, weighted_history_T)
        
        # Combine: x^2 + y^2 - 2xy
        # Broadcasting: [B,1] + [1,H] - [B,H] = [B,H]
        dist_sq = Q_sq + H_sq.t() - 2 * cross_term
        
        # 数值稳定性 (防止浮点误差导致负数)
        dist_sq = torch.clamp(dist_sq, min=0.0)
        
        # ---------------------------------------------------------
        # [关键] 防止泄露：如果是训练模式，屏蔽自己
        # ---------------------------------------------------------
        if training_mode:
            # 策略：不仅屏蔽自己，还要屏蔽过于接近的（可能是重复数据）
            # 我们生成一个 mask，把极小距离设为无穷大
            # 注意：这假设 query 就是 history 中的一部分，且距离确实为0
            # 更稳妥的做法是在 Dataset 层面对 indices 进行后处理（剔除 idx == query_idx）
            # 但在这里，我们可以简单地把距离 < 1e-6 的设为 Inf，强迫它找"第二像"的
            
            # 只有当 query 和 history 是同一个数据集时有效
            mask = dist_sq < 1e-5
            dist_sq[mask] = float('inf')

        # 获取 Top-K
        # values, indices = torch.topk(dist_sq, k=self.top_k, dim=1, largest=False)
        # 开根号才是欧氏距离，但为了排序，不开根号结果一样，省算力
        _, indices = torch.topk(dist_sq, k=self.top_k, dim=1, largest=False)
        
        return indices.cpu()