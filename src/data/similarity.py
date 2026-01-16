import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import torch
from ..utils.logger import setup_logger

class SimilarityEngine:
    """
    [核心创新模块] 基于熵权法(Entropy Weighting)的相似日检索引擎
    
    Research Goal:
    不依赖人工经验，自动计算不同特征(温度、电价、湿度)在相似度计算中的权重。
    """
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger(config['paths']['log_dir'], "SimilarityEngine")
        self.top_k = config['model']['top_k'] # 需要在yaml里加这个参数
        self.feature_weights = None
        self.historical_data = None # 存储归一化后的历史数据
        self.scaler = MinMaxScaler()
        
    def fit(self, df_train: pd.DataFrame, feature_cols: List[str]):
        """
        在训练集上计算特征权重 (Entropy Weights)。
        必须只用训练集，防止 Data Leakage。
        """
        self.logger.info("正在计算特征熵权 (Entropy Weights)...")
        
        # 1. 提取数值特征矩阵
        X = df_train[feature_cols].values
        
        # 2. 归一化 (熵权法要求非负)
        # 注意: 这里fit的scaler只用于相似度计算，与模型输入的scaler独立
        X_scaled = self.scaler.fit_transform(X)
        self.historical_data = torch.tensor(X_scaled, dtype=torch.float32)
        
        # 3. 计算熵权 (Entropy Method)
        # 添加微小量防止log(0)
        P = X_scaled / (X_scaled.sum(axis=0, keepdims=True) + 1e-9)
        
        # 计算熵值 e_j = -k * sum(p_ij * ln(p_ij))
        k = 1.0 / np.log(X_scaled.shape[0])
        entropy = -k * np.sum(P * np.log(P + 1e-9), axis=0)
        
        # 计算差异系数 d_j = 1 - e_j
        divergence = 1 - entropy
        
        # 归一化得到权重
        self.feature_weights = divergence / divergence.sum()
        
        # 转为Tensor方便GPU加速 (如果数据量极大)
        self.weights_tensor = torch.tensor(self.feature_weights, dtype=torch.float32)
        
        # 打印权重供分析 (可解释性)
        weight_dict = dict(zip(feature_cols, self.feature_weights))
        # 按权重大小排序打印前5个
        sorted_weights = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        self.logger.info(f"Top-5 重要特征权重: {sorted_weights}")

    def search(self, query_features: np.ndarray) -> torch.Tensor:
        """
        为查询目标寻找 Top-K 相似日。
        
        Args:
            query_features: [Sequence_Len, Features] 或 [Batch, Features]
            这里我们简化逻辑: 每次查询一天的'日特征向量'(Day Profile)
            
        Returns:
            top_k_indices: [Top_K] 对应训练集中的行索引
        """
        # 1. 预处理查询向量
        # 假设 query_features 是单日的平均特征向量 [Features]
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
            
        query_norm = self.scaler.transform(query_features)
        query_tensor = torch.tensor(query_norm, dtype=torch.float32)
        
        # 2. 计算加权欧氏距离 (Weighted Euclidean Distance)
        # Dist = sqrt( sum( w_i * (x_i - y_i)^2 ) )
        # 利用广播机制: [History_Len, Feats] - [1, Feats]
        diff = self.historical_data - query_tensor
        weighted_sq_diff = self.weights_tensor * (diff ** 2)
        dist = torch.sqrt(weighted_sq_diff.sum(dim=1))
        
        # 3. 获取 Top-K (最小距离)
        # largest=False 表示取最小的距离
        values, indices = torch.topk(dist, k=self.top_k, largest=False)
        
        return indices