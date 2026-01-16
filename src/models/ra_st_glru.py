import torch
import torch.nn as nn
from ..layers.dynamic_graph import DynamicGraphInteraction
from ..layers.glru import NativeGLRU
from ..layers.fusion import TextAdapter, RetrievalFusion

class RA_ST_GLRU(nn.Module):
    """
    [最终模型架构] RA-ST-GLRU
    Retrieval-Augmented Spatio-Temporal Gated Linear Recurrent Unit
    
    Update Log:
    - [Fix]: 在 __init__ 中增加了 dropout 参数，并传递给所有子模块，用于防止过拟合。
    - [Logic]: 修正了 DynamicGraphInteraction 的输入维度参数，确保与 Embedding 后的维度一致。
    """
    def __init__(self, 
                 num_nodes: int, 
                 in_features: int, 
                 d_model: int = 64, 
                 layers: int = 2, 
                 out_len: int = 24, 
                 top_k: int = 3,
                 use_retrieval: bool = True,
                 dropout: float = 0.1): # <--- [新增] 接收配置中的 dropout
        super().__init__()
        self.use_retrieval = use_retrieval
        
        # 1. 特征嵌入 (Embedding)
        # 将原始数值特征映射到 d_model
        # 例如: 输入 [Batch, Seq, 30] -> 输出 [Batch, Seq, 64]
        self.num_emb = nn.Linear(in_features, d_model)
        
        # 文本适配器 (将 768维 BERT 向量融合进来)
        self.text_adapter = TextAdapter(text_dim=768, d_model=d_model)
        
        # 2. 时空编码器骨干 (Backbone)
        self.st_layers = nn.ModuleList()
        for _ in range(layers):
            self.st_layers.append(nn.ModuleDict({
                # 空间层: 输入是 d_model (因为已经经过了 Embedding)
                # 这里的 input_dim=d_model, output_dim=d_model
                'spatial': DynamicGraphInteraction(
                    input_dim=d_model, 
                    num_nodes=num_nodes, 
                    d_model=d_model, 
                    dropout=dropout # <--- 传递 dropout
                ),
                # 时间层: 输入是 d_model
                'temporal': NativeGLRU(
                    d_model=d_model, 
                    dropout=dropout # <--- 传递 dropout
                )
            }))
            
        # 3. 相似日融合 (Retrieval)
        if use_retrieval:
            # 相似日特征也需要投影 (假设相似日只有数值特征)
            self.sim_emb = nn.Linear(in_features, d_model)
            
            # 融合层
            self.retrieval_fusion = RetrievalFusion(
                d_model=d_model, 
                dropout=dropout # <--- 传递 dropout
            )
            
        # 4. 预测头 (Prediction Head)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout), # <--- 这里的 dropout 也很重要
            nn.Linear(d_model // 2, out_len) # 直接预测未来 Out_Len 个步长
        )

    def forward(self, x_inputs, x_sim=None):
        """
        Args:
            x_inputs: Tuple (x_num, x_text)
                x_num: [Batch, Seq, Total_Features] (原始数值)
                x_text: [Batch, Seq, 768] (BERT向量)
            x_sim: [Batch, Top_K, Seq, Total_Features] (相似日数据)
        """
        x_num, x_text = x_inputs
        
        # [Step 1] Embedding & Text Fusion
        # 先把原始维度(30) 升维到 d_model(64)
        h = self.num_emb(x_num) # -> [B, S, d_model]
        
        # 注入文本信息
        h = self.text_adapter(h, x_text) # -> [B, S, d_model]
        
        # [Step 2] ST-GLRU Backbone
        for layer in self.st_layers:
            # 先过空间层 (Dynamic Graph)
            # Input: [B, S, 64] -> Output: [B, S, 64]
            h = layer['spatial'](h)
            
            # 再过时间层 (GLRU)
            # Input: [B, S, 64] -> Output: [B, S, 64]
            h = layer['temporal'](h)
            
        # [Step 3] Retrieval Augmentation
        if self.use_retrieval and x_sim is not None:
            # 相似日也需要过 Embedding: [B, K, S, 30] -> [B, K, S, 64]
            sim_h = self.sim_emb(x_sim)
            
            # Cross-Attention: 用当前的 h 去查询历史 sim_h
            h = self.retrieval_fusion(x_curr=h, x_sim=sim_h)
            
        # [Step 4] Prediction
        # 我们取最后一个时间步的特征进行预测
        # h: [Batch, Seq_Len, d_model] -> h_last: [Batch, d_model]
        h_last = h[:, -1, :] 
        
        out = self.head(h_last) # [Batch, Out_Len]
        
        return out.unsqueeze(-1) # [Batch, Out_Len, 1]