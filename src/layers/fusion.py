import torch
import torch.nn as nn
import torch.nn.functional as F

class TextAdapter(nn.Module):
    """
    [科研组件 A] 文本适配器 (Text Adapter)
    
    Function:
    将预训练大模型 (DistilBERT) 的高维语义向量 (768维) 映射到模型内部空间 (d_model)，
    并使用门控机制 (Gating) 动态控制文本信息注入的强度。
    
    Why:
    直接相加会破坏数值特征的分布。门控机制允许模型在"天气描述无用"时自动关闭该通道。
    """
    def __init__(self, text_dim: int = 768, d_model: int = 64):
        super().__init__()
        # 1. 降维投影
        self.proj = nn.Linear(text_dim, d_model)
        
        # 2. 注入门控 (Injection Gate)
        # sigmoid(Gate) * Text + (1 - sigmoid(Gate)) * Numeric
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_num: torch.Tensor, x_text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_num: 数值特征 [Batch, Seq, d_model]
            x_text: 文本嵌入 [Batch, Seq, 768]
        Returns:
            Fused Tensor [Batch, Seq, d_model]
        """
        # [Step 1] 投影文本特征
        # [B, S, 768] -> [B, S, d_model]
        text_emb = self.proj(x_text)
        
        # [Step 2] 计算融合门控
        # 拼接数值和文本: [B, S, d_model * 2]
        cat_feat = torch.cat([x_num, text_emb], dim=-1)
        gate = self.gate_net(cat_feat)
        
        # [Step 3] 加权融合
        # 类似于 LSTM 的 forget/input gate
        out = (1 - gate) * x_num + gate * text_emb
        
        return self.norm(out)


class RetrievalFusion(nn.Module):
    """
    [科研组件 B] 检索增强融合层 (Retrieval-Augmented Fusion Layer)
    
    Function:
    利用 Cross-Attention 机制，让当前的时间序列 (Query) 去'查询'历史相似日 (Key/Value)，
    从而捕获历史上的重复模式 (Recurring Patterns)。
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Cross-Attention
        # batch_first=True: 输入输出格式为 [Batch, Seq, Feature]
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 简单的 Feed Forward 网络 (FFN) 用于处理融合后的特征
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x_curr: torch.Tensor, x_sim: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_curr: 当前时刻特征 (Query) [Batch, Seq_Len, d_model]
            x_sim:  相似日特征 (Key/Value) [Batch, Top_K, Seq_Len, d_model] 
                    (注意: 这里假设 x_sim 已经经过了 Embedding)
                    
        Returns:
            Enhanced Tensor [Batch, Seq_Len, d_model]
        """
        batch, top_k, seq_len, d_model = x_sim.shape
        
        # [Step 1] 构造 Key/Value 记忆库
        # 策略: 将 Top_K 个相似日的所有时间步拼在一起，形成一个巨大的"历史参考序列"
        # View: [Batch, Top_K * Seq_Len, d_model]
        # 物理意义: 模型可以在 3天 * 96小时 = 288个历史时间点中自由搜索最相似的模式
        kv_memory = x_sim.view(batch, top_k * seq_len, d_model)
        
        # [Step 2] Cross-Attention
        # Query: 当前序列
        # Key/Value: 历史记忆
        # attn_output: [Batch, Seq_Len, d_model]
        attn_output, _ = self.attn(query=x_curr, key=kv_memory, value=kv_memory)
        
        # [Step 3] 残差连接 (Residual)
        # 原始特征 + 历史修正特征
        x_fused = self.norm(x_curr + self.dropout(attn_output))
        
        # [Step 4] FFN
        x_out = self.norm_ffn(x_fused + self.ffn(x_fused))
        
        return x_out