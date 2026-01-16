import torch
import torch.nn as nn
import torch.nn.functional as F

class NativeGLRU(nn.Module):
    """
    [科研核心组件] Gated Linear Recurrent Unit (Native PyTorch Implementation)
    
    Function:
    捕捉长短期时间依赖 (Long-Short Term Dependencies)。
    
    Why this for Top-Tier Paper:
    1. 性能 (Performance): 它是 State Space Model (Mamba/S4) 的简化变体，比 Transformer 更快，比 LSTM 记忆更长。
    2. 效率 (Efficiency): 利用 1D 卷积实现训练时的并行化 (Parallel Training)，避免了 RNN 的串行瓶颈。
    3. 鲁棒性 (Robustness): 门控机制 (Gating) 有效抑制噪声。
    """
    def __init__(self, d_model: int, expand_ratio: int = 2, dropout: float = 0.1, kernel_size: int = 4):
        """
        Args:
            d_model: 输入特征维度
            expand_ratio: 内部维度扩展倍数 (类似于 Transformer FFN 的 expansion)
            kernel_size: 局部感受野大小 (用于捕捉短期局部特征)
        """
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_ratio
        self.kernel_size = kernel_size
        
        # 1. 输入投影 (Input Projection)
        # 将输入映射到更高的维度，分为两个分支: 
        # Branch A:用于卷积处理 (Content)
        # Branch B:用于门控信号 (Gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # 2. 因果深度卷积 (Causal Depthwise Conv1d)
        # Depthwise: 每个通道独立卷积，参数量少，防过拟合
        # Causal: 必须保证 t 时刻只能看到 t 及以前的数据，不能看到 t+1
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, 
            out_channels=self.d_inner, 
            kernel_size=kernel_size, 
            groups=self.d_inner, # Depthwise
            padding=kernel_size - 1 # 填充在左边，稍后切片实现因果
        )
        
        # 3. 激活函数
        # SiLU (Swish) 是目前 GLU 变体中最常用的激活函数
        self.act = nn.SiLU()
        
        # 4. 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # 5. 正则化
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input Tensor [Batch, Seq_Len, d_model]
        """
        batch, seq_len, _ = x.shape
        residual = x
        
        # [Step 1] 投影与分流
        # x: [B, S, D] -> [B, S, D_inner * 2]
        u = self.in_proj(x)
        # Split: content=[B, S, D_inner], gate=[B, S, D_inner]
        content, gate = u.chunk(2, dim=-1)
        
        # [Step 2] 因果卷积处理 (Temporal Mixing)
        # Conv1d 需要输入 [Batch, Channels, Length]
        content = content.permute(0, 2, 1) # -> [B, D_inner, S]
        
        # Conv: [B, D_inner, S + Kernel-1]
        content_conv = self.conv1d(content)
        
        # Causal Slicing (关键科研细节!)
        # 截取前 Seq_Len 长度，丢弃后面多余的 padding
        # 这样确保了 t 时刻的输出只依赖于 [t-(k-1), ..., t]
        content_conv = content_conv[..., :seq_len]
        
        # 还原维度: [B, S, D_inner]
        content_conv = content_conv.permute(0, 2, 1)
        
        # [Step 3] 门控机制 (Gating)
        # Information Flow = Activated_Content * Sigmoid(Gate)
        # 这一步实现了 "Selective Information Passing"
        # 门控分支控制了多少卷积提取的信息能流向下一层
        out_inner = self.act(content_conv) * torch.sigmoid(gate)
        
        # [Step 4] 输出投影
        out = self.out_proj(out_inner)
        out = self.dropout(out)
        
        # [Step 5] 残差连接与归一化
        # Result = Norm(x + GLRU(x))
        return self.norm(residual + out)