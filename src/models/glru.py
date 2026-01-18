import torch
import torch.nn as nn

class GLRU(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.rnn = nn.GRU(d_model, d_model, 1, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # [ASSERT] 输入形状检查
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"GLRU Input shape mismatch. Expected [B, L, {self.d_model}], got {x.shape}")
            
        rnn_out, _ = self.rnn(x)
        x = self.norm1(x + self.dropout1(rnn_out)) 
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x