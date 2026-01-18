import torch
import torch.nn as nn

class RetrievalAttention(nn.Module):
    def __init__(self, d_model, top_k, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, keys_values):
        # [ASSERT] 维度检查
        # query: [B, 1, D]
        # kv:    [B, K, D]
        if query.shape[-1] != keys_values.shape[-1]:
             raise ValueError(f"Dim mismatch: Query {query.shape[-1]} vs KV {keys_values.shape[-1]}")
             
        attn_out, _ = self.attn(query, keys_values, keys_values)
        return self.norm(query + self.dropout(attn_out))