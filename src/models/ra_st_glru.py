import torch
import torch.nn as nn
from .glru import GLRU
from .retrieval_attention import RetrievalAttention

class RA_ST_GLRU(nn.Module):
    """
    [Logic Audit] 
    Architecture: Residual Time-Series Network
    Flow: 
      1. Input (History) -> Lag-24 Baseline (Shortcut)
      2. Input (History) -> GLRU + Retrieval -> Residual Correction (Main Branch)
      3. Output = Baseline + Residual
    """
    def __init__(self, num_nodes, in_features, d_model, layers, out_len, top_k, target_idx, use_retrieval=True, dropout=0.1):
        super().__init__()
        
        # --- 参数校验 ---
        self.d_model = d_model
        self.out_len = out_len
        self.target_idx = target_idx
        self.in_features = in_features
        self.use_retrieval = use_retrieval
        
        if target_idx >= in_features:
            raise ValueError(f"Target Index {target_idx} out of bounds for features {in_features}")

        # --- 1. 编码层 (Projections) ---
        # 为什么是 +768? 因为 BERT embedding 是 768 维
        self.input_proj_current = nn.Sequential(
            nn.Linear(in_features + 768, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        if self.use_retrieval:
            self.input_proj_sim = nn.Sequential(
                nn.Linear(in_features, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )

        # --- 2. 时序骨干 (Backbone) ---
        # 使用 ModuleList 以便后续扩展或检查
        self.glru_layers = nn.ModuleList([
            GLRU(d_model, dropout) for _ in range(layers)
        ])
        
        # --- 3. 检索增强 (Retrieval) ---
        if self.use_retrieval:
            self.retrieval_attn = RetrievalAttention(d_model, top_k, dropout)
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid() 
            )

        # --- 4. 解码头 (Decoder Head) ---
        # [逻辑升级] 使用 MLP 而不是单层 Linear
        # 原因：残差与特征的关系是非线性的，MLP 能更好地拟合"误差"
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),              # 更好的激活函数
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_len)
        )

    def forward(self, x_current, x_sim):
        """
        Input Flow:
        x_num: [Batch, Seq, Feat]
        x_text: [Batch, Seq, 768]
        """
        (x_num, x_text) = x_current
        
        # [Check 1] 输入完整性
        B, Seq, F = x_num.shape
        if Seq < self.out_len: 
            raise ValueError(f"Sequence length {Seq} too short for shortcut {self.out_len}")

        # ==========================================
        # Branch A: 物理捷径 (Shortcut)
        # ==========================================
        # 逻辑：取 x_num 的最后 24 个时间步的 Target 值。
        # 含义：假设明天和今天同一时间完全一样 (Persistence Model)。
        # 形状流向：[B, Seq, F] -> [B, Out_Len]
        baseline = x_num[:, -self.out_len:, self.target_idx] 
        
        # ==========================================
        # Branch B: 残差预测 (Neural Network)
        # ==========================================
        
        # 1. 融合
        # [B, Seq, F+768]
        x_fused = torch.cat([x_num, x_text], dim=-1)
        
        # 2. 投影
        # [B, Seq, d_model]
        x_emb = self.input_proj_current(x_fused)
        
        # 3. GLRU 处理
        h_seq = x_emb
        for layer in self.glru_layers:
            h_seq = layer(h_seq)
        
        # 4. 提取上下文 (Last Step)
        # [B, d_model]
        context = h_seq[:, -1, :] 
        
        # 5. 检索增强
        if self.use_retrieval:
            b, k, l, f = x_sim.shape
            # [Check 2] 历史特征维度对齐
            if f != self.in_features: raise ValueError("Retrieval feature mismatch")
            
            # [B*K, L, F] -> [B*K, L, d_model]
            x_sim_emb = self.input_proj_sim(x_sim.view(b*k, l, f))
            
            # Mean Pooling over time (压缩时间维)
            # [B*K, d_model]
            x_sim_vec = x_sim_emb.mean(dim=1)
            
            # [B, K, d_model]
            keys_values = x_sim_vec.view(b, k, self.d_model)
            
            # Attention
            retrieval_out = self.retrieval_attn(context.unsqueeze(1), keys_values).squeeze(1)
            
            # Gating
            g = self.fusion_gate(torch.cat([context, retrieval_out], dim=-1))
            h_final = context + g * retrieval_out
        else:
            h_final = context

        # 6. 预测残差
        # [B, d_model] -> [B, out_len]
        pred_residual = self.output_head(h_final)
        
        # ==========================================
        # Merge: Summation
        # ==========================================
        
        # [Check 3] 维度严格检查
        if baseline.shape != pred_residual.shape:
            raise ValueError(f"Shape Mismatch: Baseline {baseline.shape} vs Residual {pred_residual.shape}")
            
        # Final = Yesterday + Delta
        final_pred = baseline + pred_residual
        
        # [B, out_len, 1] - 增加最后一维以匹配 Label
        return final_pred.unsqueeze(-1)