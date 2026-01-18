import torch
import torch.nn as nn
from .glru import GLRU
from .retrieval_attention import RetrievalAttention
from .revin import RevIN

class RA_ST_GLRU(nn.Module):
    def __init__(self, num_nodes, in_features, d_model, layers, out_len, top_k, target_idx, use_retrieval=True, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.out_len = out_len
        self.target_idx = target_idx
        self.in_features = in_features
        self.use_retrieval = use_retrieval
        
        self.revin = RevIN(in_features)

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

        self.glru_layers = nn.ModuleList([
            GLRU(d_model, dropout) for _ in range(layers)
        ])
        
        if self.use_retrieval:
            self.retrieval_attn = RetrievalAttention(d_model, top_k, dropout)
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid() 
            )

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_len)
        )

    def forward(self, x_current, x_sim, debug=False):
        """
        完整的前向传播逻辑 (SOTA 严谨版)
        包含：RevIN -> No Masking -> Neural Net -> Shortcut -> Rigorous Denorm
        """
        (x_num, x_text) = x_current
        
        if debug: print("\n🔍 [Model Internals] Start Forward Pass...")

        # ==========================================
        # Step 1: RevIN Normalization (安检入口)
        # ==========================================
        # 消除非平稳性。注意：这里包含了 Affine Transform (乘 weight 加 bias)
        x_num_norm = self.revin(x_num, mode='norm')
        
        if debug:
            print(f"   1. Input Norm | Mean: {x_num_norm.mean():.4f} | Std: {x_num_norm.std():.4f}")

        # ==========================================
        # Step 2: Main Branch (神经网络主路)
        # ==========================================
        # 🚨 关键：无 Masking！保留完整视力，让模型看到趋势。
        x_fused = torch.cat([x_num_norm, x_text], dim=-1)
        x_emb = self.input_proj_current(x_fused)
        
        # GLRU 提取时序特征
        h_seq = x_emb
        for layer in self.glru_layers:
            h_seq = layer(h_seq)
        context = h_seq[:, -1, :] 
        
        # RAG 检索增强
        if self.use_retrieval:
            b, k, l, f = x_sim.shape
            x_sim_flat = x_sim.view(b * k, l, f)
            x_sim_emb = self.input_proj_sim(x_sim_flat)
            x_sim_vec = x_sim_emb.mean(dim=1)
            keys_values = x_sim_vec.view(b, k, self.d_model)
            
            retrieval_out = self.retrieval_attn(context.unsqueeze(1), keys_values).squeeze(1)
            g = self.fusion_gate(torch.cat([context, retrieval_out], dim=-1))
            h_final = context + g * retrieval_out
        else:
            h_final = context

        # MLP Head 预测残差 (在归一化空间下)
        pred_residual_norm = self.output_head(h_final)

        # ==========================================
        # Step 3: Direct Method / Shortcut (物理捷径)
        # ==========================================
        # 这就是你要找的“直接方法”！
        # 逻辑：直接截取 normalized input 的最后 24 个点
        # 意义：假设归一化后的明天 = 归一化后的昨天
        baseline_norm = x_num_norm[:, -self.out_len:, self.target_idx]
        
        # 融合：捷径 + 残差
        final_pred_norm = baseline_norm + pred_residual_norm
        
        if debug:
            print(f"   2. Pred(Norm) | Mean: {final_pred_norm.mean():.4f} | Std: {final_pred_norm.std():.4f}")

        # ==========================================
        # Step 4: RevIN Denormalization (严谨反归一化)
        # ==========================================
        # 必须先逆转 Affine，再逆转 Mean/Std
        
        # A. 逆转仿射变换 (Reverse Affine)
        # 公式: x = (x - bias) / weight
        if self.revin.affine:
            # 取出 Target 列对应的标量参数
            target_weight = self.revin.affine_weight[self.target_idx]
            target_bias = self.revin.affine_bias[self.target_idx]
            
            # 广播计算 (Batch, 24) - Scalar
            final_pred_norm = (final_pred_norm - target_bias) / (target_weight + 1e-10)

        # B. 逆转统计量 (Reverse Stats)
        # 公式: x = x * std + mean
        # 取出 Target 列对应的统计量 [Batch, 1, F] -> [Batch, 1]
        target_mean = self.revin.mean[:, :, self.target_idx]
        target_std = self.revin.stdev[:, :, self.target_idx]
        
        # 广播计算 (Batch, 24) * (Batch, 1)
        final_pred = final_pred_norm * target_std + target_mean
        
        if debug:
            print(f"   3. Final Output | Mean: {final_pred.mean():.4f} | Std: {final_pred.std():.4f}")
            print("✅ [Model Internals] Forward Pass Complete.\n")

        # 恢复形状 [Batch, Out_Len, 1]
        return final_pred.unsqueeze(-1)