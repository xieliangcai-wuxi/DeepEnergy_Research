import torch
import torch.nn as nn
from .glru import GLRU
from .retrieval_attention import RetrievalAttention
from .revin import RevIN

class RA_ST_GLRU(nn.Module):
    """
    [Final SOTA Architecture: Zero-Init Residual]
    策略变更：
    1. 移除 Masking：保证训练和测试数据分布一致。
    2. 零初始化 (Zero-Init)：强制模型从 Baseline (4.19%) 起跑，只学习偏差。
    """
    def __init__(self, num_nodes, in_features, d_model, layers, out_len, top_k, target_idx, use_retrieval=True, dropout=0.1):
        super().__init__()
        
        # --- Configs ---
        self.d_model = d_model
        self.out_len = out_len
        self.target_idx = target_idx
        self.in_features = in_features
        self.use_retrieval = use_retrieval
        
        # --- 1. RevIN ---
        self.revin = RevIN(in_features, affine=True)

        # --- 2. Projections ---
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

        # --- 3. Backbone ---
        self.glru_layers = nn.ModuleList([
            GLRU(d_model, dropout) for _ in range(layers)
        ])
        
        # --- 4. RAG ---
        if self.use_retrieval:
            self.retrieval_attn = RetrievalAttention(d_model, top_k, dropout)
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid() 
            )

        # --- 5. Output Heads (关键修改) ---
        
        # Head A: Residual Content
        # 最后一层 Linear 初始化为 0，确保初始输出为 0
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_len)
        )
        
        # Head B: Confidence Gate
        # 初始化为让 Gate 接近 0 (完全信任 Shortcut)
        self.confidence_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, out_len), 
            nn.Sigmoid() 
        )
        
        # 🚨【必杀技】零初始化 (Zero Initialization)
        # 强迫模型一开始"闭嘴"，完全等同于 Shortcut
        self._zero_init_head()

    def _zero_init_head(self):
        """
        [修正版] 零初始化策略 v2
        目标：保持初始 Loss 低 (4.19%)，同时保证梯度畅通。
        """
        print("⚡ [Init] Applying Zero-Initialization (Gradient-Friendly Version)...")
        
        # 1. 残差内容层：必须全 0
        # 这样 Neural Output = 0
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)
        
        # 2. Gate 层：Bias 设为 0 (关键修改！)
        # 之前是 -5.0 (导致梯度消失)
        # 现在是 0.0 -> Sigmoid(0) = 0.5 -> 梯度最大！
        # 初始状态: Final = Shortcut + 0.5 * 0 = Shortcut (依然稳！)
        nn.init.xavier_uniform_(self.confidence_gate[-2].weight) # 权重保持随机，打破对称性
        nn.init.zeros_(self.confidence_gate[-2].bias) # Bias 设为 0

    def forward(self, x_current, x_sim, debug=False):
        (x_num, x_text) = x_current
        
        if debug: print("\n🔍 [Model Internals] Start Forward Pass...")

        # 1. RevIN
        x_num_norm = self.revin(x_num, mode='norm')

        # 🚨【修改】彻底移除 Masking
        # 既然我们用了 Zero-Init，就不需要 Masking 来强迫学习了。
        # 让模型看完整的数据，去寻找那微小的误差。
        x_input_for_net = x_num_norm # .clone() 也不需要了
        
        # 2. Backbone
        x_fused = torch.cat([x_input_for_net, x_text], dim=-1)
        x_emb = self.input_proj_current(x_fused)
        
        h_seq = x_emb
        for layer in self.glru_layers:
            h_seq = layer(h_seq)
        context = h_seq[:, -1, :] 
        
        # 3. RAG
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

        # 4. Prediction
        # 由于零初始化，一开始这里输出全是 0
        pred_residual_content = self.output_head(h_final)
        
        # 由于 Bias=-5，一开始这里全是 0.006 (几乎不信神经网络)
        gate_score = self.confidence_gate(h_final)
        
        if debug:
            print(f"   2. Neural Raw | Mean: {pred_residual_content.mean():.4f} (Should be ~0)")
            print(f"   ⚖️ [Gate Check] Mean Conf: {gate_score.mean():.4f} (Should be ~0)")

        # 5. Fusion
        baseline_norm = x_num_norm[:, -self.out_len:, self.target_idx]
        
        # 初始状态：Baseline + 0 * 0 = Baseline
        final_pred_norm = baseline_norm + (gate_score * pred_residual_content)

        # 6. Denorm
        if self.revin.affine:
            target_weight = self.revin.affine_weight[self.target_idx]
            target_bias = self.revin.affine_bias[self.target_idx]
            final_pred_norm = (final_pred_norm - target_bias) / (target_weight + 1e-10)

        B, L = final_pred_norm.shape
        target_mean = self.revin.mean[:, :, self.target_idx].view(B, 1)
        target_std = self.revin.stdev[:, :, self.target_idx].view(B, 1)
        
        final_pred = final_pred_norm * target_std + target_mean
        
        return final_pred.unsqueeze(-1)