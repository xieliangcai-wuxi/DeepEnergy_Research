import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicGraphInteraction(nn.Module):
    """
    [科研核心组件] 动态图交互层 (Dynamic Graph Interaction)
    
    Update Log:
    - [Fix]: 增加了自动维度对齐机制。现在支持任意 d_model 输入，
             不再强制要求 d_model 必须被 num_nodes 整除。
    """
    def __init__(self, input_dim: int, num_nodes: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        
        # [关键修复] 维度对齐逻辑
        # 1. 计算每个节点至少需要多少维度
        # 我们希望投影后的维度 >= input_dim，且能被 num_nodes 整除
        if input_dim % num_nodes == 0:
            self.aligned_dim = input_dim
            self.requires_alignment = False
        else:
            # 向上取整找到最近的倍数
            self.aligned_dim = math.ceil(input_dim / num_nodes) * num_nodes
            self.requires_alignment = True
            
        # 2. 定义对齐层 (如果需要)
        if self.requires_alignment:
            self.align_proj = nn.Linear(input_dim, self.aligned_dim)
        
        # 3. 计算节点特征维度
        self.node_dim = self.aligned_dim // num_nodes
        
        # 4. 节点特征投影
        self.input_proj = nn.Linear(self.node_dim, d_model)
        
        # 5. 动态邻接矩阵参数
        self.node_embedding_dim = 10
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, self.node_embedding_dim))
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, self.node_embedding_dim))
        
        # 6. 图卷积变换
        self.gcn_proj = nn.Linear(d_model, d_model)
        
        # 7. 正则化
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Seq_Len, Total_Features] (e.g., [B, 96, 64])
        """
        batch_size, seq_len, _ = x.shape
        
        # [Step 0] 维度自动对齐
        if self.requires_alignment:
            # Linear Projection: [B, S, 64] -> [B, S, 65] (举例)
            x_aligned = self.align_proj(x)
        else:
            x_aligned = x
            
        # [Step 1] 重塑 (Reshape)
        # [B, S, Aligned_Dim] -> [B, S, Nodes, Node_Dim]
        x_reshaped = x_aligned.view(batch_size, seq_len, self.num_nodes, self.node_dim)
        
        # [Step 2] 投影到 d_model
        # [B, S, N, Node_Dim] -> [B, S, N, d_model]
        x_emb = self.input_proj(x_reshaped)
        
        # [Step 3] 动态图学习
        node_sim = torch.mm(self.node_emb1, self.node_emb2.transpose(0, 1))
        adj = F.softmax(F.relu(node_sim), dim=1)
        
        # [Step 4] 图卷积聚合
        x_emb_flat = x_emb.view(batch_size * seq_len, self.num_nodes, -1)
        x_agg = torch.matmul(adj.unsqueeze(0), x_emb_flat)
        
        # 变换
        x_agg = self.gcn_proj(x_agg)
        x_agg = self.act(x_agg)
        
        # [Step 5] 节点池化 (Pooling) -> 回到全局向量
        # [B*S, N, D] -> [B*S, D]
        x_pooled = x_agg.mean(dim=1)
        out = x_pooled.view(batch_size, seq_len, -1)
        
        # [Step 6] 残差连接
        # 注意: 残差连接要求 x 和 out 维度一致。
        # 如果输入维度是 64，输出也是 d_model(64)，可以直接加。
        # 哪怕中间对齐到了 65，只要输出投影回 64 即可。
        # 上面的 gcn_proj 输出就是 d_model，所以可以直接加。
        
        if x.shape[-1] == out.shape[-1]:
             out = self.norm(x + self.dropout(out))
        else:
             out = self.norm(self.dropout(out))
             
        return out