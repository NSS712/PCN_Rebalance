import torch
import torch.nn as nn
import torch.nn.functional as F


# class NodeEmbedding(nn.Module):
#     def __init__(self, d_model, node_feat_dim):
#         super().__init__()
#         self.proj = nn.Linear(node_feat_dim, d_model)  # 节点特征投影
#         self.edge_attn = EdgeAwareAttention()  # 自定义边权重注意力
        
#     def forward(self, nodes, adj_matrix):
#         h = self.proj(nodes)
#         # 融入边权重
#         h = self.edge_attn(h, adj_matrix)  
#         return h


# class WeightGNN(nn.Module):
#   def forward(self, x, adj):
#       # 加权消息传递
#       return torch.matmul(adj, x)
  

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class EdgeAwareTransformer(nn.Module):
    def __init__(self, node_dim, nhead=4, dim_feedforward=256, num_layers=3):
        super().__init__()
        # 节点嵌入层
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # 边权重编码器
        self.edge_proj = nn.Linear(1, 128)
        
        # Transformer配置
        encoder_layers = TransformerEncoderLayer(
            d_model=256, nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # 启用batch_first模式
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        
        # 路径解码器
        self.path_decoder = nn.LSTM(256, 128, num_layers=2, batch_first=True)
        self.pointer_net = nn.Linear(128, 1)  # 节点选择器

    def forward(self, node_feats, adj_matrix):
        """
        node_feats: [batch, N, node_dim]
        adj_matrix: [batch, N, N] 带权邻接矩阵
        """
        batch_size, N = node_feats.shape[:2]
        
        # 节点嵌入
        node_emb = self.node_embed(node_feats)  # [B,N,128]
        
        # 边特征融合
        edge_emb = self.edge_proj(adj_matrix.unsqueeze(-1)).squeeze(-2)  # [B,N,N,128]
        edge_emb = edge_permute(edge_emb)  # 调整维度适配注意力
        
        # Transformer输入构造
        combined_emb = torch.cat([node_emb, edge_emb.mean(dim=2)], dim=-1)  # [B,N,256]
        
        # Transformer处理
        memory = self.transformer(combined_emb)  # [B,N,256]
        
        # 路径解码（使用LSTM解码器）
        start_nodes = select_start_node(memory)  # 自定义起始点选择
        path_logits = self.decode_path(memory, start_nodes)
        
        return path_logits

    def decode_path(self, memory, start_nodes):
        batch_size = memory.size(0)
        path_logits = []
        
        # 初始化隐藏状态
        current_nodes = start_nodes
        visited = torch.zeros(batch_size, memory.size(1), dtype=torch.bool)
        
        for step in range(20):  # 最大步长控制
            # 获取当前节点嵌入
            curr_emb = memory[torch.arange(batch_size), current_nodes]
            
            # 计算候选节点概 率
            logits = torch.matmul(memory, curr_emb.unsqueeze(-1)).squeeze(-1)  # [B,N]
            logits[visited] = -float('inf')  # 屏蔽已访问节点
            
            # 记录预测结果
            path_logits.append(logits)
            
            # 选择下一节点
            next_nodes = logits.argmax(dim=-1)
            visited[torch.arange(batch_size), next_nodes] = True
            
            # 更新当前节点
            current_nodes = next_nodes
            
        return torch.stack(path_logits, dim=1)  # [B,L,N]

class CycleBalanceLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, path_logits, target_paths, adj_matrix):
        """
        path_logits: [B, L, N] 各位置节点预测概率
        target_paths: [B, L] 实际路径节点索引
        """
        # 平衡性损失计算
        path_weights = gather_weights(path_logits.argmax(-1), adj_matrix)
        balance_loss = torch.abs(path_weights.sum(dim=1)).mean()
        
        # 路径有效性损失
        seq_loss = F.cross_entropy(
            path_logits.permute(0,2,1),  # 调整为[B,N,L]
            target_paths.long()  # 确保为Long类型
        )
        
        return self.alpha*balance_loss + (1-self.alpha)*seq_loss

def gather_weights(path_indices, adj_matrix):
    # 沿路径收集权重
    batch_size, seq_len = path_indices.shape
    weights = torch.zeros(batch_size, seq_len-1)
    for b in range(batch_size):
        for t in range(seq_len-1):
            src = path_indices[b, t]
            dst = path_indices[b, t+1]
            weights[b, t] = adj_matrix[b, src, dst]
    return weights

def edge_permute(x):
    # 维度调整适配注意力计算
    return x.permute(0, 2, 1, 3)

def select_start_node(memory):
    # 选择出入度差异最大的节点作为起点
    degree_diff = memory[:, :, 0] - memory[:, :, 1]  # 假设0/1维度存储出入度
    return degree_diff.argmax(dim=-1)

