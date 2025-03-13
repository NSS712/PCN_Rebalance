import torch
import torch.nn as nn

class PathEmbedding(nn.Module):
    def __init__(self, node_embed_dim=16, balance_dim=2, hidden_dim=32):
        super().__init__()
        # 节点ID嵌入层
        self.node_embed = nn.Embedding(num_embeddings=1000, 
                                     embedding_dim=node_embed_dim)
        # 余额特征处理 - 修正输入维度
        self.balance_fc = nn.Linear(balance_dim, hidden_dim)
        # 组合层
        self.combine_fc = nn.Linear(node_embed_dim*2 + hidden_dim, hidden_dim)
        
    def forward(self, path_nodes, path_balances):
        """
        输入: 
          path_nodes: [batch_size, paths, path_length] 节点ID序列
          path_balances: [batch_size, paths, path_length, 2] 双向余额
        输出:
          path_emb: [batch_size, paths, path_length-1, hidden_dim] 路径特征
        """
        # 获取维度信息
        batch_size, num_paths, path_length = path_nodes.shape
        
        # 节点嵌入
        node_emb = self.node_embed(path_nodes)  # [B, P, L, node_embed_dim]
        
        # 余额特征 - 需要重塑以适应Linear层
        # 将[B, P, L, 2]重塑为[B*P*L, 2]
        balance_flat = path_balances.view(-1, 2)
        balance_feat_flat = self.balance_fc(balance_flat)  # [B*P*L, hidden_dim]
        # 重塑回[B, P, L, hidden_dim]
        balance_feat = balance_feat_flat.view(batch_size, num_paths, path_length, -1)
        
        # 拼接特征
        combined = torch.cat([
            node_emb[:, :, :-1],  # 源节点 [B, P, L-1, node_embed_dim]
            node_emb[:, :, 1:],   # 目标节点 [B, P, L-1, node_embed_dim]
            balance_feat[:, :, :-1]  # 对应边的余额 [B, P, L-1, hidden_dim]
        ], dim=-1)
        
        # 重塑以适应Linear层
        combined_flat = combined.view(-1, node_emb.size(-1)*2 + balance_feat.size(-1))
        result_flat = self.combine_fc(combined_flat)
        # 重塑回原始维度
        return result_flat.view(batch_size, num_paths, path_length-1, -1)

class ChannelEmbedding(nn.Module):
    def __init__(self, node_embed_dim=16, balance_dim=2, hidden_dim=32):
        super().__init__()
        self.node_embed = nn.Embedding(1000, node_embed_dim)
        self.balance_fc = nn.Linear(balance_dim*2, hidden_dim)
        self.combine_fc = nn.Linear(node_embed_dim*2 + hidden_dim, hidden_dim)
        
    def forward(self, channel_nodes, channel_balances):
        """
        输入:
          channel_nodes: [batch_size, num_channels, 2] 通道两端节点ID
          channel_balances: [batch_size, num_channels, 2, 2] 双向余额矩阵
        输出:
          channel_emb: [batch_size, num_channels, hidden_dim]
        """
        # 获取维度信息
        batch_size, num_channels = channel_nodes.shape[:2]
        
        # 节点嵌入
        node_emb = self.node_embed(channel_nodes)  # [B, C, 2, node_embed_dim]
        
        # 余额处理 - 需要重塑以适应Linear层
        # 将[B, C, 2, 2]重塑为[B*C, 4]
        flat_balance = channel_balances.view(-1, 4)  # 展平双向余额
        balance_feat_flat = self.balance_fc(flat_balance)  # [B*C, hidden_dim]
        balance_feat = balance_feat_flat.view(batch_size, num_channels, -1)  # [B, C, hidden_dim]
        
        # 拼接特征
        combined = torch.cat([
            node_emb[:, :, 0],  # 源节点 [B, C, node_embed_dim]
            node_emb[:, :, 1],  # 目标节点 [B, C, node_embed_dim]
            balance_feat  # [B, C, hidden_dim]
        ], dim=-1)  # [B, C, node_embed_dim*2+hidden_dim]
        
        # 重塑以适应Linear层
        combined_flat = combined.view(-1, node_emb.size(-1)*2 + balance_feat.size(-1))
        result_flat = self.combine_fc(combined_flat)
        # 重塑回原始维度
        return result_flat.view(batch_size, num_channels, -1)

class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        # 路径更新的GRU
        self.path_gru = nn.GRUCell(input_size=hidden_dim, 
                                 hidden_size=hidden_dim)
        # 通道更新的GRU
        self.channel_gru = nn.GRUCell(input_size=hidden_dim,
                                    hidden_size=hidden_dim)
        
    def forward(self, path_states, channel_states, adj_matrix):
        """
        输入:
          path_states: [B, num_paths, hidden_dim]
          channel_states: [B, num_channels, hidden_dim]
          adj_matrix: [B, P, C] 路径-通道邻接矩阵
        输出:
          new_path_states, new_channel_states
        """
        batch_size = path_states.size(0)
        num_paths = path_states.size(1)
        num_channels = channel_states.size(1)
        hidden_dim = path_states.size(2)
        
        # 步骤1: 路径到通道的聚合
        # [B, C, P] * [B, P, H] -> [B, C, H]
        channel_msg = torch.bmm(
            adj_matrix.transpose(1, 2),
            path_states
        )
        
        # 重塑维度匹配GRU输入要求
        channel_msg = channel_msg.reshape(-1, hidden_dim)
        channel_states_flat = channel_states.reshape(-1, hidden_dim)
        
        # 更新通道状态
        new_channel = self.channel_gru(
            channel_states_flat,
            channel_msg
        ).reshape(batch_size, num_channels, hidden_dim)
        
        # 步骤2: 通道到路径的更新
        # [B, P, C] * [B, C, H] -> [B, P, H]
        path_msg = torch.bmm(adj_matrix, new_channel)
        
        # 重塑维度匹配GRU输入要求
        path_states_flat = path_states.reshape(-1, hidden_dim)
        path_msg_flat = path_msg.reshape(-1, hidden_dim)
        
        # 更新路径状态
        new_path = self.path_gru(
            path_states_flat,
            path_msg_flat
        ).reshape(batch_size, num_paths, hidden_dim)
        
        return new_path, new_channel

class Readout(nn.Module):
    def __init__(self, hidden_dim=32, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # x: [B, num_paths, hidden_dim]
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # 取最后时间步

class PCNBalanceNet(nn.Module):
    def __init__(self, 
               node_embed_dim=16,
               hidden_dim=32,
               msg_pass_layers=3):
        super().__init__()
        
        # 初始化嵌入
        self.path_embed = PathEmbedding(node_embed_dim=node_embed_dim, balance_dim=2, hidden_dim=hidden_dim)
        self.channel_embed = ChannelEmbedding(node_embed_dim=node_embed_dim, balance_dim=2, hidden_dim=hidden_dim)
        
        # 消息传递层
        self.msg_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) 
            for _ in range(msg_pass_layers)
        ])
        
        # Actor-Critic头
        self.actor_readout = Readout(hidden_dim, output_dim=2)  # 均值+方差
        self.critic_readout = Readout(hidden_dim, output_dim=1)
        
    def forward(self, batch_data):
        """
        输入数据格式:
        {
            'path_nodes': [B, P, L],
            'path_balances': [B, P, L, 2],
            'channel_nodes': [B, C, 2],
            'channel_balances': [B, C, 2, 2],
            'adj_matrix': [B, P, C]
        }
        """
        # 1. 初始化嵌入
        path_emb = self.path_embed(
            batch_data['path_nodes'], 
            batch_data['path_balances']
        )  # [B, P, L-1, D]
        
        channel_emb = self.channel_embed(
            batch_data['channel_nodes'],
            batch_data['channel_balances']
        )  # [B, C, D]
        
        # 2. 消息传递
        # 对path_emb在路径长度维度上取平均，保持三维结构 [B, P, D]
        path_states = path_emb.mean(dim=2)
        
        for layer in self.msg_layers:
            path_states, channel_emb = layer(
                path_states,
                channel_emb,
                batch_data['adj_matrix']
            )
        
        # 3. Readout
        mu_logvar = self.actor_readout(path_states)  # [B, P, 2]
        value = self.critic_readout(path_states)     # [B, 1]
        
        # 4. 分解输出
        mu = mu_logvar[..., 0]  # 均值 [B, P]
        logvar = mu_logvar[..., 1]  # 方差对数
        std = torch.exp(0.5*logvar)
        
        # 生成分布
        dist = torch.distributions.Normal(mu, std)
        return dist, value

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)


batch = {
    'path_nodes': torch.randint(0, 100, (32, 10, 5)),  # B=32, P=10, L=5
    'path_balances': torch.randn(32, 10, 5, 2),
    'channel_nodes': torch.randint(0, 100, (32, 20, 2)),
    'channel_balances': torch.randn(32, 20, 2, 2),
    'adj_matrix': (torch.rand(32, 10, 20) > 0.5).float()  # 0-1浮点矩阵
}

model = PCNBalanceNet()
model.apply(init_weights)
dist, value = model(batch)

# 采样动作
actions = dist.rsample()  # [32, 10]
print(actions)
