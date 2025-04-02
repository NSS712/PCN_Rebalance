from networkx import number_attracting_components
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# class Transformer_with_Attenion(nn.TransformerEncoder):
#     def __init__(self, config):
#         head_num = config["balance_model_config"]["head_num"]
#         layer_num = config["balance_model_config"]["layer_num"]
#         d_model = config["balance_model_config"]["feature_dim"]
#         super(Transformer_with_Attenion, self).__init__(nn.TransformerEncoderLayer(d_model=d_model, nhead=head_num),
#             num_layers=layer_num)
    

class Transformer_PolicyNet(nn.Module):
    def __init__(self, config):
        super(Transformer_PolicyNet, self).__init__()
        self.config = config
        d_model = config["balance_model_config"]["feature_dim"]
        self.output_dim = config["path_num"]
        head_num = config["balance_model_config"]["head_num"]
        layer_num = config["balance_model_config"]["layer_num"]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=head_num),
            num_layers=layer_num
        )
        self.fc1 = nn.Linear(d_model, 2)

        # 特征归一化层
        self.path_norm = nn.LayerNorm(d_model)   # 路径特征归一化
        self.channel_norm = nn.LayerNorm(d_model) # 通道特征归一化
        self.num_path = config["path_num"]

    def forward(self, x, src_mask=None):
        # x: [batch_size, seq_length, d_model]
        output = self.transformer(x.permute(1, 0, 2),mask=src_mask)  # [seq_length, batch_size, d_model]
        output = output.permute(1, 0, 2)  # [batch_size, seq_length, d_model]
        output = output[:, :self.num_path, :]  # 只取前num_path个路径的特征，即p个路径的特征
        output = self.fc1(output).transpose(1,2)  # [batch_size,2 , num_path]
        mean, std = torch.chunk(output, 2, dim=1)
        std = torch.exp(std)  # 转换为标准差
        noise = torch.randn_like(mean)  # 从标准正态分布采样  [batch_size, p]
        action = mean + std * noise  # [batch_size, p]
        action = F.tanh(action)
        ans = torch.cat([mean, std, action], dim=1) # (batch, 3, p)
        return ans

    def caculate(self, states):
        '''
        对state对象进行tensor化, 封装forward函数
        输入是 (batch, 1) 的 states
        输出是 (batch, 3, p), p是path的个数, 3是 mean, std, action
        '''
        batch_channel_features = []
        batch_path_features = []
        for state in states:
            channel_features, path_features = state.to_tensor(self.config['model_config']['feature_dim'])
            batch_channel_features.append(channel_features)
            batch_path_features.append(path_features)
        channel_features = torch.stack(batch_channel_features, dim=0)  # (batch, num_channels, 32)
        path_features = torch.stack(batch_path_features, dim=0)  # (batch, num_paths, 32)   

        # 对两种特征分别进行层归一化
        channel_features = self.channel_norm(channel_features)  # 归一化通道特征
        path_features = self.path_norm(path_features)            # 归一化路径特征

        return self.forward(torch.cat([path_features, channel_features], dim = 1), src_mask=states[-1].attention_mask)    # (batch, 3, p)

    def caculate_next_state(self, state):
        re_states, policy, rewards = self.caculate_T_steps([state])
        return re_states[0][-1]

    def caculate_T_steps(self, states):
        '''
        计算T步policy
        输入states的list (batch, 1) 的 states
        输出 batch_size , (T+1) 个state
        以及 (batch, T, 3, p), p是path的个数, 3是 mean, std, action
        以及 batch_size * T 个reward
        '''
        rewards = []
        policy = []
        re_states = [states]
        for t in range(self.config['trigger_threshold']):
            new_states = []
            tep_rewards = []
            policy_outputs = self.caculate(states)  # (batch, 3, p)
            policy.append(policy_outputs)
            for idx, state in enumerate(states):
                new_state = copy.copy(state).act(policy_outputs[idx][2])
                tep_rewards.append(state.compute_reward(new_state))
                new_states.append(new_state)
            re_states.append(new_states)
            rewards.append(tep_rewards)
            states = new_states
        
        re_states = list(map(list, zip(*re_states))) # (batch, t+1)
        policy = torch.stack(policy, dim=1) # (batch, t, 3, p)
        rewards = torch.tensor(rewards, dtype=torch.float32, device='cuda').transpose(0,1) # (batch, t)
        return re_states, policy, rewards

class Transformer_ValueNet(nn.Module):
    def __init__(self, config, transformer_model):
        super(Transformer_ValueNet, self).__init__()
        self.config = config
        d_model = config["balance_model_config"]["feature_dim"]
        self.transformer = transformer_model
        self.fc = nn.Linear(d_model, 1)

        # 特征归一化层
        self.path_norm = nn.LayerNorm(d_model)   # 路径特征归一化
        self.channel_norm = nn.LayerNorm(d_model) # 通道特征归一化

    def forward(self, x, src_mask=None):
        # x: [batch_size, seq_length, d_model]
        output = self.transformer(x.permute(1, 0, 2), mask=src_mask)  # [seq_length, batch_size, d_model]
        output = output.permute(1, 0, 2)  # [batch_size, seq_length, d_model]
        pooled_output = torch.mean(output, dim=1)  # 平均池化
        output = self.fc(pooled_output)  # [batch_size, output_dim]
        return output

    def caculate(self, states):
        '''
        对state对象进行tensor化, 封装forward函数
        输入是 (batch, 1) 的 states
        输出是 (batch, 1), p是path的个数, 3是 mean, std, action
        '''
        batch_channel_features = []
        batch_path_features = []
        for state in states:
            channel_features, path_features = state.to_tensor(self.config['model_config']['feature_dim'])
            batch_channel_features.append(channel_features)
            batch_path_features.append(path_features)
        channel_features = torch.stack(batch_channel_features, dim=0)  # (batch, num_channels, 32)
        path_features = torch.stack(batch_path_features, dim=0)  # (batch, num_paths, 32)   

        # 对两种特征分别进行层归一化
        channel_features = self.channel_norm(channel_features)  # 归一化通道特征
        path_features = self.path_norm(path_features)            # 归一化路径特征

        return self.forward(torch.cat([path_features, channel_features], dim = 1), src_mask=states[-1].attention_mask)    # (batch, 3, p)
