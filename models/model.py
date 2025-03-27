from os import path
import re
from typing import final
from networkx import find_asteroidal_triple
from pylab import f
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class GRUCell1(nn.Module):
    "最初输入为path的初始化的特征，然后每轮输入的是channel特征，输出新的path特征"
    def __init__(self, feature_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size=feature_size, hidden_size=feature_size, num_layers=1, batch_first=True)
        self.paths = []

    def forward(self, path_features, channel_features):
        '''
        path_features: (batch, num_paths, 32)
        channel_features: (batch, num_channels, 32)
        每轮输入的是channel特征,输出新的path特征
        '''
        channel_features = channel_features.transpose(1, 0)  #变成(num_channels, batch, 32)
        path_features = path_features.transpose(1, 0)
        final_path_feature = []
        for path_idx, path in enumerate(path_features):
            final_channel_feature = []
            for channel_idx in self.paths[path_idx].channels:
                final_channel_feature.append(channel_features[channel_idx])
            final_channel_feature = torch.stack(final_channel_feature, dim=0).transpose(0,1) #变成(batch, T, 32)
            path_feature = self.gru(final_channel_feature, path.unsqueeze(0))
            path_feature = path_feature[-1].squeeze(0)
            final_path_feature.append(path_feature)
        path_features = torch.stack(final_path_feature, dim=1) #变成(batch,num_path, 32)
        return path_features

class GRUCell2(nn.Module):
    '''输入channel的旧的特征，用每个channel包含的path的特征来更新channel特征'''
    def __init__(self, feature_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size=feature_size, hidden_size=feature_size, num_layers=1, batch_first=True)
        self.channels = []
    
    def forward(self, path_features, channel_features):

        channel_features = channel_features.transpose(1, 0)  #变成(num_channels, batch, 32)
        path_features = path_features.transpose(1, 0)
        final_channel_feature = []
        for channel_idx, channel in enumerate(channel_features):
            f = torch.zeros(channel_features[0].shape)
            for path_idx in self.channels[channel_idx].path:
                f = f + path_features[path_idx]
            channel = self.gru(f.unsqueeze(1), channel.unsqueeze(0))
            final_channel_feature.append(channel[-1].squeeze(0))
        return torch.stack(final_channel_feature, dim=1)

class Readout(nn.Module):
    def __init__(self, feature_size=32, k=5, hidden_dim1=8, hidden_dim2=128, out_dim=1):
        super().__init__()
        self.hidden_dim2 = hidden_dim2
        self.fc1 = nn.Linear(feature_size, hidden_dim1)
        self.lstm = nn.LSTM(input_size=hidden_dim1, hidden_size=hidden_dim2, num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim2, out_dim)
    
    def forward(self, x):
        # x: [B, num_paths, hidden_dim]
        batch_size = x.shape[0]
        h0 = torch.zeros(1, batch_size, self.hidden_dim2)
        c0 = torch.zeros(1, batch_size, self.hidden_dim2)
        x = F.relu(self.fc1(x))
        output, (hn, cn) = self.lstm(x, (h0, c0))
        return self.fc2(hn.squeeze(0))  # (batch, outdim)

class EmbeddingNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config["model_config"]['hidden_dim']
        self.feature_dim = config["model_config"]['hidden_dim']
        self.iterations = config["model_config"]['iterations']
        self.path_num = config['path_num']
        self.path_init_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.channel_init_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Gru_Cell1 = GRUCell1()
        self.Gru_Cell2 = GRUCell2()

    def forward(self, channel_features, path_features):
        '''
         输入通道和路径特征，计算均值和标准差 
         channel_features: (batch, num_channels, 32)
         path_features: (batch, num_paths, 32)
         输出: 均值和标准差
         mean: (batch, K)
         std: (batch, K)
         action: (batch, K) 采样得到的action
         '''
        channel_features = F.relu(channel_features)
        path_features = F.relu(path_features)

        for step in range(self.iterations):
            channel_features = self.Gru_Cell2(path_features, channel_features)
            path_features = self.Gru_Cell1(path_features, channel_features)


class PolicyNet(nn.Module):
    def __init__(self, config, feature_dim=32):
        super().__init__()

        self.hidden_dim = 32
        self.config = config
        self.iterations = config["model_config"]['iterations']
        self.path_num = config['path_num']
        # 均值网络: 输入k*32，输出K维均值
        self.mean_layer = Readout(k=self.path_num, feature_size=feature_dim, out_dim=self.path_num)
        # 对数标准差网络: 输入k*32，输出K维log(std)
        self.log_std_layer = Readout(k=self.path_num, feature_size=feature_dim, out_dim=self.path_num)
        self.path_init_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.channel_init_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Gru_Cell1 = GRUCell1()
        self.Gru_Cell2 = GRUCell2()


    def forward(self, channel_features, path_features):
        '''
         输入通道和路径特征，计算均值和标准差 
         channel_features: (batch, num_channels, 32)
         path_features: (batch, num_paths, 32)
         输出: 均值和标准差
         mean: (batch, K)
         std: (batch, K)
         action: (batch, K) 采样得到的action
         '''
        channel_features = F.relu(channel_features)
        path_features = F.relu(path_features)

        for step in range(self.iterations):
            channel_features = self.Gru_Cell2(path_features, channel_features)
            path_features = self.Gru_Cell1(path_features, channel_features)


        # 输入: x形状为 [k, 32]
        # 计算均值和对数标准差
        mean = self.mean_layer(path_features)  # [K]
        log_std = self.log_std_layer(path_features)  # [K]
        # 约束log_std的范围，防止数值不稳定
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)  # 转换为标准差
        
        # 重参数化采样（Reparameterization Trick）
        noise = torch.randn_like(mean)  # 从标准正态分布采样  [batch_size, p]
        action = mean + std * noise  # [batch_size, p]
        action = F.tanh(action)  
        return torch.stack([mean, std, action], dim=1)  # (batch, 3, p)

    def caculate(self, states):
        '''
        对state对象进行tensor化, 封装forward函数
        输入是 (batch, 1) 的 states
        输出是 (batch, 3, p), p是path的个数, 3是 mean, std, action
        '''
        self.states = states
        self.paths = states[0].paths
        self.Gru_Cell1.paths = self.paths
        batch_channel_features = []
        batch_path_features = []
        for state in states:
            self.state = state
            self.Gru_Cell2.channels = state.channels
            channel_features = self.channel_init(state.channels)
            path_features = self.path_init(state.paths)
            batch_channel_features.append(channel_features)
            batch_path_features.append(path_features)
        channel_features = torch.stack(batch_channel_features, dim=0)  # (batch, num_channels, 32)
        path_features = torch.stack(batch_path_features, dim=0)  # (batch, num_paths, 32)   
        return self.forward(channel_features, path_features)    # (batch, 3, p)
        
    def caculate_T_steps(self, states):
        '''
        计算T步policy
        输入states的list (batch, 1) 的 states
        输出 batch_size * (T+1) 个state
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
                tep_rewards.append(new_state.compute_reward())
                new_states.append(new_state)
            re_states.append(new_states)
            rewards.append(tep_rewards)
            states = new_states
        
        # policy 变成 （t, batch, 3, p）
        # rewards 变成 （t, batch）
        # re_states 变成 （t+1, batch）

        policy = torch.stack(policy, dim=1) # (batch, t, 3, p)
        rewards = torch.tensor(rewards, dtype=torch.float32).transpose(0,1) # (batch, t)
        return re_states, policy, rewards
    
    def channel_init(self, channels, padding_len=32):
        # 把每个channel初始化为[其包含的pathid, 0, 0]的状态
        features = []
        for channel in channels:
            features.append([channel.nodeID1, channel.nodeID2, channel.weight1, channel.weight2])
            features[-1].extend([0]*(padding_len-len(features[-1])))
        return torch.tensor(features, dtype=torch.float32)
    
    def path_init(self, paths, padding_len=32):
        # 把每个path初始化为[通道id + 双向余额 + 0]的状态
        features = []
        for path in paths:
            path_feature = []
            for idx, channelID in enumerate(path.channels):
                channel = self.state.channels[channelID]
                d = path.channel_derection[idx]
                if d == 1:
                    path_feature.extend([channel.nodeID1, channel.nodeID2, channel.weight1, channel.weight2])
                else:
                    path_feature.extend([channel.nodeID2, channel.nodeID1, channel.weight2, channel.weight1])
            if len(path_feature) > padding_len:
                raise ValueError("路径长度超过padding_len")
            path_feature.extend([0]*(padding_len-len(path_feature)))
            features.append(path_feature)
        return torch.tensor(features, dtype=torch.float32)

    def value(self):

class ValueNet(nn.Module):
    def __init__(self, config, k=5, feature_dim=32):
        super().__init__()
        self.config = config
        self.value = Readout(k=k, feature_size=feature_dim, out_dim=1)

    def forward(self, x):
        # 输入: x形状为 [k, 32]
        value = self.value(x)
        return value