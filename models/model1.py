from pylab import f
import torch
import torch.nn as nn



class GRUCell1(nn.Module):
    # 最初输入为path的初始化的特征，然后每轮输入的是channel特征，输出新的path特征
    def __init__(self, paths, feature_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size=feature_size, hidden_size=feature_size, num_layers=1, batch_first=True)
        self.paths = paths

    def forward(self, path_features, channel_features):
        for path_idx, path in enumerate(path_features):
            for channel_idx in self.paths[path_idx]['channel_idx']:
                path = self.gru(channel_features[channel_idx], path)
        return path_features

class GRUCell2(nn.Module):
    '''输入channel的旧的特征，用每个channel包含的path的特征来更新channel特征'''
    def __init__(self, channels, feature_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size=feature_size, hidden_size=feature_size, num_layers=1, batch_first=True)
        self.channels = channels
    
    def forward(self, path_features, channel_features):
        for channel_idx, channel in enumerate(channel_features):
            f = torch.zeros(channel_features[0].shape)
            for path_idx in self.channels[channel_idx]['path_idx']:
                f = f + path_features[path_idx]
            channel = self.gru(f, channel)
        return channel_features

class MessagePassingLayer(nn.Module):
    def __init__(self, paths, channels, input_size=64, feature_size=32):
        super().__init__()
        self.path_init_layer = nn.Linear(input_size, feature_size)
        self.channel_init_layer = nn.Linear(input_size, feature_size)
        self.Gru_Cell1 = GRUCell1(paths)
        self.Gru_Cell2 = GRUCell2(channels)
    
    def channel_init(self, channels, padding_len=64):
        # 把每个channel初始化为[其包含的pathid, 0, 0]的状态
        features = []
        for channel in channels:
            features.append([channel['source'], channel['destination'], channel['weight']/2, channel['weight']/2])
            features[-1].extend([0]*(padding_len-len(features[-1])))
        return features
    
    def path_init(self, paths, padding_len=64):
        # 把每个path初始化为[通道id + 双向余额 + 0]的状态
        features = []
        for path in paths:
            path_feature = []
            for channel in path:
                path_feature.append([channel['source'], channel['destination'], channel['weight']/2, channel['weight']/2])
            if len(path_feature) > padding_len:
                raise ValueError("路径长度超过padding_len")
            path_feature.extend([0]*(padding_len-len(path_feature)))
            features.append(path_feature)
        return features

    def forward(self, channels, paths, path_features, channel_features, adj_matrix, num_steps=5):
        ''' 输入网络结构，计算通道和路径特征 '''

        # 初始化网络的通道特征
        channel_features = self.channel_init(channels)
        path_features = self.path_init(paths)
        channel_features = nn.ReLU(self.channel_init_layer(channel_features))
        path_features = nn.ReLU(self.path_init_layer(path_features))

        for step in range(num_steps):
            # 计算通道的新特征
            channel_features = self.Gru_Cell1(path_features, channel_features)
            # 计算路径的新特征
            path_features = self.Gru_Cell2(path_features, channel_features)
        return path_features, channel_features

