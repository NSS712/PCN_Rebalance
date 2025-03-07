import torch
import torch.nn as nn

class DRLPCR_GNN(nn.Module):
    def __init__(self, feat_dim=32, num_iterations=5):
        super(DRLPCR_GNN, self).__init__()
        self.feat_dim = feat_dim
        self.num_iterations = num_iterations
        
        # 定义两个不同的GRU单元
        self.gru_cell1 = nn.GRUCell(input_size=feat_dim, 
                                  hidden_size=feat_dim)  # 处理路径特征
        self.gru_cell2 = nn.GRUCell(input_size=feat_dim, 
                                  hidden_size=feat_dim)  # 处理通道特征

    def forward(self, paths, channels, path_to_channel, channel_to_path):
        """
        paths: 路径特征列表 [num_paths, feat_dim]
        channels: 通道特征列表 [num_channels, feat_dim]
        path_to_channel: 字典 {path_idx: [channel_indices]}
        channel_to_path: 字典 {channel_idx: [path_indices]}
        """
        # 初始化隐藏状态
        h_paths = [path.clone() for path in paths]
        h_channels = [channel.clone() for channel in channels]

        # 进行多轮消息传递
        for _ in range(self.num_iterations):
            # 第一轮：更新路径特征
            new_h_paths = []
            for path_idx, channel_indices in path_to_channel.items():
                h_path = h_paths[path_idx]
                # 按路径中的通道顺序处理
                for chan_idx in channel_indices:
                    # GRU输入：当前通道的特征
                    gru_input = h_channels[chan_idx]
                    # 更新路径特征
                    h_path = self.gru_cell1(gru_input, h_path)
                new_h_paths.append(h_path)
            h_paths = new_h_paths

            # 第二轮：更新通道特征
            new_h_channels = []
            for chan_idx, path_indices in channel_to_path.items():
                # 聚合相关路径特征
                aggregated = sum([h_paths[p_idx] for p_idx in path_indices])
                # 更新通道特征
                h_chan = self.gru_cell2(aggregated, h_channels[chan_idx])
                new_h_channels.append(h_chan)
            h_channels = new_h_channels

        return h_paths, h_channels