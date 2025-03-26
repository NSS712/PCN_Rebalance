import torch
import torch.nn as nn

# 定义参数
seq_len = 5  # 序列长度
batch_size = 3  # 批次大小
input_size = 10  # 输入特征数量
hidden_size = 20  # 隐藏状态特征数量
num_layers = 1  # GRU 层数
num_directions = 1  # 单向

# 创建 GRU 模型
gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=False)

# 生成随机输入数据
input_data = torch.randn(seq_len, batch_size, input_size)

# 初始化隐藏状态
h_0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)

# 前向传播
output, h_n = gru(input_data, h_0)

# 打印输入输出维度
print(f"输入数据维度: {input_data.shape}")
print(f"初始隐藏状态维度: {h_0.shape}")
print(f"输出数据维度: {output.shape}")
print(f"最终隐藏状态维度: {h_n.shape}")