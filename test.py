import torch

# 创建一个需要计算梯度的张量
x = torch.tensor([1.0], requires_grad=True)
print(f"初始 x 的 id: {id(x)}")

# 原地操作
x_original = x.clone()
x += 1
print(f"原地操作后 x 的 id: {id(x)}")
print(f"原地操作后 x 是否与原 x 相同: {torch.allclose(x - 1, x_original)}")

# 创建一个新的需要计算梯度的张量
x = torch.tensor([1.0], requires_grad=True)
print(f"重新创建 x 的 id: {id(x)}")

# 非原地操作
x_original = x.clone()
x = x + 1
print(f"非原地操作后 x 的 id: {id(x)}")
print(f"非原地操作后 x 是否与原 x 相同: {torch.allclose(x - 1, x_original)}")