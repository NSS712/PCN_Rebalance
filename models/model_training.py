from model_definition import EdgeAwareTransformer, CycleBalanceLoss
import torch
from torch.utils.data import DataLoader


# 示例输入（batch_size=2, 节点数=3）
node_feats = torch.tensor([
    # 节点特征：[出入度, 平均权重, 交易类型]
    [[2, 1.5, 0], [1, 0.8, 1], [3, -0.5, 0]],
    [[1, 0.7, 1], [2, 1.2, 0], [2, -0.3, 1]]
], dtype=torch.float32)

adj_matrix = torch.tensor([
    [[0, 1.2, 0], [0, 0, 0.8], [-0.5, 0, 0]],
    [[0, 0.7, 0], [0, 0, 1.2], [-0.3, 0, 0]]
], dtype=torch.float32)

# # 预测路径示例（节点索引序列）
# pred_paths = torch.tensor([
#     [0, 1, 2, 0],  # 循环路径A->B->C->A
#     [1, 2, 0, 1]   # 循环路径B->C->A->B
# ])

target_paths = torch.tensor([
    [0, 1, 2, 0],
    [1, 2, 0, 1]
])

# 模拟数据生成
def load_batch(batch_size=32):
    # 生成随机带权图及最优路径（此处需实现具体生成逻辑）
    return node_feats, adj_matrix, target_paths

model = EdgeAwareTransformer(node_dim=3)
criterion = CycleBalanceLoss(alpha=0.7)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

for epoch in range(100):
    nodes, adj, targets = load_batch()
    
    # 前向传播
    logits = model(nodes, adj)  # 输出形状[B,L,N]
    
    # 损失计算
    loss = criterion(logits, targets, adj)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # 监控指标
    if epoch % 10 == 0:
        with torch.no_grad():
            preds = logits.argmax(-1)
            accuracy = (preds == targets).float().mean()
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Acc: {accuracy:.2f}")