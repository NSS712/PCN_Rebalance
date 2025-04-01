from scipy.optimize import linprog
from ultils.entity import *

def balance_channels(state):
    channels = state.channels
    nodes = state.node_num
    C = len(channels)
    if C == 0:
        return []
    
    # 构建节点约束矩阵 ---------------------------------------------------------
    A_eq = np.zeros((nodes, C))  # 每个节点对应一个等式约束
    for node in range(nodes):
        for i, ch in enumerate(channels):
            if ch.nodeID1 == node:
                A_eq[node][i] += 1   # 流出通道
            if ch.nodeID2 == node:
                A_eq[node][i] -= 1   # 流入通道
    
    # 构造线性规划参数 ---------------------------------------------------------
    c = [0]*C + [1]*C  # 目标函数：sum(t_i)
    
    # 不等式约束：t_i >= |d_i - 2x_i|
    A_ub = []
    b_ub = []
    for i in range(C):
        d_i = channels[i].weight1 - channels[i].weight2
        # t_i >= d_i - 2x_i --> -2x_i - t_i <= -d_i
        row1 = [0.0]*(2*C)
        row1[i] = -2
        row1[C+i] = -1
        A_ub.append(row1)
        b_ub.append(-d_i)
        
        # t_i >= -(d_i - 2x_i) --> 2x_i - t_i <= d_i 
        row2 = [0.0]*(2*C)
        row2[i] = 2
        row2[C+i] = -1
        A_ub.append(row2)
        b_ub.append(d_i)
    
    # 变量边界约束 ------------------------------------------------------------
    bounds = []
    for ch in channels:
        bounds.append( (-ch.weight2, ch.weight1) )  # x_i的边界
    for _ in range(C):
        bounds.append( (0, None) )  # t_i >=0
    
    # 求解线性规划 ------------------------------------------------------------
    result = linprog(
        c, 
        A_ub=A_ub, b_ub=b_ub,
        A_eq=np.hstack([A_eq, np.zeros((nodes, C))]),  # 等式约束仅作用在x_i
        b_eq=np.zeros(nodes),
        bounds=bounds,
        method='highs'
    )
    
    if not result.success:
        print("线性规划求解失败！")
        return state
    
    x = result.x[:C]  # 提取前C个变量作为通道调整量
    
    new_channels = []
    for i, ch in enumerate(channels):
        new_ch = Channel(ch.ID, ch.nodeID1, ch.nodeID2, ch.weight1 - x[i], ch.weight2 + x[i])
        new_channels.append(new_ch)
    
    return State(state.node_num, new_channels, state.paths.copy())


if __name__=="__main__":

    """
    # 测试用例：5节点双环结构
    节点拓扑：
    0 ↔ 1 ↔ 2
    ↑↙↖↘  ↓ 
    4 ← 3 ← 2
    通道结构：
    0: 0→1 (5,1)
    1: 1→2 (5,1)
    2: 2→3 (5,1) 
    3: 3→0 (5,1)
    4: 0→4 (5,1)
    5: 4→2 (5,1)
    6: 4→3 (5,1)
    """
    channels = [
        Channel(0, 0,1,5,1),  # channel0: 0→1
        Channel(1,1,2,5,1),    # channel1: 1→2
        Channel(2,2,3,5,1),    # channel2: 2→3
        Channel(3,3,0,5,1),    # channel3: 3→0
        Channel(4,0,4,5,1),    # channel4: 0→4 
        Channel(5,4,2,5,1),    # channel5: 4→2
        Channel(6,4,3,5,1)     # channel6: 4→3
    ]
    state = State(5, channels, paths=[])

    print("=== 初始状态 ===")
    print(f"各通道余额差: {[ch.weight1 - ch.weight2 for ch in channels]}")
    print(f"初始平衡度: {state.compute_balance_index():.3f}")

    # 执行平衡
    new_state = balance_channels(state)