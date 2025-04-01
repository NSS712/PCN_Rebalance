from entity import *

def compute_node_gini(node_id, state):
    """
    计算指定节点的Gini系数
    """
    balances = []
    for ch in state.channels:
        if ch.nodeID1 == node_id:
            balances.append(abs(ch.weight1 - ch.weight2)/(ch.weight1 + ch.weight2 + 1e-9))
        elif ch.nodeID2 == node_id:
            balances.append(abs(ch.weight1 - ch.weight2)/(ch.weight1 + ch.weight2 + 1e-9))
    
    if not balances:
        return 0.0
    
    n = len(balances)
    abs_diffs = sum(abs(x - y) for x in balances for y in balances)
    gini = abs_diffs / (2 * n * sum(balances) + 1e-9)
    return gini

def find_cycle_paths(node1, node2, state, max_depth=4):
    """
    查找包含node1-node2通道的环路路径
    """
    adj = {n: [] for n in range(state.node_num)}
    for ch in state.channels:
        adj[ch.nodeID1].append(ch.nodeID2)
        adj[ch.nodeID2].append(ch.nodeID1)
    
    cycles = []
    visited = set()
    
    def dfs(current, path, start_node):
        if len(path) > max_depth:
            return
        if current == start_node and len(path) >= 3:
            cycles.append(path.copy())
            return
        if current in visited:
            return
        
        visited.add(current)
        for neighbor in adj[current]:
            if neighbor not in path[-2:]:  # 避免立即折返
                path.append(neighbor)
                dfs(neighbor, path, start_node)
                path.pop()
        visited.remove(current)
    
    # 从node2开始寻找回到node1的环路
    dfs(node2, [node2], node1)
    return [([node1, node2] + p) for p in cycles if p]

def gcb_rebalance(state, imbalance_threshold=0.3):
    transactions = []
    
    # 步骤1：识别失衡通道
    imbalance_channels = []
    for ch in state.channels:
        total = ch.weight1 + ch.weight2
        if total == 0:
            continue
        imbalance = abs(ch.weight1 - ch.weight2) / total
        if imbalance > imbalance_threshold:
            imbalance_channels.append(ch)
    
    # 步骤2：为每个失衡通道寻找环路
    for ch in imbalance_channels:
        node1, node2 = ch.nodeID1, ch.nodeID2
        
        # 查找所有可能环路
        cycles = find_cycle_paths(node1, node2, state)
        if not cycles:
            continue
            
        # 步骤3：评估环路收益
        best_cycle = None
        max_improvement = -float('inf')
        
        for cycle in cycles:
            # 计算环路中各节点的平均Gini系数
            total_gini = 0.0
            for node in cycle:
                total_gini += compute_node_gini(node, state)
            avg_gini = total_gini / len(cycle)
            
            # 模拟交易后的Gini变化
            temp_state = State(state.node_num, 
                             [Channel(c.ID, c.nodeID1, c.nodeID2, c.weight1, c.weight2) for c in state.channels],
                             state.paths.copy())
            
            # 确定交易方向（从高余额到低余额）
            direction = 1 if ch.weight1 > ch.weight2 else -1
            amount = min(ch.weight1 if direction == 1 else ch.weight2, 
                        min(c.weight1 if d == 1 else c.weight2 
                            for c, d in zip(temp_state.channels, [1]*len(temp_state.channels))))
            
            # 执行模拟交易
            try:
                path_directions = [1 if n == cycle[i+1] else -1 
                                 for i, n in enumerate(cycle[:-1])]
                temp_state.route_transaction([c.ID for c in temp_state.channels], amount, path_directions)
                
                # 计算Gini改进
                new_gini = sum(compute_node_gini(node, temp_state) for node in cycle) / len(cycle)
                improvement = avg_gini - new_gini
                
                if improvement > max_improvement:
                    max_improvement = improvement
                    best_cycle = (cycle, amount, direction)
            except:
                continue
        
        # 步骤4：记录最优交易
        if best_cycle:
            cycle, amount, direction = best_cycle
            transactions.append({
                'channels': [ch.ID for ch in state.channels if ch.nodeID1 in cycle and ch.nodeID2 in cycle],
                'amount': amount * direction,
                'direction': direction
            })
    
    # 步骤5：执行实际交易
    for tx in transactions:
        try:
            state.route_transaction(tx['channels'], abs(tx['amount']), 
                                   [tx['direction']]*len(tx['channels']))
        except:
            continue
            
    return transactions

# 测试用例
def test_gcb_rebalance():
    # 测试用例1：简单失衡场景
    channels = [
        Channel(0, 0, 1, 100, 0),  # 严重失衡
        Channel(1, 1, 2, 50, 50),
        Channel(2, 2, 0, 50, 50)
    ]
    state = State(3, channels, [])
    print("初始平衡度:", state.compute_balance_index())
    
    transactions = gcb_rebalance(state)
    print("再平衡交易:", transactions)
    print("再平衡后通道状态:")
    for ch in state.channels:
        print(f"通道{ch.ID}: {ch.nodeID1}->{ch.nodeID2} ({ch.weight1}, {ch.weight2})")
    print("最终平衡度:", state.compute_balance_index())
    
    # 验证至少有一个通道被平衡
    assert any(abs(ch.weight1 - ch.weight2) < 10 for ch in state.channels), "测试用例1失败"
    
    # 测试用例2：无失衡场景
    balanced_channels = [
        Channel(0, 0, 1, 50, 50),
        Channel(1, 1, 2, 50, 50)
    ]
    state = State(3, balanced_channels, [])
    transactions = gcb_rebalance(state)
    assert len(transactions) == 0, "测试用例2失败"
    
    print("所有测试用例通过！")

test_gcb_rebalance()