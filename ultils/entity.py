import random
from re import T
import numpy as np

class Channel:
    def __init__(self, ID, nodeID1, nodeID2, weight1, weight2):
        self.ID = ID
        self.nodeID1 = nodeID1
        self.nodeID2 = nodeID2
        self.weight1 = weight1
        self.weight2 = weight2
        self.path = []

class Path:
    def __init__(self, ID):
        self.ID = ID
        self.channels = []
        self.channel_derection = []
        self.nodes = []

    def min_capacity(self, direction, channels):
        '''
        根据direction, 找到path中最大交易额度。
        direction: 1表示交易方向为source->destination, 
                   0表示交易方向为destination->source
        '''
        ans = float('inf')
        for idx, channel_id in enumerate(self.channels):
            channel = channels[channel_id]
            d = self.channel_derection[idx]
            if direction * d == 1:
                ans = min(ans, channel.weight1)
            else:
                ans = min(ans, channel.weight2)
        return ans

    def transaction(self, amount, channels):
        '''
        根据amounts, 对每个channel执行交易。
        '''
        for idx, channel_id in enumerate(self.channels):
            channel = channels[channel_id]
            d = self.channel_derection[idx]
            if d == 1:
                channel.weight1 -= amount
                channel.weight2 += amount
            else:
                channel.weight1 += amount
                channel.weight2 -= amount
            if channel.weight1 < 0 or channel.weight2 < 0:
                raise ValueError("余额不足，交易失败")

class State:
    def __init__(self, node_num, channels, paths):
        self.node_num = node_num
        self.channels = channels
        self.paths = paths

        # 初始化每个节点的权重，用于指数分布来采样模拟交易
        weights = np.random.exponential(scale=1, size=self.node_num)
        normalized_weights = weights / np.sum(weights)  # 归一化权重
        self.weights = normalized_weights

    def compute_reward(self):
        """
        计算当前状态的平衡度, 结果在[-1, 1]之间
        """
        ans = 0
        for channel in self.channels:
            ans += (channel.weight1 - channel.weight2) / (channel.weight1 + channel.weight2)
        return ans/len(self.channels)

    def random_transaction(self, amount):
        '''
        随机选择两个节点，生成一笔交易。
        '''
        selected_nodes = np.random.choice(list(range(self.node_num)), size=2, replace=False, p=self.weights)
        source_node = selected_nodes[0]
        destination_node = selected_nodes[1]

        transaction_channels, directions = self.routing(source_node, destination_node)
        if not transaction_channels:
            return False  # 没有路径可以交易

        return self.route_transaction(transaction_channels, amount, directions)
    
    def routing(self, node1, node2):
        '''
        根据node1和node2查找最短路径，返回包含通道ID列表和方向列表的元组
        方向1表示node1->node2方向，-1表示反向
        '''
        if node1 == node2:
            return [], []
        
        # 构建邻接表 {node: [(neighbor, channel_id, direction)]}
        adj = {}
        for channel in self.channels:
            adj.setdefault(channel.nodeID1, []).append( (channel.nodeID2, channel.ID, 1) )
            adj.setdefault(channel.nodeID2, []).append( (channel.nodeID1, channel.ID, -1) )

        visited = set()
        queue = [(node1, [])]  # (current_node, path_history)
        
        while queue:
            current_node, path = queue.pop(0)
            if current_node in visited:
                continue
                
            visited.add(current_node)

            if current_node == node2:
                channel_ids = [p[0] for p in path]
                directions = [p[1] for p in path]
                return channel_ids, directions

            for neighbor, ch_id, direction in adj.get(current_node, []):
                if neighbor not in visited:
                    new_path = path + [(ch_id, direction)]
                    queue.append( (neighbor, new_path) )
        
        return [], []  # 没有找到路径

    def route_transaction(self, transaction_channels, amount, directions):
        '''
        根据amounts, 对每个channel执行交易。
        '''
        # 先检查每个通道的余额是否足够
        for channel_id, direction in zip(transaction_channels, directions):
            channel = self.channels[channel_id]
            if direction == 1:
                if channel.weight1 < amount:
                    return False
            else:
                if channel.weight2 < amount:
                    return False

        # 执行交易
        for channel_id, direction in zip(transaction_channels, directions):
            channel = self.channels[channel_id]
            if direction == 1:
                channel.weight1 -= amount
            else:
                channel.weight2 -= amount
        return True

    def act(self, amount_ratios):
        '''
        根据action, 更新图的状态。
        先根据交易的方向, 找到path中最大交易额度, 用 交易额度 * amount 来更新每个channel的balance。
        预期的输入是一个 长度为k的向量, 表示每个path的交易额度。
        '''
        for i, path in enumerate(self.paths):
            amount_ratio = amount_ratios[i].item()
            assert -1 <= amount_ratio <= 1, "amount_ratio 不在[-1, 1]之间" 
            amount = int(path.min_capacity(1 if amount_ratio >= 0 else -1, self.channels) * amount_ratio)
            path.transaction(amount, self.channels)
