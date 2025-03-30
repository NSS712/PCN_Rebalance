import random
from re import T
import numpy as np
import torch
from torch.nn.modules import padding
import heapq

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

        transaction_channels, directions = self.routing(source_node, destination_node, amount)
        if not transaction_channels:
            return False  # 没有路径可以交易

        return self.route_transaction(transaction_channels, amount, directions)

    # def routing(self, node1, node2, amount):
    #     if node1 == node2:
    #         return [], []
        
    #     adj = {node: [] for node in range(self.node_num)}
    #     for channel in self.channels:
    #         adj[channel.nodeID1].append( (channel.nodeID2, channel.ID, 1, channel.weight1) )
    #         adj[channel.nodeID2].append( (channel.nodeID1, channel.ID, -1, channel.weight2) )

    #     visited = set([node1])
    #     current_node = node1
    #     path_channels = []
    #     path_directions = []

    #     while current_node != node2:
    #         candidates = []
    #         for neighbor, ch_id, direction, balance in adj.get(current_node, []):
    #             if balance >= amount and neighbor not in visited:
    #                 candidates.append( (-balance, ch_id, direction, neighbor) )

    #         if not candidates:
    #             return [], []

    #         candidates.sort()
    #         _, ch_id, direction, next_node = candidates[0]
    #         path_channels.append(ch_id)
    #         path_directions.append(direction)
    #         visited.add(next_node)
    #         current_node = next_node

    #     return path_channels, path_directions

    def routing(self, node1, node2, amount):
        """根据余额的最大额度来路由，每一跳选择通道余额最大的
        如果当前节点的最大通道余额小于目标交易金额，则直接终止返回false
        
        Args:
            node1: 源节点
            node2: 目标节点
            amount: 交易金额
            
        Returns:
            tuple: (通道ID列表, 方向列表) 如果没有可行路径则返回 ([], [])
        """
        if node1 == node2:
            return [], []
        
        # 构建邻接表，包含节点间的通道信息
        adj = {node: [] for node in range(self.node_num)}
        for channel in self.channels:
            # 添加通道信息：邻居节点、通道ID、方向、当前方向的余额
            adj[channel.nodeID1].append((channel.nodeID2, channel.ID, 1, channel.weight1))
            adj[channel.nodeID2].append((channel.nodeID1, channel.ID, -1, channel.weight2))
        
        # 使用优先队列进行贪心搜索，优先选择余额最大的通道
        heap = []
        visited = {node1}  # 已访问节点集合
        
        # 初始化队列：检查从起点出发的所有通道
        for neighbor, ch_id, direction, balance in adj.get(node1, []):
            # 只考虑余额大于等于交易金额的通道
            if balance >= amount:
                # 使用负余额作为优先级，这样堆顶是余额最大的通道
                heapq.heappush(
                    heap, 
                    (-balance, neighbor, [ch_id], [direction], visited.union({neighbor}))
                )
        
        # 如果没有可行的起始通道，直接返回失败
        if not heap:
            return [], []
        
        while heap:
            neg_balance, current_node, path_channels, path_directions, visited = heapq.heappop(heap)
            current_balance = -neg_balance  # 当前通道的余额
            
            # 如果到达目标节点，返回路径
            if current_node == node2:
                return path_channels, path_directions
            
            # 检查当前节点的所有邻居通道
            neighbors = []
            for neighbor, ch_id, direction, neighbor_balance in adj.get(current_node, []):
                # 只考虑未访问过且余额足够的通道
                if neighbor_balance >= amount and neighbor not in visited:
                    neighbors.append((-neighbor_balance, neighbor, ch_id, direction))
            
            # 如果没有可行的下一跳通道，继续检查其他路径
            if not neighbors:
                continue
                
            # 选择余额最大的通道作为下一跳
            neighbors.sort()  # 按余额降序排序
            neg_next_balance, next_node, next_ch_id, next_direction = neighbors[0]
            
            # 更新路径信息
            new_channels = path_channels + [next_ch_id]
            new_directions = path_directions + [next_direction]
            new_visited = visited.union({next_node})
            
            # 将新路径加入优先队列
            heapq.heappush(
                heap,
                (neg_next_balance, next_node, new_channels, new_directions, new_visited)
            )
        
        return [], []  # 没有找到可行路径

    # def routing(self, node1, node2):
    #     '''
    #     根据node1和node2查找最短路径，返回包含通道ID列表和方向列表的元组
    #     方向1表示node1->node2方向，-1表示反向
    #     '''
    #     if node1 == node2:
    #         return [], []
        
    #     # 构建邻接表 {node: [(neighbor, channel_id, direction)]}
    #     adj = {}
    #     for channel in self.channels:
    #         adj.setdefault(channel.nodeID1, []).append( (channel.nodeID2, channel.ID, 1) )
    #         adj.setdefault(channel.nodeID2, []).append( (channel.nodeID1, channel.ID, -1) )

    #     visited = set()
    #     queue = [(node1, [])]  # (current_node, path_history)
        
    #     while queue:
    #         current_node, path = queue.pop(0)
    #         if current_node in visited:
    #             continue
                
    #         visited.add(current_node)

    #         if current_node == node2:
    #             channel_ids = [p[0] for p in path]
    #             directions = [p[1] for p in path]
    #             return channel_ids, directions

    #         for neighbor, ch_id, direction in adj.get(current_node, []):
    #             if neighbor not in visited:
    #                 new_path = path + [(ch_id, direction)]
    #                 queue.append( (neighbor, new_path) )
        
    #     return [], []  # 没有找到路径

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
                channel.weight2 += amount
            else:
                channel.weight2 -= amount
                channel.weight1 += amount
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
        return self

    def to_tensor(self, feature_dim):
        '''
        将state转换为tensor, 用于神经网络的输入。
        '''

        # 把每个channel初始化为[其包含的pathid, 0, 0]的状态
        padding_len = feature_dim
        channel_features = []
        for channel in self.channels:
            channel_features.append([channel.nodeID1, channel.nodeID2, channel.weight1, channel.weight2])
            channel_features[-1].extend([0]*(padding_len-4))

        # 把每个path初始化为[通道id + 双向余额 + 0]的状态
        path_features = []
        for path in self.paths:
            path_feature = []
            for idx, channelID in enumerate(path.channels):
                channel = self.channels[channelID]
                d = path.channel_derection[idx]
                if d == 1:
                    path_feature.extend([channel.nodeID1, channel.nodeID2, channel.weight1, channel.weight2])
                else:
                    path_feature.extend([channel.nodeID2, channel.nodeID1, channel.weight2, channel.weight1])
            if len(path_feature) > padding_len :
                raise ValueError("路径长度超过padding_len")
            path_feature.extend([0]*(padding_len-len(path_feature)))
            path_features.append(path_feature)
        return torch.tensor(channel_features, dtype=torch.float32, device="cuda"), torch.tensor(path_features, dtype=torch.float32, device="cuda")
