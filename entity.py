import random
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

class Graph:
    def __init__(self, nodes, channels):
        self.nodes = nodes
        self.channels = channels
        self.paths = []

        # 初始化每个节点的权重，用于指数分布来采样模拟交易
        weights = np.random.exponential(scale=1, size=len(self.nodes))
        normalized_weights = weights / np.sum(weights)  # 归一化权重
        self.weights = normalized_weights
    
    def transaction(self, source, destination, amount):
        pass

    def rebalance_transaction(self, amounts):
        '''
        根据amounts, 对每个path执行交易。
        '''
        pass

    def compute_reward(self):
        """
        计算当前状态的平衡度, 结果在[-1, 1]之间
        """
        ans = 0
        for idx in range(len(self.channels)):
            channel = self.channels[idx]
            ans += channel['balance'] / channel['capacity']
        return ans

    def random_transaction(self):
        '''
        随机选择两个节点，生成一笔交易。
        '''
        pass
    
    def act(self, action):
        '''
        根据action，更新图的状态。
        '''
        pass

class State:
    def __init__(self, graph):
        self.graph = graph
        self.channels = graph.channels
        self.paths = graph.paths