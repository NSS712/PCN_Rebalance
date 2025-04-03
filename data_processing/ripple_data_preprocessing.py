import json
import networkx as nx
from collections import defaultdict
from typing import List, Tuple
from ultils.entity import State, Path, Channel  # 保持原有实体类结构
import random

def build_ripple_network(
    accounts_file: str, 
    trust_lines_file: str,
    transactions_file: str,
    max_nodes: int = 100
) -> Tuple[State, List[dict]]:
    """
    Ripple网络状态构建函数
    
    Args:
        accounts_file: 账户数据文件路径
        trust_lines_file: 信任线数据文件路径
        transactions_file: 交易历史数据文件路径
        max_nodes: 最大采样节点数
        
    Returns:
        State: 网络状态对象
        List[dict]: 模拟交易历史
    """
    accounts_file = "data/ripple/ripple-data-jan29.all"
    
    # ----------------- 数据加载 -----------------
    # 加载账户数据
    with open(accounts_file, 'r') as f:
        accounts = [acc['address'] for acc in json.load(f)['accounts']]
    
    # 加载信任线数据（过滤无效条目）
    trust_lines = []
    with open(trust_lines_file, 'r') as f:
        for line in f:
            tl = json.loads(line)
            if tl['balance'] > 0 and tl['limit'] > 0:
                trust_lines.append(tl)
    
    # 加载交易样本（单位: drops）
    with open(transactions_file, 'r') as f:
        tx_values = [int(tx['Amount']) for tx in json.load(f)['transactions']][:1000]

    # ----------------- 拓扑构建 -----------------
    # 创建有向图（表示资金流向）
    DG = nx.DiGraph()
    
    # 添加节点
    DG.add_nodes_from(accounts)
    
    # 添加带权边（基于信任线）
    for tl in trust_lines:
        src = tl['account']
        dst = tl['counterparty']
        # 余额为当前可用额度，限额为最大可发送量
        DG.add_edge(src, dst, 
                    balance=tl['balance'],
                    limit=tl['limit'],
                    currency=tl['currency'])
    
    # 采样最大连通子图
    if max_nodes < len(DG.nodes):
        largest_scc = max(nx.strongly_connected_components(DG), key=len)
        nodes = list(largest_scc)[:max_nodes]
        DG = DG.subgraph(nodes).copy()
    
    # ----------------- 数据结构转换 -----------------
    # 节点ID映射
    node_map = {n: idx for idx, n in enumerate(DG.nodes)}
    
    # 构建Channel对象（转换为双向通道）
    channels = []
    for i, (src, dst, data) in enumerate(DG.edges(data=True)):
        # Ripple信任线转换为双向通道：
        # weight1: src->dst方向可用余额 
        # weight2: dst->src方向初始为0（需反向信任线存在）
        channels.append(
            Channel(ID=i,
                    nodeID1=node_map[src],
                    nodeID2=node_map[dst],
                    weight1=data['balance'],
                    weight2=_get_reverse_balance(DG, dst, src))
        )
    
    # ----------------- 路径生成 -----------------
    # Ripple路径特征：网关中介路径
    paths = _generate_gateway_paths(DG, node_map)
    
    # ----------------- 交易模拟 -----------------
    state = State(node_num=len(DG.nodes),
                  channels=channels,
                  paths=paths)
    
    tx_history = []
    for amount in tx_values:
        src, dst = _random_account_pair(DG.nodes)
        path = _find_ripple_path(state, src, dst, amount)
        if path:
            tx_history.append({
                'source': node_map[src],
                'destination': node_map[dst],
                'amount': amount,
                'path': path
            })
    
    return state, tx_history

def _get_reverse_balance(DG: nx.DiGraph, src: str, dst: str) -> int:
    """获取反向信任线余额"""
    if DG.has_edge(dst, src):
        return DG[dst][src]['balance']
    return 0  # 无反向信任线

def _generate_gateway_paths(DG: nx.DiGraph, node_map: dict) -> List[Path]:
    """生成网关中介路径（示例实现）"""
    paths = []
    # 查找高度数节点作为网关
    hubs = sorted(DG.out_degree, key=lambda x: x[1], reverse=True)[:5]
    for hub_node, _ in hubs:
        # 构建以网关为中心的星型路径
        for src in DG.predecessors(hub_node):
            for dst in DG.successors(hub_node):
                if src != dst:
                    path = Path(ID=len(paths))
                    # 添加路径段：src->hub->dst
                    path.channels.extend([
                        _find_channel_id(DG, src, hub_node),
                        _find_channel_id(DG, hub_node, dst)
                    ])
                    path.channel_derection = [1, 1]  # 正向流动
                    paths.append(path)
    return paths

def _find_channel_id(DG: nx.DiGraph, src: str, dst: str) -> int:
    """查找通道ID（示例实现需完善）"""
    for i, (u, v, _) in enumerate(DG.edges(data=True)):
        if u == src and v == dst:
            return i
    return -1

def _random_account_pair(nodes):
    """随机选择交易对"""
    src = random.choice(nodes)
    dst = random.choice([n for n in nodes if n != src])
    return src, dst

def _find_ripple_path(state: State, src: int, dst: int, amount: int) -> list:
    """Ripple路径查找（简化版）"""
    # 实现思路：
    # 1. 检查直接通道
    for ch in state.channels:
        if ch.nodeID1 == src and ch.nodeID2 == dst:
            if ch.weight1 >= amount:
                return [ch.ID]
    # 2. 查找通过网关的路径
    for path in state.paths:
        if path.nodes[0] == src and path.nodes[-1] == dst:
            if _check_path_capacity(path, amount, state.channels):
                return path.channels
    return []