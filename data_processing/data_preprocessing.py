import json
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
from torch import Value
from tqdm import tqdm
import random
import pickle
from networkx.classes.filters import no_filter
from ultils.entity import State, Path, Channel

def load_nodes(file_path):
    """
    加载节点数据
    """
    print("加载节点数据...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建节点ID到节点信息的映射
    nodes = set()
    for node in data['nodes']:
        node_id = node['nodeid']
        nodes.add(node_id)
    
    print(f"成功加载 {len(nodes)} 个节点")
    return nodes

def load_channels(file_path):
    """
    加载通道数据
    """
    print("加载通道数据...")
    channels = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            channels = data.get('channels', [])
    except json.JSONDecodeError:
        print("通道文件格式错误，尝试逐行解析...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    channel = json.loads(line.strip())
                    channels.append(channel)
                except json.JSONDecodeError:
                    continue
    
    print(f"成功加载 {len(channels)} 个通道")
    return channels

def load_bitcoin_values(file_path):
    """
    加载比特币价值数据
    """
    print("加载比特币价值数据...")
    values = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                value = int(float((line.strip())))
                values.append(value)
            except ValueError:
                continue
    
    print(f"成功加载 {len(values)} 个比特币价值")
    return values

def create_graph(nodes, channels, bitcoin_values=None):
    """
    创建网络图，返回最大连通分量
    """
    print("创建网络图...")
    G = nx.Graph()
    
    # 添加节点
    for node_id in nodes:
        alias = node_id[:10] + '...'
        G.add_node(node_id, alias=alias)
    
    # 添加边（通道）
    edge_weights = defaultdict(float)
    
    shot_channal_id_set = set()
    for channel in channels:
        if not channel.get('active'):
            continue
        node1 = channel.get('source')
        node2 = channel.get('destination')
        shot_channal_id = channel.get('short_channel_id')
        if shot_channal_id in shot_channal_id_set:
            continue
        shot_channal_id_set.add(shot_channal_id)
        # 确保节点存在
        if node1 and node2 and node1 in nodes and node2 in nodes:
            capacity = int(channel.get('satoshis', 0))
            
            # 累加相同节点对之间的容量
            edge_key = tuple(sorted([node1, node2]))
            edge_weights[edge_key] += capacity
    
    # 将累加后的边添加到图中
    for (node1, node2), weight in edge_weights.items():
        G.add_edge(node1, node2, weight=weight)
    
    print(f"初始拓扑图创建完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")

    # 获取最大连通分量
    connected_components = list(nx.connected_components(G))
    largest_cc = max(connected_components, key=len)
    G_largest_cc = G.subgraph(largest_cc).copy()
    print(f"最大连通分量包含 {G_largest_cc.number_of_nodes()} 个节点和 {G_largest_cc.number_of_edges()} 条边")
    return G_largest_cc

def visualize_graph(G, output_path=None, max_nodes=None):
    """
    可视化网络图
    """
    print("可视化网络图...")
    
    # 设置节点大小基于度
    node_size = [100]
    
    # 设置边宽度基于权重
    # edge_width = [0.1 + 0.01 * G_largest_cc[u][v].get('weight', 0) / 10000 for u, v in G_largest_cc.edges()]
    edge_width = 3
    # 使用spring布局
    pos = nx.spring_layout(G, k=0.05, iterations=100)
    # pos = nx.kamada_kawai_layout(G_largest_cc)
    # pos = nx.spectral_layout(G_largest_cc)
    plt.figure(figsize=(20, 20))
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', alpha=0.8)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.3, edge_color='gray')
    
    # 为重要节点添加标签
    # important_nodes = sorted(G_largest_cc.nodes(), key=lambda x: G_largest_cc.degree(x), reverse=True)[:20]
    # labels = {node: G_largest_cc.nodes[node].get('alias', node[:10]) for node in important_nodes}
    # nx.draw_networkx_labels(G_largest_cc, pos, labels=labels, font_size=8, font_family='sans-serif')
    
    plt.title("Lightning Network")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到 {output_path}")
    
    plt.show()

def generate_candidate_paths(G, num_paths=10, weight_increment=10, edge_coverage_threshold=1.0):
    """
    优化后的生成候选循环路径集，基于边的最短环检测方法
    """
    print(f"开始生成 {num_paths} 条候选循环路径...")
    
    H = G.copy()
    # pylint: disable=unused-variable
    # 为图中所有边设置权重为1
    nx.set_edge_attributes(H, {edge: 1 for edge in H.edges()}, "weight")
    
    paths = []
    path_edges = set()
    total_edges = G.number_of_edges()
    
    if total_edges < 3:
        print("边数量不足，无法形成循环路径")
        return paths
    
    try:
        for _ in range(num_paths):
            min_cycle = None
            min_cycle_weight = float('inf')
            
            # 遍历所有边，寻找包含该边的最小环
            for u, v in list(H.edges()):
                # 暂时移除当前边
                original_weight = H[u][v]['weight']
                H.remove_edge(u, v)
                
                # 计算u到v的最短路径
                try:
                    if original_weight == 1 and all(d['weight'] == 1 for _, _, d in H.edges(data=True)):
                        # 权重全为1时使用BFS
                        path = nx.shortest_path(H, u, v)
                        path_length = len(path) - 1  # 边数
                    else:
                        # 使用Dijkstra算法
                        path_length = nx.shortest_path_length(H, u, v, weight='weight')
                        path = nx.shortest_path(H, u, v, weight='weight')
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    H.add_edge(u, v, weight=original_weight)
                    continue
                
                # 恢复边
                H.add_edge(u, v, weight=original_weight)
                
                # 计算环的总权重
                cycle_weight = original_weight + path_length
                cycle_edges = list(zip(path, path[1:] + path[0:1]))  # 组成环的边
                # cycle_edges.append((v, u))  # 添加被移除的边
                
                # 确保环至少3个节点
                if len(cycle_edges) >= 3 and cycle_weight < min_cycle_weight:
                    min_cycle = {
                        'path': path,
                        'weight': cycle_weight,
                        'edges': cycle_edges
                    }
                    min_cycle_weight = cycle_weight
            
            if min_cycle:
                paths.append(min_cycle)
                # 更新边权重
                for u, v in min_cycle['edges']:
                    if H.has_edge(u, v):
                        H[u][v]['weight'] += weight_increment
                        path_edges.add(tuple(sorted((u, v))))
                
                current_coverage = len(path_edges) / total_edges
                print(f"已生成 {len(paths)} 条循环路径，当前路径权重: {min_cycle['weight']}，边覆盖率: {current_coverage:.2%}")
                
                if current_coverage >= edge_coverage_threshold:
                    print(f"已达到目标边覆盖率 {edge_coverage_threshold:.2%}，停止查找")
                    break
            else:
                print(f"无法找到更多循环路径，已生成 {len(paths)} 条路径")
                break
                
    except Exception as e:
        print(f"生成路径时发生错误: {str(e)}")
        raise
    
    # 输出统计信息
    total_edges_covered = len(path_edges)
    avg_length = sum(len(p['path']) for p in paths) / len(paths) if paths else 0
    
    print(f"\n路径生成完成:")
    print(f"共生成 {len(paths)} 条循环路径")
    print(f"平均路径长度: {avg_length:.2f}")
    print(f"使用的不同边数量: {total_edges_covered}")
    
    return paths

def pross_data(sample=100,need_path=True):
    # 设置文件路径
    config = json.load(open("config/DRLPCR.json", "r"))
    base_dir = "d:/Doc/毕设/code/data/raw"
    nodes_file = os.path.join(base_dir, "allnodes.txt")
    channels_file = os.path.join(base_dir, "channels.txt")
    bitcoin_values_file = os.path.join(base_dir, "BitcoinVal.txt")
    
    # 加载数据
    nodes = load_nodes(nodes_file)
    channels = load_channels(channels_file)
    bitcoin_values = load_bitcoin_values(bitcoin_values_file)
    
    # 创建图,只保留最大连通分量
    G = create_graph(nodes, channels)
    if sample:
        G_sub = sample_graph(G, sample)
    else:
        G_sub = G
    
    # 迭代移除度为1的节点
    G_sub = romve_degree_one_nodes(G_sub)
    # visualize_graph(G_sub)
    if need_path:
        # 生成循环路径
        num_path = config['path_num']
        path_rate = config['path_rate']
        row_paths = generate_candidate_paths(G_sub, num_paths=num_path, weight_increment=10, edge_coverage_threshold=path_rate)
    else:
        row_paths = []
    # 替换掉原始的节点ID为索引
    row_nodes = G_sub.nodes()
    raw_channels = G_sub.edges(data=True)
    nodeID_to_index = {nodeID: i for i, nodeID in enumerate(row_nodes)}
    channel_to_index = {}
    channels = []
    paths = []
    for raw_channel in raw_channels:
        source = nodeID_to_index[raw_channel[0]]
        destination = nodeID_to_index[raw_channel[1]]
        channel = Channel(len(channels), source, destination, raw_channel[2]['weight']//2, raw_channel[2]['weight']- raw_channel[2]['weight']//2)
        channels.append(channel)
        channel_to_index[(channel.nodeID1, channel.nodeID2)] = len(channels) - 1

    for raw_path in row_paths:
        pathID = len(paths)
        path = Path(pathID)
        for raw_channel in raw_path['edges']:
            key = (nodeID_to_index[raw_channel[0]], nodeID_to_index[raw_channel[1]])
            if key in channel_to_index:
                path.channels.append(channel_to_index[key])
                path.channel_derection.append(1)
                channels[channel_to_index[key]].path.append(pathID)
            else:
                key = (nodeID_to_index[raw_channel[1]], nodeID_to_index[raw_channel[0]])
                path.channels.append(channel_to_index[key])
                path.channel_derection.append(-1)
                channels[channel_to_index[key]].path.append(pathID)
        for node in raw_path['path']:
            path.nodes.append(nodeID_to_index[node])
        paths.append(path)

    # 计算平均通道余额
    total_capacity = sum((channel.weight1 + channel.weight2) for channel in channels)
    average_capacity = total_capacity / len(channels)
    print(f"平均通道余额: {average_capacity:.2f}")

    # 计算平均交易金额
    average_transaction_amount = sum(bitcoin_values) / len(bitcoin_values)
    print(f"平均交易金额: {average_transaction_amount:.2f}")

    state = State(G_sub.number_of_nodes(), channels, paths)
    return state, bitcoin_values

def sample_graph(G, n):
    """
    从图中采样n个连通的节点，创建连通子图并移除度为1的节点
    
    参数:
    G: NetworkX图对象
    n: 需要采样的节点数量
    
    返回:
    G_sampled: 采样后的连通子图
    """
    print(f"开始从{G.number_of_nodes()}个节点中采样{n}个连通节点...")
    
    # 确保采样数量不超过总节点数
    n = min(n, G.number_of_nodes())
    
    # 使用BFS算法采样连通子图
    sampled_nodes = []
    
    # 随机选择起始节点
    start_node = random.choice(list(G.nodes()))
    sampled_nodes.append(start_node)
    
    # 使用BFS扩展连通子图
    frontier = list(G.neighbors(start_node))  # 当前边界节点（候选节点）
    
    while len(sampled_nodes) < n and frontier:
        # 从边界中随机选择一个节点
        next_node_idx = random.randrange(len(frontier))
        next_node = frontier.pop(next_node_idx)
        
        # 如果节点已经被采样，则跳过
        if next_node in sampled_nodes:
            continue
        
        # 添加到采样节点集合
        sampled_nodes.append(next_node)
        
        # 更新边界，添加新节点的邻居（如果不在已采样节点中）
        for neighbor in G.neighbors(next_node):
            if neighbor not in sampled_nodes and neighbor not in frontier:
                frontier.append(neighbor)
    
    # 如果BFS无法找到足够的节点（可能是因为图不够大），则使用已有的节点
    print(f"成功采样 {len(sampled_nodes)} 个连通节点")
    
    # 创建子图
    G_sampled = G.subgraph(sampled_nodes).copy()
    print(f"采样后的图包含 {G_sampled.number_of_nodes()} 个节点和 {G_sampled.number_of_edges()} 条边")

    return G_sampled

def romve_degree_one_nodes(G_sampled):
    """
    迭代移除度为1的节点
    """
    while True:
            degree_one_nodes = [node for node in G_sampled.nodes() if G_sampled.degree(node) == 1]
            if not degree_one_nodes:
                break
            G_sampled.remove_nodes_from(degree_one_nodes)
            print(f"移除 {len(degree_one_nodes)} 个度为1的节点")
        
    print(f"处理后的图包含 {G_sampled.number_of_nodes()} 个节点和 {G_sampled.number_of_edges()} 条边")
    return G_sampled

if __name__ == "__main__":
    # 运行复杂测试
    # test_generate_paths_complex()
    
    # 运行简单测试
    # test_generate_paths()
    
    # 运行环路测试
    # test_find_cycles()

    # 运行数据预处理
    paths = pross_data()
    with open('data/processed/paths.pkl', 'wb') as file:
        pickle.dump(paths, file)
