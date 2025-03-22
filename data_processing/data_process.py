import json
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import pickle
from networkx.classes.filters import no_filter

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
                value = float(line.strip())
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
    
    print(f"图创建完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")

    # 获取最大连通分量
    connected_components = list(nx.connected_components(G))
    largest_cc = max(connected_components, key=len)
    G_largest_cc = G.subgraph(largest_cc).copy()
    print(f"最大连通分量包含 {G_largest_cc.number_of_nodes()} 个节点和 {G_largest_cc.number_of_edges()} 条边")

    # 迭代移除度为1的节点
    while True:
        degree_one_nodes = [node for node in G_largest_cc.nodes() if G_largest_cc.degree(node) == 1]
        if not degree_one_nodes:
            break
        G_largest_cc.remove_nodes_from(degree_one_nodes)
        print(f"移除 {len(degree_one_nodes)} 个度为1的节点")
    
    print(f"处理后的图包含 {G_largest_cc.number_of_nodes()} 个节点和 {G_largest_cc.number_of_edges()} 条边")
    return G_largest_cc
    """
    可视化网络图
    """
    print("可视化网络图...")
    
    # 获取最大连通分量
    connected_components = list(nx.connected_components(G))
    largest_cc = max(connected_components, key=len)
    G_largest_cc = G.subgraph(largest_cc)
    print(f"选择最大连通分量进行可视化，包含 {G_largest_cc.number_of_nodes()} 个节点")
    
    # 如果节点太多，选择度最大的节点子集
    if max_nodes and G_largest_cc.number_of_nodes() > max_nodes:
        print(f"节点数量过多，选择度最大的 {max_nodes} 个节点进行可视化")
        degrees = dict(G_largest_cc.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:max_nodes]
        G_largest_cc = G_largest_cc.subgraph(top_nodes)
    
    # 设置节点大小基于度
    node_size = [20 + G_largest_cc.degree(node) * 0.05 for node in G_largest_cc.nodes()]
    
    # 设置边宽度基于权重
    # edge_width = [0.1 + 0.01 * G_largest_cc[u][v].get('weight', 0) / 10000 for u, v in G_largest_cc.edges()]
    edge_width = 0.3
    # 使用spring布局
    pos = nx.spring_layout(G_largest_cc, k=0.15, iterations=50)
    # pos = nx.kamada_kawai_layout(G_largest_cc)
    # pos = nx.spectral_layout(G_largest_cc)
    plt.figure(figsize=(20, 20))
    
    # 绘制节点
    nx.draw_networkx_nodes(G_largest_cc, pos, node_size=node_size, node_color='skyblue', alpha=0.8)
    
    # 绘制边
    nx.draw_networkx_edges(G_largest_cc, pos, width=edge_width, alpha=0.3, edge_color='gray')
    
    # 为重要节点添加标签
    # important_nodes = sorted(G_largest_cc.nodes(), key=lambda x: G_largest_cc.degree(x), reverse=True)[:20]
    # labels = {node: G_largest_cc.nodes[node].get('alias', node[:10]) for node in important_nodes}
    # nx.draw_networkx_labels(G_largest_cc, pos, labels=labels, font_size=8, font_family='sans-serif')
    
    plt.title("Lightning Network 最大连通分量拓扑图")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到 {output_path}")
    
    plt.show()

def generate_candidate_paths(G, num_paths=10, weight_increment=10, edge_coverage_threshold=1.0):
    return []

def pross_data():
    # 设置文件路径
    base_dir = "d:/Doc/毕设/code/data/raw"
    nodes_file = os.path.join(base_dir, "allnodes.txt")
    channels_file = os.path.join(base_dir, "channels.txt")
    bitcoin_values_file = os.path.join(base_dir, "BitcoinVal.txt")
    
    # 加载数据
    nodes = load_nodes(nodes_file)
    channels = load_channels(channels_file)

    nodes_list = list(nodes)
    nodes = set(range(len(nodes_list)))
    nodeID2index = {nodeID: index for index, nodeID in enumerate(nodes_list)}

    for channel in channels:
        channel = {
            'source': nodeID2index[channel['source']],
            'destination': nodeID2index[channel['destination']],
            'weight0': int(channel['satoshis']/2),
            'weight1': int(channel['satoshis']/2),
            'path_idx': []
        }
    
    # 创建图,只保留最大连通分量
    G = create_graph(nodes, channels)
    
    # 分析图的基本特性
    print("\n图的最大连通分量的基本特性:")
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边数量: {G.number_of_edges()}")
    
    # 生成循环路径
    paths = generate_candidate_paths(G, num_paths=1000, weight_increment=10, edge_coverage_threshold=1.0)
    '''
        paths返回一个list, 预期的组成结构是每个paths包含Node id的序列, 以及channel的序列。
        需要的：
    '''
    channels_idx = []
    for channel in channels:
        for path_idx, path in enumerate(paths):
            if channel in path['channels']:
                channels_idx.append(channels)
                path['channel_idx'].append(len(channels_idx)-1)
                channel['path_idx'].append(path_idx)
    return channels_idx, paths

    # # 可视化
    # output_path = os.path.join(base_dir, "..", "processed", "lightning_network_topology.png")
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # visualize_graph(G, output_path)



    
    return G

if __name__ == "__main__":
    paths = pross_data()
    with open('data/processed/paths.pkl', 'wb') as file:
        pickle.dump(paths, file)
