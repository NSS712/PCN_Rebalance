import json
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random

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

def find_cycle_paths(G, min_length=3, max_length=None):
    """
    在图中查找所有循环路径
    
    参数:
    G: NetworkX图对象
    min_length: 最小循环长度（默认3）
    max_length: 最大循环长度（默认None）
    
    返回:
    unique_cycles: 所有唯一循环的集合
    """
    print("开始查找循环路径...")
    
    def normalize_cycle(cycle):
        """标准化循环路径，使其从最小节点ID开始"""
        # 将路径转换为列表以支持旋转
        cycle = list(cycle)
        # 找到最小节点的索引
        min_idx = cycle.index(min(cycle))
        # 旋转路径使最小节点在开头
        return tuple(cycle[min_idx:] + cycle[:min_idx])

    # 使用NetworkX的内置函数查找简单环路
    cycles = set()
    
    # 根据最大长度限制查找环路
    if max_length:
        for length in range(min_length, max_length + 1):
            print(f"查找长度为 {length} 的环路...")
            # 使用simple_cycles查找定长环路
            current_cycles = nx.simple_cycles(G, length)
            # 标准化并添加到结果集
            for cycle in current_cycles:
                if len(cycle) >= min_length:
                    normalized = normalize_cycle(cycle)
                    if normalized not in cycles:
                        cycles.add(normalized)
                        if len(cycles) % 1000 == 0:  # 每找到1000个环路打印一次进度
                            print(f"已找到 {len(cycles)} 个唯一环路")
    else:
        # 如果没有最大长度限制，直接使用simple_cycles
        print("查找所有环路...")
        current_cycles = nx.simple_cycles(G)
        for cycle in current_cycles:
            if len(cycle) >= min_length:
                normalized = normalize_cycle(cycle)
                if normalized not in cycles:
                    cycles.add(normalized)
                    if len(cycles) % 1000 == 0:  # 每找到1000个环路打印一次进度
                        print(f"已找到 {len(cycles)} 个唯一环路")

    print(f"循环路径查找完成，共找到 {len(cycles)} 个唯一环路")
    return cycles

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

    connected_components = list(nx.connected_components(G))
    largest_cc = max(connected_components, key=len)
    G_largest_cc = G.subgraph(largest_cc)
    print(f"最大连通分量包含 {G_largest_cc.number_of_nodes()} 个节点和 {G_largest_cc.number_of_edges()} 条边")

    count = 0
    for node in G.nodes():
        if G.degree(node) == 0:
            count += 1
    print(f"度为0的节点数量: {count}")
    return G_largest_cc

def visualize_graph(G, output_path=None, max_nodes=30):
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

def test_find_cycles():
    """
    测试find_cycle_paths函数的有效性
    创建一个5个节点10条边的简单图进行测试
    图的结构如下：
    1 --- 2
    |\\   /|
    | \\ / |
    |  X  |
    | / \\ |
    |/   \\|
    4 --- 3
    5连接到1、2、3、4
    """
    # 创建测试图
    G = nx.Graph()
    
    # 添加节点
    nodes = ['1', '2', '3', '4', '5']
    G.add_nodes_from(nodes)
    
    # 添加边
    edges = [
        ('1', '2'), ('1', '3'), ('1', '4'), ('1', '5'),
        ('2', '3'), ('2', '4'), ('2', '5'),
        ('3', '4'), ('3', '5'),
        ('4', '5')
    ]
    G.add_edges_from(edges)
    
    print("测试图信息：")
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边数量: {G.number_of_edges()}")
    
    # 可视化测试图
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16, font_weight='bold')
    plt.title("测试图")
    plt.show()
    
    # 测试不同长度的环路查找
    print("\n测试不同长度的环路查找：")
    
    # 测试长度为3的环路
    print("\n查找长度为3的环路：")
    cycles3 = find_cycle_paths(G, min_length=3, max_length=3)
    print(f"找到 {len(cycles3)} 个三角形环路：")
    for i, cycle in enumerate(cycles3, 1):
        print(f"环路 {i}: {' -> '.join(cycle)}")
    
    # 测试长度为4的环路
    print("\n查找长度为4的环路：")
    cycles4 = find_cycle_paths(G, min_length=4, max_length=4)
    print(f"找到 {len(cycles4)} 个四边形环路：")
    for i, cycle in enumerate(cycles4, 1):
        print(f"环路 {i}: {' -> '.join(cycle)}")
    
    # 测试长度为5的环路
    print("\n查找长度为5的环路：")
    cycles5 = find_cycle_paths(G, min_length=5, max_length=5)
    print(f"找到 {len(cycles5)} 个五边形环路：")
    for i, cycle in enumerate(cycles5, 1):
        print(f"环路 {i}: {' -> '.join(cycle)}")
    
    # 验证结果
    expected_counts = {
        3: 10,  # 应该有10个三角形
        4: 5,   # 应该有5个四边形
        5: 1    # 应该有1个五边形
    }
    
    print("\n验证结果：")
    for length, expected in expected_counts.items():
        cycles = find_cycle_paths(G, min_length=length, max_length=length)
        actual = len(cycles)
        print(f"长度为{length}的环路：预期 {expected} 个，实际找到 {actual} 个",
              "✓" if actual == expected else "✗")

def generate_candidate_paths(G, num_paths=10, weight_increment=10, edge_coverage_threshold=1.0):
    """
    基于迪杰斯特拉算法生成候选循环路径集
    
    参数:
    G: NetworkX图对象
    num_paths: 需要生成的路径数量（默认10）
    weight_increment: 每次找到路径后增加的权重（默认10）
    edge_coverage_threshold: 边覆盖率阈值（0.0-1.0），达到该阈值时停止查找（默认1.0，即100%覆盖）
    
    返回:
    paths: 包含所有候选循环路径的列表，每个路径是一个字典，包含path、weight和edges
    """
    print(f"开始生成 {num_paths} 条候选循环路径...")
    
    # 创建一个新的图用于权重调整
    H = G.copy()
    
    # 初始化所有边的权重为1
    nx.set_edge_attributes(H, 1, 'weight')
    
    # 存储找到的路径
    paths = []
    
    # 记录已经找到的路径的边集合
    path_edges = set()
    
    # 获取图中所有边的总数
    total_edges = G.number_of_edges()
    
    try:
        # 获取所有节点
        nodes = list(H.nodes())
        
        # 确保图中有足够的节点形成循环
        if len(nodes) < 3:
            print("节点数量不足，无法形成循环路径")
            return paths
        
        # 迭代生成指定数量的循环路径
        for i in range(num_paths):
            # 尝试找到权重最小的循环
            min_cycle = None
            min_cycle_weight = float('inf')
            
            # 对每个节点，尝试找到以它为起点的最小权重循环
            for source in nodes:
                # 创建一个临时图，移除起点
                temp_graph = H.copy()
                
                # 获取起点的所有邻居
                neighbors = list(H.neighbors(source))
                
                # 如果节点没有邻居，跳过
                if not neighbors:
                    continue
                
                # 移除起点但保留其连接信息
                temp_graph.remove_node(source)
                
                # 对每对邻居节点，尝试找到它们之间的最短路径
                for i, neighbor1 in enumerate(neighbors):
                    for neighbor2 in neighbors[i+1:]:
                        # 检查两个邻居是否在临时图中
                        if not (temp_graph.has_node(neighbor1) and temp_graph.has_node(neighbor2)):
                            continue
                            
                        # 检查两个邻居之间是否有路径
                        if not nx.has_path(temp_graph, neighbor1, neighbor2):
                            continue
                        
                        try:
                            # 找到两个邻居之间的最短路径
                            path = nx.dijkstra_path(temp_graph, neighbor1, neighbor2, weight='weight')
                            path_weight = nx.dijkstra_path_length(temp_graph, neighbor1, neighbor2, weight='weight')
                            
                            # 计算完整循环的权重（加上从source到两个邻居的边的权重）
                            total_weight = path_weight
                            total_weight += H[source][neighbor1].get('weight', 1)
                            total_weight += H[source][neighbor2].get('weight', 1)
                            
                            # 如果找到更小权重的循环，更新最小循环
                            if total_weight < min_cycle_weight:
                                # 构建完整的循环路径
                                cycle_path = [source, neighbor1] + path[1:] + [source]
                                cycle_edges = [(cycle_path[j], cycle_path[j+1]) for j in range(len(cycle_path)-1)]
                                
                                min_cycle = {
                                    'path': cycle_path,
                                    'weight': total_weight,
                                    'edges': cycle_edges
                                }
                                min_cycle_weight = total_weight
                        
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            continue
            
            # 如果找到了最小循环，添加到路径集合并更新边权重
            if min_cycle:
                paths.append(min_cycle)
                
                # 更新路径上所有边的权重
                for u, v in min_cycle['edges']:
                    # 获取当前权重
                    current_weight = H[u][v].get('weight', 1)
                    # 增加权重
                    H[u][v]['weight'] = current_weight + weight_increment
                    # 记录这条边
                    edge = tuple(sorted([u, v]))
                    path_edges.add(edge)
                
                # 计算当前边覆盖率
                current_coverage = len(path_edges) / total_edges
                print(f"已生成 {len(paths)} 条循环路径，当前路径权重: {min_cycle['weight']}，边覆盖率: {current_coverage:.2%}")
                
                # 检查是否达到边覆盖率阈值
                if current_coverage >= edge_coverage_threshold:
                    print(f"已达到目标边覆盖率 {edge_coverage_threshold:.2%}，停止查找")
                    break
            else:
                print(f"无法找到更多循环路径，已生成 {len(paths)} 条路径")
                break
                
    except Exception as e:
        print(f"生成路径时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 计算路径的多样性
    total_edges = len(path_edges)
    avg_path_length = sum(len(p['path']) for p in paths) / len(paths) if paths else 0
    
    print(f"\n路径生成完成:")
    print(f"共生成 {len(paths)} 条循环路径")
    print(f"平均路径长度: {avg_path_length:.2f}")
    print(f"使用的不同边数量: {total_edges}")
    
    return paths

def test_generate_paths():
    """
    测试generate_candidate_paths函数
    使用之前的测试图进行测试
    """
    # 创建测试图
    G = nx.Graph()
    
    # 添加节点
    nodes = ['1', '2', '3', '4', '5']
    G.add_nodes_from(nodes)
    
    # 添加边
    edges = [
        ('1', '2'), ('1', '3'), ('1', '4'), ('1', '5'),
        ('2', '3'), ('2', '4'), ('2', '5'),
        ('3', '4'), ('3', '5'),
        ('4', '5')
    ]
    G.add_edges_from(edges)
    
    print("测试图信息：")
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边数量: {G.number_of_edges()}")
    
    # 可视化测试图
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16, font_weight='bold')
    plt.title("测试图")
    plt.show()
    
    # 测试路径生成
    num_paths = 5
    
    print(f"\n测试生成 {num_paths} 条最短循环路径:")
    paths = generate_candidate_paths(G, num_paths)
    
    # 打印所有生成的路径
    print("\n生成的路径:")
    for i, path_info in enumerate(paths, 1):
        path = path_info['path']
        weight = path_info['weight']
        print(f"路径 {i}: {' -> '.join(path)}, 权重: {weight}")
        
        # 可视化当前路径
        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=16, font_weight='bold')
        # 高亮显示当前路径
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                             edge_color='r', width=3)
        plt.title(f"路径 {i} (权重: {weight})")
        plt.show()

def test_generate_paths_complex():
    """
    测试generate_candidate_paths函数
    创建一个10个节点20条边的复杂图进行测试
    测试边覆盖率参数和可视化所有路径
    """
    # 创建测试图
    G = nx.Graph()
    
    # 添加节点
    nodes = [str(i) for i in range(1, 11)]
    G.add_nodes_from(nodes)
    
    # 添加边，确保图是连通的且存在多个循环
    edges = [
        # 外圈
        ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '6'),
        ('6', '7'), ('7', '8'), ('8', '9'), ('9', '10'), ('10', '1'),
        
        # 内圈
        ('1', '5'), ('2', '6'), ('3', '7'), ('4', '8'), ('5', '9'),
        ('6', '10'), ('7', '1'), ('8', '2'), ('9', '3'), ('10', '4'),
        
        # 添加一些额外的边以增加循环路径
        ('1', '3'), ('2', '4'), ('3', '5'), ('4', '6'), ('5', '7'),
        ('6', '8'), ('7', '9'), ('8', '10'), ('9', '1'), ('10', '2')
    ]
    G.add_edges_from(edges)
    
    # 设置边的初始权重，使路径权重更有区分度
    edge_weights = {
        # 外圈权重较小
        ('1', '2'): 1, ('2', '3'): 1, ('3', '4'): 1, ('4', '5'): 1, ('5', '6'): 1,
        ('6', '7'): 1, ('7', '8'): 1, ('8', '9'): 1, ('9', '10'): 1, ('10', '1'): 1,
        
        # 内圈权重较大
        ('1', '5'): 2, ('2', '6'): 2, ('3', '7'): 2, ('4', '8'): 2, ('5', '9'): 2,
        ('6', '10'): 2, ('7', '1'): 2, ('8', '2'): 2, ('9', '3'): 2, ('10', '4'): 2,
        
        # 额外边的权重最大
        ('1', '3'): 3, ('2', '4'): 3, ('3', '5'): 3, ('4', '6'): 3, ('5', '7'): 3,
        ('6', '8'): 3, ('7', '9'): 3, ('8', '10'): 3, ('9', '1'): 3, ('10', '2'): 3
    }
    
    # 设置边的权重
    for (u, v), weight in edge_weights.items():
        G[u][v]['weight'] = weight
    
    print("测试图信息：")
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边数量: {G.number_of_edges()}")
    
    # 计算并显示图的基本特性
    print(f"平均度: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"最大度: {max(dict(G.degree()).values())}")
    print(f"最小度: {min(dict(G.degree()).values())}")
    
    # 可视化原始图
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=700, font_size=16, font_weight='bold',
            edge_color='gray', width=1)
    plt.title("原始测试图")
    plt.show()
    
    # 测试路径生成
    num_paths = 50
    edge_coverage = 1  # 设置60%的边覆盖率阈值
    print(f"\n测试生成最多 {num_paths} 条最短循环路径或达到 {edge_coverage:.0%} 边覆盖率:")
    paths = generate_candidate_paths(G, num_paths, edge_coverage_threshold=edge_coverage)
    
    # 打印所有生成的路径
    print("\n生成的路径:")
    for i, path_info in enumerate(paths, 1):
        path = path_info['path']
        weight = path_info['weight']
        print(f"路径 {i}: {' -> '.join(path)} -> {path[0]}, 权重: {weight}")
        
        # 可视化当前路径
        plt.figure(figsize=(12, 12))
        # 绘制所有边（灰色）
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.3)
        # 绘制所有节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=700, alpha=0.7)
        # 添加节点标签
        nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')
        # 高亮显示当前路径（红色）
        path_edges = list(zip(path[:-1], path[1:]))
        # 添加首尾相连的边
        path_edges.append((path[-1], path[0]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                             edge_color='r', width=3)
        plt.title(f"路径 {i} (权重: {weight})")
        plt.axis('off')
        plt.show()
    
    # 可视化所有路径（展示边覆盖率）
    plt.figure(figsize=(12, 12))
    # 绘制所有边（灰色）
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.3)
    # 绘制所有节点
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                         node_size=700, alpha=0.7)
    # 添加节点标签
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')
    
    # 收集所有路径的边
    all_path_edges = set()
    for path_info in paths:
        for u, v in path_info['edges']:
            all_path_edges.add((u, v))
    
    # 高亮显示所有路径的边（红色）
    nx.draw_networkx_edges(G, pos, edgelist=list(all_path_edges), 
                         edge_color='r', width=2)
    
    # 计算边覆盖率
    edge_coverage = len(all_path_edges) / G.number_of_edges()
    plt.title(f"所有路径的边覆盖率: {edge_coverage:.2%}")
    plt.axis('off')
    plt.show()
    
    # 分析路径的多样性
    all_edges = set()
    for path_info in paths:
        all_edges.update(path_info['edges'])
    
    print("\n路径分析:")
    print(f"使用的不同边数量: {len(all_edges)}")
    print(f"总边数量: {G.number_of_edges()}")
    print(f"边覆盖率: {len(all_edges) / G.number_of_edges():.2%}")
    
    # 计算路径长度的分布
    path_lengths = [len(p['path']) for p in paths]
    print("\n路径长度分布:")
    if path_lengths:
        print(f"最短路径长度: {min(path_lengths)}")
        print(f"最长路径长度: {max(path_lengths)}")
        print(f"平均路径长度: {sum(path_lengths) / len(path_lengths):.2f}")
    else:
        print("未找到任何路径")

def main():
    # 设置文件路径
    base_dir = "d:/Doc/毕设/code/data/raw"
    nodes_file = os.path.join(base_dir, "allnodes.txt")
    channels_file = os.path.join(base_dir, "channels.txt")
    bitcoin_values_file = os.path.join(base_dir, "BitcoinVal.txt")
    
    # 加载数据
    nodes = load_nodes(nodes_file)
    channels = load_channels(channels_file)
    # bitcoin_values = load_bitcoin_values(bitcoin_values_file)
    
    # 创建图,只保留最大连通分量
    G = create_graph(nodes, channels)
    
    # 查找循环路径
    print("\n查找循环路径...")
    cycles = find_cycle_paths(G)
    print(f"找到 {len(cycles)} 个唯一循环路径")
    
    # 打印前5个循环路径示例
    print("\n循环路径示例（前5个）:")
    for i, cycle in enumerate(cycles[:5]):
        print(f"\n循环 {i+1}:")
        print(f"节点: {' -> '.join(cycle['nodes'])}")
        print(f"通道: {', '.join(cycle['channels'])}")
    
    # 分析图的基本特性
    print("\n图的基本特性:")
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边数量: {G.number_of_edges()}")
    
    # 计算连通分量
    connected_components = list(nx.connected_components(G))
    print(f"连通分量数量: {len(connected_components)}")
    print(f"最大连通分量大小: {len(max(connected_components, key=len))}")
    
    # 计算平均度
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    print(f"平均度: {avg_degree:.2f}")
    
    # 可视化
    output_path = os.path.join(base_dir, "..", "processed", "lightning_network_topology.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    visualize_graph(G, output_path)

if __name__ == "__main__":
    # 运行复杂测试
    test_generate_paths_complex()
    
    # 运行简单测试
    # test_generate_paths()
    
    # 运行环路测试
    # test_find_cycles()
    
    # 运行主程序
    # main()