import matplotlib.pyplot as plt
import networkx as nx
import random

class WeightedDirectedGraph:
    def __init__(self, random=False):
        # 用字典存储邻接表结构，键为顶点，值为包含元组(目标顶点,权值)的列表
        self.adjacency_list = {}

    def add_vertex(self, vertex):
        """添加顶点，若已存在则不操作"""
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []

    def add_edge(self, source, destination, weight):
        """
        添加带权有向边
        :param source: 源顶点
        :param destination: 目标顶点
        :param weight: 边权值
        """
        if source not in self.adjacency_list:
            self.add_vertex(source)
        self.adjacency_list[source].append((destination, weight))

    def get_vertices(self):
        """获取所有顶点"""
        return list(self.adjacency_list.keys())

    def get_edges(self, vertex):
        """获取指定顶点的所有出边"""
        return self.adjacency_list.get(vertex, [])
    
    def to_networkx(self):
        """将存储结构转换为networkx图对象"""
        G = nx.DiGraph()
        for source in self.adjacency_list:
            for destination, weight in self.adjacency_list[source]:
                G.add_weighted_edges_from([(source, destination, weight)])
        return G
    
    def show_network(self):
        """绘制拓扑图"""
        G = self.to_networkx()
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=12)
        edges = nx.draw_networkx_edges(
            G, pos, 
            arrowstyle='->', arrowsize=30, 
            edge_color='gray', width=2
        )
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(
            G, pos, 
            edge_labels=edge_labels,
            font_color='red', 
            label_pos=0.5
        )
        plt.axis('off')
        plt.title("Weighted Directed Graph Visualization", fontsize=14)
        plt.show()
    
    @classmethod
    def generate_random_dag(cls, m_nodes, n_edges):
        """
        生成带权有向无环图的类方法
        :param m_nodes: 节点数量(≥1)
        :param n_edges: 边数量(需满足0 ≤ n ≤ m(m-1)/2)
        :return: WeightedDirectedGraph实例
        """
        # 参数校验
        max_edges = m_nodes * (m_nodes - 1)
        if n_edges > max_edges:
            raise ValueError(f"对于{m_nodes}个节点，最大边数为{max_edges}")

        # 生成节点(A,B,C,...)
        vertices = [chr(65 + i) for i in range(m_nodes)]
        random.shuffle(vertices)  # 生成随机拓扑序列‌:ml-citation{ref="2,3" data="citationList"}

        # 创建图实例
        graph = cls()
        for v in vertices:
            graph.add_vertex(v)

        # 生成有效边集合
        edges = set()
        while len(edges) < n_edges:
            # 在拓扑序列前半部分随机选起点，后半部分选终点‌:ml-citation{ref="2,4" data="citationList"}
            i = random.randint(0, len(vertices)-2)
            j = random.randint(i+1, len(vertices)-1)
            edges.add((vertices[i], vertices[j]))

        # 添加带权边
        for src, dest in edges:
            weight = random.randint(1, 10)
            graph.add_edge(src, dest, weight)
        
        return graph
    
    def find_all_cycles(self):
        """改进版DFS检测简单循环路径‌:ml-citation{ref="1,2" data="citationList"}"""
        visited = {v: False for v in self.adjacency_list}
        cycles = set()

        def backtrack(node, path):
            visited[node] = True
            current_path = path + [node]

            for neighbor, _ in self.adjacency_list.get(node, []):
                if neighbor in current_path:
                    cycle_start = current_path.index(neighbor)
                    cycle = current_path[cycle_start:] + [neighbor]
                    normalized = self._normalize_cycle(cycle)
                    cycles.add(tuple(normalized))
                elif not visited[neighbor]:
                    backtrack(neighbor, current_path)
            visited[node] = False

        for vertex in self.get_vertices():
            backtrack(vertex, [])

        return [list(c) for c in cycles]

    
    def _normalize_cycle(self, path):
        min_char = min(path)
        start_idx = path.index(min_char)
        rotated = path[start_idx:-1] + path[:start_idx+1]
        return rotated

    def print_cycles(self):
        """增强打印功能"""
        cycles = self.find_all_cycles()
        if not cycles:
            print("未检测到有效循环路径")
            return

        print(f"发现{len(cycles)}个有效循环：")
        for i, path in enumerate(cycles, 1):
            print(f"路径{i}: {' → '.join(path)}")

if __name__ == "__main__":

    graph = WeightedDirectedGraph()
    graph.add_edge('A', 'B', 5)
    graph.add_edge('B', 'C', 3)
    graph.add_edge('C', 'D', 2)
    graph.add_edge('D', 'A', 7)
    graph.add_edge('B', 'D', 10)
    graph.show_network()
    graph.print_cycles()

    # a = WeightedDirectedGraph.generate_random_dag(10, 50)
    # a.show_network()
    # a.print_cycles()