import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import random
from sklearn.cluster import SpectralClustering
import heapq

# 定义一个 Graph 类来存储图的结构
class Graph:
    def __init__(self):
        self.graph = nx.Graph()  # 使用 networkx 创建一个无向图
        self.node_positions = {}  # 存储节点坐标
        self.edge_dirtiness = {}  # 存储每条边的脏度
        self.edge_decay = {}  # 存储每条边的脏度增长系数
        self.cleaners_positions = {}  # 用于存储每个清洁车的位置

    def add_edge(self, start, end, length):
        """
        添加边到图中，并初始化脏度和变化系数
        """
        self.graph.add_edge(start, end, weight=length)
        # 使用 (min, max) 来处理无向边，保证键的一致性
        edge_key = (min(start, end), max(start, end))
        # 随机初始化脏度在 [10, 20] 之间
        self.edge_dirtiness[edge_key] = random.uniform(10, 20)
        # 随机初始化脏度增长系数 a 在 [0.5, 1.0] 之间，线性增长
        self.edge_decay[edge_key] = random.uniform(0.5, 1.0)

    def load_edges_from_csv(self, file_path):
        """
        从 CSV 文件加载边的数据
        """
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 跳过标题行
            for row in reader:
                start, end, length = int(row[0]), int(row[1]), float(row[2])
                self.add_edge(start, end, length)

    def load_points_from_csv(self, file_path):
        """
        从 CSV 文件加载点的坐标数据
        """
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 跳过标题行
            for row in reader:
                point_id = int(row[0])
                x, y = float(row[1]), float(row[2])
                self.node_positions[point_id] = (x, y)

    def smooth_line(self, start, end, curvature_factor=0.1):
        """
        计算平滑的直线，边缘稍微过渡自然
        """
        p0 = self.node_positions[start]
        p1 = self.node_positions[end]

        # 计算控制点，使得边略微弯曲
        control_point = (
            (p0[0] + p1[0]) / 2 + np.random.uniform(-curvature_factor, curvature_factor),
            (p0[1] + p1[1]) / 2 + np.random.uniform(-curvature_factor, curvature_factor)
        )

        # 生成插值的平滑线段
        t_values = np.linspace(0, 1, 100)  # 从0到1的参数值
        curve_points = []

        for t in t_values:
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * control_point[0] + t ** 2 * p1[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * control_point[1] + t ** 2 * p1[1]
            curve_points.append((x, y))

        return curve_points

    def compute_adjacency_matrix(self):
        """
        计算邻接矩阵，用于谱聚类
        """
        nodes = list(self.graph.nodes())
        n = len(nodes)
        adj_matrix = np.zeros((n, n))

        node_index = {node: idx for idx, node in enumerate(nodes)}  # Map node to index

        # 为每对节点之间的边赋值
        for edge in self.graph.edges():
            u, v = edge
            edge_key = (min(u, v), max(u, v))
            dirtiness = self.edge_dirtiness.get(edge_key, 0)
            i, j = node_index[u], node_index[v]
            adj_matrix[i][j] = dirtiness
            adj_matrix[j][i] = dirtiness  # Because the graph is undirected

        return adj_matrix, nodes

    def cluster_nodes(self, n_clusters=3):
        """
        使用谱聚类对图中的节点进行区域划分
        """
        adj_matrix, nodes = self.compute_adjacency_matrix()

        # 使用 SpectralClustering 进行谱聚类
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        labels = clustering.fit_predict(adj_matrix)

        # 返回每个节点的聚类标签
        node_clusters = {node: labels[i] for i, node in enumerate(nodes)}
        return node_clusters

    def update_dirtiness(self):
        """
        在每个时间步更新所有道路的脏度，脏度线性增加
        """
        for edge_key in self.edge_dirtiness:
            self.edge_dirtiness[edge_key] += self.edge_decay[edge_key]
            # 确保脏度最大为100
            if self.edge_dirtiness[edge_key] > 100:
                self.edge_dirtiness[edge_key] = 100

    def decrease_dirtiness(self, edge_key, amount):
        """
        减少指定道路的脏度
        """
        if edge_key in self.edge_dirtiness:
            self.edge_dirtiness[edge_key] = max(0, self.edge_dirtiness[edge_key] - amount)

    def draw(self, ax, node_clusters):
        """
        绘制图，使用 matplot 库进行可视化
        根据脏度随时间变化的计算来动态调整边的颜色和宽度
        """
        ax.clear()  # 清除当前图形
        ax.set_title(f"校园清洁模拟", fontsize=16, fontweight='bold')

        # 设置背景颜色
        ax.set_facecolor('whitesmoke')

        # 更新每条边的颜色和宽度
        for edge in self.graph.edges:
            start, end = edge
            edge_key = (min(start, end), max(start, end))
            dirtiness = self.edge_dirtiness[edge_key]  # 当前脏度

            # 通过脏度映射到颜色和边的宽度（脏度越高，颜色越深，边宽越大）
            color = plt.cm.Reds(dirtiness / 100)  # 使用红色 colormap
            width = 1 + (dirtiness / 100) * 4  # 使脏度越高边宽越大

            # 计算平滑的直线
            curve_points = self.smooth_line(start, end)
            x_values, y_values = zip(*curve_points)

            # 绘制曲线
            ax.plot(x_values, y_values, color=color, lw=width)

        # 绘制节点，节点根据聚类进行颜色分配
        node_size = [self.graph.degree(node) * 50 for node in self.graph.nodes()]  # 调整节点大小
        colors = []
        for node in self.graph.nodes():
            cluster_id = node_clusters[node]
            if cluster_id == 0:
                colors.append('skyblue')
            elif cluster_id == 1:
                colors.append('lightgreen')
            else:
                colors.append('salmon')

        nx.draw_networkx_nodes(self.graph, self.node_positions, node_size=node_size, node_color=colors, alpha=0.8, ax=ax)

        # 去除数字标签，不再调用 nx.draw_networkx_labels
        # nx.draw_networkx_labels(self.graph, self.node_positions, font_size=10, font_color="black", font_weight="bold",
        #                         bbox=dict(facecolor='none', edgecolor='none'), ax=ax)

        # 添加网格和轴标签
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(-100, 1100)
        ax.set_ylim(-100, 1100)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)

        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Cluster 0', markerfacecolor='skyblue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Cluster 1', markerfacecolor='lightgreen', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Cluster 2', markerfacecolor='salmon', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper right')

# 定义 Cleaner 类
class Cleaner:
    def __init__(self, graph, speed=None, start_node=None, cleaner_id=0):
        """
        初始化清洁车
        graph: 图对象
        speed: 清洁车的速度，单位：米/秒，随机在[1, 1.5]范围内
        start_node: 清洁车的起始位置
        cleaner_id: 清洁车的唯一ID
        """
        self.graph = graph  # 图对象
        self.speed = random.uniform(1, 1.5) if speed is None else speed  # 清洁车速度随机初始化
        self.cleaned_edges = set()  # 记录已清扫的边
        self.current_node = start_node  # 当前节点
        self.path = []  # 清洁车的路径
        self.progress = 0  # 当前路径上的进度（0到1之间）
        self.cleaner_id = cleaner_id  # 清洁车ID

    def set_path(self, path):
        """ 设置清洁车的路径 """
        self.path = path
        self.progress = 0  # 重置路径进度

    def move(self):
        """
        清洁车沿路径移动，每次移动到下一个节点，并更新经过路径的脏度。
        """
        if not self.path:
            return  # 如果没有路径，停止移动

        next_node = self.path.pop(0)  # 获取下一个节点
        edge_key = (min(self.current_node, next_node), max(self.current_node, next_node))

        # 清洁车经过的边脏度减少[10-20]
        dirtiness_decrease = random.uniform(10, 20)  # 脏度减少范围
        self.graph.decrease_dirtiness(edge_key, dirtiness_decrease)

        # 更新当前节点
        self.current_node = next_node

    def get_position(self):
        """
        返回清洁车的当前位置（当前节点）
        """
        return self.graph.node_positions[self.current_node]

def astar(graph, start, end):
    """
    使用A*算法计算从起点到终点的最短路径
    """
    def heuristic(node, end):
        # 使用欧几里得距离作为启发式函数
        x1, y1 = graph.node_positions[node]
        x2, y2 = graph.node_positions[end]
        return math.hypot(x2 - x1, y2 - y1)

    # 初始化开放列表、关闭列表和父节点字典
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, end), 0, start))  # (f, g, node)
    came_from = {}
    g_costs = {start: 0}

    while open_list:
        _, g, current_node = heapq.heappop(open_list)

        if current_node == end:
            # 找到路径，回溯
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            return path[::-1]  # 返回反向路径

        for neighbor in graph.graph.neighbors(current_node):
            edge_key = (min(current_node, neighbor), max(current_node, neighbor))
            distance = graph.graph[current_node][neighbor]["weight"]
            tentative_g = g + distance

            if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                came_from[neighbor] = current_node
                g_costs[neighbor] = tentative_g
                f_cost = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_list, (f_cost, tentative_g, neighbor))

    return []  # 如果没有路径

def plan_cleaning_paths(graph, cleaners, node_clusters):
    """
    为每个清洁车规划路径，确保覆盖其所属区域内的所有边
    """
    for cleaner in cleaners:
        cluster_id = cleaner.cleaner_id
        # 获取该区域内的所有节点
        nodes_in_cluster = [node for node, cluster in node_clusters.items() if cluster == cluster_id]
        if not nodes_in_cluster:
            continue

        # 选择一个随机的目标节点
        target_node = random.choice(nodes_in_cluster)

        if cleaner.current_node != target_node:
            path = astar(graph, cleaner.current_node, target_node)
            if path:
                # 路径中的第一个节点是当前节点，移除
                path = path[1:]
                cleaner.set_path(path)

def animate_simulation(time_step, graph, cleaners, node_clusters, ax):
    """
    动画中每个帧的更新函数
    """
    # 每个时间步更新道路脏度
    graph.update_dirtiness()

    # 为每个清洁车规划路径
    plan_cleaning_paths(graph, cleaners, node_clusters)

    # 移动每个清洁车
    for cleaner in cleaners:
        cleaner.move()

    # 绘制图形，反映脏度更新和清洁车位置
    graph.draw(ax, node_clusters)

    # 绘制清洁车的位置
    for cleaner in cleaners:
        pos = cleaner.get_position()
        ax.plot(pos[0], pos[1], marker='o', markersize=10, markeredgecolor='black',
                markerfacecolor='blue', label=f"Cleaner {cleaner.cleaner_id}" if time_step == 0 else "")

    # 添加时间步信息
    ax.set_title(f"校园清洁模拟 - 时间步: {time_step}", fontsize=16, fontweight='bold')

def main():
    # 创建图对象并加载数据
    graph = Graph()
    graph.load_edges_from_csv('Edges.csv')  # 加载边数据
    graph.load_points_from_csv('Points.csv')  # 加载点数据

    # 定义区域数
    n_clusters = 3  # 假设将地图分为3个区域

    # 获取节点的聚类结果
    node_clusters = graph.cluster_nodes(n_clusters=n_clusters)

    # 初始化清洁车
    cleaners = []
    for cluster_id in range(n_clusters):
        # 获取当前区域的节点
        nodes_in_cluster = [node for node, cluster in node_clusters.items() if cluster == cluster_id]
        if nodes_in_cluster:
            # 随机选择一个节点作为清洁车的起始位置
            start_node = random.choice(nodes_in_cluster)
            graph.cleaners_positions[cluster_id] = graph.node_positions[start_node]

            # 创建 Cleaner 实例
            cleaner = Cleaner(graph=graph, start_node=start_node, cleaner_id=cluster_id)
            cleaners.append(cleaner)

    # 创建动画
    fig, ax = plt.subplots(figsize=(12, 10))

    def update(frame):
        animate_simulation(frame, graph, cleaners, node_clusters, ax)

    ani = animation.FuncAnimation(fig, update, frames=range(1, 100), interval=500, repeat=False)

    # 显示图形
    plt.show()

if __name__ == "__main__":
    main()
