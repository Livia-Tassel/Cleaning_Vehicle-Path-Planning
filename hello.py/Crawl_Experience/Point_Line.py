import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import random
from sklearn.cluster import SpectralClustering
from matplotlib.lines import Line2D

# 定义一个 Graph 类来存储图的结构
class Graph:
    def __init__(self):
        self.graph = nx.Graph()  # 使用 networkx 创建一个无向图
        self.node_positions = {}  # 存储节点坐标
        self.edge_dirtiness = {}  # 存储每条边的脏度
        self.edge_decay = {}  # 存储每条边的脏度变化系数
        self.cleaners_positions = {}  # 存储清洁车的位置

    def add_edge(self, start, end, length):
        """
        添加边到图中，并初始化脏度和变化系数
        """
        self.graph.add_edge(start, end, weight=length)
        # 使用 (min, max) 来处理无向边，保证键的一致性
        edge_key = (min(start, end), max(start, end))
        # 随机初始化脏度在 [10, 20] 之间
        self.edge_dirtiness[edge_key] = random.uniform(10, 20)
        # 随机初始化脏度变化系数 a 在 [20, 25] 之间
        self.edge_decay[edge_key] = random.uniform(20, 25)

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

    def calculate_dirtiness(self, edge_key, time):
        """
        计算道路的脏度，使用 alnt 对数函数来计算脏度随时间的变化，脏度最大为 100。
        """
        initial_dirtiness = self.edge_dirtiness[edge_key]  # 获取该边的初始脏度
        decay_factor = self.edge_decay[edge_key]  # 获取该边的脏度变化系数
        dirtiness = initial_dirtiness + decay_factor * math.log(time + 1)  # 使用对数函数计算脏度
        return min(dirtiness, 100)  # 确保脏度最大为 100

    def draw(self, time):
        """
        绘制图，使用 matplot 库进行可视化
        根据脏度随时间变化的计算来动态调整边的颜色和宽度
        """
        plt.clf()  # 清除当前图形
        plt.title(f"Time: {time} hours", fontsize=16, fontweight='bold')

        # 设置背景颜色渐变
        plt.gcf().set_facecolor('whitesmoke')

        # 进行聚类并获取区域划分
        node_clusters = self.cluster_nodes(n_clusters=3)

        # 更新每条边的脏度和颜色
        for edge in self.graph.edges:
            start, end = edge
            edge_key = (min(start, end), max(start, end))
            dirtiness = self.calculate_dirtiness(edge_key, time)  # 根据时间计算脏度

            # 通过脏度映射到颜色和边的宽度（脏度越高，颜色越深，边宽越大）
            color = plt.cm.Greens(dirtiness / 100)  # 使用绿色 colormap
            width = 1 + (dirtiness / 100) * 3  # 使脏度越高边宽越大

            # 计算平滑的直线
            curve_points = self.smooth_line(start, end)
            x_values, y_values = zip(*curve_points)

            # 绘制曲线
            plt.plot(x_values, y_values, color=color, lw=width)  # 绘制边（平滑曲线）

        # 绘制节点，节点根据聚类进行颜色分配
        node_size = [self.graph.degree(node) * 30 for node in self.graph.nodes()]  # 减小节点大小
        colors = ['skyblue' if node_clusters[node] == 0 else 'lightgreen' if node_clusters[node] == 1 else 'salmon' for
                  node in self.graph.nodes()]
        nx.draw_networkx_nodes(self.graph, self.node_positions, node_size=node_size, node_color=colors, alpha=0.8)

        # 去除数字标签，不再调用 nx.draw_networkx_labels
        # nx.draw_networkx_labels(self.graph, self.node_positions, font_size=10, font_color="black", font_weight="bold",
        #                         bbox=dict(facecolor='none', edgecolor='none'))

        # Add grid and axis labels
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(-100, 1100)
        plt.ylim(-100, 1100)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        # Display the graph
        plt.tight_layout()  # Adjust the layout of the plot
        plt.show()

# 创建图对象并加载数据
graph = Graph()
graph.load_edges_from_csv('Edges.csv')  # 加载边数据
graph.load_points_from_csv('Points.csv')  # 加载点数据

# 随机初始化清洁车在每个区域内的起始位置
n_clusters = 3  # 区域数
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

        # 这里用 enumerate() 给每个清洁车分配一个唯一的 id
        cleaner_id = cluster_id  # 使用 cluster_id 作为清洁车的 id
        cleaners.append({
            'id': cleaner_id,  # 使用 cleaner_id 代替 undefined variable
            'current_position': start_node,
            'path': [],  # 存储清洁车的路径
            'cleaned_edges': set()  # 记录清洁过的边
        })

# A* 算法，用于规划清洁车的路径
import heapq


def astar(graph, start, end):
    """
    A* 算法，用于找到从起点到终点的最短路径。
    """

    # 获取节点的坐标位置
    def heuristic(node, end):
        # 使用曼哈顿距离作为启发式函数
        x1, y1 = graph.node_positions[node]
        x2, y2 = graph.node_positions[end]
        return abs(x1 - x2) + abs(y1 - y2)

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


def plan_cleaning_paths(graph, cleaners, time):
    """
    This function plans and draws the cleaning paths for all cleaners.
    It ensures that each cleaner covers all the roads in its region.
    """
    time = int(time)  # Convert time to an integer to avoid slicing errors
    start_time = int(time - 1)  # Get the start time for the previous hour
    # plt.clf()

    # Iterate through each cleaner to draw its cleaning path
    for cleaner in cleaners:
        # Ensure the cleaner has a path and slice it based on the current and previous hour
        if 'path' in cleaner and cleaner['path']:
            # Get the path segment for the current time frame (previous hour)
            path_segment = cleaner['path'][start_time:time]

            # Draw the path segment (just the previous hour's segment)
            plt.plot([graph.node_positions[node][0] for node in path_segment],
                     [graph.node_positions[node][1] for node in path_segment],
                     color='red', marker='o', markersize=5, label=f"Cleaner {cleaner['id']}")



        # 获取该区域内所有节点
        nodes_in_cluster = [node for node, cluster in node_clusters.items() if cluster == cleaner['id']]

        # 初始化清洁路径
        cleaning_path = []

        # 选取一个初始节点进行路径规划
        current_node = cleaner['current_position']
        for next_node in nodes_in_cluster:
            path = astar(graph, current_node, next_node)
            cleaner['path'].extend(path[1:])  # 跳过当前节点，避免重复
            current_node = next_node

        # 可视化当前时间下的路径
        plt.plot([graph.node_positions[node][0] for node in cleaner['path'][start_time-1:time]],
                 [graph.node_positions[node][1] for node in cleaner['path'][start_time-1:time]], 'r-',
                 label=f"Cleaner {cleaner['id'] + 1}")


# 创建动画函数
def animate(time):
    """
    动画中每个帧的更新函数
    """
    graph.draw(time)

    # 路径规划并绘制清洁车路径
    plan_cleaning_paths(graph, cleaners, time)


# 创建动画
fig, ax = plt.subplots(figsize=(10, 8))
ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, 24, 0.5), interval=500)  # interval设置为500毫秒

# 显示图形
plt.show()