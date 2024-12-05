import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import random
from sklearn.cluster import SpectralClustering


# 定义一个 Graph 类来存储图的结构
class Graph:
    def __init__(self):
        self.graph = nx.Graph()  # 使用 networkx 创建一个无向图
        self.node_positions = {}  # 存储节点坐标
        self.edge_dirtiness = {}  # 存储每条边的脏度
        self.edge_decay = {}  # 存储每条边的脏度变化系数

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

        # 为每对节点之间的边赋值
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if self.graph.has_edge(node1, node2):
                    # 使用脏度或距离等作为权重（可以进一步改进权重的定义）
                    edge_key = (min(node1, node2), max(node1, node2))
                    adj_matrix[i][j] = self.edge_dirtiness.get(edge_key, 0)  # 使用脏度作为边权重

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
            weight = self.graph[start][end]['weight']  # 获取边的权重
            dirtiness = self.calculate_dirtiness(edge_key, time)  # 根据时间计算脏度

            # 通过脏度映射到颜色和边的宽度（脏度越高，颜色越深，边宽越大）
            color = plt.cm.YlGn(dirtiness / 100)  # 通过脏度映射到颜色（YlGn colormap）
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

        # 添加节点标签，去除背景
        nx.draw_networkx_labels(self.graph, self.node_positions, font_size=10, font_color="black", font_weight="bold",
                                bbox=dict(facecolor='none', edgecolor='none'))

        # 增加网格和坐标轴
        plt.grid(True, linestyle='--', alpha=0.5)  # 显示网格
        plt.xlim(-100, 1100)  # 设置 X 轴范围
        plt.ylim(-100, 1100)  # 设置 Y 轴范围
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)

        # 显示图形
        plt.tight_layout()  # 自动调整子图参数
        plt.show()

    def calculate_dirtiness(self, edge_key, time):
        """
        计算道路的脏度，使用 alnt 对数函数来计算脏度随时间的变化，脏度最大为 100。
        """
        initial_dirtiness = self.edge_dirtiness[edge_key]  # 获取该边的初始脏度
        decay_factor = self.edge_decay[edge_key]  # 获取该边的脏度变化系数
        dirtiness = initial_dirtiness + decay_factor * math.log(time + 1)  # 使用对数函数计算脏度
        return min(dirtiness, 100)  # 确保脏度最大为 100

# 创建图对象并加载数据
graph = Graph()
graph.load_edges_from_csv('Edges.csv')  # 加载边数据
graph.load_points_from_csv('Points.csv')  # 加载点数据

# 创建动画函数
def animate(time):
    graph.draw(time)

# 创建动画
fig, ax = plt.subplots(figsize=(10, 8))
ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, 18, 0.5), interval=500)  # interval 设置为300毫秒

# 展示图形
plt.show()
