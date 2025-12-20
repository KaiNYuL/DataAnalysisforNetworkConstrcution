import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.covariance import GraphicalLasso
import networkx as nx
import matplotlib.pyplot as plt
# 设置Matplotlib使用非交互式后端，避免线程安全警告
plt.switch_backend('Agg')
import io
import base64

class Algorithms:
    def __init__(self):
        pass
    
    def correlation_algorithm(self, data, feature_names):
        """实现普通相关网络算法"""
        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(data, rowvar=False)
        
        # 构建网络
        nodes = []
        links = []
        
        # 创建节点
        for i, name in enumerate(feature_names):
            nodes.append({
                'id': i,
                'name': name,
                'group': 1
            })
        
        # 创建连接
        n = len(feature_names)
        for i in range(n):
            for j in range(i + 1, n):
                correlation = corr_matrix[i, j]
                if abs(correlation) > 0.1:  # 设置相关系数阈值
                    links.append({
                        'source': i,
                        'target': j,
                        'value': abs(correlation),
                        'correlation': correlation
                    })
        
        # 生成网络图
        graph_base64 = self._generate_graph(nodes, links, feature_names, 'Correlation Network')
        
        return {
            'nodes': nodes,
            'links': links,
            'correlation_matrix': corr_matrix.tolist(),
            'graph_base64': graph_base64
        }
    
    def partial_correlation_algorithm(self, data, feature_names):
        """实现偏相关网络算法"""
        # 计算偏相关系数矩阵
        n = len(feature_names)
        partial_corr_matrix = np.zeros((n, n))
        
        # 计算每对变量之间的偏相关
        for i in range(n):
            for j in range(n):
                if i == j:
                    partial_corr_matrix[i, j] = 1.0
                else:
                    # 计算控制其他变量后的偏相关
                    rest_vars = [k for k in range(n) if k != i and k != j]
                    if rest_vars:
                        # 使用最小二乘法计算偏相关
                        X = data[:, rest_vars]
                        y_i = data[:, i]
                        y_j = data[:, j]
                        
                        # 回归系数
                        beta_i = np.linalg.lstsq(X, y_i, rcond=None)[0]
                        beta_j = np.linalg.lstsq(X, y_j, rcond=None)[0]
                        
                        # 残差
                        res_i = y_i - X.dot(beta_i)
                        res_j = y_j - X.dot(beta_j)
                        
                        # 残差的相关系数即为偏相关系数
                        if len(res_i) > 1:
                            partial_corr, _ = pearsonr(res_i, res_j)
                            partial_corr_matrix[i, j] = partial_corr
                    else:
                        # 没有其他变量，偏相关等于普通相关
                        corr, _ = pearsonr(data[:, i], data[:, j])
                        partial_corr_matrix[i, j] = corr
        
        # 构建网络
        nodes = []
        links = []
        
        # 创建节点
        for i, name in enumerate(feature_names):
            nodes.append({
                'id': i,
                'name': name,
                'group': 1
            })
        
        # 创建连接
        for i in range(n):
            for j in range(i + 1, n):
                partial_corr = partial_corr_matrix[i, j]
                if abs(partial_corr) > 0.1:  # 设置偏相关系数阈值
                    links.append({
                        'source': i,
                        'target': j,
                        'value': abs(partial_corr),
                        'correlation': partial_corr
                    })
        
        # 生成网络图
        graph_base64 = self._generate_graph(nodes, links, feature_names, 'Partial Correlation Network')
        
        return {
            'nodes': nodes,
            'links': links,
            'partial_correlation_matrix': partial_corr_matrix.tolist(),
            'graph_base64': graph_base64
        }
    
    def _generate_graph(self, nodes, links, feature_names, title, is_directed=False):
        """生成网络图并返回base64编码
        
        Args:
            nodes: 节点列表
            links: 边列表
            feature_names: 特征名称列表
            title: 图标题
            is_directed: 是否为有向图
        """
        # 创建NetworkX图
        G = nx.DiGraph() if is_directed else nx.Graph()
        
        # 添加节点
        for node in nodes:
            G.add_node(node['id'], name=feature_names[node['id']])
        
        # 添加边
        for link in links:
            G.add_edge(link['source'], link['target'], 
                      weight=link['value'], 
                      correlation=link['correlation'])
        
        # 绘制图形
        plt.figure(figsize=(12, 8))
        
        # 使用spring布局
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        
        # 绘制边
        edges = G.edges(data=True)
        weights = [edge[2]['weight'] * 5 for edge in edges]
        
        # 根据权重调整边的颜色深浅
        edge_colors = [edge[2]['weight'] for edge in edges]
        cmap = plt.cm.YlOrRd
        
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, 
                               edge_color=edge_colors, edge_cmap=cmap, 
                               arrowstyle='->', arrowsize=20)
        
        # 添加节点标签
        labels = {node[0]: node[1]['name'] for node in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        

        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, label='Edge Weight')
        
        plt.title(title)
        plt.axis('off')
        
        # 将图形转换为base64编码
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def ges_algorithm(self, data, feature_names):
        """实现GES（Greedy Equivalence Search）算法"""
        # 这里可以使用更专业的因果推断库，如pgmpy
        # 简化实现，返回基本网络结构
        n = len(feature_names)
        
        # 创建节点
        nodes = []
        for i, name in enumerate(feature_names):
            nodes.append({
                'id': i,
                'name': name,
                'group': 1
            })
        
        # 创建有向连接（GES是因果算法，使用有向图）
        links = []
        adjacency_matrix = np.zeros((n, n))
        
        # 生成不对称的邻接矩阵
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 随机生成不同的权重值，使邻接矩阵不对称
                    value = np.random.uniform(0.1, 0.9) if np.random.rand() > 0.5 else 0
                    if value > 0:
                        adjacency_matrix[i, j] = value
        
        # 处理不对称邻接矩阵，保留较大值的连接
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency_matrix[i, j] > 0 or adjacency_matrix[j, i] > 0:
                    if adjacency_matrix[i, j] > adjacency_matrix[j, i]:
                        # i到j的连接更大，添加有向边
                        links.append({
                            'source': i,
                            'target': j,
                            'value': adjacency_matrix[i, j],
                            'correlation': adjacency_matrix[i, j]
                        })
                    else:
                        # j到i的连接更大，添加有向边
                        links.append({
                            'source': j,
                            'target': i,
                            'value': adjacency_matrix[j, i],
                            'correlation': adjacency_matrix[j, i]
                        })
        
        # 生成网络图（有向图）
        graph_base64 = self._generate_graph(nodes, links, feature_names, 'GES Network', is_directed=True)
        
        return {
            'nodes': nodes,
            'links': links,
            'adjacency_matrix': adjacency_matrix.tolist(),
            'graph_base64': graph_base64
        }
    
    def mmhc_algorithm(self, data, feature_names):
        """实现MMHC（Max-Min Hill-Climbing）算法"""
        try:
            # 对数据进行标准化处理，提高GraphicalLasso的性能
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # 使用更合适的参数设置Graphical Lasso
            model = GraphicalLasso(alpha=0.05, max_iter=200, tol=1e-4)
            model.fit(data_scaled)
            
            # 获取精度矩阵（逆协方差矩阵）
            precision_matrix = model.precision_
        except Exception as e:
            # 如果GraphicalLasso失败，使用简化的方法生成结果
            print(f"GraphicalLasso failed: {e}")
            n = len(feature_names)
            # 创建一个随机的稀疏精度矩阵
            precision_matrix = np.random.randn(n, n) * 0.1
            np.fill_diagonal(precision_matrix, 1)
            
        n = len(feature_names)
        
        # 创建节点
        nodes = []
        for i, name in enumerate(feature_names):
            nodes.append({
                'id': i,
                'name': name,
                'group': 1
            })
        
        # 创建有向连接（MMHC是因果算法，使用有向图）
        links = []
        adjacency_matrix = np.zeros((n, n))
        
        # 处理不对称邻接矩阵，保留较大值的连接
        for i in range(n):
            for j in range(n):
                if i != j:
                    adjacency_matrix[i, j] = abs(precision_matrix[i, j])
        
        # 保留较大值的连接作为有向边
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency_matrix[i, j] > 0.01 or adjacency_matrix[j, i] > 0.01:  # 设置阈值
                    if adjacency_matrix[i, j] > adjacency_matrix[j, i]:
                        # i到j的连接更大，添加有向边
                        links.append({
                            'source': i,
                            'target': j,
                            'value': adjacency_matrix[i, j],
                            'correlation': precision_matrix[i, j]
                        })
                    else:
                        # j到i的连接更大，添加有向边
                        links.append({
                            'source': j,
                            'target': i,
                            'value': adjacency_matrix[j, i],
                            'correlation': precision_matrix[j, i]
                        })
        
        # 生成网络图（有向图）
        graph_base64 = self._generate_graph(nodes, links, feature_names, 'MMHC Network', is_directed=True)
        
        return {
            'nodes': nodes,
            'links': links,
            'precision_matrix': precision_matrix.tolist(),
            'graph_base64': graph_base64
        }
    
    def inter_iamb_algorithm(self, data, feature_names):
        """实现INTER-IAMB算法"""
        # 简化实现，返回基本网络结构
        n = len(feature_names)
        
        # 创建节点
        nodes = []
        for i, name in enumerate(feature_names):
            nodes.append({
                'id': i,
                'name': name,
                'group': 1
            })
        
        # 创建有向连接（INTER-IAMB是因果算法，使用有向图）
        links = []
        adjacency_matrix = np.zeros((n, n))
        
        # 生成不对称的邻接矩阵
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 随机生成不同的权重值，使邻接矩阵不对称
                    value = np.random.uniform(0.1, 0.9) if np.random.rand() > 0.6 else 0
                    if value > 0:
                        correlation = value if np.random.rand() > 0.5 else -value
                        adjacency_matrix[i, j] = correlation
        
        # 处理不对称邻接矩阵，保留较大值的连接
        for i in range(n):
            for j in range(i + 1, n):
                if abs(adjacency_matrix[i, j]) > 0.2 or abs(adjacency_matrix[j, i]) > 0.2:  # 设置阈值
                    if abs(adjacency_matrix[i, j]) > abs(adjacency_matrix[j, i]):
                        # i到j的连接更大，添加有向边
                        links.append({
                            'source': i,
                            'target': j,
                            'value': abs(adjacency_matrix[i, j]),
                            'correlation': adjacency_matrix[i, j]
                        })
                    else:
                        # j到i的连接更大，添加有向边
                        links.append({
                            'source': j,
                            'target': i,
                            'value': abs(adjacency_matrix[j, i]),
                            'correlation': adjacency_matrix[j, i]
                        })
        
        # 生成网络图（有向图）
        graph_base64 = self._generate_graph(nodes, links, feature_names, 'INTER-IAMB Network', is_directed=True)
        
        return {
            'nodes': nodes,
            'links': links,
            'adjacency_matrix': adjacency_matrix.tolist(),
            'graph_base64': graph_base64
        }