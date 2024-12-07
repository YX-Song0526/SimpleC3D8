import numpy as np
from stiffness_matrix import cal_K_total
from preprocess import coord, elements, num_dof, num_nodes
from visualize import visualize_unique_faces, visualize_volume_elements

# 计算全局刚度矩阵
K = cal_K_total(coord, elements)

# 初始化力向量和位移向量
F = np.zeros(num_dof)
U = np.zeros(num_dof)

# 识别顶面节点
top_z = coord[:, 2].max()
top_nodes = np.where(coord[:, 2] == top_z)[0]

# 分类顶面节点
# Nx, Ny = 11, 11  # 从 preprocess.py 中获取
top_nodes_coords = coord[top_nodes]
corner_nodes = []
edge_nodes = []
inner_nodes = []

for idx, node in zip(top_nodes, top_nodes_coords):
    x, y, z = node
    if (x in [coord[:, 0].min(), coord[:, 0].max()] and
            y in [coord[:, 1].min(), coord[:, 1].max()]):  # 角点
        corner_nodes.append(idx)
    elif (x in [coord[:, 0].min(), coord[:, 0].max()] or
          y in [coord[:, 1].min(), coord[:, 1].max()]):  # 边缘节点
        edge_nodes.append(idx)
    else:  # 内圈节点
        inner_nodes.append(idx)

# 施加力
force_per_unit = -1e10  # 每个内圈节点的单位力

# 角点节点力
for node in corner_nodes:
    F[3 * node + 2] = force_per_unit / 4  # z 方向力

# 边缘节点力
for node in edge_nodes:
    F[3 * node + 2] = force_per_unit / 2  # z 方向力

# 内圈节点力
for node in inner_nodes:
    F[3 * node + 2] = force_per_unit  # z 方向力

# 减少自由度矩阵，排除固定节点
K_reduced = K[48:, 48:]
F_reduced = F[48:]

# 求解位移
U_reduced = np.linalg.solve(K_reduced, F_reduced)
U[48:] = U_reduced

# 变形后节点坐标
U = U.reshape(num_nodes, 3)
coord_new = coord + U

# 输出结果并可视化
print("节点位移向量 (U):")
print(U)
visualize_volume_elements(coord_new, elements, show_edges=True, color="lightblue", opacity=1)