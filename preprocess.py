import numpy as np

# 节点数
Nx = 4
Ny = 4
Nz = 39

num_nodes = Nx * Ny * Nz  # 节点总数
num_elements = (Nx - 1) * (Ny - 1) * (Nz - 1)  # 单元总数

x_start, x_end = 0, 5
y_start, y_end = 0, 5
z_start, z_end = 0, 40

x_step, y_step, z_step = (x_end - x_start) / (Nx - 1), (y_end - y_start) / (Ny - 1), (z_end - z_start) / (Nz - 1)

# 填充坐标矩阵
coord = np.array([[x_start + i * x_step, y_start + j * y_step, z_start + k * z_step]
                  for k in range(Nz) for j in range(Ny) for i in range(Nx)])

# 单元节点索引矩阵，每一列代表每个单元的8个节点索引
elements = np.zeros((num_elements, 8), dtype=int)

# 填充单元节点索引矩阵
ie = 0
for k in range(Nz - 1):
    for j in range(Ny - 1):
        for i in range(Nx - 1):
            n1 = k * Nx * Ny + j * Nx + i
            n2 = n1 + 1
            n3 = n2 + Nx
            n4 = n3 - 1
            n5 = n1 + Nx * Ny
            n6 = n2 + Nx * Ny
            n7 = n3 + Nx * Ny
            n8 = n4 + Nx * Ny

            elements[ie, :] = [n1, n2, n3, n4, n5, n6, n7, n8]
            ie += 1

# 节点自由度索引矩阵，每一列代表每个节点的三个自由度索引
node_dof_idx = np.zeros((num_nodes, 3), dtype=int)

# 填充自由度索引矩阵
n = 0
for i in range(num_nodes):
    node_dof_idx[i, :] = [n, n + 1, n + 2]
    n += 3

# 自由度总数
num_dof = n

l = np.arange(3*num_nodes).reshape(num_nodes, 3)
# print(coord)
# print(elements)
# print(node_dof_idx[-111])
# print(l-node_dof_idx)
