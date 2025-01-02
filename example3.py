import numpy as np
from stiffness_matrix import cal_K_total, cal_M_total, calculate_element_stress
from preprocess import coord, elements, num_dof, num_nodes, Nx, Ny
from visualize import visualize_unique_faces, visualize_volume_elements, visualize_cell_stress
import pandas as pd
from scipy.linalg import eigh

b = 3 * Nx * Ny

K = cal_K_total(coord, elements)
M = cal_M_total(coord, elements)

K_reduced = K[b:-b, b:-b]
M_reduced = M[b:-b, b:-b]

# print(np.linalg.det(K_reduced))

# 求解广义特征值问题
eigenvalues, eigenvectors = eigh(K_reduced, M_reduced)

# print(eigenvalues)
# print(eigenvectors)

# 计算模态频率（以Hz表示）
frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

U = np.zeros(num_dof)
U[b:-b] = eigenvectors[:, 6]  # 第3阶振型
U = U.reshape(num_nodes, 3)

coord_new = coord + 50 * U  # 将位移放大十倍
print(len(U))

stress = calculate_element_stress(coord_new, elements, U.flatten())

visualize_cell_stress(coord_new, elements, stress)
