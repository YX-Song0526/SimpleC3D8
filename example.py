import numpy as np
from stiffness_matrix import *
from preprocess import coord, elements, num_dof, num_nodes
from visualize import visualize_unique_faces, visualize_outer_faces, visualize_volume_elements

K = cal_K_total(coord, elements)  # 总刚
F = np.zeros(num_dof)
U = np.zeros(num_dof)

F[-11::3] = -8e8
F[-2] = -4e8
F[-11] = -4e8

K_reduced = K[48:, 48:]

F_reduced = F[48:]
U_reduced = np.linalg.solve(K_reduced, F_reduced)

U[48:] = U_reduced
U = U.reshape(num_nodes, 3)
coord_new = coord + U

print(U)

# print(np.linalg.matrix_rank(K))
visualize_volume_elements(coord_new, elements, show_edges=True, color="lightblue", opacity=1)
