import numpy as np
from stiffness_matrix import *
from preprocess import coord, elements, num_dof, num_nodes
from visualize import visualize_unique_faces, visualize_stress_cloud, visualize_volume_elements, visualize_cell_stress

K = cal_K_total(coord, elements)  # 总刚
F = np.zeros(num_dof)
U = np.zeros(num_dof)

factor = 5

F[958] = -1e8 * factor
F[955] = -2e8 * factor
F[952] = -2e8 * factor
F[949] = -1e8 * factor
F[958-48] = -1e8 * factor
F[955-48] = -2e8 * factor
F[952-48] = -2e8 * factor
F[949-48] = -1e8 * factor
F[958+48] = -1e8 * factor
F[955+48] = -2e8 * factor
F[952+48] = -2e8 * factor
F[949+48] = -1e8 * factor


K_reduced = K[48:-48, 48:-48]

F_reduced = F[48:-48]
U_reduced = np.linalg.solve(K_reduced, F_reduced)

U[48:-48] = U_reduced
U = U.reshape(num_nodes, 3)
coord_new = coord + U

stresses = calculate_element_stress(coord, elements, U.flatten())
print(len(stresses))


visualize_cell_stress(coord_new, elements, stresses[:, 5])

