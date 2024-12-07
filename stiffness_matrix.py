import numpy as np
from preprocess import num_dof, node_dof_idx

# 材料属性
E = 69e9  # 杨氏模量
niu = 0.28  # 泊松比

# 弹性矩阵
D = (E / (1 + niu) * (1 - 2 * niu)) * np.array([[1 - niu, niu, niu, 0, 0, 0],
                                                [niu, 1 - niu, niu, 0, 0, 0],
                                                [niu, niu, 1 - niu, 0, 0, 0],
                                                [0, 0, 0, (1 - 2 * niu) / 2, 0, 0],
                                                [0, 0, 0, 0, (1 - 2 * niu) / 2, 0],
                                                [0, 0, 0, 0, 0, (1 - 2 * niu) / 2]])

# 定义高斯积分点和权重
gauss_points = [-np.sqrt(1 / 3), np.sqrt(1 / 3)]
gauss_weights = [1, 1]


# 定义形函数
def shape_func(s, n, t):
    N = np.array([(1 - s) * (1 - n) * (1 - t),
                  (1 + s) * (1 - n) * (1 - t),
                  (1 + s) * (1 + n) * (1 - t),
                  (1 - s) * (1 + n) * (1 - t),
                  (1 - s) * (1 - n) * (1 + t),
                  (1 + s) * (1 - n) * (1 + t),
                  (1 + s) * (1 + n) * (1 + t),
                  (1 - s) * (1 + n) * (1 + t)]) / 8

    return N


# 定义形函数的导数矩阵
def shape_func_derivatives(s, n, t):
    dN_dxi = np.array([
        [-(1 - n) * (1 - t), -(1 - s) * (1 - t), -(1 - s) * (1 - n)],
        [(1 - n) * (1 - t), -(1 + s) * (1 - t), -(1 + s) * (1 - n)],
        [(1 + n) * (1 - t), (1 + s) * (1 - t), -(1 + s) * (1 + n)],
        [-(1 + n) * (1 - t), (1 - s) * (1 - t), -(1 - s) * (1 + n)],
        [-(1 - n) * (1 + t), -(1 - s) * (1 + t), (1 - s) * (1 - n)],
        [(1 - n) * (1 + t), -(1 + s) * (1 + t), (1 + s) * (1 - n)],
        [(1 + n) * (1 + t), (1 + s) * (1 + t), (1 + s) * (1 + n)],
        [-(1 + n) * (1 + t), (1 - s) * (1 + t), (1 - s) * (1 + n)]
    ]) / 8
    return dN_dxi


def cal_Ke(node_coords):
    Ke = np.zeros((24, 24))
    for gp_t, wt in zip(gauss_points, gauss_weights):
        for gp_n, wn in zip(gauss_points, gauss_weights):
            for gp_s, ws in zip(gauss_points, gauss_weights):

                # 形函数导数
                dN_dxi = shape_func_derivatives(gp_s, gp_n, gp_t)

                # Jacobian矩阵
                J = dN_dxi.T @ node_coords
                # print(dN_dxi.T)

                detJ = np.linalg.det(J)
                if detJ <= 0:
                    raise ValueError("Jacobian determinant is non-positive!")
                J_inv = np.linalg.inv(J)

                # B矩阵计算
                q = J_inv @ dN_dxi.T
                qx, qy, qz = q[0, :], q[1, :], q[2, :]

                B = np.zeros((6, 24))

                B[0, 0::3] = qx
                B[1, 1::3] = qy
                B[2, 2::3] = qz
                B[3, 0::3], B[3, 1::3] = qy, qx
                B[4, 1::3], B[4, 2::3] = qz, qy
                B[5, 0::3], B[5, 2::3] = qz, qx

                # print(B)

                Ke += B.T @ D @ B * detJ * ws * wn * wt
    # print(B)
    return Ke


def cal_K_total(node_coords, elements):
    K_total = np.zeros((num_dof, num_dof))
    for element in elements:
        ele_nodes = node_coords[element]
        Ke = cal_Ke(ele_nodes)
        dof = node_dof_idx[element].reshape(24)
        # print(dof)
        for i in range(24):
            for j in range(24):
                K_total[dof[i], dof[j]] += Ke[i, j]

    return K_total
