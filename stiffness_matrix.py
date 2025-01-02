import numpy as np
from preprocess import num_dof, node_dof_idx

# 材料属性
E = 69e9  # 杨氏模量
niu = 0.28  # 泊松比
rho = 1

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


def cal_Me(node_coords):
    """
    计算单元质量质量矩阵

    Args:
        node_coords: 节点坐标矩阵 (num_nodes, 3)
    """
    Me = np.zeros((24, 24))  # 修改大小以适应 2D 单元
    for gp_t, wt in zip(gauss_points, gauss_weights):
        for gp_n, wn in zip(gauss_points, gauss_weights):
            for gp_s, ws in zip(gauss_points, gauss_weights):

                # 计算形函数
                N = shape_func(gp_s, gp_n, gp_t)
                N_matrix = np.zeros((3, 24))
                N_matrix[0, 0::3] = N
                N_matrix[1, 1::3] = N
                N_matrix[2, 2::3] = N

                # 雅可比矩阵计算
                dN_dxi = shape_func_derivatives(gp_s, gp_n, gp_t)
                J = dN_dxi.T @ node_coords
                detJ = np.linalg.det(J)
                if detJ <= 0:
                    raise ValueError("Jacobian determinant is non-positive!")

                # 质量矩阵累加
                Me += rho * (N_matrix.T @ N_matrix) * detJ * ws * wn * wt
    return Me


def cal_M_total(node_coords, elements):
    """
    计算系统总体质量矩阵

    Args:
        elements: 单元连接矩阵 (num_elements, 4)
        node_coords: 节点坐标矩阵 (num_nodes, 2)
    """
    M_total = np.zeros((num_dof, num_dof))
    for element in elements:
        ele_nodes = node_coords[element]
        Me = cal_Me(ele_nodes)
        # print(Me)
        dof = node_dof_idx[element].reshape(24)
        # print(dof)
        for i in range(24):
            for j in range(24):
                M_total[dof[i], dof[j]] += Me[i, j]

    return M_total


# 为每个单元计算应力
def calculate_element_stress(node_coords, elements, U):
    """
    计算单元应力。

    Args:
        U: 节点位移向量
        elements: 单元连接矩阵 (num_elements, 8)
        node_coords: 节点坐标矩阵 (num_nodes, 3)
    """
    element_stresses = []
    for element in elements:
        ele_nodes = node_coords[element]
        ele_dof = node_dof_idx[element].reshape(-1)
        U_e = U[ele_dof]

        # 初始化高斯点应力累加
        # stress_at_gauss_points = []

        B = cal_B_matrix(ele_nodes)

        strain = B @ U_e
        stress = D @ strain

        element_stresses.append(stress)

        # for gp_t, wt in zip(gauss_points, gauss_weights):
        #     for gp_n, wn in zip(gauss_points, gauss_weights):
        #         for gp_s, ws in zip(gauss_points, gauss_weights):
        #             # 计算形函数导数
        #             dN_dxi = shape_func_derivatives(gp_s, gp_n, gp_t)
        #             J = dN_dxi.T @ ele_nodes
        #             J_inv = np.linalg.inv(J)
        #             dN_dxy = J_inv @ dN_dxi.T
        #
        #             # B矩阵计算
        #             q = J_inv @ dN_dxi.T
        #             qx, qy, qz = q[0, :], q[1, :], q[2, :]
        #
        #             B = np.zeros((6, 24))
        #
        #             B[0, 0::3] = qx
        #             B[1, 1::3] = qy
        #             B[2, 2::3] = qz
        #             B[3, 0::3], B[3, 1::3] = qy, qx
        #             B[4, 1::3], B[4, 2::3] = qz, qy
        #             B[5, 0::3], B[5, 2::3] = qz, qx
        #
        #             # 计算应变和应力
        #             strain = B @ U_e
        #             stress = D @ strain
        #             stress_at_gauss_points.append(stress)

        # 对所有高斯点的应力取平均
        # element_stresses.append(np.mean(stress_at_gauss_points, axis=0))

    return np.array(element_stresses)


def cal_B_matrix(ele_nodes):
    """
    计算八节点六面体单元的B矩阵。
    :param ele_nodes: 当前单元的节点坐标矩阵
    :return: B矩阵
    """
    # 计算单元的形函数导数
    s, n, t = np.array([0, 0, 0])  # Gauss积分点位置（需要调整根据高斯积分）
    dN_dxi = shape_func_derivatives(s, n, t)

    # 计算Jacobian矩阵并求逆
    J = dN_dxi.T @ ele_nodes
    J_inv = np.linalg.inv(J)

    # 计算B矩阵
    B = np.zeros((6, 24))
    q = J_inv @ dN_dxi.T
    qx, qy, qz = q[0, :], q[1, :], q[2, :]

    B[0, 0::3] = qx
    B[1, 1::3] = qy
    B[2, 2::3] = qz
    B[3, 0::3], B[3, 1::3] = qy, qx
    B[4, 1::3], B[4, 2::3] = qz, qy
    B[5, 0::3], B[5, 2::3] = qz, qx

    return B


def map_element_stresses_to_nodes(elements, element_stresses, num_nodes):
    """
    将单元应力值映射到节点。
    :param elements: 单元连接矩阵 (num_elements, 8)
    :param element_stresses: 单元应力值 (num_elements,)
    :param num_nodes: 节点总数
    :return: 每个节点的应力值 (num_nodes,)
    """
    node_stress = np.zeros(num_nodes)
    node_count = np.zeros(num_nodes)  # 记录每个节点被多少个单元共享

    for element, stress in zip(elements, element_stresses):
        for node in element:
            node_stress[node] += stress
            node_count[node] += 1

    # 取平均值
    node_stress /= node_count
    return node_stress
