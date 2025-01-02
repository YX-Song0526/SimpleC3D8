import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import pyvista as pv


def visualize_unique_faces(coord, elements):
    # 定义单元的面（每个面由4个节点组成，保持正确的节点次序）
    faces_per_element = [
        [0, 3, 2, 1],  # 底面
        [4, 5, 6, 7],  # 顶面
        [0, 1, 5, 4],  # 前面
        [1, 2, 6, 5],  # 右面
        [2, 3, 7, 6],  # 后面
        [3, 0, 4, 7]  # 左面
    ]

    # 生成唯一的面集合，保持面节点的顺序
    unique_faces = set()
    unique_check = set()

    for element in elements:
        for face_indices in faces_per_element:
            face = tuple(element[i] for i in face_indices)
            # 检查面或其反向是否已存在
            face_sorted = tuple(sorted(face))
            if face_sorted not in unique_check:
                unique_check.add(face_sorted)
                unique_faces.add(face)

    # print(len(unique_faces))

    # 绘制唯一的面
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for face in unique_faces:
        vertices = [coord[i, :] for i in face]
        ax.add_collection3d(Poly3DCollection([vertices], edgecolor='k', alpha=1))

    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Finite Element Mesh')

    x_min, x_max = coord[:, 0].min(), coord[:, 0].max()
    y_min, y_max = coord[:, 1].min(), coord[:, 1].max()
    z_min, z_max = coord[:, 2].min(), coord[:, 2].max()

    x_span, y_span, z_span = x_max - x_min, y_max - y_min, z_max - z_min

    ax.set_box_aspect([x_span, y_span, z_span])

    ax.set_xlim([x_min - 0.1 * x_span, x_max + 0.1 * x_span])
    ax.set_ylim([y_min - 0.1 * y_span, y_max + 0.1 * y_span])
    ax.set_zlim([z_min, z_max + 0.2 * z_span])

    ax.view_init(elev=0, azim=0)

    plt.savefig('result', dpi=1080)
    plt.show()


def visualize_outer_faces(coord, coord_new, elements):
    """
    可视化网格的最外层面，保持节点顺序并避免重复绘制。
    :param coord: 原始节点坐标矩阵 (num_nodes, 3)，每行是一个节点的 [x, y, z] 坐标
    :param coord_new: 变形后的节点坐标矩阵 (num_nodes, 3)，每行是一个节点的 [x, y, z] 坐标
    :param elements: 单元连接矩阵 (num_elements, 8)，每行是一个单元的节点编号
    """
    # 找到最外层节点编号
    x_min, y_min, z_min = coord.min(axis=0)
    x_max, y_max, z_max = coord.max(axis=0)

    # 最外层节点编号集合
    outer_node_ids = set(
        i for i, (x, y, z) in enumerate(coord)
        if x in (x_min, x_max) or y in (y_min, y_max) or z in (z_min, z_max)
    )

    # 定义单元的面（每个面由4个节点组成，保持正确的节点次序）
    faces_per_element = [
        [0, 3, 2, 1],  # 底面
        [4, 5, 6, 7],  # 顶面
        [0, 1, 5, 4],  # 前面
        [1, 2, 6, 5],  # 右面
        [2, 3, 7, 6],  # 后面
        [3, 0, 4, 7]  # 左面
    ]

    # 筛选最外层的面，使用 check 集合避免重复
    outer_faces = set()
    check = set()
    for element in elements:
        for face_indices in faces_per_element:
            face = tuple(element[i] for i in face_indices)
            # 检查面是否为最外层面
            if all(node in outer_node_ids for node in face):
                # 如果面或其反向尚未添加到 check 集合，则添加
                if face not in check and face[::-1] not in check:
                    check.add(face)
                    outer_faces.add(face)

    # 可视化最外层的面
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for face in outer_faces:
        vertices = [coord_new[i] for i in face]
        ax.add_collection3d(Poly3DCollection([vertices], edgecolor='k', alpha=0.6, facecolors='lightcoral'))

    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Outer Faces of 3D Mesh')

    # 设置显示范围
    x_min, x_max = coord_new[:, 0].min(), coord_new[:, 0].max()
    y_min, y_max = coord_new[:, 1].min(), coord_new[:, 1].max()
    z_min, z_max = coord_new[:, 2].min(), coord_new[:, 2].max()

    x_span, y_span, z_span = x_max - x_min, y_max - y_min, z_max - z_min

    ax.set_box_aspect([x_span, y_span, z_span])

    ax.set_xlim([x_min - 0.1 * x_span, x_max + 0.1 * x_span])
    ax.set_ylim([y_min - 0.1 * y_span, y_max + 0.1 * y_span])
    ax.set_zlim([z_min, z_max + 0.2 * z_span])

    plt.show()


def visualize_outer_faces_pyvista(coord, elements):
    """
    使用 PyVista 可视化最外层的面。
    :param coord: 节点坐标矩阵 (num_nodes, 3)
    :param elements: 单元连接矩阵 (num_elements, 8)
    """
    # 定义单元的面
    faces_per_element = [
        [0, 3, 2, 1],  # 底面
        [4, 5, 6, 7],  # 顶面
        [0, 1, 5, 4],  # 前面
        [1, 2, 6, 5],  # 右面
        [2, 3, 7, 6],  # 后面
        [3, 0, 4, 7]  # 左面
    ]

    # 找到网格的最外层节点
    x_min, y_min, z_min = coord.min(axis=0)
    x_max, y_max, z_max = coord.max(axis=0)

    outer_node_ids = set(
        i for i, (x, y, z) in enumerate(coord)
        if x in (x_min, x_max) or y in (y_min, y_max) or z in (z_min, z_max)
    )

    # 筛选最外层面
    outer_faces = []
    for element in elements:
        for face_indices in faces_per_element:
            face = [element[i] for i in face_indices]
            if all(node in outer_node_ids for node in face):
                outer_faces.append(face)

    # 构建 PyVista 多面体网格
    faces = []
    for face in outer_faces:
        faces.append(len(face))  # 面的节点数（始终为 4）
        faces.extend(face)

    # 创建网格对象
    mesh = pv.PolyData(coord, np.array(faces))

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color="lightblue", opacity=1)
    plotter.add_axes()
    plotter.set_background("white")
    plotter.show()


def visualize_volume_elements(coord, elements, show_edges=True, color="lightblue", opacity=0.8):
    """
    使用 PyVista 可视化单元实体。
    :param coord: 节点坐标矩阵 (num_nodes, 3)
    :param elements: 单元连接矩阵 (num_elements, 8)
    :param show_edges: 是否显示单元边框
    :param color: 单元的颜色
    :param opacity: 单元透明度
    """
    # 构造 cells 和 cell_types
    cells = []
    cell_types = []
    VTK_HEXAHEDRON = 12  # PyVista 中的八节点六面体单元类型

    for element in elements:
        cells.append(8)  # 八节点单元的节点数
        cells.extend(element)  # 添加单元节点索引
        cell_types.append(VTK_HEXAHEDRON)

    # 转换为 PyVista 的 UnstructuredGrid
    cells = np.array(cells)
    cell_types = np.array(cell_types)
    grid = pv.UnstructuredGrid(cells, cell_types, coord)

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=show_edges, color=color, opacity=opacity)
    plotter.add_axes()
    plotter.set_background("white")
    plotter.view_xy()
    plotter.show()


def visualize_stress_cloud(stresses, coord_new, elements, component='von_mises'):
    # 为每个单元生成面列表
    faces = []
    for element in elements:
        faces_per_element = [
            [0, 3, 2, 1],  # 底面
            [4, 5, 6, 7],  # 顶面
            [0, 1, 5, 4],  # 前面
            [1, 2, 6, 5],  # 右面
            [2, 3, 7, 6],  # 后面
            [3, 0, 4, 7]  # 左面
        ]
        for face in faces_per_element:
            faces.append([4] + [element[i] for i in face])  # 每个面有4个节点

    # Flatten faces 数组以便 PyVista 使用
    faces = np.array(faces).flatten()

    # 创建 PolyData 网格对象
    grid = pv.PolyData(coord_new, faces)

    # 可视化应力云图
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars=stresses[:, 0], cmap="jet", show_edges=True, opacity=1)
    plotter.add_axes()
    plotter.set_background("white")
    plotter.show()


import numpy as np
import pyvista as pv


def visualize_cell_stress(coord, elements, stresses, cmap="jet", show_edges=True, opacity=1.0):
    """
    可视化单元级应力分布，使用 PyVista 的 UnstructuredGrid。

    :param coord: 节点坐标矩阵 (num_nodes, 3)
    :param elements: 单元连接矩阵 (num_elements, 8)
    :param stresses: 单元应力值 (num_elements,)
    :param cmap: 颜色映射表（默认 "jet"）
    :param show_edges: 是否显示单元边框（默认 True）
    :param opacity: 单元透明度（默认 1.0）
    """
    # VTK_HEXAHEDRON 表示八节点六面体单元的类型
    VTK_HEXAHEDRON = 12

    # 构造单元连接数组和类型数组
    cells = []
    cell_types = []
    for element in elements:
        cells.append(8)  # 每个单元有8个节点
        cells.extend(element)  # 添加单元节点索引
        cell_types.append(VTK_HEXAHEDRON)

    # 转换为 NumPy 数组
    cells = np.array(cells, dtype=np.int32)
    cell_types = np.array(cell_types, dtype=np.uint8)

    # 创建 UnstructuredGrid 对象
    grid = pv.UnstructuredGrid(cells, cell_types, coord)

    # 将单元应力值添加到网格中
    grid.cell_data["Stress"] = stresses  # 修正为 cell_data

    # 创建 PyVista 绘图器
    plotter = pv.Plotter()

    # 添加网格并设置颜色映射
    plotter.add_mesh(grid, scalars="Stress", cmap=cmap, show_edges=show_edges, opacity=opacity)

    # 添加坐标轴
    plotter.add_axes()

    # 设置背景颜色
    plotter.set_background("white")

    # 显示绘图
    plotter.show()
