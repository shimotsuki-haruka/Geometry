import numpy as np
import nibabel as nib
import os
import subprocess
import plotly.graph_objects as go
import trimesh
import tempfile
import meshio

from skimage import measure
from lapy import Solver, TetMesh
from scipy.interpolate import griddata
from plotly.subplots import make_subplots
from visualize_brain import extract_nuclei_mesh, visualize_mask_html, add_axes_RPD
from scipy.ndimage import gaussian_filter,binary_dilation

def mask_to_tetmesh_gmsh(mask, mask_img, max_cell_size=2.0, gmsh_path="gmsh"):
    #tmp_dir = tempfile.mkdtemp()
    data_dir = "/home/wmy/geometry/result/volume_modes"
    stl_surface = os.path.join(data_dir, "surface.stl")
    geo_file = os.path.join(data_dir, "surface.geo")
    tetra_file = os.path.join(data_dir, "surface.tetra.vtk")

    # 1. marching cubes\
    mask_smooth = gaussian_filter(mask.astype(float), sigma=0.5)
    verts, faces, _, _ = measure.marching_cubes(mask_smooth, level=0.5)
    verts = verts.astype(np.float64)
    faces = faces.astype(np.int32)
    verts = nib.affines.apply_affine(mask_img.affine, verts)

    visualize_mask_html(
    mask_processed=mask_smooth,
    affine=mask_img.affine,
    out_html="nuclei_closed_smooth.html",
    original_mask=None,   # 若不想对比，可传 None
    title="Original vs Closed (dilation+erosion, iters=2)"
    )

    # 2. 导出 STL 文件（避免 VTK 导出问题）
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(stl_surface)

    # 3. 写 geo 文件
    geo_content = f"""
Mesh.Algorithm3D=4;
Mesh.Optimize=1;
Mesh.OptimizeNetgen=1;
Merge "{stl_surface}";
Surface Loop(1) = {{1}};
Volume(1) = {{1}};
Physical Volume(1) = {{1}};
Mesh.CharacteristicLengthMax = {max_cell_size};
"""
    with open(geo_file, "w") as f:
        f.write(geo_content)

    # 4. 调用 gmsh
    cmd = [gmsh_path, "-3", "-o", tetra_file, geo_file]
    subprocess.run(cmd, check=True)

    # 5. 读取网格
    mesh_data = meshio.read(tetra_file)
    nodes = mesh_data.points
    tets = mesh_data.get_cells_type("tetra")

    return TetMesh(nodes, tets)

def merge_tet_meshes(tet_list):
    """
    手动合并多个 TetMesh 对象。

    参数
    ----
    tet_list : list of TetMesh
        每个对象都有 v (顶点) 和 t (四面体)

    返回
    ----
    TetMesh
        合并后的单一四面体网格
    """
    all_verts = []
    all_tets = []
    v_offset = 0

    for tet in tet_list:
        all_verts.append(tet.v)
        all_tets.append(tet.t + v_offset)
        v_offset += tet.v.shape[0]

    merged_verts = np.vstack(all_verts)
    merged_tets = np.vstack(all_tets)

    return TetMesh(merged_verts, merged_tets)


def compute_volume_eigenmodes_from_mask(mask, subcortical_img, num_modes=10):
    """
    使用 pygalmesh 从 mask 生成四面体网格并计算本征模。
    """
    tet = mask_to_tetmesh_gmsh(mask, subcortical_img)
    fem = Solver(tet)
    evals, emodes = fem.eigs(k=num_modes)
    print(f"完成: 计算 {num_modes} 个本征模, 节点数 = {tet.v.shape[0]}")
    return evals, emodes, tet

def compute_merged_nuclei_eigenmodes(struct_data_dir, nuclei_dict, num_modes=10):
    """
    分别为多个脑区生成 tetrahedral 网格并合并，再计算本征模。
    
    参数
    ----
    struct_data_dir : str
        解剖数据目录，必须包含 wmparc.nii.gz
    nuclei_dict : dict
        键为名称，值为 FreeSurfer label 列表，如 {"左尾状核": [11], "左壳核": [12]}
    num_modes : int
        要计算的特征模数量
    
    返回
    ----
    evals, emodes, tet : np.ndarray, np.ndarray, TetMesh
    """
    from lapy import TetMesh, Solver

    tet_list = []
    
    for name, labels in nuclei_dict.items():
        print(f"处理脑区：{name} 标签：{labels}")
        verts, faces, subcortical_img, mask = extract_nuclei_mesh(struct_data_dir, labels)
        if np.count_nonzero(mask) == 0:
            print(f"跳过 {name}：无有效体素")
            continue
        try:
            tet = mask_to_tetmesh_gmsh(mask, subcortical_img)
            print(f"{name} 网格节点数: {tet.v.shape[0]} 元素数: {tet.t.shape[0]}")
            tet_list.append(tet)
        except Exception as e:
            print(f"{name} 网格生成失败: {e}")
    
    if len(tet_list) == 0:
        raise RuntimeError("未成功生成任何脑区的 tetra 网格。")

    if len(tet_list) == 1:
        tet_merged = tet_list[0]
    else:
        print("合并多个脑区网格...")
        tet_merged = merge_tet_meshes(tet_list)

    print(f"合并后总节点数: {tet_merged.v.shape[0]} 元素数: {tet_merged.t.shape[0]}")
    
    # 求解 Laplacian eigenmodes
    fem = Solver(tet_merged)
    evals, emodes = fem.eigs(k=num_modes)
    print(f"完成：计算 {num_modes} 个本征模")

    return evals, emodes, tet_merged

def visualize_volume_eigenmodes_mesh(verts, faces, emodes, max_modes=6, output_dir="/home/wmy/work/geometry/volume_eigenmodes_vis"):
    """
    可视化体积本征模（将每个模态保存为 HTML）。
    
    参数
    ----
    verts : np.ndarray
        网格顶点坐标 (N x 3)
    faces : np.ndarray
        网格三角面片 (M x 3)
    emodes : np.ndarray
        本征模矩阵 (N x num_modes)
    max_modes : int
        可视化的模态数量
    output_dir : str
        输出 HTML 文件的目录
    """
    os.makedirs(output_dir, exist_ok=True)
    num_modes = min(max_modes, emodes.shape[1])
    smin = emodes.min()
    smax = emodes.max()
    for i in range(num_modes):
        mode_values = emodes[:, i]
        mesh_trace = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=mode_values,
            colorscale=[[0.0, 'blue'], [0.5, 'white'], [1.0, 'red']],
            cmin=smin,
            cmax=smax,
            showscale=True,
            opacity=0.9
        )
        fig = go.Figure([mesh_trace])
        fig.update_layout(
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            title=f"Volume Eigenmode {i + 1}"
        )
        out_file = os.path.join(output_dir, f"volume_eigenmode_{i + 1}.html")
        add_axes_RPD(fig, verts, scale_ratio=0.18, margin_ratio=0.06)
        fig.write_html(out_file)
        print(f"已保存：{out_file}")

###mutiple test

def visualize_multiple_nuclei_eigenmodes_mesh(nuclei_mesh_data, mode_index=0, output_file="all_nuclei_mode.html"):
    """
    同时可视化多个脑区在某一模态下的本征模。

    参数
    ----
    nuclei_mesh_data : dict
        字典，key 为核团名称，value 为 (verts, faces, emodes)。
        - verts: np.ndarray (N x 3)
        - faces: np.ndarray (M x 3)
        - emodes: np.ndarray (N x num_modes)
    mode_index : int
        显示的模态序号 (从0开始)。
    output_file : str
        输出 HTML 文件路径。
    """
    fig = go.Figure()

    # 为每个脑区生成一个 Mesh3d Trace
    color_scales = [
        [[0.0, 'blue'], [0.5, 'white'], [1.0, 'red']],
    ]
    
    for i, (name, (verts, faces, emodes)) in enumerate(nuclei_mesh_data.items()):
        if mode_index >= emodes.shape[1]:
            print(f"警告：{name} 的模态数不足，跳过 mode_index={mode_index}")
            continue
        smin = emodes.min()
        smax = emodes.max()
        mode_values = emodes[:, mode_index]
        color_scale = color_scales[i % len(color_scales)]
        mesh_trace = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=mode_values,
            colorscale=color_scale,
            cmin=smin,
            cmax=smax,
            showscale=True,
            name=name,
            opacity=0.8
        )
        fig.add_trace(mesh_trace)

    fig.update_layout(
        scene=dict(xaxis=dict(visible=False),
                   yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
        title=f"Multiple Nuclei Eigenmode {mode_index + 1}"
    )
    fig.write_html(output_file)
    print(f"已保存：{output_file}")

def visualize_all_nuclei_eigenmodes_combined(struct_data_dir, max_modes=6, num_modes=10, output_dir="/home/wmy/work/geometry/volume_eigenmodes_vis"):
    """
    将所有脑区的本征模态依次可视化，每个模态输出一个 HTML 文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    nuclei_dict = {
        "左尾状核": [11],
        "左壳核": [12],
    }

    nuclei_mesh_data = {}

    # 先计算每个脑区的 emodes
    for name, labels in nuclei_dict.items():
        print(f"正在处理{name} ...")
        verts, faces, subcortical_img, mask = extract_nuclei_mesh(struct_data_dir, labels)
        if np.count_nonzero(mask) == 0:
            print(f"跳过 {name}：无有效体素")
            continue

        evals, emodes, tet = compute_volume_eigenmodes_from_mask(mask, subcortical_img, num_modes=num_modes)
        tetfaces = tet.boundary_tria().t
        nuclei_mesh_data[name] = (tet.v, tetfaces, emodes)

    # 逐模态生成文件
    num_modes_to_visualize = min(max_modes, num_modes)
    for mode_index in range(num_modes_to_visualize):
        out_file = os.path.join(output_dir, f"all_nuclei_mode_{mode_index + 1}.html")
        visualize_multiple_nuclei_eigenmodes_mesh(nuclei_mesh_data, mode_index=mode_index, output_file=out_file)


def main():
    # 设置数据目录
    #BASE_PATH = '/mnt/'
    BASE_PATH = '/home/wmy/Documents/'
    sub = '100307'
    resting_data_dir = os.path.join(BASE_PATH, 'Resting_1', sub)
    struct_data_dir = os.path.join(BASE_PATH, 'Structure', sub)
    out_data_dir = "/home/wmy/work/geometry/volume_modes/"
    #help(pygalmesh.generate_volume_mesh_from_surface_mesh)
    #print(dir(TetMesh))

    verts, faces, subcortical_img, mask = extract_nuclei_mesh(struct_data_dir, [11, 12])

    '''
    纹状体：
        11: Left-Caudate尾状核
        12: Left-Putamen壳核
        26: Left-Accumbens伏隔核
    '''

    #mask = binary_dilation(mask, iterations=2)

    evals, emodes, tet = compute_volume_eigenmodes_from_mask(mask, subcortical_img, num_modes=10)

    tetfaces = tet.boundary_tria().t
    visualize_volume_eigenmodes_mesh(tet.v, tetfaces, emodes, max_modes=6)

    #visualize_all_nuclei_eigenmodes_combined(struct_data_dir, max_modes=6, num_modes=10, output_dir="/home/wmy/work/geometry/volume_eigenmodes_vis")

if __name__ == "__main__":
    main()

#wrong smooth and projecting
'''def project_emode_to_volume(tet, emode_values, mask_img, smooth_sigma=1.0):
    """
    将四面体网格的本征模值投影到mask对应的体积空间。
    参数
    ----
    tet : lapy.TetMesh
        四面体网格对象
    emode_values : np.ndarray
        对应每个网格顶点的模态值 (N,)
    mask_img : nib.Nifti1Image
        原始核团掩码
    smooth_sigma : float
        高斯平滑系数
    返回
    ----
    emode_volume : 3D numpy array
        投影到体积空间的模态值
    """
    mask_data = mask_img.get_fdata()
    shape = mask_data.shape
    affine = mask_img.affine

    print("TetMesh vertex range:", np.min(tet.v, axis=0), np.max(tet.v, axis=0))
    print("Mask voxel-to-world origin:", mask_img.affine)
    print("Mask shape:", mask_img.shape)
    
    # 获取mask内体素坐标（世界坐标）
    voxel_coords = np.array(np.nonzero(mask_data)).T
    world_coords = nib.affines.apply_affine(affine, voxel_coords)

    # 插值
    interpolated = griddata(tet.v, emode_values, world_coords, method='linear', fill_value=0)

    emode_volume = np.zeros(shape)
    emode_volume[mask_data > 0] = interpolated

    # 平滑
    if smooth_sigma > 0:
        emode_volume = gaussian_filter(emode_volume, sigma=smooth_sigma)

    return emode_volume


def visualize_emode_on_volume_mesh(emode_volume, affine, output_file="emode_surface.html"):
    """
    将投影后的本征模体积数据可视化为表面Mesh3d。
    """

    print("emode_volume min:", np.min(emode_volume))
    print("emode_volume max:", np.max(emode_volume))
    print("Non-zero voxels:", np.count_nonzero(emode_volume))

    verts, faces, _, _ = measure.marching_cubes(emode_volume, level = (emode_volume.min() + emode_volume.max()) / 2)
    verts_world = nib.affines.apply_affine(affine, verts)

    mesh_trace = go.Mesh3d(
        x=verts_world[:, 0],
        y=verts_world[:, 1],
        z=verts_world[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=emode_volume[tuple(verts.astype(int).T)],
        colorscale=[[0, 'blue'], [0.5, 'white'], [1, 'red']],
        opacity=0.9,
        showscale=True
    )
    fig = go.Figure([mesh_trace])
    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)),
                      title="Projected Volume Eigenmode")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.write_html(output_file)
    print(f"已保存：{output_file}")
'''
#main():
'''print("emodes min:", emodes[:,0].min(), "max:", emodes[:,0].max())
emode_volume = project_emode_to_volume(tet, emodes[:, 0], subcortical_img)
visualize_emode_on_volume_mesh(emode_volume, subcortical_img.affine, output_file="/home/wmy/work/geometry/volume_modes/projected_mode_1.html")'''