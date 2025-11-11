import os
import numpy as np
import nibabel as nib
import subprocess
import tempfile
import pyvista as pv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from lapy import TriaMesh, Solver
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap
from visualize_brain import load_surface_and_activity
from brainspace.mesh import mesh_operations, mesh_io
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_OSMESA"] = "true"

def calc_eig_from_vertices_faces(tria, num_modes=20):
    """
    根据给定的表面网格 (vertices, faces) 计算特征值和特征模态。
    
    参数
    ----
    vertices : np.ndarray (N x 3)  网格顶点坐标
    faces : np.ndarray (M x 3)  网格三角面片（顶点索引）
    num_modes : int  需要计算的特征模态数量
        
    返回
    ----
    evals : np.ndarray (num_modes,)  特征值
    emodes : np.ndarray (N x num_modes)  特征模态
    """
    fem = Solver(tria)
    evals, emodes = fem.eigs(k=num_modes)
    return evals, emodes


def visualize_surface_eigenmodes(vertices, faces, emodes):
    """
    可视化脑表面的特征模态。可以通过修改col、height、width来调整子图布局/大小
    
    参数
    ----
    vertices : np.ndarray (N x 3)  网格顶点坐标
    faces : np.ndarray (M x 3)  网格三角形面片
    emodes : np.ndarray (N x K)  特征模态矩阵 (每列为一个模态)
    num_modes : int  可视化的模态数量
    multi_plot : bool {True：在一张图中显示多个模态  False：每个模态生成一个单独的 HTML 文件} 
    output_dir : str  保存输出文件的目录
    """
    num_modes = min(9, emodes.shape[1])
    multi_plot = True
    output_dir="/home/wmy/work/geometry/eigenmodes_vis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果是多模态模式，先创建子图
    if multi_plot:
        cols = 3
        rows = (num_modes + cols - 1) // cols
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{'type': 'scene'}] * cols] * rows,
            subplot_titles=[f"Eigenmode {i+1}" for i in range(num_modes)]
        )
    
    # 公共绘制逻辑
    min_val, max_val = emodes.min(), emodes.max()
    for i in range(num_modes):
        mode_values = emodes[:, i]
        mesh_trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=mode_values,
            colorscale=[
                    [0.0, 'blue'],   # 最小值为蓝色
                    [0.5, 'white'],  # 中间值为白色
                    [1.0, 'red']     # 最大值为红色
            ],
            cmin=min_val,
            cmax=max_val,
            showscale=(not multi_plot),  # 多子图时关闭多余颜色条
            opacity=0.9
        )

        if multi_plot:
            row, col = divmod(i, cols)
            fig.add_trace(mesh_trace, row=row + 1, col=col + 1)
        else:
            # 单模态输出
            single_fig = go.Figure([mesh_trace])
            single_fig.update_layout(
                scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                title=f"Eigenmode {i + 1}"
            )
            output_file = os.path.join(output_dir, f"eigenmode_{i + 1}.html")
            single_fig.write_html(output_file)
            print(f"已保存单模态文件：{output_file}")

    # 多模态输出
    if multi_plot:
        fig.update_layout(
            height=600 * rows,
            width=600 * cols,
            title_text="Eigenmodes Visualization"
        )
        output_file = os.path.join(output_dir, "eigenmodes_multi.html")
        fig.write_html(output_file)
        print(f"已保存多模态文件：{output_file}")
        return fig
    
def bluewhitered_cmap():
    """Create a MATLAB-style blue-white-red colormap."""
    n = 256
    cdict = {'red':   [[0.0, 0.0, 0.0],
                       [0.5, 1.0, 1.0],
                       [1.0, 1.0, 1.0]],
             'green': [[0.0, 0.0, 0.0],
                       [0.5, 1.0, 1.0],
                       [1.0, 0.0, 0.0]],
             'blue':  [[0.0, 1.0, 1.0],
                       [0.5, 1.0, 1.0],
                       [1.0, 0.0, 0.0]]}
    return LinearSegmentedColormap('bluewhitered', cdict, N=n)

def draw_surface_bluewhitered_dull_save(surface_to_plot, data_to_plot, hemisphere='lh',
                                        medial_wall=None, with_medial=False, save_path="surface.png"):
    """
    Render surface data with blue-white-red colormap (Pang 2022 style)
    and save directly to file (headless, off-screen).
    """

    # === 数据准备 ===
    vertices = np.asarray(surface_to_plot['vertices'])
    faces = np.asarray(surface_to_plot['faces'])
    data_to_plot = np.asarray(data_to_plot).flatten()
    cmap = bluewhitered_cmap()
    vmin, vmax = np.min(data_to_plot), np.max(data_to_plot)
    if vmax <= 0:
        vmax = 0.01

    if medial_wall is not None:
        data_to_plot[medial_wall] = np.min(data_to_plot) * 1.1

    # === 生成 PyVista 网格 ===
    faces_vtk = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(vertices, faces_vtk)

    # === 创建离屏渲染器 ===
    if with_medial:
        plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 600))
    else:
        plotter = pv.Plotter(off_screen=True, window_size=(800, 800))

    # === 添加渲染 ===
    def add_view(p, mesh, look_dir, data_to_plot, vmin, vmax, cmap):
        """为每个 subplot 添加独立的 mesh 视图"""
        mesh_copy = mesh.copy()   # ✅ 必须复制，否则多个 subplot 会共享渲染状态

        p.add_mesh(
            mesh_copy, scalars=data_to_plot, cmap=cmap, clim=[vmin, vmax],
            show_edges=False, lighting=True, smooth_shading=True,
            diffuse=0.8, specular=0.05
        )
        p.enable_eye_dome_lighting()
        p.background_color = "white"
        p.remove_scalar_bar()

        # === 设置相机 ===
        center = np.array(mesh_copy.center)
        dist   = float(mesh_copy.length) * 1.8
        pos    = center + look_dir * dist
        viewup = (0.0, 0.0, 1.0)

        p.camera.position    = tuple(pos)
        p.camera.focal_point = tuple(center)
        p.camera.up          = viewup
        p.reset_camera_clipping_range()


    if with_medial:
        plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 600))
    else:
        plotter = pv.Plotter(off_screen=True, window_size=(800, 800))

    # === 左外侧 ===
    plotter.subplot(0, 0)
    if hemisphere.lower() == "rh":
        add_view(plotter, mesh, np.array([-1.0, 0.0, 0.0]), data_to_plot, vmin, vmax, cmap)
    else:
        add_view(plotter, mesh, np.array([1.0, 0.0, 0.0]), data_to_plot, vmin, vmax, cmap)

    # === 右内侧 ===
    if with_medial:
        plotter.subplot(0, 1)
        if hemisphere.lower() == "rh":
            add_view(plotter, mesh, np.array([1.0, 0.0, 0.0]), data_to_plot, vmin, vmax, cmap)
        else:
            add_view(plotter, mesh, np.array([-1.0, 0.0, 0.0]), data_to_plot, vmin, vmax, cmap)


    plotter.show(screenshot=save_path)
    print(f"[SAVED] surface image -> {save_path}")



def cal_and_visualize(resting_data_dir, struct_data_dir, sub, masked):
    
    # 加载数据和可视化表面
    vertices, faces, activity_timeseries = load_surface_and_activity(resting_data_dir, struct_data_dir)
    
    if masked:
        # 去除medial wall
        mask_path = os.path.join(
            struct_data_dir,
            f"MNINonLinear/Native/{sub}.L.atlasroi.native.shape.gii"
        )
        mask_gii = nib.load(mask_path)
        mask_data = mask_gii.darrays[0].data
        mask_bool = mask_data > 0  # True = cortex, False = medial wall

        print(f"Loaded mask: {mask_path}")
        #print(f"Mask cortex vertex count = {mask_data.sum()} / {len(mask_data)}")    有个.0
        print(f"Mask cortex vertex count = {mask_bool.sum()} / {len(mask_bool)}")

        surf_gii_path = os.path.join(struct_data_dir, f"MNINonLinear/Native/{sub}.L.white.native.surf.gii")
        vtk_tmp = tempfile.NamedTemporaryFile(suffix=".vtk", delete=False).name

        # 调用 FreeSurfer 自带的 mris_convert
        subprocess.run(["bash", "-c", f"source $FREESURFER_HOME/SetUpFreeSurfer.sh && mris_convert {surf_gii_path} {vtk_tmp}"], check=True)

        # 读入 .vtk 文件
        tria = mesh_io.read_surface(vtk_tmp)
        # 裁剪得到新的 surface
        surface_cut = mesh_operations.mask_points(tria, mask_bool)

        tria = TriaMesh.read_vtk(vtk_tmp)
        tria.v = surface_cut.Points
        tria.t = np.reshape(surface_cut.Polygons, [surface_cut.n_cells, 4])[:, 1:4]
        print(f"After masking: {tria.v.shape[0]} vertices, {tria.t.shape[0]} faces")

        print("surface_cut.Points shape:", surface_cut.Points.shape)
        print("surface_cut.Polygons shape:", surface_cut.Polygons.shape)
        print("surface_cut.n_cells:", surface_cut.n_cells)
        print("Polygons[:12]:", surface_cut.Polygons[:12])
        print("Max polygon index:", surface_cut.Polygons.max())

        evals, emodes = calc_eig_from_vertices_faces(tria, num_modes=200)
        np.savetxt("/home/wmy/work/geometry/eigenvalues_1.txt", evals)
        np.savetxt("/home/wmy/work/geometry/eigenmodes_1.txt", emodes)

        visualize_surface_eigenmodes(tria.v, tria.t, emodes)

        for i in range(9):
            draw_surface_bluewhitered_dull_save(
                surface_to_plot={'vertices': tria.v, 'faces': tria.t},
                data_to_plot=emodes[:, i],
                hemisphere='lh',
                medial_wall=None,
                with_medial=True,
                save_path=f"/home/wmy/work/geometry/result/mode_{i+1:02d}_lh.png"
            )

    else:
        tria = TriaMesh(vertices, faces)
        evals, emodes = calc_eig_from_vertices_faces(tria, num_modes=200)
        np.savetxt("/home/wmy/work/geometry/eigenvalues.txt", evals)
        np.savetxt("/home/wmy/work/geometry/eigenmodes.txt", emodes)

        visualize_surface_eigenmodes(vertices, faces, emodes)
    

def main():
    # 设置数据目录
    #BASE_PATH = '/mnt/'
    BASE_PATH = '/home/wmy/Documents/'
    sub = '100307'
    resting_data_dir = os.path.join(BASE_PATH, 'REST1', sub)
    struct_data_dir = os.path.join(BASE_PATH, 'Structure', sub)


    cal_and_visualize(resting_data_dir, struct_data_dir, sub, masked = True)

if __name__ == "__main__":
    main()