import os
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lapy import TetMesh, Solver
from volume_eigenmode import visualize_volume_eigenmodes_mesh
from skimage import measure
from skimage.measure import marching_cubes
from nilearn import plotting, image
from skimage import measure
from nibabel.affines import apply_affine


def view_striatum_middle(wmparc_path, t1w_path=None, output_dir="."):
    """
    查看 T1w 中壳核与尾状核之间的部分（伏隔核 accumbens）。
    
    参数
    ----
    wmparc_path : str
        HCP/FreeSurfer 的 wmparc.nii.gz 路径
    t1w_path : str or None
        T1w.nii.gz 路径；若 None 会尝试在 wmparc 同目录自动查找
    output_dir : str
        输出文件保存目录

    返回
    ----
    result : dict
        包含统计结果与输出文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 自动寻找 T1w
    if t1w_path is None:
        candidates = [
            "T1w_restore.nii.gz",
            "T1w_acpc_dc_restore.nii.gz",
            "T1w_acpc_dc.nii.gz",
            "T1w.nii.gz",
        ]
        mnidir = os.path.dirname(wmparc_path)
        for c in candidates:
            p = os.path.join(mnidir, c)
            if os.path.exists(p):
                t1w_path = p
                break

    # 读取影像
    seg_img = nib.load(wmparc_path)
    seg = seg_img.get_fdata().astype(np.int32)
    aff = seg_img.affine
    t1w_img = nib.load(t1w_path) if t1w_path else None

    # 标签定义
    LAB_ACC_L, LAB_ACC_R = 26, 58
    LAB_PUT, LAB_CAU = [12, 51], [11, 50]

    acc_mask = np.isin(seg, [LAB_ACC_L, LAB_ACC_R]).astype(np.uint8)
    put_mask = np.isin(seg, LAB_PUT).astype(np.uint8)
    cau_mask = np.isin(seg, LAB_CAU).astype(np.uint8)

    acc_img = nib.Nifti1Image(acc_mask, aff)
    put_img = nib.Nifti1Image(put_mask, aff)
    cau_img = nib.Nifti1Image(cau_mask, aff)

    # 体素统计
    voxel_vol = abs(np.linalg.det(aff[:3, :3]))
    cnt_L, cnt_R = np.count_nonzero(seg == LAB_ACC_L), np.count_nonzero(seg == LAB_ACC_R)
    vol_L, vol_R = cnt_L * voxel_vol, cnt_R * voxel_vol

    # 质心
    if acc_mask.any():
        vz = np.column_stack(np.nonzero(acc_mask))
        mm = apply_affine(aff, vz)
        cut_coords = tuple(np.round(mm.mean(axis=0), 1))
    else:
        cut_coords = None

    # 绘制叠加图
    out_png = os.path.join(output_dir, "acc_on_T1w.png")
    disp = plotting.plot_anat(
        t1w_img if t1w_img else seg_img,
        display_mode="ortho",
        cut_coords=cut_coords,
        title="Accumbens (between Caudate & Putamen)"
    )
    disp.add_contours(acc_img, levels=[0.5], linewidths=2)
    disp.add_contours(put_img, levels=[0.5], linewidths=1)
    disp.add_contours(cau_img, levels=[0.5], linewidths=1)
    disp.savefig(out_png, dpi=200)

    # 交互式 3D
    out_html = os.path.join(output_dir, "acc_3d_view.html")
    acc_smooth = image.smooth_img(acc_img, fwhm=1.5)
    view = plotting.view_img(acc_smooth, threshold=0.5, black_bg=True)
    view.save_as_html(out_html)

    # 网格导出
    out_obj = os.path.join(output_dir, "accumbens_mesh.obj")
    try:
        verts, faces, _, _ = measure.marching_cubes(acc_mask.astype(bool), level=0.5)
        hom = np.c_[verts, np.ones(len(verts))]
        verts_mm = (aff @ hom.T).T[:, :3]
        with open(out_obj, "w") as f:
            for v in verts_mm:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for tri in faces + 1:
                f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")
    except Exception as e:
        out_obj = None
        print(f"[WARN] 网格导出失败: {e}")

    return {
        "count_L": int(cnt_L),
        "count_R": int(cnt_R),
        "volume_L_mm3": float(vol_L),
        "volume_R_mm3": float(vol_R),
        "out_png": out_png,
        "out_html": out_html,
        "out_obj": out_obj,
    }

def vis_tetra():
    base_dir = "/home/wmy/work/geometry/data"
    
    # 1. 加载已有的 tetrahedral 网格
    vtk_path = os.path.join(base_dir, "hcp_striatum-lh_thr25.nii.gz.tetra.vtk")
    tet = TetMesh.read_vtk(vtk_path)
    
    # 2. 计算前 10 个本征模态
    solver = Solver(tet)
    evals, emodes = solver.eigs(k=10)
    
    # 3. 提取顶点和边界三角面
    verts = tet.v
    faces = tet.boundary_tria().t

    # 4. 可视化前 6 个模态
    visualize_volume_eigenmodes_mesh(verts, faces, emodes, max_modes=6)

def vis_mask():
    nii_path = "/home/wmy/work/geometry/data/hcp_striatum-lh_thr25.nii.gz"
    out_html = "striatum_lh_mesh.html"
    level = 0.5         # marching cubes 等值面（掩膜0/1，取0.5即可）
    min_voxels = 100    # 过滤过小的连通块（防噪点）；设为0则不滤

    # === 读取掩膜 ===
    img = nib.load(nii_path)
    data = img.get_fdata()
    mask = (data > 0).astype(np.uint8)

    # （可选）简单去噪：保留最大连通块
    if min_voxels > 0:
        try:    
            from scipy.ndimage import label
            labeled, nlab = label(mask)
            if nlab > 1:
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0
                keep = sizes.argmax()
                mask = (labeled == keep).astype(np.uint8)
        except Exception:
            pass  # 没装scipy也没关系

    # === marching cubes（在体素坐标系）===
    # 注意：这里先不设置 spacing；我们用仿射矩阵把顶点从体素坐标变换到世界坐标
    verts_vox, faces, normals, values = marching_cubes(
        volume=mask, level=level, allow_degenerate=False
    )

    # === 体素坐标 -> 世界坐标（mm）===
    aff = img.affine  # 4x4
    verts_mm = nib.affines.apply_affine(aff, verts_vox)

    # === 画 Mesh 并保存 HTML ===
    i, j, k = faces.T  # plotly 需要三角面索引
    mesh = go.Mesh3d(
        x=verts_mm[:, 0], y=verts_mm[:, 1], z=verts_mm[:, 2],
        i=i, j=j, k=k,
        opacity=1.0,
        flatshading=True,
        lighting=dict(ambient=0.3, diffuse=0.8, specular=0.2, roughness=0.6),
        lightposition=dict(x=100, y=200, z=300),
        name="Striatum (LH)"
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title="Left Striatum (marching cubes)",
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    print(f"Saved: {os.path.abspath(out_html)}")


def _load_LH_native_timeseries_and_surface(resting_dir: str, struct_dir: str, sub: str):
    """
    返回：
      Y_TxV : (T, V_all)  —— LH native.func.gii 的全部顶点时序（不去 medial wall）
      V     : (V_all, 3)  —— LH white.native.surf.gii 顶点
      F     : (F_all, 3)  —— LH white.native.surf.gii 面
      m     : (V_all,)    —— atlasroi 掩膜(仅打印校验，不用于裁剪)
    """
    # 时序（dense vertex-wise, 不裁剪）
    func_gii = os.path.join(resting_dir,
        "MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.L.native.func.gii")
    g = nib.load(func_gii)
    Y_TxV = np.vstack([arr.data for arr in g.darrays]).astype(np.float64)  # (T,V_all)

    # 白质表面（你也可以换成 midthickness.native.surf.gii，视觉更接近皮层中层）
    surf_gii = os.path.join(struct_dir, f"MNINonLinear/Native/{sub}.L.white.native.surf.gii")
    s = nib.load(surf_gii)
    V = s.darrays[0].data.astype(np.float64)   # (V_all,3)
    F = s.darrays[1].data.astype(np.int32)     # (F_all,3)

    # atlasroi（1=有效皮层；0=medial wall 或非皮层）
    mask_gii = os.path.join(struct_dir, f"MNINonLinear/Native/{sub}.L.atlasroi.native.shape.gii")
    m = nib.load(mask_gii).darrays[0].data.astype(bool)  # (V_all,)

    # 一致性校验
    print(f"[TEST] func 顶点数 (V_all)   = {Y_TxV.shape[1]}")
    print(f"[TEST] surface 顶点数        = {V.shape[0]}")
    print(f"[TEST] 三角形数 (faces)      = {F.shape[0]}")
    print(f"[TEST] atlasroi 皮层顶点数   = {int(m.sum())} / {m.size} ({m.sum()/m.size*100:.2f}%)")
    if Y_TxV.shape[1] != V.shape[0] or V.shape[0] != m.size:
        raise ValueError("func 顶点数、surface 顶点数、atlasroi 长度不一致。")

    return Y_TxV, V, F, m

def _faces_split_by_medial_wall(F: np.ndarray, mask_cortex: np.ndarray):
    """
    输入：
      F: (F_all,3) 三角面顶点索引
      mask_cortex: (V_all,) 布尔向量，True=有效皮层；False=medial wall
    返回：
      F_cortex: 全部顶点都在皮层的三角面
      F_medial: 至少有1个顶点在 medial wall 的三角面
    """
    # 对每个面，检查三个顶点是否都在皮层
    all_cortex = mask_cortex[F].all(axis=1)
    F_cortex = F[all_cortex]
    F_medial = F[~all_cortex]
    return F_cortex, F_medial

def _set_axes_equal(ax):
    """让3D坐标轴比例一致，避免脑表面看起来被拉伸。"""
    x, y, z = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xrange = x[1]-x[0]; yrange = y[1]-y[0]; zrange = z[1]-z[0]
    max_range = max([xrange, yrange, zrange])
    xmid = np.mean(x); ymid = np.mean(y); zmid = np.mean(z)
    ax.set_xlim3d([xmid - max_range/2, xmid + max_range/2])
    ax.set_ylim3d([ymid - max_range/2, ymid + max_range/2])
    ax.set_zlim3d([zmid - max_range/2, zmid + max_range/2])

def _faces_split_by_medial_wall(F: np.ndarray, cortex_mask: np.ndarray):
    """
    F: (F_all,3) int32 三角面顶点索引
    cortex_mask: (V_all,) bool  True=有效皮层；False=medial wall
    返回: F_cortex, F_medial
    """
    all_cortex = cortex_mask[F].all(axis=1)
    F_cortex = F[all_cortex]
    F_medial = F[~all_cortex]
    return F_cortex, F_medial

def plot_LH_native_with_medial_wall_interactive(
        V: np.ndarray,
        F: np.ndarray,
        mask_atlasroi: np.ndarray,
        title: str = "LH native (white) — interactive, medial wall highlighted",
        save_html: str = None):
    """
    用 Plotly 交互式 3D 展示左半球 native 表面：
      - “全部面”(F_all)：浅灰底
      - “皮层 faces”(F_cortex)：稍深灰
      - “medial faces”(F_medial)：高亮(红/橙)

    参数
    ----
    V: (V_all,3) float 顶点坐标
    F: (F_all,3) int   三角面顶点索引（0-based）
    mask_atlasroi: (V_all,) bool   True=有效皮层；False=medial wall
    title: 图题
    save_html: 若提供路径，将保存为独立可交互 HTML
    """
    V = V.astype(np.float64, copy=False)
    F = F.astype(np.int32, copy=False)
    cortex_mask = mask_atlasroi.astype(bool, copy=False)

    # 拆分三角面
    F_cortex, F_medial = _faces_split_by_medial_wall(F, cortex_mask)

    # 统计信息
    nV = V.shape[0]
    nF = F.shape[0]
    n_ctxV = int(cortex_mask.sum()); n_mwV = nV - n_ctxV
    n_ctxF = F_cortex.shape[0];      n_mwF = F_medial.shape[0]

    # 底层：全部 faces（浅灰）
    mesh_all = go.Mesh3d(
        x=V[:,0], y=V[:,1], z=V[:,2],
        i=F[:,0], j=F[:,1], k=F[:,2],
        color="lightgray", opacity=0.20,
        name=f"All faces (V: {nV}, F: {nF})",
        showscale=False,
        hoverinfo="skip",
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.05, roughness=0.9),
        flatshading=True,
        visible=True
    )

    # 纯皮层 faces（深一点灰）
    mesh_ctx = go.Mesh3d(
        x=V[:,0], y=V[:,1], z=V[:,2],
        i=F_cortex[:,0] if len(F_cortex) else None,
        j=F_cortex[:,1] if len(F_cortex) else None,
        k=F_cortex[:,2] if len(F_cortex) else None,
        color="gray", opacity=0.55,
        name=f"Cortex faces (V: {n_ctxV}, F: {n_ctxF})",
        showscale=False,
        hoverinfo="skip",
        lighting=dict(ambient=0.55, diffuse=0.9, specular=0.10, roughness=0.85),
        flatshading=True,
        visible=True
    )

    # medial faces（高亮：红/橙）
    mesh_med = go.Mesh3d(
        x=V[:,0], y=V[:,1], z=V[:,2],
        i=F_medial[:,0] if len(F_medial) else None,
        j=F_medial[:,1] if len(F_medial) else None,
        k=F_medial[:,2] if len(F_medial) else None,
        color="orangered", opacity=0.85,
        name=f"Medial faces (V: {n_mwV}, F: {n_mwF})",
        showscale=False,
        hoverinfo="skip",
        lighting=dict(ambient=0.45, diffuse=0.9, specular=0.15, roughness=0.8),
        flatshading=True,
        visible=True
    )

    # 视角预设
    cam_outer = dict(eye=dict(x=1.75, y=-0.4, z=0.4))
    cam_inner = dict(eye=dict(x=-1.75, y=0.4, z=0.4))
    cam_top   = dict(eye=dict(x=0.0, y=0.0, z=2.0))
    cam_bottom= dict(eye=dict(x=0.0, y=0.0, z=-2.0))
    cam_front = dict(eye=dict(x=0.0, y=2.0, z=0.0))
    cam_back  = dict(eye=dict(x=0.0, y=-2.0, z=0.0))

    fig = go.Figure(data=[mesh_all, mesh_ctx, mesh_med])

    # 工具栏与布局
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02),
        updatemenus=[
            # 显隐切换
            dict(
                type="buttons",
                direction="right",
                x=0.02, y=1.08, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="显示全部", method="update",
                         args=[{"visible": [True, True, True]}]),
                    dict(label="只看皮层", method="update",
                         args=[{"visible": [False, True, False]}]),
                    dict(label="只看Medial", method="update",
                         args=[{"visible": [False, False, True]}]),
                ]
            ),
            # 视角切换
            dict(
                type="buttons",
                direction="right",
                x=0.02, y=1.02, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="外侧",   method="relayout", args=[{"scene.camera": cam_outer}]),
                    dict(label="内侧",   method="relayout", args=[{"scene.camera": cam_inner}]),
                    dict(label="顶视",   method="relayout", args=[{"scene.camera": cam_top}]),
                    dict(label="底视",   method="relayout", args=[{"scene.camera": cam_bottom}]),
                    dict(label="前视",   method="relayout", args=[{"scene.camera": cam_front}]),
                    dict(label="后视",   method="relayout", args=[{"scene.camera": cam_back}]),
                ]
            ),
        ],
        # 右上角还有 Plotly 自带的旋转/缩放/重置/保存PNG工具
    )

    # 初始视角：外侧
    fig.update_layout(scene_camera=cam_outer)

    # 保存为 HTML（单文件，便于分享/汇报）
    if save_html is not None:
        os.makedirs(os.path.dirname(save_html) or ".", exist_ok=True)
        fig.write_html(save_html, include_plotlyjs="cdn", full_html=True)
        print(f"[SAVE] interactive HTML -> {save_html}")

    # 打印统计
    print(f"[INFO] Vertices: cortex={n_ctxV}, medial={n_mwV}, total={nV}")
    print(f"[INFO] Faces:    cortex={n_ctxF}, medial={n_mwF}, total={nF}")

    return fig


def inspect_dtseries(dtseries_path: str):
    """
    读取并检查 CIFTI-2 Dense Timeseries (.dtseries.nii) 文件的信息。
    
    参数：
        dtseries_path : str
            文件路径，如 'subject_rfMRI_REST.dtseries.nii'
    
    返回：
        data : np.ndarray, shape = (T, N)
            fMRI 时间序列矩阵
        brain_models : list
            每个脑区（皮层/亚皮层）模型的信息
    """
    # 读取 CIFTI 文件
    img = nib.load(dtseries_path)
    data = img.get_fdata(dtype=np.float32)
    
    # 获取 header 信息
    header = img.header
    cifti_axes = [img.header.get_axis(i) for i in range(img.ndim)]
    
    print(f"文件路径: {dtseries_path}")
    print(f"数据维度: {data.shape}  -> [时间点 × 顶点/体素]")
    print(f"轴信息:")
    for i, ax in enumerate(cifti_axes):
        print(f"  轴 {i}: {type(ax).__name__}")
        print(f"    长度: {len(ax)}")
        print(f"    名称: {getattr(ax, 'name', None)}")
        if hasattr(ax, 'brain_models'):
            print(f"    包含 {len(ax.brain_models)} 个 brain models:")
            for bm in ax.brain_models[:4]:  # 打印前几个
                print(f"      - {bm.brain_structure} ({bm.index_offset}:{bm.index_offset + bm.index_count})")
    
    # 获取脑区信息
    brain_models = []
    for ax in cifti_axes:
        if hasattr(ax, 'brain_models'):
            brain_models = ax.brain_models
            break

    bm_axis = img.header.get_axis(1)

    # bm_axis.name 是每个灰质点的结构名（长度91282的numpy数组）
    names = np.asarray(bm_axis.name)

    # 找出左半球索引
    lh_indices = np.where(names == 'CIFTI_STRUCTURE_CORTEX_LEFT')[0]

    # 提取左半球时间序列
    lh_ts = data[:, lh_indices]   # shape (1200, 32492)
    print("左半球索引数量:", len(lh_indices))
    print("左半球时序形状:", lh_ts.shape)

    return data, brain_models

def main():
    BASE_PATH = "/home/wmy/Documents/"
    sub = "100307"
    resting_dir = os.path.join(BASE_PATH, "REST1", sub)
    struct_dir  = os.path.join(BASE_PATH, "Structure",  sub)
    
    '''Y_TxV, V, F, m = _load_LH_native_timeseries_and_surface(resting_dir, struct_dir, sub)
    fig = plot_LH_native_with_medial_wall_interactive(
        V, F, m,
        title=f"{sub} LH native (white) — medial wall highlighted",
        save_html=f"./{sub}_LH_native_medialwall_interactive.html"
    )'''

    '''result = view_striatum_middle(
        wmparc_path="/mnt/Structure/100307/MNINonLinear/wmparc.nii.gz",
        t1w_path=None,              # 可指定 T1w；None 表示自动查找
        output_dir="./outputs"      # 输出目录
    )

    print(result)'''

    data, models = inspect_dtseries("/home/wmy/work/geometry/data/subject_rfMRI_REST.dtseries.nii")
    #print(data.shape)
    
if __name__ == "__main__":
    main()


