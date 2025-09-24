import os
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
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




def main():
    result = view_striatum_middle(
        wmparc_path="/mnt/Structure/100307/MNINonLinear/wmparc.nii.gz",
        t1w_path=None,              # 可指定 T1w；None 表示自动查找
        output_dir="./outputs"      # 输出目录
    )

    print(result)
    
if __name__ == "__main__":
    main()

