import os
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from skimage import measure

from visualize_brain import extract_nuclei_mesh  # 你的文件名/模块名按实际调整

def compare_mesh(struct_data_dir, nuclei_labels, out_html="mesh_compare.html"):
    # 1) 得到正确 verts_world
    verts_world_ok, faces, subcortical_img, mask = extract_nuclei_mesh(struct_data_dir, nuclei_labels)

    # 2) 在测试脚本里复刻“原始错误逻辑”：得到 verts_world_wrong
    verts_vox, faces2, _, _ = measure.marching_cubes(mask, level=0.5)  # 体素索引坐标
    voxel_size = subcortical_img.header.get_zooms()
    verts_wrong = verts_vox * voxel_size  # ❌ 错误点：多此一举（会导致重复缩放）
    affine = subcortical_img.affine
    verts_world_wrong = (np.hstack([verts_wrong, np.ones((verts_wrong.shape[0], 1))]) @ affine.T)[:, :3]

    # 3) 叠加可视化（两个 Mesh3d）
    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=verts_world_wrong[:,0], y=verts_world_wrong[:,1], z=verts_world_wrong[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        name="WRONG (double scaling)",
        opacity=0.25,
        color="red",
        showscale=False
    ))

    fig.add_trace(go.Mesh3d(
        x=verts_world_ok[:,0], y=verts_world_ok[:,1], z=verts_world_ok[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        name="OK (affine once)",
        opacity=0.6,
        color="blue",
        showscale=False
    ))

    fig.update_layout(
        title="Mesh Compare: WRONG (red) vs OK (blue)",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    print("Saved:", os.path.abspath(out_html))

if __name__ == "__main__":
    # 例：左尾状核 11（你也可以换成 [10]/[12]/[13] 等）
    compare_mesh(struct_data_dir="/home/wmy/Documents/Structure/100307", nuclei_labels=[11])
