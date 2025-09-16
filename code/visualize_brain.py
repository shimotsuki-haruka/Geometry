import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from nibabel import gifti
from nilearn import surface, plotting, image, datasets
from nilearn.regions import connected_regions
from skimage import measure
from scipy import ndimage as ndi
from skimage.measure import marching_cubes


def load_surface_and_activity(resting_data_dir, struct_data_dir):
    # 加载左半球表面数据 (functional data)
    surf_left = nib.load(os.path.join(resting_data_dir, 'MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.L.native.func.gii'))
    
    # 加载左半球表面网格 (structural data - white matter surface, uninflated)
    surf_mesh_left = nib.load(os.path.join(struct_data_dir, 'MNINonLinear/Native/100307.L.white.native.surf.gii'))
    
    # 获取顶点坐标和面片信息
    vertices_left = surf_mesh_left.darrays[0].data  # 顶点坐标
    faces_left = surf_mesh_left.darrays[1].data     # 面片信息
    
    # 获取所有时间点的神经活动数据
    print(len(surf_left.darrays), surf_left.darrays[0].data.shape)
    activity_timeseries = []
    for i in range(len(surf_left.darrays)):
        activity_left = surf_left.darrays[i].data
        activity_timeseries.append(activity_left)
    
    # 将时间序列数据转换为 numpy 数组
    activity_timeseries = np.array(activity_timeseries)
    
    print(activity_timeseries.shape, len(vertices_left))
    
    # 确保活动数据长度与顶点数匹配
    if activity_timeseries.shape[1] != len(vertices_left):
        print(f"Warning: Activity data length ({activity_timeseries.shape[1]}) does not match vertex count ({len(vertices_left)})")
        # 在实际应用中，你可能需要进行数据重新采样或投影
    
    print(f"Loaded {activity_timeseries.shape[0]} time points of data")
    
    return vertices_left, faces_left, activity_timeseries

def visualize_brain_surface_over_time(vertices, faces, activity_timeseries, max_frames=100, colorscale='RdYlBu_r'):
    # 限制帧数以防止文件过大
    num_frames = min(activity_timeseries.shape[0], max_frames)
    # 计算步长，以便均匀采样时间点
    step = activity_timeseries.shape[0] // num_frames if activity_timeseries.shape[0] > num_frames else 1
    
    # 获取全局活动数据的最小值和最大值，以便颜色范围统一
    min_val = np.min(activity_timeseries)
    max_val = np.max(activity_timeseries)
    
    # 创建 3D 可视化
    fig = go.Figure()
    
    # 为每个时间点创建一个帧
    frames = []
    for i in range(0, activity_timeseries.shape[0], step):
        if i >= num_frames * step:
            break
            
        frames.append(
            go.Frame(
                data=[
                    go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        intensity=activity_timeseries[i],
                        colorscale=colorscale,
                        cmin=min_val,  # 设置统一的颜色范围
                        cmax=max_val,
                        showscale=True,
                        opacity=0.8
                    )
                ],
                name=f'Time {i}'
            )
        )
    
    # 添加第一帧作为初始显示
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=activity_timeseries[0],
            colorscale=colorscale,
            cmin=min_val,
            cmax=max_val,
            showscale=True,
            opacity=0.8
        )
    )
    
    # 将帧添加到图形中
    fig.frames = frames
    
    # 添加播放控件
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        title="Brain Activity Over Time",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0}
                            }
                        ]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    }
                ]
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "label": f"{i}",
                        "args": [
                            [f"Time {i*step}"],
                            {
                                "mode": "immediate",
                                "transition": {"duration": 0},
                                "frame": {"duration": 0, "redraw": False}
                            }
                        ]
                    }
                    for i in range(len(frames))
                ],
                "active": 0,
                "currentvalue": {"prefix": "Time point: "}
            }
        ]
    )
    
    return fig

def load_and_visualize_subcortical(resting_data_dir, struct_data_dir):
    """加载并可视化核团上的神经活动"""
    # 加载结构数据 (T1 加权图像)
    t1_file = os.path.join(struct_data_dir, 'MNINonLinear/T1w.nii.gz')
    t1_img = nib.load(t1_file)
    
    # 加载核团掩码 (例如基底节，丘脑等)
    # HCP 数据通常有一个分割文件，包含了不同脑区的标签
    subcortical_file = os.path.join(struct_data_dir, 'MNINonLinear/wmparc.nii.gz')
    subcortical_img = nib.load(subcortical_file)
    
    # 加载功能 MRI 数据
    func_file = os.path.join(resting_data_dir, 'MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz')
    func_img = nib.load(func_file)
    
    # 创建核团区域的掩码
    # wmparc 文件中的标签：
    # 10 = 左丘脑，11 = 左尾状核，12 = 左壳核，13 = 左苍白球
    # 49 = 右丘脑，50 = 右尾状核，51 = 右壳核，52 = 右苍白球
    subcortical_labels = [10, 11, 12, 13, 49, 50, 51, 52]
    subcortical_data = subcortical_img.get_fdata()
    
    # 创建核团掩码
    mask = np.zeros_like(subcortical_data)
    for label in subcortical_labels:
        mask[subcortical_data == label] = 1
    
    # 保存掩码为 nifti 文件
    mask_img = nib.Nifti1Image(mask, subcortical_img.affine)
    
    # 使用 nilearn 提取核团区域的时间序列
    # 先获取第一个时间点作为示例
    first_frame = image.index_img(func_img, 0)
    
    # 应用掩码到功能数据
    masked_func = image.math_img("img1 * img2", img1=first_frame, img2=mask_img)
    
    # 可视化核团区域
    # 创建一个包含结构和功能的混合图
    display = plotting.plot_stat_map(
        masked_func, 
        bg_img=t1_img,
        display_mode='ortho', 
        cut_coords=(0, 0, 0),
        title='核团区域的神经活动',
        output_file='subcortical_activity.png'
    )
    
    # 返回显示对象以便进一步处理
    return display

def visualize_subcortical_over_time(resting_data_dir, struct_data_dir, output_dir='subcortical_frames'):
    """创建核团活动随时间变化的可视化序列"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载结构数据 (T1 加权图像)
    t1_file = os.path.join(struct_data_dir, 'MNINonLinear/T1w.nii.gz')
    t1_img = nib.load(t1_file)
    
    # 加载核团掩码
    subcortical_file = os.path.join(struct_data_dir, 'MNINonLinear/wmparc.nii.gz')
    subcortical_img = nib.load(subcortical_file)
    
    # 加载功能 MRI 数据
    func_file = os.path.join(resting_data_dir, 'MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz')
    func_img = nib.load(func_file)
    
    # 创建核团区域的掩码
    subcortical_labels = [10, 11, 12, 13, 49, 50, 51, 52]  # 主要核团的标签
    subcortical_data = subcortical_img.get_fdata()
    
    # 创建核团掩码
    mask = np.zeros_like(subcortical_data)
    for label in subcortical_labels:
        mask[subcortical_data == label] = 1
    
    # 保存掩码为 nifti 文件
    mask_img = nib.Nifti1Image(mask, subcortical_img.affine)
    
    # 获取时间点数量
    n_timepoints = func_img.shape[3]
    max_frames = min(50, n_timepoints)  # 限制帧数
    step = n_timepoints // max_frames if n_timepoints > max_frames else 1
    
    # 首先将掩码重采样到功能数据的空间
    # 这解决了形状和仿射矩阵不匹配的问题
    mask_resampled = image.resample_to_img(mask_img, func_img.slicer[..., 0], 
                                           interpolation='nearest')
    
    print(f"原始掩码形状：{mask_img.shape}, 功能数据形状：{func_img.shape[:3]}")
    print(f"重采样后掩码形状：{mask_resampled.shape}")
    
    # 为每个时间点创建一个图像
    for i in range(0, n_timepoints, step):
        if i >= max_frames * step:
            break
            
        # 获取当前时间点的功能数据
        frame = image.index_img(func_img, i)
        
        # 应用掩码到功能数据 (现在它们有相同的形状和仿射矩阵)
        masked_func = image.math_img("img1 * img2", img1=frame, img2=mask_resampled)
        
        # 可视化并保存图像
        output_file = os.path.join(output_dir, f'subcortical_time_{i:04d}.png')
        plotting.plot_stat_map(
            masked_func, 
            bg_img=image.resample_to_img(t1_img, frame, interpolation='linear'),  # 也重采样背景图像
            display_mode='ortho', 
            cut_coords=(0, 0, 0),
            title=f'核团神经活动 (时间点 {i})',
            output_file=output_file
        )
        
        print(f"已生成图像：{output_file}")
    
    print(f"所有图像已保存到目录：{output_dir}")
    print("您可以使用这些图像创建动画或以幻灯片形式查看活动变化")


###draw

def add_axes_RPD(fig, verts_mm, scale_ratio=0.15, margin_ratio=0.05,
                 r_dir=(0,1,0), p_dir=(1,0,0), d_dir=(0,0,1)):
    """
    在已有 fig 上添加 R/P/D 三轴。
    verts_mm: (N,3) 世界坐标顶点
    r_dir/p_dir/d_dir: 三个方向单位向量（默认 RAS+ 假设）
    """
    mins = verts_mm.min(0)
    maxs = verts_mm.max(0)
    diag = np.linalg.norm(maxs - mins)
    L = diag * scale_ratio
    origin = mins - (maxs - mins) * margin_ratio

    def axis_trace(o, v, label):
        return go.Scatter3d(
            x=[o[0], o[0]+v[0]], y=[o[1], o[1]+v[1]], z=[o[2], o[2]+v[2]],
            mode="lines+text",
            text=[None, label],
            textposition="top center",
            line=dict(width=6),
            hoverinfo="skip",
            showlegend=False,
        )

    fig.add_trace(axis_trace(origin, np.array(r_dir)*L, "R"))
    fig.add_trace(axis_trace(origin, np.array(p_dir)*L, "P"))
    fig.add_trace(axis_trace(origin, np.array(d_dir)*L, "D"))
    fig.update_layout(scene=dict(aspectmode="data"))

def visualize_mask_html(mask_processed,
                        affine,
                        out_html,
                        level=0.5,
                        original_mask=None,
                        title="Mask visualization",
                        processed_name="Processed",
                        original_name="Original",
                        processed_opacity=1.0,
                        original_opacity=0.35):
    """
    将三维二值掩膜可视化为 HTML（Plotly Mesh3d）。
    若提供 original_mask，则在同一图中对比显示。

    Parameters
    ----------
    mask_processed : (X,Y,Z) bool/uint8
        已处理好的掩膜（如 dilation+erosion 后）
    affine : (4,4) array
        NIfTI 的仿射矩阵，用于把体素坐标变换到世界坐标（mm）
    out_html : str
        输出 HTML 文件路径
    level : float
        marching_cubes 等值面（对二值掩膜设为 0.5 即可）
    original_mask : (X,Y,Z) bool/uint8 or None
        原始掩膜；若提供则一起可视化对比
    title : str
        图标题
    processed_name : str
        图例中处理后掩膜名称
    original_name : str
        图例中原始掩膜名称
    processed_opacity : float
        处理后网格不透明度
    original_opacity : float
        原始网格不透明度（建议半透明便于对比）
    """
    def mask_to_mesh(bin_mask, name, opacity):
        bin_mask = (bin_mask > 0).astype(np.uint8)
        if bin_mask.sum() == 0:
            return None
        verts_vox, faces, _, _ = marching_cubes(
            volume=bin_mask, level=level, allow_degenerate=False
        )
        # 体素坐标 -> 世界坐标（mm）
        verts_mm = nib.affines.apply_affine(affine, verts_vox)
        i, j, k = faces.T
        return go.Mesh3d(
            x=verts_mm[:, 0], y=verts_mm[:, 1], z=verts_mm[:, 2],
            i=i, j=j, k=k,
            name=name, opacity=opacity,
            flatshading=True,
            lighting=dict(ambient=0.3, diffuse=0.8, specular=0.2, roughness=0.6),
            lightposition=dict(x=120, y=180, z=250),
            showscale=False
        )

    traces = []
    if original_mask is not None:
        m = mask_to_mesh(original_mask, original_name, original_opacity)
        if m is not None:
            traces.append(m)

    m = mask_to_mesh(mask_processed, processed_name, processed_opacity)
    if m is not None:
        traces.append(m)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, x=0.02)
    )
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    print(f"Saved HTML: {os.path.abspath(out_html)}")

def extract_nuclei_mesh(struct_data_dir, nuclei_labels):
    """从分割数据中提取核团的 3D 网格"""
    # 加载核团分割数据
    subcortical_file = os.path.join(struct_data_dir, 'MNINonLinear/wmparc.nii.gz')
    subcortical_img = nib.load(subcortical_file)
    subcortical_data = subcortical_img.get_fdata()
    
    # 创建掩码
    mask = np.zeros_like(subcortical_data)
    for label in nuclei_labels:
        mask[subcortical_data == label] = 1
    
    '''structure = ndi.generate_binary_structure(3, 2)  # 26连通邻域
    mask = ndi.binary_dilation(mask, structure=structure, iterations=2)
    mask = ndi.binary_erosion(mask, structure=structure, iterations=2)'''

    visualize_mask_html(
    mask_processed=mask,
    affine=subcortical_img.affine,
    out_html="nuclei_closed.html",
    original_mask=None,   # 若不想对比，可传 None
    title="Original vs Closed"
    )
    
    # 获取体素尺寸，用于正确缩放网格
    voxel_size = subcortical_img.header.get_zooms()
    
    # 使用 marching cubes 算法生成 3D 网格
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    
    # 应用体素尺寸缩放
    verts = verts * voxel_size
    
    # 将顶点坐标转换到世界坐标系
    affine = subcortical_img.affine
    verts_world = np.dot(np.hstack((verts, np.ones((verts.shape[0], 1)))), affine.T)[:, :3]
    
    return verts_world, faces, subcortical_img, mask

def map_activity_to_mesh(func_img, verts_world, mask_img):
    """将功能活动映射到网格顶点"""
    # 检查功能数据是否为 3D 或 4D
    if len(func_img.shape) == 4:
        # 如果是 4D，获取第一个时间点
        first_frame = image.index_img(func_img, 0)
    else:
        # 如果已经是 3D，直接使用
        first_frame = func_img
    
    # 将网格顶点坐标转换为功能数据中的索引
    # 这需要将世界坐标反转换为体素索引
    inv_affine = np.linalg.inv(first_frame.affine)
    verts_func_indices = np.dot(np.hstack((verts_world, np.ones((verts_world.shape[0], 1)))), inv_affine.T)[:, :3]
    
    # 确保索引是整数
    verts_func_indices = np.round(verts_func_indices).astype(int)
    
    # 获取功能数据
    func_data = first_frame.get_fdata()
    
    # 确保索引在有效范围内
    valid_indices = (
        (verts_func_indices[:, 0] >= 0) & (verts_func_indices[:, 0] < func_data.shape[0]) &
        (verts_func_indices[:, 1] >= 0) & (verts_func_indices[:, 1] < func_data.shape[1]) &
        (verts_func_indices[:, 2] >= 0) & (verts_func_indices[:, 2] < func_data.shape[2])
    )
    
    # 初始化活动数据
    activity = np.zeros(verts_world.shape[0])
    
    # 对有效索引，获取相应的功能值
    valid_verts = verts_func_indices[valid_indices]
    activity[valid_indices] = func_data[
        valid_verts[:, 0], valid_verts[:, 1], valid_verts[:, 2]
    ]
    
    return activity

def visualize_nuclei_mesh(verts, faces, activity, nucleus_name):
    """可视化核团网格和活动"""
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=activity,
            colorscale='RdBu',
            showscale=True,
            opacity=0.8
        )
    ])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        title=f"{nucleus_name}的 3D 模型与神经活动"
    )
    
    return fig

def visualize_all_nuclei(resting_data_dir, struct_data_dir):
    """可视化所有感兴趣的核团"""
    # 定义感兴趣的核团及其标签
    nuclei_dict = {
        "左丘脑": [10],
        "左尾状核": [11],
        "左壳核": [12],
        "左苍白球": [13],
        "右丘脑": [49],
        "右尾状核": [50],
        "右壳核": [51],
        "右苍白球": [52],
        "所有核团": [10, 11, 12, 13, 49, 50, 51, 52]
    }
    
    # 加载功能数据
    func_file = os.path.join(resting_data_dir, 'MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz')
    func_img = nib.load(func_file)
    
    # 为每个核团创建可视化
    for name, labels in nuclei_dict.items():
        print(f"正在处理{name}...")
        
        # 提取网格
        verts, faces, subcortical_img, mask = extract_nuclei_mesh(struct_data_dir, labels)
        
        # 创建掩码图像
        mask_img = nib.Nifti1Image(mask, subcortical_img.affine)
        
        # 将功能活动映射到网格
        activity = map_activity_to_mesh(func_img, verts, mask_img)
        
        # 可视化
        fig = visualize_nuclei_mesh(verts, faces, activity, name)
        
        # 保存和显示
        fig.write_html(f"{name}_3D_mesh.html")
        fig.show()

def visualize_nuclei_activity_over_time(resting_data_dir, struct_data_dir, nucleus_name, labels, max_frames=30):
    """创建核团活动随时间变化的可视化"""
    # 提取网格
    verts, faces, subcortical_img, mask = extract_nuclei_mesh(struct_data_dir, labels)
    
    # 创建掩码图像
    mask_img = nib.Nifti1Image(mask, subcortical_img.affine)
    
    # 加载功能数据
    func_file = os.path.join(resting_data_dir, 'MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz')
    func_img = nib.load(func_file)
    
    # 确保函数文件是 4D 的
    if len(func_img.shape) != 4:
        raise ValueError(f"功能数据应为 4D，但发现它是{len(func_img.shape)}D")
    
    # 获取时间点数量
    n_timepoints = func_img.shape[3]
    max_frames = min(max_frames, n_timepoints)
    step = n_timepoints // max_frames if n_timepoints > max_frames else 1
    
    print(f"处理{n_timepoints}个时间点中的{max_frames}个...")
    
    # 为所有时间点获取活动数据
    activity_timeseries = []
    
    for i in range(0, n_timepoints, step):
        if i >= max_frames * step:
            break
            
        # 获取当前时间点
        frame = image.index_img(func_img, i)
        
        # 将顶点映射到当前帧
        activity = map_activity_to_mesh(frame, verts, mask_img)
        activity_timeseries.append(activity)
    
    # 获取全局最小值和最大值
    all_activity = np.concatenate(activity_timeseries)
    min_val = np.min(all_activity)
    max_val = np.max(all_activity)
    
    # 创建动画
    fig = go.Figure()
    
    # 创建帧
    frames = []
    for i, activity in enumerate(activity_timeseries):
        frames.append(
            go.Frame(
                data=[go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    intensity=activity,
                    colorscale='RdBu',
                    cmin=min_val,
                    cmax=max_val,
                    showscale=True,
                    opacity=0.8
                )],
                name=f'Time {i}'
            )
        )
    
    # 添加初始数据
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=activity_timeseries[0],
        colorscale='RdBu',
        cmin=min_val,
        cmax=max_val,
        showscale=True,
        opacity=0.8
    ))
    
    # 将帧添加到图中
    fig.frames = frames
    
    # 添加播放控件
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        title=f"{nucleus_name}随时间变化的神经活动",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "播放",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0}
                            }
                        ]
                    },
                    {
                        "label": "暂停",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    }
                ]
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "label": f"{i}",
                        "args": [
                            [f"Time {i}"],
                            {
                                "mode": "immediate",
                                "transition": {"duration": 0},
                                "frame": {"duration": 0, "redraw": False}
                            }
                        ]
                    }
                    for i in range(len(frames))
                ],
                "active": 0,
                "currentvalue": {"prefix": "时间点："}
            }
        ]
    )
    
    # 保存和显示
    fig.write_html(f"{nucleus_name}_activity_over_time.html")
    return fig

def main():
    # 设置数据目录
    #BASE_PATH = '/mnt/'
    BASE_PATH = '/home/wmy/Documents/'
    sub = '100307'
    resting_data_dir = os.path.join(BASE_PATH, 'Resting_1', sub)
    struct_data_dir = os.path.join(BASE_PATH, 'Structure', sub)
    

    # 加载数据和可视化表面
    #vertices, faces, activity_timeseries = load_surface_and_activity(resting_data_dir, struct_data_dir)

    # 可视化脑表面活动
    #fig = visualize_brain_surface_over_time(vertices, faces, activity_timeseries, max_frames=50)
    #fig.show()
    #fig.write_html("brain_activity_over_time.html")

 

    
    # 加载和可视化核团
    print("\n正在生成核团可视化...")
    visualize_subcortical_over_time(resting_data_dir, struct_data_dir)
    
    # 可视化单个核团随时间的变化
    # 这里以左丘脑为例
    fig = visualize_nuclei_activity_over_time(
        resting_data_dir, 
        struct_data_dir, 
        "左丘脑", 
        [10], 
        max_frames=30
    )
    fig.show()
    
    # 如果需要可视化所有核团的 3D 模型和活动
    # visualize_all_nuclei(resting_data_dir, struct_data_dir)


if __name__ == "__main__":
    main()