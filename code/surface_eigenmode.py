import numpy as np
import os
import plotly.graph_objects as go
import mpmath as mp
from numpy.linalg import lstsq
from typing import Dict, List, Tuple
from plotly.subplots import make_subplots
from visualize_brain import load_surface_and_activity, visualize_brain_surface_over_time
from lapy import TriaMesh, Solver

def calc_eig_from_vertices_faces(vertices, faces, num_modes=20):
    """
    根据给定的表面网格 (vertices, faces) 计算特征值和特征模态。
    
    参数
    ----
    vertices : np.ndarray (N x 3)
        网格顶点坐标
    faces : np.ndarray (M x 3)
        网格三角面片（顶点索引）
    num_modes : int
        需要计算的特征模态数量
        
    返回
    ----
    evals : np.ndarray (num_modes,)
        特征值
    emodes : np.ndarray (N x num_modes)
        特征模态
    """
    # 将输入传给 lapy 的 TriaMesh
    tria = TriaMesh(vertices, faces)
    
    # 求解特征值和特征模态
    fem = Solver(tria)
    evals, emodes = fem.eigs(k=num_modes)
    return evals, emodes


def visualize_surface_eigenmodes(vertices, faces, emodes):
    """
    可视化脑表面的特征模态。
    
    参数
    ----
    vertices : np.ndarray (N x 3)
        网格顶点坐标
    faces : np.ndarray (M x 3)
        网格三角形面片
    emodes : np.ndarray (N x K)
        特征模态矩阵 (每列为一个模态)
    num_modes : int
        可视化的模态数量
    multi_plot : bool
        True：在一张图中显示多个模态
        False：每个模态生成一个单独的 HTML 文件
    output_dir : str
        保存输出文件的目录
    可以通过修改col、height、width来调整子图布局/大小
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

def visualize_two_mode_beating(
    vertices,
    faces,
    emodes,
    mode_idx1: int,
    mode_idx2: int,
    f1: float = 10.0,         # Hz, 模态1频率（α波附近）
    f2: float = 10.6,         # Hz, 模态2频率（与 f1 接近 -> 产生拍频）
    gamma: float = 0.8,       # s^-1, 阻尼率（e^{-gamma t}）
    duration: float = 2.0,    # s, 动画时长
    fps: int = 20,            # 帧率（每秒多少帧）
    amp1: float = 1.0,        # 模态1幅度系数
    amp2: float = 1.0,        # 模态2幅度系数
    phase1: float = 0.0,      # 模态1初相（rad）
    phase2: float = 0.0,      # 模态2初相（rad）
    max_frames: int = 120     # 传给可视化函数的最大帧数限制
):
    """
    生成两个空间模态在接近α频率下的拍频动态，并用已有的
    visualize_brain_surface_over_time(...) 进行三维动画可视化。
    这正是论文 Fig. 11 所展示的核心动态：结点线在站立与旋转之间切换。

    参数
    ----
    mode_idx1, mode_idx2 : 选择参与拍频的两个空间模态的列索引（从0开始）
    f1, f2 : 两个模态的振荡频率（Hz），尽量接近以产生清晰的拍频
    gamma : 阻尼率（s^-1），决定 e^{-gamma t} 的衰减速度
    duration : 动画总时长（秒）
    fps : 帧率
    其余参数为幅度与相位设定
    """

    # 基本时间轴
    num_frames = int(np.round(duration * fps))
    t = np.linspace(0.0, duration, num_frames, endpoint=False)  # [0, duration)

    # 取出两个空间模态（每列一个模态），做 L2 归一化，避免尺度不一导致某一模态过强
    m1 = emodes[:, mode_idx1].astype(float)
    m2 = emodes[:, mode_idx2].astype(float)
    m1 = m1 / (np.linalg.norm(m1) + 1e-12)
    m2 = m2 / (np.linalg.norm(m2) + 1e-12)

    # 构造时间因子：衰减 * 余弦振荡
    # activity(t, v) = e^{-gamma t} * ( amp1 * m1[v] * cos(2π f1 t + phase1) + amp2 * m2[v] * cos(2π f2 t + phase2) )
    decay = np.exp(-gamma * t)[:, None]  # (T,1)
    cos1  = np.cos(2 * np.pi * f1 * t + phase1)[:, None]  # (T,1)
    cos2  = np.cos(2 * np.pi * f2 * t + phase2)[:, None]  # (T,1)

    # 组装时空数据 (T, N_vertices)
    activity_timeseries = decay * (amp1 * cos1 * m1[None, :] + amp2 * cos2 * m2[None, :])

    # 用你已有的三维动画函数进行可视化
    fig = visualize_brain_surface_over_time(
        vertices=vertices,
        faces=faces,
        activity_timeseries=activity_timeseries,
        max_frames=max_frames,
        colorscale=[
                    [0.0, 'blue'],   # 最小值为蓝色
                    [0.5, 'white'],  # 中间值为白色
                    [1.0, 'red']     # 最大值为红色
            ],
    )
    
    fig.write_html("/home/wmy/work/geometry/two_mode_beating.html")
    print("动画已保存：two_mode_beating.html")

NFT_PARAMS = dict(
    alpha=50.0,   # s^-1
    beta=200.0,   # s^-1
    tau_es=0.04,  # s
    tau_se=0.04,  # s
    tau_re=0.04,  # s
    gamma_ee=116.0,  # s^-1
    r_ee=0.086,      # m
    Gee=6.8,
    Gei=-8.1,
    GesGse=4.25,
    GesGsrGre=-3.23,
    GsrGrs=-0.36,
    Gsn=1.0,      # 仅作比例常数使用
    Ges=1.0       # 仅作比例常数使用
)

def L_of_omega(omega: complex, alpha: float, beta: float) -> complex:
    # Eq. (9): L(ω) = (1 - iω/α)^-1 (1 - iω/β)^-1
    return 1.0 / (1 - 1j*omega/alpha) / (1 - 1j*omega/beta)

def q2_of_omega(omega: complex, p: dict) -> complex:
    # Eq. (12): q^2(ω) r_ee^2 = (...)
    alpha, beta = p['alpha'], p['beta']
    tau_es, tau_se, tau_re = p['tau_es'], p['tau_se'], p['tau_re']
    gamma_ee, r_ee = p['gamma_ee'], p['r_ee']
    Gee, Gei = p['Gee'], p['Gei']
    GesGse, GesGsrGre, GsrGrs = p['GesGse'], p['GesGsrGre'], p['GsrGrs']

    L = L_of_omega(omega, alpha, beta)
    exp_es_se = np.exp(1j * omega * (tau_es + tau_se))
    exp_es_re = np.exp(1j * omega * (tau_es + tau_re))

    num = (1 - 1j*omega/gamma_ee)**2 - (1.0 / (1 - Gei*L)) * (
        L*Gee
        + (L**2)*GesGse*exp_es_se
        + (L**3)*GesGsrGre*exp_es_re / (1 - (L**2)*GsrGrs)
    )
    return num / (r_ee**2)  # 返回 q^2(ω)

def T_exact(k2: float, omega: complex, p: dict) -> complex:
    """
    Eq. (11): T(k,ω) = [L^2 * Ges * Gsn * e^{iω τ_es} / ((1-L^2 Gsr Grs)(1-Gei L))] * 1/(k^2 r_ee^2 + q^2(ω) r_ee^2)
    常数前因子只影响幅度，不影响极点位置。我们可直接用 1 / (k^2 + q^2(ω)) 的结构来拟合。
    """
    r_ee = p['r_ee']
    q2 = q2_of_omega(omega, p)
    denom = (k2 * r_ee**2 + q2 * r_ee**2)
    if denom == 0:
        return np.inf
    # 只保留极点结构（分母）；前因子省略不改变极点
    return 1.0 / denom

# ====== (b) 用式(18)思路拟合分母多项式（等价拟合 1/T）======
def fit_denominator_poly_via_least_squares(
    k2: float,
    params: dict,
    n: int = 14,
    w_max_hz: float = 40.0,
    num_samples: int = 2000
) -> np.ndarray:
    """
    在实频轴 [0, w_max] 上采样精确 T(k, ω)，
    用复数最小二乘拟合： 1/T(ω) ≈ sum_{q=0}^{n} A_q * (-i ω)^q
    返回 A（长度 n+1），多项式变量为 s = (-i ω)
    注：只要分母拟合得好，其根（在复平面）就是 T 的极点。
    """
    w = np.linspace(1e-3, 2*np.pi*w_max_hz, num_samples)  # rad/s
    T_vals = np.array([T_exact(k2, wi, params) for wi in w], dtype=complex)
    Y = 1.0 / T_vals  # 目标：分母多项式值（可差个比例常数不影响根）
    # 构造范德蒙德矩阵 V_{t,q} = (-i ω_t)^q
    s = -1j * w
    V = np.vstack([s**q for q in range(n+1)]).T  # shape (num_samples, n+1)
    # 复数最小二乘
    A, *_ = lstsq(V, Y, rcond=None)
    return A  # A[0] + A[1] s + ... + A[n] s^n

def poles_from_den_poly(A: np.ndarray) -> np.ndarray:
    """
    已有分母多项式 D(s) = sum_{q=0}^n A_q s^q, 变量 s = (-i ω)。
    求 D(s)=0 的根 s_j，然后映射回 ω_j = i * s_j。
    """
    # numpy 根求解默认最高次在前，这里先翻转系数顺序
    coeffs = A[::-1]  # 最高次到常数项
    s_roots = np.roots(coeffs)
    omega_roots = 1j * s_roots
    return omega_roots  # 复频率 ω = Ω - i γ

def eigenfreqs_via_eq18_for_mode(
    k2: float,
    params: dict = NFT_PARAMS,
    n: int = 14,
    w_max_hz: float = 40.0,
    num_samples: int = 2000
) -> List[Tuple[float, float, complex]]:
    """
    对单个空间模态（给定 k^2）：
    1) 采样真实 T
    2) 用式(18)思路拟合分母多项式
    3) 求根得到极点（频率与阻尼）
    返回列表 [(f_Hz, gamma_s^-1, omega_complex), ...]，按阻尼从小到大排序。
    """
    A = fit_denominator_poly_via_least_squares(k2, params, n=n, w_max_hz=w_max_hz, num_samples=num_samples)
    roots = poles_from_den_poly(A)

    out = []
    for w in roots:
        # 只保留稳定极点（Im(ω) < 0）
        if np.imag(w) < 0:
            f = abs(np.real(w))/(2*np.pi)  # Hz
            gamma = -np.imag(w)           # s^-1
            out.append((f, gamma, w))
    out.sort(key=lambda x: x[1])  # 阻尼从小到大（更主导在前）
    return out

def eigenfreqs_for_modes_via_eq18(evals: np.ndarray, mode_indices: List[int], **kwargs):
    """
    批量：对若干模态索引计算 (f, gamma) 列表
    """
    res = {}
    for idx in mode_indices:
        k2 = max(float(evals[idx]), 0.0)   # 拉普拉斯本征值 ~ k^2
        res[idx] = eigenfreqs_via_eq18_for_mode(k2, **kwargs)
    return res

def pick_alpha_or_fallback(vals, band=(8.0, 13.0), wide=(5.0, 20.0)):
    """
    从 (f, gamma, ω) 列表中优先挑 α 带(8–13Hz)阻尼最小的一个；
    若没有 -> 扩大到 5–20Hz；
    再没有 -> 挑全频里阻尼最小的一个；
    若列表本身为空 -> 返回一个 None，同时给出清晰的诊断。
    返回：(f, gamma, ω) 或 None
    """
    if not vals:
        print("[pick_alpha] 没有任何极点（列表为空）。")
        return None

    def in_band(x, lo, hi): return (lo <= x <= hi)

    # 已经按 gamma 升序排序的话，这里不需要再排序
    for lo, hi in (band, wide):
        cand = [t for t in vals if in_band(t[0], lo, hi)]
        if cand:
            best = cand[0]
            print(f"[pick_alpha] 命中频带 {lo}-{hi} Hz: 选 f={best[0]:.3f} Hz, γ={best[1]:.3f} s^-1")
            return best

    # 全频最小阻尼
    best = vals[0]
    print(f"[pick_alpha] α/宽频带都未命中，改取全频阻尼最小: f={best[0]:.3f} Hz, γ={best[1]:.3f} s^-1")
    return best

def main():
    # 设置数据目录
    #BASE_PATH = '/mnt/'
    BASE_PATH = '/home/wmy/Documents/'
    sub = '100307'
    resting_data_dir = os.path.join(BASE_PATH, 'Resting_1', sub)
    struct_data_dir = os.path.join(BASE_PATH, 'Structure', sub)
    

    # 加载数据和可视化表面
    vertices, faces, activity_timeseries = load_surface_and_activity(resting_data_dir, struct_data_dir)

    # 计算脑表面的特征模态
    evals, emodes = calc_eig_from_vertices_faces(vertices, faces, num_modes=50)
    np.savetxt("/home/wmy/work/geometry/eigenvalues.txt", evals)
    np.savetxt("/home/wmy/work/geometry/eigenmodes.txt", emodes)
    visualize_surface_eigenmodes(vertices, faces, emodes)

    mode_idx1, mode_idx2 = 1, 2
    res = eigenfreqs_for_modes_via_eq18(evals, [mode_idx1, mode_idx2], params=NFT_PARAMS, n=14, w_max_hz=40.0)
    f1, g1, _ = pick_alpha_or_fallback(res[mode_idx1])
    f2, g2, _ = pick_alpha_or_fallback(res[mode_idx2])
    gamma = 0.5*(g1 + g2)

    visualize_two_mode_beating(
        vertices=vertices, faces=faces, emodes=emodes,
        mode_idx1=mode_idx1, mode_idx2=mode_idx2,
        f1=f1,      # 第一个模态频率
        f2=f2,      # 第二个模态频率
        gamma=gamma,    # 阻尼率
        duration=1.0, # 动画时长
        fps=20,       # 帧率
    )

if __name__ == "__main__":
    main()