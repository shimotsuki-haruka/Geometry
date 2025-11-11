import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Tuple
from mpmath import mp
from plotly.subplots import make_subplots
from visualize_brain import visualize_brain_surface_over_time, load_surface_and_activity
from surface_eigenmode import cal_and_visualize


def visualize_two_mode_beating(
    vertices,
    faces,
    emodes,
    mode_idx1: int,
    mode_idx2: int,
    f1: float = 10.0,         # Hz, 模态1频率
    f2: float = 10.6,         # Hz, 模态2频率
    gamma: float = 0.8,       # s^-1, 阻尼率
    duration: float = 2.0,    # s, 动画时长
    fps: int = 20,            # 帧率
    amp1: float = 1.0,        # 模态1幅度
    amp2: float = 1.0,        # 模态2幅度
    phase1: float = 0.0,      # 模态1初相（rad）
    phase2: float = 0.0,      # 模态2初相（rad）
    max_frames: int = 120     # 最大帧数限制
):
    """
    可视化两个模态在接近 α 频率下的拍频动态。
    phase1 / phase2 控制两个模态的初始相位差：
      - phase2=0      -> 驻波
      - phase2=π/2    -> 旋转波 / 行波
    """

    # 基本时间轴
    num_frames = int(np.round(duration * fps))
    num_frames = min(num_frames, max_frames)   # 限制帧数
    t = np.linspace(0.0, duration, num_frames, endpoint=False)

    # 取出两个空间模态并 L2 归一化
    m1 = emodes[:, mode_idx1].astype(float)
    m2 = emodes[:, mode_idx2].astype(float)
    m1 = m1 / (np.linalg.norm(m1) + 1e-12)
    m2 = m2 / (np.linalg.norm(m2) + 1e-12)

    # 构造时间因子：衰减 * 余弦振荡（带初相）
    decay = np.exp(-gamma * t)[:, None]        # (T,1)
    cos1  = np.cos(2 * np.pi * f1 * t + phase1)[:, None]
    cos2  = np.cos(2 * np.pi * f2 * t + phase2)[:, None]

    # 时空数据 (T, N_vertices)
    activity_timeseries = decay * (
        amp1 * cos1 * m1[None, :] +
        amp2 * cos2 * m2[None, :]
    )

    # 用已有三维动画函数可视化
    fig = visualize_brain_surface_over_time(
        vertices=vertices,
        faces=faces,
        activity_timeseries=activity_timeseries,
        max_frames=max_frames,
        colorscale=[
            [0.0, 'blue'],
            [0.5, 'white'],
            [1.0, 'red']
        ],
    )

    fig.write_html("/home/wmy/work/geometry/two_mode_beating.html")
    print("动画已保存：two_mode_beating.html")
    return fig


NFT_PARAMS = dict(
    alpha=50.0,   # s^-1
    beta=200.0,   # s^-1
    tau_es=0.04,  # s
    tau_se=0.04,  # s
    tau_re=0.04,  # s
    gamma_ee=116.0,  # s^-1
    r_ee=86.0,      # mm
    Gee=6.8,
    Gei=-8.1,
    GesGse=4.25,
    GesGsrGre=-3.23,
    GsrGrs=-0.36,
    Gsn=1.0,      # 仅作比例常数使用
    Ges=1.0       # 仅作比例常数使用
)

seeds = [
        # α 波段 (8–12 Hz)
        2*np.pi*10 - 15j,
        2*np.pi*9  - 13j,
        2*np.pi*11 - 13j,
        2*np.pi*8  - 16j,
        2*np.pi*12 - 16j,
        2*np.pi*10 - 11j,

        # 慢波 (f≈0 Hz, γ≈20 s^-1)
        2*np.pi*1 - 15j,  # γ≈15
        2*np.pi*2 - 20j,  # γ≈20
        2*np.pi*3 - 25j   # γ≈25
    ]

def F_or_T(k2: float, params: dict):
    #返回公式 F(ω) 和 T(ω)
    alpha, beta  = params['alpha'], params['beta']
    tau_es, tau_se, tau_re = params['tau_es'], params['tau_se'], params['tau_re']
    gamma_ee, r_ee = params['gamma_ee'], params['r_ee']
    Gee, Gei = params['Gee'], params['Gei']
    GesGse, GesGsrGre, GsrGrs = params['GesGse'], params['GesGsrGre'], params['GsrGrs']

    def q2_common(w, use_mp: bool):
        if use_mp:
            exp = lambda z: mp.e**(z)
        else:
            exp = np.exp

        L = 1 / (1 - 1j*w/alpha) / (1 - 1j*w/beta)
        exp_es_se = exp(1j * w * (tau_es + tau_se))
        exp_es_re = exp(1j * w * (tau_es + tau_re))
        num = (1 - 1j*w/gamma_ee)**2 - (1.0 / (1 - Gei*L)) * (
            L*Gee
            + ((L**2)*GesGse*exp_es_se + (L**3)*GesGsrGre*exp_es_re) / (1 - (L**2)*GsrGrs)
        )
        return num / (r_ee**2)  # q^2(ω)

    def F_mp(w):
        q2 = q2_common(w, use_mp=True)
        return k2 + q2
    
    def T_exact(omega):
        q2 = q2_common(omega, use_mp=False)
        denom = k2 * r_ee**2 + q2 * r_ee**2
        return 1.0 / denom

    return F_mp, T_exact

#==========多项式求根==========#

def fit_rational_T_pade(
    k2: float,
    n: int = 14,          # 分母阶
    m: int = 8,           # 分子阶 (m < n)
    f_max_hz: float = 40.0,
    num_samples: int = 2000,
    weight_alpha: float = 1.0,  # α 带权重（>1 可加强 8–13Hz）
):
    """
    拟合 T(ω) ≈ N(s)/D(s)，s=-iω。
    返回 (A, B)，A.shape=(n+1,), B.shape=(m+1,)
    且 A[0] 固定为 1（归一化）；A, B 全为实数（KK/因果性约束）。
    """
    # 频率采样
    w = np.linspace(1e-3, 2*np.pi*f_max_hz, num_samples)  # rad/s
    F, T_exact = F_or_T(k2, NFT_PARAMS)
    T_vals = np.array([T_exact(wi) for wi in w], dtype=complex)
    s = -1j * w

    # 权重（可对 α 带加权）
    if weight_alpha != 1.0:
        f = w / (2*np.pi)
        wts = np.ones_like(w)
        wts[(f >= 8.0) & (f <= 13.0)] = weight_alpha
    else:
        wts = np.ones_like(w)

    # 构建设计矩阵（实部/虚部拆开；未知量是 real B[0..m], real A[1..n]）
    # 方程：sum_p B_p s^p - T * (A_0 + sum_{q>=1} A_q s^q) = 0,  A_0=1
    # => sum_p B_p s^p - T - T * sum_{q>=1} A_q s^q = 0
    A0 = 1.0
    # 组装实值最小二乘： [Re, Im] 两倍点数
    M = 2*len(w)
    K = (m+1) + n  # B[m+1] + A[1..n] 共 m+1 + n 个未知
    Mtx = np.zeros((M, K), dtype=float)
    rhs = np.zeros((M,), dtype=float)

    # 预先构造幂
    Sp = np.vstack([s**p for p in range(m+1)]).T      # (num_samples, m+1)
    Sq = np.vstack([s**q for q in range(1, n+1)]).T   # (num_samples, n)

    # 左边：sum_p B_p s^p - T - T*sum_{q>=1} A_q s^q = 0
    # -> 实部行
    row = 0
    for t in range(len(w)):
        # 列顺序：B[0..m] | A[1..n]
        # 实部: Re(Sp) * B  - Re(T)  - Re(T*Sq) * A_tail = 0
        # 虚部: Im(Sp) * B  - Im(T)  - Im(T*Sq) * A_tail = 0
        Sp_t = Sp[t]
        Sq_t = Sq[t]
        Tt = T_vals[t]
        Tsq = Tt * Sq_t  # 逐 q 项

        wt = wts[t]
        # 实部
        Mtx[row, 0:(m+1)]  =  np.real(Sp_t) * wt
        Mtx[row, (m+1):]   = -np.real(Tsq) * wt
        rhs[row]           =  np.real(Tt) * wt  # 移项：= Re(T)
        row += 1
        # 虚部
        Mtx[row, 0:(m+1)]  =  np.imag(Sp_t) * wt
        Mtx[row, (m+1):]   = -np.imag(Tsq) * wt
        rhs[row]           =  np.imag(Tt) * wt
        row += 1

    # 解实数最小二乘
    x, *_ = np.linalg.lstsq(Mtx, rhs, rcond=None)
    B = x[0:(m+1)]
    A_tail = x[(m+1):]              # A[1..n]
    A = np.empty(n+1, dtype=float)
    A[0] = A0
    A[1:] = A_tail
    return A, B

def poles_from_rational_den(A: np.ndarray):
    """
    D(s) = sum_{q=0}^n A_q s^q, A 实系数。
    返回 ω_j = i s_j （复频率）。
    """
    coeffs = A[::-1]    # np.roots 最高次在前
    s_roots = np.roots(coeffs)
    omega_roots = 1j * s_roots
    return omega_roots

def eigenfreqs_via_eq18_pade(
    k2: float,
    params: dict,
    n: int = 14,
    m: int = 8,
    f_max_hz: float = 40.0,
    num_samples: int = 2000,
    weight_alpha: float = 1.0
):

    A, B = fit_rational_T_pade(k2, n=n, m=m, f_max_hz=f_max_hz,
                               num_samples=num_samples, weight_alpha=weight_alpha)
    roots = poles_from_rational_den(A)
    out = []
    for w in roots:
        if np.imag(w) < 0:
            f = abs(np.real(w))/(2*np.pi)
            g = -np.imag(w)
            # print(f"  ω=({np.real(w):.3f})+i({np.imag(w):.3f}), f={f:.3f} Hz, γ={g:.3f}")
            out.append((float(f), float(g), complex(w)))
    out.sort(key=lambda x: x[1])
    return out

#==========直接求根==========#

def find_roots_dispersion_direct(k2: float, params: dict, seeds: List[complex]) -> List[Tuple[float, float, complex]]:
    out, roots = [], []
    F, T = F_or_T(k2, params)
    for z0 in seeds:
        try:
            w_root = mp.findroot(F, z0)  # 复初值
            # 合并去重
            if not any(abs(w_root - z) < 1e-6 for z in roots):
                roots.append(w_root)
        except:  # 没收敛就跳过
            pass

    for w in roots:
        if mp.im(w) < 0:  # 稳定根
            f = abs(mp.re(w)) / (2*mp.pi)
            g = -mp.im(w)
            out.append((float(f), float(g), complex(w)))
    out.sort(key=lambda x: x[1])
    return out

#==========呈现==========#

def pick_root_by_band(cands, lo_hz, hi_hz):
    if not cands:
        return None
    band = [x for x in cands if (lo_hz <= x[0] <= hi_hz)]
    if band:
        band.sort(key=lambda t: t[1])
        return (band[0][0], band[0][1])
    return None

def compute_sw_alpha_for_modes(evals,mode_indices,params,slow_band=(0.0, 3.5),alpha_band=(8.0, 13.0)):
    results = {}
    for midx in mode_indices:
        k2 = float(evals[midx])
        print(f"\n[direct scan] MODE {midx}, k^2={k2:.6g}")
        direct = find_roots_dispersion_direct(k2, params, seeds)
        sw_d   = pick_root_by_band(direct, slow_band[0], slow_band[1])
        alpha_d = pick_root_by_band(direct, alpha_band[0], alpha_band[1])
        print(f"[direct pick] MODE {midx}")
        if sw_d:
            print(f"  慢波: f={sw_d[0]:.3f} Hz, γ={sw_d[1]:.3f} s^-1")
        if alpha_d:
            print(f"  α 波: f={alpha_d[0]:.3f} Hz, γ={alpha_d[1]:.3f} s^-1")

        approx_list = eigenfreqs_via_eq18_pade(k2, params, n=14, m=8,
                                               f_max_hz=20.0, num_samples=2000)
        sw_a   = pick_root_by_band(approx_list, slow_band[0], slow_band[1])
        alpha_a= pick_root_by_band(approx_list, alpha_band[0], alpha_band[1])
        print(f"[approx pick] MODE {midx}")
        if sw_a:
            print(f"  慢波: f={sw_a[0]:.3f} Hz, γ={sw_a[1]:.3f} s^-1")
        if alpha_a:
            print(f"  α 波: f={alpha_a[0]:.3f} Hz, γ={alpha_a[1]:.3f} s^-1")

        results[midx] = {"slow": sw_d, "alpha": alpha_d, "k2": k2}
    return results

def plot_sw_alpha_bars(results, mode_order, mode_labels=None):
    if mode_labels is None:
        mode_labels = [str(m) for m in mode_order]

    fs, gs = [], []   # 慢波
    fa, ga = [], []   # α波
    for midx in mode_order:
        sw = results[midx]["slow"]
        al = results[midx]["alpha"]
        fs.append(sw[0] if sw else np.nan)
        gs.append(sw[1] if sw else np.nan)
        fa.append(al[0] if al else np.nan)
        ga.append(al[1] if al else np.nan)

    x = np.arange(len(mode_order))
    figsize=(8, 3)
    fig1, axes1 = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    # (a) f_s
    axes1[0].bar(x, fs)
    axes1[0].set_xticks(x)
    axes1[0].set_xticklabels(mode_labels)
    axes1[0].set_ylabel(r"$f_s$ (Hz)")
    axes1[0].set_xlabel(r"$\lambda\mu$")
    axes1[0].set_title("(a) Slow-wave frequency")
    # (b) γ_s
    axes1[1].bar(x, gs)
    axes1[1].set_xticks(x)
    axes1[1].set_xticklabels(mode_labels)
    axes1[1].set_ylabel(r"$\gamma_s$ (s$^{-1}$)")
    axes1[1].set_xlabel(r"$\lambda\mu$")
    axes1[1].set_title("(b) Slow-wave damping")

    fig2, axes2 = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    # (a) f_α
    axes2[0].bar(x, fa)
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels(mode_labels)
    axes2[0].set_ylabel(r"$f_\alpha$ (Hz)")
    axes2[0].set_xlabel(r"$\lambda\mu$")
    axes2[0].set_title("(a) Alpha frequency")
    # (b) γ_α
    axes2[1].bar(x, ga)
    axes2[1].set_xticks(x)
    axes2[1].set_xticklabels(mode_labels)
    axes2[1].set_ylabel(r"$\gamma_\alpha$ (s$^{-1}$)")
    axes2[1].set_xlabel(r"$\lambda\mu$")
    axes2[1].set_title("(b) Alpha damping")

    return (fig1, fig2)

def main():
    # 设置数据目录
    #BASE_PATH = '/mnt/'
    BASE_PATH = '/home/wmy/Documents/'
    sub = '100307'
    resting_data_dir = os.path.join(BASE_PATH, 'Resting_1', sub)
    struct_data_dir = os.path.join(BASE_PATH, 'Structure', sub)
    
    evals = np.loadtxt("/home/wmy/work/geometry/eigenvalues.txt")

    mode_order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    mode_labels = ["00","1−1","10","11","2−2","2−1","20","21","22"]
    results = compute_sw_alpha_for_modes(evals, mode_order, NFT_PARAMS)
    fig_slow, fig_alpha = plot_sw_alpha_bars(results, mode_order, mode_labels)
    fig_slow.savefig("fig6_like_slow.png", dpi=200)
    fig_alpha.savefig("fig7_like_alpha.png", dpi=200)
    plt.show()

    '''f1, g1 = alpha_results[2]
    f2, g2 = alpha_results[3]
    gamma = 0.5*(g1 + g2)  # 使用真实阻尼；你也可以用 min(g1, g2)

    # —— 放慢播放—— #
    fB = abs(f2 - f1)                         # 拍频
    duration = max(3.0/max(fB, 1e-6), 30.0)   # 至少覆盖 ~3 个拍频周期
    fps = 10                                  # 降低帧率
    max_frames = min(int(duration * fps), 30) 

    # —— 非零相位差才能看到旋转/行波 —— #
    visualize_two_mode_beating(
        vertices=vertices, faces=faces, emodes=emodes,
        mode_idx1=2, mode_idx2=3,
        f1=f1, f2=f2,gamma=gamma,
        duration=duration,fps=fps,
        phase1=0.0, phase2=np.pi/2,
        max_frames=max_frames
    )'''

if __name__ == "__main__":
    main()