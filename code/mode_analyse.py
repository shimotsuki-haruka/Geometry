'''import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from visualize_brain import load_surface_and_activity

# ===== 基础工具 =====

def calc_eigendecomposition(data: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    最小二乘 beta: basis @ beta ≈ data
    data: (N,) 或 (N, T)
    basis: (N, K)
    return: beta (K,) 或 (K, T)
    """
    return np.linalg.pinv(basis) @ data

def calc_parcellate(parc: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    把 vertex-level data (N, ...) 平均到 parcels，parc 是 (N,)，值>0 为 parcel index
    return: (P, ...) ；P = unique(parc>0) 的个数
    """
    parc = np.asarray(parc)
    labels = np.unique(parc[parc > 0])
    out = []
    for lab in labels:
        mask = (parc == lab)
        vals = data[mask]
        out.append(np.mean(vals, axis=0) if vals.ndim > 1 else np.mean(vals))
    return np.asarray(out)

def zscore(ts: np.ndarray) -> np.ndarray:
    mu  = ts.mean(axis=0, keepdims=True)
    std = ts.std(axis=0, keepdims=True) + 1e-12
    return (ts - mu) / std

def upper_triangle_vector(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(M, k=1)
    return M[iu]

def modal_power(beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """beta (K,) 或 (K,T) -> 每个模态功率 & 归一化功率"""
    if beta.ndim == 1:
        p = beta**2
    else:
        p = np.sum(beta**2, axis=1)
    norm = p / (np.sum(p) + 1e-12)
    return p, norm

# ===== 1) 任务 fMRI 空间图重建 =====

def reconstruct_task_map(
    eigenmodes: np.ndarray,       # (N, K)
    task_map: np.ndarray,         # (N,)
    cortex_mask: Optional[np.ndarray], # (N,) 0/1；为 None 则用全体
    parc: Optional[np.ndarray],   # (N,) parcel id；为 None 则不做 parcel 比较
    num_modes: int = 50
):
    if cortex_mask is None:
        cortex_idx = np.arange(eigenmodes.shape[0])
    else:
        cortex_idx = np.where(cortex_mask > 0)[0]

    recon_corr_vertex = []
    recon_corr_parc   = []

    # 存系数（便于复现“用前 m 个模态”的重建结果）
    recon_beta = np.zeros((num_modes, num_modes))

    for m in range(1, num_modes+1):
        B = eigenmodes[cortex_idx, :m]
        beta = calc_eigendecomposition(task_map[cortex_idx], B)
        recon_beta[:m, m-1] = beta

        # 顶点水平的相关
        recon_vertex = B @ beta
        c_v = np.corrcoef(task_map[cortex_idx], recon_vertex)[0, 1]
        recon_corr_vertex.append(c_v)

        # parcel 水平的相关
        if parc is not None:
            recon_full = eigenmodes[:, :m] @ recon_beta[:m, m-1]
            c_p = np.corrcoef(
                calc_parcellate(parc, task_map),
                calc_parcellate(parc, recon_full)
            )[0, 1]
        else:
            c_p = np.nan
        recon_corr_parc.append(c_p)

    # 画图
    plt.figure(figsize=(6,4))
    plt.plot(range(1, num_modes+1), recon_corr_vertex, 'k-', lw=2, label='vertex')
    if not np.all(np.isnan(recon_corr_parc)):
        plt.plot(range(1, num_modes+1), recon_corr_parc, 'b-', lw=2, label='parcellated')
        plt.legend(loc='lower right', frameon=False)
    plt.ylim(0, 1); plt.xlim(1, num_modes)
    plt.xlabel('number of modes'); plt.ylabel('reconstruction accuracy (corr)')
    plt.title('Task map reconstruction')
    plt.tight_layout()
    plt.savefig('task_reconstruction_accuracy.png', dpi=200)

    # 返回“用全部 num_modes 模态”的重建结果，以便你可视化
    recon_full = eigenmodes[:, :num_modes] @ recon_beta[:num_modes, num_modes-1]
    return recon_full, np.array(recon_corr_vertex), np.array(recon_corr_parc)

# ===== 2) 静息态时空图 + FC 重建 =====

def reconstruct_rest_FC(
    eigenmodes: np.ndarray,     # (N, K)
    timeseries: np.ndarray,     # (T, N)
    cortex_mask: Optional[np.ndarray], # (N,)
    parc: np.ndarray,           # (N,) —— 这个部分必须要有 parcel 才能算 FC 的 parcel 级相关
    num_modes: int = 50
):
    if cortex_mask is None:
        cortex_idx = np.arange(eigenmodes.shape[0])
    else:
        cortex_idx = np.where(cortex_mask > 0)[0]

    T = timeseries.shape[0]
    # 真实 FC（parcel 级）
    data_parc_emp = calc_parcellate(parc, timeseries.T)   # (P, T)
    data_parc_emp = zscore(data_parc_emp.T)               # (T, P)
    FC_emp = (data_parc_emp.T @ data_parc_emp) / T        # (P, P)
    FC_emp_vec = upper_triangle_vector(FC_emp)

    corr_parc_list = []

    for m in range(1, num_modes+1):
        B = eigenmodes[cortex_idx, :m]                    # (Nc, m)
        beta = calc_eigendecomposition(timeseries[:, cortex_idx].T, B)  # (m, T)
        recon_full = eigenmodes[:, :m] @ beta             # (N, T)

        data_parc_rec = calc_parcellate(parc, recon_full) # (P, T)
        data_parc_rec = zscore(data_parc_rec.T)           # (T, P)
        FC_rec = (data_parc_rec.T @ data_parc_rec) / T
        FC_rec_vec = upper_triangle_vector(FC_rec)

        corr_parc_list.append(np.corrcoef(FC_emp_vec, FC_rec_vec)[0, 1])

    # 画图
    plt.figure(figsize=(5,4))
    plt.plot(range(1, num_modes+1), corr_parc_list, 'b-', lw=2)
    plt.ylim(0, 1); plt.xlim(1, num_modes)
    plt.xlabel('number of modes'); plt.ylabel('FC reconstruction accuracy (corr)')
    plt.title('Rest FC reconstruction')
    plt.tight_layout()
    plt.savefig('rest_reconstruction_accuracy.png', dpi=200)

    return np.array(corr_parc_list)

# ===== 3) 空间图的模态功率谱 =====

def modal_power_of_map(
    eigenmodes: np.ndarray,   # (N, K)
    spatial_map: np.ndarray,  # (N,)
    cortex_mask: Optional[np.ndarray],
    num_modes: int = 50
):
    if cortex_mask is None:
        cortex_idx = np.arange(eigenmodes.shape[0])
    else:
        cortex_idx = np.where(cortex_mask > 0)[0]

    B = eigenmodes[cortex_idx, :num_modes]
    beta = calc_eigendecomposition(spatial_map[cortex_idx], B)  # (num_modes,)
    p, pnorm = modal_power(beta)

    plt.figure(figsize=(6,3))
    plt.bar(np.arange(1, num_modes+1), pnorm)
    plt.yscale('log'); plt.xlim(1, num_modes)
    plt.xlabel('mode'); plt.ylabel('normalized power (log scale)')
    plt.title('Modal power spectrum')
    plt.tight_layout()
    plt.savefig('modal_power_spectrum.png', dpi=200)
    return beta, pnorm
'''


import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import image
from volume_eigenmode import compute_volume_eigenmodes_from_mask

def zscore_rows(X: np.ndarray) -> np.ndarray:
    m  = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-12
    return (X - m) / sd

def vertex_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    计算每个顶点的面积权重（每个三角形面积均分到三个顶点）。
    verts: (V,3), faces: (F,3) int
    return: area (V,), >=0
    """
    v0 = verts[faces[:,0]]
    v1 = verts[faces[:,1]]
    v2 = verts[faces[:,2]]
    tri_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)  # (F,)
    area = np.zeros(verts.shape[0], dtype=np.float64)
    np.add.at(area, faces[:,0], tri_area/3.0)
    np.add.at(area, faces[:,1], tri_area/3.0)
    np.add.at(area, faces[:,2], tri_area/3.0)
    area += 1e-18  # 避免 0
    return area

def _load_LH_native_timeseries_and_surface(resting_dir: str, struct_dir: str, sub: str):
    """
    返回：
      Y_TxV : (T, V_all)  —— LH native.func.gii 的全部顶点时序（不去 medial wall）
      V     : (V_all, 3)  —— LH white.native.surf.gii 顶点
      F     : (F_all, 3)  —— LH white.native.surf.gii 面
      m     : (V_all,)    —— atlasroi 掩膜(仅打印校验，不用于裁剪)
    """
    # 时序
    func_gii = os.path.join(resting_dir,
        "MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.L.native.func.gii")
    g = nib.load(func_gii)
    Y_TxV = np.vstack([arr.data for arr in g.darrays]).astype(np.float64)  # (T,V_all)

    # 表面（white）
    surf_gii = os.path.join(struct_dir, f"MNINonLinear/Native/{sub}.L.white.native.surf.gii")
    s = nib.load(surf_gii)
    V = s.darrays[0].data.astype(np.float64)   # (V_all,3)
    F = s.darrays[1].data.astype(np.int32)     # (F_all,3)

    # atlasroi（仅用于核对）
    mask_gii = os.path.join(struct_dir, f"MNINonLinear/Native/{sub}.L.atlasroi.native.shape.gii")
    m = nib.load(mask_gii).darrays[0].data.astype(bool)  # (V_all,)

    # 调试输出（你要的节点数核对）
    print(f"[TEST] func 顶点数 (V_all)   = {Y_TxV.shape[1]}")
    print(f"[TEST] surface 顶点数        = {V.shape[0]}")
    print(f"[TEST] 三角形数 (faces)      = {F.shape[0]}")
    print(f"[TEST] atlasroi 皮层顶点数   = {int(m.sum())} / {m.size} ({m.sum()/m.size*100:.2f}%)")
    if Y_TxV.shape[1] != V.shape[0] or V.shape[0] != m.size:
        raise ValueError("func 顶点数、surface 顶点数、atlasroi 长度不一致。")

    return Y_TxV, V, F, m

def _load_LH_native_parcellation_a2009s(struct_dir: str, sub: str, parc_path: str|None=None):
    if parc_path is None:
        parc_path = os.path.join(struct_dir, f"MNINonLinear/Native/{sub}.L.aparc.a2009s.native.label.gii")
    lab_full = nib.load(parc_path).darrays[0].data.astype(int)  # (V_all,)
    # 只在聚合时用 lab>0；此处不裁掉 medial wall
    print(f"[TEST] a2009s 非零标签数量(去重) = {np.unique(lab_full[lab_full>0]).size}")
    return lab_full

def _parcellate_vertex_timeseries(Y_TxV: np.ndarray, labels_V: np.ndarray):
    """(T,V_all) + (V_all,) -> (P,T), parcel_ids(P,)；仅 labels>0 参与平均"""
    labs = np.unique(labels_V[labels_V > 0])
    P, T = labs.size, Y_TxV.shape[0]
    X = Y_TxV.T  # (V_all,T)
    pt = np.zeros((P, T), dtype=np.float32)
    for i, pid in enumerate(labs):
        pt[i] = X[labels_V == pid].mean(axis=0)
    return pt, labs


# -------- A) 经验 FC（LH native + aparc.a2009s）--------
def ptseries_and_fc_surface_native_LH_a2009s_noMW(resting_data_dir: str,
                                                  struct_data_dir: str,
                                                  sub: str,
                                                  out_prefix: str="LH_a2009s_full"):
    Y_TxV, V, F, m = _load_LH_native_timeseries_and_surface(resting_data_dir, struct_data_dir, sub)
    lab_full = _load_LH_native_parcellation_a2009s(struct_data_dir, sub, parc_path=None)

    pt_emp, parcel_ids = _parcellate_vertex_timeseries(Y_TxV, lab_full)  # (P,T)
    # z-score along time per parcel
    Z = (pt_emp - pt_emp.mean(1, keepdims=True)) / (pt_emp.std(1, keepdims=True)+1e-12)
    FC_emp = np.corrcoef(Z)

    # 保存 & 画图
    np.save(f"{out_prefix}.empirical.ptseries.npy", pt_emp)
    np.save(f"{out_prefix}.empirical.FC.npy", FC_emp)
    np.save(f"{out_prefix}.parcel_ids.npy", parcel_ids)

    plt.figure(figsize=(5.2,4.4))
    im = plt.imshow(FC_emp, vmin=-1.0, vmax=1.0, cmap="RdBu_r", interpolation="nearest")
    plt.title(f"LH empirical FC (a2009s, P={len(parcel_ids)})"); plt.xlabel("Parcel"); plt.ylabel("Parcel")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04); cbar.set_label("r")
    plt.tight_layout(); plt.savefig(f"{out_prefix}.empirical.FC.png", dpi=200); plt.close()
    print(f"[saved] {out_prefix}.empirical.FC.png")

    return pt_emp, FC_emp, parcel_ids, (Y_TxV, lab_full)

# -------- B) 模态重建评估（LH native + aparc.a2009s）--------
def reconstruct_FC_with_modes_surface_LH_a2009s_noMW(eigenmodes: np.ndarray,  # (V_all,K)
                                                     resting_data_dir: str,
                                                     struct_data_dir: str,
                                                     sub: str,
                                                     out_prefix: str="LH_a2009s_full"):
    # 经验 FC（也取回 Y_TxV 和 labels）
    pt_emp, FC_emp, parcel_ids, (Y_TxV, labels_V) = ptseries_and_fc_surface_native_LH_a2009s_noMW(
        resting_data_dir, struct_data_dir, sub, out_prefix=out_prefix
    )
    T, V_all = Y_TxV.shape
    P = len(parcel_ids)

    # 基检查
    print(f"[TEST] eigenmodes 形状: {eigenmodes.shape}；应为 (V_all={V_all}, K)")
    if eigenmodes.shape[0] != V_all:
        raise ValueError("eigenmodes 行数与 V_all 不一致，请确认模态是在 LH white.native 顶点顺序上计算的。")

    # 预计算
    Phi = eigenmodes.astype(np.float64)      # (V_all,K)
    Y   = Y_TxV.astype(np.float64)           # (T,V_all)
    C   = Phi.T @ Y.T                        # (K,T)
    G   = Phi.T @ Phi                        # (K,K)
    K   = Phi.shape[1]
    ridge = 1e-8 * (np.trace(G)/K)
    G.flat[::G.shape[0]+1] += ridge
    Ginv = np.linalg.inv(G)

    # 上三角索引
    iu = np.triu_indices(P, k=1)
    v_emp = FC_emp[iu]

    # 逐模态重建
    Mmax = min(200, K)
    acc_curve = []
    FC_last = None
    for M in range(1, Mmax+1):
        Beta = Ginv[:M,:M] @ C[:M,:]          # (M,T)
        Yhat = (Phi[:,:M] @ Beta).T           # (T,V_all)

        pt_rec, _ = _parcellate_vertex_timeseries(Yhat, labels_V)
        Zr = (pt_rec - pt_rec.mean(1, keepdims=True)) / (pt_rec.std(1, keepdims=True)+1e-12)
        FC_rec = np.corrcoef(Zr)

        r = np.corrcoef(v_emp, FC_rec[iu])[0,1]
        acc_curve.append(float(r))
        if M == Mmax:
            FC_last = FC_rec

    # 精度曲线
    plt.figure(figsize=(6,4))
    plt.plot(range(1, Mmax+1), acc_curve, "b-", lw=2)
    plt.xlabel("number of modes"); plt.ylabel("FC reconstruction accuracy (r)")
    plt.title("LH (a2009s, full surface) – FC reconstruction")
    plt.ylim(0,1); plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_prefix}.fc_recon_curve.png", dpi=200); plt.close()
    print(f"[saved] {out_prefix}.fc_recon_curve.png")

    # 上=经验/下=重建
    mix = np.zeros_like(FC_emp); il = np.tril_indices(P, k=-1)
    mix[iu] = FC_emp[iu]; mix[il] = FC_last[il]; np.fill_diagonal(mix, 1.0)
    plt.figure(figsize=(5.2,4.4))
    im = plt.imshow(mix, vmin=-1.0, vmax=1.0, cmap="RdBu_r", interpolation="nearest")
    plt.title("Empirical (upper) vs Reconstructed (lower)"); plt.xlabel("Parcel"); plt.ylabel("Parcel")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04); cbar.set_label("FC (r)")
    plt.tight_layout(); plt.savefig(f"{out_prefix}.emp_upper_rec_lower.png", dpi=200); plt.close()

    # 上三角散点 + 单独重建 FC
    plt.figure(figsize=(5,5))
    plt.scatter(v_emp, FC_last[iu], s=6, alpha=0.5)
    plt.plot([-1,1], [-1,1], "k--", lw=1)
    plt.xlim(-1,1); plt.ylim(-1,1)
    r_last = np.corrcoef(v_emp, FC_last[iu])[0,1]
    plt.xlabel("Empirical FC (upper triangle)")
    plt.ylabel(f"Reconstructed FC (M={Mmax})")
    plt.title(f"Upper-triangle corr r = {r_last:.3f}")
    plt.tight_layout(); plt.savefig(f"{out_prefix}.emp_vs_recon_scatter.png", dpi=200); plt.close()

    plt.figure(figsize=(5.2,4.4))
    im = plt.imshow(FC_last, vmin=-1.0, vmax=1.0, cmap="RdBu_r", interpolation="nearest")
    plt.title(f"Reconstructed FC (M={Mmax})"); plt.xlabel("Parcel"); plt.ylabel("Parcel")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04); cbar.set_label("FC (r)")
    plt.tight_layout(); plt.savefig(f"{out_prefix}.FC_reconstructed_M{Mmax}.png", dpi=200); plt.close()

    return np.asarray(acc_curve), FC_emp, FC_last

#--------------系数&timestep-----------------#
def compute_modal_coefficients(Y_TxV: np.ndarray,
                               eigenmodes_VxK: np.ndarray,
                               ridge_scale: float = 1e-8):
    """
    输入:
      Y_TxV          : (T, V_all) 顶点时序（不去墙）
      eigenmodes_VxK : (V_all, K) 表面模态（与白质表面顶点顺序一致）
    返回:
      Beta_KxT  : (K, T) 每个时间点的模态系数
      Ginv_KxK  : (K, K) (Phi^T Phi)^(-1)（可复用）
    """
    Phi = eigenmodes_VxK.astype(np.float64)  # (V,K)
    Y   = Y_TxV.astype(np.float64)           # (T,V)

    # 预计算
    C = Phi.T @ Y.T                # (K,T) = Phi^T * Y^T
    G = Phi.T @ Phi                # (K,K)
    K = G.shape[0]
    ridge = ridge_scale * (np.trace(G) / max(1, K))
    G.flat[::K+1] += ridge
    Ginv = np.linalg.inv(G)

    Beta = Ginv @ C                # (K,T)
    return Beta, Ginv

def compute_modal_coefficients_weighted(Y_TxV: np.ndarray,
                                        eigenmodes_VxK: np.ndarray,
                                        verts_Vx3: np.ndarray,
                                        faces_Fx3: np.ndarray,
                                        demean_spatial: bool = True):
    """
    在面积权 A 下做正交化与投影：
      - S = Φᵀ A Φ
      - Φ̃ = Φ · S^{-1/2}   （后正交化，使 Φ̃ᵀ A Φ̃ = I）
      - C = Φ̃ᵀ A Yᵀ
      - Beta = C            （因为 G=I）
    返回:
      Beta_KxT, Phi_tilde_VxK, S_KxK
    """
    V = Y_TxV.shape[1]
    if eigenmodes_VxK.shape[0] != V:
        raise ValueError(f"V 不一致: Y has V={V}, modes has V={eigenmodes_VxK.shape[0]}")

    # (可选) 每个时间点去掉空间均值（常量模态不主导）
    Y = Y_TxV.copy().astype(np.float64)
    if demean_spatial:
        Y -= Y.mean(axis=1, keepdims=True)

    # 顶点面积向量 a -> 质量矩阵 A（对角）
    a = vertex_areas(verts_Vx3, faces_Fx3)            # (V,)
    # 计算 S = Φᵀ A Φ
    Phi = eigenmodes_VxK.astype(np.float64)           # (V,K)
    S = (Phi.T * a) @ Phi                              # (K,K)

    # S^{-1/2}：对称的逆平方根
    evals, evecs = np.linalg.eigh(S)
    evals_clamped = np.clip(evals, 1e-12, None)
    S_inv_sqrt = (evecs * (evals_clamped**-0.5)) @ evecs.T   # (K,K)

    # 后正交化的基 Φ̃
    Phi_tilde = Phi @ S_inv_sqrt                              # (V,K) 使 Φ̃ᵀ A Φ̃ ≈ I

    # 面积加权的投影：C = Φ̃ᵀ A Yᵀ
    Beta = (Phi_tilde.T * a) @ Y.T                            # (K,T)

    return Beta, Phi_tilde, S

def plot_representative_beta(Beta_KxT: np.ndarray,
                             out_prefix: str = "LH_a2009s_full",
                             max_show: int = 8):
    """
    选择“有代表性”的模态并画系数随时间曲线：
      - 固定长波模态: 1,2,3
      - 最高方差的模态: 若干个（补足到 max_show 条）
      - 再加中/高频代表 (K//2, K)（如果还没被选中）
    """
    K, T = Beta_KxT.shape
    # 1) 先选低阶（长波）
    chosen = set([0, 1, 2])  # 0-based -> 模态1,2,3

    # 2) 选方差最高的模态
    var = Beta_KxT.var(axis=1)
    top_var_idx = list(np.argsort(var)[::-1])
    for idx in top_var_idx:
        chosen.add(int(idx))
        if len(chosen) >= max_show - 2:
            break

    # 3) 加入中/高频代表
    for idx in [K//2 - 1, K - 1]:
        if 0 <= idx < K:
            chosen.add(int(idx))

    chosen = sorted(chosen)[:max_show]

    # 画图
    plt.figure(figsize=(10, 4))
    for i in chosen:
        plt.plot(Beta_KxT[i], lw=1.2, label=f"mode {i+1}")
    plt.xlabel("time (frames)")
    plt.ylabel("β_n(t)")
    plt.title("Representative modal coefficients over time")
    plt.legend(ncol=4, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.beta_representatives.png", dpi=200)
    plt.close()
    print(f"[saved] {out_prefix}.beta_representatives.png")

#---------------power spectrum----------------#
def plot_modal_power_spectrum_from_beta(Beta_KxT: np.ndarray, out_prefix: str,
                                        logy: bool = True):
    power = np.sum(Beta_KxT**2, axis=1)            # (K,)
    pnorm = power / (power.sum() + 1e-12)
    pcum  = np.cumsum(pnorm)

    np.save(f"{out_prefix}.modal_power.npy", power)
    np.save(f"{out_prefix}.modal_power_norm.npy", pnorm)
    np.save(f"{out_prefix}.modal_power_cum.npy", pcum)

    x = np.arange(1, Beta_KxT.shape[0]+1)
    fig, ax1 = plt.subplots(figsize=(8,3.6))
    ax1.bar(x, pnorm, width=1.0)
    ax1.set_xlim(1, x[-1])
    ax1.set_xlabel("mode")
    ax1.set_ylabel("normalized power")
    if logy:
        ax1.set_yscale("log")
        ax1.set_ylabel("normalized power (log)")

    ax2 = ax1.twinx()
    ax2.plot(x, pcum, "k-", lw=1.5)
    ax2.set_ylabel("cumulative power")
    ax2.set_ylim(0, 1.02)

    plt.title("Modal power spectrum (β² summed over time)")
    fig.tight_layout()
    plt.savefig(f"{out_prefix}.modal_power_spectrum.png", dpi=200)
    plt.close()
    print(f"[saved] {out_prefix}.modal_power_spectrum.png")

def main():
    # 路径设置（与你环境一致）
    BASE_PATH = "/home/wmy/Documents/"
    sub = "100307"
    resting_dir = os.path.join(BASE_PATH, "REST1", sub)
    struct_dir  = os.path.join(BASE_PATH, "Structure",  sub)
    out_dir = os.path.join(BASE_PATH, "work/geometry/results/mode_analyse", f"{sub}.LH")
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    # emodes_lh: (V_all, K)，与 LH native 顶点顺序一致
    # 例如 emodes_lh = np.load("/path/to/LH_native_eigenmodes.npy")
    # 若 emodes_lh 已经是 (Vmask, K)，也可以正常运行。


    emodes_path = "/home/wmy/work/geometry/eigenmodes.txt"
    emodes = np.loadtxt(emodes_path)  # (29696, K)  —— 必须是 LH-native 计算的模态
    
    pt_emp, FC_emp, pids, (Y_TxV, labels_V) = ptseries_and_fc_surface_native_LH_a2009s_noMW(
        resting_dir, struct_dir, sub, out_prefix=out_dir
    )

    # B) 重建评估
    acc, FC_emp2, FC_last = reconstruct_FC_with_modes_surface_LH_a2009s_noMW(
        emodes, resting_dir, struct_dir, sub, out_prefix=out_dir
    )

    Y_TxV2, Vxyz, Ftri, _ = _load_LH_native_timeseries_and_surface(resting_dir, struct_dir, sub)
    # 注意：Y_TxV2==Y_TxV，只是为了拿到 Vxyz/Ftri。也可以把 Vxyz/Ftri在别处传入。

    # 1) 面积加权的系数（关键修复）
    Beta_w, Phi_tilde, S = compute_modal_coefficients_weighted(
        Y_TxV, emodes, Vxyz, Ftri, demean_spatial=True
    )

    # 2) 代表性 β 曲线（用 Beta_w）
    plot_representative_beta(Beta_w, out_prefix=out_dir)

    # 3) 模态功率谱（用 Beta_w）
    plot_modal_power_spectrum_from_beta(Beta_w, out_prefix=out_dir)

    print("[done] 三张图已保存到当前目录。")

if __name__ == "__main__":
    main()



#写成体积的了
'''def ptseries_and_fc(resting_data_dir: str, struct_data_dir: str):
    """
    读取 wmparc 与 rfMRI；将 wmparc 重采样到功能空间（nearest）；
    用所有非0标签做 parcel；返回 (ptseries_PxT, FC_PxP, parcel_ids, labels_1d)
    并保存经验 FC 热图到 'wmparc_alllabels.FC.png'。
    """
    # 固定参数（内部定义即可）
    out_prefix  = "empFC"
    min_vox     = 1          # 真·所有非0标签：改为 >1 可滤除极小簇
    zscore_time = False      # 构造经验 FC 前是否先逐行z-score
    fisher_z    = False      # 是否对FC做Fisher z

    # --- 读取 & 重采样 wmparc 到功能空间 ---
    lab_img  = nib.load(os.path.join(struct_data_dir,  "MNINonLinear/wmparc.nii.gz"))
    func_img = nib.load(os.path.join(resting_data_dir, "MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"))
    lab_res  = image.resample_to_img(lab_img, func_img.slicer[..., 0], interpolation="nearest")

    lab  = lab_res.get_fdata().astype(np.int32)       # (X,Y,Z)
    func = func_img.get_fdata()                       # (X,Y,Z,T)
    labels_1d     = lab.reshape(-1)                   # (N,)
    timeseries_TxN = func.reshape(-1, func.shape[3]).T# (T,N)

    # --- 体素过滤（可关闭）---
    if min_vox > 1:
        for pid in np.unique(labels_1d[labels_1d > 0]):
            if np.count_nonzero(labels_1d == pid) < min_vox:
                labels_1d[labels_1d == pid] = 0

    # --- parcel 汇聚：(T,N)->(P,T) ---
    labs = np.unique(labels_1d[labels_1d > 0])
    P, T = labs.size, timeseries_TxN.shape[0]
    pt_PxT = np.zeros((P, T), dtype=np.float32)
    X = timeseries_TxN.T  # (N,T)
    for i, pid in enumerate(labs):
        pt_PxT[i] = X[labels_1d == pid].mean(axis=0)

    # --- 经验 FC（可选zscore、fisher_z内部控制）---
    X_fc = zscore_rows(pt_PxT) if zscore_time else pt_PxT
    FC = np.corrcoef(X_fc)
    if fisher_z:
        FC = np.arctanh(np.clip(FC, -0.999999, 0.999999))

    plt.figure(figsize=(6,5))
    im = plt.imshow(FC, vmin=-1.0, vmax=1.0, cmap="RdBu_r", interpolation="nearest")
    plt.title(f"Empirical FC (P={P})"); plt.xlabel("Region"); plt.ylabel("Region")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04); cbar.set_label("FC (r)" if not fisher_z else "Fisher z")
    plt.tight_layout(); plt.savefig(f"{out_prefix}.png", dpi=200); plt.close()

    return pt_PxT, FC, labs, labels_1d
    
def reconstruct_FC_with_modes(eigenmodes: np.ndarray,      # (N,K)
                              timeseries_TxN: np.ndarray,  # (T,N)
                              labels_1d: np.ndarray):      # (N,)
    """
    用几何本征模态重建 -> parcel 汇聚 -> FC；
    输出重建精度曲线、上=经验/下=重建对照图、上三角散点图。
    返回 (acc_curve, FC_emp, FC_last)。
    """
    # 固定参数（内部定义）
    zscore_time  = True
    use_max_modes = 200   # 最多用多少个模态（受限于K）
    ridge_eps    = 1e-8

    T, N = timeseries_TxN.shape
    K = eigenmodes.shape[1]
    assert eigenmodes.shape[0] == N and labels_1d.shape[0] == N
    num_modes_list = list(range(1, min(use_max_modes, K) + 1))

    # --- 经验 FC（parcel级）---
    labs = np.unique(labels_1d[labels_1d > 0])
    P = labs.size
    pt_emp = np.zeros((P, T), dtype=np.float32)
    X = timeseries_TxN.T
    for i, pid in enumerate(labs):
        pt_emp[i] = X[labels_1d == pid].mean(axis=0)
    FC_emp = np.corrcoef(zscore_rows(pt_emp) if zscore_time else pt_emp)
    iu = np.triu_indices(P, k=1)        # 上三角索引（不含对角）
    v_emp = FC_emp[iu]

    # --- 预计算（和 MATLAB demo 同步）：C, Ginv ---
    Phi = eigenmodes.astype(np.float64) # (N,K)
    Y   = timeseries_TxN.astype(np.float64)
    C   = Phi.T @ Y.T                   # (K,T)
    G   = Phi.T @ Phi                   # (K,K)
    G.flat[::G.shape[0]+1] += ridge_eps * (np.trace(G) / K)
    Ginv = np.linalg.inv(G)

    # --- 不同模态数下的重建精度曲线 ---
    acc_curve = []
    FC_last = None
    for idx, M in enumerate(num_modes_list):
        Beta = Ginv[:M,:M] @ C[:M,:]            # (M,T)
        Yhat = (Phi[:,:M] @ Beta).T             # (T,N)

        # parcel 汇聚
        pt_rec = np.zeros((P, T), dtype=np.float32)
        Xh = Yhat.T
        for i, pid in enumerate(labs):
            pt_rec[i] = Xh[labels_1d == pid].mean(axis=0)

        FC_rec = np.corrcoef(zscore_rows(pt_rec) if zscore_time else pt_rec)
        r = np.corrcoef(v_emp, FC_rec[iu])[0, 1]
        acc_curve.append(float(r))
        if idx == len(num_modes_list) - 1:
            FC_last = FC_rec

    # --- 单独保存最后一个模态数的重建 FC 热图 ---
    plt.figure(figsize=(6,5))
    im = plt.imshow(FC_last, vmin=-1.0, vmax=1.0, cmap="RdBu_r", interpolation="nearest")
    plt.title(f"Reconstructed FC (M={num_modes_list[-1]})")
    plt.xlabel("Region"); plt.ylabel("Region")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04); cbar.set_label("FC (r)")
    plt.tight_layout()
    plt.savefig(f"FC_reconstructed_M{num_modes_list[-1]}.png", dpi=200)
    plt.close()
    #print(f"[saved] {out_prefix}.FC_reconstructed_M{num_modes_list[-1]}.png")

    # --- 绘图：精度曲线 ---
    plt.figure(figsize=(6,4))
    plt.plot(num_modes_list, acc_curve, "b-", lw=2)
    plt.xlabel("number of modes"); plt.ylabel("FC reconstruction accuracy (r)")
    plt.title("Rest FC reconstruction (upper-triangle correlation)")
    plt.ylim(0, 1); plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"fc_recon_curve.png", dpi=200); plt.close()

    # --- 上=经验/下=重建 合成图 ---
    Mmix = np.zeros((P, P), dtype=np.float32)
    il = np.tril_indices(P, k=-1)
    Mmix[iu] = FC_emp[iu]; Mmix[il] = FC_last[il]; np.fill_diagonal(Mmix, 1.0)
    plt.figure(figsize=(6,5))
    im = plt.imshow(Mmix, vmin=-1.0, vmax=1.0, cmap="RdBu_r", interpolation="nearest")
    plt.title("Empirical (upper) vs Reconstructed (lower) FC")
    plt.xlabel("Region"); plt.ylabel("Region")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04); cbar.set_label("FC (r)")
    plt.tight_layout(); plt.savefig(f"emp_vs_rec_FC.png", dpi=200); plt.close()

    # --- 上三角散点 ---
    plt.figure(figsize=(5,5))
    plt.scatter(v_emp, FC_last[iu], s=6, alpha=0.5)
    plt.plot([-1,1], [-1,1], "k--", lw=1)
    plt.xlim(-1,1); plt.ylim(-1,1)
    r_last = np.corrcoef(v_emp, FC_last[iu])[0, 1]
    plt.xlabel("Empirical FC (upper triangle)"); plt.ylabel(f"Reconstructed FC (M={num_modes_list[-1]})")
    plt.title(f"Upper-triangle corr r = {r_last:.3f}")
    plt.tight_layout(); plt.savefig(f"emp_vs_rec_scatter.png", dpi=200); plt.close()

    return np.asarray(acc_curve), FC_emp, FC_last

    #——————————main()————————————#


    wmparc_path = os.path.join(struct_dir, "MNINonLinear/wmparc.nii.gz")
    wmparc_img  = nib.load(wmparc_path)
    wmparc_data = wmparc_img.get_fdata()
    mask = wmparc_data > 0  # 所有非0标签的总体 mask
    
    print("开始计算体积本征模态...")
    evals, emodes, tet = compute_volume_eigenmodes_from_mask(mask, wmparc_img, num_modes=200)
    print(f"完成体积本征模态计算：nodes={tet.v.shape[0]}, modes={emodes.shape[1]}")

    # ② 把功能数据 reshape 成 (T,N)
    func_path = os.path.join(resting_dir, "MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz")
    func_img = nib.load(func_path)
    func_data = func_img.get_fdata()
    T = func_data.shape[3]
    flat_func = func_data.reshape(-1, T)  # (V, T)
    flat_mask = mask.reshape(-1)
    timeseries_TxN = flat_func[flat_mask, :].T  # (T,N)

    # ③ 同样取 mask 位置的 parcel 标签
    labels_full = wmparc_data.reshape(-1).astype(np.int32)
    labels_1d   = labels_full[flat_mask]  # (N,)

    # ④ 使用重建函数（体素空间）
    acc_curve, FC_emp, FC_last = reconstruct_FC_with_modes(emodes, timeseries_TxN, labels_1d)

    print("重建完成。精度曲线保存在 reconFC.fc_recon_curve.png")'''