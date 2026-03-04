#此版本存在计算问题。请使用testanalyse
#___________________________________

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# =========================
# Utils
# =========================
def zscore_time(X_PxT: np.ndarray) -> np.ndarray:
    """对每个 parcel/模态沿时间维做 z-score： (P,T) -> (P,T)"""
    mu = X_PxT.mean(axis=1, keepdims=True)
    sd = X_PxT.std(axis=1, keepdims=True) 
    return (X_PxT - mu) / sd


def mask_mesh(verts: np.ndarray, faces: np.ndarray, mask: np.ndarray):
    """
    用 mask (V,) 裁剪 mesh，并重映射 faces 索引。
    返回: verts2, faces2, keep_idx
    """
    keep_idx = np.where(mask)[0]
    remap = -np.ones(mask.shape[0], dtype=np.int64)
    remap[keep_idx] = np.arange(keep_idx.size)

    # 保留三个顶点都在 cortex 的三角形
    fmask = mask[faces[:, 0]] & mask[faces[:, 1]] & mask[faces[:, 2]]
    faces2 = remap[faces[fmask]]
    verts2 = verts[keep_idx]
    return verts2, faces2.astype(np.int32), keep_idx.astype(np.int32)


# =========================
# IO (LH native)
# =========================
def load_LH_native(resting_dir: str, struct_dir: str, sub: str, masked: bool):
    """
    读取 LH native:
      - rfMRI func.gii -> Y_TxV (T,V_all)
      - white.native.surf.gii -> verts(V_all,3), faces(F,3)
      - atlasroi.native.shape.gii -> mask_bool(V_all,) True=cortex
      - aparc.a2009s.native.label.gii -> labels(V_all,) int

    若 masked=True，则裁掉 medial wall，返回裁剪后的 Y/verts/faces/labels，并额外返回 keep_idx。
    """
    func_gii = os.path.join(
        resting_dir,
        "MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.L.native.func.gii"
    )
    g = nib.load(func_gii)
    Y_TxV = np.vstack([arr.data for arr in g.darrays]).astype(np.float64)  # (T,V_all)

    surf_gii = os.path.join(struct_dir, f"MNINonLinear/Native/{sub}.L.white.native.surf.gii")
    s = nib.load(surf_gii)
    verts = s.darrays[0].data.astype(np.float64)   # (V_all,3)
    faces = s.darrays[1].data.astype(np.int32)     # (F,3)

    mask_gii = os.path.join(struct_dir, f"MNINonLinear/Native/{sub}.L.atlasroi.native.shape.gii")
    mask_bool = nib.load(mask_gii).darrays[0].data.astype(bool)  # True=cortex

    parc_gii = os.path.join(struct_dir, f"MNINonLinear/Native/{sub}.L.aparc.a2009s.native.label.gii")
    labels = nib.load(parc_gii).darrays[0].data.astype(int)

    # 基本一致性检查
    V_all = verts.shape[0]
    if Y_TxV.shape[1] != V_all or labels.size != V_all or mask_bool.size != V_all:
        raise ValueError("func/surf/labels/mask 的顶点数不一致，请检查输入文件是否同一空间与同一分辨率。")

    print(f"[LOAD] V_all={V_all}, F={faces.shape[0]}, T={Y_TxV.shape[0]}, cortex={int(mask_bool.sum())}")

    keep_idx = None
    if masked:
        # 裁掉 medial wall：Y/verts/faces/labels 同步裁剪
        verts2, faces2, keep_idx = mask_mesh(verts, faces, mask_bool)
        Y_TxV = Y_TxV[:, keep_idx]
        labels = labels[keep_idx]
        mask_bool = mask_bool[keep_idx]  # 全 True（仅作占位）

        verts, faces = verts2, faces2
        print(f"[MASK] after masking: V={verts.shape[0]}, F={faces.shape[0]}")

    return Y_TxV, verts, faces, labels, mask_bool, keep_idx


# =========================
# Parcellation & FC
# =========================
def parcellate_vertex_timeseries(Y_TxV: np.ndarray, labels_V: np.ndarray):
    """
    (T,V) + (V,) -> pt(P,T), parcel_ids(P,)
    仅 labels>0 参与平均；label=0 (medial wall 等) 自动忽略。
    """
    parcel_ids = np.unique(labels_V[labels_V > 0])
    T = Y_TxV.shape[0]
    P = parcel_ids.size

    X_VxT = Y_TxV.T  # (V,T)
    pt = np.zeros((P, T), dtype=np.float32)
    for i, pid in enumerate(parcel_ids):
        pt[i] = X_VxT[labels_V == pid].mean(axis=0)
    return pt, parcel_ids


def fc_from_ptseries(pt_PxT: np.ndarray) -> np.ndarray:
    """pt(P,T) -> FC(P,P)"""
    Z = zscore_time(pt_PxT)
    return np.corrcoef(Z)


# =========================
# Modal coefficients (Euclidean by default; keep mass interface)
# =========================
def compute_modal_coefficients(Y_TxV: np.ndarray,
                               eigenmodes_VxK: np.ndarray,
                               ridge_scale: float = 1e-8,
                               mass=None):
    """
    计算 Beta：欧式内积 (mass=None)；保留 mass 接口以便未来使用质量矩阵。

    输入:
      Y_TxV: (T,V)
      Phi : (V,K)
      mass: None / (V,) / (V,V)稀疏(未来可扩展)
    返回:
      Beta: (K,T)
    """
    Phi = eigenmodes_VxK.astype(np.float64)
    Y = Y_TxV.astype(np.float64)
    V = Y.shape[1]
    if Phi.shape[0] != V:
        raise ValueError(f"V mismatch: Y has {V}, Phi has {Phi.shape[0]}")

    # --- Euclidean inner product (current default) ---
    if mass is None:
        C = Phi.T @ Y.T          # (K,T)
        G = Phi.T @ Phi          # (K,K)
    else:
        # 仅保留接口：你未来把质量矩阵/向量存下来后可直接传进来
        # mass 为向量 (V,) 时，相当于 A=diag(mass)
        if isinstance(mass, np.ndarray) and mass.ndim == 1:
            w = mass.astype(np.float64)
            C = (Phi.T * w) @ Y.T           # Phi^T A Y^T
            G = (Phi.T * w) @ Phi           # Phi^T A Phi
        else:
            raise NotImplementedError("目前仅保留 mass 向量接口 (V,)。未来可扩展到稀疏矩阵。")

    K = G.shape[0]
    ridge = ridge_scale * (np.trace(G) / max(1, K))
    G = G.copy()
    G.flat[::K+1] += ridge

    # 用 solve 比 inv 更稳
    Beta = np.linalg.solve(G, C)   # (K,T)
    return Beta


# =========================
# FC reconstruction with modes
# =========================
def reconstruct_fc_with_modes(Y_TxV: np.ndarray,
                              labels_V: np.ndarray,
                              FC_emp: np.ndarray,
                              eigenmodes_VxK: np.ndarray,
                              Mmax: int = 200,
                              ridge_scale: float = 1e-8):
    """
    逐模态数 M 重建 -> parcel -> FC -> 与经验 FC 上三角相关
    返回: acc_curve(Mmax,), FC_last
    """
    T, V = Y_TxV.shape
    Phi = eigenmodes_VxK.astype(np.float64)
    if Phi.shape[0] != V:
        raise ValueError(f"eigenmodes rows != V: Phi={Phi.shape[0]}, V={V}")

    P = FC_emp.shape[0]
    iu = np.triu_indices(P, k=1)
    v_emp = FC_emp[iu]

    # 预计算（你原逻辑：C & G）
    C = Phi.T @ Y_TxV.T                 # (K,T)
    G = Phi.T @ Phi                     # (K,K)
    K = G.shape[0]
    ridge = ridge_scale * (np.trace(G) / max(1, K))
    G = G.copy()
    G.flat[::K+1] += ridge

    # 注意：不要用 inv(G) 再切片（(G^-1)[:M,:M] != (G[:M,:M])^-1）
    # 我这里保持“同一数学目标”不变，只是把求解方式换成更正确/稳定的 solve。
    Mmax = min(Mmax, K)
    acc_curve = []
    FC_last = None

    for M in range(1, Mmax + 1):
        GM = G[:M, :M]
        CM = C[:M, :]                     # (M,T)
        BetaM = np.linalg.solve(GM, CM)   # (M,T)

        Yhat = (Phi[:, :M] @ BetaM).T     # (T,V)

        pt_rec, _ = parcellate_vertex_timeseries(Yhat, labels_V)
        FC_rec = fc_from_ptseries(pt_rec)

        r = np.corrcoef(v_emp, FC_rec[iu])[0, 1]
        acc_curve.append(float(r))

        if M == Mmax:
            FC_last = FC_rec

    return np.asarray(acc_curve), FC_last


# =========================
# Plots
# =========================
def save_fc_png(FC: np.ndarray, save_path: str, title: str):
    plt.figure(figsize=(5.2, 4.4))
    im = plt.imshow(FC, vmin=-1.0, vmax=1.0, cmap="RdBu_r", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Parcel")
    plt.ylabel("Parcel")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("r")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_representative_beta(Beta_KxT: np.ndarray, save_path: str, max_show: int = 8):
    K, T = Beta_KxT.shape
    chosen = set([0, 1, 2])  # mode 1,2,3

    var = Beta_KxT.var(axis=1)
    top_var_idx = list(np.argsort(var)[::-1])
    for idx in top_var_idx:
        chosen.add(int(idx))
        if len(chosen) >= max_show - 2:
            break

    for idx in [K // 2 - 1, K - 1]:
        if 0 <= idx < K:
            chosen.add(int(idx))

    chosen = sorted(chosen)[:max_show]

    plt.figure(figsize=(10, 4))
    for i in chosen:
        plt.plot(Beta_KxT[i], lw=1.2, label=f"mode {i+1}")
    plt.xlabel("time (frames)")
    plt.ylabel("β_n(t)")
    plt.title("Representative modal coefficients over time")
    plt.legend(ncol=4, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[saved] {save_path}")


def plot_modal_power_spectrum(Beta_KxT: np.ndarray, save_path: str, out_prefix: str, logy: bool = True):
    power = np.sum(Beta_KxT ** 2, axis=1)               # (K,)
    pnorm = power / (power.sum() + 1e-12)
    pcum = np.cumsum(pnorm)

    np.save(f"{out_prefix}.modal_power.npy", power)
    np.save(f"{out_prefix}.modal_power_norm.npy", pnorm)
    np.save(f"{out_prefix}.modal_power_cum.npy", pcum)

    x = np.arange(1, Beta_KxT.shape[0] + 1)
    fig, ax1 = plt.subplots(figsize=(8, 3.6))
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

    plt.title("Modal power spectrum (sum_t β^2)")
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[saved] {save_path}")


# =========================
# Main (surface-only)
# =========================
def main():
    BASE_PATH = "/home/wmy/Documents/"
    sub = "100307"
    masked = True   # True: 去 medial wall；

    resting_dir = os.path.join(BASE_PATH, "REST1", sub)
    struct_dir  = os.path.join(BASE_PATH, "Structure", sub)

    out_prefix = os.path.join("work/geometry/results/mode_analyse", f"{sub}.LH")
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # 读取数据（按 masked 决定是否裁掉 medial wall）
    Y_TxV, verts, faces, labels_V, mask_bool, keep_idx = load_LH_native(
        resting_dir, struct_dir, sub, masked=masked
    )

    # 载入 eigenmodes：要求与当前 Y_TxV 的顶点顺序一致
    eigenmodes_path = "/home/wmy/work/geometry/eigenmodes_1.txt"
    Phi = np.loadtxt(eigenmodes_path).astype(np.float64)  # (V,K)

    if Phi.shape[0] != Y_TxV.shape[1]:
        raise ValueError(
            f"eigenmodes 行数与当前 V 不一致：Phi={Phi.shape[0]}, V={Y_TxV.shape[1]}\n"
            f"提示：若你 eigenmodes 是 masked 版本，请把 masked=True；若是 full 版本，请 masked=False。"
        )
    print(f"[OK] eigenmodes shape = {Phi.shape}")

    # A) 经验 FC（parcel级）
    pt_emp, parcel_ids = parcellate_vertex_timeseries(Y_TxV, labels_V)
    FC_emp = fc_from_ptseries(pt_emp)

    np.save(f"{out_prefix}.empirical.ptseries.npy", pt_emp)
    np.save(f"{out_prefix}.empirical.FC.npy", FC_emp)
    np.save(f"{out_prefix}.parcel_ids.npy", parcel_ids)

    save_fc_png(FC_emp, f"{out_prefix}.empirical.FC.png", title=f"LH empirical FC (a2009s, P={len(parcel_ids)})")
    print(f"[saved] {out_prefix}.empirical.FC.png")

    # B) 模态重建评估（FC accuracy curve + 最终重建 FC）
    acc_curve, FC_last = reconstruct_fc_with_modes(
        Y_TxV=Y_TxV,
        labels_V=labels_V,
        FC_emp=FC_emp,
        eigenmodes_VxK=Phi,
        Mmax=200,
        ridge_scale=1e-8
    )

    # 精度曲线
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(acc_curve) + 1), acc_curve, "b-", lw=2)
    plt.xlabel("number of modes")
    plt.ylabel("FC reconstruction accuracy (r)")
    plt.title("LH (a2009s) – FC reconstruction")
    plt.ylim(0, 1)
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.fc_recon_curve.png", dpi=200)
    plt.close()
    print(f"[saved] {out_prefix}.fc_recon_curve.png")

    # 上=经验/下=重建 合成图
    P = FC_emp.shape[0]
    iu = np.triu_indices(P, k=1)
    il = np.tril_indices(P, k=-1)
    mix = np.zeros_like(FC_emp)
    mix[iu] = FC_emp[iu]
    mix[il] = FC_last[il]
    np.fill_diagonal(mix, 1.0)

    save_fc_png(mix, f"{out_prefix}.emp_upper_rec_lower.png",
                title="Empirical (upper) vs Reconstructed (lower)")
    print(f"[saved] {out_prefix}.emp_upper_rec_lower.png")

    # 上三角散点
    v_emp = FC_emp[iu]
    r_last = np.corrcoef(v_emp, FC_last[iu])[0, 1]
    plt.figure(figsize=(5, 5))
    plt.scatter(v_emp, FC_last[iu], s=6, alpha=0.5)
    plt.plot([-1, 1], [-1, 1], "k--", lw=1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Empirical FC (upper triangle)")
    plt.ylabel(f"Reconstructed FC (M={len(acc_curve)})")
    plt.title(f"Upper-triangle corr r = {r_last:.3f}")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.emp_vs_recon_scatter.png", dpi=200)
    plt.close()
    print(f"[saved] {out_prefix}.emp_vs_recon_scatter.png")

    # 最终重建 FC 单独保存
    save_fc_png(FC_last, f"{out_prefix}.FC_reconstructed_M{len(acc_curve)}.png",
                title=f"Reconstructed FC (M={len(acc_curve)})")
    print(f"[saved] {out_prefix}.FC_reconstructed_M{len(acc_curve)}.png")

    # C) 模态系数 beta（当前：欧式内积；保留 mass 接口）
    Beta = compute_modal_coefficients(
        Y_TxV=Y_TxV,
        eigenmodes_VxK=Phi,
        ridge_scale=1e-8,
        mass=None   # 未来你可以传入质量向量 (V,)
    )

    plot_representative_beta(Beta, save_path=f"{out_prefix}.beta_representatives.png")

    plot_modal_power_spectrum(
        Beta,
        save_path=f"{out_prefix}.modal_power_spectrum.png",
        out_prefix=out_prefix,
        logy=True
    )

    print("[done] surface-only pipeline finished.")


if __name__ == "__main__":
    main()
