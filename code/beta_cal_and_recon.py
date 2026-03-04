#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute eigenmode coefficients (beta) for HCP task-fMRI *native surface* time series.

This script is intentionally aligned with the "standard" Pang et al. MATLAB/Python logic:
  beta = pinv(Phi_cortex) @ X_cortex
where
  Phi_cortex: (V_cortex, K) eigenmodes restricted to cortical vertices
  X_cortex  : (V_cortex, T) task-fMRI time series restricted to cortical vertices

Key design choices (per your request)
-------------------------------------
1) Surface-only (no volume code).
2) Medial wall handling is OPTIONAL:
   - masked=True  : physically crop data/eigenmodes to cortical vertices (atlasroi True)
   - masked=False : keep full arrays in memory, but computations still use cortex_ind
                   (this matches the typical "use cortex mask" logic while not cropping).
3) No manual area-weighting for now.
   - A `mass` interface is reserved for the future (vector (V_cortex,) or None),
     but NOT used by default.
4) Logic follows your provided standard code: `calc_eigendecomposition` uses pseudoinverse.

Outputs
-------
- beta_all: (K, T) using all K modes
- (optional) beta_cumulative: (K, T, K) where beta_cumulative[:m,:,m-1] are coefficients using first m modes
  (faithful to the MATLAB-style reconstruction-curve workflow)

"""

from __future__ import annotations
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
from tqdm import tqdm

# ============================================================
#  Standard helper (faithful)
# ============================================================
def calc_eigendecomposition(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Faithful to your standard code: beta = pinv(basis) @ X

    Parameters
    ----------
    X : ndarray, shape (V, T)
    basis : ndarray, shape (V, K)

    Returns
    -------
    beta : ndarray, shape (K, T)
    """
    pinv_basis = np.linalg.pinv(basis)
    beta = pinv_basis @ X
    return beta


# ============================================================
#  IO helpers (HCP native surface)
# ============================================================
def load_func_gii_as_VxT(func_gii_path: str) -> np.ndarray:
    """Load a .func.gii where each darray is one timepoint. Returns (V, T)."""
    g = nib.load(func_gii_path)
    X = np.column_stack([arr.data for arr in g.darrays]).astype(np.float32)
    return X


def load_atlasroi_mask(struct_dir: str, sub: str, hemi: str) -> np.ndarray:
    """Load HCP native atlasroi mask: True for cortex vertices."""
    hemi = hemi.lower()
    hemi_letter = "L" if hemi in ("lh", "l", "left") else "R"
    mask_path = os.path.join(struct_dir, f"MNINonLinear/Native/{sub}.{hemi_letter}.atlasroi.native.shape.gii")
    mask = nib.load(mask_path).darrays[0].data.astype(bool)
    return mask


def resolve_hcp_task_func_gii(subject_dir: str, task_name: str, phase_enc: str, hemi: str) -> str:
    """Build a typical HCP task-fMRI native func.gii path.

    Example:
      {subject_dir}/MNINonLinear/Results/tfMRI_SOCIAL_LR/tfMRI_SOCIAL_LR.L.native.func.gii
    """
    hemi = hemi.lower()
    hemi_letter = "L" if hemi in ("lh", "l", "left") else "R"
    phase_enc = phase_enc.upper()
    task_name = task_name.upper()
    folder = f"tfMRI_{task_name}_{phase_enc}"
    fname = f"{folder}.{hemi_letter}.native.func.gii"
    return os.path.join(subject_dir, "MNINonLinear/Results", folder, fname)


# ============================================================
#  Core: compute beta for native task time series
# ============================================================
def compute_task_native_beta(
    eigenmodes_VxK: np.ndarray,
    X_VxT: np.ndarray,
    cortex_mask_V: np.ndarray,
    num_modes: int = 200,
    masked: bool = True,
    cumulative: bool = False,
    beta_path: str = None,
    mass: np.ndarray | None = None,
) -> dict:
    """Compute beta coefficients for task native time series using pseudoinverse.

    Parameters
    ----------
    eigenmodes_VxK : (V, K0)
        Eigenmodes on the same vertex ordering as X_VxT.
    X_VxT : (V, T)
        Task native time series (vertices × time).
    cortex_mask_V : (V,)
        True for cortical vertices (atlasroi).
    num_modes : int
        Number of modes K to use.
    masked : bool
        If True, conceptually 'remove medial wall' by restricting computations to cortex vertices.
        If False, keep full arrays but still use cortex vertices for computations.
    cumulative : bool
        If True, compute beta_cumulative cube (K, T, K) like MATLAB workflow.
    mass : optional
        Reserved for future mass-weighting. Not applied currently.

    Returns
    -------
    dict with keys:
      - beta_all : (K, T)
      - beta_cumulative : (K, T, K) or None
      - cortex_ind : (V_cortex,) indices used
    """
    if X_VxT.shape[0] != cortex_mask_V.shape[0]:
        raise ValueError(f"X_VxT V={X_VxT.shape[0]} != mask V={cortex_mask_V.shape[0]}")
    if eigenmodes_VxK.shape[0] != X_VxT.shape[0]:
        raise ValueError(f"eigenmodes V={eigenmodes_VxK.shape[0]} != X_VxT V={X_VxT.shape[0]}")
    if eigenmodes_VxK.shape[1] < num_modes:
        raise ValueError(f"eigenmodes K={eigenmodes_VxK.shape[1]} < num_modes={num_modes}")

    # ============================================================
    #  Cache logic: if not recompute and cache exists -> load & return
    # ============================================================

    cortex_ind = np.where(cortex_mask_V)[0]
    Phi = eigenmodes_VxK[:, :num_modes].astype(np.float64)
    X = X_VxT.astype(np.float64)

    # Interface reserved; not used yet
    if mass is not None:
        raise NotImplementedError(
            "mass-weighted projection is intentionally disabled for now; "
            "you asked to keep the interface only."
        )

    # We always compute on cortex vertices (masked or not), matching standard scripts.
    Phi_c = Phi[cortex_ind, :]
    X_c = X[cortex_ind, :]

    beta_all = calc_eigendecomposition(X_c, Phi_c).astype(np.float32)  # (K, T)

    beta_cum = None
    if cumulative:
        T = X_c.shape[1]
        K = num_modes
        beta_cum = np.zeros((K, T, K), dtype=np.float32)
        for m in range(1, K + 1):
            bm = calc_eigendecomposition(X_c, Phi_c[:, :m]).astype(np.float32)  # (m, T)
            beta_cum[:m, :, m - 1] = bm

    if cumulative:
        beta = beta_cum[:, :, -1]  # (K, T)
    else:
        beta = beta_all

    print(f"[BETA] beta_all shape (K,T) = {beta.shape}")

    if cumulative:
        recon_beta = beta_cum   # (K,T,K)
        np.savez_compressed(beta_path, recon_beta=recon_beta)
    else:
        recon_beta = beta  # (K,T)
        np.save(beta_path, recon_beta)


def run_reconstruct_fc_accuracy_from_saved(
    *,
    out_dir: str,
    num_modes: int,
    hemisphere: str,
    eigenmodes: np.ndarray,      # (V_new, K)
    func_gii_path: str,          # 原始 func.gii (V_orig, T)
    parc_gii_path: str,          # 原始 native parcellation .label.gii (V_orig,)
    keep_idx_path: str,          # keep_idx: (V_new,) 映射到原始 V_orig
    mode: str,                   # "masked" / "unmasked" (用于拼文件名)
):
    """
    按 testanalyse.py:reconstruct_fc_accuracy 的“逻辑/思路”计算 FC 重建准确率：
    - 若存在 out_dir/L.native.{mode}_recon_beta.npz: 用 recon_beta 画准确率曲线
    - 否则只用 out_dir/L.native.{mode}_beta.npy: 计算最终(全模态)准确率(单点)

    注意：本函数不进行任何 beta 的再计算；只读你之前保存的 beta / recon_beta。
    """

    # ---- helpers (从 testanalyse.py 逻辑原样搬过来) ----
    def calc_triu_ind(mat_shape):
        return np.triu_indices(mat_shape[0], k=1)

    def calc_parcellate(parc, data):
        labels = np.unique(parc[parc > 0])
        P = len(labels)
        data_parc = np.zeros((P, data.shape[1]), dtype=np.float32)
        for i, label in enumerate(labels):
            idx = np.where(parc == label)[0]
            if len(idx) > 0:
                data_parc[i, :] = np.nanmean(data[idx, :], axis=0)
        return data_parc

    def calc_normalize_timeseries(X_TxP):
        return zscore(X_TxP, axis=0, nan_policy='omit')

    # --------------------------------------------------------
    # Load fMRI data (.func.gii) —— 与 testanalyse.py 一致
    # --------------------------------------------------------
    print(f"[RECON][LOAD] Loading surface fMRI from {func_gii_path}")
    func_gii = nib.load(func_gii_path)
    data_to_reconstruct = np.column_stack([arr.data for arr in func_gii.darrays]).astype(np.float32)
    print(f"[RECON][LOAD] Loaded .func.gii fMRI data: {data_to_reconstruct.shape} (vertices × time)")
    V, T = data_to_reconstruct.shape

    print(f"[RECON][LOAD] Eigenmodes: {eigenmodes.shape}, Timeseries: {data_to_reconstruct.shape}")

    # --------------------------------------------------------
    # Decide whether to use cortex mask —— 与 testanalyse.py 一致
    # (parc_candidate.endswith(".gii") 分支)
    # --------------------------------------------------------
    parc_candidate = parc_gii_path
    if parc_candidate.endswith(".gii"):
        keep = np.load(keep_idx_path)  # <= 仅把原来 hardcode 路径改成你已经在 main 里使用的 keep_idx_path
        eigen_cortex = eigenmodes
        data_cortex = data_to_reconstruct[keep, :]

        parc_gii = nib.load(parc_gii_path)
        parc = parc_gii.darrays[0].data.astype(int)
        parc = parc[keep]
    else:
        raise ValueError("[RECON] This embedded version expects a .gii parcellation (native label).")

    # --------------------------------------------------------
    # 读取你已经保存的 beta / recon_beta（不允许再计算）
    # --------------------------------------------------------
    recon_beta_path = os.path.join(out_dir, f"L.native.{mode}_recon_beta.npz")
    beta_path = os.path.join(out_dir, f"L.native.{mode}_beta.npy")

    has_recon_beta = os.path.exists(recon_beta_path)
    has_beta = os.path.exists(beta_path)

    if not has_recon_beta and not has_beta:
        raise FileNotFoundError(
            f"[RECON] Neither recon_beta nor beta found in out_dir:\n  {recon_beta_path}\n  {beta_path}"
        )

    recon_beta = None
    beta_all = None

    if has_recon_beta:
        z = np.load(recon_beta_path, allow_pickle=False)
        recon_beta = z["recon_beta"]  # (K,T,K)
        print(f"[RECON][LOAD] recon_beta loaded: {recon_beta.shape} <- {recon_beta_path}")
    else:
        beta_all = np.load(beta_path).astype(np.float32)  # (K,T)
        print(f"[RECON][LOAD] beta_all loaded: {beta_all.shape} <- {beta_path}")

    # --------------------------------------------------------
    # Parcellation & empirical FC 
    # --------------------------------------------------------
    labels = np.unique(parc[parc > 0])
    P = len(labels)
    triu_ind = calc_triu_ind((P, P))

    len_triu = triu_ind[0].size
    print(f"[RECON][CHECK] parcels: P={P}, upper-tri length={len_triu}")

    print(f"[RECON][DEBUG] parcellation shape: {parc.shape}")
    print(f"[RECON][DEBUG] unique labels (>0): {np.unique(parc[parc>0]).size}")

    data_parc_emp = calc_parcellate(parc, data_cortex)
    data_parc_emp = calc_normalize_timeseries(data_parc_emp.T)
    data_parc_emp[np.isnan(data_parc_emp)] = 0
    FC_emp = (data_parc_emp.T @ data_parc_emp) / T
    FCvec_emp = FC_emp[triu_ind]
    print(f"[RECON][STEP] Empirical FC computed, parcels={P}, vec_len={len(FCvec_emp)}")
    print(f"[RECON][CHECK] FCvec_emp length = {FCvec_emp.size}")

    # --------------------------------------------------------
    # Reconstruction + FC accuracy —— 与 testanalyse.py 一致
    #   - 若有 recon_beta：逐 mode 画曲线
    #   - 否则仅全模态算一次（单点）
    # --------------------------------------------------------
    if recon_beta is not None:
        FCvec_recon = np.zeros((len(FCvec_emp), num_modes), dtype=np.float32)
        recon_corr_parc = np.zeros(num_modes, dtype=np.float32)
        print("[RECON][STEP] Computing reconstruction accuracy across modes (curve) ...")

        for mode_i in tqdm(range(1, num_modes + 1)):
            recon_temp = eigen_cortex[:, :mode_i] @ recon_beta[:mode_i, :, mode_i - 1]  # V×T

            data_parc_recon = calc_parcellate(parc, recon_temp)
            data_parc_recon = calc_normalize_timeseries(data_parc_recon.T)
            data_parc_recon[np.isnan(data_parc_recon)] = 0

            FC_recon_temp = (data_parc_recon.T @ data_parc_recon) / T
            FCvec_recon[:, mode_i - 1] = FC_recon_temp[triu_ind]

            r, _ = pearsonr(FCvec_emp, FCvec_recon[:, mode_i - 1])
            recon_corr_parc[mode_i - 1] = r

        print("[RECON][DONE] Reconstruction correlation (per mode):")
        print(recon_corr_parc)

        # ---- 保存曲线数据 + 画图（保存到 out_dir）----
        np.save(os.path.join(out_dir, f"L.native.{mode}_recon_corr_parc.npy"), recon_corr_parc)

        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(recon_corr_parc) + 1), recon_corr_parc, 'o-', lw=2)
        plt.xlabel('Number of modes used')
        plt.ylabel('FC reconstruction correlation')
        plt.title('Reconstruction accuracy vs. number of eigenmodes')
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"L.native.{mode}_fc_reconstruction_accuracy.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[RECON][SAVE] curve fig -> {fig_path}")
    else:
        # ---- 计算最终(全模态)一次 ----
        K = beta_all.shape[0]
        if K != num_modes:
            print(f"[RECON][WARN] beta_all K={K} != num_modes={num_modes}; will use K from beta_all.")
            num_modes_use = K
        else:
            num_modes_use = num_modes

        print("[RECON][STEP] No recon_beta file. Compute final (all-modes) reconstruction accuracy only ...")
        recon_temp = eigen_cortex[:, :num_modes_use] @ beta_all[:num_modes_use, :]  # V×T

        data_parc_recon = calc_parcellate(parc, recon_temp)
        data_parc_recon = calc_normalize_timeseries(data_parc_recon.T)
        data_parc_recon[np.isnan(data_parc_recon)] = 0

        FC_recon_temp = (data_parc_recon.T @ data_parc_recon) / T
        FCvec_recon = FC_recon_temp[triu_ind]

        r, _ = pearsonr(FCvec_emp, FCvec_recon)
        print(f"[RECON][DONE] Final all-modes FC recon corr = {r}")


# ============================================================
#  Example main: your HCP-native task pipeline
# ============================================================
def main():
    # ================= user config (batch) =================
    BASE_PATH = "/home/wmy/Documents"

    # 你想跑的任务 / 编码 / 半球
    TASKS = ["SOCIAL","MOTOR","WM","RELATIONAL","EMOTION","LANGUAGE","GAMBLING"]  # e.g., ["SOCIAL","MOTOR","WM","RELATIONAL","EMOTION","LANGUAGE","GAMBLING"]
    ENCS = ["LR","RL"]      # ["LR","RL"]
    HEMIS = ["lh"]     # ["lh","rh"]

    # 选择被试：
    # - 如果 SUBJECTS = [] ：自动扫描 BASE_PATH/{task}/ 下面的所有子目录作为被试
    # - 如果 SUBJECTS 非空：只跑你指定的这些被试
    SUBJECTS = []  # e.g., ["100610","101107"] ; or [] for auto-scan

    num_modes = 200
    masked = True
    cumulative = False
    listed = False  # 是否从预先列出的被试名单加载被试（而不是自动扫描）。如果 True，代码会从 hardcoded 的路径加载被试名单；如果 False，则自动扫描 BASE_PATH/{task}/ 下的子目录。
    RECOMPUTE = False  # 是否强制重新计算 beta（如果之前已经保存过了）。True 则无视之前的文件，重新计算并覆盖保存；False 则如果发现之前的文件存在，就直接加载返回，不再计算。
    # 结构文件根目录（你原来是 BASE_PATH/Structure/sub）
    STRUCT_ROOT = os.path.join(BASE_PATH, "Structure")
    
    mode = "masked" if masked else "unmasked"
    KEEP_IDX_TMPL     = f"/mnt2/wmy/geometry/Eigenmodes/{mode}" + "/{sub}/L.native_idx.npy"
    EIGENMODES_TMPL   = f"/mnt2/wmy/geometry/Eigenmodes/{mode}" + "/{sub}/eigenmodes_{sub}.txt"
    OUT_ROOT_TMPL     = "/mnt2/wmy/geometry/{task_name}/{sub}"
    eigens_root       = f"/mnt2/wmy/geometry/Eigenmodes/{mode}"

    # =======================================================

    n_ok, n_skip = 0, 0

    for task_name in TASKS:
        task_dir = os.path.join(BASE_PATH, task_name)

        if not os.path.isdir(task_dir):
            print(f"[SKIP] task folder not found: {task_dir}")
            n_skip += 1
            continue
        
        if listed:
            sub_list_path = "/home/wmy/work/geometry/data/subject_list_HCP.txt"  

            with open(sub_list_path, "r", encoding="utf-8") as f:
                SUBJECTS = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
                
        # 自动扫描被试
        if SUBJECTS:
            subs = SUBJECTS
        else:
            # subs = sorted([d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]) #数据目录
            subs = sorted([d for d in os.listdir(eigens_root) if os.path.isdir(os.path.join(eigens_root, d))])

        for sub in subs:
            subject_task_dir = os.path.join(BASE_PATH, task_name, sub)
            struct_dir = os.path.join(STRUCT_ROOT, sub)

            for phase_enc in ENCS:
                for hemi in HEMIS:
                    try:
                        # ---------------- 原来 main 里的路径逻辑（保持不变） ----------------
                        keep_idx_path = KEEP_IDX_TMPL.format(sub=sub)
                        func_gii_path = resolve_hcp_task_func_gii(subject_task_dir, task_name, phase_enc, hemi)
                        OUT_ROOT = OUT_ROOT_TMPL.format(task_name=task_name, sub=sub)
                        eigenmodes_path = EIGENMODES_TMPL.format(sub=sub)
                        # -------------------------------------------------------------------

                        if not os.path.exists(func_gii_path):
                            raise FileNotFoundError(f"Task func.gii not found:\n  {func_gii_path}")
                        if not os.path.exists(eigenmodes_path):
                            raise FileNotFoundError(f"Eigenmodes not found: {eigenmodes_path}")
                        if not os.path.exists(struct_dir):
                            raise FileNotFoundError(f"Struct dir not found: {struct_dir}")
                        
                        print("\n" + "=" * 78)
                        print(f"[RUN] task={task_name} sub={sub} enc={phase_enc} hemi={hemi}")
                        
                        out_dir = os.path.join(
                            OUT_ROOT,
                            f"tfMRI_{task_name}_{phase_enc}/"
                        )
                        os.makedirs(out_dir, exist_ok=True)

                        if cumulative:
                            beta_path = os.path.join(out_dir, f"L.native.{mode}_recon_beta.npz")
                        else:
                            beta_path = os.path.join(out_dir, f"L.native.{mode}_beta.npy")

                        if (not RECOMPUTE) and os.path.exists(beta_path):
                            print(f"[SKIP] exists -> {beta_path}")
                            continue

                        print(f"[LOAD] func.gii: {func_gii_path}")
                        X_VxT = load_func_gii_as_VxT(func_gii_path)
                        print(f"[LOAD] X shape (V,T) = {X_VxT.shape}")

                        Phi_VxK = np.loadtxt(eigenmodes_path).astype(np.float32)
                        print(f"[LOAD] eigenmodes shape (V,K) = {Phi_VxK.shape}")

                        cortex_mask = load_atlasroi_mask(struct_dir, sub, hemi)
                        print(f"[LOAD] cortex vertices = {int(cortex_mask.sum())} / {cortex_mask.size}")

                        # --- Align func vertices to eigenmode vertex order if needed ---
                        if Phi_VxK.shape[0] != X_VxT.shape[0]:
                            if not os.path.exists(keep_idx_path):
                                raise FileNotFoundError(
                                    f"keep_idx needed but not found:\n  {keep_idx_path}\n"
                                    "Because eigenmodes V != func V, code requires keep_idx to align."
                                )
                            keep = np.load(keep_idx_path).astype(np.int64)
                            # 关键检查：keep 必须和 eigenmodes 的 V 完全一致
                            if keep.size != Phi_VxK.shape[0]:
                                raise ValueError(f"keep_idx len={keep.size} != eigenmodes V={Phi_VxK.shape[0]}")
                            if keep.max() >= X_VxT.shape[0]:
                                raise ValueError(f"keep_idx max={keep.max()} >= func V={X_VxT.shape[0]}")

                            X_VxT = X_VxT[keep, :]
                            cortex_mask = cortex_mask[keep]
                            print(f"[ALIGN] after keep_idx: X={X_VxT.shape}, cortex={int(cortex_mask.sum())}/{cortex_mask.size}")

                        compute_task_native_beta(
                            eigenmodes_VxK=Phi_VxK,
                            X_VxT=X_VxT,
                            cortex_mask_V=cortex_mask,
                            num_modes=num_modes,
                            masked=masked,
                            cumulative=cumulative,
                            beta_path=beta_path,
                            mass=None,
                        )
                        
                        # ============================================================
                        # FC 重建准确率/曲线
                        # ============================================================
                        '''hemi_letter = "L" if hemi.lower() in ("lh", "l", "left") else "R"
                        parc_gii_path = os.path.join(struct_dir, f"MNINonLinear/Native/{sub}.{hemi_letter}.aparc.a2009s.native.label.gii")

                        if not os.path.exists(parc_gii_path):
                            raise FileNotFoundError(f"parcellation not found: {parc_gii_path}")

                        run_reconstruct_fc_accuracy_from_saved(
                            out_dir=out_dir,
                            num_modes=num_modes,
                            hemisphere=hemi,
                            eigenmodes=Phi_VxK[:, :num_modes],
                            func_gii_path=func_gii_path,
                            parc_gii_path=parc_gii_path,
                            keep_idx_path=keep_idx_path,
                            mode=mode,
                        )'''

                        print(f"[SAVE] -> {out_dir}")
                        print("[DONE]")
                        n_ok += 1

                    except Exception as e:
                        print(f"[FAIL] task={task_name} sub={sub} enc={phase_enc} hemi={hemi} : {e}")
                        n_skip += 1

    print(f"\nAll finished. OK={n_ok}, SKIP/FAIL={n_skip}")


if __name__ == "__main__":
    main()
