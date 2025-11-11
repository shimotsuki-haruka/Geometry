#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstruct single-subject resting-state fMRI spatiotemporal map and FC matrix
using fsLR-32k eigenmodes (Python version faithfully following MATLAB code).

Author: adapted for clarity from James Pang et al.
"""

import os
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
from tqdm import tqdm

# ============================================================
#  Helper functions
# ============================================================
def load_mat_timeseries(mat_path, varname='timeseries'):
    """
    通用加载 .mat 文件的函数，自动识别 v7.3 (HDF5) 或旧版本。
    返回 np.ndarray。
    """
    try:
        # 尝试普通 loadmat (v7.2 及以下)
        mat = sio.loadmat(mat_path)
        if varname in mat:
            return np.array(mat[varname])
        else:
            # 兼容不同 key（MATLAB 可能保存为 struct）
            for k in mat.keys():
                if not k.startswith("__"):
                    return np.array(mat[k])
    except NotImplementedError:
        # HDF5 (v7.3)
        with h5py.File(mat_path, 'r') as f:
            # MATLAB 保存时变量名可能是 'timeseries' 或类似
            if varname in f:
                data = np.array(f[varname])
            else:
                # 自动选择第一个非内部键
                keys = [k for k in f.keys() if not k.startswith('#')]
                data = np.array(f[keys[0]])
            # MATLAB 保存顺序是反的（C-order vs Fortran-order），需要转置
            data = data.T
        return data
    
def calc_eigendecomposition(X, basis):
    """
    Calculate eigendecomposition coefficients (β) for data X in basis Φ.
    Equivalent to MATLAB calc_eigendecomposition(...,'matrix').

    Parameters
    ----------
    X : ndarray, shape (V, T)
        Data to reconstruct.
    basis : ndarray, shape (V, k)
        Basis eigenmodes.

    Returns
    -------
    beta : ndarray, shape (k, T)
        Reconstruction coefficients β = (ΦᵀΦ)^(-1) Φᵀ X.
        If Φ columns are orthonormal, simplifies to Φᵀ X.
    """
    # pseudo-inverse for numerical stability
    pinv_basis = np.linalg.pinv(basis)
    beta = pinv_basis @ X
    return beta


def calc_triu_ind(mat_shape):
    """Return indices of upper triangle (excluding diagonal)."""
    return np.triu_indices(mat_shape[0], k=1)


def calc_parcellate(parc, data):
    """
    Average vertexwise data into parcels.
    parc: array of parcel labels (len V)
    data: (V, T)
    return: (P, T)
    """
    labels = np.unique(parc[parc > 0])
    P = len(labels)
    data_parc = np.zeros((P, data.shape[1]), dtype=np.float32)
    for i, label in enumerate(labels):
        idx = np.where(parc == label)[0]
        if len(idx) > 0:
            data_parc[i, :] = np.nanmean(data[idx, :], axis=0)
    return data_parc


def calc_normalize_timeseries(X_TxP):
    """
    Z-score normalize each parcel's time series (column-wise if T×P).
    MATLAB: calc_normalize_timeseries(data_parc_emp')
    """
    return zscore(X_TxP, axis=0, nan_policy='omit')


# ============================================================
#  Main reconstruction procedure
# ============================================================
def reconstruct_task_activation_accuracy(
    hemisphere='lh',
    num_modes=200,
    eigenmode_path_template='geodemo/data/examples/fsLR_32k_midthickness-{hemi}_emode_{K}.txt',
    tfmri_mat_template='geodemo/data/examples/subject_tfMRI_zstat-{hemi}.mat',
    parc_template='geodemo/data/parcellations/fsLR_32k_{parcname}-{hemi}.txt',
    parc_name='Glasser360'
):
    """
    Fully faithful Python translation of the MATLAB task-fMRI reconstruction script.
    Functionally identical to:
      subject_tfMRI_zstat reconstruction and correlation (James Pang et al.)
    """

    # ============================================================
    # Load eigenmodes and empirical z-stat data
    # ============================================================
    eigenmode_path = eigenmode_path_template.format(hemi=hemisphere, K=num_modes)
    eigenmodes = np.loadtxt(eigenmode_path)  # (V × num_modes)

    data_mat = load_mat_timeseries(
        tfmri_mat_template.format(hemi=hemisphere),
        varname='zstat'
    )
    if data_mat.ndim > 1:
        data_to_reconstruct = np.squeeze(data_mat)
    else:
        data_to_reconstruct = data_mat
    V = data_to_reconstruct.shape[0]

    print(f"[LOAD] Eigenmodes shape: {eigenmodes.shape}")
    print(f"[LOAD] z-stat shape: {data_to_reconstruct.shape}")

    # ============================================================
    # Load cortex mask
    # ============================================================
    mask_path = f"geodemo/data/template_surfaces_volumes/fsLR_32k_cortex-{hemisphere}_mask.txt"
    cortex_mask = np.loadtxt(mask_path).astype(bool)
    cortex_ind = cortex_mask

    # ============================================================
    # Calculate reconstruction beta coefficients
    # ============================================================
    recon_beta = np.zeros((num_modes, num_modes), dtype=np.float32)
    print("[STEP] Calculating reconstruction coefficients β ...")

    for mode in tqdm(range(1, num_modes + 1)):
        basis = eigenmodes[cortex_ind, :mode]
        beta = calc_eigendecomposition(
            data_to_reconstruct[cortex_ind, np.newaxis],
            basis
        )
        recon_beta[:mode, mode - 1] = beta[:, 0]

    # ============================================================
    # Calculate reconstruction accuracy (vertex-level)
    # ============================================================
    recon_corr_vertex = np.zeros(num_modes, dtype=np.float32)
    print("[STEP] Calculating vertex-level reconstruction correlation ...")

    for mode in tqdm(range(1, num_modes + 1)):
        recon_temp = eigenmodes[cortex_ind, :mode] @ recon_beta[:mode, mode - 1]
        r, _ = pearsonr(data_to_reconstruct[cortex_ind], recon_temp)
        recon_corr_vertex[mode - 1] = r

    # ============================================================
    # Calculate reconstruction accuracy (parcellated-level)
    # ============================================================
    parc_path = parc_template.format(parcname=parc_name, hemi=hemisphere)
    parc = np.loadtxt(parc_path).astype(int)
    parc = parc[cortex_ind]

    recon_corr_parc = np.zeros(num_modes, dtype=np.float32)
    print("[STEP] Calculating parcellation-level reconstruction correlation ...")

    for mode in tqdm(range(1, num_modes + 1)):
        recon_temp = eigenmodes[:, :mode] @ recon_beta[:mode, mode - 1]

        data_parc_emp = calc_parcellate(parc, data_to_reconstruct[:, np.newaxis])
        data_parc_recon = calc_parcellate(parc, recon_temp[:, np.newaxis])

        r, _ = pearsonr(data_parc_emp.flatten(), data_parc_recon.flatten())
        recon_corr_parc[mode - 1] = r

    print("[DONE] Reconstruction finished.")
    print(f"[INFO] recon_corr_vertex shape: {recon_corr_vertex.shape}")
    print(f"[INFO] recon_corr_parc shape: {recon_corr_parc.shape}")

    return {
        'recon_beta': recon_beta,
        'recon_corr_vertex': recon_corr_vertex,
        'recon_corr_parc': recon_corr_parc
    }

def reconstruct_fc_accuracy(hemisphere='lh', num_modes=50,
                            eigenmode_path_template='data/examples/fsLR_32k_midthickness-{hemi}_emode_{K}.txt',
                            fmri_mat_template='data/examples/subject_rfMRI_timeseries-{hemi}.mat',
                            parc_template='data/parcellations/fsLR_32k_{parcname}-{hemi}.txt',
                            parc_name='Glasser360'):
    """
    Fully reproduce MATLAB logic for fMRI reconstruction and FC accuracy curve.
    """

    # --------------------------------------------------------
    # Load eigenmodes
    # --------------------------------------------------------
    eigenmode_path = eigenmode_path_template.format(hemi=hemisphere, K=num_modes)
    eigenmodes = np.loadtxt(eigenmode_path)  # (V_ctx × num_modes)

    # --------------------------------------------------------
    # Load single-subject rfMRI data
    # --------------------------------------------------------
    data_to_reconstruct = load_mat_timeseries(fmri_mat_template.format(hemi=hemisphere),varname='timeseries').astype(np.float32)  # (V × T)
    V, T = data_to_reconstruct.shape
    print(f"[LOAD] Eigenmodes: {eigenmodes.shape}, Timeseries: {data_to_reconstruct.shape}")

    # --------------------------------------------------------
    # Load cortex mask (与 MATLAB 对齐)
    # --------------------------------------------------------
    mask_path = f"geodemo/data/template_surfaces_volumes/fsLR_32k_cortex-{hemisphere}_mask.txt"
    cortex_mask = np.loadtxt(mask_path).astype(bool)
 
    # Apply mask to all spatial variables
    eigen_cortex = eigenmodes[cortex_mask, :]
    data_cortex = data_to_reconstruct[cortex_mask, :]

    # parcellation 也要对应裁剪
    parc = np.loadtxt(parc_template.format(parcname=parc_name, hemi=hemisphere)).astype(int)
    if len(parc) != len(cortex_mask):
        print(f"[WARN] Parcellation length {len(parc)} != mask length {len(cortex_mask)}; applying mask cut.")
        parc = parc[cortex_mask]
    else:
        parc = parc[cortex_mask]

    # --------------------------------------------------------
    # Cortex mask (if provided)
    # --------------------------------------------------------
    '''if cortex_ind is None:
        cortex_ind = np.ones(V, dtype=bool)
    data_cortex = data_to_reconstruct[cortex_ind, :]
    eigen_cortex = eigenmodes[cortex_ind, :]'''

    # --------------------------------------------------------
    # Calculate reconstruction beta coefficients (β)
    # recon_beta[k_modes, T, num_modes]
    # --------------------------------------------------------
    recon_beta = np.zeros((num_modes, T, num_modes), dtype=np.float32)
    print("[STEP] Calculating reconstruction coefficients β ...")
    for mode in tqdm(range(1, num_modes + 1)):
        basis = eigen_cortex[:, :mode]
        beta = calc_eigendecomposition(data_cortex, basis)
        recon_beta[:mode, :, mode - 1] = beta

    # --------------------------------------------------------
    # Parcellation & empirical FC
    # --------------------------------------------------------
    parc_path = parc_template.format(parcname=parc_name, hemi=hemisphere)
    parc = np.loadtxt(parc_path).astype(int)
    labels = np.unique(parc[parc > 0])
    P = len(labels)
    triu_ind = calc_triu_ind((P, P))

    print(f"[DEBUG] eigenmodes shape: {eigen_cortex.shape}")
    print(f"[DEBUG] fMRI data shape: {data_cortex.shape}")
    print(f"[DEBUG] parcellation shape: {parc.shape}")
    print(f"[DEBUG] unique labels (>0): {np.unique(parc[parc>0]).size}")

    # Empirical FC
    data_parc_emp = calc_parcellate(parc, data_to_reconstruct)
    data_parc_emp = calc_normalize_timeseries(data_parc_emp.T)
    data_parc_emp[np.isnan(data_parc_emp)] = 0
    FC_emp = (data_parc_emp.T @ data_parc_emp) / T
    FCvec_emp = FC_emp[triu_ind]
    print(f"[STEP] Empirical FC computed, parcels={P}, vec_len={len(FCvec_emp)}")

    # --------------------------------------------------------
    # Reconstruction + FC accuracy
    # --------------------------------------------------------
    FCvec_recon = np.zeros((len(FCvec_emp), num_modes), dtype=np.float32)
    recon_corr_parc = np.zeros(num_modes, dtype=np.float32)
    print("[STEP] Computing reconstruction accuracy across modes ...")

    for mode in tqdm(range(1, num_modes + 1)):
        recon_temp = eigenmodes[:, :mode] @ recon_beta[:mode, :, mode - 1]  # V×T

        data_parc_recon = calc_parcellate(parc, recon_temp)
        data_parc_recon = calc_normalize_timeseries(data_parc_recon.T)
        data_parc_recon[np.isnan(data_parc_recon)] = 0

        FC_recon_temp = (data_parc_recon.T @ data_parc_recon) / T
        FCvec_recon[:, mode - 1] = FC_recon_temp[triu_ind]

        r, _ = pearsonr(FCvec_emp, FCvec_recon[:, mode - 1])
        recon_corr_parc[mode - 1] = r

    print("[DONE] Reconstruction correlation (per mode):")
    print(recon_corr_parc)

    return {
        'recon_beta': recon_beta,
        'FCvec_emp': FCvec_emp,
        'FCvec_recon': FCvec_recon,
        'recon_corr_parc': recon_corr_parc
    }



def calc_power_spectrum(data):
    """
    Python translation of calc_power_spectrum.m
    data: ndarray [N, P]
        N = number of modes
        P = number of independent samples (here, e.g. time points)
    returns:
        power_spectrum: [N, P]
        power_spectrum_norm: [N, P]
    """
    N, P = data.shape
    power_spectrum = np.abs(data) ** 2
    power_spectrum_norm = power_spectrum / np.nansum(power_spectrum, axis=0, keepdims=True)
    return power_spectrum, power_spectrum_norm

# ============================================================
#  Power spectrum
# ============================================================

def plot_power_spectrum_at_time(recon_beta, time_index, save_path="power_spectrum_t.png"):
    """
    绘制指定时间点的模态功率谱并保存为 PNG。
    参数
    ----
    recon_beta : np.ndarray
        (num_modes, T, num_modes)，由你的重构脚本返回。
        其中 recon_beta[:mode,:,mode-1] 是前 mode 个模态的系数。
    time_index : int
        要绘制的时间点索引（0-based）
    save_path : str
        保存路径（.png）

    """
    num_modes = recon_beta.shape[0]
    mode = num_modes  # 使用所有模态
    # 取完整模态重构下的系数（即 recon_beta[1:num_modes,:,num_modes-1]）
    beta = recon_beta[:mode, :, mode - 1]  # (num_modes, T)
    coeffs_t = beta[:, time_index:time_index+1]  # (num_modes, 1)

    power_spectrum, power_spectrum_norm = calc_power_spectrum(coeffs_t)

    # 绘图
    plt.figure(figsize=(7, 4))
    # 模态编号（1~num_modes）
    modes = np.arange(1, num_modes + 1)

    # 绘制对数功率谱（10进制对数）
    plt.plot(modes, power_spectrum_norm[:, 0],
            marker='o', markersize=2, linewidth=2, color='tab:blue')

    # 设置y轴为对数坐标（以10为底）
    plt.yscale('log', base=10)

    # 设置y轴刻度与范围
    #ymin = np.max([power_spectrum_norm[:, 0].min(), 1e-4])  # 防止0取log
    plt.ylim(power_spectrum_norm[:, 0].min(), power_spectrum_norm[:, 0].max())

    plt.yticks([1e-1, 1e-2, 1e-3, 1e-4],
            [ '10⁻¹', '10⁻²', '10⁻³', '10⁻⁴'])

    plt.xlabel("Mode number", fontsize=12)
    plt.ylabel("Normalized Power", fontsize=12)
    plt.title(f"Power spectrum at time point {time_index}", fontsize=13)
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.tight_layout()


    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVE] Power spectrum plot -> {save_path}")

    return power_spectrum, power_spectrum_norm

def plot_power_spectrum_bar(recon_beta,time_index,):
    """
    绘制指定时间点的模态功率谱（论文风格柱状图，log10坐标）并保存为 PNG。
    自动根据 time_index 命名输出文件。
    """
    save_dir="geodemo/result"
    save_name_prefix="power_spectrum_bar"

    num_modes = recon_beta.shape[0]
    mode = num_modes
    beta = recon_beta[:mode, :, mode - 1]
    coeffs_t = beta[:, time_index:time_index+1]

    power_spectrum, power_spectrum_norm = calc_power_spectrum(coeffs_t)
    ps = power_spectrum_norm[:, 0]

    modes = np.arange(1, num_modes + 1)
    plt.figure(figsize=(7.5, 4))
    plt.bar(modes, ps, color="#2878B5", width=1.2, alpha=0.8, align='center', edgecolor='none')
    plt.yscale('log', base=10)
    plt.ylim(power_spectrum_norm[:, 0].min(), power_spectrum_norm[:, 0].max())
    plt.yticks([1e-1, 1e-2, 1e-3, 1e-4],
               ['10⁻¹', '10⁻²', '10⁻³', '10⁻⁴'])
    plt.xlim(0, num_modes + 1)
    plt.xticks([0, 50, 100, 150, 200],
               ['0', '50', '100', '150', '200'])
    plt.xlabel("Mode", fontsize=12)
    plt.ylabel("Normalized Power (log scale)", fontsize=12)
    plt.title(f"Power spectrum at time point {time_index}", fontsize=13, pad=15)

    plt.axvline(x=50, color='k', linestyle='--', linewidth=0.9, alpha=0.8)
    plt.axvline(x=100, color='k', linestyle='--', linewidth=0.9, alpha=0.8)
    plt.axvline(x=200, color='k', linestyle='--', linewidth=0.9, alpha=0.8)
    y_text = 1.2e-1
    plt.text(25, y_text, "~60 mm", fontsize=10, ha='center')
    plt.text(75, y_text, "~40 mm", fontsize=10, ha='center')
    plt.text(170, y_text, "~30 mm", fontsize=10, ha='center')
    plt.grid(True, which='major', ls='--', alpha=0.3)
    plt.grid(True, which='minor', ls=':', alpha=0.2)
    plt.tight_layout(pad=2)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name_prefix}_t{time_index}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] Power spectrum bar plot -> {save_path}")

    return power_spectrum, power_spectrum_norm

def plot_power_spectrum_task(recon_beta, save_path="geodemo/result/power_spectrum_task.png"):
    """
    绘制任务态重建结果的模态功率谱（严格对应 MATLAB calc_power_spectrum.m）

    输入：
        recon_beta : ndarray, shape (num_modes, num_modes)
            任务态重建函数返回的重建系数矩阵。
            第 k 列为使用前 k 个模态的系数 β。
        save_path : str
            保存路径 (.png)

    输出：
        power_spectrum, power_spectrum_norm : ndarray
            功率谱及其归一化形式，形状 (num_modes, 1)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # ===== 取最后一组（使用所有模态）的 β 系数 =====
    num_modes = recon_beta.shape[0]
    beta_final = recon_beta[:num_modes, num_modes - 1]     # (K,)
    coeffs = beta_final[:, None]                           # (K,1) for consistency

    # ===== 计算功率谱 =====
    power_spectrum, power_spectrum_norm = calc_power_spectrum(coeffs)
    ps = power_spectrum_norm[:, 0]                         # (K,)

    # ===== 绘制 =====
    modes = np.arange(1, num_modes + 1)
    plt.figure(figsize=(7.5, 4))
    plt.bar(modes, ps, color="#2878B5", width=1.2, alpha=0.8, align='center', edgecolor='none')

    plt.yscale('log', base=10)
    plt.ylim(ps.min(), ps.max())
    plt.yticks([1e-1, 1e-2, 1e-3, 1e-4], ['10⁻¹', '10⁻²', '10⁻³', '10⁻⁴'])
    plt.xlim(0, num_modes + 1)
    plt.xticks([0, 50, 100, 150, 200], ['0', '50', '100', '150', '200'])
    plt.xlabel("Mode number", fontsize=12)
    plt.ylabel("Normalized Power (log scale)", fontsize=12)
    plt.title("Task-fMRI power spectrum (all modes)", fontsize=13, pad=15)

    # 分段标记（对应论文图）
    plt.axvline(x=50, color='k', linestyle='--', linewidth=0.9, alpha=0.8)
    plt.axvline(x=100, color='k', linestyle='--', linewidth=0.9, alpha=0.8)
    plt.axvline(x=200, color='k', linestyle='--', linewidth=0.9, alpha=0.8)
    y_text = 1.2e-1
    plt.text(25, y_text, "~60 mm", fontsize=10, ha='center')
    plt.text(75, y_text, "~40 mm", fontsize=10, ha='center')
    plt.text(170, y_text, "~30 mm", fontsize=10, ha='center')

    plt.grid(True, which='major', ls='--', alpha=0.3)
    plt.grid(True, which='minor', ls=':', alpha=0.2)
    plt.tight_layout(pad=2)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SAVE] Task-fMRI power spectrum plot -> {save_path}")
    return power_spectrum, power_spectrum_norm



# ============================================================
#  Example usage
# ============================================================
def rest_state():
    results = reconstruct_fc_accuracy(
        hemisphere='lh',
        num_modes=200,
        eigenmode_path_template='geodemo/data/examples/fsLR_32k_midthickness-{hemi}_emode_{K}.txt',
        fmri_mat_template='geodemo/data/examples/subject_rfMRI_timeseries-{hemi}.mat',
        parc_template='geodemo/data/parcellations/fsLR_32k_{parcname}-{hemi}.txt',
        parc_name='Glasser360'
    )

    recon_beta = results['recon_beta']
    '''power, power_norm = plot_power_spectrum_at_time(
        recon_beta=recon_beta,
        time_index=100,
        save_path="geodemo/result/power_spectrum_t100.png"
    )'''
    power, power_norm = plot_power_spectrum_bar(
        recon_beta=recon_beta,
        time_index=1
    )
    # 可视化 FC 重构准确度曲线
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(results['recon_corr_parc']) + 1),
             results['recon_corr_parc'], 'o-', lw=2)
    plt.xlabel('Number of modes used')
    plt.ylabel('FC reconstruction correlation')
    plt.title('Reconstruction accuracy vs. number of eigenmodes')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"fc_reconstruction_accuracy.png", dpi=300)

def task_state():
    results_task = reconstruct_task_activation_accuracy(
        hemisphere='lh',
        num_modes=200,
        eigenmode_path_template='geodemo/data/examples/fsLR_32k_midthickness-{hemi}_emode_{K}.txt',
        tfmri_mat_template='geodemo/data/examples/subject_tfMRI_zstat-{hemi}.mat',
        parc_template='geodemo/data/parcellations/fsLR_32k_{parcname}-{hemi}.txt',
        parc_name='Glasser360'
    )

    # 绘制结果（如果需要）
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(results_task['recon_corr_vertex']) + 1),
             results_task['recon_corr_vertex'], 'o-', lw=2, label='Vertex-level')
    plt.plot(np.arange(1, len(results_task['recon_corr_parc']) + 1),
             results_task['recon_corr_parc'], 's--', lw=2, label='Parcellated-level')
    plt.xlabel('Number of modes')
    plt.ylabel('Correlation')
    plt.title('Task-fMRI reconstruction accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("geodemo/result/task_reconstruction_accuracy.png", dpi=300)

    power_task, power_task_norm = plot_power_spectrum_task(
        recon_beta=results_task['recon_beta'],
        save_path="geodemo/result/power_spectrum_task.png"
    )
    
if __name__ == "__main__":
    task_state()