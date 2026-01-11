#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

# ==== 配置：按需改这两行 ====
NPZ_PATH = "/home/wmy/geodemo/data/beta/recon_beta_lh_200_Glasser360.npz"
OUT_PNG  = "geodemo/result/beta_glasser360.png"

# 代表性模态选择：前5 + 中间5 + 后5
HEAD, CENTER, TAIL = 5, 5, 5    # 如需更少/更多曲线，改这三个数即可
CUSTOM_MODES_1BASED = [1]

def _beta_to_2d(beta):
    """把 beta 统一成 (K,T)。若是 (K,T,K)，取最后一层（用全部模态重建的那层）。"""
    if beta.ndim == 2:
        return beta
    if beta.ndim == 3 and beta.shape[0] == beta.shape[2]:
        return beta[:, :, -1]
    raise ValueError(f"Unexpected beta shape {beta.shape}. Need (K,T) or (K,T,K).")

def pick_modes(K, head=5, center=5, tail=5, custom_1based=None):
    """
    返回 0-based 索引列表，按照：
    1) 自定义模态（1-based）优先；
    2) 前 head 个；
    3) 中间 center 个；
    4) 最后 tail 个；
    并去重保序。
    """
    sel = []

    # 1) 自定义（1-based -> 0-based）
    if custom_1based:
        for m in custom_1based:
            i = int(m) - 1
            if 0 <= i < K:
                sel.append(i)

    # 2) 低频
    h = min(max(head, 0), K)
    sel += list(range(h))

    # 3) 中频
    cnum = min(max(center, 0), K)
    if cnum > 0:
        half = max(1, cnum // 2)
        c = K // 2
        s = max(0, c - half)
        e = min(K, s + cnum)
        s = max(0, e - cnum)
        sel += list(range(s, e))

    # 4) 高频
    t = min(max(tail, 0), K)
    if t > 0:
        sel += list(range(max(0, K - t), K))

    # 去重保序
    seen = set()
    sel = [i for i in sel if (0 <= i < K and (i not in seen and not seen.add(i)))]
    return sel

def main():
    os.makedirs(os.path.dirname(OUT_PNG) or ".", exist_ok=True)

    z = np.load(NPZ_PATH, allow_pickle=False)
    if "recon_beta" not in z.files:
        raise KeyError(f"'recon_beta' not in {NPZ_PATH}. Found keys: {z.files}")
    beta = z["recon_beta"]                  # 形状 (K,T) 或 (K,T,K)
    beta_2d = _beta_to_2d(beta)             # -> (K,T)

    K, T = beta_2d.shape
    modes = pick_modes(K, HEAD, CENTER, TAIL, CUSTOM_MODES_1BASED)
    B = beta_2d[modes].copy()               # (M,T)

    # 每条曲线做 z-score（更便于同图对比）
    # B -= B.mean(axis=1, keepdims=True)
    # B /= (B.std(axis=1, keepdims=True))

    plt.figure(figsize=(12, 4.5))
    for i, idx in enumerate(modes):
        plt.plot(B[i], lw=1.2, label=f"mode {idx+1}")  # 显示 1-based

    plt.xlabel("time (frames)")
    plt.ylabel("β")
    plt.title("β(t) for representative modes")
    plt.legend(ncol=4, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()
    print(f"[SAVE] {OUT_PNG}")

if __name__ == "__main__":
    main()
