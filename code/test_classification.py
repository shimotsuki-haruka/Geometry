#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import re
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset, DataLoader

# --------------------------
# 配置
# --------------------------
MAT_PATH = "/home/wmy/geodemo/data/beta/beta_S255_lh_struct.mat"   # v7 mat
USE_PREPROC = "slog1p"   # "slog1p" 或 "asinh"
USE_MLP = True           # 需要 MLP 做个对照就设 True


# ==========================
# 工具 & 核心函数
# ==========================
def map_to_superclass(name: str) -> str:
    n = name.lower()
    if   n.startswith("social"):     return "social"
    elif n.startswith("motor"):      return "motor"
    elif n.startswith("gambling"):   return "gambling"
    elif n.startswith("wm"):         return "wm"
    elif n.startswith("language"):   return "language"
    elif n.startswith("emotion"):    return "emotion"
    elif n.startswith("relational"): return "relational"
    else:
        m = re.match(r"([a-z]+)", n)  # 兜底：取首段字母
        return m.group(1) if m else "unknown"


def load_struct_mat(mat_path):
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    B = m["beta_struct"]               # numpy object array, len 47
    task_names = [str(B[i].name) for i in range(B.size)]
    K, Nsub = B[0].beta.shape
    Ntask = B.size
    print(f"[INFO] K={K}, Nsub={Nsub}, Ntask={Ntask}")
    return B, task_names, K, Nsub, Ntask


def build_X_y_groups(B, task_names, Ntask, Nsub, K):
    # 样本顺序：task 外层 × subject 内层
    X_list, ytask, subj = [], [], []
    for t in range(Ntask):
        X_list.append(B[t].beta.T)            # [Nsub, K]
        ytask.extend([t]*Nsub)
        subj.extend(list(range(Nsub)))
    X = np.vstack(X_list)                      # [Nsub*Ntask, K]
    ytask = np.array(ytask, dtype=int)
    groups = np.array(subj, dtype=int)         # 被试分组
    return X, ytask, groups


def make_classes(task_names):
    super_names = [map_to_superclass(n) for n in task_names]
    classes = sorted(list({n for n in super_names if n!="unknown"}))
    cls_to_id = {c:i for i,c in enumerate(classes)}
    return super_names, classes, cls_to_id


def data_check(X, B, task_names, Ntask, Nsub):
    print("\n[CHECK] === Global matrix X ===")
    print(f"[CHECK] X shape = {X.shape}")
    total_nan = int(np.isnan(X).sum())
    total_inf = int(np.isinf(X).sum())
    print(f"[CHECK] X NaN={total_nan}, Inf={total_inf}")

    if total_nan or total_inf:
        bad_rows = np.where(np.any(~np.isfinite(X), axis=1))[0]
        print(f"[CHECK] bad sample rows count = {bad_rows.size}")
        if bad_rows.size:
            print("[CHECK] first bad rows (task, subj, global_row, task_name):")
            for r in bad_rows[:20]:
                t = r // Nsub
                s = r %  Nsub
                print(f"  ({t:2d}, {s:3d}, {r:5d})  {task_names[t]}")
    else:
        print("[CHECK] no NaN/Inf rows in X")

    bad_cols = np.where(np.any(~np.isfinite(X), axis=0))[0]
    if bad_cols.size:
        print(f"[CHECK] modes having NaN/Inf (count={bad_cols.size}): "
              f"{bad_cols[:20]}{' ...' if bad_cols.size>20 else ''}")
    var_per_mode = np.nanvar(X, axis=0)
    zero_var_cols = np.where((~np.isfinite(var_per_mode)) | (var_per_mode == 0.0))[0]
    if zero_var_cols.size:
        print(f"[CHECK] zero-variance modes (count={zero_var_cols.size}): "
              f"{zero_var_cols[:20]}{' ...' if zero_var_cols.size>20 else ''}")

    print("\n[CHECK] === Per-task origin (B[t].beta) ===")
    tasks_with_issues = []
    for t in range(Ntask):
        Bt = B[t].beta  # [K, Nsub]
        nan_cnt = int(np.isnan(Bt).sum())
        inf_cnt = int(np.isinf(Bt).sum())
        if nan_cnt or inf_cnt:
            nan_per_sub = np.isnan(Bt).sum(axis=0) + np.isinf(Bt).sum(axis=0)
            bad_sub_cols = np.where(nan_per_sub > 0)[0]
            print(f"[TASK] {task_names[t]} : NaN={nan_cnt}, Inf={inf_cnt}, "
                  f"bad_subject_cols={bad_sub_cols[:10]}{' ...' if bad_sub_cols.size>10 else ''}")
            tasks_with_issues.append((t, task_names[t], nan_cnt, inf_cnt))

    print("\n[CHECK] === Summary ===")
    if total_nan or total_inf:
        print(f"[SUMMARY] X contains NaN/Inf -> NaN={total_nan}, Inf={total_inf}")
    else:
        print("[SUMMARY] X has no NaN/Inf")
    if bad_cols.size:
        print(f"[SUMMARY] {bad_cols.size} mode columns contain NaN/Inf.")
    else:
        print("[SUMMARY] no mode column has NaN/Inf")
    if zero_var_cols.size:
        print(f"[SUMMARY] {zero_var_cols.size} zero-variance modes (uninformative for classification).")
    if tasks_with_issues:
        print(f"[SUMMARY] {len(tasks_with_issues)} / {Ntask} tasks have NaN/Inf in original beta.")
    else:
        print("[SUMMARY] All tasks' original beta are finite.")
    print("[CHECK] ===============================\n")


def find_invalid_subjects(B, task_names, Ntask, Nsub):
    invalid_subs = set()
    affected = {}
    for t in range(Ntask):
        Bt = B[t].beta  # [K, Nsub]
        whole_nan = np.all(np.isnan(Bt), axis=0)
        bad_idx = np.where(whole_nan)[0]
        for s in bad_idx:
            invalid_subs.add(int(s))
            affected.setdefault(int(s), []).append(task_names[t])

    invalid_subs = sorted(list(invalid_subs))
    if invalid_subs:
        print(f"[FILTER] invalid subjects (0-based): {invalid_subs}")
        for s in invalid_subs:
            print(f"         subj {s} -> affected tasks: {affected.get(s, [])}")
    else:
        print("[FILTER] invalid subjects (0-based): None")
    return invalid_subs


def filter_rows_by_subjects(X, ytask, groups, invalid_subs, Nsub):
    if not invalid_subs:
        return X, ytask, groups
    rows = np.arange(X.shape[0])
    subj_of_row = rows % Nsub
    keep_mask = ~np.isin(subj_of_row, invalid_subs)
    X = X[keep_mask]
    ytask = ytask[keep_mask]
    groups = groups[keep_mask]
    print(f"[FILTER] kept rows: {keep_mask.sum()} / {len(keep_mask)}")
    return X, ytask, groups


def make_labels_after_filter(ytask, super_names, classes):
    cls_to_id = {c:i for i,c in enumerate(classes)}
    y = np.array([cls_to_id[super_names[t]] for t in ytask], dtype=int)
    print("[INFO] classes:", classes)
    return y


def preprocess(X, method="slog1p"):
    eps = 1e-6
    if method == "slog1p":
        Xp = np.sign(X) * np.log1p(np.abs(X))
    elif method == "asinh":
        med = np.median(np.abs(X), axis=0) + eps
        Xp = np.arcsinh(X / med)
    else:
        Xp = X.copy()
    Xp[~np.isfinite(Xp)] = 0.0
    scaler = StandardScaler()
    Xn = scaler.fit_transform(Xp)
    return Xn, scaler


def train_linear(Xtr, ytr):
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=2000, n_jobs=-1  # 留空 multi_class，避免未来版本告警
    )
    clf.fit(Xtr, ytr)
    return clf


class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)


def train_mlp(Xtr, ytr, Xval, yval, num_classes, epochs=60, lr=1e-3, wd=1e-4, bs=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(Xtr.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr).float(),
                                         torch.from_numpy(ytr).long()),
                           batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(Xval).float(),
                                          torch.from_numpy(yval).long()),
                            batch_size=bs, shuffle=False)
    best_acc, best = -1, None
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        model.eval(); preds=[]; gts=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pr = model(xb).argmax(1).cpu().numpy()
                preds.append(pr); gts.append(yb.numpy())
        acc = (np.concatenate(preds) == np.concatenate(gts)).mean()
        if acc > best_acc:
            best_acc, best = acc, {k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best)
    return model, best_acc


def run_cv_and_print(Xn, y, groups, classes, use_mlp=True):
    gkf = GroupKFold(n_splits=5)
    lin_accs, mlp_accs = [], []
    cm_sum = np.zeros((len(classes), len(classes)), dtype=int)
    W_accum = []
    Gsig_folds = []   # <<< 新增：收集每折的有符号 SHAP（按类聚合）

    for fold, (tr, te) in enumerate(gkf.split(Xn, y, groups), 1):
        Xtr, Xte = Xn[tr], Xn[te]
        ytr, yte = y[tr],  y[te]

        # 取 10% 训练做 val（仅用于 MLP 早停）
        n = len(tr); idx = np.random.permutation(n)
        nv = max(1, int(0.1*n))
        val_sel, tr_sel = idx[:nv], idx[nv:]

        # 线性
        lin = train_linear(Xtr[tr_sel], ytr[tr_sel])
        ypred = lin.predict(Xte)
        acc = accuracy_score(yte, ypred)
        lin_accs.append(acc); cm_sum += confusion_matrix(yte, ypred, labels=range(len(classes)))
        W_accum.append(lin.coef_)          # [C,K]

        # <<< 这里：计算本折 SHAP（baseline=0, 已标准化），并追加到列表
        _, G_signed, fid = shap_linear_zero_fold(lin, Xte, yte, classes)
        Gsig_folds.append(G_signed)
        print(f"[Fold {fold} SHAP] ArgmaxCons={fid['argmax_consistency']:.3f} "
              f"R2={np.round(fid['r2_per_class'],3)} MAE={np.round(fid['mae_per_class'],3)}")
        
        '''# 更细的控制：R2 用小数，MAE 用科学计数法
        r2_str  = np.array2string(fid["r2_per_class"],
                                formatter={'float_kind': lambda x: f"{x:.6f}"})
        mae_str = np.array2string(fid["mae_per_class"],
                                formatter={'float_kind': lambda x: f"{x:.3e}"})
        print(f"[Fold {fold} SHAP] ArgmaxCons={fid['argmax_consistency']:.3f} "
              f"R2={r2_str} MAE={mae_str}")'''

        # MLP（可选）
        if use_mlp:
            model, bestval = train_mlp(Xtr[tr_sel], ytr[tr_sel], Xtr[val_sel], ytr[val_sel], len(classes))
            device = next(model.parameters()).device
            with torch.no_grad():
                xb = torch.from_numpy(Xte).float().to(device)
                logits = model(xb)
                acc_m = (logits.argmax(1).cpu().numpy()==yte).mean()
            mlp_accs.append(acc_m)
            print(f"[Fold {fold}] Linear {acc:.3f} | MLP {acc_m:.3f}")
        else:
            print(f"[Fold {fold}] Linear {acc:.3f}")

    print(f"\n[CV] Linear  mean acc: {np.mean(lin_accs):.3f} ± {np.std(lin_accs):.3f}")
    if use_mlp:
        print(f"[CV] MLP     mean acc: {np.mean(mlp_accs):.3f} ± {np.std(mlp_accs):.3f}")
    print("\n[Linear Confusion Matrix (sum over folds)]:\n", cm_sum)

    W_mean = np.mean(np.stack(W_accum, axis=0), axis=0)   # [C,K]
    return W_mean, Gsig_folds   # <<< 返回 SHAP 折列表



# ========= A. Top-K 可视化（发散条形）- 保存版 =========
def visualize_topk_bars(W_mean, classes, topk=10, show_neg=True,
                        out_dir=None, filename_prefix="topk", show=False):
    """
    对每个任务类 c，绘制发散条形图并保存：
    - 文件名：{filename_prefix}_{classname}.png
    - out_dir：保存目录；None 则不保存
    - show：是否 plt.show()（默认不弹窗）
    """
    C, K = W_mean.shape
    _ensure_dir(out_dir)

    for ci, cname in enumerate(classes):
        w = W_mean[ci]
        pos_idx = np.argsort(w)[::-1][:topk]
        neg_idx = np.argsort(w)[:topk] if show_neg else np.array([], dtype=int)

        pos_vals = w[pos_idx]
        neg_vals = w[neg_idx]

        labels = [f"m{int(k)}" for k in neg_idx[::-1]] + [f"m{int(k)}" for k in pos_idx]
        vals   = list(neg_vals[::-1]) + list(pos_vals)

        plt.figure(figsize=(8, 4.5))
        y = np.arange(len(vals))
        colors = ["#d95f02"]*len(neg_vals) + ["#1b9e77"]*len(pos_vals)  # 左负右正
        plt.barh(y, vals, color=colors)
        plt.yticks(y, labels, fontsize=9)
        plt.axvline(0, color="k", lw=0.8)
        plt.title(f"Top-K modes for class: {cname}  (K={topk})")
        plt.xlabel("linear weight (mean over folds)")
        plt.tight_layout()

        if out_dir:
            fpath = os.path.join(out_dir, f"{filename_prefix}_{cname}.png")
            plt.savefig(fpath, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

def print_top_modes(W_mean, classes, topk=10):
    W_mean = np.asarray(W_mean)
    for ci, cname in enumerate(classes):
        w = np.asarray(W_mean[ci]).ravel()    # [K]
        pos_idx = np.argsort(w)[::-1][:topk]
        neg_idx = np.argsort(w)[:topk]
        print(f"\n=== Class: {cname} ===")
        print(" Top + modes (support the class):")
        for r, k in enumerate(pos_idx, 1):
            kk = int(k)
            wk = float(w[kk])                 # 转为 Python float
            print(f"  #{r:2d} mode {kk:3d}  w={wk:+.4f}")
        print(" Top - modes (against the class):")
        for r, k in enumerate(neg_idx, 1):
            kk = int(k)
            wk = float(w[kk])                 # 转为 Python float
            print(f"  #{r:2d} mode {kk:3d}  w={wk:+.4f}")


# ========= B. t-SNE / 聚类：签名构造与可视化 =========
def _softmax(a, axis=-1):
    a = a - np.max(a, axis=axis, keepdims=True)
    e = np.exp(a)
    return e / np.sum(e, axis=axis, keepdims=True)

def build_model_signatures(W_mean, use_softmax=False, l2norm=True):
    """
    模型视角签名：每个模态 k 的签名 = W_mean[:, k] (7维)。
    - use_softmax=True 时，将7维做softmax归一化为“归属分布”
    - l2norm=True 时，对每个签名除以其L2范数（保方向，去量纲）
    返回: S_model [K, C]
    """
    C, K = W_mean.shape
    S = W_mean.T.copy()  # [K, C]
    if use_softmax:
        S = _softmax(S, axis=1)
    if l2norm:
        n = np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
        S = S / n
    return S

def build_data_signatures(Xn, y, num_classes, l2norm=True):
    """
    数据视角签名：每个模态 k 的签名 = [μ_{1,k},..., μ_{C,k}],
    其中 μ_{c,k} 是 y==c 的样本在第 k 维上的均值。
    返回: S_data [K, C]
    """
    K = Xn.shape[1]
    S = np.zeros((K, num_classes), dtype=np.float64)
    for c in range(num_classes):
        mask = (y == c)
        if mask.any():
            S[:, c] = Xn[mask].mean(axis=0)
        else:
            S[:, c] = 0.0
    if l2norm:
        n = np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
        S = S / n
    return S

def _ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ========= B. t-SNE / 聚类：签名 + 可视化保存版 =========
def tsne_and_cluster(signatures, classes, method_name="model",
                     perplexity=30, n_clusters=6, random_state=0,
                     out_dir=None, filename_prefix="tsne", show=False):
    """
    在 [K, C] 的签名上做 t-SNE (2D) 并 KMeans 聚类，保存散点图：
    - 文件名：{filename_prefix}_{method_name}_p{perplexity}_k{n_clusters}.png
    - 点颜色：该模态的“主归属类”（签名 argmax）
    - 点形状：聚类簇 id
    只保存/展示图像；终端仍打印簇信息。
    """
    _ensure_dir(out_dir)
    K, C = signatures.shape
    primary = np.argmax(signatures, axis=1)

    Y = TSNE(
        n_components=2, perplexity=perplexity, init="pca",
        learning_rate="auto", random_state=random_state
    ).fit_transform(signatures)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    cid = km.fit_predict(signatures)

    plt.figure(figsize=(7.5, 6))
    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
    for k in range(K):
        plt.scatter(
            Y[k, 0], Y[k, 1],
            c=f"C{primary[k]}", marker=markers[cid[k] % len(markers)],
            s=30, edgecolors="none"
        )
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=classes[i],
                                 markerfacecolor=f"C{i}", markersize=8)
                      for i in range(C)]
    plt.legend(handles=legend_handles, title="Primary class", loc="best", fontsize=8)
    plt.title(f"t-SNE of modes ({method_name} signatures), perplexity={perplexity}, k={n_clusters}")
    plt.tight_layout()

    if out_dir:
        fname = f"{filename_prefix}_{method_name}_p{perplexity}_k{n_clusters}.png"
        fpath = os.path.join(out_dir, fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # 终端打印簇信息
    print(f"\n[TSNE/CLUSTER] {method_name} signatures")
    for g in range(n_clusters):
        idx = np.where(cid == g)[0]
        if idx.size == 0:
            print(f" - Cluster {g}: empty")
            continue
        counts = np.bincount(primary[idx], minlength=C)
        top_classes = np.argsort(counts)[::-1]
        top_classes_names = [f"{classes[c]}:{counts[c]}" for c in top_classes[:3]]
        scores = signatures[idx].max(axis=1)
        rep_local = idx[np.argsort(scores)[::-1][:10]]
        print(f" - Cluster {g}: size={len(idx)} | top classes={top_classes_names} | representative modes={rep_local.tolist()}")

# ==========================
# SHAP
# ==========================

def shap_linear_zero_fold(lin_clf, Xte, yte, classes):
    """
    线性探针在“标准化后 + baseline=0”下的 SHAP（解释 logit）。
    输入:
      lin_clf : 已训练好的 sklearn LogisticRegression (multinomial)
      Xte     : [Nte,K]  —— 已做你同样的预处理 + z-score 后的特征
      yte     : [Nte]     —— 测试集标签（0..C-1）
      classes : list[str] —— 类名
    输出:
      G_abs   : [C,K]     —— 按类聚合( mean |phi| )的“模态→类”强度
      G_signed: [C,K]     —— 按类聚合( mean phi )的有符号贡献（看方向）
      fid     : dict      —— 忠实度（MAE、R2、argmax一致率）
    """
    C = len(classes); K = Xte.shape[1]
    W = lin_clf.coef_          # [C,K]
    b = lin_clf.intercept_     # [C]

    # 真 logit 及 SHAP 重建
    Z_true = lin_clf.decision_function(Xte)             # [Nte,C]
    Z_hat  = (Xte @ W.T) + b[None, :]                   # [Nte,C]  因为 phi0=b, sum_k phi = x·w

    # 忠实度
    mae = np.mean(np.abs(Z_hat - Z_true), axis=0)       # per-class
    r2  = []
    for c in range(C):
        yt = Z_true[:, c]; yh = Z_hat[:, c]
        ss_res = np.sum((yh - yt)**2)
        ss_tot = np.sum((yt - yt.mean())**2) + 1e-12
        r2.append(1.0 - ss_res/ss_tot)
    r2 = np.array(r2)
    y_pred_true = Z_true.argmax(axis=1)
    y_pred_hat  = Z_hat.argmax(axis=1)
    consistency = float(np.mean(y_pred_true == y_pred_hat))
    fid = {"mae_per_class": mae, "r2_per_class": r2, "argmax_consistency": consistency}

    # 逐类样本的 SHAP：phi_c(x)=x * w_c
    # 聚合成“模态→类”
    G_abs    = np.zeros((C, K), dtype=float)
    G_signed = np.zeros((C, K), dtype=float)
    for c in range(C):
        idx = (yte == c)
        if not np.any(idx): continue
        Phi_c = Xte[idx] * W[c][None, :]          # [Nc,K]
        G_abs[c]    = np.mean(np.abs(Phi_c), axis=0)
        G_signed[c] = np.mean(Phi_c, axis=0)

    return G_abs, G_signed, fid

def plot_heatmap_G(G, classes, title, save_path, vmax=None):
    """
    G: [C,K]  行=类, 列=模态索引; 建议传 G_abs_bar 或 G_signed_bar 的某种归一化版本
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 3 + 0.35*len(classes)))
    if vmax is None: vmax = np.percentile(np.abs(G), 99)
    plt.imshow(G, aspect='auto', cmap='bwr', vmin=-vmax, vmax=+vmax)
    plt.colorbar(label='Contribution (logit space)')
    plt.yticks(range(len(classes)), classes)
    plt.xticks(np.arange(0, G.shape[1], 20))
    plt.xlabel('Mode index (k)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

def plot_topk_bars(G_row, k_top, row_name, signed=True, save_path=''):
    """
    对单个类的贡献向量 G_row[K] 画 Top-K 条形图：
      - signed=True: 用有符号均值排序（显示方向）
      - signed=False: 用强度 |.| 排序
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if signed:
        order = np.argsort(G_row)[::-1]
        vals  = G_row[order][:k_top]
        labels= order[:k_top]
        title = f"{row_name}: Top-{k_top} modes by signed mean SHAP"
    else:
        order = np.argsort(np.abs(G_row))[::-1]
        vals  = G_row[order][:k_top]
        labels= order[:k_top]
        title = f"{row_name}: Top-{k_top} modes by |mean SHAP|"

    plt.figure(figsize=(9, 4))
    colors = ['tab:red' if v>=0 else 'tab:blue' for v in vals]
    plt.bar(np.arange(k_top), vals, tick_label=labels, color=colors)
    plt.axhline(0, lw=1, color='k')
    plt.xlabel('Mode index')
    plt.ylabel('Contribution (logit)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

# ==========================
# main
# ==========================
def main():
    # 读取 & 组装
    B, task_names, K, Nsub, Ntask = load_struct_mat(MAT_PATH)
    X, ytask, groups = build_X_y_groups(B, task_names, Ntask, Nsub, K)
    super_names, classes, _ = make_classes(task_names)

    # 体检
    data_check(X, B, task_names, Ntask, Nsub)

    # 仅在 Python 端剔除整列 NaN 的被试
    invalid_subs = find_invalid_subjects(B, task_names, Ntask, Nsub)
    X, ytask, groups = filter_rows_by_subjects(X, ytask, groups, invalid_subs, Nsub)

    # 过滤后再做 7 大类标签
    y = make_labels_after_filter(ytask, super_names, classes)
    print(f"[CHECK] After subject filtering: X NaN={np.isnan(X).sum()}, Inf={np.isinf(X).sum()}, shape={X.shape}")

    # 预处理
    Xn, _ = preprocess(X, method=USE_PREPROC)

    # 交叉验证训练，打印结果并取平均权重
    W_mean, Gsig_folds = run_cv_and_print(Xn, y, groups, classes, use_mlp=USE_MLP)

    # 打印 Top-K 模态
    print_top_modes(W_mean, classes, topk=10)

    # Top-K 发散条形，按类各生成一张图到目录
    visualize_topk_bars(W_mean, classes, topk=10, show_neg=True,
                        out_dir="/home/wmy/geodemo/figs/topk", filename_prefix="topk", show=False)

    # 模型视角签名（W_mean[:,k]）
    S_model = build_model_signatures(W_mean, use_softmax=False, l2norm=True)  # [K,C]
    tsne_and_cluster(S_model, classes, method_name="model",
                    perplexity=30, n_clusters=7,
                    out_dir="/home/wmy/geodemo/figs/tsne", filename_prefix="tsne", show=False)

    # 数据视角签名（类内均值）
    S_data = build_data_signatures(Xn, y, num_classes=len(classes), l2norm=True)  # [K,C]
    tsne_and_cluster(S_data, classes, method_name="data",
                    perplexity=30, n_clusters=7,
                    out_dir="/home/wmy/geodemo/figs/tsne", filename_prefix="tsne", show=False)

    # === 汇总（仅有符号贡献） ===
    # 交叉验证训练 + 取平均权重 + 收集每折 SHAP（有符号）
    #W_mean, Gsig_folds = run_cv_and_print(Xn, y, groups, classes, use_mlp=USE_MLP)

    # ...（Top-K 与 t-SNE 的代码保持不变） ...

    # === 仅“有符号贡献”的 SHAP 可视化（折间平均） ===
    if not Gsig_folds:
        raise RuntimeError("No SHAP results collected. Ensure run_cv_and_print() appends G_signed per fold.")
    G_sig_bar = np.mean(np.stack(Gsig_folds, axis=0), axis=0)    # [C,K]

    plot_heatmap_G(
        G_sig_bar, classes,
        title="SHAP signed mean (baseline=0, logit)",
        save_path="/home/wmy/geodemo/figs/SHAP/shap_signed_heatmap.png"  # <<< 修正路径
    )

    topk = 10
    for ci, cname in enumerate(classes):
        plot_topk_bars(
            G_sig_bar[ci], topk, cname,
            signed=True,
            save_path=f"/home/wmy/geodemo/figs/SHAP/top{topk}_{cname}_signed.png"  # <<< 修正路径
        )

    print("[SHAP] Saved: /home/wmy/geodemo/figs/SHAP/shap_signed_heatmap.png and per-class signed Top-K bars.")



if __name__ == "__main__":
    main()
