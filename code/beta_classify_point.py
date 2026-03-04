#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt

import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# ==========================
# 配置（按你 beta_task 的目录结构）
# ==========================
BASE_GEOM = "/mnt2/wmy/geometry"          # beta_task.py OUT_ROOT_TMPL 的上级
MODE = "masked"                           # "masked" or "unmasked"
HEMI = "L"                                # 目前你保存文件是 L.native.*
ENCS = ["LR"]
TASKS = ["SOCIAL","MOTOR","WM","RELATIONAL","LANGUAGE","GAMBLING"] #"EMOTION",

# 被试：空列表就自动扫描 BASE_GEOM/{task}/ 下所有 sub；否则只跑列出的
SUBJECTS = []   # or []

# 标签文件名：默认同目录下找 *.tsv（若你固定名可直接写死）
TAG_FILENAME_CANDIDATES = [
    f"{HEMI}.native_time_tags.tsv",
    f"{HEMI.lower()}.native_time_tags.tsv",
    "L.native_time_tags.tsv",
]

# 预处理 & 模型（与你原脚本一致）
USE_PREPROC = "none"      # "slog1p" 或 "asinh"
USE_MLP = True              # True/False
listed = True
N_SPLITS = 5                # k-fold


# ==========================
# 工具 & 核心函数（基本沿用你的）
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
        m = re.match(r"([a-z]+)", n)
        return m.group(1) if m else "unknown"


def preprocess(X, method="none"):
    eps = 1e-6
    if method == "slog1p":
        Xp = np.sign(X) * np.log1p(np.abs(X))
    elif method == "asinh":
        med = np.median(np.abs(X), axis=0) + eps
        Xp = np.arcsinh(X / med)
    else:
        Xp = X.copy()
    Xp[~np.isfinite(Xp)] = 0.0
    return Xp, None   # 不做标准化


def train_linear(Xtr, ytr):
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=2000, n_jobs=-1
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
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pr = model(xb).argmax(1).cpu().numpy()
                preds.append(pr); gts.append(yb.numpy())
        acc = (np.concatenate(preds) == np.concatenate(gts)).mean()
        if acc > best_acc:
            best_acc = acc
            best = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best)
    return model, best_acc


def _find_tag_file(run_dir: str) -> str | None:
    # 1) 先按候选名字找
    for fn in TAG_FILENAME_CANDIDATES:
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            return p
    # 2) 再兜底找任意 .tsv
    cand = [f for f in os.listdir(run_dir) if f.endswith(".tsv")]
    if len(cand) == 1:
        return os.path.join(run_dir, cand[0])
    return None


def _split_two_level(tag: str):
    """
    兼容你常见的 tag 写法：
      - "SOCIAL::mental" / "SOCIAL:mental"
      - "SOCIAL_mental" / "SOCIAL-mental"
      - 只有 "SOCIAL"（则 level2='na'）
    返回 (level1, level2) 均为小写
    """
    s = str(tag).strip()
    if not s:
        return "unknown", "na"

    s2 = s.replace("__", "_").replace("-", "_")
    # 优先处理 :: 或 :
    if "::" in s2:
        a, b = s2.split("::", 1)
    elif ":" in s2:
        a, b = s2.split(":", 1)
    elif "_" in s2:
        a, b = s2.split("_", 1)
    else:
        a, b = s2, "na"

    return a.strip().lower(), b.strip().lower()


def _load_tags_tsv_levels(tag_path: str):
    """
    返回两个 list，长度 T：
      - level1_tags: 每个 TR 的一级标签（大任务）
      - level2_tags: 每个 TR 的二级标签（phase/condition/子任务），若缺失则 'na'
    读取优先级：
      1) tsv 中存在 task_level1 + task_level2（或 phase/condition）列
      2) tsv 中存在单列 task/label，则用 _split_two_level() 从字符串解析
      3) 兜底：用最后一列解析
    """
    with open(tag_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    data = np.genfromtxt(tag_path, dtype=str, delimiter="\t", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]

    cols_lower = [h.lower() for h in header]

    # 尝试找到 level1 / level2 列
    l1_col = None
    for key in ["task_level1", "task", "task_name", "label_task", "super_task"]:
        if key in cols_lower:
            l1_col = cols_lower.index(key)
            break

    l2_col = None
    for key in ["task_level2", "phase", "condition", "subtask", "label_phase"]:
        if key in cols_lower:
            l2_col = cols_lower.index(key)
            break

    level1_tags, level2_tags = [], []

    if l1_col is not None and l2_col is not None:
        # 两列齐全：直接读
        for i in range(data.shape[0]):
            a = str(data[i, l1_col]).strip().lower()
            b = str(data[i, l2_col]).strip().lower()
            level1_tags.append(a if a else "unknown")
            level2_tags.append(b if b else "na")
        return level1_tags, level2_tags

    # 否则：从单列解析
    if l1_col is None:
        l1_col = data.shape[1] - 1  # 兜底最后一列

    for x in data[:, l1_col].tolist():
        a, b = _split_two_level(x)
        level1_tags.append(a)
        level2_tags.append(b)

    return level1_tags, level2_tags



def build_dataset_tr_level(label_mode="super"):
    """
    label_mode:
      - "super": y=7大类
      - "sub"  : y=二级标签(例如 social:mental / wm:2bk ...)
      - "both" : 返回 y_super, y_sub（你可以后面做 multi-task）

    返回：
      super: X, y, groups, classes
      sub  : X, y, groups, classes
      both : X, y_super, y_sub, groups, classes_super, classes_sub
    """
    X_list, y1_list, y2_list, g_list = [], [], [], []

    # 固定 7 大类
    classes_super = ["social","motor","gambling","wm","language","emotion","relational"]
    cls1_to_id = {c:i for i,c in enumerate(classes_super)}

    # 二级类：动态收集（建议用 “super:sub” 拼接，避免不同任务同名 phase 混淆）
    cls2_to_id = {}
    classes_sub = []

    for task_name in TASKS:
        task_root = os.path.join(BASE_GEOM, task_name)
        valid_subjects = set()

        if not os.path.isdir(task_root):
            print(f"[SKIP] missing task root: {task_root}")
            continue
        
        if listed:
            sub_list_path = "/home/wmy/work/geometry/data/subject_list_HCP.txt"  

            with open(sub_list_path, "r", encoding="utf-8") as f:
                SUBJECTS = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]

        subs = SUBJECTS if SUBJECTS else sorted(
            [d for d in os.listdir(task_root) if os.path.isdir(os.path.join(task_root, d))]
        )

        for sub in subs:
            for enc in ENCS:
                run_dir = os.path.join(task_root, sub, f"tfMRI_{task_name}_{enc}")
                if not os.path.isdir(run_dir):
                    # print("Missing beta:", task_name, sub)
                    continue

                beta_path = os.path.join(run_dir, f"{HEMI}.native.{MODE}_beta.npy")
                if not os.path.exists(beta_path):
                    continue

                valid_subjects.add(sub)

                tag_path = _find_tag_file(run_dir)
                if tag_path is None or (not os.path.exists(tag_path)):
                    print(f"[SKIP] tag not found for: {run_dir}")
                    continue

                beta = np.load(beta_path)  # (K,T)
                tags_l1, tags_l2 = _load_tags_tsv_levels(tag_path)  # length T
                K, T = beta.shape
                if len(tags_l1) != T:
                    print(f"[SKIP] T mismatch: beta T={T} vs tags={len(tags_l1)} | {run_dir}")
                    continue

                # for t in range(T):
                #     # 统一一级：你原本的 map_to_superclass 仍可复用
                #     super_name = map_to_superclass(tags_l1[t])

                #     if super_name not in cls1_to_id:
                #         continue

                #     # 二级：拼接为 super:sub
                #     sub_name = str(tags_l2[t]).strip().lower()
                #     if not sub_name:
                #         sub_name = "na"
                #     sub_full = f"{super_name}:{sub_name}"

                #     # 收集样本
                #     X_list.append(beta[:, t].astype(np.float32))
                #     y1_list.append(cls1_to_id[super_name])

                #     if sub_full not in cls2_to_id:
                #         cls2_to_id[sub_full] = len(classes_sub)
                #         classes_sub.append(sub_full)
                #     y2_list.append(cls2_to_id[sub_full])

                #     g_list.append(int(sub))  # groups=subject

                for t in range(T):
                    super_name = map_to_superclass(tags_l1[t])

                    # 只保留 SOCIAL
                    if super_name != "social":
                        continue

                    sub_name = str(tags_l2[t]).strip().lower()

                    # 只保留 stim_mental 和 stim_random
                    if sub_name not in ["stim_mental", "stim_random"]:
                        continue

                    sub_full = f"{super_name}:{sub_name}"

                    X_list.append(beta[:, t].astype(np.float32))

                    # 二分类标签
                    if sub_name == "stim_mental":
                        y2_list.append(0)
                    else:
                        y2_list.append(1)

                    # 手动定义类别名
                    if not classes_sub:
                        classes_sub.extend(["social:stim_mental", "social:stim_random"])

                    g_list.append(int(sub))

        print(f"[STAT] Task {task_name}: valid subjects = {len(valid_subjects)}")


    if not X_list:
        raise RuntimeError("No samples collected. Check paths, beta files, and tag files.")

    X = np.stack(X_list, axis=0)
    y1 = np.array(y1_list, dtype=int)
    y2 = np.array(y2_list, dtype=int)
    groups = np.array(g_list, dtype=int)

    print(f"[DATA] X={X.shape}, y1(super)={y1.shape}, y2(sub)={y2.shape}, groups(unique)={np.unique(groups).size}")
    print(f"[DATA] num super classes={len(classes_super)}, num sub classes={len(classes_sub)}")

    cnt1 = Counter(y1.tolist())
    cnt2 = Counter(y2.tolist())
    print("[LABEL super] n_classes =", len(classes_super))
    for cid, n in cnt1.most_common():
        print(f"  {cid:4d}  {n:8d}  {classes_super[cid]}")

    print("[LABEL sub] n_classes =", len(classes_sub))
    for cid, n in cnt2.most_common():
        print(f"  {cid:4d}  {n:8d}  {classes_sub[cid]}")

    TOPK = 50  # 你可以改成 len(classes_sub) 画全量，但一般会挤爆
    items = cnt2.most_common(TOPK)
    cids  = [cid for cid, _ in items]
    vals  = [n   for _,   n in items]
    names = [classes_sub[cid] for cid in cids]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), names, rotation=75, ha="right", fontsize=8)
    plt.ylabel("Num samples (TRs)")
    plt.title(f"Count of sub labels")
    plt.tight_layout()

    # 保存（沿用你脚本的 out_dir 逻辑；若在 builder 里没有 out_dir，就先写死到 BASE_GEOM 下）
    out_dir = os.path.join(BASE_GEOM, "_classify_tr_results")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"sub_label_counts_top{TOPK}.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[SAVE] sub label count bar -> {fig_path}")

    if label_mode == "super":
        return X, y1, groups, classes_super
    elif label_mode == "sub":
        return X, y2, groups, classes_sub
    elif label_mode == "both":
        return X, y1, y2, groups, classes_super, classes_sub
    else:
        raise ValueError("label_mode must be one of: 'super', 'sub', 'both'")


def run_cv_and_print_no_shap(Xn, y, groups, classes, use_mlp=True, n_splits=5):
    """
    完全按你原 run_cv_and_print 的训练/评估方式，但不做 SHAP。
    """
    uniq_g = np.unique(groups)
    if uniq_g.size >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(Xn, y, groups)
        split_name = "GroupKFold(subject)"
    else:
        # 兜底：否则你单被试根本跑不了
        splitter = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=0)
        split_iter = splitter.split(Xn, y)
        split_name = "KFold(fallback)"

    print(f"[CV] splitter = {split_name}, n_splits={n_splits}")

    lin_accs, mlp_accs = [], []
    cm_sum = np.zeros((len(classes), len(classes)), dtype=int)

    for fold, sp in enumerate(split_iter, 1):
        if split_name.startswith("GroupKFold"):
            tr, te = sp
        else:
            tr, te = sp

        Xtr, Xte = Xn[tr], Xn[te]
        ytr, yte = y[tr],  y[te]

        # 取 10% 训练做 val（仅用于 MLP 早停）——与你原代码一致
        n = len(tr); idx = np.random.permutation(n)
        nv = max(1, int(0.1*n))
        val_sel, tr_sel = idx[:nv], idx[nv:]

        # 线性
        lin = train_linear(Xtr[tr_sel], ytr[tr_sel])
        ypred = lin.predict(Xte)
        acc = accuracy_score(yte, ypred)
        lin_accs.append(acc)
        cm_sum += confusion_matrix(yte, ypred, labels=range(len(classes)))

        # MLP（可选）
        if use_mlp:
            model, _ = train_mlp(
                Xtr[tr_sel], ytr[tr_sel],
                Xtr[val_sel], ytr[val_sel],
                len(classes)
            )
            device = next(model.parameters()).device
            with torch.no_grad():
                xb = torch.from_numpy(Xte).float().to(device)
                logits = model(xb)
                acc_m = (logits.argmax(1).cpu().numpy() == yte).mean()
            mlp_accs.append(acc_m)
            print(f"[Fold {fold}] Linear {acc:.3f} | MLP {acc_m:.3f}")
        else:
            print(f"[Fold {fold}] Linear {acc:.3f}")

    print(f"\n[CV] Linear  mean acc: {np.mean(lin_accs):.3f} ± {np.std(lin_accs):.3f}")
    if use_mlp and mlp_accs:
        print(f"[CV] MLP     mean acc: {np.mean(mlp_accs):.3f} ± {np.std(mlp_accs):.3f}")
    print("\n[Linear Confusion Matrix (sum over folds)]:\n", cm_sum)

    return cm_sum


def save_confusion_matrix_png(cm, classes, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.title("Confusion Matrix (sum over folds)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[SAVE] CM fig -> {save_path}")


# ==========================
# main
# ==========================
def main():
    # 1) 组装数据（TR级，单TR beta）
    LABEL_MODE = "sub"   # "super" / "sub"

    X, y, groups, classes = build_dataset_tr_level(label_mode=LABEL_MODE)

    # 2) 预处理（与你原脚本一致）
    Xn, _ = preprocess(X, method=USE_PREPROC)

    # 3) CV 训练评估（不做 SHAP）
    cm_sum = run_cv_and_print_no_shap(
        Xn, y, groups, classes,
        use_mlp=USE_MLP,
        n_splits=N_SPLITS
    )

    # 4) 输出一张 CM 图（不影响你训练逻辑，只是保存可视化）
    out_dir = os.path.join(BASE_GEOM, "_classify_tr_results")
    os.makedirs(out_dir, exist_ok=True)
    save_confusion_matrix_png(
        cm_sum, classes,
        save_path=os.path.join(out_dir, f"cm_tr_{MODE}_{USE_PREPROC}.png")
    )


if __name__ == "__main__":
    main()
