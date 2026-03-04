#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
beta_classify_transformer_window_flex.py

Transformer window classification on raw eigenmode beta time series.

This version adds an option to classify:
  (A) Between-task (superclass): social/motor/gambling/wm/language/relational
  (B) Within-task phase (e.g., SOCIAL: stim_mental vs stim_random) using TSV columns

Key constraints preserved:
  - Keep raw time-series waveform within each window (no mean/std aggregation).
  - Preserve magnitude information BUT stabilize cross-subject training via subject-level
    robust scaling (divide by subject median(|beta|)) and pass log(scale) as an aux feature.

Outputs:
  - Grid search results:  BASE_GEOM/_classify_window_results/grid_W_stride_<LABEL_MODE>.tsv
  - Debug figures:        BASE_GEOM/_debug_transformer_flex/
"""

import os
import re
import time
from dataclasses import dataclass
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import confusion_matrix


# ==========================
# Paths / data layout
# ==========================
BASE_GEOM = "/mnt2/wmy/geometry"
MODE = "masked"                           # "masked" or "unmasked"
HEMI = "L"
ENCS = ["LR"]
TASKS = ["SOCIAL", "MOTOR", "WM", "RELATIONAL", "LANGUAGE", "GAMBLING"]  # add "EMOTION" if needed

SUBJECTS: List[str] = []
LISTED = True
SUBJECT_LIST_PATH = "/home/wmy/work/geometry/data/subject_list_HCP.txt"

TAG_FILENAME_CANDIDATES = [
    f"{HEMI}.native_time_tags.tsv",
    f"{HEMI.lower()}.native_time_tags.tsv",
    "L.native_time_tags.tsv",
]

# ==========================
# Labeling mode (NEW)
# ==========================
# Choose one:
#   "between_task"       -> classify tasks across TASKS
#   "within_task_phase"  -> classify phases within a single task (e.g., SOCIAL mental vs random)
LABEL_MODE = "between_task"

# When LABEL_MODE="within_task_phase":
TARGET_TASK = "SOCIAL"
PHASE_COLUMN = "phase"   # "phase" or "tag" column from tsv
TARGET_PHASES = ["stim_mental", "stim_random"]  # set None to include all phases

# For between_task mode: (emotion excluded by default)
BETWEEN_TASK_CLASSES = ["social", "motor", "gambling", "wm", "language", "relational"]

# ==========================
# Window settings
# ==========================
WINDOW_SIZE = 9
STRIDE = 2
MAX_WIN_FRAC = 0.5          # W <= T * MAX_WIN_FRAC
LABEL_PURITY = 0.8
STRICT_WINDOW_LABEL = False  # True => require purity==1.0

# Aux magnitude features
USE_AUX_MAG_FEATURES = True
AUX_INCLUDE_LOG_SUBJECT_SCALE = True

# ==========================
# Training settings
# ==========================
N_SPLITS = 5
EPOCHS = 50
BATCH_SIZE = 64
LR = 3e-5
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
VAL_FRAC = 0.10
SEED = 0

# Transformer hyperparams
D_MODEL = 200
N_HEAD = 4
N_LAYERS = 2
D_FF = 256
DROPOUT = 0.1
LABEL_SMOOTHING = 0.1

# ==========================
# Debug / Visualization
# ==========================
DEBUG_PRINT_INPUT_STATS = False
DEBUG_SAVE_TRAIN_CURVES = True
DEBUG_SAVE_ATTENTION = True
DEBUG_OUTDIR = os.path.join(BASE_GEOM, "_debug_transformer_flex")
os.makedirs(DEBUG_OUTDIR, exist_ok=True)


# ==========================
# Utilities
# ==========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def map_to_superclass(name: str) -> str:
    n = str(name).lower()
    if n.startswith("social"): return "social"
    if n.startswith("motor"): return "motor"
    if n.startswith("gambling"): return "gambling"
    if n.startswith("wm"): return "wm"
    if n.startswith("language"): return "language"
    if n.startswith("emotion"): return "emotion"
    if n.startswith("relational"): return "relational"
    m = re.match(r"([a-z]+)", n)
    return m.group(1) if m else "unknown"


def _find_tag_file(run_dir: str) -> str | None:
    for fn in TAG_FILENAME_CANDIDATES:
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            return p
    cand = [f for f in os.listdir(run_dir) if f.endswith(".tsv")]
    if len(cand) == 1:
        return os.path.join(run_dir, cand[0])
    return None


def _load_tsv(tag_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Return (task_list, phase_list, tag_list), all lowercase, length T."""
    with open(tag_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    data = np.genfromtxt(tag_path, dtype=str, delimiter="\t", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]
    cols = [h.lower() for h in header]

    def col_idx(name: str):
        return cols.index(name) if name in cols else None

    task_i = col_idx("task")
    phase_i = col_idx("phase")
    tag_i = col_idx("tag")

    if task_i is None:
        for k in ["task_level1", "task_name", "label_task", "super_task"]:
            if k in cols:
                task_i = cols.index(k); break
    if phase_i is None:
        for k in ["task_level2", "condition", "subtask", "label_phase"]:
            if k in cols:
                phase_i = cols.index(k); break

    T = data.shape[0]
    tasks = [str(data[i, task_i]).strip().lower() if task_i is not None else "unknown" for i in range(T)]
    phases = [str(data[i, phase_i]).strip().lower() if phase_i is not None else "na" for i in range(T)]
    tags = [str(data[i, tag_i]).strip().lower() if tag_i is not None else "na" for i in range(T)]
    return tasks, phases, tags


def _majority_with_frac(labels: List[str]) -> Tuple[str, float]:
    if not labels:
        return "unknown", 0.0
    cnt = Counter(labels)
    lab, n = cnt.most_common(1)[0]
    frac = float(n) / float(len(labels))
    return lab, frac


# ==========================
# Dataset builder
# ==========================
def build_dataset_window_sequence(
    window_size: int,
    stride: int,
    max_win_frac: float,
    purity: float,
    label_mode: str,
):
    """
    Returns:
      X_seq: (N, W, K) float32
      X_aux: (N, A) float32
      y: (N,) int
      groups: (N,) int subject
      classes: list[str]
    """
    X_seq_list, X_aux_list, y_list, g_list = [], [], [], []
    stat_total, stat_kept = 0, 0
    scale_cache = {}  # subject -> scale

    if label_mode == "between_task":
        classes = BETWEEN_TASK_CLASSES[:]
        cls_to_id = {c: i for i, c in enumerate(classes)}
        task_folders = TASKS
    elif label_mode == "within_task_phase":
        task_folders = [TARGET_TASK]
        if TARGET_PHASES is not None:
            classes = [f"{TARGET_TASK.lower()}:{p.lower()}" for p in TARGET_PHASES]
            cls_to_id = {c: i for i, c in enumerate(classes)}
        else:
            classes = []
            cls_to_id = {}
    else:
        raise ValueError("LABEL_MODE must be 'between_task' or 'within_task_phase'")

    # subject list
    subs = SUBJECTS
    if LISTED and os.path.exists(SUBJECT_LIST_PATH):
        with open(SUBJECT_LIST_PATH, "r", encoding="utf-8") as f:
            subs = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    if not subs:
        root0 = os.path.join(BASE_GEOM, task_folders[0])
        if os.path.isdir(root0):
            subs = sorted([d for d in os.listdir(root0) if os.path.isdir(os.path.join(root0, d))])

    for task_name in task_folders:
        task_root = os.path.join(BASE_GEOM, task_name)
        if not os.path.isdir(task_root):
            print(f"[SKIP] missing task root: {task_root}")
            continue

        valid_subjects = set()

        for sub in subs:
            for enc in ENCS:
                run_dir = os.path.join(task_root, sub, f"tfMRI_{task_name}_{enc}")
                if not os.path.isdir(run_dir):
                    continue

                beta_path = os.path.join(run_dir, f"{HEMI}.native.{MODE}_beta.npy")
                if not os.path.exists(beta_path):
                    continue

                tag_path = _find_tag_file(run_dir)
                if tag_path is None or (not os.path.exists(tag_path)):
                    continue

                beta = np.load(beta_path)  # (K, T)
                beta = np.sign(beta) * np.log1p(np.abs(beta))  # stabilize small values while preserving sign
                K, T = beta.shape

                tasks_l1, phases_l2, tags = _load_tsv(tag_path)
                if len(tasks_l1) != T:
                    print(f"[SKIP] T mismatch: beta T={T} vs tags={len(tasks_l1)} | {run_dir}")
                    continue

                # subject-level robust scale
                if sub not in scale_cache:
                    scale_cache[sub] = float(np.median(np.abs(beta)) + 1e-8)
                s_sub = scale_cache[sub]

                w_max = int(np.floor(T * float(max_win_frac)))
                if window_size > w_max:
                    continue

                valid_subjects.add(sub)

                # label per TR
                if label_mode == "between_task":
                    lab_seq = [map_to_superclass(tasks_l1[t]) for t in range(T)]
                else:
                    if PHASE_COLUMN.lower() == "phase":
                        lab_seq = [str(phases_l2[t]).strip().lower() for t in range(T)]
                    elif PHASE_COLUMN.lower() == "tag":
                        lab_seq = [str(tags[t]).strip().lower() for t in range(T)]
                    else:
                        raise ValueError("PHASE_COLUMN must be 'phase' or 'tag'")

                # sliding windows
                for t0 in range(0, T - window_size + 1, int(stride)):
                    stat_total += 1
                    win_labels = lab_seq[t0:t0 + window_size]
                    lab, frac = _majority_with_frac(win_labels)

                    req_purity = 1.0 if STRICT_WINDOW_LABEL else float(purity)
                    if frac < req_purity:
                        continue

                    # map to y
                    if label_mode == "between_task":
                        if lab not in cls_to_id:
                            continue
                        y_id = cls_to_id[lab]
                    else:
                        if TARGET_PHASES is not None and lab not in [p.lower() for p in TARGET_PHASES]:
                            continue
                        lab_full = f"{TARGET_TASK.lower()}:{lab}"
                        if lab_full not in cls_to_id:
                            if TARGET_PHASES is None:
                                cls_to_id[lab_full] = len(classes)
                                classes.append(lab_full)
                            else:
                                continue
                        y_id = cls_to_id[lab_full]

                    # segment
                    seg = beta[:, t0:t0 + window_size]  # (K, W)

                    # stabilize across subjects (keep magnitude in aux)
                    seg = seg / s_sub

                    seg_wk = np.ascontiguousarray(seg.T, dtype=np.float32)  # (W, K)

                    if USE_AUX_MAG_FEATURES:
                        abs_seg = np.abs(seg)
                        aux_list = [
                            float(abs_seg.mean()),
                            float(abs_seg.sum()),
                            float((seg ** 2).mean()),
                            float((seg ** 2).sum()),
                        ]
                        if AUX_INCLUDE_LOG_SUBJECT_SCALE:
                            aux_list.append(float(np.log(s_sub)))
                        aux = np.array(aux_list, dtype=np.float32)
                    else:
                        aux = np.zeros((0,), dtype=np.float32)

                    X_seq_list.append(seg_wk)
                    X_aux_list.append(aux)
                    y_list.append(int(y_id))
                    g_list.append(int(sub))
                    stat_kept += 1

        print(f"[STAT] Task {task_name}: valid subjects = {len(valid_subjects)}")

    if not X_seq_list:
        raise RuntimeError("No samples collected. Check paths / filters / phases.")

    X_seq = np.stack(X_seq_list, axis=0)
    X_aux = np.stack(X_aux_list, axis=0) if USE_AUX_MAG_FEATURES else np.zeros((len(X_seq_list), 0), dtype=np.float32)
    y = np.array(y_list, dtype=int)
    groups = np.array(g_list, dtype=int)

    print(f"[DATA] X_seq={X_seq.shape}, X_aux={X_aux.shape}, groups(unique)={np.unique(groups).size}")
    print(f"[WINDOW] total tried={stat_total}, kept={stat_kept}, keep_ratio={stat_kept/max(1,stat_total):.3f}")

    cnt = Counter(y.tolist())
    print(f"[LABEL] n_classes={len(classes)}")
    for cid, n in cnt.most_common(30):
        print(f"  {cid:4d}  {n:8d}  {classes[cid]}")

    return X_seq, X_aux, y, groups, classes


# ==========================
# Model
# ==========================
class EncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_out, attn_w = self.mha(x1, x1, x1, need_weights=True, average_attn_weights=False)
        x = x + self.drop(attn_out)
        x2 = self.norm2(x)
        x = x + self.ffn(x2)
        return x, attn_w


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(x + self.pos)


class TransformerWindowClassifier(nn.Module):
    def __init__(self, k_in, window_size, n_classes,
                 d_model=200, n_head=4, n_layers=2, d_ff=256,
                 dropout=0.1, aux_dim=0):
        super().__init__()
        self.aux_dim = aux_dim
        self.in_proj = nn.Linear(k_in, d_model)
        self.in_scale = nn.Parameter(torch.tensor(0.1))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.pos = LearnablePositionalEncoding(window_size + 1, d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayerWithAttn(d_model, n_head, d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        head_in = d_model + aux_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x_seq, x_aux=None, return_attn=False):
        x_seq = x_seq * self.in_scale
        h = self.in_proj(x_seq)
        B = h.shape[0]
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos(h)
        attn_all = []
        for layer in self.layers:
            h, attn = layer(h)
            attn_all.append(attn)
        h = h[:, 0, :]
        if self.aux_dim > 0:
            if x_aux is None:
                raise ValueError("aux_dim>0 but x_aux is None")
            h = torch.cat([h, x_aux], dim=1)
        logits = self.head(h)
        if return_attn:
            return logits, attn_all
        return logits


# ==========================
# Training
# ==========================
@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        if len(batch) == 3:
            xb, xa, yb = batch
            xb, xa = xb.to(device), xa.to(device)
            logits = model(xb, xa)
        else:
            xb, yb = batch
            xb = xb.to(device)
            logits = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
        gts.append(yb.numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    return float((preds == gts).mean()), preds, gts

@torch.no_grad()
def eval_loss_acc(model, loader, device, crit):
    model.eval()
    total_loss, total_n = 0.0, 0
    preds, gts = [], []
    for batch in loader:
        if len(batch) == 3:
            xb, xa, yb = batch
            xb, xa, yb = xb.to(device), xa.to(device), yb.to(device)
            logits = model(xb, xa)
        else:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
        loss = crit(logits, yb)
        bs = int(yb.shape[0])
        total_loss += float(loss.detach().cpu()) * bs
        total_n += bs
        preds.append(logits.argmax(1).cpu().numpy())
        gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    acc = float((preds == gts).mean())
    mean_loss = total_loss / max(1, total_n)
    return mean_loss, acc, preds, gts

def train_transformer_one_fold(X_seq, X_aux, y, tr_idx, te_idx, n_classes: int, classes: List[str], fold: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_idx = np.array(tr_idx)
    rng = np.random.default_rng(SEED + fold)
    perm = rng.permutation(len(tr_idx))
    nv = max(1, int(VAL_FRAC * len(tr_idx)))
    val_sel = tr_idx[perm[:nv]]
    tr_sel = tr_idx[perm[nv:]]

    xtr = torch.from_numpy(X_seq[tr_sel]).float()
    xva = torch.from_numpy(X_seq[val_sel]).float()
    xte = torch.from_numpy(X_seq[te_idx]).float()

    ytr = torch.from_numpy(y[tr_sel]).long()
    yva = torch.from_numpy(y[val_sel]).long()
    yte = torch.from_numpy(y[te_idx]).long()

    aux_dim = X_aux.shape[1]
    if aux_dim > 0:
        atr = torch.from_numpy(X_aux[tr_sel]).float()
        ava = torch.from_numpy(X_aux[val_sel]).float()
        ate = torch.from_numpy(X_aux[te_idx]).float()
        tr_ds = TensorDataset(xtr, atr, ytr)
        va_ds = TensorDataset(xva, ava, yva)
        te_ds = TensorDataset(xte, ate, yte)
    else:
        tr_ds = TensorDataset(xtr, ytr)
        va_ds = TensorDataset(xva, yva)
        te_ds = TensorDataset(xte, yte)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    # class weights
    cls_counts = np.bincount(y[tr_sel], minlength=n_classes).astype(np.float32)
    cls_weights = (cls_counts.sum() / (cls_counts + 1e-6))
    cls_weights = cls_weights / cls_weights.mean()
    w_t = torch.from_numpy(cls_weights).float().to(device)

    model = TransformerWindowClassifier(
        k_in=X_seq.shape[2],
        window_size=X_seq.shape[1],
        n_classes=n_classes,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        aux_dim=aux_dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss(weight=w_t, label_smoothing=LABEL_SMOOTHING)

    best_acc = -1.0
    best_state = None
    patience = 8
    bad = 0
    hist_tr_loss, hist_va_acc = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_n = 0
        for batch in tr_loader:
            opt.zero_grad(set_to_none=True)
            if len(batch) == 3:
                xb, xa, yb = batch
                xb, xa, yb = xb.to(device), xa.to(device), yb.to(device)
                logits = model(xb, xa)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)

            loss = crit(logits, yb)
            loss.backward()
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            epoch_loss += float(loss.detach().cpu()) * int(yb.shape[0])
            epoch_n += int(yb.shape[0])

        epoch_loss = epoch_loss / max(1, epoch_n)
        hist_tr_loss.append(epoch_loss)

        va_acc, _, _ = eval_model(model, va_loader, device)
        hist_va_acc.append(float(va_acc))

        if va_acc > best_acc + 1e-4:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if DEBUG_SAVE_TRAIN_CURVES:
        fig_path = os.path.join(DEBUG_OUTDIR, f"fold{fold}_curve_{LABEL_MODE}_W{X_seq.shape[1]}_d{D_MODEL}.png")
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(hist_tr_loss)
        plt.title("Train loss")
        plt.xlabel("epoch"); plt.ylabel("loss")
        plt.subplot(1, 2, 2)
        plt.plot(hist_va_acc)
        plt.title("Val acc")
        plt.xlabel("epoch"); plt.ylabel("acc")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"[SAVE] train curves -> {fig_path}")

    if DEBUG_SAVE_ATTENTION:
        batch = next(iter(te_loader))
        if len(batch) == 3:
            xb, xa, _ = batch
            xb, xa = xb.to(device), xa.to(device)
            _, attn_all = model(xb, xa, return_attn=True)
        else:
            xb, _ = batch
            xb = xb.to(device)
            _, attn_all = model(xb, return_attn=True)

        attn = attn_all[-1][0].mean(dim=0).detach().cpu().numpy()
        fig_path = os.path.join(DEBUG_OUTDIR, f"fold{fold}_attn_{LABEL_MODE}_W{X_seq.shape[1]}_d{D_MODEL}.png")
        plt.figure(figsize=(6, 5))
        plt.imshow(attn, aspect="auto")
        plt.colorbar()
        plt.title("Attention (last layer, mean heads)")
        plt.xlabel("Key position"); plt.ylabel("Query position")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"[SAVE] attention heatmap -> {fig_path}")

    tr_acc, tr_pred, _ = eval_model(model, tr_loader, device)
    te_acc, te_pred, te_gt = eval_model(model, te_loader, device)

    def dist(arr, n):
        c = np.bincount(arr, minlength=n)
        return {classes[i]: int(c[i]) for i in range(n) if c[i] > 0}

    print("[TRAIN] acc", f"{tr_acc:.3f}", "pred_dist", dist(tr_pred, n_classes))
    print("[TEST ] acc", f"{te_acc:.3f}", "pred_dist", dist(te_pred, n_classes))

    tr_loss_e, tr_acc_e, tr_pred_e, _ = eval_loss_acc(model, tr_loader, device, crit)
    va_loss_e, va_acc_e, va_pred_e, _ = eval_loss_acc(model, va_loader, device, crit)

    cnt = np.bincount(va_pred_e, minlength=n_classes)
    print(f"[Epoch {epoch}] tr_loss={tr_loss_e:.3f} tr_acc={tr_acc_e:.3f} | "
        f"va_loss={va_loss_e:.3f} va_acc={va_acc_e:.3f} | va_pred_dist={cnt.tolist()}")
    
    return float(te_acc), te_pred, te_gt


# ==========================
# CV runner
# ==========================
def run_cv_transformer(X_seq, X_aux, y, groups, classes, n_splits=5):
    uniq_g = np.unique(groups)
    if uniq_g.size >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X_seq, y, groups)
        split_name = "GroupKFold(subject)"
    else:
        splitter = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=0)
        split_iter = splitter.split(X_seq, y)
        split_name = "KFold(fallback)"

    print(f"[CV] splitter = {split_name}, n_splits={n_splits}")

    fold_accs = []
    cm_sum = np.zeros((len(classes), len(classes)), dtype=int)

    def _dist_y(arr, n_cls):
        c = np.bincount(arr, minlength=n_cls)
        return {classes[i]: int(c[i]) for i in range(n_cls) if c[i] > 0}

    for fold, (tr, te) in enumerate(split_iter, 1):
        print(f"\n[Fold {fold}] y_train dist:", _dist_y(y[tr], len(classes)))
        print(f"[Fold {fold}] y_test  dist:", _dist_y(y[te], len(classes)))

        acc, pred, gt = train_transformer_one_fold(
            X_seq, X_aux, y, tr, te, n_classes=len(classes), classes=classes, fold=fold
        )
        fold_accs.append(float(acc))
        cm_sum += confusion_matrix(gt, pred, labels=list(range(len(classes))))
        print(f"[Fold {fold}] Transformer acc = {acc:.3f}")

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))

    print(f"\n[CV] Transformer mean acc: {mean_acc:.3f} ± {std_acc:.3f}")
    print("\n[Confusion Matrix (sum over folds)]:\n", cm_sum)

    return mean_acc, std_acc, fold_accs, cm_sum


# ==========================
# Grid search
# ==========================
@dataclass
class WSResult:
    W: int
    stride: int
    n_samples: int
    n_subjects: int
    mean_acc: float
    std_acc: float


def grid_search_W_stride(
    W_list: List[int],
    stride_list: List[int],
    max_win_frac: float,
    purity: float,
    label_mode: str,
    min_samples: int = 300,
    topk: int = 5,
    save_path: Optional[str] = None,
):
    all_results: List[WSResult] = []
    W_list = sorted(set(int(w) for w in W_list if w > 0))
    stride_list = sorted(set(int(s) for s in stride_list if s > 0))

    for W in W_list:
        for stride in stride_list:
            if stride > W:
                continue

            t0 = time.time()
            try:
                X_seq, X_aux, y, groups, classes = build_dataset_window_sequence(
                    window_size=W,
                    stride=stride,
                    max_win_frac=max_win_frac,
                    purity=purity,
                    label_mode=label_mode,
                )
            except Exception as e:
                print(f"[GRID][SKIP] W={W}, stride={stride} | build failed: {e}")
                continue

            n_samples = int(len(y))
            n_subjects = int(len(set(groups.tolist())))
            print(f"[GRID] W={W}, stride={stride} | X_seq={X_seq.shape}, X_aux={X_aux.shape}, y={y.shape}, groups={groups.shape}")

            if n_samples < min_samples or n_subjects < N_SPLITS:
                print(f"[GRID][SKIP] W={W}, stride={stride} | samples={n_samples}, subjects={n_subjects}")
                continue

            mean_acc, std_acc, _, _ = run_cv_transformer(X_seq, X_aux, y, groups, classes, n_splits=N_SPLITS)

            dt = time.time() - t0
            print(f"[GRID] W={W:3d} stride={stride:3d} | n={n_samples:5d} subj={n_subjects:4d} | acc={mean_acc:.3f}±{std_acc:.3f} | {dt:.1f}s")

            all_results.append(WSResult(W, stride, n_samples, n_subjects, float(mean_acc), float(std_acc)))

    if not all_results:
        print("[GRID] No valid (W, stride) results.")
        return []

    all_results.sort(key=lambda r: (-r.mean_acc, r.std_acc, -r.n_samples))

    print("\n========== TOP RESULTS ==========")
    for i, r in enumerate(all_results[:topk], 1):
        print(f"[Top {i}] W={r.W}, stride={r.stride} | acc={r.mean_acc:.3f}±{r.std_acc:.3f} | n={r.n_samples}, subj={r.n_subjects}")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("W\tstride\tmean_acc\tstd_acc\tn_samples\tn_subjects\n")
            for r in all_results:
                f.write(f"{r.W}\t{r.stride}\t{r.mean_acc:.6f}\t{r.std_acc:.6f}\t{r.n_samples}\t{r.n_subjects}\n")
        print(f"[SAVE] grid results -> {save_path}")

    return all_results


def main():
    # set_seed(SEED)

    W_list = list(range(5, 16, 2))      # 5,7,9,11,13,15
    stride_list = [1, 3, 5, 7]

    out_dir = os.path.join(BASE_GEOM, "_classify_window_results")
    os.makedirs(out_dir, exist_ok=True)

    grid_search_W_stride(
        W_list=W_list,
        stride_list=stride_list,
        max_win_frac=MAX_WIN_FRAC,
        purity=LABEL_PURITY,
        label_mode=LABEL_MODE,
        min_samples=200,
        topk=5,
        save_path=os.path.join(out_dir, f"grid_W_stride_{LABEL_MODE}.tsv"),
    )


if __name__ == "__main__":
    main()