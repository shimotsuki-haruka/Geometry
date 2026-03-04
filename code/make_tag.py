'''import numpy as np
import pandas as pd

def read_tab(tab_path):
    return pd.read_csv(tab_path, sep=r"\t", engine="python")

def read_ev(ev_path):
    # FSL EV: onset duration amplitude
    arr = np.loadtxt(ev_path, ndmin=2)
    if arr.size == 0:
        return np.array([])
    return arr[:, 0]  # onset seconds

def fit_a_b(tab_sec, ev_sec):
    tab_sec = np.asarray(tab_sec, float)
    ev_sec  = np.asarray(ev_sec, float)

    m = np.isfinite(tab_sec) & np.isfinite(ev_sec)
    tab_sec, ev_sec = tab_sec[m], ev_sec[m]

    n = min(len(tab_sec), len(ev_sec))
    tab_sec, ev_sec = tab_sec[:n], ev_sec[:n]

    A = np.c_[tab_sec, np.ones_like(tab_sec)]
    a, b = np.linalg.lstsq(A, ev_sec, rcond=None)[0]
    pred = a * tab_sec + b
    resid = ev_sec - pred
    return float(a), float(b), resid, tab_sec, ev_sec, pred

# --------- 你需要改成自己的路径 ----------
TAB_PATH = "/home/wmy/Documents/SOCIAL/100610/MNINonLinear/Results/tfMRI_SOCIAL_LR/SOCIAL_run2_TAB.txt"     # LR 对应 run2
EV_MENTAL = "/home/wmy/Documents/SOCIAL/100610/MNINonLinear/Results/tfMRI_SOCIAL_LR/EVs/mental.txt"
EV_RANDOM = "/home/wmy/Documents/SOCIAL/100610/MNINonLinear/Results/tfMRI_SOCIAL_LR/EVs/rnd.txt"
# ----------------------------------------

tab = read_tab(TAB_PATH)

# 以 SOCIAL 为例：用每个 movie 的开始作为 TAB anchor（与你之前 MovieSlide.OnsetTime 一致）
social_rows = tab[tab["Procedure"].eq("SOCIALrunPROC")].copy()
tab_onsets_sec = social_rows["MovieSlide.OnsetTime"].to_numpy() / 1000.0

# EV anchor：把 mental + random onset 合并，再按时间排序
ev_onsets = np.concatenate([read_ev(EV_MENTAL), read_ev(EV_RANDOM)])
ev_onsets = np.sort(ev_onsets)

a, b, resid, tab_used, ev_used, pred = fit_a_b(tab_onsets_sec, ev_onsets)

print("Fitted mapping: ev_time ≈ a * tab_time + b")
print("a =", a)
print("b =", b, "(seconds)")
print("resid: mean =", resid.mean(), "sec, max abs =", np.max(np.abs(resid)), "sec")

# 看看前几对点对齐情况
for i in range(min(10, len(tab_used))):
    print(i, "TAB", tab_used[i], "EV", ev_used[i], "pred", pred[i], "resid", resid[i])'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

# -------------------------
# 基础：读取 Sync / TAB / GIFTI
# -------------------------
def read_sync_seconds(sync_path: Path) -> float:
    txt = sync_path.read_text().strip()
    # Sync.txt 只有一个数字
    return float(txt)

def read_tab_tsv(tab_path: Path) -> pd.DataFrame:
    # HCP TAB.txt 通常是 TSV
    return pd.read_csv(tab_path, sep=r"\t", engine="python")

def get_T_from_func_gii(gii_path: Path) -> int:
    import nibabel as nib
    g = nib.load(str(gii_path))
    # HCP native func.gii 通常每个 darray = 1 个 timepoint
    return len(g.darrays)

def find_tab_file(run_dir: Path, task: str, enc: str) -> Path:
    """
    TAB.txt 在 tfMRI_{task}_{enc} 目录下（不是 EVs 里）
    根据规则：RL=run1, LR=run2 优先匹配，否则兜底选唯一 TAB。
    """
    run = 1 if enc.upper() == "RL" else 2

    # 常见候选：*run{run}*_TAB*.txt
    cands = sorted(run_dir.glob(f"*run{run}*TAB*.txt"))
    if not cands:
        # 宽松：任何包含 TAB 的 txt
        cands = sorted(run_dir.glob("*TAB*.txt"))
    if not cands:
        raise FileNotFoundError(f"No TAB.txt found under {run_dir}")

    task_low = task.lower()
    # 优先：同时包含 task 和 run 关键词
    for p in cands:
        name = p.name.lower()
        if task_low in name and f"run{run}" in name:
            return p
    # 其次：包含 task
    for p in cands:
        if task_low in p.name.lower():
            return p
    # 兜底：第一个
    return cands[0]

# -------------------------
# 事件与区间
# -------------------------
@dataclass
class Event:
    onset_sec: float
    phase: str
    cond: str | None = None

def _col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def _get_numeric_series(df: pd.DataFrame, col: str) -> np.ndarray:
    v = pd.to_numeric(df[col], errors="coerce").to_numpy()
    v = v[np.isfinite(v)]
    return v

def _ms_to_scan_sec(ms: np.ndarray, sync_sec: float) -> np.ndarray:
    # 核心对齐：TAB(ms)/1000 - Sync(sec)  -> 扫描时间轴
    return ms / 1000.0 - sync_sec

def _emit_events_from_onset_col(
    df: pd.DataFrame,
    onset_col: str,
    phase: str,
    sync_sec: float,
    cond_col: str | None = None,
    cond_map: dict | None = None
) -> list[Event]:
    if onset_col not in df.columns:
        return []
    ms = pd.to_numeric(df[onset_col], errors="coerce").to_numpy()
    m = np.isfinite(ms)
    if not m.any():
        return []
    onsets = _ms_to_scan_sec(ms[m], sync_sec)

    cond_vals = None
    if cond_col and cond_col in df.columns:
        raw = df.loc[m, cond_col].astype(str).to_numpy()
        if cond_map:
            raw = np.array([cond_map.get(x, x) for x in raw], dtype=object)
        cond_vals = raw

    evs = []
    if cond_vals is None:
        for t in onsets:
            evs.append(Event(float(t), phase, None))
    else:
        for t, c in zip(onsets, cond_vals):
            evs.append(Event(float(t), phase, str(c)))
    return evs

def _phase_tag(task: str, phase: str, cond: str | None) -> tuple[str, str, str]:
    # 二级标签：phase 可带 cond，例如 stim_mental / stim_random
    if cond is None or cond.strip() == "" or cond.lower() == "nan":
        ph = phase
    else:
        ph = f"{phase}_{cond}".lower()
    tag = f"{task.upper()}:{ph}"
    return task.upper(), ph, tag

# -------------------------
# 各任务：按 Appendix VI 的“关键变量”选择列
# -------------------------
def build_events(task: str, tab: pd.DataFrame, sync_sec: float) -> list[Event]:
    t = task.upper()
    evs: list[Event] = []

    # 通用：如果能识别 fixation 的 OnsetTime 列，就先收集
    # （不同任务 fixation 列名不同，这里只是兜底）
    for c in tab.columns:
        if c.lower().endswith(".onsettime") and "fix" in c.lower():
            # 不直接全收集，只作为兜底时可以用；下面每个任务会更精确
            pass

    if t == "SOCIAL":
        # Appendix VI: Type, MovieSlide.OnsetTime, CountDownSlide.OnsetTime, ResponseSlide...:contentReference[oaicite:5]{index=5}
        evs += _emit_events_from_onset_col(tab, "CountDownSlide.OnsetTime", "instruction", sync_sec)
        evs += _emit_events_from_onset_col(tab, "MovieSlide.OnsetTime", "stim", sync_sec, cond_col="Type")
        evs += _emit_events_from_onset_col(tab, "ResponseSlide.OnsetTime", "response", sync_sec)
        evs += _emit_events_from_onset_col(tab, "FixationBlock.OnsetTime", "fixation", sync_sec)

    elif t == "MOTOR":
        # Appendix VI: 各种 Cue.OnsetTime, Cross*.OnsetTime, SyncSlide/CountDownSlide:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
        evs += _emit_events_from_onset_col(tab, "SyncSlide.OnsetTime", "instruction", sync_sec)
        evs += _emit_events_from_onset_col(tab, "CountDownSlide.OnsetTime", "instruction", sync_sec)

        # block cue（可理解为 stim/cue）
        for cue in ["LeftHandCue.OnsetTime", "LeftFootCue.OnsetTime",
                    "RightHandCue.OnsetTime", "RightFootCue.OnsetTime",
                    "TongueCue.OnsetTime"]:
            evs += _emit_events_from_onset_col(tab, cue, "stim", sync_sec)

        # 单次动作（更细粒度也归为 stim）
        for cross in ["CrossLeft.OnsetTime", "CrossRight.OnsetTime", "CrossCenter.OnsetTime"]:
            evs += _emit_events_from_onset_col(tab, cross, "response", sync_sec)  # 这里把动作执行看成 response/执行期

        # 休息点（Fixdot）
        evs += _emit_events_from_onset_col(tab, "Fixdot.OnsetTime", "fixation", sync_sec)

    elif t == "WM":
        # Appendix VI: Stim.OnsetTime, Fix.OnsetTime, Fix15sec.OnsetTime, Cue2Back/CueTarget, BlockType/StimType:contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
        evs += _emit_events_from_onset_col(tab, "SyncSlide.OnsetTime", "instruction", sync_sec)
        evs += _emit_events_from_onset_col(tab, "Cue2Back.OnsetTime", "cue", sync_sec, cond_col="BlockType")
        evs += _emit_events_from_onset_col(tab, "CueTarget.OnsetTime", "cue", sync_sec, cond_col="BlockType")
        evs += _emit_events_from_onset_col(tab, "Stim.OnsetTime", "stim", sync_sec, cond_col="BlockType")
        evs += _emit_events_from_onset_col(tab, "Fix.OnsetTime", "fixation", sync_sec)
        evs += _emit_events_from_onset_col(tab, "Fix15sec.OnsetTime", "fixation", sync_sec)
        evs += _emit_events_from_onset_col(tab, "FeelFreeToRest.OnsetTime", "other", sync_sec)

    elif t == "LANGUAGE":
        # Appendix VI: GetReady.FinishTime(同步)、PresentStoryFile/PresentMathFile、ResponsePeriod等:contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
        evs += _emit_events_from_onset_col(tab, "SyncSlide.OnsetTime", "instruction", sync_sec)
        evs += _emit_events_from_onset_col(tab, "GetReady.FinishTime", "instruction", sync_sec)

        # story/math 刺激
        evs += _emit_events_from_onset_col(tab, "PresentStoryFile.OnsetTime", "stim", sync_sec, cond_col="StimType")
        evs += _emit_events_from_onset_col(tab, "PresentMathFile.OnsetTime", "stim", sync_sec, cond_col="StimType")

        # 提问/选项与反应
        for c in ["ThatWasAbout.OnsetTime", "PresentMathOptions.OnsetTime", "ResponsePeriod.OnsetTime"]:
            evs += _emit_events_from_onset_col(tab, c, "response", sync_sec, cond_col="StimType")

        # fixation（有的 language 有 Fix.OnsetTime / Fix15sec.OnsetTime）
        evs += _emit_events_from_onset_col(tab, "Fix.OnsetTime", "fixation", sync_sec)
        evs += _emit_events_from_onset_col(tab, "Fix15sec.OnsetTime", "fixation", sync_sec)
        evs += _emit_events_from_onset_col(tab, "FeelFreeToRest.OnsetTime", "other", sync_sec)

    elif t == "RELATIONAL":
        # Appendix VI: RelationalSlide.OnsetTime, ControlSlide.OnsetTime, Prompts, SyncSlide:contentReference[oaicite:12]{index=12}
        evs += _emit_events_from_onset_col(tab, "SyncSlide.OnsetTime", "instruction", sync_sec)
        evs += _emit_events_from_onset_col(tab, "RelationalPrompt.OnsetTime", "cue", sync_sec)
        evs += _emit_events_from_onset_col(tab, "ControlPrompt.OnsetTime", "cue", sync_sec)
        evs += _emit_events_from_onset_col(tab, "RelationalSlide.OnsetTime", "stim", sync_sec, cond_col="Procedure")
        evs += _emit_events_from_onset_col(tab, "ControlSlide.OnsetTime", "stim", sync_sec, cond_col="Procedure")

    elif t == "GAMBLING":
        # Appendix VI: TrialType, QuestionMark.OnsetTime, SyncSlide.OnsetTime:contentReference[oaicite:13]{index=13}
        evs += _emit_events_from_onset_col(tab, "SyncSlide.OnsetTime", "instruction", sync_sec)
        evs += _emit_events_from_onset_col(tab, "QuestionMark.OnsetTime", "stim", sync_sec, cond_col="TrialType")

        # fixation block 常可由 Procedure[Trial] 识别为 FixationBlockPROC:contentReference[oaicite:14]{index=14}
        if "Procedure[Trial]" in tab.columns and "FixationBlock.OnsetTime" in tab.columns:
            # 如果存在更明确的 fixation onset，就用它
            evs += _emit_events_from_onset_col(tab, "FixationBlock.OnsetTime", "fixation", sync_sec)
        else:
            # 兜底：找可能的 Fixation onset 列
            for c in tab.columns:
                if c.lower().endswith(".onsettime") and "fixation" in c.lower():
                    evs += _emit_events_from_onset_col(tab, c, "fixation", sync_sec)

    elif t == "EMOTION":
        # Appendix VI: Procedure, StimSlide.OnsetTime, SyncSlide.OnsetTime:contentReference[oaicite:15]{index=15}
        evs += _emit_events_from_onset_col(tab, "SyncSlide.OnsetTime", "instruction", sync_sec)
        # StimSlide.OnsetTime 有时带 [Block]
        if "StimSlide.OnsetTime[Block]" in tab.columns:
            evs += _emit_events_from_onset_col(tab, "StimSlide.OnsetTime[Block]", "stim", sync_sec, cond_col="Procedure")
        evs += _emit_events_from_onset_col(tab, "StimSlide.OnsetTime", "stim", sync_sec, cond_col="Procedure")

        # prompts 可视为 cue
        for p in ["ShapesPrompt.OnsetTime", "FacePromt.OnsetTime", "FacePrompt.OnsetTime"]:
            evs += _emit_events_from_onset_col(tab, p, "cue", sync_sec)

    else:
        raise NotImplementedError(f"Task '{task}' not supported yet.")

    # 清理：只保留合理时间（>=0），并去重（同一 phase+cond 同一时间可能重复）
    out = []
    seen = set()
    for e in evs:
        if e.onset_sec < 0:
            continue
        key = (round(e.onset_sec, 6), e.phase, e.cond or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(e)

    out.sort(key=lambda x: x.onset_sec)
    return out

def events_to_intervals(events: list[Event], scan_end_sec: float) -> list[tuple[float, float, str, str | None]]:
    """
    简化策略：按事件 onset 排序，区间 = [onset_i, onset_{i+1})
    最后一个事件延伸到 scan_end_sec
    """
    if not events:
        return []

    intervals = []
    for i, e in enumerate(events):
        s = e.onset_sec
        e_next = events[i + 1].onset_sec if i + 1 < len(events) else scan_end_sec
        if e_next <= s:
            continue
        intervals.append((float(s), float(min(e_next, scan_end_sec)), e.phase, e.cond))
    return intervals

def label_timepoints(
    task: str,
    intervals: list[tuple[float, float, str, str | None]],
    TR: float,
    T: int,
) -> pd.DataFrame:
    t = np.round((np.arange(T)) * TR, 2)
    phase = np.array(["other"] * T, dtype=object)
    tag = np.array([f"{task.upper()}:other"] * T, dtype=object)

    for s, e, ph, cond in intervals:
        mask = (t >= s) & (t < e)
        _, ph2, tag2 = _phase_tag(task, ph, cond)
        phase[mask] = ph2
        tag[mask] = tag2

    df = pd.DataFrame({
        "frame": np.arange(T, dtype=int),
        "t_sec": t,
        "task": [task.upper()] * T,
        "phase": phase,
        "tag": tag
    })
    return df

# -------------------------
# 主流程：遍历目录
# -------------------------
def process_one_run(run_dir: Path, task: str, sub: str, enc: str, TR: float,
                    out_root: Path, out_name: str) -> Path:
    """
    run_dir: .../MNINonLinear/Results/tfMRI_{task}_{enc}
    TAB.txt: 位于 run_dir 下
    Sync.txt: 位于 run_dir/EVs/Sync.txt
    输出：out_root/{task}/{sub}/tfMRI_{task}_{enc}/{out_name}
    """
    evs_dir = run_dir / "EVs"
    sync_path = evs_dir / "Sync.txt"
    if not sync_path.exists():
        raise FileNotFoundError(f"Missing Sync.txt: {sync_path}")

    sync_sec = read_sync_seconds(sync_path)

    # TAB 在 run_dir 下
    tab_path = find_tab_file(run_dir, task=task, enc=enc)
    tab = read_tab_tsv(tab_path)

    # L.native func gii
    gii = run_dir / f"tfMRI_{task}_{enc}.L.native.func.gii"
    if not gii.exists():
        cands = sorted(run_dir.glob("*.L.native.func.gii"))
        if not cands:
            raise FileNotFoundError(f"Cannot find L.native.func.gii under {run_dir}")
        gii = cands[0]

    T = get_T_from_func_gii(gii)
    scan_end = T * TR

    events = build_events(task, tab, sync_sec=sync_sec)
    intervals = events_to_intervals(events, scan_end_sec=scan_end)
    labels_df = label_timepoints(task, intervals, TR=TR, T=T)

    # 输出到用户指定目录，并保持层级
    out_dir = out_root / task / sub / f"tfMRI_{task}_{enc}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    labels_df.to_csv(out_path, sep="\t", index=False)

    # 也建议保存一个简短 meta，方便追溯（可选）
    meta_path = out_dir / (Path(out_name).stem + "_meta.txt")
    meta_path.write_text(
        f"task={task}\nsub={sub}\nenc={enc}\nTR={TR}\nT={T}\n"
        f"sync_sec={sync_sec}\n"
        f"tab={tab_path}\nfunc={gii}\n",
        encoding="utf-8"
    )

    return out_path


def main():
    ROOT = Path("/home/wmy/Documents")
    OUT_ROOT = Path("/mnt2/wmy/geometry/")
    TASKS = ["SOCIAL", "MOTOR", "WM", "LANGUAGE", "EMOTION", "GAMBLING", "RELATIONAL"]
    ENCS = ["LR", "RL"]                   
    TR = 0.72                              
    OUT_NAME = "L.native_time_tags.tsv"
    SUBJECTS = []  # 如果只想处理特定 subject，可以在这里列出，例如 ["100610", "100307"]

    # =========================
    # 批处理
    # =========================
    n_ok, n_skip = 0, 0

    for task in TASKS:
        task_dir = ROOT / task
        if not task_dir.exists():
            print(f"[SKIP] task folder not found: {task_dir}")
            n_skip += 1
            continue

        subs = SUBJECTS if SUBJECTS else sorted([p.name for p in task_dir.iterdir() if p.is_dir()])

        for sub in subs:
            for enc in ENCS:
                run_dir = task_dir / sub / "MNINonLinear" / "Results" / f"tfMRI_{task}_{enc}"
                if not run_dir.exists():
                    print(f"[SKIP] missing run dir: {run_dir}")
                    n_skip += 1
                    continue

                try:
                    out_path = process_one_run(
                        run_dir=run_dir, task=task, sub=sub, enc=enc, TR=TR,
                        out_root=OUT_ROOT, out_name=OUT_NAME
                    )
                    print(f"[OK] {task}/{sub} {enc} -> {out_path}")
                    n_ok += 1
                except Exception as e:
                    print(f"[FAIL] {task}/{sub} {enc}: {e}")
                    n_skip += 1

    print(f"\nDone. OK={n_ok}, SKIP/FAIL={n_skip}")

if __name__ == "__main__":
    main()
