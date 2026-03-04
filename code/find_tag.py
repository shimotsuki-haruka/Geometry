#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from collections import defaultdict

# =============================
# 修改成你的根目录
# =============================
BASE_GEOM = "/home/wmy/Documents"
TASKS = ["SOCIAL","MOTOR","WM","RELATIONAL","EMOTION","LANGUAGE","GAMBLING"]


def is_time_column(col):
    c = col.lower()
    return "onsettime" in c or "finishtime" in c


def find_tab_files(task_root):
    tabs = []
    for root, _, files in os.walk(task_root):
        for f in files:
            if f.lower().endswith("tab.txt"):
                tabs.append(os.path.join(root, f))
    return tabs


def detect_stim_onset_columns(df):
    """
    自动检测可能是 stim 的 OnsetTime 列
    规则：
      - 包含 Stim
      - 或包含 Movie
      - 或包含 Trial
    """
    cols = []
    for c in df.columns:
        cl = c.lower()
        if "onsettime" in cl:
            if ("stim" in cl) or ("movie" in cl) or ("trial" in cl):
                cols.append(c)
    return cols


def analyze_task(task_name):

    print("\n" + "="*100)
    print(f"🔎 TASK: {task_name}")
    print("="*100)

    task_root = os.path.join(BASE_GEOM, task_name)
    tabs = find_tab_files(task_root)

    if not tabs:
        print("No TAB files found.")
        return

    dfs = []
    for tab in tabs:
        try:
            df = pd.read_csv(tab, sep="\t")
            dfs.append(df)
        except:
            continue

    df = pd.concat(dfs, ignore_index=True)

    print("\n📌 Columns containing 'type' (possible condition columns):\n")

    for col in df.columns:
        if "type" not in col.lower():
            continue

        series = df[col].dropna()

        if len(series) == 0:
            continue

        n_unique = series.nunique()

        if n_unique < 2:
            continue

        if n_unique > 20:
            continue

        # 排除纯数字
        try:
            pd.to_numeric(series)
            continue
        except:
            pass

        value_counts = series.value_counts()

        print("-"*80)
        print(f"Column: {col}")
        print(f"Unique values: {n_unique}")
        print("Top subclasses:")
        print(value_counts.head(10))

def main():
    for task in TASKS:
        analyze_task(task)

    print("\nDone.")


if __name__ == "__main__":
    main()
