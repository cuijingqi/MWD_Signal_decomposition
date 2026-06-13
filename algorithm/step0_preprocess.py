# ══════════════════════════════════════════════════════════════
# step0_preprocess.py
# 前处理：读取稳定段 → 每孔取最长段 → 两两截齐配对
# 不做归一化，保留原始物理幅值（量纲一致，幅值含岩性信息）
# ══════════════════════════════════════════════════════════════

import os
import re
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict

from config import (
    INPUT_MODE, INPUT_FOLDER, INPUT_EXCEL, SIGNAL_TYPES,
    SAVE_INTERMEDIATE, OUTPUT_STEP0,
    make_label, make_dir, get_pair_meta,
    MIXED_MAX_SEG_LEN,
)


# ──────────────────────────────────────────────────────────────
# 文件解析
# ──────────────────────────────────────────────────────────────

def parse_filename(filename):
    """
    解析文件名，提取孔号和信号类型。
    期望格式：{前缀}_{孔号}{信号类型}[_segments].xlsx
    孔号支持纯数字（如 1）或带连字符（如 1-1）。
    返回 (hole_id, signal_type) 或 None，其中 hole_id 含前缀（如 2TRD_1）以避免跨系列重名。
    """
    basename = os.path.splitext(filename)[0]
    pattern = r"(.+)_([0-9]+(?:-[0-9]+)?)(" + "|".join(SIGNAL_TYPES) + r")(?:_segments)?$"
    m = re.match(pattern, basename, re.IGNORECASE)
    if not m:
        return None
    hole_id = f"{m.group(1)}_{m.group(2)}"
    return (hole_id, m.group(3).lower())


def read_stable_sheets(filepath):
    """
    读取 xlsx 中所有 stable* sheet。
    每个 sheet 两列：[x（深度/序号）, y（信号值）]。
    返回 list of (x_array, y_array)，各段独立，不拼接。
    """
    xl = pd.ExcelFile(filepath)
    sheets = [s for s in xl.sheet_names if s.lower().startswith("stable")]
    if not sheets:
        print(f"    ⚠  未找到 stable sheet：{filepath}")
        return []

    result = []
    for sheet in sheets:
        df = pd.read_excel(filepath, sheet_name=sheet).dropna()
        if df.shape[1] < 2 or len(df) == 0:
            continue
        x = df.iloc[:, 0].values.astype(float)
        y = df.iloc[:, 1].values.astype(float)
        result.append((x, y))
    return result


def parse_sheet_name(sheet_name):
    """
    解析单文件模式下的 sheet 名，提取孔号和信号类型。
    期望格式：{孔号}{信号类型}，如 "1pressure"、"6torque"、"13pressure"。
    孔号支持纯数字或带连字符（如 1-1）。
    返回 (hole_id, signal_type) 或 None。
    """
    pattern = r"^([0-9]+(?:-[0-9]+)?)(" + "|".join(SIGNAL_TYPES) + r")$"
    m = re.match(pattern, sheet_name.strip(), re.IGNORECASE)
    return (m.group(1), m.group(2).lower()) if m else None


def read_from_folder_all_segments(folder):
    """
    混合模式：扫描文件夹内所有 *_segments.xlsx，保留每个文件每个稳定段，
    孔号命名为 {原孔号}-{段序号}（如 1-1、1-2），全部参与两两配对。
    返回与其他模式相同结构的 hole_data。
    """
    all_files = [f for f in os.listdir(folder) if f.endswith(".xlsx")]
    if not all_files:
        raise FileNotFoundError(f"在 {folder} 中未找到 .xlsx 文件")

    print(f"\n[混合模式扫描]  路径: {folder}")
    print(f"  找到 {len(all_files)} 个 *_segments.xlsx 文件")
    print(f"  策略: 每孔每个稳定段独立保留，孔号命名为 {{原孔号}}-{{段序号}}")

    file_groups = defaultdict(dict)
    for fname in sorted(all_files):
        parsed = parse_filename(fname)
        if parsed is None:
            print(f"  ⚠ 无法解析文件名: {fname}，跳过")
            continue
        hole_id, sig_type = parsed
        file_groups[sig_type][hole_id] = os.path.join(folder, fname)
        print(f"  解析成功: {fname}  →  信号类型={sig_type}  孔号={hole_id}")

    hole_data = defaultdict(dict)

    subdivide = (MIXED_MAX_SEG_LEN is not None and MIXED_MAX_SEG_LEN > 0)
    if subdivide:
        print(f"  细分开关: ON  — 超过 {MIXED_MAX_SEG_LEN} 点的稳定段将均分为子段，孔内重排序")
    else:
        print(f"  细分开关: OFF — 保留原始稳定段长度")

    for sig_type, hole_files in file_groups.items():
        print(f"\n[读取所有稳定段]  信号类型: {sig_type}  共{len(hole_files)}个孔")
        for hole_id, filepath in sorted(hole_files.items()):
            segments = read_stable_sheets(filepath)
            if not segments:
                print(f"  孔{hole_id}: ⚠ 未读到任何稳定段，跳过")
                continue

            raw_lengths = [len(seg[1]) for seg in segments]
            print(f"  孔{hole_id}: 共{len(segments)}段  各段原始长度={raw_lengths}点")

            # ── 细分长段 ───────────────────────────────────────
            all_sub_segs = []
            for seg_i, (x_seg, y_seg) in enumerate(segments, start=1):
                n = len(y_seg)
                if subdivide and n > MIXED_MAX_SEG_LEN:
                    n_chunks = n // MIXED_MAX_SEG_LEN
                    tail = n - n_chunks * MIXED_MAX_SEG_LEN
                    total_subs = n_chunks + (1 if tail > 0 else 0)
                    tail_note = f" + 末尾{tail}点" if tail > 0 else ""
                    print(f"    原始段{seg_i}（{n}点）→ 细分为{total_subs}个子段"
                          f"（{n_chunks}×{MIXED_MAX_SEG_LEN}点{tail_note}）")
                    for ci in range(n_chunks):
                        s = ci * MIXED_MAX_SEG_LEN
                        all_sub_segs.append(
                            (x_seg[s:s + MIXED_MAX_SEG_LEN],
                             y_seg[s:s + MIXED_MAX_SEG_LEN],
                             seg_i)
                        )
                    if tail > 0:
                        s = n_chunks * MIXED_MAX_SEG_LEN
                        all_sub_segs.append((x_seg[s:], y_seg[s:], seg_i))
                else:
                    all_sub_segs.append((x_seg, y_seg, seg_i))

            # ── 重排序：孔内子段统一编号 -1, -2, ... ──────────
            sub_lengths = [len(t[1]) for t in all_sub_segs]
            print(f"    → 孔内共{len(all_sub_segs)}个子段，长度={sub_lengths}，重排序编号")

            for seg_idx, (x_sub, y_sub, _src) in enumerate(all_sub_segs, start=1):
                new_hole_id = f"{hole_id}-{seg_idx}"
                mean_v, std_v = np.mean(y_sub), np.std(y_sub)
                print(f"    子段{seg_idx} → 孔号={new_hole_id}  N={len(y_sub)}"
                      f"  均值={mean_v:.4f}  std={std_v:.4f}"
                      f"  范围=[{y_sub.min():.4f}, {y_sub.max():.4f}]")
                hole_data[sig_type][new_hole_id] = {
                    "y": y_sub,
                    "x": x_sub,
                    "n_segments": len(all_sub_segs),
                    "seg_lengths": sub_lengths,
                }

    return hole_data


def read_from_excel(filepath):
    """
    单文件模式：读取一个 xlsx，每个 sheet 为一个人工选定的稳定段。
    sheet 名格式：{孔号}{信号类型}，如 "1pressure"。
    返回与文件夹模式相同结构的 hole_data（每孔只有1段，直接作为最终段）。
    """
    xl = pd.ExcelFile(filepath)
    hole_data = defaultdict(dict)

    print(f"\n[读取单文件稳定段]  路径: {filepath}")
    print(f"  共 {len(xl.sheet_names)} 个 sheet")

    for sheet in xl.sheet_names:
        parsed = parse_sheet_name(sheet)
        if parsed is None:
            print(f"  ⚠ 无法解析 sheet 名: '{sheet}'，跳过"
                  f"（期望格式：孔号+信号类型，如 '1pressure'）")
            continue
        hole_id, sig_type = parsed

        df = pd.read_excel(filepath, sheet_name=sheet).dropna()
        if df.shape[1] < 2 or len(df) == 0:
            print(f"  ⚠ sheet '{sheet}' 数据不足，跳过")
            continue

        x = df.iloc[:, 0].values.astype(float)
        y = df.iloc[:, 1].values.astype(float)

        mean_v, std_v = np.mean(y), np.std(y)
        print(f"  sheet '{sheet}'  →  孔={hole_id}  信号={sig_type}"
              f"  N={len(y)}  均值={mean_v:.4f}  std={std_v:.4f}"
              f"  范围=[{y.min():.4f}, {y.max():.4f}]")

        hole_data[sig_type][hole_id] = {
            "y": y,
            "x": x,
            "n_segments": 1,
            "seg_lengths": [len(y)],
        }

    return hole_data


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def run_preprocess():
    """
    前处理主流程。
    返回 hole_data, pair_data。
    """
    print("\n" + "=" * 60)
    print("  Step0: 前处理 — 读取稳定段 → 取最长段 → 两两截齐配对")
    print("  注：保留原始物理幅值，不做任何归一化")
    print(f"  输入模式: {INPUT_MODE}")
    print("=" * 60)

    # ── 按模式读取数据 ─────────────────────────────────────────
    if INPUT_MODE == "excel":
        if not os.path.isfile(INPUT_EXCEL):
            raise FileNotFoundError(f"单文件模式：找不到 {INPUT_EXCEL}")
        hole_data = read_from_excel(INPUT_EXCEL)
        if not hole_data:
            raise ValueError("单文件模式：未能从 Excel 中解析任何有效数据")

    elif INPUT_MODE == "mixed":
        if not os.path.isdir(INPUT_FOLDER):
            raise FileNotFoundError(f"混合模式：找不到文件夹 {INPUT_FOLDER}")
        hole_data = read_from_folder_all_segments(INPUT_FOLDER)
        if not hole_data:
            raise ValueError("混合模式：未能从文件夹中解析任何有效数据")

    else:  # "folder"
        all_files = [f for f in os.listdir(INPUT_FOLDER)
                     if f.endswith(".xlsx")]
        if not all_files:
            raise FileNotFoundError(
                f"在 {INPUT_FOLDER} 中未找到 .xlsx 文件")

        print(f"\n[文件扫描]  路径: {INPUT_FOLDER}")
        print(f"  找到 {len(all_files)} 个 *_segments.xlsx 文件")

        file_groups = defaultdict(dict)
        for fname in sorted(all_files):
            parsed = parse_filename(fname)
            if parsed is None:
                print(f"  ⚠ 无法解析文件名: {fname}，跳过")
                continue
            hole_id, sig_type = parsed
            file_groups[sig_type][hole_id] = os.path.join(INPUT_FOLDER, fname)
            print(f"  解析成功: {fname}  →  信号类型={sig_type}  孔号={hole_id}")

        if not file_groups:
            raise ValueError("未能解析任何文件，请检查命名格式")

        # ── 各信号类型孔集合一致性校验 ──────────────────────────
        hole_sets = {sig: set(hfiles.keys()) for sig, hfiles in file_groups.items()}
        unique_sets = {frozenset(s) for s in hole_sets.values()}
        if len(unique_sets) > 1:
            print(f"\n  ⚠ [警告] 各信号类型的孔集合不一致，跨类型对比结果需谨慎：")
            for sig, hs in hole_sets.items():
                print(f"    {sig}: 孔{sorted(hs)}")
            all_holes = set().union(*hole_sets.values())
            for sig, hs in hole_sets.items():
                missing = all_holes - hs
                if missing:
                    print(f"    → {sig} 缺少孔{sorted(missing)}的数据文件")

        # ── 读取各孔最长稳定段 ─────────────────────────────────
        hole_data = defaultdict(dict)

        for sig_type, hole_files in file_groups.items():
            print(f"\n[读取稳定段]  信号类型: {sig_type}  共{len(hole_files)}个孔")
            print(f"  策略: 每孔读取所有标注稳定段，取最长一段用于后续分析")

            for hole_id, filepath in sorted(hole_files.items()):
                segments = read_stable_sheets(filepath)
                if not segments:
                    print(f"  孔{hole_id}: ⚠ 未读到任何稳定段，跳过")
                    continue

                lengths = [len(seg[1]) for seg in segments]
                longest_idx = int(np.argmax(lengths))
                x_best, y_best = segments[longest_idx]

                mean_v = np.mean(y_best)
                std_v = np.std(y_best)
                print(f"  孔{hole_id}: 稳定段共{len(segments)}段  "
                      f"各段长度={lengths}点")
                print(f"         → 取第{longest_idx + 1}段（最长，{lengths[longest_idx]}点）"
                      f"  均值={mean_v:.4f}  std={std_v:.4f}"
                      f"  范围=[{y_best.min():.4f}, {y_best.max():.4f}]")
                print(f"           注：幅值保留原始物理量，std反映信号变异程度")

                hole_data[sig_type][hole_id] = {
                    "y": y_best,
                    "x": x_best,
                    "n_segments": len(segments),
                    "seg_lengths": lengths,
                }

    # ── 两两配对截齐 ──────────────────────────────────────────
    pair_data = defaultdict(dict)

    print(f"\n[两两配对截齐]")
    print(f"  策略: 取两孔中较短的长度截齐，确保时间轴对齐后才能做逐点比较")

    for sig_type, holes in hole_data.items():
        hole_ids = sorted(holes.keys())
        if len(hole_ids) < 2:
            print(f"  ⚠ {sig_type} 不足2个孔，跳过配对")
            continue

        n_pairs = len(list(itertools.combinations(hole_ids, 2)))
        print(f"\n  信号类型: {sig_type}  {len(hole_ids)}个孔 → {n_pairs}对")
        for hole_A, hole_B in itertools.combinations(hole_ids, 2):
            y_A = holes[hole_A]["y"]
            y_B = holes[hole_B]["y"]

            n = min(len(y_A), len(y_B))
            y_A = y_A[:n]
            y_B = y_B[:n]

            label_A = make_label(hole_A, sig_type)
            label_B = make_label(hole_B, sig_type)
            meta = get_pair_meta(hole_A, hole_B)
            tag = meta["tag"]

            # 截断量
            dropped_A = len(holes[hole_A]["y"]) - n
            dropped_B = len(holes[hole_B]["y"]) - n
            corr = float(np.corrcoef(y_A, y_B)[0, 1])

            max_len = max(len(holes[hole_A]["y"]), len(holes[hole_B]["y"]))
            trunc_ratio = 1.0 - n / max_len if max_len > 0 else 0.0

            print(f"  {label_A} vs {label_B}  [{tag}]")
            print(f"    原始长度: A={len(holes[hole_A]['y'])}  "
                  f"B={len(holes[hole_B]['y'])}  → 截齐后={n}点"
                  f"（A截去{dropped_A}点，B截去{dropped_B}点）")
            if trunc_ratio > 0.3:
                print(f"    ⚠ 截齐损失{trunc_ratio*100:.0f}%数据，建议检查两孔稳定段长度差异是否合理")
            print(f"    截齐后Pearson相关={corr:.4f}"
                  f"  {'↑相关较强，两孔响应相似' if abs(corr) > 0.6 else '↓相关较弱，两孔响应差异明显'}")

            pair_data[sig_type][(hole_A, hole_B)] = {
                "y_A": y_A,
                "y_B": y_B,
                "length": n,
                "label_A": label_A,
                "label_B": label_B,
                "meta": meta,
            }

    # ── 保存数据 ───────────────────────────────────────────────
    if SAVE_INTERMEDIATE:
        print(f"\n[保存中间数据]  → {OUTPUT_STEP0}/")
        for sig_type, pairs in pair_data.items():
            out_dir = make_dir(OUTPUT_STEP0, sig_type)
            for (hA, hB), d in pairs.items():
                path = os.path.join(
                    out_dir,
                    f"{d['label_A']}_vs_{d['label_B']}_segment.npz")
                np.savez(path,
                         y_A=d["y_A"], y_B=d["y_B"],
                         label_A=np.array([d["label_A"]]),
                         label_B=np.array([d["label_B"]]),
                         n_samples=np.array([d["length"]]))
                print(f"  保存: {os.path.basename(path)}")

    fig_all_holes(hole_data)
    fig_pair_overlay(pair_data)
    print(f"\n[Step0完成]  hole_data: {list(hole_data.keys())}  "
          f"pair_data: { {k: len(v) for k, v in pair_data.items()} }")
    return hole_data, pair_data


# ══════════════════════════════════════════════════════════════

