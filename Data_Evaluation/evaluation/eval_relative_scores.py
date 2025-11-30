#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1 + Rank Correlations (Spearman's rho, Kendall's tau-b) evaluator.

- 仍然使用 mapping.json 将 h5 的 test_keys 映射到预测 json 的键名
- 相关性使用 h5 的 gtscore；若 gtscore 为下采样，先块状上采样到与“预测序列”相同长度再计算
- 需要你仓库里的:
    from evaluator import evaluate_summaries
    from model.heuristic_prediction import heuristic_predicator
"""

import argparse
import json
import os
import sys
import numpy as np
import h5py
import time

# 自动添加 src 路径，确保可以 import model.*
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
SRC_PATH = os.path.join(ROOT_PATH, 'src')
sys.path.append(SRC_PATH)

from evaluator import evaluate_summaries
from model.heuristic_prediction import heuristic_predicator

# =========================
# SciPy 可选加速
# =========================
_HAS_SCIPY = False
try:
    from scipy.stats import spearmanr, kendalltau
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# =========================
# 工具函数（无需外部依赖）
# =========================
def block_upsample_like_pred(gtscore, target_len: int) -> np.ndarray:
    """
    按“整段复制”的方式把 gtscore 扩展到 target_len（和你发我的上采样逻辑一致）:
      设 len(gtscore)=M, 目标长度 N，因子 f = N/M
      第 i 段复制到 [floor(i*f), floor((i+1)*f))
    """
    g = np.asarray(gtscore, dtype=float).ravel()
    N = int(target_len)
    M = g.size
    if N <= 0:
        return np.zeros(0, dtype=float)
    if M == 0:
        return np.zeros(N, dtype=float)
    if M == N:
        return g.copy()

    f = N / M
    up = np.zeros(N, dtype=float)
    end_last = 0
    for i in range(M):
        start = int(i * f)
        end = min(int((i + 1) * f), N)
        if end <= start:
            end = min(start + 1, N)  # 极端浮点边界，至少占 1
        up[start:end] = g[i]
        end_last = end
    if end_last < N:  # 末尾如果没填满，补最后一个值
        up[end_last:] = g[-1]
    return up


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """
    纯 numpy 的“平均秩”实现（ties 用平均秩），用于 Spearman 备用。
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        # 找 tie 段
        while j + 1 < n and x[order[j + 1]] == x[order[j]]:
            j += 1
        # 平均秩：从 i 到 j 的位置（1-based）
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def _spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    sx = rx.std()
    sy = ry.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _kendall_tau_b_np(x: np.ndarray, y: np.ndarray, max_n: int = 5000) -> float:
    """
    朴素 Kendall τ-b（O(n^2)）；为避免超长数组算时过久，长度>max_n时均匀抽样到 max_n。
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if n < 2:
        return np.nan
    if n > max_n:
        idx = np.linspace(0, n - 1, num=max_n).astype(int)
        x = x[idx]
        y = y[idx]
        n = max_n

    conc = 0
    disc = 0
    tie_x = 0
    tie_y = 0
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        sx = np.sign(dx)
        sy = np.sign(dy)
        prod = sx * sy
        conc += np.sum(prod > 0)
        disc += np.sum(prod < 0)
        tie_x += np.sum(sx == 0)
        tie_y += np.sum(sy == 0)
    n_pairs = n * (n - 1) / 2.0
    denom = np.sqrt((n_pairs - tie_x) * (n_pairs - tie_y))
    if denom == 0:
        return np.nan
    return float((conc - disc) / denom)


def _fetch_gtscore(hdf: h5py.File, vid_key: str) -> np.ndarray:
    name = f"{vid_key}/gtscore"
    if name not in hdf.keys():
        raise KeyError(f"'{name}' not found in h5")
    return np.array(hdf[name], dtype=float).ravel()


def evaluate_rank_correlations_aligned(frames_score_file: str,
                                       gt_file: str,
                                       keys,
                                       mapping: dict):
    """
    用 gtscore 作为 GT；若抽帧则把 gtscore 块状上采样到“预测向量”的长度，再计算 Spearman ρ / Kendall τ-b。
    - frames_score_file: heuristic_predicator 产出的 eval.json（video_key -> 帧级预测分数列表）
    - gt_file: h5
    - keys: split['test_keys']（h5 内视频键名）
    - mapping: h5 键 -> 预测 json 的键（即你保存 eval.json 的键）
    """
    with open(frames_score_file, 'r') as f:
        pred_data = json.load(f)
    hdf = h5py.File(gt_file, 'r')

    rhos, taus = [], []
    per_video = {}

    for vid_key in list(keys):
        pred_key = mapping[vid_key]
        if pred_key not in pred_data:
            print(f"[WARN] Missing prediction for '{pred_key}' (mapped from '{vid_key}')")
            continue

        pred = np.array(pred_data[pred_key], dtype=float).ravel()
        T_pred = pred.size
        if T_pred < 2 or np.allclose(pred.std(), 0.0):
            print(f"[WARN] Degenerate prediction for '{pred_key}', skip correlation")
            continue

        try:
            gts = _fetch_gtscore(hdf, vid_key)   # 可能是下采样长度
        except Exception as e:
            print(f"[WARN] GT not found for '{vid_key}': {e}")
            continue

        # 与你的处理一致：如果长度不同，就把 gtscore 上采样到与预测相同的长度
        gts_up = block_upsample_like_pred(gts, T_pred)

        # 清洗 NaN/inf
        x = pred
        y = gts_up
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            rho = tau = np.nan
        else:
            if _HAS_SCIPY:
                try:
                    rho = float(spearmanr(x, y).correlation)
                except Exception:
                    rho = _spearman_np(x, y)
                try:
                    tau = float(kendalltau(x, y, variant='b').correlation)
                except Exception:
                    tau = _kendall_tau_b_np(x, y)
            else:
                rho = _spearman_np(x, y)
                tau = _kendall_tau_b_np(x, y)

        per_video[vid_key] = {
            "spearman_rho": None if rho != rho else rho,
            "kendall_tau": None if tau != tau else tau,
            "T_pred": int(T_pred),
            "T_gts": int(gts.size)
        }
        if rho == rho:
            rhos.append(rho)
        if tau == tau:
            taus.append(tau)

    hdf.close()
    mean_rho = float(np.mean(rhos)) if rhos else float('nan')
    mean_tau = float(np.mean(taus)) if taus else float('nan')
    return mean_rho, mean_tau, per_video


# =========================
# 主流程
# =========================
def run(args):
    work_dir = args.work_dir
    gt_file = args.gt_file
    splits_file = args.splits_file
    mapping_file = args.mapping_file
    meta_data_dir = args.meta_data_dir
    metric = args.metric
    norm = args.norm

    print("\n" + "=" * 60)
    print("DEBUG: Input Arguments")
    print(f"work_dir      : {work_dir}")
    print(f"gt_file       : {gt_file}")
    print(f"splits_file   : {splits_file}")
    print(f"mapping_file  : {mapping_file}")
    print(f"meta_data_dir : {meta_data_dir}")
    print(f"metric        : {metric}")
    print(f"norm          : {norm}")
    print("=" * 60 + "\n")

    # 读取 splits / mapping
    with open(splits_file, 'r') as jf:
        splits = json.load(jf)
    with open(mapping_file, 'r') as jf:
        mapping = json.load(jf)

    # 以 PredMetaData_1 作为“视频名清单”，随后对 1..4 四个目录读取对应文件
    meta_root = os.path.join(meta_data_dir, 'PredMetaData_1')
    meta_files = [os.path.splitext(f)[0] for f in os.listdir(meta_root) if f.endswith('.json')]
    print(f"[INFO] Found {len(meta_files)} meta files under PredMetaData_1")

    # 载入 4 份元数据（你的 heuristic_predicator 需要）
    data = {}
    missing_files = []
    for vn in meta_files:
        data[vn] = {}
        for i in range(1, 5):
            fp = os.path.join(meta_data_dir, f'PredMetaData_{i}', f'{vn}.json')
            try:
                with open(fp, 'r') as f:
                    content = json.load(f)
                # 基本字段检查
                for field in ('n_frames', 'scene_scores', 'scene_frames'):
                    if field not in content:
                        raise ValueError(f"Missing field '{field}'")
                data[vn][i] = content
            except Exception as e:
                print(f"[WARN] load fail: {fp} -> {e}")
                missing_files.append(fp)
    if missing_files:
        print(f"[WARN] Missing/invalid files: {len(missing_files)} (show 3)")
        for x in missing_files[:3]:
            print("  -", x)

    f1_splits = []
    rho_splits = []
    tau_splits = []

    for i, split in enumerate(splits):
        print(f"\n{'=' * 24} Split {i + 1} {'=' * 24}")
        test_keys = split['test_keys']             # h5 名称
        test_videos = [mapping[k] for k in test_keys]  # 预测 json 键名

        # 检查数据是否齐全
        miss = [v for v in test_videos if v not in data]
        if miss:
            print(f"[ERROR] Missing test videos (meta): {miss[:3]} ...")
            continue

        test_data = {k: data[k] for k in test_videos}
        output_file = os.path.join(work_dir, 'eval.json')

        # 产生帧级预测分数（写到 eval.json）
        heuristic_predicator(test_data, output_file, norm=norm)

        # —— F1 —— #
        split_f1, _, _ = evaluate_summaries(output_file, gt_file, test_keys, mapping, metric, test=True)
        f1_splits.append(split_f1)
        print(f"[Split {i+1}] F1: {split_f1:.4f}")

        # —— Rank correlations（用 gtscore，上采样到与预测相同长度） —— #
        mean_rho, mean_tau, _ = evaluate_rank_correlations_aligned(
            frames_score_file=output_file,
            gt_file=gt_file,
            keys=test_keys,
            mapping=mapping
        )
        rho_splits.append(mean_rho)
        tau_splits.append(mean_tau)
        print(f"[Split {i+1}] Spearman ρ: {mean_rho:.4f} | Kendall τ: {mean_tau:.4f}")

        # 清理中间文件
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
        except Exception:
            pass

    print("\n" + "=" * 60)
    print(f"Final Results for {metric} dataset:")
    if f1_splits:
        print(f"F1-score (mean over splits): {np.mean(f1_splits):.6f}")
    else:
        print("F1-score: N/A")
    if rho_splits:
        print(f"Spearman ρ (mean over splits): {np.mean(rho_splits):.6f}")
    else:
        print("Spearman ρ: N/A")
    if tau_splits:
        print(f"Kendall τ (mean over splits): {np.mean(tau_splits):.6f}")
    else:
        print("Kendall τ: N/A")
    print("=" * 60 + "\n")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate F1 + Spearman/Kendall on tvsum/summe")
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--gt_file", type=str, required=True, help="Ground truth .h5")
    parser.add_argument("--splits_file", type=str, required=True)
    parser.add_argument("--mapping_file", type=str, required=True, help="h5 key -> pred key mapping json")
    parser.add_argument("--meta_data_dir", type=str, required=True, help="dir that contains PredMetaData_{1..4}")
    parser.add_argument("--metric", type=str, choices=['summe', 'tvsum'], required=True)
    parser.add_argument("--norm", type=str, choices=['None', 'MinMax', 'Exp', 'MinMax+Exp'], default='None')

    args = parser.parse_args()
    run(args)
