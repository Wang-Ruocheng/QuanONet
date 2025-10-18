#!/usr/bin/env python3
import os
import json
import math
from glob import glob
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator


def safe_min_train_loss(training_history: List[Dict[str, Any]]) -> float:
    if not isinstance(training_history, list) or len(training_history) == 0:
        return math.nan
    vals = []
    for rec in training_history:
        try:
            v = float(rec.get("train_loss"))
            if math.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    return float(min(vals)) if vals else math.nan


def safe_results_mse(results: Dict[str, Any]) -> float:
    if not isinstance(results, dict):
        return math.nan
    try:
        v = float(results.get("MSE"))
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def collect_rank_metrics(base_dir: str, rank_from: int = 1, rank_to: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ranks = []
    train_means = []
    train_stds = []
    test_means = []
    test_stds = []

    for r in range(rank_from, rank_to + 1):
        rank_dir = os.path.join(base_dir, f"rank{r}")
        if not os.path.isdir(rank_dir):
            # 跳过不存在的 rank 目录
            continue
        files = sorted(glob(os.path.join(rank_dir, "*.json")))
        if not files:
            continue

        train_vals = []
        test_vals = []
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            train_vals.append(safe_min_train_loss(data.get("training_history")))
            test_vals.append(safe_results_mse(data.get("results")))

        # 过滤 NaN
        train_arr = np.array([v for v in train_vals if isinstance(v, (int, float)) and math.isfinite(v)], dtype=float)
        test_arr = np.array([v for v in test_vals if isinstance(v, (int, float)) and math.isfinite(v)], dtype=float)
        if train_arr.size == 0 and test_arr.size == 0:
            continue

        ranks.append(r)
        train_means.append(float(np.mean(train_arr)) if train_arr.size else math.nan)
        train_stds.append(float(np.std(train_arr, ddof=0)) if train_arr.size else math.nan)
        test_means.append(float(np.mean(test_arr)) if test_arr.size else math.nan)
        test_stds.append(float(np.std(test_arr, ddof=0)) if test_arr.size else math.nan)

    return (
        np.array(ranks, dtype=int),
        np.array(train_means, dtype=float),
        np.array(train_stds, dtype=float),
        np.array(test_means, dtype=float),
        np.array(test_stds, dtype=float),
    )


def plot_with_shade(x: np.ndarray, y: np.ndarray, ystd: np.ndarray, label: str, color: str, marker: str = "o", ax=None):
    ax = ax or plt.gca()
    ax.plot(x, y, label=label, color=color, marker=marker, linewidth=1.8, markersize=5)
    # 仅对非 NaN 位置绘制阴影
    mask = np.isfinite(y) & np.isfinite(ystd)
    if np.any(mask):
        ax.fill_between(x[mask], y[mask] - ystd[mask], y[mask] + ystd[mask], color=color, alpha=0.18, linewidth=0)


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "logs", "Homogeneous")
    ranks, train_mean, train_std, test_mean, test_std = collect_rank_metrics(base_dir, 1, 32)

    if ranks.size == 0:
        print(f"未在 {base_dir} 下找到任何 rank 目录或有效 json 文件。")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_with_shade(ranks, train_mean, train_std, label="Train (min train_loss)", color="#1f77b4", marker="o", ax=ax)
    plot_with_shade(ranks, test_mean, test_std, label="Test (MSE)", color="#d62728", marker="s", ax=ax)

    # 轴与网格美化
    ax.set_xlabel("Hamiltonian rank")
    ax.set_ylabel("Error")
    ax.set_xlim(1, 32)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which="major", alpha=0.35, linestyle="--", linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.12, linestyle=":", linewidth=0.6)

    # Y 轴以 ×10^{-4} 显示
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
    ax.yaxis.get_offset_text().set_size(10)

    ax.legend(frameon=False)
    fig.tight_layout()

    out_dir = base_dir
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "rank_errors.png")
    plt.savefig(png_path, dpi=180)
    print(f"保存图像: {png_path}")

    # 同时导出一个 CSV 便于后续处理
    csv_path = os.path.join(out_dir, "rank_errors.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "train_mean", "train_std", "test_mean", "test_std"])
        for i in range(len(ranks)):
            writer.writerow([int(ranks[i]), train_mean[i], train_std[i], test_mean[i], test_std[i]])
    print(f"保存数据: {csv_path}")


if __name__ == "__main__":
    main()
