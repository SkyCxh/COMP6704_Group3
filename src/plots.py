# src/plots.py
"""
High-quality plotting utilities for constrained optimization experiments.

Design goals:
- Clean and minimal style (适合论文/报告).
- Clear comparison between iterative methods (PGD / SPGD / Penalty / Barrier).
- 不再在图中展示 SciPy solver，避免压缩坐标或引入额外视觉噪音。
"""

from typing import Dict, Any, List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------
# Global plotting configuration
# -----------------------------

mpl.rcParams.update({
    # Figure size: 单图默认 5.5 x 3.5 英寸，适合论文列宽
    "figure.figsize": (5.5, 3.5),
    "savefig.dpi": 300,
    "figure.dpi": 300,

    # Font & text
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    # Lines
    "lines.linewidth": 1.8,
    "lines.markersize": 4,

    # Grid
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
})


# 简洁、颜色盲友好的颜色
_ALGO_COLORS = {
    "pgd": "#1f77b4",      # 蓝
    "spgd": "#ff7f0e",     # 橙
    "penalty": "#2ca02c",  # 绿
    "barrier": "#d62728",  # 红
}

# legend 中使用的名字
_ALGO_DISPLAY_NAMES = {
    "pgd": "PGD",
    "spgd": "SPGD",
    "penalty": "Penalty",
    "barrier": "Barrier",
}

# 推荐绘制顺序
_ALGO_ORDER = ["pgd", "spgd", "penalty", "barrier"]


# -----------------------------
# Helpers
# -----------------------------

def _clean_axis(ax: plt.Axes):
    """简化坐标轴样式：去掉顶部/右侧边框，只保留底部和左侧。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", width=0.8)


def _display_name(key: str) -> str:
    return _ALGO_DISPLAY_NAMES.get(key, key)


def _color_for(key: str) -> str:
    return _ALGO_COLORS.get(key, "#7f7f7f")


def _iterative_algo_keys(histories: Dict[str, Any]) -> List[str]:
    """
    只保留我们关心的迭代算法，并且要求 train_loss 序列长度 > 1。
    不包含 SciPy solver 等一次性 baseline。
    """
    keys = []
    for k in _ALGO_ORDER:  # 按既定顺序扫描
        h = histories.get(k, {})
        losses = h.get("train_loss", [])
        if len(losses) > 1:
            keys.append(k)
    return keys


def _gather_all_losses(histories: Dict[str, Any], algo_keys: List[str], key: str = "train_loss") -> np.ndarray:
    vals: List[float] = []
    for k in algo_keys:
        h = histories.get(k, {})
        seq = h.get(key, [])
        vals.extend([float(v) for v in seq if np.isfinite(v)])
    return np.array(vals, dtype=float) if vals else np.array([], dtype=float)


# -----------------------------
# Main plotting functions
# -----------------------------

def plot_convergence(histories: Dict[str, Any], save_path: str):
    """
    画不同算法的训练误差收敛曲线 (train loss vs. iteration)。

    特点：
    - 只画 PGD / SPGD / Penalty / Barrier 等迭代算法；
    - 自动根据这几条曲线的取值范围设置 y 轴上下界；
    - legend 放在图内右上角，利用好画布空间；
    - PGD / SPGD 用 marker 辅助区分，Penalty / Barrier 用不同线型。
    """
    algo_keys = _iterative_algo_keys(histories)
    if not algo_keys:
        print("[WARN] No iterative algorithms with train_loss history to plot.")
        return

    all_losses = _gather_all_losses(histories, algo_keys, key="train_loss")

    fig, ax = plt.subplots()
    _clean_axis(ax)

    # 不同算法的线型
    linestyle_map = {
        "pgd": "-",
        "spgd": "--",
        "penalty": "-.",
        "barrier": ":",
    }

    marker_algos = {"pgd", "spgd"}

    max_len = 0
    for k in algo_keys:
        losses = histories[k].get("train_loss", [])
        if losses:
            max_len = max(max_len, len(losses))

    for key in algo_keys:
        h = histories.get(key, {})
        losses = h.get("train_loss", [])
        if not losses:
            continue

        y = np.asarray(losses, dtype=float)
        x = np.arange(len(y))

        color = _color_for(key)
        label = _display_name(key)
        linestyle = linestyle_map.get(key, "-")

        if key in marker_algos and len(y) > 5:
            # marker 稀疏一点，避免太密
            markevery = max(1, len(y) // 15)
            ax.plot(
                x,
                y,
                linestyle=linestyle,
                color=color,
                marker="o",
                markevery=markevery,
                alpha=0.95,
                label=label,
            )
        else:
            ax.plot(x, y, linestyle=linestyle, color=color, alpha=0.95, label=label)

    # 自适应 y 轴：只围绕这些算法的 loss 做放大
    if all_losses.size > 0:
        y_min, y_max = float(all_losses.min()), float(all_losses.max())
        if y_max > y_min:
            margin = 0.08 * (y_max - y_min)
        else:
            margin = 0.02
        ax.set_ylim(y_min - margin, y_max + margin)

    # x 轴范围覆盖所有迭代次数
    if max_len > 1:
        ax.set_xlim(0, max_len - 1)
        # x 轴 tick 不用太密，最多 6 个
        n_ticks = min(6, max_len)
        ticks = np.linspace(0, max_len - 1, n_ticks, dtype=int)
        ax.set_xticks(ticks)

    ax.set_xlabel("Iteration / Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss convergence")

    # legend 放在图内右上角，透明背景，不挡主要曲线
    leg = ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
    )
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()  # 不再预留右侧空白
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved convergence plot to {save_path}")


def plot_final_accuracy_bar(histories: Dict[str, Any], save_path: str):
    """
    画各迭代算法最终验证集准确率的水平条形图。

    - 只展示 PGD / SPGD / Penalty / Barrier（与收敛图保持一致）；
    - 按准确率从高到低排序；
    - 每个算法颜色与收敛曲线一致；
    - 在条形末端标注数值（3 位小数）。
    """
    algo_keys: List[str] = []
    final_accs: List[float] = []

    for key in _ALGO_ORDER:
        h = histories.get(key, {})
        vals = h.get("val_acc", [])
        if not vals:
            continue
        algo_keys.append(key)
        final_accs.append(float(vals[-1]))

    if not algo_keys:
        print("[WARN] No accuracy data to plot.")
        return

    algo_keys = np.array(algo_keys)
    final_accs = np.array(final_accs, dtype=float)

    # 按准确率从高到低排序
    order = np.argsort(-final_accs)
    algo_keys = algo_keys[order]
    final_accs = final_accs[order]

    display_names = [_display_name(k) for k in algo_keys]
    colors = [_color_for(k) for k in algo_keys]
    y_pos = np.arange(len(display_names))

    fig, ax = plt.subplots()
    _clean_axis(ax)

    bars = ax.barh(y_pos, final_accs, color=colors, alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.set_xlabel("Final validation accuracy")
    ax.set_title("Final validation accuracy by algorithm")
    ax.set_xlim(0.0, 1.05)

    # 在条形末端标出数值
    for rect, acc in zip(bars, final_accs):
        width = rect.get_width()
        ax.text(
            width + 0.01,
            rect.get_y() + rect.get_height() / 2.0,
            f"{acc:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.grid(axis="x", linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved final accuracy bar plot to {save_path}")
