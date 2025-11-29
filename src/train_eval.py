# src/train_eval.py

import os
import json
import argparse
from typing import Dict, Any

import numpy as np

from .config import get_config
from .data_utils import load_processed_data
from .model import LogisticRegressionModel, SoftmaxRegressionModel
from .algorithms.pgd import pgd_optimize
from .algorithms.spgd import spgd_optimize
from .algorithms.penalty import penalty_optimize
from .algorithms.barrier import barrier_optimize
from .algorithms.solver_baseline import solver_optimize
from .plots import plot_convergence, plot_final_accuracy_bar


def run_all_algorithms(cfg) -> Dict[str, Any]:
    """
    统一运行所有算法，并返回各自的 history。
    根据标签类别数自动选择模型类型：
      - 2 类: LogisticRegressionModel（二分类）
      - >2 类: SoftmaxRegressionModel（多分类）
    """
    print("[INFO] Loading processed embeddings and labels...")
    X_train, y_train, X_val, y_val = load_processed_data()
    dim = X_train.shape[1]
    num_classes = int(np.max(y_train)) + 1
    print(f"[INFO] Train shape: {X_train.shape}, Val shape: {X_val.shape}, num_classes={num_classes}")

    def make_model():
        if num_classes == 2:
            return LogisticRegressionModel(dim)
        else:
            return SoftmaxRegressionModel(dim, num_classes)

    algorithms = {
        "pgd": pgd_optimize,
        "spgd": spgd_optimize,
        "penalty": penalty_optimize,
        "barrier": barrier_optimize,
    }

    histories: Dict[str, Any] = {}

    # 1. 自己实现的四个算法
    for name, algo_fn in algorithms.items():
        print(f"[INFO] Running algorithm: {name} ...")
        model = make_model()
        history = algo_fn(model, X_train, y_train, cfg, X_val, y_val)
        histories[name] = history

    # 2. SciPy baseline（只在二分类时运行）
    if num_classes == 2:
        print("[INFO] Running SciPy solver baseline ...")
        try:
            baseline_history = solver_optimize(X_train, y_train, cfg, X_val, y_val)
            histories["solver"] = baseline_history
        except NotImplementedError as e:
            print(f"[WARN] Solver baseline skipped: {e}")
        except ImportError as e:
            print(f"[WARN] SciPy solver not available: {e}")
    else:
        print("[INFO] Multi-class task detected; skipping SciPy solver baseline.")

    return histories


def save_metrics(histories: Dict[str, Any], out_dir: str):
    """
    将每个算法的最终 train/val 指标写入 JSON，方便后续查阅或导入表格。
    这里把 numpy 类型转换成 Python 原生类型，避免 json.dump 报错。
    """
    os.makedirs(out_dir, exist_ok=True)

    def last(seq):
        return seq[-1] if seq else None

    def to_py_float(x):
        """
        将 numpy 标量安全地转换为 Python float；None 保持不变。
        """
        if x is None:
            return None
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, (float, int)):
            return x
        try:
            return float(x)
        except Exception:
            return None

    summary = {}
    for name, h in histories.items():
        summary[name] = {
            "final_train_loss": to_py_float(last(h.get("train_loss", []))),
            "final_val_loss": to_py_float(last(h.get("val_loss", []))),
            "final_train_acc": to_py_float(last(h.get("train_acc", []))),
            "final_val_acc": to_py_float(last(h.get("val_acc", []))),
        }

    out_path = os.path.join(out_dir, "metrics_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved metrics summary to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config yaml (e.g. configs/imdb.yaml or configs/ag_news.yaml)",
    )
    args = parser.parse_args()

    # 第一次调用 get_config 时指定路径，后续模块都会用同一个配置
    cfg = get_config(args.config)

    # === 关键改动：根据 config 名字生成一个实验标签，用于分目录保存结果 ===
    if args.config is not None:
        exp_tag = os.path.splitext(os.path.basename(args.config))[0]  # e.g. imdb / ag_news
    else:
        # 没传 --config 就用 dataset.name 或 default
        exp_tag = getattr(cfg.dataset, "name", "default")

    print(f"[INFO] Experiment tag: {exp_tag}")

    histories = run_all_algorithms(cfg)

    # 项目根目录
    root_dir = os.path.dirname(os.path.dirname(__file__))

    # 数值结果目录：results/metrics/<exp_tag>/
    metrics_dir = os.path.join(root_dir, "results", "metrics", exp_tag)
    save_metrics(histories, metrics_dir)

    # 图像结果目录：results/figures/<exp_tag>/
    figures_dir = os.path.join(root_dir, "results", "figures", exp_tag)
    os.makedirs(figures_dir, exist_ok=True)

    # 1) 收敛曲线：train loss
    plot_convergence(
        histories,
        save_path=os.path.join(figures_dir, "train_loss_convergence.png"),
    )

    # 2) 最终验证集准确率柱状图
    plot_final_accuracy_bar(
        histories,
        save_path=os.path.join(figures_dir, "final_val_accuracy.png"),
    )

    print("[INFO] All experiments finished.")


if __name__ == "__main__":
    main()
