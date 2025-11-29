# src/algorithms/penalty.py

from typing import Dict, Any

import numpy as np

from ..config import get_config


def _compute_accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    preds = model.predict(X)
    return float((preds == y).mean())


def _constraint_violation(W: np.ndarray, R: float, C: float) -> float:
    """
    统一的约束违背度量（兼容一维 w 和二维 W）。
    """
    norm_W = np.linalg.norm(W)
    l2_violation = max(0.0, norm_W - R)
    box_violation_vec = np.maximum(0.0, np.abs(W) - C)
    box_violation = float(box_violation_vec.sum())
    return float(l2_violation + box_violation)


def penalty_optimize(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg=None,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> Dict[str, Any]:
    """
    二次罚函数法：

        L_total = base_loss
                  + (rho_l2 / 2) * max(0, ||W||_F - R)^2
                  + (rho_box / 2) * sum_i max(0, |W_i| - C)^2

    注意：和 barrier 一样，history 中的 train_loss / val_loss
    记录的都是 base_loss（分类损失），罚项只用来决定更新方向。
    """
    if cfg is None:
        cfg = get_config()
    opt_cfg = cfg.optimization

    lr = float(opt_cfg.lr)
    max_iter = int(opt_cfg.max_iter)
    R = float(opt_cfg.R)
    C = float(opt_cfg.C)
    rho_l2 = float(opt_cfg.rho_l2)
    rho_box = float(opt_cfg.rho_box)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "constraint_violation": [],
    }

    W = model.get_weights().copy()
    b = model.get_bias()

    for it in range(max_iter):
        # 写回参数以复用 model.loss_and_grad
        model.set_weights(W)
        model.set_bias(b)
        base_loss, grad_W_base, grad_b_base = model.loss_and_grad(X_train, y_train)

        # ---- L2 约束罚项 ----
        norm_W = np.linalg.norm(W)
        l2_violation = max(0.0, norm_W - R)
        if norm_W > 1e-12 and l2_violation > 0.0:
            penalty_l2 = 0.5 * rho_l2 * (l2_violation ** 2)
            # d/dW penalty_l2 = rho_l2 * l2_violation * W / ||W||
            grad_penalty_l2 = rho_l2 * l2_violation * (W / norm_W)
        else:
            penalty_l2 = 0.0
            grad_penalty_l2 = np.zeros_like(W)

        # ---- box 约束罚项 ----
        abs_W = np.abs(W)
        box_violation_vec = np.maximum(0.0, abs_W - C)
        penalty_box = 0.5 * rho_box * np.sum(box_violation_vec ** 2)

        # d/dW_i penalty_box = rho_box * (|W_i| - C)_+ * sign(W_i)
        sign_W = np.sign(W)
        grad_penalty_box = rho_box * box_violation_vec * sign_W

        # 总目标（只用来算梯度）
        total_loss = base_loss + penalty_l2 + penalty_box
        grad_W = grad_W_base + grad_penalty_l2 + grad_penalty_box
        grad_b = grad_b_base  # 罚项不依赖 b

        # ---- 梯度下降 ----
        W = W - lr * grad_W
        b = b - lr * grad_b

        # 更新模型 & 记录
        model.set_weights(W)
        model.set_bias(b)

        # 注意：记录的是 base_loss，而不是 total_loss
        train_loss = float(base_loss)
        train_acc = _compute_accuracy(model, X_train, y_train)

        if X_val is not None and y_val is not None:
            val_loss, _, _ = model.loss_and_grad(X_val, y_val)
            val_acc = _compute_accuracy(model, X_val, y_val)
        else:
            val_loss, val_acc = float("nan"), float("nan")

        cv = _constraint_violation(W, R, C)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["constraint_violation"].append(cv)

    return history
