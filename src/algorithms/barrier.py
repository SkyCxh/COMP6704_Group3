# src/algorithms/barrier.py

from typing import Dict, Any

import numpy as np

from ..config import get_config


def _compute_accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    preds = model.predict(X)
    return float((preds == y).mean())


def _constraint_violation(W: np.ndarray, R: float, C: float) -> float:
    """
    统一的约束违背度量（兼容一维 w 和二维 W）：
      - L2/Frobenius 约束：max(0, ||W|| - R)
      - box 约束：sum_i max(0, |W_i| - C)
    """
    norm_W = np.linalg.norm(W)
    l2_violation = max(0.0, norm_W - R)
    box_violation_vec = np.maximum(0.0, np.abs(W) - C)
    box_violation = float(box_violation_vec.sum())
    return float(l2_violation + box_violation)


def barrier_optimize(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg=None,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> Dict[str, Any]:
    """
    对数障碍法（简单版本）：

        L_total = base_loss
                  - (1/t) * [ log(R^2 - ||W||_F^2)
                              + sum_i (log(C - W_i) + log(C + W_i)) ]

    注意：为了便于和 PGD / SPGD / Penalty 对比，
    **history['train_loss'] / ['val_loss'] 始终记录 base_loss
    （即纯 logistic / softmax 交叉熵），
    而不是含障碍项的 L_total。障碍项只用于更新方向。
    """
    if cfg is None:
        cfg = get_config()
    opt_cfg = cfg.optimization

    max_iter = int(opt_cfg.max_iter)
    R = float(opt_cfg.R)
    C = float(opt_cfg.C)
    t = float(opt_cfg.barrier_t)
    lr = float(opt_cfg.barrier_lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "constraint_violation": [],
    }

    # 统一使用 get_weights / get_bias 接口，兼容二分类 & 多分类
    W = model.get_weights().copy()
    b = model.get_bias()

    eps = 1e-4  # 用于避免太贴近约束边界

    for it in range(max_iter):
        # ---- 0) 确保 W 在“缩小”后的可行域内（留一点距离） ----
        norm_W = np.linalg.norm(W)
        max_norm = (1.0 - eps) * R
        if norm_W >= max_norm and norm_W > 0:
            W = W * (max_norm / norm_W)

        W = np.clip(W, -C * (1.0 - eps), C * (1.0 - eps))

        # ---- 1) 基础损失和梯度（不含障碍项） ----
        model.set_weights(W)
        model.set_bias(b)
        base_loss, grad_W_base, grad_b_base = model.loss_and_grad(X_train, y_train)

        # ---- 2) 障碍项及其梯度 ----
        # L2/Frobenius 约束：R^2 - ||W||_F^2 > 0
        norm_W_sq = float(np.sum(W * W))
        s_l2 = R * R - norm_W_sq
        if s_l2 <= 0:
            # 数值出了可行域，提前退出
            break

        barrier_loss_l2 = - (1.0 / t) * np.log(s_l2)
        grad_barrier_l2 = (2.0 * W) / (t * s_l2)

        # box 约束：C - W_i > 0, C + W_i > 0
        s1 = C - W
        s2 = C + W
        if np.any(s1 <= 0) or np.any(s2 <= 0):
            # 数值出界，提前退出
            break

        barrier_loss_box = - (1.0 / t) * (np.log(s1).sum() + np.log(s2).sum())
        # d/dW_i：
        grad_barrier_box = (1.0 / (t * s1)) - (1.0 / (t * s2))

        # 总目标（只用于更新，不记录在 history 里）
        barrier_loss = barrier_loss_l2 + barrier_loss_box
        total_loss = base_loss + barrier_loss

        grad_W = grad_W_base + grad_barrier_l2 + grad_barrier_box
        grad_b = grad_b_base  # 障碍项不依赖 b

        # ---- 3) 梯度下降更新 ----
        W = W - lr * grad_W
        b = b - lr * grad_b

        # ---- 4) 记录指标（使用 base_loss） ----
        model.set_weights(W)
        model.set_bias(b)

        # 此处记录的是“纯分类 loss”，和其它算法保持一致
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
