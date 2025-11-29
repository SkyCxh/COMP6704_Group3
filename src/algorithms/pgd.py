# src/algorithms/pgd.py

from typing import Dict, Any

import numpy as np

from ..config import get_config


def _compute_accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    preds = model.predict(X)
    return float((preds == y).mean())


def _constraint_violation(W: np.ndarray, R: float, C: float) -> float:
    """
    简单度量：L2 约束违背 + box 约束违背之和。
    对于二分类（一维 w）和多分类（二维 W）都适用。
    """
    norm_W = np.linalg.norm(W)
    l2_violation = max(0.0, norm_W - R)
    box_violation_vec = np.maximum(0.0, np.abs(W) - C)
    box_violation = float(box_violation_vec.sum())
    return float(l2_violation + box_violation)


def pgd_optimize(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg=None,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> Dict[str, Any]:
    """
    全批量投影梯度下降（Projected Gradient Descent, PGD）
    对权重 W 进行两步投影：先 box 再 L2 球。
    返回 history，用于画收敛曲线等。
    """
    if cfg is None:
        cfg = get_config()
    opt_cfg = cfg.optimization

    lr = float(opt_cfg.lr)
    max_iter = int(opt_cfg.max_iter)
    R = float(opt_cfg.R)
    C = float(opt_cfg.C)

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
        # 1. 用当前参数计算 loss 和 grad
        model.set_weights(W)
        model.set_bias(b)
        loss, grad_W, grad_b = model.loss_and_grad(X_train, y_train)

        # 2. 梯度下降一步
        W = W - lr * grad_W
        b = b - lr * grad_b

        # 3. 投影到 box [-C, C]
        W = np.clip(W, -C, C)

        # 4. 投影到 L2 球 ||W||_F <= R
        norm_W = np.linalg.norm(W)
        if norm_W > R > 0:
            W = W * (R / norm_W)

        # 更新模型参数
        model.set_weights(W)
        model.set_bias(b)

        # 5. 记录指标
        train_loss = loss
        train_acc = _compute_accuracy(model, X_train, y_train)

        if X_val is not None and y_val is not None:
            val_loss, _, _ = model.loss_and_grad(X_val, y_val)
            val_acc = _compute_accuracy(model, X_val, y_val)
        else:
            val_loss, val_acc = float("nan"), float("nan")

        cv = _constraint_violation(W, R, C)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["constraint_violation"].append(cv)

    return history
