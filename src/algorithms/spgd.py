# src/algorithms/spgd.py

from typing import Dict, Any

import numpy as np

from ..config import get_config


def _compute_accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    preds = model.predict(X)
    return float((preds == y).mean())


def _constraint_violation(W: np.ndarray, R: float, C: float) -> float:
    norm_W = np.linalg.norm(W)
    l2_violation = max(0.0, norm_W - R)
    box_violation_vec = np.maximum(0.0, np.abs(W) - C)
    box_violation = float(box_violation_vec.sum())
    return float(l2_violation + box_violation)


def spgd_optimize(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg=None,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> Dict[str, Any]:
    """
    随机投影梯度下降（Stochastic PGD）
    每个 epoch 使用 mini-batch 随机梯度更新，多次投影约束。
    """
    if cfg is None:
        cfg = get_config()
    opt_cfg = cfg.optimization
    exp_cfg = cfg.experiment

    lr = float(opt_cfg.lr)
    batch_size = int(opt_cfg.batch_size)
    epochs = int(opt_cfg.epochs)
    R = float(opt_cfg.R)
    C = float(opt_cfg.C)
    seed = int(exp_cfg.seed)

    rng = np.random.default_rng(seed)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "constraint_violation": [],
    }

    W = model.get_weights().copy()
    b = model.get_bias()

    N = X_train.shape[0]
    indices = np.arange(N)

    for epoch in range(epochs):
        # 打乱数据
        rng.shuffle(indices)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = indices[start:end]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            # 用 mini-batch 计算梯度
            model.set_weights(W)
            model.set_bias(b)
            _, grad_W, grad_b = model.loss_and_grad(X_batch, y_batch)

            # 梯度更新
            W = W - lr * grad_W
            b = b - lr * grad_b

            # 投影到 box
            W = np.clip(W, -C, C)
            # 投影到 L2 球
            norm_W = np.linalg.norm(W)
            if norm_W > R > 0:
                W = W * (R / norm_W)

        # 一个 epoch 结束后，记录一次全数据指标
        model.set_weights(W)
        model.set_bias(b)

        train_loss, _, _ = model.loss_and_grad(X_train, y_train)
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
