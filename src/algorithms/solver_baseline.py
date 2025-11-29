# src/algorithms/solver_baseline.py

from typing import Dict, Any

import numpy as np

from ..model import LogisticRegressionModel
from ..config import get_config

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


def _compute_accuracy(model: LogisticRegressionModel, X: np.ndarray, y: np.ndarray) -> float:
    preds = model.predict(X)
    return float((preds == y).mean())


def _pack_params(w: np.ndarray, b: float) -> np.ndarray:
    return np.concatenate([w, np.array([b], dtype=w.dtype)])


def _unpack_params(theta: np.ndarray) -> (np.ndarray, float):
    return theta[:-1], float(theta[-1])


def solver_optimize(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg=None,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> Dict[str, Any]:
    """
    使用 SciPy 的通用约束优化求解器（例如 SLSQP）作为 baseline。
    注意：当前版本仅用于二分类任务 (label ∈ {0,1})。
    """
    if minimize is None:
        raise ImportError("scipy 未安装，请先 pip install scipy")

    # 如果不是二分类，直接报错，由外层决定是否跳过
    num_classes = int(np.max(y_train)) + 1
    if num_classes != 2:
        raise NotImplementedError("SciPy solver baseline is only implemented for binary classification.")


    if cfg is None:
        cfg = get_config()
    opt_cfg = cfg.optimization

    R = float(opt_cfg.R)
    C = float(opt_cfg.C)

    N, dim = X_train.shape

    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def objective(theta: np.ndarray) -> float:
        w, b = _unpack_params(theta)
        z = X_train @ w + b
        p = sigmoid(z)
        eps = 1e-12
        loss = -np.mean(y_train * np.log(p + eps) + (1 - y_train) * np.log(1 - p + eps))
        return float(loss)

    def objective_grad(theta: np.ndarray) -> np.ndarray:
        w, b = _unpack_params(theta)
        z = X_train @ w + b
        p = sigmoid(z)
        diff = (p - y_train).astype(np.float64)
        grad_w = (X_train.T @ diff) / N
        grad_b = float(np.mean(diff))
        return _pack_params(grad_w, grad_b)

    # 不等式约束：c_i(theta) >= 0
    # 1) L2：R^2 - ||w||^2 >= 0
    def cons_l2(theta: np.ndarray) -> float:
        w, _ = _unpack_params(theta)
        return R * R - float(np.dot(w, w))

    # 2) box：C - w_j >= 0, C + w_j >= 0
    cons = [{"type": "ineq", "fun": cons_l2}]
    for j in range(dim):
        cons.append({
            "type": "ineq",
            "fun": lambda theta, j=j: C - _unpack_params(theta)[0][j]
        })
        cons.append({
            "type": "ineq",
            "fun": lambda theta, j=j: C + _unpack_params(theta)[0][j]
        })

    theta0 = _pack_params(np.zeros(dim, dtype=np.float64), 0.0)

    res = minimize(
        objective,
        theta0,
        method="SLSQP",
        jac=objective_grad,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-6, "disp": False},
    )

    w_opt, b_opt = _unpack_params(res.x)

    # 使用 LogisticRegressionModel 接一下，便于后面统一计算 acc
    model = LogisticRegressionModel(dim)
    model.w = w_opt.astype(np.float32)
    model.b = b_opt

    train_loss = objective(res.x)
    train_acc = _compute_accuracy(model, X_train, y_train)

    if X_val is not None and y_val is not None:
        z_val = X_val @ w_opt + b_opt
        p_val = sigmoid(z_val)
        eps = 1e-12
        val_loss = -np.mean(y_val * np.log(p_val + eps) + (1 - y_val) * np.log(1 - p_val + eps))
        val_acc = _compute_accuracy(model, X_val, y_val)
    else:
        val_loss, val_acc = float("nan"), float("nan")

    # 为了与其他算法的 history 结构保持一致，这里也返回单元素列表
    history = {
        "train_loss": [float(train_loss)],
        "val_loss": [float(val_loss)],
        "train_acc": [float(train_acc)],
        "val_acc": [float(val_acc)],
        "constraint_violation": [],  # 如有需要，可以在此计算
    }

    return history
