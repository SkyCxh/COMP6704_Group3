# src/model.py

from typing import Tuple

import numpy as np


class LogisticRegressionModel:
    """
    基于 LLM 嵌入的二分类逻辑回归模型：
        p(y=1|x) = sigmoid(w^T x + b)
    """

    def __init__(self, dim: int):
        """
        dim: 特征维度（embedding 维度）
        """
        self.w = np.zeros(dim, dtype=np.float32)
        self.b = 0.0

    # ---- 统一的参数接口（供算法使用） ----
    def get_weights(self) -> np.ndarray:
        return self.w

    def set_weights(self, w: np.ndarray):
        self.w = np.array(w, dtype=np.float32)

    def get_bias(self):
        return self.b

    def set_bias(self, b):
        # b 是标量
        self.b = float(b)

    # ---- 二分类逻辑回归本身 ----
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        输入特征 X: shape [N, d]
        返回预测概率 p(y=1|x): shape [N,]
        """
        z = X @ self.w + self.b
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        根据给定阈值输出 0/1 标签。
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(np.int64)

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """
        计算 logistic loss 以及对 w, b 的梯度。
        y: shape [N,]，取值为 0 或 1
        """
        N = X.shape[0]
        z = X @ self.w + self.b
        p = self._sigmoid(z)

        eps = 1e-12  # 防止 log(0)
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

        diff = (p - y).astype(np.float32)
        grad_w = (X.T @ diff) / N
        grad_b = float(np.mean(diff))

        return float(loss), grad_w, grad_b


class SoftmaxRegressionModel:
    """
    多类 softmax 逻辑回归：
        p(y=k|x) = exp(w_k^T x + b_k) / sum_j exp(w_j^T x + b_j)
    参数：
        W: [d, K]
        b: [K]
    """

    def __init__(self, dim: int, num_classes: int):
        self.W = np.zeros((dim, num_classes), dtype=np.float32)
        self.b = np.zeros(num_classes, dtype=np.float32)

    # ---- 统一参数接口 ----
    def get_weights(self) -> np.ndarray:
        return self.W

    def set_weights(self, W: np.ndarray):
        self.W = np.array(W, dtype=np.float32)

    def get_bias(self) -> np.ndarray:
        return self.b

    def set_bias(self, b):
        self.b = np.array(b, dtype=np.float32)

    # ---- 模型前向与损失 ----
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        # logits: [N, K]
        logits = logits - logits.max(axis=1, keepdims=True)  # 防止溢出
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        输入 X: [N, d]
        返回 p(y=k|x): [N, K]
        """
        logits = X @ self.W + self.b  # [N, K]
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        返回预测类别标签，shape: [N,]，取值 0..K-1
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        多类交叉熵损失及梯度。
        y: [N,]，取值 0..K-1
        """
        N, d = X.shape
        K = self.W.shape[1]

        logits = X @ self.W + self.b  # [N, K]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # [N, K]

        eps = 1e-12
        log_probs = np.log(probs + eps)
        loss = -np.mean(log_probs[np.arange(N), y])

        # one-hot
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(N), y] = 1.0
        diff = (probs - one_hot).astype(np.float32)  # [N, K]

        grad_W = (X.T @ diff) / N          # [d, K]
        grad_b = diff.mean(axis=0)         # [K]

        return float(loss), grad_W, grad_b
