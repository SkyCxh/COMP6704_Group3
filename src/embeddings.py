# src/embeddings.py

from typing import List

import numpy as np

from .config import get_config

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def load_embedding_model(model_name: str = None):
    """
    加载预训练句向量模型。
    默认使用 configs/default.yaml 中的 model_name。
    """
    cfg = get_config()
    if model_name is None:
        model_name = cfg.embeddings.model_name

    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers 库未安装，请先 pip install sentence-transformers"
        )

    model = SentenceTransformer(model_name)
    return model


def encode_texts(model, texts: List[str], batch_size: int = None) -> np.ndarray:
    """
    使用预训练模型将一批文本编码成 embeddings。
    返回 shape: [N, d] 的 numpy 数组。
    """
    cfg = get_config()
    if batch_size is None:
        batch_size = cfg.embeddings.batch_size

    # SentenceTransformer 自带批处理 encode
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # 这里先不做归一化，有需要再在算法中处理
    )
    return embeddings.astype(np.float32)
