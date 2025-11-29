# src/data_utils.py

import os
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split

from .config import get_config

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def load_raw_dataset(name: str, max_samples: int = None) -> Tuple[List[str], np.ndarray]:
    """
    加载原始文本数据和标签。
    当前支持：
      - imdb: 电影评论情感二分类 (label ∈ {0,1})
      - ag_news: 新闻主题四分类 (label ∈ {0,1,2,3})
    返回：
        texts: List[str]
        labels: np.ndarray (shape: [N,], 元素为 0..K-1 的整数)
    """
    if load_dataset is None:
        raise ImportError("datasets 库未安装，请先 pip install datasets")

    name = name.lower()

    if name == "imdb":
        dataset = load_dataset("imdb")

        texts = []
        labels = []

        # 这里简单只用 train split，也可以合并 test 再重新划分
        for example in dataset["train"]:
            texts.append(str(example["text"]))
            labels.append(int(example["label"]))

            if max_samples is not None and len(texts) >= max_samples:
                break

        return texts, np.array(labels, dtype=np.int64)

    elif name == "ag_news":
        # AG News 原始有 4 类: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
        dataset = load_dataset("ag_news")

        texts = []
        labels = []

        for example in dataset["train"]:
            texts.append(str(example["text"]))
            labels.append(int(example["label"]))
            if max_samples is not None and len(texts) >= max_samples:
                break

        return texts, np.array(labels, dtype=np.int64)

    else:
        raise ValueError(f"Unsupported dataset name: {name}")


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对特征和标签进行训练/验证划分。
    返回：
        X_train, y_train, X_val, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=train_ratio, random_state=seed, stratify=y
    )
    return X_train, y_train, X_val, y_val


def load_processed_data(
    processed_dir: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从 data/processed 中加载已经生成好的 embeddings 和 labels。
    默认文件名：
        train_embeddings.npy
        train_labels.npy
        val_embeddings.npy
        val_labels.npy
    """
    cfg = get_config()
    if processed_dir is None:
        processed_dir = cfg.paths.processed_dir

    train_emb_path = os.path.join(processed_dir, "train_embeddings.npy")
    train_label_path = os.path.join(processed_dir, "train_labels.npy")
    val_emb_path = os.path.join(processed_dir, "val_embeddings.npy")
    val_label_path = os.path.join(processed_dir, "val_labels.npy")

    if not all(os.path.exists(p) for p in [train_emb_path, train_label_path, val_emb_path, val_label_path]):
        raise FileNotFoundError(
            "Processed data not found. "
            "Please run scripts/extract_embeddings.py first to generate embeddings."
        )

    X_train = np.load(train_emb_path)
    y_train = np.load(train_label_path)
    X_val = np.load(val_emb_path)
    y_val = np.load(val_label_path)

    return X_train, y_train, X_val, y_val
