# scripts/extract_embeddings.py

import os
import sys
import argparse

import numpy as np

# 确保可以导入 src 包
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import get_config
from src.data_utils import load_raw_dataset, train_val_split
from src.embeddings import load_embedding_model, encode_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config yaml (e.g. configs/default.yaml or configs/ag_news.yaml)",
    )
    args = parser.parse_args()

    cfg = get_config(args.config)

    # 1. 加载原始数据
    dataset_name = cfg.dataset.name
    max_samples = cfg.dataset.max_samples
    print(f"[INFO] Loading raw dataset '{dataset_name}' (max_samples={max_samples})...")
    texts, labels = load_raw_dataset(dataset_name, max_samples=max_samples)
    print(f"[INFO] Loaded {len(texts)} samples.")

    # 2. 加载预训练句向量模型
    print(f"[INFO] Loading embedding model '{cfg.embeddings.model_name}'...")
    model = load_embedding_model(cfg.embeddings.model_name)

    # 3. 编码文本为 embeddings
    print("[INFO] Encoding texts into embeddings...")
    embeddings = encode_texts(model, texts, batch_size=cfg.embeddings.batch_size)
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    # 4. 训练/验证划分
    train_ratio = cfg.experiment.train_ratio
    seed = cfg.experiment.seed
    X_train, y_train, X_val, y_val = train_val_split(embeddings, labels, train_ratio, seed)
    print(f"[INFO] Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

    # 5. 保存到 data/processed/（或配置指定的子目录）
    processed_dir = cfg.paths.processed_dir
    os.makedirs(processed_dir, exist_ok=True)

    np.save(os.path.join(processed_dir, "train_embeddings.npy"), X_train)
    np.save(os.path.join(processed_dir, "train_labels.npy"), y_train)
    np.save(os.path.join(processed_dir, "val_embeddings.npy"), X_val)
    np.save(os.path.join(processed_dir, "val_labels.npy"), y_val)

    print(f"[INFO] Saved processed data to '{processed_dir}'")


if __name__ == "__main__":
    main()
