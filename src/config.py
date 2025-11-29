# src/config.py

import os
import yaml


class Config(dict):
    """
    简单的字典封装，支持点号访问：
        cfg['dataset']['name'] 或 cfg.dataset.name 都可以。
    """

    def __getattr__(self, item):
        value = self.get(item)
        if isinstance(value, dict):
            return Config(value)
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# 全局缓存：第一次加载的 config 会被缓存，之后调用 get_config() 都返回它
_CACHED_CONFIG: Config | None = None


def get_config(path: str = None) -> Config:
    """
    读取 YAML 配置文件并返回 Config 对象。
    - 如果第一次调用传入 path，则后续所有 get_config() 都返回同一份配置。
    - 如果从未传入 path，则默认读取 configs/default.yaml。
    """
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None:
        return _CACHED_CONFIG

    if path is None:
        # 默认路径：项目根目录下的 configs/default.yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "default.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    _CACHED_CONFIG = Config(cfg_dict)
    return _CACHED_CONFIG
