# utils/seed.py
import os, random, numpy as np
def set_seed(seed: int = 2025):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)  # 统一 TF 随机源
    except Exception:
        pass
