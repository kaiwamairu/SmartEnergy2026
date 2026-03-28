"""
seed_utils.py — Global seed management เพื่อ reproducibility
ควบคุม random state ของทุก library ที่ใช้ใน project
"""
import os
import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """
    ตั้งค่า seed สำหรับทุก library เพื่อให้ผลลัพธ์ reproducible

    Args:
        seed: ค่า seed (ดึงจาก config['project']['seed'])

    Example:
        set_global_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
