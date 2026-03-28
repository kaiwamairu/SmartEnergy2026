"""
mlflow_helpers.py — MLflow tracking utilities
จัดการ experiment setup, run naming, และ artifact logging
"""
import os
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import mlflow


def setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """
    ตั้งค่า MLflow tracking URI และ experiment

    Args:
        tracking_uri: URI ของ MLflow server เช่น 'sqlite:///runs/mlruns.db' หรือ 'http://...'
        experiment_name: ชื่อ experiment ใน MLflow

    Returns:
        experiment_id ที่สร้างหรือดึงมา
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    return exp.experiment_id


def make_run_name(algo: str, version: int) -> str:
    """
    สร้าง run name ตาม convention: {algo}_v{version}_{timestamp}

    Args:
        algo: ชื่อ algorithm เช่น 'xgb', 'rf', 'ridge'
        version: version number

    Returns:
        run name string

    Example:
        make_run_name('xgb', 1) → 'xgb_v1_20260328_143022'
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{algo}_v{version}_{ts}"


def hash_file(file_path: str) -> str:
    """
    คำนวณ MD5 hash ของไฟล์ข้อมูล เพื่อ track data version ใน MLflow

    Args:
        file_path: path ของไฟล์

    Returns:
        MD5 hash string (first 12 chars)
    """
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def log_pip_freeze(artifact_dir: str = ".") -> None:
    """
    บันทึก pip freeze output เป็น artifact ใน MLflow
    ทำทุกครั้งหลังเทรนสำเร็จเพื่อ reproducibility

    Args:
        artifact_dir: directory ชั่วคราวสำหรับเก็บไฟล์ก่อน log
    """
    freeze_path = Path(artifact_dir) / "requirements_freeze.txt"
    result = subprocess.run(
        ["pip", "freeze"], capture_output=True, text=True
    )
    freeze_path.write_text(result.stdout, encoding="utf-8")
    mlflow.log_artifact(str(freeze_path))


def log_scaler_params(scaler, prefix: str = "scaler") -> None:
    """
    บันทึก Scaler parameters (mean, std) ลง MLflow เพื่อใช้ตอน Inference

    Args:
        scaler: fitted sklearn scaler (StandardScaler / MinMaxScaler)
        prefix: prefix สำหรับชื่อ param
    """
    if hasattr(scaler, "mean_"):
        for i, (mean, std) in enumerate(zip(scaler.mean_, scaler.scale_)):
            mlflow.log_param(f"{prefix}_mean_{i}", round(float(mean), 6))
            mlflow.log_param(f"{prefix}_std_{i}", round(float(std), 6))
