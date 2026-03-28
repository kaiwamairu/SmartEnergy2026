"""
train_pipeline.py — Training pipeline หลักสำหรับทุก algorithm
บันทึกผลลัพธ์ลง MLflow พร้อม artifacts ครบถ้วน
"""
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.utils.config_loader import load_config
from src.utils.seed_utils import set_global_seed
from src.utils.mlflow_helpers import setup_mlflow, make_run_name, hash_file, log_pip_freeze, log_scaler_params
from src.data.data_loader import load_raw_data, resample_hourly, load_processed_data
from src.data.preprocessor import prepare_features, time_series_split, fit_scaler, apply_scaler


def compute_metrics(y_true, y_pred) -> dict:
    """
    คำนวณ regression metrics ทั้งหมด

    Args:
        y_true: actual values
        y_pred: predicted values

    Returns:
        dict ของ rmse, mae, r2
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def plot_actual_vs_pred(y_true, y_pred, title: str, save_path: str) -> None:
    """
    สร้าง actual vs predicted plot และบันทึกเป็น artifact

    Args:
        y_true: actual values (array-like)
        y_pred: predicted values (array-like)
        title: plot title
        save_path: path บันทึกรูป
    """
    plt.figure(figsize=(14, 5))
    plt.plot(np.array(y_true)[:200], label="Actual", alpha=0.8)
    plt.plot(np.array(y_pred)[:200], label="Predicted", alpha=0.8)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Global Active Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_feature_importance(model, feature_names: list, save_path: str) -> None:
    """
    สร้าง feature importance plot (รองรับ XGBoost และ Random Forest)

    Args:
        model: trained model ที่มี feature_importances_
        feature_names: list ของ feature names
        save_path: path บันทึกรูป
    """
    if not hasattr(model, "feature_importances_"):
        return

    importance = model.feature_importances_
    indices = np.argsort(importance)[-20:]  # top 20 features

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Feature Importance (Top 20)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def build_model(algo: str, params: dict, seed: int):
    """
    สร้าง model instance จาก algorithm name และ params

    Args:
        algo: 'xgboost', 'random_forest', หรือ 'ridge'
        params: hyperparameters dict จาก config
        seed: random seed

    Returns:
        sklearn-compatible model
    """
    if algo == "xgboost":
        p = {k: v for k, v in params.items() if k not in ("early_stopping_rounds",)}
        return XGBRegressor(**p, random_state=seed, n_jobs=-1)
    elif algo == "random_forest":
        return RandomForestRegressor(**params, random_state=seed)
    elif algo == "ridge":
        return Ridge(**params)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def run_training(
    model_name: str,
    data_path: str,
    tracking_uri: str,
    experiment_file: str = None,
    version: int = 1,
    config_dir: str = None,
) -> dict:
    """
    Training pipeline หลัก — โหลดข้อมูล → feature engineering → train → evaluate → log MLflow

    Args:
        model_name: 'xgboost', 'random_forest', หรือ 'ridge'
        data_path: path ของไฟล์ข้อมูล (raw หรือ processed)
        tracking_uri: MLflow tracking URI
        experiment_file: YAML experiment override (optional)
        version: model version number
        config_dir: path ของ configs/ directory

    Returns:
        dict ของ metrics และ run_id

    Example:
        result = run_training('xgboost', '/content/drive/MyDrive/SmartEnergy2026/data/processed_hourly.csv',
                              tracking_uri='http://localhost:5000')
    """
    # โหลด config
    cfg = load_config(model_name, experiment_file, config_dir)
    seed = cfg["project"]["seed"]
    set_global_seed(seed)

    # ตั้งค่า MLflow
    setup_mlflow(tracking_uri, cfg["mlflow"]["experiment_name"])

    # โหลดข้อมูล
    print(f"[INFO] โหลดข้อมูลจาก: {data_path}")
    if data_path.endswith(".txt") or "household_power" in data_path:
        df_raw = load_raw_data(data_path)
        df = resample_hourly(df_raw)
    else:
        df = load_processed_data(data_path)

    # Feature engineering
    X, y = prepare_features(df, cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"]
    )

    # Scale features (บันทึก scaler params ใน MLflow)
    X_train_s, scaler = fit_scaler(X_train)
    X_val_s = apply_scaler(X_val, scaler)
    X_test_s = apply_scaler(X_test, scaler)

    # สร้างชื่อ run
    algo_short = {"xgboost": "xgb", "random_forest": "rf", "ridge": "ridge"}[model_name]
    run_name = make_run_name(algo_short, version)
    data_hash = hash_file(data_path) if Path(data_path).exists() else "unknown"

    # Enable autolog
    if model_name == "xgboost":
        mlflow.xgboost.autolog(log_models=True, log_datasets=False)
    else:
        mlflow.sklearn.autolog(log_models=True, log_datasets=False)

    with mlflow.start_run(run_name=run_name):
        # Log tags
        mlflow.set_tags({
            "algo": model_name,
            "data_version": data_hash,
            "resample_type": cfg["data"]["resample_period"],
            "seed": seed,
        })

        # Log scaler params
        log_scaler_params(scaler)

        # Build และ train model
        params = cfg["params"]
        model = build_model(model_name, params, seed)

        print(f"[INFO] เริ่มเทรน {model_name}...")
        if model_name == "xgboost":
            es_rounds = cfg["params"].get("early_stopping_rounds", 30)
            model.fit(
                X_train_s, y_train,
                eval_set=[(X_val_s, y_val)],
                early_stopping_rounds=es_rounds,
                verbose=50,
            )
        else:
            model.fit(X_train_s, y_train)

        # Evaluate
        train_metrics = compute_metrics(y_train, model.predict(X_train_s))
        val_metrics = compute_metrics(y_val, model.predict(X_val_s))
        test_metrics = compute_metrics(y_test, model.predict(X_test_s))

        # Log metrics
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # แสดงผลลัพธ์ตาราง (ตาม training-monitor)
        print("\n" + "="*60)
        print(f"{'Metric':<12} {'Train':>10} {'Val':>10} {'Test':>10}")
        print("-"*60)
        for k in ["rmse", "mae", "r2"]:
            print(f"{k.upper():<12} {train_metrics[k]:>10.4f} {val_metrics[k]:>10.4f} {test_metrics[k]:>10.4f}")
        print("="*60)

        # ตรวจสอบ Production threshold
        rmse_threshold = cfg["baseline"]["rmse_threshold"]
        if test_metrics["rmse"] < rmse_threshold:
            print(f"[PASS] RMSE {test_metrics['rmse']:.4f} < threshold {rmse_threshold} → สามารถ Register ได้")
        else:
            print(f"[FAIL] RMSE {test_metrics['rmse']:.4f} >= threshold {rmse_threshold} → ยังไม่ผ่านเกณฑ์")

        # Save artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            # actual vs predicted plot
            pred_test = model.predict(X_test_s)
            plot_path = os.path.join(tmpdir, "actual_vs_pred_plot.png")
            plot_actual_vs_pred(y_test, pred_test, f"{run_name} — Test Set", plot_path)
            mlflow.log_artifact(plot_path)

            # feature importance plot
            fi_path = os.path.join(tmpdir, "feature_importance.png")
            plot_feature_importance(model, list(X.columns), fi_path)
            if os.path.exists(fi_path):
                mlflow.log_artifact(fi_path)

            # pip freeze
            log_pip_freeze(tmpdir)

        run_id = mlflow.active_run().info.run_id
        print(f"\n[INFO] Run ID: {run_id}")

    return {
        "run_id": run_id,
        "run_name": run_name,
        "test_metrics": test_metrics,
        "passed_threshold": test_metrics["rmse"] < rmse_threshold,
    }
