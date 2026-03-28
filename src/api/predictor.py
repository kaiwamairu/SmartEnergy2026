"""
predictor.py — โหลด Production model และทำ inference
รองรับทั้ง Ridge และ XGBoost โดยอ่าน algo จาก metadata
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.data.preprocessor import (
    create_lag_features,
    create_rolling_features,
    create_time_features,
    apply_scaler,
)
from src.utils.config_loader import load_config


class EnergyPredictor:
    """
    Inference wrapper สำหรับ Smart Home Energy Predictor
    โหลดจาก model_registry/ และรับ raw hourly data

    Example:
        predictor = EnergyPredictor.load('model_registry/')
        result = predictor.predict(records=[
            {"datetime": "2009-01-01 10:00:00", "Global_active_power": 1.2, ...}
        ])
    """

    def __init__(self, model, scaler: StandardScaler, cfg: dict, algo: str, version: str):
        self.model   = model
        self.scaler  = scaler
        self.cfg     = cfg
        self.algo    = algo
        self.version = version

    @classmethod
    def load(cls, registry_dir: str = "model_registry", config_dir: str = "configs"):
        """
        โหลด model, scaler, metadata จาก registry_dir

        Args:
            registry_dir: path ของ model_registry/
            config_dir: path ของ configs/

        Returns:
            EnergyPredictor instance พร้อมใช้งาน
        """
        registry_path = Path(registry_dir)

        with open(registry_path / "metadata.json") as f:
            meta = json.load(f)

        model  = joblib.load(registry_path / "model.joblib")
        scaler = joblib.load(registry_path / "scaler.joblib")
        cfg    = load_config(meta["algo_name"], config_dir=config_dir)

        return cls(model, scaler, cfg, meta["algo"], str(meta["version"]))

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """แปลง raw hourly DataFrame → scaled feature matrix"""
        target_col      = self.cfg["data"]["target_column"]
        lag_features    = self.cfg["features"]["lag_features"]
        rolling_windows = self.cfg["features"]["rolling_windows"]

        df = create_time_features(df.copy())
        df = create_lag_features(df, target_col, lag_features)
        df = create_rolling_features(df, target_col, rolling_windows)
        df = df.dropna()

        feature_cols = [c for c in df.columns if c != target_col]
        return apply_scaler(df[feature_cols], self.scaler), df.index

    def predict(self, records: list) -> list:
        """
        ทำนายจาก list of records (JSON-friendly)

        Args:
            records: list of dict เช่น
                [{"datetime": "2009-01-01 10:00:00",
                  "Global_active_power": 1.2,
                  "Global_reactive_power": 0.1, ...}]
                ต้องมีอย่างน้อย 25 records (24h lag buffer + 1)

        Returns:
            list of {"datetime": ..., "predicted_kw": ...}
        """
        df = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")

        X_scaled, index = self._build_features(df)
        preds = self.model.predict(X_scaled)

        return [
            {"datetime": str(idx), "predicted_kw": round(float(p), 4)}
            for idx, p in zip(index, preds)
        ]

    def predict_next(self, records: list) -> dict:
        """
        ทำนาย 1 ชั่วโมงถัดไปจาก history

        Args:
            records: list of dict ย้อนหลัง 24+ ชั่วโมง

        Returns:
            {"next_datetime": ..., "predicted_kw": ..., "algo": ..., "version": ...}
        """
        results = self.predict(records)
        last = results[-1]
        next_dt = pd.Timestamp(last["datetime"]) + pd.Timedelta(hours=1)
        return {
            "next_datetime": str(next_dt),
            "predicted_kw":  last["predicted_kw"],
            "algo":          self.algo,
            "version":       self.version,
        }
