"""
preprocessor.py — Feature engineering และ train/test split สำหรับ Time-series
ป้องกัน Data Leakage โดยใช้ TimeSeriesSplit เท่านั้น (ห้าม random split)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List


def create_lag_features(df: pd.DataFrame, target_col: str, lag_periods: List[int]) -> pd.DataFrame:
    """
    สร้าง lag features จาก target column

    Args:
        df: DataFrame พร้อม datetime index
        target_col: ชื่อ column เป้าหมาย เช่น 'Global_active_power'
        lag_periods: list ของ lag เช่น [1, 2, 3, 6, 12, 24]

    Returns:
        DataFrame พร้อม lag features ที่เพิ่มมา
    """
    df = df.copy()
    for lag in lag_periods:
        df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str, windows: List[int]) -> pd.DataFrame:
    """
    สร้าง rolling mean และ rolling std features

    Args:
        df: DataFrame พร้อม lag features
        target_col: ชื่อ column เป้าหมาย
        windows: list ของ window sizes เช่น [3, 6, 12, 24]

    Returns:
        DataFrame พร้อม rolling features
    """
    df = df.copy()
    for w in windows:
        df[f"{target_col}_roll_mean_{w}h"] = df[target_col].shift(1).rolling(window=w).mean()
        df[f"{target_col}_roll_std_{w}h"] = df[target_col].shift(1).rolling(window=w).std()
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    สร้าง time-based features จาก datetime index

    Args:
        df: DataFrame พร้อม DatetimeIndex

    Returns:
        DataFrame พร้อม hour, day_of_week, month, is_weekend features
    """
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df


def prepare_features(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Pipeline หลักสำหรับ feature engineering ทั้งหมด

    Args:
        df: hourly resampled DataFrame
        cfg: config dict จาก load_config()

    Returns:
        (X, y) — feature matrix และ target series

    Example:
        X, y = prepare_features(df_hourly, cfg)
    """
    target_col = cfg["data"]["target_column"]
    lag_features = cfg["features"]["lag_features"]
    rolling_windows = cfg["features"]["rolling_windows"]

    df = create_time_features(df)
    df = create_lag_features(df, target_col, lag_features)
    df = create_rolling_features(df, target_col, rolling_windows)

    # ลบ rows ที่มี NaN (เกิดจาก lag features ต้น series)
    df = df.dropna()

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    return X, y


def time_series_split(X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float):
    """
    แบ่ง train/val/test โดยรักษาลำดับเวลา (ห้าม shuffle)

    Args:
        X: feature matrix
        y: target series
        test_size: สัดส่วน test set เช่น 0.2
        val_size: สัดส่วน validation set เช่น 0.1

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n = len(X)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))

    X_train = X.iloc[:val_idx]
    X_val = X.iloc[val_idx:test_idx]
    X_test = X.iloc[test_idx:]

    y_train = y.iloc[:val_idx]
    y_val = y.iloc[val_idx:test_idx]
    y_test = y.iloc[test_idx:]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler บน train set เท่านั้น

    Args:
        X_train: training feature matrix

    Returns:
        (X_train_scaled, fitted_scaler)
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    return X_scaled, scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Apply fitted scaler ไปยัง val หรือ test set

    Args:
        X: feature matrix
        scaler: fitted StandardScaler จาก fit_scaler()

    Returns:
        Scaled DataFrame
    """
    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
