"""
data_loader.py — โหลดและ resample UCI Household Power Consumption dataset
Dataset: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
"""
import pandas as pd
from pathlib import Path


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    โหลด raw UCI dataset (semicolon-separated, มี '?' เป็น missing values)

    Args:
        file_path: path ของ household_power_consumption.txt หรือ .csv

    Returns:
        DataFrame พร้อม datetime index

    Example:
        df = load_raw_data('/content/drive/MyDrive/SmartEnergy2026/data/household_power_consumption.txt')
    """
    df = pd.read_csv(
        file_path,
        sep=";",
        na_values=["?"],
        low_memory=False,
        parse_dates={"datetime": ["Date", "Time"]},
        dayfirst=True,
        index_col="datetime",
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample ข้อมูลจาก 1-minute → hourly เพื่อลด Noise และประหยัด Memory

    Args:
        df: DataFrame จาก load_raw_data()

    Returns:
        Hourly resampled DataFrame ที่ไม่มี NaN

    Example:
        df_hourly = resample_hourly(df_raw)
    """
    df_hourly = df.resample("H").mean()

    # ตรวจสอบ NaN หลัง resample (ตาม data-pipeline-audit)
    nan_count = df_hourly.isna().sum().sum()
    if nan_count > 0:
        print(f"[WARNING] พบ {nan_count} NaN values หลัง resample — ใช้ interpolation แก้ไข")
        df_hourly = df_hourly.interpolate(method="time").ffill().bfill()

    return df_hourly


def load_processed_data(file_path: str) -> pd.DataFrame:
    """
    โหลด processed CSV ที่ผ่านการ resample มาแล้ว

    Args:
        file_path: path ของ processed_hourly.csv

    Returns:
        DataFrame พร้อม datetime index
    """
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df
