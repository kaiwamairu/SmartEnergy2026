"""
app.py — FastAPI REST API สำหรับ Smart Home Energy Predictor
รัน: uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import traceback

from src.api.predictor import EnergyPredictor

app = FastAPI(
    title="Smart Home Energy Predictor API",
    description="ทำนายการใช้ไฟฟ้ารายชั่วโมง ด้วย Production model จาก MLflow",
    version="1.0.0",
)

# โหลด model ครั้งเดียวตอนเริ่ม server
predictor: Optional[EnergyPredictor] = None

@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = EnergyPredictor.load(
            registry_dir="model_registry",
            config_dir="configs",
        )
        print(f"[INFO] Loaded model: {predictor.algo} v{predictor.version}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")


# ─── Request / Response schemas ───────────────────────────────────────────────

class PowerRecord(BaseModel):
    datetime: str = Field(..., example="2009-01-01 10:00:00")
    Global_active_power: float = Field(..., example=1.2)
    Global_reactive_power: Optional[float] = Field(0.0, example=0.1)
    Voltage: Optional[float] = Field(240.0, example=240.0)
    Global_intensity: Optional[float] = Field(5.0, example=5.0)
    Sub_metering_1: Optional[float] = Field(0.0, example=0.0)
    Sub_metering_2: Optional[float] = Field(0.0, example=0.0)
    Sub_metering_3: Optional[float] = Field(0.0, example=0.0)

class PredictRequest(BaseModel):
    records: List[PowerRecord] = Field(
        ...,
        min_length=25,
        description="ข้อมูลย้อนหลังอย่างน้อย 25 ชั่วโมง (24h lag + 1)"
    )

class PredictResponse(BaseModel):
    datetime: str
    predicted_kw: float

class NextHourResponse(BaseModel):
    next_datetime: str
    predicted_kw: float
    algo: str
    version: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """ตรวจสอบสถานะ API"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model": predictor.algo,
        "version": predictor.version,
    }


@app.get("/model-info")
def model_info():
    """ดูข้อมูล Production model ที่โหลดอยู่"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "algo":    predictor.algo,
        "version": predictor.version,
        "target":  predictor.cfg["data"]["target_column"],
        "features": {
            "lag":     predictor.cfg["features"]["lag_features"],
            "rolling": predictor.cfg["features"]["rolling_windows"],
        },
    }


@app.post("/predict", response_model=List[PredictResponse])
def predict(req: PredictRequest):
    """
    ทำนายการใช้ไฟฟ้าทุก record ที่ส่งมา

    ส่ง hourly records อย่างน้อย 25 ชั่วโมง
    คืนค่าเป็น list of predicted_kw พร้อม datetime
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        records = [r.model_dump() for r in req.records]
        results = predictor.predict(records)
        return results
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/next-hour", response_model=NextHourResponse)
def predict_next_hour(req: PredictRequest):
    """
    ทำนาย 1 ชั่วโมงถัดไปจาก history ที่ส่งมา

    ส่ง hourly records ย้อนหลัง 24+ ชั่วโมง
    คืนค่าเป็น next_datetime และ predicted_kw
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        records = [r.model_dump() for r in req.records]
        result = predictor.predict_next(records)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
