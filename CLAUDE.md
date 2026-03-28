# CLAUDE.md — AI / RL Project Template

> เปลี่ยนค่าใน {braces} ให้ตรงกับโปรเจคจริง แล้วลบ comment คำอธิบายออก

## Project

- **ชื่อ**: {Smart Home Energy Predictor 2026 (Green-AI)}
- **เป้าหมาย**: {ทำนายการใช้ไฟฟ้าล่วงหน้าในแต่ละชั่วโมงเพื่อจัดการพลังงานในบ้านอัจฉริยะอย่างมีประสิทธิภาพ}
- **Environment**: {Individual Household Electric Power Consumption (UCI Dataset - Hourly Resampled)}
- **Algorithms**: {XGBoost Regressor, Random Forest, Ridge Regression (Baseline)}

## Tech Stack

- **Language**: {Python 3.10+}
- **Framework**: {Scikit-learn 1.4+, XGBoost 2.0+, Pandas}
- **Simulator**: {N/A (Data-driven Forecasting)}
- **CUDA**: {12.x (บน Google Colab Pro)}
- **MLOps**: {MLflow} — project name: `{smart-energy-optimizer-2026}`

## Conventions

- **Comment language**: {ภาษาไทย / English}
- **Run ID format**: `{algo}_v{version}_{timestamp}`
- **Branch strategy**: feature branch → PR → develop → main
- **Config**: YAML only — ห้าม hardcode hyperparameters

## Paths

```
{base_path เช่น /content/drive/MyDrive/SmartEnergy2026}/
├── runs/           ← MLflow tracking database (mlruns.db)
├── data/           ← raw & processed csv files (DVC tracked)
├── model_registry/ ← registered best models
└── shared/         ← Colab notebooks สำหรับ EDA และ Testing
```

## Team

| ชื่อ | บทบาท | Branch หลัก |
|------|--------|-------------|
| {Ryu} | {Lead Data Scientist (XGBoost & MLflow)} | `feature/{xgb-pipeline}` |
| {Ryu} | {ML Engineer (Data Prep & Baseline)} | `feature/{data-engineering}` |

## Key Commands
# ติดตั้ง dependencies และตั้งค่า MLflow
pip install mlflow pyngrok xgboost pandas scikit-learn

```bash
# รันการเทรนผ่าน script (สมมติว่าใช้ Makefile)
make train-xgboost           # เทรน XGBoost พร้อมบันทึกลง MLflow
make train-rf                # เทรน Random Forest
make search-hparams N=50     # ใช้ Optuna หาค่า Hyperparameters ที่ดีที่สุด
make ui-launch               # เปิด MLflow UI ผ่าน localtunnel
```

## Current Status

[x] P0 — Foundation: ตั้งค่า Google Drive, ติดตั้ง MLflow, เตรียม Dataset เบื้องต้น

[ ] P1 — Quality: สร้างโมเดล Baseline, เชื่อมต่อ MLflow Autologging, เก็บ Artifacts

[ ] P2 — Scale: เปรียบเทียบหลายโมเดล, ทำ Hyperparameter Tuning, วิเคราะห์ Feature Importance

[ ] P3 — Production: ลงทะเบียนโมเดลใน Registry, ทำ Inference API ขนาดเล็ก

## Notes

- ข้อมูลถูก Resample เป็นรายชั่วโมง (H) เพื่อลด Noise และประหยัด Memory ในการเทรน

- ใช้ Global_active_power เป็นเป้าหมาย (Target) หลักในการทำนาย

- เกณฑ์การตัดสิน: โมเดลต้องมี RMSE < 0.15 ถึงจะอนุญาตให้ทำการ Register เข้าสู่ Production stage
