# SKILL.md — Smart Home Energy Predictor 2026

> Reusable procedures สำหรับโปรเจค Energy Forecasting & ML Regression
> ใช้ร่วมกับ Claude.md ที่ระบุบริบทของ Smart Home Energy Predictor 2026

---

## Trigger

ใช้ SKILL นี้เมื่อโปรเจคเกี่ยวข้องกับ: Time-series forecasting, Energy consumption modeling, MLflow tracking, Hyperparameter tuning (Optuna), และ Model Deployment บน Google Colab Pro

---

## P0 — Foundation

### project-scaffold

เมื่อถูกขอให้สร้างโครงสร้างโปรเจค AI:

```
project-root/
├── configs/                # YAML config files เท่านั้น
│   ├── base.yaml           # ค่าพื้นฐาน: data_path, resample_rate (H/D), seed
│   ├── xgboost.yaml        # hparams เฉพาะ XGBoost
│   └── experiments/        # สเปคเฉพาะแต่ละ run (เช่น exp_winter_data.yaml)
├── src/
│   ├── data/               # data_loader, preprocessor (resampling, scaling)
│   ├── models/             # model definitions (XGB, Random Forest, etc.)
│   ├── pipelines/          # training & validation loop
│   └── utils/              # MLflow helpers, config loader, metrics calculation
├── notebooks/              # EDA.ipynb, Colab_Main.ipynb
├── tests/                  # unit tests สำหรับ data processing logic
├── scripts/                # setup_colab.sh, download_uci_data.sh
├── Makefile                # คำสั่งรัดกุมสำหรับ Colab (make train, make ui)
├── requirements.txt        # pinned: mlflow, xgboost, pandas, scikit-learn
├── .env.example            # สำหรับ NGROK_AUTH_TOKEN (ถ้าใช้ MLflow UI)
├── CLAUDE.md
└── README.md
```

### config-management

เมื่อถูกขอให้จัดการ config:
1. Hierarchy: base.yaml (Global) → {model}.yaml (Algo-specific) → experiments/{run}.yaml (Override)
2. Key Params: ต้องมี window_size, horizon, resample_period, และ test_size เสมอ
3. Strict Rule: ห้ามใส่ค่า learning_rate หรือ n_estimators ลงในโค้ด .py โดยตรง

### reproducibility

เมื่อถูกขอให้ setup reproducibility:
1. Seed Engine: ใช้ set_global_seed(seed) คุมทั้ง numpy, random, sklearn, และ xgboost (seed parameter)

2. Data Versioning: หากมีการกรองข้อมูล (Clean Outliers) ต้องเก็บ Hash ของไฟล์ข้อมูลไว้ใน MLflow Tag เสมอ

3. Environment: บันทึก pip freeze ลงใน MLflow Artifacts ทุกครั้งที่เทรนสำเร็จ


---

## P1 — Quality Gates

### pre-commit-setup

# mlflow-tracking-setup
1. Auto-logging: เปิด mlflow.xgboost.autolog() หรือ mlflow.sklearn.autolog() เป็นค่าเริ่มต้น
2. Custom Tags: บันทึก data_version และ resample_type (เช่น 'Hourly') เพื่อแยกแยะการทดลอง
3. Artifacts: เก็บไฟล์ feature_importance.png และ actual_vs_pred_plot.png ทุกครั้งหลังจบการเทรน

# data-pipeline-audit
1. Resampling Check: ตรวจสอบว่าไม่มีค่า NaN เกิดขึ้นหลังจากทำการ Resampling (เช่น ใช้ fillna หรือ interpolation)
2. Leakage Prevention: ตรวจสอบว่าไม่มีการใช้ข้อมูลในอนาคต (Future Data) มาทำนายอดีต (Time-series split เท่านั้น ห้าม Random split)
3. Scale Check: บันทึกพารามิเตอร์ของ Scaler (Mean/Std) ไว้ใน MLflow เพื่อใช้ตอนทำ Inference

### model-registry

เมื่อถูกขอให้จัดการ trained models:
1. `register()` — บันทึก run_id, metrics, config, tags, reward_version
2. `get_best(metric, algorithm)` — query model ที่ดีที่สุด
3. `compare(run_ids)` — เปรียบเทียบหลาย models
4. `promote(run_id, status)` — lifecycle: registered → staging → production → archived
5. Registration Criteria: โมเดลที่จะเข้าสู่ Staging ต้องมี RMSE ต่ำกว่า Baseline ที่ตั้งไว้ใน base.yaml
6. Transition: ใช้คำสั่ง mlflow.register_model() เมื่อโมเดลผ่านการทดสอบบน Test Set เท่านั้น

### data-versioning

เมื่อถูกขอให้ setup DVC:
1. `dvc init` + remote add (Google Drive / S3)
2. Track: environment assets, mesh, texture, large datasets
3. ไม่ track: configs (ใช้ Git), source code (ใช้ Git), training checkpoints (อยู่บน Drive)
4. Commit .dvc files เข้า Git
5. สร้าง .dvcignore

---

## P2 — Scale

### optuna-search

เมื่อถูกขอให้ search hyperparameters:
1. กำหนด search space แยกต่อ algorithm (ดูจาก Claude.md ว่าใช้ algorithm อะไร)
2. Sampler: TPESampler (ดีกว่า random), Pruner: MedianPruner (ตัด trial ที่แย่เร็ว)
3. Train แบบสั้น (10% ของ full training) ต่อ trial — ประหยัดเวลา
4. Save results: best_params.json, all_trials.csv, optimization_history.html, param_importance.html
5. Auto-generate experiment config จาก best params — พร้อมรันเลย
6. จำนวน trials แนะนำ: 50-100
7. Search Space: เน้น max_depth, learning_rate, subsample สำหรับ XGBoost
8. Pruning: ใช้ TimeSeriesSplit ของ Sklearn ร่วมกับ Optuna เพื่อป้องกัน Overfitting
9. Persistence: เซฟไฟล์ study.db (SQLite) ไว้ใน Google Drive เพื่อให้รันต่อได้ถ้า Colab ตัดการทำงาน

### domain-randomization

เมื่อถูกขอให้ทำ domain randomization:
1. สร้าง gymnasium Wrapper ที่ randomize ทุก reset()
2. Randomize 3 ประเภท: Visual (แสง, สี, กล้อง), Physics (friction, mass, damping), Position (object, target)
3. เพิ่ม noise ใน action (motor imprecision) + observation (sensor noise) ทุก step()
4. สร้าง preset levels: none / light / medium / heavy
5. Log DR parameters ใน info dict สำหรับ reproduce
6. ใช้ config จาก YAML — ไม่ hardcode

### distributed-training

เมื่อถูกขอให้ scale training:
1. `get_device_info()` — สำรวจ GPU ทั้งหมด
2. `auto_select_device()` — เลือก GPU ที่ว่างมากสุด
3. `create_vectorized_envs()` — รัน N environments ขนาน (แนะนำ 4-16 ต่อ GPU)
4. `estimate_training_time()` — ประมาณเวลา train
5. ใช้ ManiSkill3 GPU vectorization ก่อน (เร็วสุด), fallback เป็น SubprocVecEnv

### training-monitor

เมื่อถูกขอให้ monitor training:
1. ตรวจ: LR anomaly, gradient explosion, reward plateau, loss divergence, NaN/Inf, GPU memory leak
2. Alert levels: info / warning / critical
3. เรียก `monitor.on_step(step, metrics)` ทุก N steps
4. Save alerts ลง JSONL สำหรับดูย้อนหลัง
5. Critical alert → หยุด training อัตโนมัติ (optional)
6. `print_summary()` ตอนจบ training
7. Resource Watch: ตรวจสอบ RAM บน Colab Pro (โดยเฉพาะตอน Resampling ข้อมูลขนาดใหญ่)
8. Convergence: ตรวจสอบ Log ของ MLflow ว่า Loss เริ่มนิ่งหรือยัง หากมีอาการ Diverge ให้หยุดรันทันที
9. Alert: แสดงผลลัพธ์ RMSE และ MAE ในรูปแบบตารางเปรียบเทียบในขั้นตอนสุดท้าย


### makefile

เมื่อถูกขอให้สร้าง Makefile:
- `make help` — แสดงคำสั่งทั้งหมด
- `make setup` — ติดตั้ง dependencies + pre-commit
- `make train-{algo}` — train แต่ละ algorithm
- `make search-{algo}` — Optuna search
- `make eval` — evaluate model
- `make compare` — เปรียบเทียบ algorithms
- `make test` — รัน tests + coverage
- `make lint` / `make format` — code quality
- `make docker-build` / `make docker-train-{algo}`
- `make monitor` — เปิด Tensorboard
- Override ค่าได้: `make train-sac SEED=99`

---

## P3 — Production

### safety-constraints

เมื่อถูกขอให้เพิ่ม safety:
1. Force limits — กำหนด max force ที่ gripper ใช้ได้
2. Collision check — ตรวจชนวัตถุที่ไม่ควรชน
3. Workspace boundary — จำกัดพื้นที่ทำงานของหุ่นยนต์
4. Emergency stop — หยุดทันทีเมื่อเกิน threshold
5. Log ทุก violation สำหรับ audit
6. Portable Model: Export โมเดลเป็นรูปแบบ mlflow-model เพื่อให้เรียกใช้ผ่าน mlflow.pyfunc.load_model() ได้ง่าย
7. Preprocessing Wrapper: สร้าง Function ที่รับ Raw Data แล้วทำทุกขั้นตอน (Resample -> Scale -> Predict) ให้จบในที่เดียว


### deploy-strategy

เมื่อถูกขอให้ deploy model:
1. Export model เป็น format ที่ใช้ได้จริง (ONNX, TorchScript)
2. A/B model serving — switch ระหว่าง 2 models ได้
3. Rollback — กลับไปใช้ model เก่าได้ทันที
4. Sim-to-real gap test — ทดสอบใน sim ที่ randomize หนักก่อน deploy

---

## Comment Convention

- Comment ภาษาตามที่ระบุใน Claude.md (default: English)
- ทุก function ต้องมี docstring อธิบาย parameters + return + ตัวอย่าง
- ทุกไฟล์ต้องมี module-level comment อธิบายว่าไฟล์นี้ทำอะไร

---

## Anti-patterns (สิ่งที่ห้ามทำ)

1. ห้าม hardcode hyperparameters ใน source code — ใช้ config YAML เท่านั้น
2. ห้าม commit checkpoints / model weights เข้า Git — ใช้ DVC หรือ external storage
3. ห้าม commit .env / API keys — ใช้ .env.example เป็นตัวอย่าง
4. ห้ามแก้ reward function เก่า — สร้างเวอร์ชันใหม่ + log audit
5. ห้าม train โดยไม่ระบุ seed — ทุก run ต้อง reproducible
6. ห้ามสร้าง comparison plots ถ้าใช้ MLOps tool (W&B/MLflow) อยู่แล้ว — ซ้ำซ้อน
7. Random Shuffle: ห้ามใช้ train_test_split(shuffle=True) กับข้อมูลไฟฟ้า เพราะลำดับเวลาสำคัญมาก
8. Hardcoding Paths: ห้ามใช้ /content/drive/... ตรงๆ ในโค้ด ให้ดึงจาก config.yaml เท่านั้น
9. No Baseline: ห้ามเริ่มเทรนโมเดลซับซ้อนโดยไม่มีโมเดลพื้นฐาน (เช่น Moving Average หรือ Linear Regression) มาเปรียบเทียบ