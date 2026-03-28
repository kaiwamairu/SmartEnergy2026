# -*- coding: utf-8 -*-
"""
export_model.py — Export Production model from MLflow -> model_registry/
Run: python scripts/export_model.py

Use after P3 is done so the API can load the model without MLflow server.
"""
import json
import joblib
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
TRACKING_URI  = "sqlite:///G:/My Drive/SmartEnergy2026/runs/mlruns.db"
PROCESSED_FILE = "G:/My Drive/SmartEnergy2026/data/processed_hourly.csv"
REGISTRY_DIR  = Path("model_registry")
CONFIG_DIR    = Path("configs")
BEST_PARAMS_FILE = "G:/My Drive/SmartEnergy2026/runs/best_xgb_params.json"

# ─── Setup ────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()

REGISTRY_DIR.mkdir(exist_ok=True)

# ─── หา Production model ──────────────────────────────────────────────────────
print("Searching for Production model...")
versions = client.search_model_versions("name='SmartEnergyPredictor'")
prod = next((v for v in versions if v.current_stage == "Production"), None)

if prod is None:
    # fallback: เลือก RMSE ต่ำสุด
    best_rmse = float("inf")
    for v in versions:
        rmse = client.get_run(v.run_id).data.metrics.get("test_rmse", float("inf"))
        if rmse < best_rmse:
            best_rmse = rmse
            prod = v
    print(f"[WARN] No Production stage found — using lowest RMSE (v{prod.version})")

prod_run  = client.get_run(prod.run_id)
prod_algo = prod_run.data.tags.get("algo", "unknown")
prod_rmse = prod_run.data.metrics.get("test_rmse", 0)
print(f"Production model: {prod_algo} v{prod.version} | RMSE: {prod_rmse:.4f}")

# ─── โหลด data และ build model ────────────────────────────────────────────────
from src.utils.config_loader import load_config
from src.utils.seed_utils import set_global_seed
from src.data.data_loader import load_processed_data
from src.data.preprocessor import prepare_features, time_series_split, fit_scaler, apply_scaler

algo_name = "xgboost" if "xgboost" in prod_algo else "ridge"
cfg = load_config(algo_name, config_dir=str(CONFIG_DIR))
set_global_seed(cfg["project"]["seed"])

print("Loading data and building features...")
df = load_processed_data(PROCESSED_FILE)
X, y = prepare_features(df, cfg)
X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(
    X, y, cfg["data"]["test_size"], cfg["data"]["val_size"]
)
X_train_s, scaler = fit_scaler(X_train)

# ─── Train model ──────────────────────────────────────────────────────────────
print(f"Training {prod_algo}...")
if "xgboost" in prod_algo:
    from xgboost import XGBRegressor
    with open(BEST_PARAMS_FILE) as f:
        bp = json.load(f)["best_params"]
    bp.update({"objective": "reg:squarederror", "n_jobs": -1,
               "verbosity": 0, "early_stopping_rounds": 30})
    model = XGBRegressor(**bp, random_state=cfg["project"]["seed"])
    X_val_s = apply_scaler(X_val, scaler)
    model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)

elif "ridge" in prod_algo:
    from sklearn.linear_model import Ridge
    model = Ridge(**cfg["params"])
    model.fit(X_train_s, y_train)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import mean_squared_error
X_test_s = apply_scaler(X_test, scaler)
test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_s)))
print(f"Test RMSE: {test_rmse:.4f}")

# ─── Save to model_registry/ ──────────────────────────────────────────────────
joblib.dump(model,  REGISTRY_DIR / "model.joblib")
joblib.dump(scaler, REGISTRY_DIR / "scaler.joblib")

metadata = {
    "algo":       prod_algo,
    "algo_name":  algo_name,
    "version":    prod.version,
    "run_id":     prod.run_id,
    "test_rmse":  round(test_rmse, 6),
    "features":   list(X.columns),
}
with open(REGISTRY_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nExported to {REGISTRY_DIR}/")
print(f"   model.joblib   -- {(REGISTRY_DIR / 'model.joblib').stat().st_size // 1024} KB")
print(f"   scaler.joblib  -- {(REGISTRY_DIR / 'scaler.joblib').stat().st_size} bytes")
print(f"   metadata.json  -- {prod_algo} v{prod.version}")
print(f"\nStart API with:")
print(f"   uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload")
