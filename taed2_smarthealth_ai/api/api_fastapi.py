# taed2_smarthealth_ai/api.py
from fastapi import FastAPI, HTTPException, Body
from pathlib import Path
import pandas as pd
import joblib, yaml, json

app = FastAPI(
    title="SmartHealth-AI API",
    version="1.0.0",
    description="FastAPI server for obesity classification model"
)

# Globals
model = None
feature_order = []
class_labels = []

@app.on_event("startup")
def load_artifacts():
    """Load model and metadata on startup"""
    global model, feature_order, class_labels

    ROOT = Path(__file__).resolve().parents[2]
    params_path = ROOT / "params.yaml"
    if not params_path.exists():
        raise RuntimeError(f"params.yaml not found: {params_path}")

    params = yaml.safe_load(open(params_path, "r", encoding="utf-8"))
    model_dir = ROOT / params["train"]["model_dir"]
    model_path = model_dir / "model.joblib"
    feat_path  = model_dir / "features.json"
    cls_path   = model_dir / "classes.json"

    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")
    model = joblib.load(model_path)

    feature_order = json.load(open(feat_path, "r", encoding="utf-8"))["feature_order"]
    if cls_path.exists():
        class_labels = json.load(open(cls_path, "r", encoding="utf-8"))["classes"]
    else:
        class_labels = list(getattr(model, "classes_", []))

@app.get("/")
def root():
    return {"message": "SmartHealth-AI API running. See /docs or /predict/schema."}

@app.get("/healthz")
def healthz():
    ok = (model is not None) and len(feature_order) > 0
    return {"status": "ok" if ok else "not_ready", "model_loaded": ok, "n_features": len(feature_order)}

@app.get("/predict/schema")
def schema():
    if not feature_order:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"feature_order": feature_order, "n_features": len(feature_order)}

@app.post("/predict")
def predict(
    record: dict = Body(
        ...,
        example={
            "Gender": 1,
            "Age": 21.0,
            "Height": 1.62,
            "Weight": 64.0,
            "family_history_with_overweight": 1,
            "FAVC": 0,
            "FCVC": 2.0,
            "NCP": 3.0,
            "CAEC": 1,
            "SMOKE": 0,
            "CH2O": 2.0,
            "SCC": 0,
            "FAF": 0.0,
            "TUE": 1.0,
            "CALC": 0,
            "MTRANS_automobile": 0,
            "MTRANS_bike": 0,
            "MTRANS_motorbike": 0,
            "MTRANS_walking": 0
        }
    )
):
    """Predict obesity class for one input record"""
    if not feature_order:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Check missing features
    missing = [c for c in feature_order if c not in record]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    # Build dataframe in correct order
    row = {k: record[k] for k in feature_order}
    df = pd.DataFrame([row])
    try:
        df = df.apply(pd.to_numeric)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Type error in input: {e}")

    # Predict
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            labels = class_labels or list(getattr(model, "classes_", []))
            top_i = int(proba.argmax())
            return {
                "label": str(labels[top_i]),
                "confidence": float(proba[top_i]),
                "probabilities": {str(labels[i]): float(proba[i]) for i in range(len(labels))}
            }
        pred = model.predict(df)[0]
        return {"label": str(pred), "confidence": 1.0, "probabilities": {str(pred): 1.0}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
