from fastapi import FastAPI, HTTPException, Body
import joblib, yaml, json
import pandas as pd
from pathlib import Path
import numpy as np

def _to_scalar(v, default=0):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(v)
        if arr.size == 0:
            return default
        x = arr.ravel()[0]
        return x.item() if isinstance(x, np.generic) else x
    return v

def generate_advice(vals: dict) -> dict:
    tips = []
    # genetics
    if int(vals.get("family_history_with_overweight", 0)) == 1:
        tips.append("Family history increases risk. Monitor weight, keep meals balanced, and stay active.")
    # diet
    if int(vals.get("FAVC", 0)) == 1:
        tips.append("Cut down on energy-dense foods and sugary drinks.")
    if float(vals.get("FCVC", 0.0)) < 2:
        tips.append("Add more vegetables and fiber to meals (aim for 2–3 servings/day).")
    if int(vals.get("CAEC", 0)) >= 2:
        tips.append("Reduce snacking; plan protein- and fiber-rich meals for satiety.")
    if float(vals.get("NCP", 0.0)) < 3:
        tips.append("Keep a regular meal pattern (~3 balanced meals/day).")
    if int(vals.get("CALC", 0)) >= 2:
        tips.append("Lower alcohol frequency or portion size.")
    # activity
    if float(vals.get("FAF", 0.0)) < 2:
        tips.append("Build up to ≥150 min/week moderate activity (e.g., brisk walking).")
    else:
        tips.append("Maintain activity and add 2×/week strength training.")
    if float(vals.get("TUE", 0.0)) >= 2:
        tips.append("Break up screen time with short standing or walking breaks each hour.")
    if int(vals.get("MTRANS_walking", 0)) == 0 and int(vals.get("MTRANS_bike", 0)) == 0 and int(vals.get("MTRANS_automobile", 0)) == 1:
        tips.append("Swap short car trips for walking or cycling when possible.")
    # hydration
    if float(vals.get("CH2O", 0.0)) < 2:
        tips.append("Increase water intake across the day.")
    # smoking
    if int(vals.get("SMOKE", 0)) == 1:
        tips.append("Consider a quit plan; it supports cardiometabolic health.")

    return {
        "advice": list(dict.fromkeys(tips)),  # de-dup
        "note": "Model-based suggestions; not a medical diagnosis."
    }

params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
model_dir = Path(params["train"]["model_dir"])
model_path = model_dir / "model.joblib"
feat_path  = model_dir / "features.json"
cls_path   = model_dir / "classes.json"

app = FastAPI(
    title="SmartHealth-AI API",
    version="1.0.0",
    description="Simple FastAPI server for obesity model (RandomForest)",
)

@app.get("/")
def root():
    return {"message": "SmartHealth-AI API is running. See /docs, /healthz, /predict/schema."}

model = joblib.load(model_path)
feature_order = json.load(open(feat_path, "r", encoding="utf-8"))["feature_order"]
try:
    class_labels = json.load(open(cls_path, "r", encoding="utf-8"))["classes"]
except FileNotFoundError:
    class_labels = getattr(model, "classes_", [])

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": True, "n_features": len(feature_order)}

@app.get("/predict/schema")
def schema():
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
            "MTRANS_walking": 0,
        },
    )
):
    missing = [c for c in feature_order if c not in record]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    clean = {k: _to_scalar(record[k]) for k in feature_order}
    df = pd.DataFrame([clean])
    try:
        df = df.apply(pd.to_numeric)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Type error in input: {e}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        labels = list(class_labels) if len(class_labels) else list(getattr(model, "classes_", []))
        top_i = int(np.argmax(proba))

        resp = {
            "label": str(labels[top_i]),
            "confidence": float(proba[top_i]),
            "probabilities": {str(labels[i]): float(proba[i]) for i in range(len(labels))},
        }

        resp["personalized_advice"] = generate_advice(clean)
        return resp

    pred = model.predict(df)[0]
    return {"label": str(pred), "confidence": 1.0, "probabilities": {str(pred): 1.0}}
