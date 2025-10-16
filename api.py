# api.py (simpel, men robust mod feature-rækkefølge)
from fastapi import FastAPI, HTTPException, Body
import joblib, yaml, json
import pandas as pd
from pathlib import Path

# reading the model 
params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
model_dir = Path(params["train"]["model_dir"])   # samme key som i din træning
model_path = model_dir / "model.joblib"
feat_path  = model_dir / "features.json"
cls_path   = model_dir / "classes.json"


app = FastAPI(
    title="SmartHealth-AI API",
    version="1.0.0",
    description="Simple FastAPI server for obesity model (RandomForest)"
)


@app.get("/")
def root():
    return {"message": "SmartHealth-AI API is running. See /docs, /healthz, /predict/schema."}

# load model and artifacts
model = joblib.load(model_path)
feature_order = json.load(open(feat_path))["feature_order"]
try:
    class_labels = json.load(open(cls_path))["classes"]
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
            "MTRANS_walking": 0
        }
    )
):
    # check for missing features
    missing = [c for c in feature_order if c not in record]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    # create DataFrame in correct order
    df = pd.DataFrame([{k: record[k] for k in feature_order}])
    try:
        df = df.apply(pd.to_numeric)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Type error in input: {e}")

    # predict + proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        labels = list(class_labels) if class_labels else list(getattr(model, "classes_", []))
        top_i = int(proba.argmax())
        return {
            "label": str(labels[top_i]),
            "confidence": float(proba[top_i]),
            "probabilities": {str(labels[i]): float(proba[i]) for i in range(len(labels))}
        }
    else:
        pred = model.predict(df)[0]
        return {"label": str(pred), "confidence": 1.0, "probabilities": {str(pred): 1.0}}
