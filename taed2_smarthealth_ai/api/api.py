# src/api/api.py
from contextlib import asynccontextmanager
from typing import List
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from taed2_smarthealth_ai.api.schema import PredictionRequest, PredictionResponse
from data.processed.config import MODEL_PATH

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise ValueError(f"Model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    yield

app = FastAPI(title="Obesity Classification API", version="1.0.0", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Welcome to the Obesity app!"}

@app.post("/prediction", response_model=List[PredictionResponse])
def predict_items(requests: PredictionRequest) -> List[PredictionResponse]:
    try:
        X = pd.DataFrame([it.model_dump() for it in requests.items])
        y = model.predict(X)
        try:
            proba = model.predict_proba(X)
        except Exception:
            proba = None
        out = []
        for i, item in enumerate(requests.items):
            score = float(max(proba[i])) if proba is not None else None
            out.append(PredictionResponse(item=item, label=str(y[i]), score=score))
        return out
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Bad Request: {ve}") from ve
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {ex}") from ex
