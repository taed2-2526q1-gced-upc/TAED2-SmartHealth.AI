# src/api/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, List, Optional


class ObesityInput(BaseModel):
    Gender: Literal["Female", "Male", "Other"]
    Age: int = Field(..., ge=5, le=100)
    Height: float = Field(..., gt=0, le=2.5)
    Weight: float = Field(..., gt=0, le=350)
    family_history_with_overweight: Literal["yes", "no"]
    FAVC: Literal["yes", "no"]
    FCVC: float = Field(..., ge=0, le=3)
    NCP: float = Field(..., ge=0, le=4)
    CAEC: Literal["no", "Sometimes", "Frequently", "Always"]
    SMOKE: Literal["yes", "no"]
    CH2O: float = Field(..., ge=0, le=3)
    SCC: Literal["yes", "no"]
    FAF: float = Field(..., ge=0, le=3)
    TUE: float = Field(..., ge=0, le=2)
    CALC: Literal["no", "Sometimes", "Frequently", "Always"]
    MTRANS: Literal["Automobile", "Bike", "Motorbike", "Walking", "Public_Transportation"]


class PredictionRequest(BaseModel):
    """Liste af inputs – matcher lærerens “reviews: list[Review]” idé."""
    items: List[ObesityInput]


class PredictionResponse(BaseModel):
    """Ekkó input + modelens label (+ optional score)."""
    item: ObesityInput
    label: str
    score: Optional[float] = None  # hvis modellen har predict_proba
