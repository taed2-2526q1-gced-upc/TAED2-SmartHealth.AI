import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

# Importér modulet, så vi kan ændre modulvariabler (fx GOOGLE_API_KEY)
from taed2_smarthealth_ai.api import api as api_module

client = TestClient(api_module.app)

# Eksempelkrop, der matcher dine features
EXAMPLE_INPUT = {
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
}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "SmartHealth-AI API" in body["message"]


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert isinstance(body["n_features"], int)
    # Feltet findes altid; True/False afhænger af miljøet
    assert "gemini_api_configured" in body


def test_predict_missing_fields():
    # Bevidst ufuldstændig payload -> skal give 400
    r = client.post("/predict", json={"Age": 25, "Weight": 70})
    assert r.status_code == 400
    assert "Missing features" in r.json()["detail"]


def test_predict_valid_request():
    r = client.post("/predict", json=EXAMPLE_INPUT)
    assert r.status_code == 200
    body = r.json()

    # Strukturtjek
    assert "label" in body
    assert "confidence" in body
    assert "probabilities" in body
    assert isinstance(body["probabilities"], dict)

    # Værdier i [0,1]
    assert 0.0 <= body["confidence"] <= 1.0
    for p in body["probabilities"].values():
        assert 0.0 <= float(p) <= 1.0

    # Sandsynligheder ~ summerer til 1
    total_p = sum(float(v) for v in body["probabilities"].values())
    assert total_p == pytest.approx(1.0, rel=1e-3, abs=1e-3)

    # Label bør være blandt kendte klasser (hvis tilgængelige)
    if getattr(api_module, "class_labels", None):
        assert body["label"] in [str(c) for c in api_module.class_labels]


def test_generate_advice_returns_503_when_no_key(monkeypatch):
    # Tving API'en til at opføre sig som om der ikke er en nøgle
    monkeypatch.setattr(api_module, "GOOGLE_API_KEY", None, raising=False)

    payload = {
        "user_inputs": {**EXAMPLE_INPUT},
        "prediction": {"label": "Normal_Weight", "confidence": 0.85},
    }
    r = client.post("/generate-advice", json=payload)
    assert r.status_code == 503
    assert "GOOGLE_API_KEY not configured" in r.json()["detail"]


def test_predict_with_advice_monkeypatched_llm(monkeypatch):
    """
    Tester /predict-with-advice ved at:
    - sætte en 'falsk' GOOGLE_API_KEY (forbi 503-check)
    - mocke google.generativeai.GenerativeModel til at returnere forudbestemt JSON
    """
    # 1) Bypass nøgle-check
    monkeypatch.setattr(api_module, "GOOGLE_API_KEY", "test-key", raising=False)

    # 2) Mock GenerativeModel
    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate_content(self, prompt):
            # Returnér noget, der ligner det rigtige Gemini-svar-objekt
            fake_json = {
                "advice": [
                    "Walk 30 minutes 3 times a week.",
                    "Increase vegetables at lunch and dinner.",
                ],
                "note": "These are AI-generated suggestions based on your answers to the form. Please consult with healthcare professionals for personalized medical advice.",
            }
            return SimpleNamespace(text=json.dumps(fake_json))

    # Patch faktisk klasse i modulet api_module.genai
    monkeypatch.setattr(api_module.genai, "GenerativeModel", DummyModel, raising=True)

    # 3) Kald /predict-with-advice
    r = client.post("/predict-with-advice", json=EXAMPLE_INPUT)
    assert r.status_code == 200
    body = r.json()

    # Basale felter fra predict
    assert "label" in body
    assert "confidence" in body
    assert "probabilities" in body

    # Rådgivningsfelt
    assert "personalized_advice" in body
    advice = body["personalized_advice"]
    assert "advice" in advice and isinstance(advice["advice"], list) and len(advice["advice"]) >= 1
    assert "note" in advice and isinstance(advice["note"], str)
