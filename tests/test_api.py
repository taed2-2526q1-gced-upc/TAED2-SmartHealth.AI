from fastapi.testclient import TestClient
from taed2_smarthealth_ai.api.api import app  # <-- updated import

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "SmartHealth-AI API" in response.json()["message"]

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "n_features" in body

def test_predict_missing_fields():
    # send incomplete data to trigger validation
    response = client.post("/predict", json={"Age": 25, "Weight": 70})
    assert response.status_code == 400
    assert "Missing features" in response.json()["detail"]

def test_predict_valid_request():
    # example request with all required fields (adjust keys if your feature_order differs)
    example_input = {
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

    response = client.post("/predict", json=example_input)
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "confidence" in body
    assert "probabilities" in body
