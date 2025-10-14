# tests/test_api.py
from http import HTTPStatus
from fastapi.testclient import TestClient
import pytest

from taed2_smarthealth_ai.api.api import app

@pytest.fixture(scope="module", autouse=True)
def client():
    with TestClient(app) as c:
        yield c

def test_root(client):
    r = client.get("/")
    assert r.status_code == HTTPStatus.OK
    assert r.json()["message"] == "Welcome to the Obesity app!"

def test_single_item(client):
    payload = {
        "items": [
            {
                "Gender":"Female","Age":21,"Height":1.62,"Weight":64,
                "family_history_with_overweight":"yes","FAVC":"no","FCVC":2,"NCP":3,
                "CAEC":"Sometimes","SMOKE":"no","CH2O":2,"SCC":"no",
                "FAF":0,"TUE":1,"CALC":"no","MTRANS":"Public_Transportation"
            }
        ]
    }
    r = client.post("/prediction", json=payload)
    assert r.status_code == HTTPStatus.OK
    data = r.json()
    assert len(data) == 1
    assert "label" in data[0]
    # score er optional – kan være None hvis model ikke har predict_proba
    assert "score" in data[0]
