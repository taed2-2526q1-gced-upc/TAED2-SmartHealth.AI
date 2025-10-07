import pytest
import joblib
import pandas as pd
import yaml

@pytest.fixture
def pipe():
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    model_path = params["train"]["model_dir"] + "/model.joblib"
    return joblib.load(model_path)

@pytest.fixture
def test_ds():
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    test_path = params["data"]["validation"]
    return pd.read_csv(test_path)

def test_model_accuracy(pipe, test_ds):
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    target = params["split"]["target"]
    X = test_ds.drop(columns=[target])
    y = test_ds[target]
    acc = pipe.score(X, y)
    assert acc > 0.85  # justér threshold efter behov

@pytest.mark.parametrize("sample", [
    {"Age": 23, "Gender": 1, "Height": 170, "Weight": 70},  # tilpas til dine features
    {"Age": 45, "Gender": 0, "Height": 160, "Weight": 90},
])
def test_model_predictions(pipe, sample):
    df = pd.DataFrame([sample])
    pred = pipe.predict(df)
    assert pred is not None