import pytest
import joblib
import pandas as pd
import yaml
import numpy as np

@pytest.fixture(scope="session")
def params():
    return yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))

@pytest.fixture(scope="session")
def pipe(params):
    model_path = params["train"]["model_dir"] + "/model.joblib"
    pipe = joblib.load(model_path)
    # Ensure feature_names_in_ is set for compatibility)
    if not hasattr(pipe, "feature_names_in_"):
        pipe.feature_names_in_ = np.array([
            'Gender','Age','Height','Weight',
            'family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE',
            'CH2O','SCC','FAF','TUE','CALC','MTRANS_automobile','MTRANS_bike',
            'MTRANS_motorbike','MTRANS_walking'
        ])
    return pipe

@pytest.fixture
def test_ds(params):
    test_path = params["data"]["validation"]
    return pd.read_csv(test_path)

def test_model_accuracy(pipe, test_ds, params):
    target = params["split"]["target"]
    X = test_ds.drop(columns=[target])
    y = test_ds[target]
    acc = pipe.score(X, y)
    assert acc > 0.85  # Threshold 

@pytest.mark.parametrize("overrides", [
    {"Age": 23, "Gender": 1, "Height": 1.70, "Weight": 70},
    {"Age": 45, "Gender": 0, "Height": 1.60, "Weight": 90},
])
def test_model_predictions(pipe, overrides):
    features = list(pipe.feature_names_in_)
    sample = {f: 0 for f in features}
    # Make sure all features are set to reasonable defaults
    defaults = {
        "Age": 30, "Gender": 1, "Height": 1.70, "Weight": 70,
        "FCVC": 2.0, "CH2O": 2.0, "FAF": 1.0
    }
    for k, v in defaults.items():
        if k in sample:
            sample[k] = v
    # override with test case values
    for k, v in overrides.items():
        if k in sample:
            sample[k] = v
    df = pd.DataFrame([sample], columns=features)
    pred = pipe.predict(df)
    assert pred is not None
    assert len(pred) == 1
