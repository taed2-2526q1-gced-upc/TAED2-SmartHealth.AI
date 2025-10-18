"""Model test: Tests for data preprocessing and validation."""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import precision_recall_fscore_support

# Ensure project root is on sys.path (so `src.*` imports work if needed later)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------- fixtures ----------------
@pytest.fixture(scope="session")
def params():
    """Load params.yaml once per session."""
    return yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))


@pytest.fixture(scope="session")
def pipe(params):
    """Load the trained pipeline and ensure feature names are available."""
    model_path = params["train"]["model_dir"] + "/model.joblib"
    pipe = joblib.load(model_path)
    if not hasattr(pipe, "feature_names_in_"):
        pipe.feature_names_in_ = np.array([
            "Gender", "Age", "Height", "Weight",
            "family_history_with_overweight", "FAVC", "FCVC", "NCP", "CAEC", "SMOKE",
            "CH2O", "SCC", "FAF", "TUE", "CALC",
            "MTRANS_automobile", "MTRANS_bike", "MTRANS_motorbike", "MTRANS_walking",
        ])
    return pipe


@pytest.fixture
def test_ds(params):
    """Load the validation dataset defined in params.yaml."""
    test_path = params["data"]["validation"]
    return pd.read_csv(test_path)


# Default label names (used if not provided in params.yaml)
DEFAULT_LABELS = [
    "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
    "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]


@pytest.fixture(scope="session")
def label_index_to_name(params):
    """Map class indices -> human-readable labels."""
    return params.get("labels", {}).get("index_to_name", DEFAULT_LABELS)


@pytest.fixture(scope="session")
def label_set(label_index_to_name):
    return set(label_index_to_name)

def to_label_name(y_data, label_index_to_name):
    """
    Convert model output (possibly numeric code) to human-readable label.
    """
    # if it's already a string label, just return it
    if isinstance(y_data, str) and y_data in label_index_to_name or y_data in set(label_index_to_name):
        return str(y_data)
    # try integer mapping
    try:
        idx = int(y_data)
        if 0 <= idx < len(label_index_to_name):
            return label_index_to_name[idx]
    except Exception:
        pass
    # fallback
    return str(y_data)

# ---------------- tests ----------------
def test_model_accuracy(pipe, test_ds, params):
    target = params["split"]["target"]
    X_data = test_ds.drop(columns=[target])
    y_data = test_ds[target]
    acc = pipe.score(X_data, y_data)
    assert acc >= 0.85  # threshold per your request

# Rer-class recall >= 0.75 
def test_per_class_recall(pipe, test_ds, params, label_index_to_name):
    target = params["split"]["target"]
    X_data = test_ds.drop(columns=[target])
    y_true = pd.Series(test_ds[target]).astype(str)

    y_pred_raw = pipe.predict(X_data)
    y_pred = pd.Series([to_label_name(p, label_index_to_name) for p in y_pred_raw])

    labels = list(label_index_to_name)  # fixed order
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    min_support = 10
    for lbl, r, sup in zip(labels, rec, support):
        if sup < min_support:
            continue
        assert r >= 0.75, f"Recall for '{lbl}'={r:.2f} (support={sup}) < 0.75"


@pytest.mark.parametrize("overrides", [
    {"Age": 23, "Gender": 1, "Height": 1.70, "Weight": 70},
    {"Age": 45, "Gender": 0, "Height": 1.60, "Weight": 90},
])
def test_model_predictions(pipe, overrides, label_set, label_index_to_name):
    features = list(pipe.feature_names_in_)
    sample = {f: 0 for f in features}
    defaults = {
        "Age": 30, "Gender": 1, "Height": 1.70, "Weight": 70,
        "FCVC": 2.0, "CH2O": 2.0, "FAF": 1.0
    }
    for k, v in defaults.items():
        if k in sample:
            sample[k] = v
    for k, v in overrides.items():
        if k in sample:
            sample[k] = v
    df = pd.DataFrame([sample], columns=features)
    pred = pipe.predict(df)
    assert len(pred) == 1
    label = to_label_name(pred[0], label_index_to_name)
    assert label in label_set


@pytest.mark.parametrize("overrides,expected_label", [
    # Insufficient Weight
    ({"Age": 15, "Gender": 1, "Height": 1.90, 
    "Weight": 40, "FCVC": 2, "CH2O": 2.0, "FAF": 1.0},
     "Insufficient_Weight"),

    # Normal Weight
    ({"Age": 50, "Gender": 1, "Height": 1.62, 
    "Weight": 64, "FCVC": 2.0, "CH2O": 2.0, "FAF": 0.0},
     "Normal_Weight"),

    # Overweight Level I
    ({"Age": 28, "Gender": 0, "Height": 1.70, 
      "Weight": 74, "FCVC": 1.5, "CH2O": 1.5, "FAF": 0.5},
     "Overweight_Level_I"),

    # Overweight Level II
    ({"Age": 30, "Gender": 0, "Height": 1.78, 
      "Weight": 90, "FCVC": 0.0, "CH2O": 2.0, "FAF": 0.0},
     "Overweight_Level_II"),

    # Obesity Level I
    ({"Age": 25, "Gender": 1, "Height": 1.50, 
      "Weight": 100, "FCVC": 0.0, "CH2O": 0.0, "FAF": 0.0},
     "Obesity_Type_I"),

    # Obesity Level II
    ({"Age": 31, "Gender": 1, "Height": 1.68, 
      "Weight": 125, "FCVC": 0.5, "CH2O": 1.0, "FAF": 0.0},
     "Obesity_Type_II"),

    # Obesity Level III
    ({"Age": 23, "Gender": 1, "Height": 1.73, 
      "Weight": 140, "FCVC": 3.0, "CH2O": 3.0, "FAF": 1.0},
     "Obesity_Type_III"),
])


def test_model_expected_labels(pipe, overrides, expected_label, label_index_to_name):
    features = list(pipe.feature_names_in_)
    sample = {f: 0 for f in features}
    defaults = {"Age": 30, "Gender": 1, "Height": 1.70, "Weight": 70, "FCVC": 2.0, "CH2O": 2.0, "FAF": 1.0}
    for k, v in defaults.items():
        if k in sample:
            sample[k] = v
    for k, v in overrides.items():
        if k in sample:
            sample[k] = v
    df = pd.DataFrame([sample], columns=features)
    pred = pipe.predict(df)[0]
    label = to_label_name(pred, label_index_to_name)
    assert label == expected_label

def test_stability_near_duplicates(pipe, label_index_to_name):
    features = list(pipe.feature_names_in_)
    base = {f: 0 for f in features}
    base.update({"Age": 32, "Gender": 1, "Height": 1.75, "Weight": 82, "FCVC": 2.0, "CH2O": 2.0, "FAF": 1.0})
    a = pd.DataFrame([base], columns=features)

    b = base.copy()
    if "Weight" in b: 
        b["Weight"] = b["Weight"] + 0.3
    if "Height" in b:
        b["Height"] = b["Height"] - 0.05
    b = pd.DataFrame([b], columns=features)

    pa = to_label_name(pipe.predict(a)[0], label_index_to_name)
    pb = to_label_name(pipe.predict(b)[0], label_index_to_name)
    assert pa == pb