import json
from pathlib import Path

from codecarbon import EmissionsTracker
import dagshub
import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import yaml

REPORTS_DIR = Path("reports")
METRICS_PATH = REPORTS_DIR / "metrics.json"  # reports/metrics.json

# load config
params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
D, S, T = params["data"], params["split"], params["train"]
target = S["target"]


def load_xy(fp: str):
    df = pd.read_csv(fp)
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


tracker = EmissionsTracker(
    output_dir=str(REPORTS_DIR),
    output_file="emissions.csv",
    save_to_file=True,
    gpu_ids=[],  # disable GPU detection
    measure_power_secs=0.1,
)
tracker.start()
# training code starts here

Xtr, ytr = load_xy(D["train"])
Xva, yva = load_xy(D["validation"])

# model
clf = RandomForestClassifier(
    n_estimators=T["n_estimators"],
    max_depth=T["max_depth"],
    random_state=T["random_state"],
    n_jobs=-1,
)
clf.fit(Xtr, ytr)
pva = clf.predict(Xva)

# metrics
metrics = {
    "val_accuracy": float(accuracy_score(yva, pva)),
    "val_f1_macro": float(f1_score(yva, pva, average="macro")),
}


# Ensure model dir exists
model_dir = Path(T["model_dir"])
model_dir.mkdir(parents=True, exist_ok=True)

# 1) Save model
model_path = model_dir / "model.joblib"
joblib.dump(clf, model_path)

# 2) Save feature order (critical for API input order)
features_out = model_dir / "features.json"
with open(features_out, "w", encoding="utf-8") as f:
    json.dump({"feature_order": list(Xtr.columns)}, f, indent=2)

# 3) Save classes (map model outputs to labels)
classes_path = model_dir / "classes.json"
with open(classes_path, "w", encoding="utf-8") as f:
    json.dump({"classes": clf.classes_.tolist()}, f, indent=2)

# 4) Save feature importances (if available)
fi_path = model_dir / "feature_importances.json"
try:
    importances = clf.feature_importances_
    fi = sorted(
        [
            {"feature": feat, "importance": float(imp)}
            for feat, imp in zip(Xtr.columns, importances)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )
    with open(fi_path, "w", encoding="utf-8") as f:
        json.dump(fi, f, indent=2)
except Exception:
    pass  # OK if model doesn't expose importances

# 5) Save metrics (ensure parent folder exists)
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# 6) Status logs
print("[train] saved model     ->", model_path)
print("[train] features order  ->", features_out)
print("[train] classes         ->", classes_path)
print("[train] feature imprt.  ->", fi_path)
print("metrics                  → ", METRICS_PATH)
print("emissions                → ", REPORTS_DIR / "emissions.csv")
tracker.stop()

# ML Flow tracking via DagsHub
dagshub.init(
    repo_owner="RenauxNt",
    repo_name="TAED2-SmartHealth.AI",
    mlflow=True,
)

with mlflow.start_run(run_name="rf_train_eval"):
    mlflow.log_params(
        {
            "model_type": "RandomForestClassifier",
            "n_estimators": T["n_estimators"],
            "max_depth": T["max_depth"],
            "random_state": T["random_state"],
            "n_features": Xtr.shape[1],
            "target": target,
            "train_rows": Xtr.shape[0],
            "val_rows": Xva.shape[0],
        }
    )

    # Metrics
    mlflow.log_metrics(metrics)

    # Artifacts
    mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")
    mlflow.log_artifact(str(REPORTS_DIR / "emissions.csv"), artifact_path="reports")
    mlflow.log_artifact(str(features_out), artifact_path="model_assets")
    mlflow.log_artifact(str(classes_path), artifact_path="model_assets")
    if fi_path and fi_path.exists():
        mlflow.log_artifact(str(fi_path), artifact_path="model_assets")
