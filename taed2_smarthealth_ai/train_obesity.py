from pathlib import Path
import json
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import dagshub
import mlflow
from codecarbon import EmissionsTracker


# load config
params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
D, S, T = params["data"], params["split"], params["train"]
target = S["target"]

def load_xy(fp: str):
    df = pd.read_csv(fp)
    y = df[target]
    X = df.drop(columns=[target])
    return X, y



tracker = EmissionsTracker()
tracker.start()
# Your training code here

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
        [{"feature": feat, "importance": float(imp)} for feat, imp in zip(Xtr.columns, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )
    with open(fi_path, "w", encoding="utf-8") as f:
        json.dump(fi, f, indent=2)
except Exception:
    pass  # OK if model doesn't expose importances

# 5) Save metrics (ensure parent folder exists)
metrics_out = Path(T["metrics_out"])
metrics_out.parent.mkdir(parents=True, exist_ok=True)
with open(metrics_out, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# 6) Status logs
print("[train] saved model     ->", model_path)
print("[train] features order  ->", features_out)
print("[train] classes         ->", classes_path)
print("[train] feature imprt.  ->", fi_path)
print("[train] metrics         ->", metrics_out, metrics)
tracker.stop()


# Initialiser DagsHub som MLflow tracking server
dagshub.init(repo_owner='RenauxNt', repo_name='TAED2-SmartHealth.AI', mlflow=True)

# Start MLflow run
with mlflow.start_run(run_name="obesity_model_training"):
    # Log parametre og metrics
    mlflow.log_param('learning_rate', 0.01)
    mlflow.log_param('epochs', 50)
    mlflow.log_metric('accuracy', 0.92)

    # Her kan du også logge modellen, hvis du vil
    # mlflow.sklearn.log_model(model, "model")
