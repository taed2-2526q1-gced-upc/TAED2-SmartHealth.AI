from pathlib import Path
import json, yaml, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# load config
params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
D, S, T = params["data"], params["split"], params["train"]
target = S["target"]

def load_xy(fp: str):
    df = pd.read_csv(fp)
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

# data
from codecarbon import EmissionsTracker

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

# save artifacts
model_dir = Path(T["model_dir"]); model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, model_dir / "model.joblib")

with open(T["metrics_out"], "w") as f:
    json.dump(metrics, f, indent=2)

print("[train] saved model ->", model_dir / "model.joblib")
print("[train] metrics ->", T["metrics_out"], metrics)

tracker.stop()



