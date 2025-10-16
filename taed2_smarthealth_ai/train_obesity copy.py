# train_obesity.py
from __future__ import annotations
from pathlib import Path
import json
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

#from codecarbon import EmissionsTracker

##tracker = EmissionsTracker()
#tracker.start()


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_xy(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target-kolonnen '{target_col}' findes ikke i {csv_path}")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def main():
    params = load_params("params.yaml")
    D, S, T = params["data"], params["split"], params["train"]
    target = S["target"]

    # Data
    Xtr, ytr = load_xy(D["train"], target)
    Xva, yva = load_xy(D["validation"], target)

    # output folder
    model_dir = Path(T["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    # feature order til API server
    features_out = model_dir / "features.json"
    with open(features_out, "w", encoding="utf-8") as f:
        json.dump({"feature_order": list(Xtr.columns)}, f, indent=2)

    # model
    clf = RandomForestClassifier(
        n_estimators=T.get("n_estimators", 300),
        max_depth=T.get("max_depth", None),
        random_state=T.get("random_state", 42),
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)

    # validation
    pva = clf.predict(Xva)
    metrics = {
        "val_accuracy": float(accuracy_score(yva, pva)),
        "val_f1_macro": float(f1_score(yva, pva, average="macro")),
    }

    # save model
    model_path = model_dir / "model.joblib"
    joblib.dump(clf, model_path)

    # classes (to API)
    classes_path = model_dir / "classes.json"
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump({"classes": clf.classes_.tolist()}, f, indent=2)

    # Feature importance
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
        pass 

    # save metrics for api
    metrics_out = Path(T["metrics_out"])
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # status
    print("[train] saved model     ->", model_path)
    print("[train] features order  ->", features_out)
    print("[train] classes        ->", classes_path)
    print("[train] feature imprt. ->", fi_path)
    print("[train] metrics        ->", metrics_out, metrics)


if __name__ == "__main__":
    main()

#tracker.stop()
