from __future__ import annotations
from typing import Dict, Any, List, Mapping
import numpy as np
import pandas as pd

try:
    import shap
except Exception:
    shap = None

FEATURE_THEMES: Dict[str, List[str]] = {
    "genetics": ["family_history_with_overweight"],
    "diet": ["FAVC", "FCVC", "NCP", "CAEC", "CALC"],
    "activity": ["FAF", "TUE", "MTRANS_walking", "MTRANS_bike", "MTRANS_motorbike", "MTRANS_automobile"],
    "hydration": ["CH2O"],
    "smoking": ["SMOKE"],
    "anthropometrics": ["Age", "Height", "Weight"],
}

CAT_MAP = {
    "Gender": {0: "Female", 1: "Male"},
    "FAVC": {0: "No", 1: "Yes"},
    "CAEC": {0: "No", 1: "Sometimes", 2: "Frequently", 3: "Always"},
    "CALC": {0: "No", 1: "Sometimes", 2: "Frequently", 3: "Always"},
    "SMOKE": {0: "No", 1: "Yes"},
    "family_history_with_overweight": {0: "No", 1: "Yes"},
    "MTRANS_automobile": {0: "No", 1: "Yes"},
    "MTRANS_bike": {0: "No", 1: "Yes"},
    "MTRANS_motorbike": {0: "No", 1: "Yes"},
    "MTRANS_walking": {0: "No", 1: "Yes"},
}

def _to_scalar(v, default=0):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(v)
        if arr.size == 0:
            return default
        x = arr.ravel()[0]
        return x.item() if isinstance(x, np.generic) else x
    return v

def _num(v, default=0.0) -> float:
    v = _to_scalar(v, default)
    try:
        return float(v)
    except Exception:
        return float(default)

def _num_int(v, default=0) -> int:
    v = _to_scalar(v, default)
    try:
        return int(float(v))
    except Exception:
        return int(default)

def _get_shap_contribs(clf, X_row: pd.DataFrame, target_class: int | None) -> np.ndarray:
    if shap is None:
        raise RuntimeError("shap not installed")
    explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        if target_class is None:
            proba = clf.predict_proba(X_row)[0]
            target_class = int(np.argmax(proba))
        return sv[target_class][0]
    return sv[0]

def _aggregate_by_theme(contribs: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    idx = {f: i for i, f in enumerate(feature_names)}
    out = {}
    for theme, feats in FEATURE_THEMES.items():
        s = 0.0
        for f in feats:
            if f in idx:
                s += float(abs(contribs[idx[f]]))
        out[theme] = s
    return out

# ---- advice functions now read from a plain dict (vals) ----

def _advice_genetics(vals: Mapping[str, Any]) -> List[str]:
    return ["Family history increases risk. Monitor weight, keep meals balanced, and stay active."]

def _advice_diet(vals: Mapping[str, Any]) -> List[str]:
    tips = []
    favc = _num_int(vals.get("FAVC", 0))
    fcvc = _num(vals.get("FCVC", 0.0))
    ncp  = _num(vals.get("NCP", 0.0))
    caec = _num_int(vals.get("CAEC", 0))
    calc = _num_int(vals.get("CALC", 0))
    if favc == 1:
        tips.append("Cut down on energy-dense foods and sugary drinks.")
    if fcvc < 2:
        tips.append("Add more vegetables and fiber to meals (aim for 2–3 servings/day).")
    if caec >= 2:
        tips.append("Reduce snacking; plan protein- and fiber-rich meals for satiety.")
    if ncp < 3:
        tips.append("Keep a regular meal pattern (~3 balanced meals/day).")
    if calc >= 2:
        tips.append("Lower alcohol frequency or portion size.")
    if not tips:
        tips.append("Focus on minimally processed foods and portion control.")
    return tips

def _advice_activity(vals: Mapping[str, Any]) -> List[str]:
    tips = []
    faf  = _num(vals.get("FAF", 0.0))
    tue  = _num(vals.get("TUE", 0.0))
    walk = _num_int(vals.get("MTRANS_walking", 0))
    bike = _num_int(vals.get("MTRANS_bike", 0))
    car  = _num_int(vals.get("MTRANS_automobile", 0))
    if faf < 2:
        tips.append("Build up to ≥150 min/week moderate activity (e.g., brisk walking).")
    else:
        tips.append("Maintain activity and add 2×/week strength training.")
    if tue >= 2:
        tips.append("Break up screen time with short standing or walking breaks each hour.")
    if walk == 0 and bike == 0 and car == 1:
        tips.append("Swap short car trips for walking or cycling when possible.")
    return tips

def _advice_hydration(vals: Mapping[str, Any]) -> List[str]:
    ch2o = _num(vals.get("CH2O", 0.0))
    return ["Increase water intake across the day."] if ch2o < 2 else ["Hydration looks OK."]

def _advice_smoking(vals: Mapping[str, Any]) -> List[str]:
    smoke = _num_int(vals.get("SMOKE", 0))
    return ["Consider a quit plan; it supports cardiometabolic health."] if smoke == 1 else ["No smoking reported."]

def _advice_anthropometrics(vals: Mapping[str, Any]) -> List[str]:
    return ["Track trends over time and pair goals with behavior changes."]

THEME_FUN = {
    "genetics": _advice_genetics,
    "diet": _advice_diet,
    "activity": _advice_activity,
    "hydration": _advice_hydration,
    "smoking": _advice_smoking,
    "anthropometrics": _advice_anthropometrics,
}

def tailored_advice_for_instance(
    clf,
    X_row: pd.DataFrame,
    raw_input: Dict[str, Any],
    predicted_class: int | None = None,
    top_k_themes: int = 2,
) -> Dict[str, Any]:
    X_row = X_row.copy().reset_index(drop=True)
    assert X_row.shape[0] == 1

    # SHAP or fallback
    try:
        contribs = _get_shap_contribs(clf, X_row, predicted_class)
    except Exception:
        imp = getattr(clf, "feature_importances_", None)
        contribs = np.asarray(imp, dtype=float) if imp is not None else np.zeros(X_row.shape[1], dtype=float)

    feats = list(X_row.columns)
    theme_scores = _aggregate_by_theme(contribs, feats)
    ordered = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
    main = [t for t, s in ordered[:top_k_themes] if s > 0]

    # convert the single row to a pure-scalar dict
    row_series = X_row.iloc[0]
    vals = {k: _to_scalar(row_series[k]) for k in feats}

    advice = []
    for t in main:
        try:
            tips = THEME_FUN[t](vals)
        except Exception:
            tips = ["Advice unavailable for this theme."]
        advice.append({"theme": t, "score": round(theme_scores[t], 4), "advice": tips})

    echo = {}
    for k, v in raw_input.items():
        vv = _to_scalar(v)
        echo[k] = CAT_MAP[k][vv] if (k in CAT_MAP and vv in CAT_MAP[k]) else vv

    return {
        "themes_ranked": [{"theme": t, "score": round(s, 4)} for t, s in ordered],
        "advice": advice,
        "note": "Model-based suggestions; not a medical diagnosis.",
        "input_summary": echo,
    }
