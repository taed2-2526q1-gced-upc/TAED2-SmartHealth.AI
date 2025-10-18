import pandas as pd
import yaml
import sys
from pathlib import Path


import numpy as np


# ---- functions to test ----
def norm(s: pd.Series) -> pd.Series:
    """Normalize Unicode (NFKC), strip spaces, lowercase, keep NaN."""
    return s.astype("string").str.normalize("NFKC").str.strip().str.lower()

def warn_unmapped(series: pd.Series, used_keys, colname: str):
    """Print warning if any non-NA values not in used_keys."""
    bad = series[~series.isna() & ~series.isin(used_keys)].unique()
    if len(bad) > 0:
        print(f"[WARN] Unmapped values in {colname}: {bad}")


# ---- pytest tests ----

def test_norm_trims_and_lowercases():
    s = pd.Series(["  Female ", "MALE", None])
    out = norm(s)
    assert list(out) == ["female", "male", pd.NA]

def test_norm_nfkc_normalizes():
    s = pd.Series(["ＦＥＭＡＬＥ", "Ｐｕｂｌｉｃ＿Ｔｒａｎｓｐｏｒｔａｔｉｏｎ"])
    out = norm(s)
    assert out.iloc[0] == "female"
    assert out.iloc[1] == "public_transportation"

def test_warn_unmapped_prints_only_unmapped(capsys):
    s = pd.Series(["yes", "no", "maybe", np.nan])
    s_norm = norm(s)
    warn_unmapped(s_norm, used_keys={"yes", "no"}, colname="FAVC")
    captured = capsys.readouterr().out
    assert "[WARN]" in captured
    assert "maybe" in captured
    assert "yes" not in captured
    assert "no" not in captured

def test_warn_unmapped_no_output_when_all_valid(capsys):
    s = pd.Series(["yes", "no", None])
    warn_unmapped(norm(s), used_keys={"yes", "no"}, colname="FAVC")
    assert capsys.readouterr().out == ""

def main():
    # DVC runs from repo root → read params.yaml from CWD
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))

    RAW = Path(params["data"]["raw"])          # e.g., data/raw/ObesityDataSet_raw_and_data_sinthetic.csv
    OUT = Path(params["data"]["interim"])      # e.g., data/interim/obesity_clean.csv
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # --- load ---
    data = pd.read_csv(RAW)

    # --- normalize relevant text cols (avoid NaNs from case/space mismatches) ---
    for col in ["CAEC","CALC","Gender","NObeyesdad",
                "family_history_with_overweight","FAVC","SMOKE","SCC","MTRANS"]:
        if col in data.columns:
            data[col] = norm(data[col])

    # --- mappings  ---
    caec_calc_mapping = {'no':0, 'sometimes':1, 'frequently':2, 'always':3}
    obesity_mapping = {
        "insufficient_weight":0, "normal_weight":1,
        "overweight_level_i":2, "overweight_level_ii":3,
        "obesity_type_i":4, "obesity_type_ii":5, "obesity_type_iii":6
    }
    gender_mapping = {"male":0, "female":1}
    binary_mapping = {"yes":1, "no":0}

    for col in ["CAEC","CALC"]:
        if col in data.columns:
            warn_unmapped(data[col], caec_calc_mapping.keys(), col)
            data[col] = data[col].map(caec_calc_mapping).astype("Int64")

    if "NObeyesdad" in data.columns:
        warn_unmapped(data["NObeyesdad"], obesity_mapping.keys(), "NObeyesdad")
        data["Obesity"] = data["NObeyesdad"].map(obesity_mapping).astype("Int64")

    if "Gender" in data.columns:
        warn_unmapped(data["Gender"], gender_mapping.keys(), "Gender")
        data["Gender"] = data["Gender"].map(gender_mapping).astype("Int64")

    for col in ['family_history_with_overweight','FAVC','SMOKE','SCC']:
        if col in data.columns:
            warn_unmapped(data[col], binary_mapping.keys(), col)
            data[col] = data[col].map(binary_mapping).astype("Int64")

    # One-hot for MTRANS, force 0/1 ints; drop a reference dummy if present
    if "MTRANS" in data.columns:
        data = pd.get_dummies(data, columns=["MTRANS"], drop_first=False, dtype=int)
        ref_col = "MTRANS_public_transportation"
        if ref_col in data.columns:
            data.drop(columns=[ref_col], inplace=True)

    # Drop original categorical target after successful mapping
    if "NObeyesdad" in data.columns:
        if data["Obesity"].isna().sum() == 0:
            data.drop(columns=["NObeyesdad"], inplace=True)
        else:
            print("[WARN] Some 'Obesity' values are NaN; kept NObeyesdad for inspection.")

    # --- save CSV ---
    if OUT.suffix.lower() != ".csv":
        OUT = OUT.with_suffix(".csv")
    data.to_csv(OUT, index=False)

    print(f"[preprocess] raw -> {RAW}")
    print(f"[preprocess] cleaned -> {OUT}")

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        raise