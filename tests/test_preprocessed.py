from pathlib import Path
import pandas as pd
from taed2_smarthealth_ai.preprocess_obesity import norm, warn_unmapped
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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
