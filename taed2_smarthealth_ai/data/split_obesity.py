from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

# Load params
params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
cfgd, cfgs = params["data"], params["split"]

# Load cleaned dataset
df = pd.read_csv(cfgd["interim"])
target = cfgs["target"]

y = df[target]
X = df.drop(columns=[target])

# First: test split
X_trv, X_test, y_trv, y_test = train_test_split(
    X,
    y,
    test_size=cfgs["test_size"],
    random_state=cfgs["random_state"],
    stratify=y if cfgs["stratify"] else None,
)

# Then: validation split from remaining
val_fraction = cfgs["validation_size"] / (1 - cfgs["test_size"])
X_train, X_val, y_train, y_val = train_test_split(
    X_trv,
    y_trv,
    test_size=val_fraction,
    random_state=cfgs["random_state"],
    stratify=y_trv if cfgs["stratify"] else None,
)

# Save outputs
Path(cfgd["processed_dir"]).mkdir(parents=True, exist_ok=True)
pd.concat([X_train, y_train], axis=1).to_csv(cfgd["train"], index=False)
pd.concat([X_val, y_val], axis=1).to_csv(cfgd["validation"], index=False)
pd.concat([X_test, y_test], axis=1).to_csv(cfgd["test"], index=False)

print("[split] train:", cfgd["train"])
print("[split] validation:", cfgd["validation"])
print("[split] test:", cfgd["test"])
