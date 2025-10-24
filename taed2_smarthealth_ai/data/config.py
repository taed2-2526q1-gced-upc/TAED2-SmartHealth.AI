# ROOT_DIR = Path(__file__).resolve().parents[2]
# DATA_DIR = ROOT_DIR / "data"
# RAW_DATA_DIR = DATA_DIR / "raw"
# INTERIM_DATA_DIR = DATA_DIR / "interim"
# PROCESSED_DATA_DIR = DATA_DIR / "processed"

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

proj_root = os.getenv("PROJ_ROOT") or os.getenv("ROOT")
if not proj_root:
    proj_root = Path(__file__).resolve().parents[2]

ROOT_DIR = Path(proj_root).resolve()

logger.info(f"ROOT_DIR path is: {ROOT_DIR}")

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

TESTS_DIR = ROOT_DIR / "tests"

REPORTS_DIR = ROOT_DIR / "reports"

MODELS_DIR = ROOT_DIR / "models"
PROD_MODEL = "obesity-clf"

SEED = 2025
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1


# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# Environment variable (MODEL_PATH) takes priority over the default model path
_env_model = os.getenv("MODEL_PATH")
if _env_model:
    MODEL_PATH = Path(_env_model)
else:
    MODEL_PATH = MODELS_DIR / PROD_MODEL

logger.info(f"ROOT_DIR: {ROOT_DIR}")
logger.info(f"MODEL_PATH: {MODEL_PATH}")
