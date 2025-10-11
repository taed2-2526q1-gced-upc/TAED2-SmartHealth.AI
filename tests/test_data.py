

from pathlib import Path




# tests/test_data.py
import great_expectations as gx
from great_expectations import ValidationDefinition
from data.processed.config import ROOT_DIR
from data.processed.gx_context_configuration import CLEAN_DATA_VALIDATOR
from pytest import fixture

import sys
from pathlib import Path

# Find projektets rodmappe (to niveauer op fra denne fil)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


@fixture
def clean_data_validator() -> ValidationDefinition:
    context = gx.get_context(mode="file", project_root_dir=ROOT_DIR)
    return context.validation_definitions.get(CLEAN_DATA_VALIDATOR)

def test_clean_data(clean_data_validator: ValidationDefinition):
    validation_result = clean_data_validator.run(result_format="BOOLEAN_ONLY")
    expectations_failed = validation_result["statistics"]["unsuccessful_expectations"]
    assert expectations_failed == 0, f"There were {expectations_failed} failing expectations."
