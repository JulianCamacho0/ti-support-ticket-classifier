"""Tests unitarios para el contrato de datos definido en src/schema.py."""

import sys
from pathlib import Path
import pandas as pd
import pytest
from pandera.errors import SchemaError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schema import validate_data

def _base_row(**overrides):
    row = {
        "subject": "Printer not working",
        "body": "The office printer stopped responding after the update.",
        "type": "Incident",
        "queue": "Technical Support",
        "priority": "high",
        "language": "en",
    }
    row.update(overrides)
    return row

def test_validate_data_accepts_valid_dataframe():
    df = pd.DataFrame([_base_row()])
    validated = validate_data(df)
    pd.testing.assert_frame_equal(validated, df)

def test_validate_data_rejects_disallowed_category():
    df = pd.DataFrame([_base_row(priority="urgent")])
    with pytest.raises(SchemaError):
        validate_data(df)


def test_validate_data_rejects_missing_required_column():
    df = pd.DataFrame([_base_row()]).drop(columns=["body"])
    with pytest.raises(SchemaError):
        validate_data(df)