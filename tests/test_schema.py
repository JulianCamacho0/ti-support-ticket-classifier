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
        "answer": "Please restart the printer and try again.",
        "type": "Incident",
        "queue": "Technical Support",
        "priority": "high",
        "language": "en",
        "version": 51,
        "tag_1": "Hardware",
        "tag_2": None,
        "tag_3": None,
        "tag_4": None,
        "tag_5": None,
        "tag_6": None,
        "tag_7": None,
        "tag_8": None,
    }
    row.update(overrides)
    return row


def test_validate_data_accepts_valid_dataframe():
    df = pd.DataFrame([_base_row()])

    validated = validate_data(df)

    pd.testing.assert_frame_equal(validated, df)


def test_validate_data_rejects_invalid_type_in_numeric_column():
    df = pd.DataFrame([_base_row(version="not-a-number")])

    with pytest.raises(SchemaError):
        validate_data(df)


def test_validate_data_rejects_disallowed_category():
    df = pd.DataFrame([_base_row(priority="urgent")])

    with pytest.raises(SchemaError):
        validate_data(df)


def test_validate_data_rejects_missing_required_column():
    df = pd.DataFrame([_base_row()]).drop(columns=["body"])

    with pytest.raises(SchemaError):
        validate_data(df)
