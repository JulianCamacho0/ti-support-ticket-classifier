"""Tests unitarios para el pipeline de ML: lectura, validación, split y entrenamiento."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.preprocessing import split_features_target
from src.train import train_model
from src.validation import validate_dataset


def _sample_df(n_per_class: int = 4) -> pd.DataFrame:
    """DataFrame pequeño y válido, con varias filas por clase para poder entrenar."""
    rows = []
    for i in range(n_per_class):
        rows.append(
            {
                "subject": f"Printer issue {i}",
                "body": "The printer is not responding after the latest update.",
                "answer": "Please restart the device.",
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
        )
        rows.append(
            {
                "subject": f"Billing question {i}",
                "body": "I would like clarification about my last invoice.",
                "answer": "Here are the billing details you requested.",
                "type": "Request",
                "queue": "Billing and Payments",
                "priority": "low",
                "language": "en",
                "version": 51,
                "tag_1": "Billing",
                "tag_2": None,
                "tag_3": None,
                "tag_4": None,
                "tag_5": None,
                "tag_6": None,
                "tag_7": None,
                "tag_8": None,
            }
        )
    return pd.DataFrame(rows)


def test_load_data_reads_existing_dataset():
    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_missing_file_raises_error(tmp_path):
    missing_path = tmp_path / "no_existe.csv"

    with pytest.raises(FileNotFoundError):
        load_data(missing_path)


def test_validate_dataset_accepts_valid_dataframe():
    df = _sample_df()

    validated = validate_dataset(df)

    pd.testing.assert_frame_equal(validated, df)


def test_validate_dataset_rejects_invalid_dataframe():
    df = _sample_df()
    df.loc[0, "priority"] = "urgent"

    with pytest.raises(SchemaError):
        validate_dataset(df)


def test_split_features_target_combines_text_and_extracts_label():
    df = _sample_df()

    X, y = split_features_target(df)

    assert len(X) == len(df)
    assert "Printer issue 0" in X.iloc[0]
    assert sorted(y.unique()) == sorted(df["queue"].unique())


def test_train_model_returns_fitted_pipeline_able_to_predict():
    df = _sample_df()
    X, y = split_features_target(df)

    model = train_model(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
