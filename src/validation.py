"""Punto único de validación del contrato de datos del pipeline."""

import pandas as pd

from src.schema import validate_data


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Valida `df` contra el contrato de datos definido en `src.schema`."""
    return validate_data(df)
