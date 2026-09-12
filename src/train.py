"""Entrenamiento del modelo de clasificación de tickets."""

import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import build_pipeline


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Crea y entrena el pipeline (vectorizador + modelo) con los datos de train."""
    model = build_pipeline()
    model.fit(X_train, y_train)
    return model
