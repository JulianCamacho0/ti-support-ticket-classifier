"""Preparación de features/target y definición del pipeline de preprocesamiento."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import MODEL_PARAMS, TARGET_COLUMN, TEXT_COLUMNS, TFIDF_PARAMS


def build_text_feature(df: pd.DataFrame) -> pd.Series:
    """Combina las columnas de texto en una única serie de entrada al modelo."""
    text = df[TEXT_COLUMNS[0]].fillna("")
    for column in TEXT_COLUMNS[1:]:
        text = text + " " + df[column].fillna("")
    return text


def split_features_target(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Separa el DataFrame en features de texto (X) y variable objetivo (y)."""
    X = build_text_feature(df)
    y = df[TARGET_COLUMN]
    return X, y


def build_pipeline() -> Pipeline:
    """Crea el pipeline de vectorización de texto + modelo clasificador."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf", LogisticRegression(**MODEL_PARAMS)),
        ]
    )
