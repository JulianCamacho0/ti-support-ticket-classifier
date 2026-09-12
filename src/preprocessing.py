"""Preparación de features/target y definición del pipeline de preprocesamiento."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    MODEL_PARAMS,
    ONEHOT_PARAMS,
    TARGET_COLUMN,
    TEXT_COLUMNS,
    TFIDF_PARAMS,
)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona las variables predictoras y rellena los textos nulos con ''."""
    features = df[FEATURE_COLUMNS].copy()
    for column in TEXT_COLUMNS:
        features[column] = features[column].fillna("")
    return features


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa el DataFrame en variables predictoras (X) y variable objetivo (y)."""
    X = prepare_features(df)

    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Vectoriza `subject`/`body` como texto (TF-IDF) y codifica las columnas
    categóricas (`type`, `language`, `priority`) con One-Hot Encoding."""
    return ColumnTransformer(
        transformers=[
            ("subject_tfidf", TfidfVectorizer(**TFIDF_PARAMS), "subject"),
            ("body_tfidf", TfidfVectorizer(**TFIDF_PARAMS), "body"),
            ("categorical", OneHotEncoder(**ONEHOT_PARAMS), CATEGORICAL_COLUMNS),
        ]
    )


def build_pipeline() -> Pipeline:
    """Crea el pipeline completo: preprocesamiento (texto + categóricas) + modelo."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("clf", LogisticRegression(**MODEL_PARAMS)),
        ]
    )
