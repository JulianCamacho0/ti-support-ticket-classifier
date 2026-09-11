"""Configuraciones y constantes centrales del pipeline de ML."""

from pathlib import Path

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "dataset-ti-tickets.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "ticket_classifier.joblib"

# Columnas del dataset
TEXT_COLUMNS = ["subject", "body"]  # se combinan en un único texto de entrada
TARGET_COLUMN = "queue"  # cola/área de soporte a la que se enruta el ticket

# Split de entrenamiento/test
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hiperparámetros básicos del baseline (sin búsqueda de hiperparámetros)
TFIDF_PARAMS = {"max_features": 5000, "ngram_range": (1, 2)}
MODEL_PARAMS = {"max_iter": 1000, "random_state": RANDOM_STATE}
