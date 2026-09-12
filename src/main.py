"""Punto de entrada del pipeline de Machine Learning: python src/main.py"""

import sys
from pathlib import Path

# Permite ejecutar este archivo directamente (python src/main.py) sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from sklearn.model_selection import train_test_split

from src.config import MODEL_FILE, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.data import load_data
from src.evaluate import evaluate_model
from src.preprocessing import split_features_target
from src.train import train_model
from src.validation import validate_dataset


def main() -> None:
    print("Iniciando pipeline de entrenamiento...")

    df = load_data()
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    df = validate_dataset(df)
    print("Validación del esquema completada correctamente")
    
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} filas | Test: {len(X_test)} filas")

    print("Entrenando modelo...")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"Modelo guardado en: {MODEL_FILE}")


if __name__ == "__main__":
    main()
