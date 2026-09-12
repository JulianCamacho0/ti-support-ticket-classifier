"""API REST sencilla para servir el modelo de clasificación de tickets.

Ejecutar localmente con:
    uvicorn src.api:app --reload
"""

import sys
from pathlib import Path

# Permite ejecutar/importar este módulo sin instalar el paquete (igual que main.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_FILE
from src.preprocessing import prepare_features

app = FastAPI(title="TI Support Ticket Classifier", version="0.0.0")

class TicketRequest(BaseModel):
    """Datos de entrada de un ticket a clasificar."""

    subject: str | None = Field(default=None, description="Asunto del ticket")
    body: str = Field(..., min_length=1, description="Descripción del ticket")
    type: str = Field(..., description="Tipo de mensaje (Incident, Request, Problem, Change)")
    language: str = Field(..., description="Idioma del mensaje (de, en)")
    priority: str = Field(..., description="Prioridad del mensaje (high, medium, low)")


class PredictionResponse(BaseModel):
    """Resultado de la clasificación del ticket."""

    queue: str
    probability: float | None = None


def _load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en: {MODEL_FILE}. "
            "Ejecuta primero 'python src/main.py' para entrenarlo y guardarlo."
        )
    return joblib.load(MODEL_FILE)


# El modelo se carga una única vez al iniciar la API.
model = _load_model()


@app.get("/health")
def health() -> dict:
    """Verifica que la API esté disponible y el modelo cargado."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(ticket: TicketRequest) -> PredictionResponse:
    """Predice la cola de soporte (`queue`) para un ticket dado.

    El pipeline cargado (mismo artefacto usado en entrenamiento) se encarga de
    todo el preprocesamiento (TF-IDF para texto, One-Hot para categóricas).
    """
    try:
        df = pd.DataFrame([ticket.model_dump()])
        X = prepare_features(df)

        prediction = model.predict(X)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X).max())

        return PredictionResponse(queue=prediction, probability=probability)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
