"""Tests unitarios para la API REST (src/api.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_accepts_all_predictors_and_returns_valid_prediction():
    payload = {
        "subject": "Printer not working",
        "body": "The office printer stopped responding after the latest update.",
        "type": "Incident",
        "language": "en",
        "priority": "high",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["queue"], str) and data["queue"]
    assert data["probability"] is None or 0.0 <= data["probability"] <= 1.0


def test_predict_without_optional_subject_still_works():
    payload = {
        "body": "I would like clarification about my last invoice.",
        "type": "Request",
        "language": "en",
        "priority": "low",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200


def test_predict_missing_required_field_returns_validation_error():
    payload = {
        "subject": "Printer not working",
        "body": "The office printer stopped responding.",
        "language": "en",
        "priority": "high",
        # falta "type"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
