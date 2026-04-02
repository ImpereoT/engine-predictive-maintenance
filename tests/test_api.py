"""
Базовые тесты API.
Запуск: pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# Мокаем predictor до импорта app
@pytest.fixture
def client():
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = {
        "prediction": 0,
        "probability": 0.12,
        "status": "OK",
        "message": "✅ Двигатель работает в норме. Вероятность отказа: 12.0%.",
    }

    with patch("src.api.main.predictor", mock_predictor):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


SAMPLE_PAYLOAD = {
    "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
    "s_1": 518.67, "s_2": 641.82, "s_3": 1589.70, "s_4": 1400.60,
    "s_5": 14.62,  "s_6": 21.61,  "s_7": 554.36,  "s_8": 2388.06,
    "s_9": 9046.19, "s_10": 1.30, "s_11": 47.47,  "s_12": 521.66,
    "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195, "s_16": 0.03,
    "s_17": 392.0, "s_18": 2388.0, "s_19": 100.0,
    "s_20": 39.06, "s_21": 23.4190,
}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_ok(client):
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["status"] in ("OK", "WARNING")


def test_predict_missing_field(client):
    bad_payload = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "s_1"}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422  # Unprocessable Entity
