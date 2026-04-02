import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# Порядок фичей — тот же, что использовался при обучении
FEATURE_COLUMNS = (
    ["setting_1", "setting_2", "setting_3"]
    + [f"s_{i}" for i in range(1, 22)]
)


class EnginePredictor:
    """
    Singleton-обёртка над CatBoost-моделью.
    Загружается один раз при старте FastAPI-приложения.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена по пути: {model_path}\n"
                "Сначала запустите train.py, чтобы обучить и сохранить модель."
            )
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        self._model_path = model_path

    def predict(self, sensor_data: dict) -> dict:
        """
        Принимает словарь с данными датчиков,
        возвращает словарь с prediction, probability, status, message.
        """
        # Собираем DataFrame с правильным порядком колонок
        df = pd.DataFrame([sensor_data])[FEATURE_COLUMNS]

        prediction = int(self.model.predict(df)[0])
        probability = float(self.model.predict_proba(df)[0][1])

        if prediction == 1:
            status = "WARNING"
            message = (
                f"⚠️  Риск поломки! Вероятность отказа в течение 30 циклов: "
                f"{probability:.1%}. Рекомендуется техническое обслуживание."
            )
        else:
            status = "OK"
            message = (
                f"✅ Двигатель работает в норме. "
                f"Вероятность отказа в течение 30 циклов: {probability:.1%}."
            )

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "status": status,
            "message": message,
        }
