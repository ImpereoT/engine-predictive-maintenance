from pydantic import BaseModel, Field
from typing import Optional


class SensorData(BaseModel):
    """
    Данные одного snapshot двигателя:
    3 операционные настройки + 21 датчик.
    """
    # Операционные настройки
    setting_1: float = Field(..., example=-0.0007)
    setting_2: float = Field(..., example=-0.0004)
    setting_3: float = Field(..., example=100.0)

    # Датчики s_1 ... s_21
    s_1:  float = Field(..., example=518.67)
    s_2:  float = Field(..., example=641.82)
    s_3:  float = Field(..., example=1589.70)
    s_4:  float = Field(..., example=1400.60)
    s_5:  float = Field(..., example=14.62)
    s_6:  float = Field(..., example=21.61)
    s_7:  float = Field(..., example=554.36)
    s_8:  float = Field(..., example=2388.06)
    s_9:  float = Field(..., example=9046.19)
    s_10: float = Field(..., example=1.30)
    s_11: float = Field(..., example=47.47)
    s_12: float = Field(..., example=521.66)
    s_13: float = Field(..., example=2388.02)
    s_14: float = Field(..., example=8138.62)
    s_15: float = Field(..., example=8.4195)
    s_16: float = Field(..., example=0.03)
    s_17: float = Field(..., example=392.0)
    s_18: float = Field(..., example=2388.0)
    s_19: float = Field(..., example=100.0)
    s_20: float = Field(..., example=39.06)
    s_21: float = Field(..., example=23.4190)

    class Config:
        json_schema_extra = {
            "example": {
                "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
                "s_1": 518.67, "s_2": 641.82, "s_3": 1589.70, "s_4": 1400.60,
                "s_5": 14.62, "s_6": 21.61, "s_7": 554.36, "s_8": 2388.06,
                "s_9": 9046.19, "s_10": 1.30, "s_11": 47.47, "s_12": 521.66,
                "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195, "s_16": 0.03,
                "s_17": 392.0, "s_18": 2388.0, "s_19": 100.0,
                "s_20": 39.06, "s_21": 23.4190,
            }
        }


class PredictionResponse(BaseModel):
    """Ответ модели."""
    model_config = {"protected_namespaces": ()}
    prediction: int = Field(...,
                            description="0 — норма, 1 — риск поломки ≤30 циклов")
    probability: float = Field(...,
                               description="Вероятность поломки (класс 1)")
    status: str = Field(..., description="OK | WARNING")
    message: str = Field(..., description="Человекочитаемый вывод")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"
