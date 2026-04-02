import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.api.schemas import SensorData, PredictionResponse, HealthResponse
from src.utils.predictor import EnginePredictor

MODEL_PATH = os.getenv("MODEL_PATH", "models/engine_model.cbm")
TEMPLATES_DIR = Path(__file__).parent / "templates"

predictor: EnginePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = EnginePredictor(MODEL_PATH)
        print(f"✅ Модель загружена из {MODEL_PATH}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
    yield


app = FastAPI(
    title="Engine Predictive Maintenance API",
    description=(
        "Предсказывает риск поломки авиадвигателя на основе показаний датчиков.\n\n"
        "Датасет: NASA Turbofan Jet Engine (CMAPSS FD001).\n"
        "Модель: CatBoostClassifier. Цель: предсказать отказ за **30 циклов**."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    html_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(status="ok", model_loaded=predictor is not None)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: SensorData):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена.")
    result = predictor.predict(data.model_dump())
    return PredictionResponse(**result)
