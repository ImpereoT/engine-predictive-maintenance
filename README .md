# ✈️ Engine Predictive Maintenance

ML-сервис для предсказания отказа авиадвигателя на основе данных телеметрии.  
Датасет: [NASA CMAPSS Turbofan Engine Degradation](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) (FD001).

---

## 📌 Задача

Бинарная классификация: предсказать, откажет ли двигатель в течение следующих **30 рабочих циклов** на основе показаний 21 датчика.

Метрика успеха — **AUC-ROC** (важнее precision/recall из-за дисбаланса классов).

---

## 🏗️ Архитектура проекта

```
engine_predictive_maintenance/
├── data/
│   ├── raw/                      # Исходные данные NASA
│   └── processed/                # Данные с рассчитанным RUL и label
├── models/                       # Сохранённая модель (.cbm)
├── src/
│   ├── api/
│   │   ├── main.py               # FastAPI приложение
│   │   ├── schemas.py            # Pydantic-схемы запроса/ответа
│   │   └── templates/
│   │       └── index.html        # Кастомный дашборд (localhost:8000)
│   ├── models/
│   │   ├── preprocess.py         # Препроцессинг и расчёт RUL
│   │   └── train.py              # Обучение CatBoost
│   └── utils/
│       └── predictor.py          # Инференс-обёртка
├── tests/
│   └── test_api.py
├── conftest.py
├── pytest.ini
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🖥️ Интерфейс

При открытии `http://localhost:8000` отображается кастомный дашборд:

- Слайдеры для всех 24 параметров (3 операционные настройки + 21 датчик)
- Цветной результат: **зелёный** (норма) / **красный** (риск отказа)
- Вероятность отказа в процентах с прогресс-баром
- История последних 8 запросов
- Счётчики запросов и предупреждений
- Индикатор состояния API (online / offline)

Swagger UI по-прежнему доступен по адресу `http://localhost:8000/docs`.

---

## 🚀 Быстрый старт

### 1. Локально

```bash
# Установка зависимостей
pip install -r requirements.txt

# Препроцессинг данных
python -m src.models.preprocess

# Обучение модели
python -m src.models.train

# Запуск API
uvicorn src.api.main:app --reload
```

Открой `http://localhost:8000` — дашборд готов к работе.

### 2. Docker (рекомендуется)

```bash
# Сборка образа
docker build -t engine-maintenance .

# Запуск через Docker Compose
docker-compose up

# Или вручную
docker run -p 8000:8000 -v $(pwd)/models:/app/models engine-maintenance
```

> **Важно:** перед сборкой Docker-образа модель должна быть обучена —  
> файл `models/engine_model.cbm` должен существовать.

---

## 🔌 API

### `GET /`
Кастомный интерактивный дашборд.

### `GET /health`
Проверка состояния сервиса.

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### `POST /predict`
Принимает snapshot датчиков, возвращает прогноз.

**Пример запроса:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
    "s_1": 518.67, "s_2": 641.82, "s_3": 1589.70, "s_4": 1400.60,
    "s_5": 14.62, "s_6": 21.61, "s_7": 554.36, "s_8": 2388.06,
    "s_9": 9046.19, "s_10": 1.30, "s_11": 47.47, "s_12": 521.66,
    "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195, "s_16": 0.03,
    "s_17": 392.0, "s_18": 2388.0, "s_19": 100.0,
    "s_20": 39.06, "s_21": 23.419
  }'
```

**Пример ответа:**
```json
{
  "prediction": 0,
  "probability": 0.0842,
  "status": "OK",
  "message": "✅ Двигатель работает в норме. Вероятность отказа в течение 30 циклов: 8.4%."
}
```

---

## 🧪 Тесты

```bash
pytest tests/ -v
```

---

## 🛠️ Стек технологий

| Слой | Инструмент |
|---|---|
| ML | CatBoost, scikit-learn |
| Data | pandas, numpy |
| API | FastAPI, Pydantic v2 |
| Frontend | Vanilla HTML/CSS/JS |
| Server | Uvicorn |
| Контейнеризация | Docker, Docker Compose |
| Тесты | pytest, httpx |
