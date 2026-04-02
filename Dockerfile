# ── Stage 1: builder ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Системные зависимости (нужны catboost для компиляции)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ───────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Копируем установленные пакеты из builder
COPY --from=builder /install /usr/local

# Копируем исходники
COPY src/ ./src/
COPY models/ ./models/

# Переменные окружения
ENV MODEL_PATH=models/engine_model.cbm \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Открываем порт
EXPOSE 8000

# Healthcheck — Docker сам проверит живость сервиса
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Запуск
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
