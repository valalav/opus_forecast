"""
СИРЕНА-КБР REST API v4.0
=========================

FastAPI приложение для прогнозирования инфляции КБР.

Запуск:
    uvicorn api.main:app --reload --port 8000

Документация:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import (
    forecast_router,
    models_router,
    backtest_router,
    weekly_router,
    batch_router,
    health_router,
    metrics_router,
)
from api.schemas.models import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    # Startup: загрузка моделей
    print("СИРЕНА-КБР API v4.0 запускается...")

    # Track uptime
    try:
        from api.routes.health import set_start_time

        set_start_time()
    except Exception as e:
        print(f"Warning: Could not set start time for uptime tracking: {e}")

    # Initialize Prometheus metrics with MAE values
    try:
        from api.routes.metrics import initialize_model_metrics

        initialize_model_metrics()
        print("Prometheus metrics initialized")
    except Exception as e:
        print(f"Warning: Could not initialize Prometheus metrics: {e}")

    try:
        from sirena.models import ModelRegistry

        models = ModelRegistry.list_models()
        print(f"Загружено моделей: {len(models)}")
        print(f"Модели: {', '.join(models)}")
    except Exception as e:
        print(f"Ошибка загрузки моделей: {e}")

    yield

    # Shutdown
    print("СИРЕНА-КБР API остановлен")


# Создаём приложение
app = FastAPI(
    title="СИРЕНА-КБР API",
    description="""
# API прогнозирования инфляции КБР

Система Интеллектуального Регионального Анализа — Кабардино-Балкарская Республика.

## Возможности

* **Прогнозирование** — production ensemble + auxiliary модели
* **Недельные цены** — прогноз и nowcasting сигнал
* **Бэктестирование** — оценка качества моделей
* **Кастомизация** — выбор моделей и весов

## Модели

**Production ensemble:** Huber, RidgeShockDummies, ElasticNet, NGBoostShock,
NGBoost, Ridge, RidgeExtended, Prophet, EBM.

**Auxiliary models (not in default ensemble):** BVAR, SARIMA, ETS,
LightGBM and other analysis models.
    """,
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Роуты
app.include_router(forecast_router)
app.include_router(models_router)
app.include_router(backtest_router)
app.include_router(weekly_router)
app.include_router(batch_router)
app.include_router(health_router)
app.include_router(metrics_router)


@app.get("/", tags=["Health"])
async def root():
    """Корневой эндпоинт."""
    return {
        "name": "СИРЕНА-КБР API",
        "version": "4.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# Old /health endpoint moved to api/routes/health.py with uptime tracking
# @app.get("/health", response_model=HealthResponse, tags=["Health"])
# async def health_check():
#     """
#     Проверка статуса сервера.
#
#     Возвращает информацию о доступности API и загруженных моделях.
#     """
#     try:
#         from sirena.models import ModelRegistry
#
#         models_count = len(ModelRegistry.list_models())
#
#         # Проверяем данные
#         data_loaded = False
#         data_paths = [
#             "data/infl_kbr.csv",
#             os.path.join(os.getcwd(), "data/infl_kbr.csv"),
#         ]
#         for path in data_paths:
#             if os.path.exists(path):
#                 data_loaded = True
#                 break
#
#         return HealthResponse(
#             status="ok",
#             version="4.0.0",
#             models_available=models_count,
#             data_loaded=data_loaded,
#         )
#
#     except Exception as e:
#         return HealthResponse(
#             status="error", version="4.0.0", models_available=0, data_loaded=False
#         )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
