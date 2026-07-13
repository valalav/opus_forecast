from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes.explain import router as explain_router
from api.routes.batch import router as batch_router
from api.routes.health import router as health_router, set_start_time
from api.routes.models import router as models_router
from api.routes.metrics import router as metrics_router, initialize_model_metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    # Startup: track start time for uptime
    set_start_time()
    # Initialize Prometheus metrics with current model MAE values
    initialize_model_metrics()
    yield
    # Shutdown: cleanup if needed
    pass


app = FastAPI(
    title="Opus Edge Lab API",
    description="API for Sirena-KBR forecasting system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(explain_router)
app.include_router(batch_router)
app.include_router(health_router)
app.include_router(models_router)
app.include_router(metrics_router)


@app.get("/")
async def root():
    return {
        "message": "Opus Edge Lab API",
        "version": "1.0.0",
        "endpoints": {
            "root": "/",
            "health": "/health",
            "explain": "/explain/",
            "batch": "/forecast/batch/",
            "models": "/models/",
            "metrics": "/metrics",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
