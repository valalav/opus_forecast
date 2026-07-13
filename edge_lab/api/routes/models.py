"""
Models endpoint for listing available forecasters with performance metrics.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import os

router = APIRouter(prefix="/models", tags=["Models"])


class ModelInfo(BaseModel):
    """Information about a forecasting model."""

    name: str
    description: Optional[str] = None
    mae: Optional[float] = None
    weight: Optional[float] = None
    is_production: bool = False


class ModelsListResponse(BaseModel):
    """Response containing list of available models."""

    models: List[ModelInfo]
    total_count: int


def _load_mae_metrics() -> Dict[str, float]:
    """
    Load MAE metrics from backtest results.

    Returns:
        Dictionary mapping model names to their MAE values
    """
    try:
        edge_lab_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        opus_forecast_dir = os.path.dirname(edge_lab_dir)

        metrics_path = os.path.join(
            opus_forecast_dir, "archive", "results", "backtest_h1_metrics.csv"
        )
        df = pd.read_csv(metrics_path)
        return dict(zip(df["Model"], df["MAE"]))
    except Exception as e:
        return {}


def _get_model_descriptions() -> Dict[str, str]:
    """
    Get descriptions for known models.

    Returns:
        Dictionary mapping model names to descriptions
    """
    descriptions = {
        "Ridge": "Ridge regression with ETS seasonality (baseline)",
        "RidgeExtended": "Ridge with extended calendar features",
        "RidgeShock": "Ridge with shock dummy variables from CBR methodology",
        "RidgeMacro": "Ridge with macro features (Ki, USD, Brent)",
        "BayesianRidge": "Ridge with automatic regularization and CI",
        "ElasticNet": "L1+L2 regularization with automatic feature selection",
        "Huber": "Robust regression resistant to outliers",
        "NGBoost": "Probabilistic gradient boosting with natural gradients",
        "NGBoostShock": "NGBoost with shock dummy variables",
        "LMMR": "Local Multivariate Matrix Regression",
        "BVAR": "Bayesian VAR with Minnesota Prior",
        "SARIMA": "Seasonal ARIMA",
        "ETS": "Exponential Smoothing",
        "Prophet": "Facebook Prophet with seasonality",
        "LightGBM": "Gradient boosting with leaf-wise growth",
        "CatBoost": "Gradient boosting optimized for small samples",
        "EBM": "Explainable Boosting Machine",
        "Stacking": "Meta-learning on base model predictions",
        "Subcomp": "Bottom-up from 3 inflation components",
        "SubcompMulti": "Multi-model bottom-up from subcomponents",
        "Microcomponent": "Bottom-up from 497 micro-components",
        "HierarchicalMicro": "Hierarchical reconciliation micro → subcomp → comp → total",
        "HorizonEnsemble": "Adaptive Huber + Micro ensemble",
        "Conformal": "Calibrated prediction intervals",
        "MicroARIMA": "External micro-component ARIMA model",
        "Fundamental": "Structural model on economic drivers (USD, Oil, Key Rate)",
        "USDForecaster": "ElasticNet for USD/RUB (Oil + Rates)",
    }
    return descriptions


@router.get("")
@router.get("/")
async def list_models():
    """
    List all available forecasting models with performance metrics.

    Returns:
        JSON array of models with name, description, MAE, and ensemble weight
    """
    try:
        import sys

        edge_lab_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        opus_forecast_dir = os.path.dirname(edge_lab_dir)

        sys.path.insert(0, edge_lab_dir)
        sys.path.insert(0, opus_forecast_dir)

        from sirena.models.registry import ModelRegistry

        registered_models = ModelRegistry.list_models()
        mae_metrics = _load_mae_metrics()
        descriptions = _get_model_descriptions()

        try:
            weights = ModelRegistry._default_weights
        except:
            weights = {}

        models_list = []
        for model_name in registered_models:
            model_info = ModelInfo(
                name=model_name,
                description=descriptions.get(
                    model_name, f"Forecasting model: {model_name}"
                ),
                mae=mae_metrics.get(model_name),
                weight=weights.get(model_name),
                is_production=model_name in weights,
            )
            models_list.append(model_info)

        return ModelsListResponse(models=models_list, total_count=len(models_list))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")


@router.get("/leaderboard")
async def get_leaderboard():
    """
    Get performance leaderboard ranked by MAE.

    Returns:
        Models sorted by MAE (ascending)
    """
    try:
        import sys

        edge_lab_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        opus_forecast_dir = os.path.dirname(edge_lab_dir)

        sys.path.insert(0, edge_lab_dir)
        sys.path.insert(0, opus_forecast_dir)

        mae_metrics = _load_mae_metrics()
        descriptions = _get_model_descriptions()

        leaderboard = []
        for model_name, mae in mae_metrics.items():
            leaderboard.append(
                {
                    "model": model_name,
                    "mae": mae,
                    "description": descriptions.get(model_name, f"Model: {model_name}"),
                }
            )

        leaderboard.sort(key=lambda x: x["mae"])

        return {"leaderboard": leaderboard, "total_count": len(leaderboard)}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading leaderboard: {str(e)}"
        )
