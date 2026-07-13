"""
Эндпоинты для информации о моделях
"""

from fastapi import APIRouter, HTTPException
from typing import List

from ..schemas.models import ModelInfo

router = APIRouter(prefix="/models", tags=["Models"])

# Описания моделей
MODEL_DESCRIPTIONS = {
    "ridge": "Ridge регрессия с ETS сезонной компонентой",
    "bvar": "Байесовская VAR с Minnesota Prior",
    "lightgbm": "Gradient Boosting для нелинейных зависимостей",
    "prophet": "Facebook Prophet с автоматической сезонностью",
    "sarima": "Сезонная авторегрессия SARIMA",
    "ets": "Exponential Smoothing (Holt-Winters)",
    "lstm": "Рекуррентная нейронная сеть LSTM",
    "ar1": "Простая авторегрессия AR(1)",
}

# Mapping between backtest model names and registry model names
BACKTEST_TO_REGISTRY_MAP = {
    "Subcomp": "subcomponent",
    "Subcomp_Multi": "subcomp_multi",
    "Ridge": "ridge",
    "Ridge_Macro": "ridge_macro",
    "Ridge_Shock": "ridge_shock_dummies",
    "Ridge_Ext": "ridge_extended",
    "Huber": "huber",
    "Ensemble": "ensemble",
    "EBM": "ebm",
    "CatBoost": "catboost",
    "Bayes_Ridge": "bayesian_ridge",
    "Prophet": "prophet",
    "ElasticNet": "elasticnet",
    "LightGBM": "lightgbm",
    "NGBoost": "ngboost",
    "NGBoost_Shock": "ngboost_shock",
    "SARIMA": "sarima",
    "ETS": "ets",
    "BVAR": "bvar",
    "Micro": "micro_arima",
    "MIDAS": "midas",
}


@router.get("", response_model=List[ModelInfo])
async def list_models():
    """
    Получить список всех доступных моделей.

    Возвращает информацию о каждой модели, включая вес в ансамбле и MAE из backtest h=1.
    """
    try:
        from sirena.models import ModelRegistry
        import pandas as pd
        import os

        # Load MAE from backtest results (h=1 is main KPI)
        mae_data = {}
        backtest_file = os.path.join(os.getcwd(), "archive/results/backtest_h1_metrics.csv")
        if os.path.exists(backtest_file):
            df = pd.read_csv(backtest_file)
            # Map backtest names to registry names
            for backtest_name, mae in zip(df["Model"], df["MAE"]):
                registry_name = BACKTEST_TO_REGISTRY_MAP.get(backtest_name)
                if registry_name:
                    mae_data[registry_name] = mae

        models = []

        for name in ModelRegistry.list_models():
            weight = ModelRegistry.get_default_weight(name)
            model_class = ModelRegistry.get_class(name)

            # Get MAE from backtest results, or None if not available
            mae_value = mae_data.get(name)
            mae = round(mae_value, 4) if mae_value is not None else None

            models.append(
                ModelInfo(
                    name=name,
                    weight=weight,
                    min_train_size=getattr(model_class, "MIN_TRAIN_SIZE", 24),
                    description=MODEL_DESCRIPTIONS.get(name, "Модель прогнозирования"),
                    mae=mae,
                )
            )

        return sorted(models, key=lambda x: -x.weight)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """
    Получить информацию о конкретной модели.
    """
    try:
        from sirena.models import ModelRegistry
        import pandas as pd
        import os

        if not ModelRegistry.is_registered(model_name):
            raise HTTPException(
                status_code=404, detail=f"Модель '{model_name}' не найдена"
            )

        # Load MAE from backtest results
        mae = None
        backtest_file = os.path.join(os.getcwd(), "archive/results/backtest_h1_metrics.csv")
        if os.path.exists(backtest_file):
            df = pd.read_csv(backtest_file)
            # Find backtest name for this registry name
            backtest_name = None
            for bt_name, reg_name in BACKTEST_TO_REGISTRY_MAP.items():
                if reg_name == model_name:
                    backtest_name = bt_name
                    break
            if backtest_name:
                model_row = df[df["Model"] == backtest_name]
                if not model_row.empty:
                    mae = round(float(model_row.iloc[0]["MAE"]), 4)

        model_class = ModelRegistry.get_class(model_name)
        weight = ModelRegistry.get_default_weight(model_name)

        return ModelInfo(
            name=model_name,
            weight=weight,
            min_train_size=getattr(model_class, "MIN_TRAIN_SIZE", 24),
            description=MODEL_DESCRIPTIONS.get(model_name, "Модель прогнозирования"),
            mae=mae,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
