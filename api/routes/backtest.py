"""
Эндпоинты для бэктестирования
"""

from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np

from ..schemas.backtest import (
    BacktestRequest,
    BacktestResponse,
    BacktestResult,
    BacktestMetrics,
    HistoryRequest,
    HistoryEntry,
    HistoryResponse,
)

router = APIRouter(prefix="/backtest", tags=["Backtest"])


def get_data():
    """Загрузка данных."""
    import os

    data_paths = ["data/infl_kbr.csv", "/home/valalav/_projects/sirena-kbr/data/infl_kbr.csv"]

    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, sep=";", decimal=",")
            df["Date"] = pd.to_datetime(df["Day"], format="%d.%m.%Y")

            if "Товар" in df.columns:
                pivot = df.pivot_table(
                    index="Date", columns="Товар", values="MoM", aggfunc="first"
                )
            else:
                pivot = df.set_index("Date")

            return pivot.sort_index()

    raise HTTPException(status_code=500, detail="Данные не найдены")


@router.post("", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """
    Запустить бэктест модели.

    Выполняет скользящий бэктест с горизонтом H=1.
    """
    try:
        from sirena.models import ModelRegistry

        if not ModelRegistry.is_registered(request.model):
            raise HTTPException(
                status_code=404, detail=f"Модель '{request.model}' не найдена"
            )

        df = get_data()
        model = ModelRegistry.get(request.model)

        # Запуск бэктеста
        results_df = model.backtest(
            df, start_date=request.start_date, target_col="Все товары и услуги"
        )

        if results_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Бэктест не дал результатов (недостаточно данных)",
            )

        # Фильтруем по end_date если указан
        if request.end_date:
            results_df = results_df[results_df["date"] <= request.end_date]

        # Вычисляем метрики
        errors = results_df["error"].abs()
        mae = float(errors.mean())
        rmse = float(np.sqrt((results_df["error"] ** 2).mean()))
        kpi = float((errors <= 0.5).sum() / len(results_df) * 100)

        # Форматируем результаты
        results = [
            BacktestResult(
                date=row["date"].strftime("%Y-%m"),
                actual=round(float(row["actual"]), 3),
                prediction=round(float(row["prediction"]), 3),
                error=round(float(row["error"]), 3),
            )
            for _, row in results_df.iterrows()
        ]

        return BacktestResponse(
            model=request.model,
            start_date=results_df["date"].min().strftime("%Y-%m-%d"),
            end_date=results_df["date"].max().strftime("%Y-%m-%d"),
            metrics=BacktestMetrics(
                MAE=round(mae, 4),
                RMSE=round(rmse, 4),
                KPI=round(kpi, 1),
                count=len(results),
            ),
            results=results,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{model_name}")
async def get_backtest_metrics(model_name: str, start_date: str = "2019-01-01"):
    """
    Получить только метрики бэктеста (без детальных результатов).
    """
    try:
        from sirena.models import ModelRegistry

        if not ModelRegistry.is_registered(model_name):
            raise HTTPException(
                status_code=404, detail=f"Модель '{model_name}' не найдена"
            )

        df = get_data()
        model = ModelRegistry.get(model_name)

        results_df = model.backtest(df, start_date=start_date)

        if results_df.empty:
            return {"error": "Нет данных для бэктеста"}

        errors = results_df["error"].abs()

        return {
            "model": model_name,
            "MAE": round(float(errors.mean()), 4),
            "RMSE": round(float(np.sqrt((results_df["error"] ** 2).mean())), 4),
            "KPI": round(float((errors <= 0.5).sum() / len(results_df) * 100), 1),
            "count": len(results_df),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    start_date: str = None, end_date: str = None, model: str = None, horizon: int = 1
):
    """
    Получить исторические прогнозы из бэктеста.

    Загружает данные из CSV файлов с результатами бэктестов.
    """
    try:
        import os

        # Проверяем корректность горизонта
        if horizon not in [1, 2, 12]:
            raise HTTPException(
                status_code=400, detail="Горизонт должен быть 1, 2 или 12"
            )

        # Определяем путь к файлу
        backtest_file = f"/home/valalav/_projects/sirena-kbr/archive/results/backtest_h{horizon}_predictions.csv"

        if not os.path.exists(backtest_file):
            raise HTTPException(
                status_code=404, detail=f"Файл для горизонта h={horizon} не найден"
            )

        # Загружаем данные
        df = pd.read_csv(backtest_file)
        df["Date"] = pd.to_datetime(df["Date"])

        # Фильтр по дате
        if start_date:
            df = df[df["Date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["Date"] <= pd.to_datetime(end_date)]

        if df.empty:
            raise HTTPException(
                status_code=404, detail="Нет данных для указанного диапазона дат"
            )

        # Фильтр по модели
        if model:
            model_columns = [col for col in df.columns if model.lower() in col.lower()]
            if not model_columns:
                available_models = [
                    col for col in df.columns if col not in ["Date", "Actual"]
                ]
                raise HTTPException(
                    status_code=404,
                    detail=f"Модель '{model}' не найдена. Доступные: {', '.join(available_models[:5])}...",
                )
            # Используем первое совпадение
            model_col = model_columns[0]
        else:
            # Если модель не указана, возвращаем Ensemble
            model_col = "Ensemble" if "Ensemble" in df.columns else df.columns[2]

        # Вычисляем ошибку
        df["error"] = df["Actual"] - df[model_col]

        # Формируем результат
        entries = [
            HistoryEntry(
                date=row["Date"].strftime("%Y-%m-%d"),
                actual=round(float(row["Actual"]), 4),
                prediction=round(float(row[model_col]), 4),
                error=round(float(row["error"]), 4),
            )
            for _, row in df.iterrows()
        ]

        return HistoryResponse(
            count=len(entries),
            horizon=horizon,
            model=model,
            start_date=df["Date"].min().strftime("%Y-%m-%d"),
            end_date=df["Date"].max().strftime("%Y-%m-%d"),
            data=entries,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
