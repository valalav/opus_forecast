"""
Асинхронный запуск моделей СИРЕНА-КБР
======================================

Параллельное выполнение моделей для ускорения прогнозирования.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from functools import partial
import time


def _run_model_sync(model_name: str, df_dict: Dict, horizon: int) -> Dict[str, Any]:
    """
    Синхронный запуск одной модели (для ProcessPoolExecutor).

    Args:
        model_name: Название модели
        df_dict: DataFrame как словарь (для сериализации)
        horizon: Горизонт прогноза

    Returns:
        Dict с результатами
    """
    try:
        from sirena.models import ModelRegistry

        # Восстанавливаем DataFrame
        df = pd.DataFrame(df_dict)
        df.index = pd.to_datetime(df.index)

        # Запускаем модель
        model = ModelRegistry.get(model_name)
        start_time = time.time()
        model.fit(df)
        forecast = model.forecast(horizon=horizon)
        elapsed = time.time() - start_time

        return {
            'model': model_name,
            'forecast': forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast),
            'elapsed': elapsed,
            'success': True
        }

    except Exception as e:
        return {
            'model': model_name,
            'forecast': None,
            'error': str(e),
            'success': False
        }


class AsyncModelRunner:
    """
    Асинхронный запуск моделей.

    Использует ProcessPoolExecutor для CPU-bound моделей
    и ThreadPoolExecutor для IO-bound операций.

    Example:
        runner = AsyncModelRunner(max_workers=4)
        results = await runner.run_models(['ridge', 'bvar'], df, horizon=12)
    """

    def __init__(self, max_workers: int = 4, use_processes: bool = True):
        """
        Инициализация.

        Args:
            max_workers: Максимум параллельных воркеров
            use_processes: Использовать процессы (True) или потоки (False)
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self._executor = None

    def _get_executor(self):
        """Создание executor при необходимости."""
        if self._executor is None:
            if self.use_processes:
                self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    async def run_model(
        self,
        model_name: str,
        df: pd.DataFrame,
        horizon: int = 12
    ) -> Dict[str, Any]:
        """
        Асинхронный запуск одной модели.

        Args:
            model_name: Название модели
            df: DataFrame с данными
            horizon: Горизонт прогноза

        Returns:
            Dict с результатами
        """
        loop = asyncio.get_event_loop()
        executor = self._get_executor()

        # Конвертируем DataFrame в словарь для сериализации
        df_dict = df.to_dict()

        result = await loop.run_in_executor(
            executor,
            partial(_run_model_sync, model_name, df_dict, horizon)
        )

        return result

    async def run_models(
        self,
        models: List[str],
        df: pd.DataFrame,
        horizon: int = 12,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Параллельный запуск нескольких моделей.

        Args:
            models: Список моделей
            df: DataFrame с данными
            horizon: Горизонт прогноза
            weights: Веса моделей (опционально)

        Returns:
            Dict с результатами всех моделей и ансамблем
        """
        start_time = time.time()

        # Запускаем все модели параллельно
        tasks = [self.run_model(m, df, horizon) for m in models]
        results = await asyncio.gather(*tasks)

        # Обрабатываем результаты
        model_results = {}
        successful_forecasts = []
        successful_weights = []

        for result in results:
            model_name = result['model']
            model_results[model_name] = result

            if result['success']:
                fc = np.array(result['forecast'])
                successful_forecasts.append(fc)

                if weights:
                    successful_weights.append(weights.get(model_name, 0))
                else:
                    successful_weights.append(1.0)

        # Вычисляем ансамбль
        ensemble = None
        if successful_forecasts:
            # Нормализуем веса
            total_weight = sum(successful_weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in successful_weights]
            else:
                normalized_weights = [1.0 / len(successful_forecasts)] * len(successful_forecasts)

            ensemble = np.zeros(horizon)
            for fc, w in zip(successful_forecasts, normalized_weights):
                ensemble += fc * w

        total_elapsed = time.time() - start_time

        return {
            'models': model_results,
            'ensemble': ensemble.tolist() if ensemble is not None else None,
            'total_elapsed': total_elapsed,
            'successful_count': len(successful_forecasts),
            'failed_count': len(models) - len(successful_forecasts)
        }

    def close(self):
        """Закрытие executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()


async def run_ensemble_async(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    horizon: int = 12,
    weights: Optional[Dict[str, float]] = None,
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Удобная функция для запуска ансамбля.

    Args:
        df: DataFrame с данными
        models: Список моделей (если None — все)
        horizon: Горизонт прогноза
        weights: Веса моделей
        max_workers: Количество воркеров

    Returns:
        Результаты прогнозирования

    Example:
        results = await run_ensemble_async(df, horizon=12)
        print(results['ensemble'])
    """
    from sirena.models import ModelRegistry

    if models is None:
        models = ModelRegistry.list_models()

    if weights is None:
        weights = {m: ModelRegistry.get_default_weight(m) for m in models}

    async with AsyncModelRunner(max_workers=max_workers) as runner:
        return await runner.run_models(models, df, horizon, weights)


# Синхронная обёртка
def run_ensemble_parallel(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    horizon: int = 12,
    weights: Optional[Dict[str, float]] = None,
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Синхронная обёртка для run_ensemble_async.

    Использует asyncio.run() для запуска.
    """
    return asyncio.run(run_ensemble_async(df, models, horizon, weights, max_workers))


if __name__ == "__main__":
    # Тестирование
    import sys
    sys.path.insert(0, '..')

    async def test():
        # Загрузка данных
        df = pd.read_csv('data/infl_kbr.csv', sep=';', decimal='.')
        df['Date'] = pd.to_datetime(df['Day'], format='%d.%m.%Y')
        df = df.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        df = df.sort_index()

        print("Тестирование AsyncModelRunner...")

        # Только Ridge и BVAR для быстрого теста
        results = await run_ensemble_async(
            df,
            models=['ridge', 'bvar'],
            horizon=6,
            max_workers=2
        )

        print(f"\nУспешно: {results['successful_count']}")
        print(f"Ошибки: {results['failed_count']}")
        print(f"Время: {results['total_elapsed']:.2f}с")
        print(f"Ансамбль: {results['ensemble']}")

    asyncio.run(test())
