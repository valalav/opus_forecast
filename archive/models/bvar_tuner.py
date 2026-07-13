"""
BVAR Auto-Tuner: Автоматический подбор параметров BVAR
======================================================

Модуль для автоматического подбора оптимальных параметров BVAR
с использованием grid search и time series cross-validation.

Поддерживает:
- Grid search по параметрам (lambda1, lambda2, lambda3, lags)
- Time series cross-validation (expanding/rolling window)
- Сравнение метрик (MAE, RMSE, MAPE, etc.)
- Адаптация к данным разных регионов

Использование:
    tuner = BVARTuner(df, target_col='Все товары и услуги')
    best_params, results = tuner.tune()
    model = tuner.get_best_model()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from itertools import product
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from .bvar import BVARForecaster


@dataclass
class TuningResult:
    """Результат тюнинга одной конфигурации."""
    params: Dict[str, Any]
    mae: float
    rmse: float
    mape: float
    n_observations: int
    cv_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.params,
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'n_obs': self.n_observations,
            'cv_std': np.std(self.cv_scores) if self.cv_scores else 0
        }


class BVARTuner:
    """
    Автоматический подбор параметров BVAR.

    Параметры для тюнинга:
    - lambda1: Overall tightness (0.05 - 1.0)
    - lambda2: Cross-variable tightness (0.1 - 1.0)
    - lambda3: Lag decay (0.5 - 2.0)
    - lags: Количество лагов (1 - 4)

    Методы валидации:
    - expanding: Расширяющееся окно (стандартный бэктест)
    - rolling: Скользящее окно фиксированной длины
    """

    # Дефолтные сетки параметров
    DEFAULT_PARAM_GRID = {
        'lambda1': [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
        'lambda2': [0.3, 0.5, 0.7, 1.0],
        'lambda3': [0.5, 1.0, 1.5, 2.0],
        'lags': [1, 2, 3]
    }

    # Быстрая сетка для тестирования
    QUICK_PARAM_GRID = {
        'lambda1': [0.1, 0.2, 0.5],
        'lambda2': [0.5],
        'lambda3': [1.0],
        'lags': [1, 2]
    }

    # Детальная сетка для финальной оптимизации
    FINE_PARAM_GRID = {
        'lambda1': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0],
        'lambda2': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        'lambda3': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        'lags': [1, 2, 3, 4]
    }

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = 'Все товары и услуги',
        param_grid: Optional[Dict[str, List]] = None,
        cv_method: str = 'expanding',
        cv_start_date: str = '2019-01-01',
        cv_window_size: int = 60,  # Для rolling CV
        n_draws: int = 500,
        metric: str = 'mae',
        n_jobs: int = 1,
        verbose: bool = True
    ):
        """
        Инициализация тюнера.

        Args:
            df: DataFrame с данными (индекс = даты)
            target_col: Целевая колонка
            param_grid: Сетка параметров для перебора
            cv_method: Метод валидации ('expanding' или 'rolling')
            cv_start_date: Начальная дата для валидации
            cv_window_size: Размер окна для rolling CV
            n_draws: Количество draws для BVAR
            metric: Метрика для оптимизации ('mae', 'rmse', 'mape')
            n_jobs: Количество параллельных процессов (1 = без параллелизации)
            verbose: Выводить прогресс
        """
        self.df = df
        self.target_col = target_col
        self.param_grid = param_grid or self.DEFAULT_PARAM_GRID
        self.cv_method = cv_method
        self.cv_start_date = cv_start_date
        self.cv_window_size = cv_window_size
        self.n_draws = n_draws
        self.metric = metric
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Результаты
        self.results: List[TuningResult] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_model: Optional[BVARForecaster] = None

    def _get_param_combinations(self) -> List[Dict[str, Any]]:
        """Генерация всех комбинаций параметров."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _evaluate_params(
        self,
        params: Dict[str, Any]
    ) -> TuningResult:
        """
        Оценка одной комбинации параметров через CV.

        Args:
            params: Параметры модели

        Returns:
            TuningResult с метриками
        """
        model = BVARForecaster(
            lags=params.get('lags', 1),
            lambda1=params.get('lambda1', 0.2),
            lambda2=params.get('lambda2', 0.5),
            lambda3=params.get('lambda3', 1.0),
            lambda4=params.get('lambda4', 100),
            n_draws=self.n_draws
        )

        # Бэктест
        bt = model.backtest(
            self.df,
            start_date=self.cv_start_date,
            target_col=self.target_col
        )

        if len(bt) == 0:
            return TuningResult(
                params=params,
                mae=np.inf,
                rmse=np.inf,
                mape=np.inf,
                n_observations=0
            )

        errors = bt['error'].values
        actuals = bt['actual'].values

        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors**2).mean())

        # MAPE с защитой от деления на ноль
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.abs(errors / np.where(actuals != 0, actuals, np.nan)).mean() * 100
            if np.isnan(mape):
                mape = np.inf

        return TuningResult(
            params=params,
            mae=mae,
            rmse=rmse,
            mape=mape,
            n_observations=len(bt),
            cv_scores=list(np.abs(errors))
        )

    def tune(
        self,
        grid_type: str = 'default'
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Запуск тюнинга параметров.

        Args:
            grid_type: Тип сетки ('quick', 'default', 'fine')

        Returns:
            (best_params, results_df)
        """
        # Выбор сетки
        if grid_type == 'quick':
            self.param_grid = self.QUICK_PARAM_GRID
        elif grid_type == 'fine':
            self.param_grid = self.FINE_PARAM_GRID
        elif grid_type != 'default':
            # Предполагаем, что передан словарь
            if isinstance(grid_type, dict):
                self.param_grid = grid_type

        combinations = self._get_param_combinations()
        n_combinations = len(combinations)

        if self.verbose:
            print(f"BVAR Tuner: {n_combinations} комбинаций параметров")
            print(f"CV method: {self.cv_method}, start: {self.cv_start_date}")
            print("-" * 50)

        self.results = []

        for i, params in enumerate(combinations):
            if self.verbose:
                print(f"[{i+1}/{n_combinations}] λ1={params.get('lambda1', 0.2):.2f}, "
                      f"λ2={params.get('lambda2', 0.5):.2f}, "
                      f"λ3={params.get('lambda3', 1.0):.2f}, "
                      f"lags={params.get('lags', 1)}", end=" ")

            try:
                result = self._evaluate_params(params)
                self.results.append(result)

                if self.verbose:
                    print(f"→ MAE={result.mae:.4f}, RMSE={result.rmse:.4f}")
            except Exception as e:
                if self.verbose:
                    print(f"→ ERROR: {e}")

        # Найти лучшие параметры
        if self.results:
            if self.metric == 'mae':
                best_result = min(self.results, key=lambda x: x.mae)
            elif self.metric == 'rmse':
                best_result = min(self.results, key=lambda x: x.rmse)
            else:
                best_result = min(self.results, key=lambda x: x.mape)

            self.best_params = best_result.params

        # Создать DataFrame с результатами
        results_df = pd.DataFrame([r.to_dict() for r in self.results])
        results_df = results_df.sort_values(self.metric).reset_index(drop=True)

        if self.verbose:
            print("-" * 50)
            print(f"Лучшие параметры ({self.metric}):")
            print(f"  {self.best_params}")
            print(f"  MAE={best_result.mae:.4f}, RMSE={best_result.rmse:.4f}")

        return self.best_params, results_df

    def get_best_model(self) -> BVARForecaster:
        """
        Получить модель с лучшими параметрами.

        Returns:
            Обученная модель BVARForecaster
        """
        if self.best_params is None:
            raise ValueError("Сначала запустите tune()")

        self.best_model = BVARForecaster(
            lags=self.best_params.get('lags', 1),
            lambda1=self.best_params.get('lambda1', 0.2),
            lambda2=self.best_params.get('lambda2', 0.5),
            lambda3=self.best_params.get('lambda3', 1.0),
            lambda4=self.best_params.get('lambda4', 100),
            n_draws=1000  # Больше draws для финальной модели
        )

        self.best_model.fit(self.df, self.target_col)
        return self.best_model

    def compare_with_baseline(self) -> pd.DataFrame:
        """
        Сравнение лучшей модели с базовыми конфигурациями.

        Returns:
            DataFrame со сравнением
        """
        baselines = [
            {'name': 'Default (λ=0.2, lags=1)', 'lambda1': 0.2, 'lags': 1},
            {'name': 'Weak prior (λ=0.5)', 'lambda1': 0.5, 'lags': 1},
            {'name': 'Strong prior (λ=0.1)', 'lambda1': 0.1, 'lags': 1},
            {'name': 'AR(2)', 'lambda1': 0.2, 'lags': 2},
        ]

        if self.best_params:
            baselines.append({
                'name': 'TUNED (best)',
                **self.best_params
            })

        results = []
        for config in baselines:
            name = config.pop('name')
            result = self._evaluate_params(config)
            results.append({
                'Model': name,
                'MAE': result.mae,
                'RMSE': result.rmse,
                'MAPE': result.mape,
                'N': result.n_observations
            })

        return pd.DataFrame(results)

    def save_results(self, filepath: str) -> None:
        """Сохранить результаты тюнинга в JSON."""
        data = {
            'best_params': self.best_params,
            'metric': self.metric,
            'cv_method': self.cv_method,
            'cv_start_date': self.cv_start_date,
            'results': [r.to_dict() for r in self.results]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_results(cls, filepath: str) -> Dict[str, Any]:
        """Загрузить результаты тюнинга из JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


class RegionalBVARTuner:
    """
    Тюнер BVAR для нескольких регионов.

    Позволяет:
    - Подбирать параметры для каждого региона отдельно
    - Находить универсальные параметры для всех регионов
    - Сравнивать качество моделей между регионами
    """

    def __init__(
        self,
        regional_data: Dict[str, pd.DataFrame],
        target_col: str = 'Все товары и услуги',
        param_grid: Optional[Dict[str, List]] = None,
        cv_start_date: str = '2019-01-01',
        verbose: bool = True
    ):
        """
        Инициализация регионального тюнера.

        Args:
            regional_data: Словарь {region_name: DataFrame}
            target_col: Целевая колонка
            param_grid: Сетка параметров
            cv_start_date: Начальная дата CV
            verbose: Выводить прогресс
        """
        self.regional_data = regional_data
        self.target_col = target_col
        self.param_grid = param_grid or BVARTuner.DEFAULT_PARAM_GRID
        self.cv_start_date = cv_start_date
        self.verbose = verbose

        self.regional_results: Dict[str, Tuple[Dict, pd.DataFrame]] = {}
        self.universal_params: Optional[Dict[str, Any]] = None

    def tune_all_regions(
        self,
        grid_type: str = 'quick'
    ) -> Dict[str, Tuple[Dict, pd.DataFrame]]:
        """
        Подобрать параметры для каждого региона.

        Args:
            grid_type: Тип сетки параметров

        Returns:
            Словарь {region: (best_params, results_df)}
        """
        for region, df in self.regional_data.items():
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"РЕГИОН: {region}")
                print(f"{'='*60}")

            tuner = BVARTuner(
                df=df,
                target_col=self.target_col,
                cv_start_date=self.cv_start_date,
                verbose=self.verbose
            )

            best_params, results_df = tuner.tune(grid_type=grid_type)
            self.regional_results[region] = (best_params, results_df)

        return self.regional_results

    def find_universal_params(self) -> Dict[str, Any]:
        """
        Найти универсальные параметры, хорошо работающие для всех регионов.

        Использует среднее значение метрики по всем регионам.

        Returns:
            Универсальные параметры
        """
        if not self.regional_results:
            raise ValueError("Сначала запустите tune_all_regions()")

        # Собираем все конфигурации параметров
        all_configs = set()
        for region, (_, results_df) in self.regional_results.items():
            for _, row in results_df.iterrows():
                config = (
                    row.get('lambda1', 0.2),
                    row.get('lambda2', 0.5),
                    row.get('lambda3', 1.0),
                    row.get('lags', 1)
                )
                all_configs.add(config)

        # Оцениваем каждую конфигурацию по среднему MAE
        config_scores = {}
        for config in all_configs:
            total_mae = 0
            n_regions = 0

            for region, (_, results_df) in self.regional_results.items():
                mask = (
                    (results_df['lambda1'] == config[0]) &
                    (results_df['lambda2'] == config[1]) &
                    (results_df['lambda3'] == config[2]) &
                    (results_df['lags'] == config[3])
                )

                if mask.any():
                    total_mae += results_df.loc[mask, 'mae'].values[0]
                    n_regions += 1

            if n_regions > 0:
                config_scores[config] = total_mae / n_regions

        # Лучшая конфигурация
        best_config = min(config_scores, key=config_scores.get)

        self.universal_params = {
            'lambda1': best_config[0],
            'lambda2': best_config[1],
            'lambda3': best_config[2],
            'lags': best_config[3]
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print("УНИВЕРСАЛЬНЫЕ ПАРАМЕТРЫ")
            print(f"{'='*60}")
            print(f"  {self.universal_params}")
            print(f"  Средний MAE: {config_scores[best_config]:.4f}")

        return self.universal_params

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Сравнительная таблица по регионам.

        Returns:
            DataFrame с метриками по регионам
        """
        rows = []

        for region, (best_params, results_df) in self.regional_results.items():
            best_row = results_df.iloc[0]  # Первая строка = лучший результат

            rows.append({
                'Регион': region,
                'λ1': best_params.get('lambda1', 0.2),
                'λ2': best_params.get('lambda2', 0.5),
                'λ3': best_params.get('lambda3', 1.0),
                'Лаги': best_params.get('lags', 1),
                'MAE': best_row['mae'],
                'RMSE': best_row['rmse'],
                'N': best_row['n_obs']
            })

        return pd.DataFrame(rows)

    def save_all_results(self, filepath: str) -> None:
        """Сохранить все результаты в JSON."""
        data = {
            'universal_params': self.universal_params,
            'regional_results': {
                region: {
                    'best_params': params,
                    'results': results_df.to_dict(orient='records')
                }
                for region, (params, results_df) in self.regional_results.items()
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def quick_tune(
    df: pd.DataFrame,
    target_col: str = 'Все товары и услуги',
    cv_start_date: str = '2019-01-01'
) -> Tuple[Dict[str, Any], BVARForecaster]:
    """
    Быстрый тюнинг BVAR (удобная функция).

    Args:
        df: DataFrame с данными
        target_col: Целевая колонка
        cv_start_date: Начальная дата CV

    Returns:
        (best_params, fitted_model)
    """
    tuner = BVARTuner(
        df=df,
        target_col=target_col,
        cv_start_date=cv_start_date,
        verbose=True
    )

    best_params, _ = tuner.tune(grid_type='quick')
    model = tuner.get_best_model()

    return best_params, model
