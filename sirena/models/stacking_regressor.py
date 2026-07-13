"""
StackingRegressor модель для прогнозирования инфляции КБР
========================================================

Meta-модель на базе sklearn.ensemble.StackingRegressor.
Комбинирует Ridge и EBM с помощью LinearRegression.

Архитектура:
    Level 1 (Base Models): Ridge + EBM
    Level 2 (Meta Learner): LinearRegression

Особенности:
- Автоматический весовой подбор через meta-learner
- Комбинация strengths разных моделей
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from typing import Dict, Any, List, Optional

import sys

# Add parent directory for main sirena imports (RidgeForecaster, EBMForecaster)
sys.path.insert(0, "/home/valalav/_projects/sirena-kbr")

from sirena.models.base import BaseForecaster
from sirena.models.ridge import RidgeForecaster
from sirena.models.ebm import EBMForecaster

from .registry import ModelRegistry


@ModelRegistry.register("stacking_regressor")
class StackingRegressorForecaster(BaseForecaster):
    """
    StackingRegressor с Ridge + EBM -> LinearRegression.

    Base Models:
    - RidgeForecaster: регуляризованная линейная модель
    - EBMForecaster: Explainable Boosting Machine (GAM)

    Meta Learner:
    - LinearRegression: оптимальная комбинация базовых моделей

    Особенности:
    - Автоматическое взвешивание через обучение meta-learner
    - Captures synergies между моделями
    """

    name = "stacking_regressor"
    MIN_TRAIN_SIZE = 36

    # Годы-выбросы (как в Ridge/EBM)
    OUTLIER_YEARS = [2022, 2010]

    # Базовые признаки (совместимы с Ridge)
    BASE_FEATURES = [
        "y_lag1",
        "y_lag2",
        "y_lag12",
        "y_ma3",
        "month_sin",
        "month_cos",
        "food_lag1",
        "nonfood_lag1",
        "services_lag1",
    ]

    def __init__(self, ridge_alpha: float = 0.3, ebm_max_bins: int = 256, **kwargs):
        """
        Инициализация модели.

        Args:
            ridge_alpha: Ridge регуляризация
            ebm_max_bins: EBM максимальное количество бинов
        """
        super().__init__(**kwargs)

        self.ridge_alpha = ridge_alpha
        self.ebm_max_bins = ebm_max_bins

        self.stacking = None
        self.scaler = None
        self.seasonal_norm = None
        self._features = None
        self._ridge_forecaster = None
        self._ebm_forecaster = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков (как в Ridge)."""
        df = df.copy()

        df["month"] = df.index.month
        df["year"] = df.index.year

        # Лаги целевой переменной
        df["y_lag1"] = df["Все товары и услуги"].shift(1)
        df["y_lag2"] = df["Все товары и услуги"].shift(2)
        df["y_lag12"] = df["Все товары и услуги"].shift(12)

        # Скользящее среднее
        df["y_ma3"] = df["Все товары и услуги"].rolling(3).mean().shift(1)

        # Сезонные признаки
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Лаги компонентов
        if "Продовольственные товары" in df.columns:
            df["food_lag1"] = df["Продовольственные товары"].shift(1)
        else:
            df["food_lag1"] = df["y_lag1"]

        if "Непродовольственные товары" in df.columns:
            df["nonfood_lag1"] = df["Непродовольственные товары"].shift(1)
        else:
            df["nonfood_lag1"] = df["y_lag1"]

        if "Услуги" in df.columns:
            df["services_lag1"] = df["Услуги"].shift(1)
        else:
            df["services_lag1"] = df["y_lag1"]

        return df

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        """Вычисление сезонной нормы без выбросных лет."""
        clean_df = df[~df["year"].isin(self.OUTLIER_YEARS)]
        return clean_df.groupby("month")["Все товары и услуги"].mean()

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "StackingRegressorForecaster":
        """
        Обучение модели.

        Args:
            df: DataFrame с данными
            target_col: Целевая колонка

        Returns:
            self
        """
        # Валидация
        series = self._validate_data(df, target_col)

        # Подготовка признаков
        df_prep = self._prepare_features(df)

        # Сезонная норма
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)

        # Исключаем выбросные годы
        train_df = df_prep[~df_prep["year"].isin(self.OUTLIER_YEARS)]

        # Очистка
        train_clean = train_df.dropna(subset=self.BASE_FEATURES + [target_col])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(
                f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}"
            )

        # Подготовка данных
        X = train_clean[self.BASE_FEATURES].values
        y = train_clean[target_col].values

        # Scaling
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Инициализируем базовые модели
        # Ridge
        from sklearn.linear_model import Ridge as SklearnRidge

        ridge = SklearnRidge(alpha=self.ridge_alpha)

        # EBM (если доступен)
        try:
            from interpret.glassbox import ExplainableBoostingRegressor

            ebm = ExplainableBoostingRegressor(
                max_bins=self.ebm_max_bins,
                interactions=0,
                outer_bags=8,
                learning_rate=0.01,
            )
            estimators = [("ridge", ridge), ("ebm", ebm)]
        except ImportError:
            # Fallback если InterpretML недоступен
            from sklearn.ensemble import GradientBoostingRegressor

            gbm = GradientBoostingRegressor(n_estimators=100, max_depth=3)
            estimators = [("ridge", ridge), ("gbm", gbm)]

        # Meta-learner: LinearRegression
        final_estimator = LinearRegression()

        # StackingRegressor
        self.stacking = StackingRegressor(
            estimators=estimators, final_estimator=final_estimator, cv=5, n_jobs=-1
        )

        # Обучение
        self.stacking.fit(X_scaled, y)

        self._is_fitted = True
        self._last_train_date = df.index.max()
        self._target_col = target_col

        # Сохраняем DataFrame для iterative_forecast
        self._train_df = df.copy()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Прогноз на горизонт через итеративный predict().

        Args:
            horizon: Количество месяцев

        Returns:
            numpy array с прогнозами (MoM в %)
        """
        self._check_fitted()

        # Используем iterative_forecast с сохранёнными данными
        if hasattr(self, "_train_df") and self._train_df is not None:
            target_col = getattr(self, "_target_col", "Все товары и услуги")
            return self.iterative_forecast(self._train_df, horizon, target_col)

        # Fallback на сезонную норму
        if self.seasonal_norm is None:
            return np.zeros(horizon)

        last_month = self._last_train_date.month if self._last_train_date else 1
        predictions = []

        for i in range(horizon):
            month = ((last_month + i) % 12) + 1
            pred = self.seasonal_norm.get(month, 100.0) - 100  # MoM%
            predictions.append(pred)

        return np.array(predictions)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Точечный прогноз на дату.

        Args:
            df: DataFrame с данными
            target_date: Дата прогноза

        Returns:
            Dict с прогнозом и компонентами
        """
        self._check_fitted()

        df_prep = self._prepare_features(df)
        test_row = df_prep.loc[[target_date]]

        # Stacking прогноз
        X_test = self.scaler.transform(test_row[self.BASE_FEATURES].values)
        pred_stacking = self.stacking.predict(X_test)[0]

        # ETS прогноз (как fallback)
        target_month = target_date.month
        pred_ets = self.seasonal_norm.get(target_month, 100.0)

        # Комбинация (weighted average)
        # Stacking уже комбинирует, но добавляем сезонный hint
        pred_combined = 0.8 * pred_stacking + 0.2 * (pred_ets - 100)

        return {
            "date": target_date,
            "prediction": pred_combined,
            "pred_stacking": pred_stacking,
            "pred_ets": pred_ets,
            "model": self.name,
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        """
        Бэктестирование модели.

        Args:
            df: DataFrame с данными
            start_date: Начало периода
            target_col: Целевая колонка

        Returns:
            DataFrame с результатами
        """
        start = pd.Timestamp(start_date)

        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            cutoff = target_date - pd.DateOffset(days=1)
            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = StackingRegressorForecaster(
                    ridge_alpha=self.ridge_alpha, ebm_max_bins=self.ebm_max_bins
                )
                model.fit(train_df, target_col)

                test_df = df[df.index <= target_date].copy()
                pred_result = model.predict(test_df, target_date)

                actual = df.loc[target_date, target_col]

                results.append(
                    {
                        "date": target_date,
                        "actual": actual,
                        "prediction": pred_result["prediction"],
                        "error": actual - pred_result["prediction"],
                        "pred_stacking": pred_result["pred_stacking"],
                    }
                )
            except Exception as e:
                print(f"StackingRegressor Error at {target_date}: {e}")
                import traceback

                traceback.print_exc()
                continue

        return pd.DataFrame(results)

    def get_meta_weights(self) -> Dict[str, float]:
        """
        Веса базовых моделей от meta-learner.

        Returns:
            Dict с весами каждого базового модели
        """
        self._check_fitted()

        if self.stacking.final_estimator_ is None:
            return {}

        coefs = self.stacking.final_estimator_.coef_
        estimators = [name for name, _ in self.stacking.estimators]

        weights = {}
        for i, name in enumerate(estimators):
            weights[name] = float(coefs[i])

        # Intercept добавляем как bias
        weights["bias"] = float(self.stacking.final_estimator_.intercept_)

        return weights
