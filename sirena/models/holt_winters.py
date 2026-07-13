"""
Holt-Winters модель для прогнозирования инфляции КБР
================================================

Dedicated Holt-Winters implementation (separate from generic ETS).
Supports both additive and multiplicative seasonal models.

Reference: Holt, C.C., and S.R. Winters (1960)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Dict, Any, Optional
import warnings

warnings.filterwarnings("ignore")

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("holt_winters")
class HoltWintersForecaster(BaseForecaster):
    """
    Holt-Winters тройное экспоненциальное сглаживание.

    Компоненты модели:
    - Level (L): текущий уровень ряда
    - Trend (T): тренд
    - Seasonality (S): сезонный компонент

    Режимы:
    - Additive: Y = L + T + S (для постоянной амплитуды)
    - Multiplicative: Y = L * T * S (для растущей амплитуды)
    """

    name = "holt_winters"
    MIN_TRAIN_SIZE = 24

    def __init__(
        self,
        trend: str = "add",
        seasonal: str = "add",
        seasonal_periods: int = 12,
        damped_trend: bool = False,
        smoothing_level: Optional[float] = None,
        smoothing_trend: Optional[float] = None,
        smoothing_seasonal: Optional[float] = None,
        **kwargs,
    ):
        """
        Инициализация Holt-Winters.

        Args:
            trend: 'add' или 'mul' - тип тренда
            seasonal: 'add' или 'mul' - тип сезонности
            seasonal_periods: период сезонности (месяцы, обычно 12)
            damped_trend: затухающий тренд
            smoothing_level: alpha параметр (0-1)
            smoothing_trend: beta параметр (0-1)
            smoothing_seasonal: gamma параметр (0-1)
        """
        super().__init__(**kwargs)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal

        self.model = None
        self.fit_result = None
        self.last_values = None

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "HoltWintersForecaster":
        """Обучение Holt-Winters."""
        series = self._validate_data(df, target_col)

        # Конвертируем в MoM если нужно
        if series.mean() > 50:
            series = series - 100

        self.last_values = series.values

        # Проверяем достаточно ли данных для сезонности
        use_seasonal = len(series) >= 2 * self.seasonal_periods

        try:
            self.model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal if use_seasonal else None,
                seasonal_periods=self.seasonal_periods if use_seasonal else None,
                damped_trend=self.damped_trend,
            )

            # Параметры сглаживания (если заданы)
            fit_kwargs = {"optimized": True}
            if self.smoothing_level is not None:
                fit_kwargs["smoothing_level"] = self.smoothing_level
            if self.smoothing_trend is not None:
                fit_kwargs["smoothing_trend"] = self.smoothing_trend
            if self.smoothing_seasonal is not None and use_seasonal:
                fit_kwargs["smoothing_seasonal"] = self.smoothing_seasonal

            self.fit_result = self.model.fit(**fit_kwargs)
            self._is_fitted = True
            self._last_train_date = df.index.max()

        except Exception as e:
            # Fallback на модель без сезонности
            self.model = ExponentialSmoothing(
                series, trend=self.trend, seasonal=None, damped_trend=self.damped_trend
            )
            self.fit_result = self.model.fit(optimized=True)
            self._is_fitted = True
            self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз Holt-Winters."""
        self._check_fitted()

        forecast = self.fit_result.forecast(steps=horizon)
        return forecast.values

    def forecast_with_intervals(
        self, horizon: int = 12, alpha: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Прогноз с доверительными интервалами.

        Args:
            horizon: горизонт прогноза
            alpha: уровень значимости (0.05 = 95% CI)

        Returns:
            dict с 'mean', 'lower', 'upper'
        """
        self._check_fitted()

        forecast = self.fit_result.forecast(steps=horizon)

        # Оценка интервалов через residuals
        residuals = self.fit_result.resid
        std_resid = np.std(residuals)

        z = 1.96  # 95% confidence
        lower = forecast - z * std_resid * np.sqrt(np.arange(1, horizon + 1))
        upper = forecast + z * std_resid * np.sqrt(np.arange(1, horizon + 1))

        return {
            "mean": forecast.values,
            "lower": lower.values,
            "upper": upper.values,
            "aic": getattr(self.fit_result, "aic", None),
            "sse": getattr(self.fit_result, "sse", None),
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        """Бэктестирование Holt-Winters."""
        if target_col in df.columns:
            series = df[target_col].dropna()
        else:
            series = df.dropna()

        # Конвертируем в MoM если нужно
        if series.mean() > 50:
            series = series - 100

        test_dates = series[series.index >= start_date].index
        results = []

        for target_date in test_dates:
            cutoff = target_date - pd.DateOffset(months=1)
            train_data = series[series.index <= cutoff]

            if len(train_data) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = HoltWintersForecaster(
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods,
                    damped_trend=self.damped_trend,
                )
                train_df = pd.DataFrame({target_col: train_data + 100})
                model.fit(train_df, target_col)
                fc = model.forecast(horizon=1)

                actual = series.loc[target_date]

                results.append(
                    {
                        "date": target_date,
                        "actual": actual,
                        "prediction": fc[0],
                        "error": actual - fc[0],
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_components(self) -> Dict[str, Any]:
        """
        Получить компоненты Holt-Winters модели.

        Returns:
            dict с level, trend, season, resid
        """
        if self.fit_result is None:
            return {}

        return {
            "level": self.fit_result.level,
            "trend": getattr(self.fit_result, "trend", None),
            "season": getattr(self.fit_result, "season", None),
            "resid": self.fit_result.resid,
        }

    def get_params(self) -> Dict[str, Any]:
        """Получить параметры модели."""
        params = super().get_params()
        params.update(
            {
                "trend": self.trend,
                "seasonal": self.seasonal,
                "seasonal_periods": self.seasonal_periods,
                "damped_trend": self.damped_trend,
            }
        )

        # Добавляем подобранные параметры
        if self.fit_result is not None:
            params.update(
                {
                    "smoothing_level": getattr(self.fit_result, "params", {}).get(
                        "smoothing_level"
                    ),
                    "smoothing_trend": getattr(self.fit_result, "params", {}).get(
                        "smoothing_trend"
                    ),
                    "smoothing_seasonal": getattr(self.fit_result, "params", {}).get(
                        "smoothing_seasonal"
                    ),
                    "aic": getattr(self.fit_result, "aic", None),
                }
            )

        return params
