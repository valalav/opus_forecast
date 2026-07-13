"""
Naive Seasonal Forecaster for KBR Inflation
===========================================

Baseline model for 'Null Hypothesis' testing.
Forecast = Last Year's Value.

Use case: Establish performance floor - any sophisticated model
should beat this naive baseline.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("naive_seasonal")
class NaiveSeasonalForecaster(BaseForecaster):
    """
    Naive Seasonal Forecaster.

    Simple baseline that forecasts last year's values:
        Forecast(t+h) = Actual(t+h-12)

    This is a 'Null Hypothesis' model - any sophisticated
    model should beat this baseline.
    """

    name = "naive_seasonal"
    MIN_TRAIN_SIZE = 13  # Need at least 1 year of data

    def __init__(self, seasonal_lag: int = 12, **kwargs):
        """
        Initialize Naive Seasonal model.

        Args:
            seasonal_lag: Number of periods to look back (default: 12 months)
        """
        super().__init__(**kwargs)
        self.seasonal_lag = seasonal_lag
        self.last_values = None

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "NaiveSeasonalForecaster":
        """
        Fit the naive seasonal model.

        Simply stores the last values for reference.
        No actual training required.

        Args:
            df: DataFrame with time index and target variable
            target_col: Target column name

        Returns:
            self for method chaining
        """
        series = self._validate_data(df, target_col)

        # Convert to MoM if needed
        if series.mean() > 50:
            series = series - 100

        # Store last values
        self.last_values = series.values
        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Generate forecast by carrying forward last year's values.

        For each horizon step h:
            prediction = value from (h - seasonal_lag) positions ago

        Args:
            horizon: Number of periods to forecast

        Returns:
            numpy array with forecasts (MoM in %)
        """
        self._check_fitted()

        forecasts = []
        n = len(self.last_values)

        for h in range(horizon):
            # Get value from seasonal_lag periods ago
            idx = n - self.seasonal_lag + h

            # If we don't have enough history, use the last available value
            if idx < 0:
                idx = 0

            forecasts.append(self.last_values[idx])

        return np.array(forecasts)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Predict for a specific date.

        Args:
            df: DataFrame with data up to target_date
            target_date: Forecast date

        Returns:
            dict with prediction and metadata
        """
        # Find the value from 12 months before target_date
        lookup_date = target_date - pd.DateOffset(months=self.seasonal_lag)

        if lookup_date in df.index:
            value = df.loc[lookup_date, "Все товары и услуги"]

            # Convert to MoM if needed
            if value > 50:
                value = value - 100
        else:
            # Fallback: use last available value
            value = df["Все товары и услуги"].dropna().iloc[-1]
            if value > 50:
                value = value - 100

        return {"date": target_date, "prediction": value, "model": self.name}

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        """
        Backtest naive seasonal model.

        Args:
            df: DataFrame with data
            start_date: Start date for backtest
            target_col: Target variable

        Returns:
            DataFrame with columns: date, actual, prediction, error
        """
        if target_col in df.columns:
            series = df[target_col].dropna()
        else:
            series = df.dropna()

        # Convert to MoM if needed
        if series.mean() > 50:
            series = series - 100

        test_dates = series[series.index >= start_date].index
        results = []

        for target_date in test_dates:
            # Look up value from 12 months ago
            lookup_date = target_date - pd.DateOffset(months=self.seasonal_lag)

            if lookup_date in series.index:
                prediction = series.loc[lookup_date]
                actual = series.loc[target_date]

                results.append(
                    {
                        "date": target_date,
                        "actual": actual,
                        "prediction": prediction,
                        "error": actual - prediction,
                    }
                )

        return pd.DataFrame(results)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return dummy feature importance (not applicable for this model).
        """
        return {"seasonal_lag_12": 1.0}
