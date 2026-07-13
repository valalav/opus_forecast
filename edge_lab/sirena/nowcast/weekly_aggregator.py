"""
Weekly Nowcast Integration for Ensemble Forecasts
=============================================

This module provides functionality to integrate weekly high-frequency signals
into monthly ensemble forecasts using weighted blending.

Key functionality:
- Aggregates weekly signals to monthly frequency
- Combines monthly model predictions with weekly signal: ensemble = 0.85*monthly + 0.15*weekly
- Validates performance on recent data periods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from ..models.base import BaseForecaster
from ..models.weekly import WeeklySignalForecaster
from ..models.registry import ModelRegistry


class WeeklyAggregator:
    """
    Integrates weekly signals into monthly ensemble forecasts.

    This class:
    1. Loads weekly signals from WeeklySignalForecaster
    2. Aggregates weekly data to monthly frequency
    3. Computes weighted blend: ensemble = 0.85 * monthly_models + 0.15 * weekly_signal
    4. Provides validation on recent test periods
    """

    def __init__(
        self,
        weekly_weight: float = 0.15,
        monthly_weight: float = 0.85,
        monthly_models: Optional[List[str]] = None,
    ):
        """
        Initialize WeeklyAggregator.

        Args:
            weekly_weight: Weight for weekly signal (default: 0.15)
            monthly_weight: Weight for monthly ensemble (default: 0.85)
            monthly_models: List of monthly model names to include in ensemble
        """
        self.weekly_weight = weekly_weight
        self.monthly_weight = monthly_weight
        self.monthly_models = monthly_models or []
        self.weekly_model = None
        self._is_fitted = False

    def _load_data(self) -> pd.DataFrame:
        """Load inflation data."""
        base_dir = Path.cwd().parent
        data_path = base_dir / "data" / "inflation_data.csv"

        if not data_path.exists():
            data_path = Path.cwd() / "data" / "inflation_data.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"Inflation data not found")

        df = pd.read_csv(data_path, sep=";", encoding="utf-8-sig")
        df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", dayfirst=True)
        df = df.set_index("Date").sort_index()

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.replace(",", ".").astype(float)

        return df

    def _ensure_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure numeric columns are properly converted."""
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
        return df

    def fit(
        self,
        df: Optional[pd.DataFrame] = None,
        target_col: str = "mom",
        start_date: Optional[str] = None,
    ) -> "WeeklyAggregator":
        """
        Fit aggregator on historical data.

        Args:
            df: DataFrame with macro data. If None, loads from default path.
            target_col: Target column name (default: "mom")
            start_date: Optional start date for fitting period

        Returns:
            self for method chaining
        """
        if df is None:
            df = self._load_data()

        df = df.copy()
        df = self._ensure_numeric_columns(df)

        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]

        self.weekly_model = WeeklySignalForecaster()
        self.weekly_model.fit(df, target_col)

        self._is_fitted = True
        return self

    def _get_monthly_ensemble_forecast(
        self, df: pd.DataFrame, forecast_date: pd.Timestamp, target_col: str = "mom"
    ) -> float:
        """
        Compute monthly ensemble forecast from registered models.

        Args:
            df: DataFrame with historical data
            forecast_date: Date to forecast for
            target_col: Target column name

        Returns:
            Ensemble forecast value (MoM in %)
        """
        df = self._ensure_numeric_columns(df)

        if not self.monthly_models:
            models = ModelRegistry.list_models()
            monthly_models = [m for m in models if m != "weekly"]
        else:
            monthly_models = self.monthly_models

        forecasts = []

        for model_name in monthly_models:
            try:
                model = ModelRegistry.get(model_name)

                cutoff = forecast_date - pd.DateOffset(months=1)
                train_df = df[df.index <= cutoff].copy()

                if len(train_df) < model.MIN_TRAIN_SIZE:
                    continue

                model.fit(train_df, target_col)
                pred = model.forecast(horizon=1)[0]
                forecasts.append(pred)
            except Exception as e:
                continue

        if not forecasts:
            return 0.0

        return float(np.mean(forecasts))

    def _get_weekly_signal(
        self, df: pd.DataFrame, forecast_date: pd.Timestamp, target_col: str = "mom"
    ) -> float:
        """
        Get weekly signal forecast.

        Args:
            df: DataFrame with historical data
            forecast_date: Date to forecast for
            target_col: Target column name

        Returns:
            Weekly signal forecast value (MoM in %)
        """
        df = self._ensure_numeric_columns(df)

        if self.weekly_model is None:
            return 0.0

        try:
            cutoff = forecast_date - pd.DateOffset(months=1)
            train_df = df[df.index <= cutoff].copy()

            self.weekly_model.fit(train_df, target_col)
            pred = self.weekly_model.forecast(horizon=1)[0]
            return float(pred)
        except Exception as e:
            return 0.0

    def forecast(
        self,
        df: Optional[pd.DataFrame] = None,
        horizon: int = 12,
        target_col: str = "mom",
    ) -> np.ndarray:
        """
        Generate ensemble forecasts with weekly signal integration.

        Args:
            df: DataFrame with macro data. If None, loads from default path.
            horizon: Forecast horizon in months
            target_col: Target column name

        Returns:
            Array of ensemble forecasts (MoM in %)
        """
        self._check_fitted()

        if df is None:
            df = self._load_data()

        df = self._ensure_numeric_columns(df)
        last_date = df.index.max()
        forecasts = []

        for h in range(horizon):
            forecast_date = last_date + pd.DateOffset(months=h + 1)

            monthly_pred = self._get_monthly_ensemble_forecast(
                df, forecast_date, target_col
            )
            weekly_pred = self._get_weekly_signal(df, forecast_date, target_col)

            ensemble_pred = (
                self.monthly_weight * monthly_pred + self.weekly_weight * weekly_pred
            )

            forecasts.append(ensemble_pred)

        return np.array(forecasts)

    def backtest(
        self,
        df: Optional[pd.DataFrame] = None,
        start_date: str = "2024-07-01",
        target_col: str = "mom",
        horizon: int = 1,
    ) -> pd.DataFrame:
        """
        Run backtest comparing ensemble with and without weekly signal.

        Args:
            df: DataFrame with macro data. If None, loads from default path.
            start_date: Start date for backtest (default: 6 months ago)
            target_col: Target column name
            horizon: Forecast horizon

        Returns:
            DataFrame with columns: date, actual, prediction_ensemble,
                                 prediction_monthly_only, error_ensemble, error_monthly
        """
        if df is None:
            df = self._load_data()

        df = self._ensure_numeric_columns(df)
        target_series = df[target_col] - 100
        results = []

        test_dates = df[df.index >= pd.Timestamp(start_date)].index

        for forecast_date in test_dates:
            try:
                cutoff = forecast_date - pd.DateOffset(months=horizon)
                train_df = df[df.index <= cutoff].copy()

                if len(train_df) < 36:
                    continue

                monthly_pred = self._get_monthly_ensemble_forecast(
                    train_df, forecast_date, target_col
                )
                weekly_pred = self._get_weekly_signal(
                    train_df, forecast_date, target_col
                )

                ensemble_pred = (
                    self.monthly_weight * monthly_pred
                    + self.weekly_weight * weekly_pred
                )

                actual = target_series.loc[forecast_date]

                results.append(
                    {
                        "date": forecast_date,
                        "actual": actual,
                        "prediction_ensemble": ensemble_pred,
                        "prediction_monthly_only": monthly_pred,
                        "weekly_signal": weekly_pred,
                        "error_ensemble": actual - ensemble_pred,
                        "error_monthly": actual - monthly_pred,
                    }
                )
            except Exception as e:
                continue

        return pd.DataFrame(results)

    def compare_mae(self, backtest_results: pd.DataFrame) -> Dict[str, float]:
        """
        Compare MAE between ensemble with and without weekly signal.

        Args:
            backtest_results: DataFrame from backtest() method

        Returns:
            Dict with MAE values and improvement
        """
        mae_ensemble = np.mean(np.abs(backtest_results["error_ensemble"]))
        mae_monthly = np.mean(np.abs(backtest_results["error_monthly"]))

        improvement = ((mae_monthly - mae_ensemble) / mae_monthly) * 100

        return {
            "mae_with_weekly": round(mae_ensemble, 4),
            "mae_without_weekly": round(mae_monthly, 4),
            "improvement_pct": round(improvement, 2),
            "is_better": mae_ensemble <= mae_monthly,
        }

    def _check_fitted(self):
        """Check if the aggregator has been fitted."""
        if not self._is_fitted:
            raise ValueError("WeeklyAggregator not fitted. Call fit() first.")

    @property
    def is_fitted(self) -> bool:
        """Check if the aggregator is fitted."""
        return self._is_fitted


def create_6_month_validation_report(
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Create a 6-month validation report comparing ensemble performance.

    Args:
        output_path: Optional path to save the CSV report

    Returns:
        Tuple of (backtest_results, metrics)
    """
    aggregator = WeeklyAggregator()

    df_path = Path.cwd().parent / "data" / "inflation_data.csv"
    if not df_path.exists():
        df_path = Path.cwd() / "data" / "inflation_data.csv"

    df = pd.read_csv(df_path, sep=";", encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", dayfirst=True)
    df = df.set_index("Date").sort_index()

    aggregator.fit(df, start_date="2019-01-01")

    max_date = df.index.max()
    if hasattr(max_date, "to_pydatetime"):
        six_months_ago = (max_date.to_pydatetime() - pd.DateOffset(months=6)).strftime(
            "%Y-%m-%d"
        )
    else:
        six_months_ago = (max_date - pd.DateOffset(months=6)).strftime("%Y-%m-%d")

    backtest_results = aggregator.backtest(start_date=six_months_ago)

    metrics = aggregator.compare_mae(backtest_results)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        backtest_results.to_csv(output_path, index=False)

    return backtest_results, metrics
