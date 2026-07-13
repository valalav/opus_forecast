"""
External statsmodels micro forecast adapter.

Loads the staged Windows-to-Linux CPI forecast matrix generated at
data/external/micro_cpi_region_export/micro_test_statsmodels.csv and exposes it
as a separate Sirena model without replacing the existing micro_test.csv flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .micro_arima import MicroARIMAForecaster
from .registry import ModelRegistry


DEFAULT_FILE_PATH = (
    "data/external/micro_cpi_region_export/micro_test_statsmodels.csv"
)


@ModelRegistry.register("micro_statsmodels_external")
class MicroStatsmodelsExternalForecaster(BaseForecaster):
    """Sirena adapter for the additional Linux statsmodels forecast."""

    name = "micro_statsmodels_external"

    def __init__(
        self,
        horizon: int = 1,
        file_path: str = DEFAULT_FILE_PATH,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.horizon = horizon
        self.file_path = Path(file_path)
        self._loader = MicroARIMAForecaster(horizon=horizon, file_path=str(self.file_path))
        self._forecasts: Optional[pd.DataFrame] = None

    def fit(
        self,
        df: Optional[pd.DataFrame] = None,
        target_col: str = "Все товары и услуги",
    ) -> "MicroStatsmodelsExternalForecaster":
        self._loader.fit()
        self._forecasts = self._loader.get_forecasts_matrix()
        self._is_fitted = True
        if df is not None and not df.empty:
            self._last_train_date = pd.Timestamp(df.index.max()).to_period("M").to_timestamp()
        return self

    def _ensure_loaded(self) -> None:
        if not self._is_fitted:
            self.fit()

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        self._ensure_loaded()
        cutoff_date = None
        if df is not None and not df.empty:
            cutoff_date = pd.Timestamp(df.index.max()).to_period("M").to_timestamp()
        value = self._loader.get_forecast(target_date, cutoff_date=cutoff_date)
        return {
            "date": pd.Timestamp(target_date),
            "prediction": np.nan if value is None else float(value),
            "model": self.name,
            "source_file": str(self.file_path),
        }

    def forecast(self, horizon: int = 12) -> np.ndarray:
        self._ensure_loaded()
        assert self._forecasts is not None

        cutoff_dates = [
            col for col in self._forecasts.columns if isinstance(col, pd.Timestamp)
        ]
        if not cutoff_dates:
            return np.full(horizon, np.nan)

        cutoff = max(cutoff_dates)
        values = []
        for step in range(1, horizon + 1):
            target_date = cutoff + pd.DateOffset(months=step)
            value = self._loader.get_forecast(target_date, cutoff_date=cutoff)
            values.append(np.nan if value is None else float(value))
        return np.asarray(values, dtype=float)

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        self.fit(df, target_col)
        rows = []
        for target_date in pd.date_range(start=start_date, end=df.index.max(), freq="MS"):
            train = df[df.index < target_date]
            result = self.predict(train, target_date)
            actual = df.loc[target_date, target_col] if target_date in df.index else np.nan
            prediction = result["prediction"]
            if pd.notna(actual) and abs(actual) <= 50:
                actual = actual + 100
            rows.append(
                {
                    "date": target_date,
                    "actual": actual,
                    "prediction": prediction,
                    "error": prediction - actual
                    if pd.notna(prediction) and pd.notna(actual)
                    else np.nan,
                }
            )
        return pd.DataFrame(rows)
