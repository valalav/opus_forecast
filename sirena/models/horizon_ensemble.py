#!/usr/bin/env python3
"""
АДАПТИВНЫЙ АНСАМБЛЬ ПО ГОРИЗОНТУ v1
====================================
Комбинирует Huber (лучший на h=1,2,3) и Micro (лучший на h=12)
с адаптивными весами в зависимости от горизонта прогноза.

Результаты исследования (MAE):
- h=1: Huber 0.288, Micro 0.361 → вес Huber 80%
- h=2: Huber 0.267, Micro 0.369 → вес Huber 75%
- h=3: Huber 0.318, Micro 0.350 → вес Huber 65%
- h=6: Huber 0.333, Micro 0.331 → вес Huber 50%
- h=12: Huber 0.331, Micro 0.297 → вес Huber 30%

Формула весов (эмпирическая):
  w_huber = max(0.3, 1.0 - 0.05 * horizon)
  w_micro = 1 - w_huber
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

warnings.filterwarnings("ignore")

from .huber import HuberForecaster
from .microcomponent import MicrocomponentForecaster


class HorizonEnsembleForecaster:
    """
    Adaptive ensemble: Huber + Micro with horizon-dependent weights.

    Short horizons: Huber dominates (autocorrelation is strong)
    Long horizons: Micro dominates (structural info matters more)
    """

    name = "horizon_ensemble"

    # Empirically derived weights based on backtest results
    WEIGHTS = {
        1: {"huber": 0.80, "micro": 0.20},
        2: {"huber": 0.75, "micro": 0.25},
        3: {"huber": 0.65, "micro": 0.35},
        6: {"huber": 0.50, "micro": 0.50},
        12: {"huber": 0.30, "micro": 0.70},
    }

    def __init__(
        self, horizon: int = 1, train_start: str = "2016-01-01", random_state: int = 42
    ) -> None:
        self.horizon = horizon
        self.train_start = train_start
        self.random_state = random_state

        self._is_fitted = False
        self.huber_model = None
        self.micro_model = None
        self.weights = self._get_weights(horizon)

    def _get_weights(self, horizon: int) -> Dict[str, float]:
        """Get weights for given horizon, interpolating if needed."""
        if horizon in self.WEIGHTS:
            return self.WEIGHTS[horizon]

        # Interpolate for other horizons
        horizons = sorted(self.WEIGHTS.keys())

        if horizon < horizons[0]:
            return self.WEIGHTS[horizons[0]]
        if horizon > horizons[-1]:
            return self.WEIGHTS[horizons[-1]]

        # Find surrounding horizons
        lower = max(h for h in horizons if h <= horizon)
        upper = min(h for h in horizons if h >= horizon)

        if lower == upper:
            return self.WEIGHTS[lower]

        # Linear interpolation
        ratio = (horizon - lower) / (upper - lower)
        w_huber = (
            self.WEIGHTS[lower]["huber"] * (1 - ratio)
            + self.WEIGHTS[upper]["huber"] * ratio
        )

        return {"huber": w_huber, "micro": 1 - w_huber}

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "HorizonEnsembleForecaster":
        """Fit both models."""
        # Fit Huber
        self.huber_model = HuberForecaster(
            horizon=self.horizon,
            train_start=self.train_start,
            random_state=self.random_state,
        )
        self.huber_model.fit(df, target_col)

        # Fit Micro
        self.micro_model = MicrocomponentForecaster(
            horizon=self.horizon,
            train_start=self.train_start,
            random_state=self.random_state,
        )
        self.micro_model.fit(df, target_col)

        self._is_fitted = True
        self.macro_df = df.copy()

        print(f"HorizonEnsembleForecaster fitted (h={self.horizon}):")
        print(
            f"  Weights: Huber {self.weights['huber']:.0%}, Micro {self.weights['micro']:.0%}"
        )

        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Predict using weighted ensemble."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        # Get predictions from both models
        try:
            huber_pred = self.huber_model.predict(df, target_date)["prediction"] - 100
        except:
            huber_pred = None

        try:
            micro_pred = self.micro_model.predict(df, target_date)["prediction"] - 100
        except:
            micro_pred = None

        # Combine with weights
        if huber_pred is not None and micro_pred is not None:
            ensemble_pred = (
                self.weights["huber"] * huber_pred + self.weights["micro"] * micro_pred
            )
        elif huber_pred is not None:
            ensemble_pred = huber_pred
        elif micro_pred is not None:
            ensemble_pred = micro_pred
        else:
            ensemble_pred = 0.0

        return {
            "prediction": 100 + ensemble_pred,
            "huber_pred": huber_pred,
            "micro_pred": micro_pred,
            "weights": self.weights,
        }

    def forecast(self, horizon: Optional[int] = None) -> np.ndarray:
        """Generate forecast trajectory."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        h = horizon or self.horizon
        forecasts = []
        last_date = self.macro_df.index[-1]

        for i in range(h):
            target_date = last_date + pd.DateOffset(months=i + 1)

            # Adjust weights for each step in trajectory
            step_weights = self._get_weights(i + 1)

            # Get predictions
            df_ext = self.macro_df.copy()
            df_ext.loc[target_date] = np.nan

            try:
                huber_pred = (
                    self.huber_model.predict(df_ext, target_date)["prediction"] - 100
                )
            except:
                huber_pred = 0.0

            try:
                micro_pred = (
                    self.micro_model.predict(df_ext, target_date)["prediction"] - 100
                )
            except:
                micro_pred = 0.0

            ensemble_pred = (
                step_weights["huber"] * huber_pred + step_weights["micro"] * micro_pred
            )
            forecasts.append(ensemble_pred)

        return np.array(forecasts)

    def get_model_contributions(self, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Get contribution of each model to the forecast."""
        if not self._is_fitted:
            return {}

        df_ext = self.macro_df.copy()
        df_ext.loc[target_date] = np.nan

        result = self.predict(df_ext, target_date)

        return {
            "ensemble": result["prediction"] - 100,
            "huber": {
                "prediction": result["huber_pred"],
                "weight": self.weights["huber"],
                "contribution": self.weights["huber"] * result["huber_pred"]
                if result["huber_pred"]
                else 0,
            },
            "micro": {
                "prediction": result["micro_pred"],
                "weight": self.weights["micro"],
                "contribution": self.weights["micro"] * result["micro_pred"]
                if result["micro_pred"]
                else 0,
            },
        }
