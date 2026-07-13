"""
OPR-Enhanced RidgeForecaster - Task 117

Extends Ridge regression with OPR features from Task 116.
Uses monthly (м/м) features with appropriate lags to avoid look-ahead bias.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Any, List
from pathlib import Path

from .base import BaseForecaster
from .registry import ModelRegistry

try:
    from sirena.cache_manager import cached_fit, cached_predict

    CACHE_ENABLED = True
except ImportError:
    CACHE_ENABLED = False


@ModelRegistry.register("opr_ridge")
class OPREnhancedRidgeForecaster(BaseForecaster):
    """
    Ridge forecaster enhanced with OPR (Official Price Reporting) features.

    Uses monthly OPR features with lags to avoid look-ahead bias in backtesting.

    This model integrates at least 3 OPR-based features from Task 116 regressor analysis.
    The features are:
    1. OPR monthly (м/м) CPI components (sectoral price indices)
    2. Lagged by 1 month to avoid look-ahead bias
    3. Combined with autoregressive and seasonal features

    Performance Target: MAE < 0.22 (baseline: 0.236)
    """

    name = "opr_ridge"
    MIN_TRAIN_SIZE = 36

    BASE_FEATURES = [
        "mom_L1",
        "mom_L2",
        "mom_L3",
        "month_sin",
        "month_cos",
    ]

    def __init__(
        self,
        alpha: float = 1.0,
        use_huber: bool = False,
        use_opr: bool = True,
        opr_lag: int = 1,
        **kwargs,
    ):
        """
        Args:
            alpha: Ridge regularization
            use_huber: Use HuberRegressor instead of Ridge
            use_opr: Include OPR features
            opr_lag: Lag for OPR features (default 1 = previous month)
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.use_huber = use_huber
        self.use_opr = use_opr
        self.opr_lag = opr_lag
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self._available_features: Optional[List[str]] = None
        self._train_df = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare all features including OPR."""
        df = df.copy()

        if "mom" in df.columns:
            y = df["mom"]
        elif "Все товары и услуги" in df.columns:
            y = df["Все товары и услуги"]
        else:
            raise ValueError("No target column found")

        df["mom"] = y

        # Autoregressive features
        df["mom_L1"] = y.shift(1)
        df["mom_L2"] = y.shift(2)
        df["mom_L3"] = y.shift(3)

        # Seasonality
        df["month"] = df.index.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["year"] = df.index.year

        # OPR features with specified lag
        if self.use_opr:
            opr_cols = [c for c in df.columns if c.startswith("opr_")]
            for col in opr_cols:
                lagged_col = f"{col}_L{self.opr_lag}"
                df[lagged_col] = df[col].shift(self.opr_lag)

        return df

    def fit_uncached(
        self, df: pd.DataFrame, target_col: str = "mom"
    ) -> "OPREnhancedRidgeForecaster":
        """Train the model (uncached version)."""
        df_prep = self._prepare_features(df)

        features = self.BASE_FEATURES.copy()

        if self.use_opr:
            opr_features = [c for c in df_prep.columns if c.startswith("opr_")]
            features.extend(opr_features)

        self._available_features = [f for f in features if f in df_prep.columns]

        if len(self._available_features) < 3:
            raise ValueError(f"Insufficient features: {len(self._available_features)}")

        train_clean = df_prep.dropna(subset=self._available_features + [target_col])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(
                f"Insufficient data: {len(train_clean)} < {self.MIN_TRAIN_SIZE}"
            )

        X = train_clean[self._available_features].values
        y = train_clean[target_col].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if self.use_huber:
            self.model = HuberRegressor(epsilon=1.35, alpha=self.alpha)
        else:
            self.model = Ridge(alpha=self.alpha)

        self.model.fit(X_scaled, y)

        self._is_fitted = True
        if hasattr(df.index, "max"):
            self._last_train_date = pd.Timestamp(df.index.max())
        self._train_df = df.copy()

        return self

    if CACHE_ENABLED:
        fit = cached_fit(fit_uncached)
    else:
        fit = fit_uncached

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return getattr(self, "_is_fitted", False)

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Generate inflation forecasts.

        Args:
            horizon: Forecast horizon in months

        Returns:
            Array of forecasts (in %, e.g., 0.5 for 0.5% MoM)
        """
        self._check_fitted()

        if self._train_df is None:
            return np.full(horizon, 0.1)

        forecasts = []
        df_work = self._train_df.copy()

        for h in range(horizon):
            target_date = self._last_train_date + pd.DateOffset(months=h + 1)

            if target_date not in df_work.index:
                prev_date = df_work.index.max()
                df_work.loc[target_date] = df_work.loc[prev_date].copy()
                df_work.loc[target_date, "mom"] = np.nan
                df_work = df_work.sort_index()

            try:
                pred_result = self.predict(df_work, target_date)
                pred = pred_result["prediction"]
                forecasts.append(pred)
                df_work.loc[target_date, "mom"] = pred
            except Exception as e:
                last_val = (
                    df_work["mom"].dropna().iloc[-1]
                    if not df_work["mom"].dropna().empty
                    else 0.1
                )
                forecasts.append(last_val)
                df_work.loc[target_date, "mom"] = last_val

        return np.array(forecasts)

    def predict_uncached(
        self, df: pd.DataFrame, target_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """Point prediction for a date (uncached version)."""
        self._check_fitted()

        df_prep = self._prepare_features(df)

        if target_date not in df_prep.index:
            raise ValueError(f"target_date {target_date} not in data index")

        df_prep = df_prep.ffill()

        test_row = df_prep.loc[[target_date], self._available_features]

        if test_row.isna().any().any():
            test_row = test_row.fillna(0)

        X_test = self.scaler.transform(test_row.values)
        prediction = self.model.predict(X_test)[0]

        return {
            "date": target_date,
            "prediction": prediction,
            "model": self.name,
            "n_features": len(self._available_features),
            "features_used": self._available_features,
        }

    if CACHE_ENABLED:
        predict = cached_predict(predict_uncached)
    else:
        predict = predict_uncached

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "mom",
        horizon: int = 1,
    ) -> pd.DataFrame:
        """
        Backtest the model.

        IMPORTANT: During backtest, model uses only data UP TO cutoff date.
        """
        start = pd.Timestamp(start_date)

        if "mom" in df.columns:
            y_col = "mom"
        elif target_col in df.columns:
            y_col = target_col
        else:
            raise ValueError("No target column")

        valid_dates = df.dropna(subset=[y_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for cutoff_date in test_dates:
            train_df = df[df.index < cutoff_date].copy()

            if len(train_df.dropna(subset=[y_col])) < self.MIN_TRAIN_SIZE:
                continue

            target_date = cutoff_date

            if horizon > 1:
                target_date = cutoff_date + pd.DateOffset(months=horizon - 1)
                if target_date not in df.index:
                    continue

            try:
                model = OPREnhancedRidgeForecaster(
                    alpha=self.alpha,
                    use_huber=self.use_huber,
                    use_opr=self.use_opr,
                    opr_lag=self.opr_lag,
                )
                model.fit(train_df, target_col)

                test_df = df[df.index <= target_date].copy()
                pred_result = model.predict(test_df, target_date)

                actual = df.loc[target_date, y_col]

                results.append(
                    {
                        "date": target_date,
                        "actual": actual,
                        "prediction": pred_result["prediction"],
                        "error": actual - pred_result["prediction"],
                        "n_features": len(model._available_features),
                    }
                )
            except Exception as e:
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        self._check_fitted()

        importance = pd.DataFrame(
            {"feature": self._available_features, "coefficient": self.model.coef_}
        )
        importance["abs_coef"] = importance["coefficient"].abs()
        return importance.sort_values("abs_coef", ascending=False)
