"""
MIDAS (Mixed Data Sampling) Model
==================================

MIDAS regression allows combining data at different frequencies:
- Monthly inflation target
- Weekly macro indicators (e.g., Brent oil, USD/RUB)
- Daily exchange rates, etc.

Key feature: Polynomial weighting functions to aggregate high-frequency data:
- Almon lag (polynomial)
- Exponential weights
- Beta weights
- Normalized exponential (NED)

Mathematical formulation:
    y_t = α + β * Σ (w_k * X_{t - k/K}) + ε_t

where:
- y_t: monthly target (inflation)
- X: high-frequency predictor (e.g., weekly Brent)
- w_k: MIDAS weights from polynomial
- K: number of high-frequency periods per month
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Any, List, Tuple
from scipy.optimize import minimize
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("midas")
class MIDASForecaster(BaseForecaster):
    """
    MIDAS (Mixed Data Sampling) Forecaster.

    Combines monthly inflation with high-frequency predictors using
    polynomial weighting functions.

    Supported weight types:
    - 'almon': Polynomial lag structure
    - 'exp': Exponential decay weights
    - 'beta': Beta function weights
    - 'normalized_exp': Normalized exponential (NED)
    """

    name = "midas"
    MIN_TRAIN_SIZE = 48
    OUTLIER_YEARS = [2010, 2022]

    # Monthly features
    MONTHLY_FEATURES = [
        "y_lag1",
        "y_lag2",
        "y_lag12",
        "y_lag3",
        "y_lag6",
        "month_sin",
        "month_cos",
        "is_jan",
        "is_jul",
        "is_dec",
    ]

    # High-frequency features with default lags
    HF_FEATURES = {
        "brent": {"freq": "W", "lags": 8, "weight_type": "almon"},
        "usd": {"freq": "W", "lags": 8, "weight_type": "exp"},
        "ki": {"freq": "W", "lags": 6, "weight_type": "almon"},
    }

    def __init__(
        self,
        weight_type: str = "almon",
        poly_order: int = 2,
        hf_features: Optional[List[str]] = None,
        alpha: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            weight_type: Type of MIDAS weights ('almon', 'exp', 'beta', 'normalized_exp')
            poly_order: Polynomial order for Almon weights
            hf_features: List of high-frequency features to use (['brent', 'usd', 'ki'])
                         If None, uses all available HF features
            alpha: Ridge regularization parameter
        """
        super().__init__(**kwargs)
        self.weight_type = weight_type
        self.poly_order = poly_order
        self.hf_features = hf_features or list(self.HF_FEATURES.keys())
        self.alpha = alpha

        self.model = None
        self.scaler = None
        self._midas_transformers = {}
        self._available_hf = []
        self._final_features = None

    def _midas_weights_almon(self, k: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Almon polynomial lag weights.

        w_k = θ_0 + θ_1*k + θ_2*k^2 + ... + θ_p*k^p
        """
        weights = np.zeros(len(k))
        for p, theta_p in enumerate(theta):
            weights += theta_p * (k**p)
        return weights

    def _midas_weights_exp(self, k: np.ndarray, theta: float) -> np.ndarray:
        """
        Exponential decay weights.

        w_k = exp(-θ * k)
        """
        return np.exp(-theta * k)

    def _midas_weights_beta(self, k: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Beta function weights (smooth hump-shaped).

        w_k = k^{θ1-1} * (K-k)^{θ2-1}
        """
        theta1, theta2 = theta
        K = len(k)
        k_norm = k / (K - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weights = (k_norm ** (theta1 - 1)) * ((1 - k_norm) ** (theta2 - 1))
        return weights

    def _midas_weights_normalized_exp(
        self, k: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        Normalized exponential (NED) weights.

        w_k = exp(θ1 * k + θ2 * k^2) / Σ(exp(...))
        """
        exp_vals = np.exp(theta[0] * k + theta[1] * (k**2))
        return exp_vals / np.sum(exp_vals)

    def _get_midas_weights(self, n_lags: int) -> np.ndarray:
        """
        Get MIDAS weights for given number of lags.
        """
        k = np.arange(n_lags)

        if self.weight_type == "almon":
            n_params = self.poly_order + 1
            theta = np.ones(n_params) / n_params  # Start with equal weights
            return self._midas_weights_almon(k, theta), theta

        elif self.weight_type == "exp":
            theta = 0.5  # Decay parameter
            return self._midas_weights_exp(k, theta), np.array([theta])

        elif self.weight_type == "beta":
            theta = np.array([2.0, 2.0])  # Symmetric hump
            return self._midas_weights_beta(k, theta), theta

        elif self.weight_type == "normalized_exp":
            theta = np.array([-0.5, 0.01])  # Peak then decay
            return self._midas_weights_normalized_exp(k, theta), theta

        else:
            raise ValueError(f"Unknown weight_type: {self.weight_type}")

    def _aggregate_hf_to_mf(
        self, hf_series: pd.Series, target_dates: pd.DatetimeIndex, n_lags: int
    ) -> pd.DataFrame:
        """
        Aggregate high-frequency data to monthly frequency using simple mean.

        Args:
            hf_series: High-frequency time series (e.g., weekly Brent)
            target_dates: Monthly target dates
            n_lags: Number of lag periods to aggregate

        Returns:
            DataFrame with aggregated features [mf_L1, mf_L2, ..., mf_Ln]
        """
        result = pd.DataFrame(index=target_dates)

        for lag in range(1, n_lags + 1):
            # Shift target dates back by lag months
            shifted_dates = target_dates - pd.DateOffset(months=lag)

            agg_values = []
            for date in shifted_dates:
                # Get high-frequency data within the month
                month_start = date
                month_end = date + pd.DateOffset(months=1) - pd.Timedelta(days=1)

                hf_in_month = hf_series[
                    (hf_series.index >= month_start) & (hf_series.index <= month_end)
                ]

                if len(hf_in_month) > 0:
                    agg_values.append(hf_in_month.mean())
                else:
                    agg_values.append(np.nan)

            result[f"hf_agg_L{lag}"] = agg_values

        return result

    def _apply_midas_weights(
        self, agg_df: pd.DataFrame, weights: np.ndarray
    ) -> pd.Series:
        """
        Apply MIDAS weights to aggregated high-frequency features.

        Args:
            agg_df: DataFrame with aggregated HF features (L1, L2, ..., Ln)
            weights: MIDAS weights for each lag

        Returns:
            Weighted sum series
        """
        weighted_values = np.zeros(len(agg_df))

        for lag, weight in enumerate(weights, start=1):
            col = f"hf_agg_L{lag}"
            if col in agg_df.columns:
                weighted_values += weight * agg_df[col].fillna(0).values

        return pd.Series(weighted_values, index=agg_df.index)

    def _prepare_monthly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare monthly features."""
        df = df.copy()
        y = df["Все товары и услуги"]

        # Lags
        df["y_lag1"] = y.shift(1)
        df["y_lag2"] = y.shift(2)
        df["y_lag3"] = y.shift(3)
        df["y_lag6"] = y.shift(6)
        df["y_lag12"] = y.shift(12)

        # Seasonality
        df["month"] = df.index.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Calendar
        df["is_jan"] = (df["month"] == 1).astype(int)
        df["is_jul"] = (df["month"] == 7).astype(int)
        df["is_dec"] = (df["month"] == 12).astype(int)

        return df

    def _prepare_hf_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Extract high-frequency features from DataFrame.

        For now, assumes monthly data has been interpolated from HF.
        In production, would load actual HF data.
        """
        hf_features = {}

        # Simulate HF features from monthly data with forward-fill
        # In real implementation, would load actual weekly/daily data
        if "brent" in df.columns:
            hf_features["brent"] = df["brent"].copy()
        if "usd_nom_i" in df.columns:
            hf_features["usd"] = df["usd_nom_i"].copy()
        if "Ki" in df.columns:
            hf_features["ki"] = df["Ki"].copy()

        return hf_features

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "MIDASForecaster":
        """
        Fit MIDAS model.

        Args:
            df: DataFrame with monthly inflation and optional HF indicators
            target_col: Target column name
        """
        series = self._validate_data(df, target_col)

        # Prepare monthly features
        df_prep = self._prepare_monthly_features(df)

        # Extract high-frequency features
        hf_data = self._prepare_hf_features(df)

        # Add MIDAS-transformed HF features
        self._available_hf = []
        for hf_name in self.hf_features:
            if hf_name not in hf_data:
                continue

            hf_config = self.HF_FEATURES.get(hf_name, {})
            n_lags = hf_config.get("lags", 8)

            # Aggregate HF to monthly
            agg_df = self._aggregate_hf_to_mf(hf_data[hf_name], df.index, n_lags)

            # Get MIDAS weights
            weights, theta = self._get_midas_weights(n_lags)
            self._midas_transformers[hf_name] = {"weights": weights, "theta": theta}

            # Apply weights
            weighted_feature = self._apply_midas_weights(agg_df, weights)
            df_prep[f"{hf_name}_midas"] = weighted_feature

            self._available_hf.append(hf_name)

        # Prepare final feature list
        features = self.MONTHLY_FEATURES.copy()
        for hf_name in self._available_hf:
            features.append(f"{hf_name}_midas")
        self._final_features = features

        # Filter training data
        train_clean = df_prep.dropna(subset=features + [target_col])

        # Exclude outlier years
        train_clean = train_clean[~train_clean.index.year.isin(self.OUTLIER_YEARS)]

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(
                f"Insufficient data: {len(train_clean)} < {self.MIN_TRAIN_SIZE}"
            )

        X = train_clean[features].values
        y = train_clean[target_col].values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit Ridge regression
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_scaled, y)

        self._is_fitted = True
        self._last_train_date = df.index.max()
        self._target_col = target_col
        self._train_df = df.copy()

        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Predict for a specific date.
        """
        self._check_fitted()

        df_prep = self._prepare_monthly_features(df)

        # Add MIDAS features for prediction
        hf_data = self._prepare_hf_features(df)

        for hf_name in self._available_hf:
            if hf_name not in hf_data:
                continue

            hf_config = self.HF_FEATURES.get(hf_name, {})
            n_lags = hf_config.get("lags", 8)

            agg_df = self._aggregate_hf_to_mf(hf_data[hf_name], df.index, n_lags)
            weights = self._midas_transformers[hf_name]["weights"]
            weighted_feature = self._apply_midas_weights(agg_df, weights)
            df_prep[f"{hf_name}_midas"] = weighted_feature

        # Get prediction row
        test_row = df_prep.loc[[target_date]]
        X_test = self.scaler.transform(test_row[self._final_features].values)
        pred = self.model.predict(X_test)[0]

        # Decompose prediction
        monthly_pred = pred
        hf_contrib = {}
        for hf_name in self._available_hf:
            idx = self._final_features.index(f"{hf_name}_midas")
            coef = self.model.coef_[idx]
            hf_contrib[hf_name] = coef * test_row[f"{hf_name}_midas"].values[0]

        return {
            "date": target_date,
            "prediction": monthly_pred,
            "model": self.name,
            "features_used": self._final_features,
            "hf_features": self._available_hf,
            "hf_contribution": hf_contrib,
            "weight_type": self.weight_type,
        }

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Forecast iteratively.
        """
        self._check_fitted()

        if hasattr(self, "_train_df") and self._train_df is not None:
            target_col = getattr(self, "_target_col", "Все товары и услуги")
            return self.iterative_forecast(self._train_df, horizon, target_col)

        return np.zeros(horizon)

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        """
        Backtest the MIDAS model.
        """
        start = pd.Timestamp(start_date)

        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = MIDASForecaster(
                    weight_type=self.weight_type,
                    poly_order=self.poly_order,
                    hf_features=self.hf_features,
                    alpha=self.alpha,
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
                        "hf_features": ", ".join(pred_result["hf_features"]),
                        "weight_type": pred_result["weight_type"],
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from model coefficients.
        """
        self._check_fitted()

        importance = pd.DataFrame(
            {"feature": self._final_features, "coefficient": self.model.coef_}
        )
        importance["abs_coef"] = importance["coefficient"].abs()
        importance["is_hf"] = importance["feature"].str.contains("_midas")

        return importance.sort_values("abs_coef", ascending=False)

    def get_midas_weights(self, hf_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get MIDAS weights for a specific high-frequency feature.

        Returns:
            (weights, theta) - weights array and parameters
        """
        if hf_name not in self._midas_transformers:
            raise ValueError(f"HF feature {hf_name} not available")

        return (
            self._midas_transformers[hf_name]["weights"],
            self._midas_transformers[hf_name]["theta"],
        )

    def get_model_info(self) -> Dict:
        """
        Get model information.
        """
        return {
            "name": self.name,
            "weight_type": self.weight_type,
            "poly_order": self.poly_order,
            "hf_features": self._available_hf,
            "alpha": self.alpha,
            "n_features": len(self._final_features) if self._final_features else 0,
            "is_fitted": self._is_fitted,
        }
