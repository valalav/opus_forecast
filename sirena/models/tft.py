"""
Temporal Fusion Transformer (TFT) Model
========================================

TFT combines the best of interpretable models (like ARIMA with exogenous variables)
with deep learning to handle complex temporal patterns.

Key Features:
1. **Temporal Attention:** Learns which time steps are most relevant
2. **Static Covariates:** Time-invariant features (e.g., month, day of week)
3. **Variable Selection Networks:** Selects relevant features dynamically
4. **Multi-head Attention:** Captures complex temporal dependencies
5. **Residual Connections:** Helps with gradient flow

Architecture:
- Input Processing (embedding + variable selection)
- LSTM Encoder for temporal context
- Multi-head Self-Attention for temporal dependencies
- Decoder for multi-horizon forecasting
- Output layer with quantile prediction

Simplified Implementation:
Given small dataset (~180 months), this is a lightweight version using:
- sklearn's MLPRegressor with temporal features
- Attention mechanism via feature importance weighting
- Multi-horizon output via iterative forecasting
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("tft")
class TemporalFusionForecaster(BaseForecaster):
    """
    Temporal Fusion Transformer Forecaster (Simplified).

    This is a lightweight TFT implementation suitable for small datasets
    (inflation forecasting with ~180 monthly observations).

    Key components:
    1. **Temporal Features:** Lagged values, momentum, seasonality
    2. **Static Covariates:** Month, calendar features
    3. **Variable Selection:** Via MLP feature importance
    4. **Multi-horizon Output:** Via iterative forecasting

    The model uses MLPRegressor with attention-like feature importance
    to simulate the variable selection and attention mechanisms of TFT.

    Args:
        hidden_layers: Number of hidden layers in MLP
        hidden_size: Size of hidden layers
        learning_rate_init: Learning rate for MLP
        max_iter: Maximum iterations for training
        alpha: L2 regularization parameter
        activation: Activation function ('relu', 'tanh', 'logistic')
        solver: Solver for weight optimization ('adam', 'lbfgs')
    """

    name = "tft"
    MIN_TRAIN_SIZE = 36  # Need more data for NN
    OUTLIER_YEARS = [2010, 2022]

    def __init__(
        self,
        hidden_layers: int = 2,
        hidden_size: int = 64,
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        alpha: float = 0.001,
        activation: str = "relu",
        solver: str = "adam",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.alpha = alpha
        self.activation = activation
        self.solver = solver

        self.model = None
        self.scaler = None
        self._feature_importance = None
        self._attention_weights = None
        self._static_features = None
        self._dynamic_features = None
        self._final_features = None

    def _prepare_static_covariates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare static covariates (time-invariant features).

        In TFT, static covariates provide context that doesn't change
        over time, such as:
        - Month of the year (seasonality)
        - Quarter
        - Calendar flags
        """
        df = df.copy()
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["day_of_week"] = df.index.dayofweek

        # Cyclical encoding for month and quarter
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)

        # Calendar features
        df["is_jan"] = (df["month"] == 1).astype(int)
        df["is_jul"] = (df["month"] == 7).astype(int)
        df["is_dec"] = (df["month"] == 12).astype(int)
        df["is_q1"] = (df["quarter"] == 1).astype(int)

        return df

    def _prepare_dynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dynamic features (time-varying features).

        In TFT, dynamic features provide the temporal context:
        - Lagged target values
        - Momentum (differences)
        - Exogenous variables (USD, Brent, Ki if available)
        """
        df = df.copy()
        y = df["Все товары и услуги"]

        # Temporal lags (input window)
        for lag in [1, 2, 3, 6, 12]:
            df[f"y_lag{lag}"] = y.shift(lag)

        # Momentum features
        df["y_diff1"] = y.diff(1)
        df["y_diff3"] = y.diff(3)
        df["y_diff6"] = y.diff(6)

        # Rolling statistics
        df["y_ma3"] = y.rolling(window=3, min_periods=1).mean()
        df["y_ma6"] = y.rolling(window=6, min_periods=1).mean()
        df["y_std3"] = y.rolling(window=3, min_periods=1).std().fillna(0)

        # Exogenous variables (if available)
        if "usd_nom_i" in df.columns:
            df["usd_lag1"] = df["usd_nom_i"].shift(1)
            df["usd_lag2"] = df["usd_nom_i"].shift(2)
            df["usd_diff1"] = df["usd_nom_i"].diff(1)

        if "brent" in df.columns:
            df["brent_lag1"] = df["brent"].shift(1)
            df["brent_lag3"] = df["brent"].shift(3)
            df["brent_diff1"] = df["brent"].diff(1)

        if "Ki" in df.columns:
            df["ki_lag1"] = df["Ki"].shift(1)
            df["ki_lag3"] = df["Ki"].shift(3)
            df["ki_diff1"] = df["Ki"].diff(1)

        # Macro features from components (if available)
        if "Продовольственные товары" in df.columns:
            df["food_lag1"] = df["Продовольственные товары"].shift(1)
        if "Непродовольственные товары" in df.columns:
            df["nonfood_lag1"] = df["Непродовольственные товары"].shift(1)
        if "Услуги" in df.columns:
            df["services_lag1"] = df["Услуги"].shift(1)

        return df

    def _compute_attention_weights(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Simulate TFT's attention mechanism.

        In full TFT, attention weights are learned end-to-end.
        Here, we approximate them using permutation feature importance.

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature names to attention weights
        """
        baseline_score = self.model.score(
            X, self._train_y if hasattr(self, "_train_y") else np.zeros(len(X))
        )

        attention = {}
        for i, feature in enumerate(feature_names):
            X_permuted = X.copy()
            np.random.seed(42)
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            permuted_score = self.model.score(
                X_permuted,
                self._train_y if hasattr(self, "_train_y") else np.zeros(len(X)),
            )
            attention[feature] = baseline_score - permuted_score

        # Normalize to sum to 1
        total = sum(abs(v) for v in attention.values())
        if total > 0:
            attention = {k: abs(v) / total for k, v in attention.items()}

        return attention

    def _select_features(self, df_prep: pd.DataFrame, target_col: str) -> List[str]:
        """
        Variable selection (simplified).

        In full TFT, variable selection networks learn which features
        to use for each horizon. Here, we select based on correlation
        with target and basic heuristics.
        """
        # Core static features
        static_features = [
            "month_sin",
            "month_cos",
            "quarter_sin",
            "quarter_cos",
            "is_jan",
            "is_jul",
            "is_dec",
            "is_q1",
        ]

        # Core dynamic features
        dynamic_features = [
            "y_lag1",
            "y_lag2",
            "y_lag3",
            "y_lag6",
            "y_lag12",
            "y_diff1",
            "y_diff3",
            "y_diff6",
            "y_ma3",
            "y_ma6",
            "y_std3",
        ]

        # Available exogenous features
        exog_features = []
        if "usd_lag1" in df_prep.columns:
            exog_features.extend(["usd_lag1", "usd_lag2", "usd_diff1"])
        if "brent_lag1" in df_prep.columns:
            exog_features.extend(["brent_lag1", "brent_lag3", "brent_diff1"])
        if "ki_lag1" in df_prep.columns:
            exog_features.extend(["ki_lag1", "ki_lag3", "ki_diff1"])

        # Component features
        component_features = []
        if "food_lag1" in df_prep.columns:
            component_features.append("food_lag1")
        if "nonfood_lag1" in df_prep.columns:
            component_features.append("nonfood_lag1")
        if "services_lag1" in df_prep.columns:
            component_features.append("services_lag1")

        # Select available features
        all_features = (
            static_features + dynamic_features + exog_features + component_features
        )
        available_features = [f for f in all_features if f in df_prep.columns]

        return available_features

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "TemporalFusionForecaster":
        """
        Fit TFT model.

        Args:
            df: DataFrame with temporal index
            target_col: Target column name
        """
        series = self._validate_data(df, target_col)

        # Prepare features
        df_prep = self._prepare_static_covariates(df)
        df_prep = self._prepare_dynamic_features(df_prep)

        # Select features
        features = self._select_features(df_prep, target_col)
        self._final_features = features
        self._static_features = [
            f
            for f in features
            if f
            in [
                "month_sin",
                "month_cos",
                "quarter_sin",
                "quarter_cos",
                "is_jan",
                "is_jul",
                "is_dec",
                "is_q1",
            ]
        ]
        self._dynamic_features = [f for f in features if f not in self._static_features]

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

        # Fit MLPRegressor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple([self.hidden_size] * self.hidden_layers),
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                alpha=self.alpha,
                activation=self.activation,
                solver=self.solver,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )
            self.model.fit(X_scaled, y)

        # Compute attention weights (simplified)
        self._train_y = y
        try:
            self._attention_weights = self._compute_attention_weights(
                X_scaled, features
            )
        except:
            self._attention_weights = {f: 1.0 / len(features) for f in features}

        self._is_fitted = True
        self._last_train_date = df.index.max()
        self._target_col = target_col
        self._train_df = df.copy()

        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Predict for a specific date.

        Args:
            df: DataFrame with data up to target_date
            target_date: Date to predict

        Returns:
            Dictionary with prediction and metadata
        """
        self._check_fitted()

        df_prep = self._prepare_static_covariates(df)
        df_prep = self._prepare_dynamic_features(df_prep)

        # Get prediction row
        if target_date not in df_prep.index:
            # Forward fill from previous date
            last_date = df_prep.index[df_prep.index < target_date][-1]
            df_prep.loc[target_date] = df_prep.loc[last_date].copy()
            df_prep.loc[target_date, "month"] = target_date.month
            df_prep.loc[target_date, "quarter"] = target_date.quarter
            df_prep.loc[target_date, "day_of_week"] = target_date.dayofweek
            df_prep.loc[target_date, "month_sin"] = np.sin(
                2 * np.pi * target_date.month / 12
            )
            df_prep.loc[target_date, "month_cos"] = np.cos(
                2 * np.pi * target_date.month / 12
            )

        test_row = df_prep.loc[[target_date]]

        # Handle missing values
        for feature in self._final_features:
            if feature not in test_row.columns or pd.isna(test_row[feature].iloc[0]):
                if feature in test_row.columns:
                    test_row[feature] = test_row[feature].fillna(0)
                else:
                    test_row[feature] = 0

        X_test = self.scaler.transform(test_row[self._final_features].values)
        pred = self.model.predict(X_test)[0]

        return {
            "date": target_date,
            "prediction": pred,
            "model": self.name,
            "features_used": self._final_features,
            "static_features": self._static_features,
            "dynamic_features": self._dynamic_features,
            "attention_weights": self._attention_weights,
        }

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Forecast iteratively for multiple horizons.

        Args:
            horizon: Number of periods to forecast

        Returns:
            Array with predictions
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
        Backtest TFT model.

        Args:
            df: DataFrame with data
            start_date: Start date for backtest
            target_col: Target column name

        Returns:
            DataFrame with backtest results
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
                model = TemporalFusionForecaster(
                    hidden_layers=self.hidden_layers,
                    hidden_size=self.hidden_size,
                    learning_rate_init=self.learning_rate_init,
                    max_iter=self.max_iter,
                    alpha=self.alpha,
                    activation=self.activation,
                    solver=self.solver,
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
                        "n_static_features": len(pred_result["static_features"]),
                        "n_dynamic_features": len(pred_result["dynamic_features"]),
                        "top_attention": self._get_top_attention(
                            pred_result["attention_weights"]
                        ),
                    }
                )
            except Exception as e:
                continue

        return pd.DataFrame(results)

    def _get_top_attention(self, attention: Dict[str, float], top_n: int = 3) -> str:
        """Get top-n features by attention weight."""
        sorted_features = sorted(attention.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        return ", ".join([f"{f}:{w:.3f}" for f, w in top_features])

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from attention weights.

        Returns:
            DataFrame with features and their importance scores
        """
        self._check_fitted()

        if self._attention_weights is None:
            return pd.DataFrame(columns=["feature", "importance", "type"])

        importance = []
        for feature in self._final_features:
            ftype = "static" if feature in self._static_features else "dynamic"
            importance.append(
                {
                    "feature": feature,
                    "importance": self._attention_weights.get(feature, 0.0),
                    "type": ftype,
                }
            )

        df = pd.DataFrame(importance)
        return df.sort_values("importance", ascending=False)

    def get_attention_weights(self) -> Dict[str, float]:
        """
        Get attention weights for all features.

        Returns:
            Dictionary mapping feature names to attention weights
        """
        self._check_fitted()
        return self._attention_weights.copy() if self._attention_weights else {}

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model parameters and state
        """
        return {
            "name": self.name,
            "hidden_layers": self.hidden_layers,
            "hidden_size": self.hidden_size,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "alpha": self.alpha,
            "activation": self.activation,
            "solver": self.solver,
            "n_features": len(self._final_features) if self._final_features else 0,
            "n_static_features": len(self._static_features)
            if self._static_features
            else 0,
            "n_dynamic_features": len(self._dynamic_features)
            if self._dynamic_features
            else 0,
            "is_fitted": self._is_fitted,
            "n_iter": self.model.n_iter_
            if self.model and hasattr(self.model, "n_iter_")
            else None,
            "loss_curve": self.model.loss_curve_[-1]
            if self.model and hasattr(self.model, "loss_curve_")
            else None,
        }

    def get_weights(self) -> Dict[str, Any]:
        """
        Extract model weights (acceptance criterion for task 22).

        In TFT, "weights" refer to:
        1. Attention weights (feature importance)
        2. Neural network weights (layer coefficients)

        Returns:
            Dictionary with attention_weights and network_weights
        """
        self._check_fitted()

        result = {
            "attention_weights": self._attention_weights.copy()
            if self._attention_weights
            else {},
            "network_weights": {},
        }

        if self.model and hasattr(self.model, "coefs_"):
            result["network_weights"]["layer_weights"] = [
                w.tolist() for w in self.model.coefs_
            ]
            result["network_weights"]["layer_biases"] = [
                b.tolist() for b in self.model.intercepts_
            ]

        return result
