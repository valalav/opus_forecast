"""
Transparent block-factor bridge challenger.

This model keeps the mandatory factor-family evidence interpretable: it builds
one train-only component factor and one train-only macro/financial factor, then
fits direct horizon equations for CPI. It is intended as a compact challenger to
the PCA/FAVAR policy model, not as a broad replacement for production models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA
from sklearn.linear_model import HuberRegressor, QuantileRegressor, Ridge
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster
from .factor_policy import COMPONENTS, DEFAULT_INFO, MACRO, TARGET, FactorPolicyForecaster
from .registry import ModelRegistry


BLOCK_COLUMNS = ("CPI", "ComponentFactor", "MacroFactor")


@dataclass
class _BridgeSpec:
    lags: int = 2
    estimator: str = "huber"
    seasonal: bool = True
    factor_method: str = "pca"
    component_cols: tuple[str, ...] = tuple(COMPONENTS)
    macro_cols: tuple[str, ...] = ("USD", "Ki_i", "Ruonia")
    max_horizon: int = 12
    rolling_window: Optional[int] = None


@ModelRegistry.register("factor_bridge")
class FactorBridgeForecaster(BaseForecaster):
    """Leakage-safe block factor bridge model with direct horizon equations."""

    name = "factor_bridge"
    MIN_TRAIN_SIZE = 72

    def __init__(
        self,
        lags: int = 2,
        estimator: str = "huber",
        seasonal: bool = True,
        factor_method: str = "pca",
        component_cols: Optional[Iterable[str]] = None,
        macro_cols: Optional[Iterable[str]] = None,
        max_horizon: int = 12,
        rolling_window: Optional[int] = None,
        ridge_alpha: float = 1.0,
        huber_epsilon: float = 1.35,
        quantile_alpha: float = 0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if lags < 1:
            raise ValueError("lags must be >= 1")
        if max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")
        estimator = estimator.lower()
        if estimator not in {"huber", "ridge", "quantile", "ols"}:
            raise ValueError("estimator must be one of: huber, ridge, quantile, ols")
        factor_method = factor_method.lower()
        if factor_method not in {"pca", "average", "weighted_pca", "sparse_pca"}:
            raise ValueError("factor_method must be one of: pca, average, weighted_pca, sparse_pca")

        self.spec = _BridgeSpec(
            lags=lags,
            estimator=estimator,
            seasonal=seasonal,
            factor_method=factor_method,
            component_cols=tuple(component_cols) if component_cols is not None else tuple(COMPONENTS),
            macro_cols=tuple(macro_cols) if macro_cols is not None else ("USD", "Ki_i", "Ruonia"),
            max_horizon=max_horizon,
            rolling_window=rolling_window,
        )
        self.ridge_alpha = ridge_alpha
        self.huber_epsilon = huber_epsilon
        self.quantile_alpha = quantile_alpha

        self.train_: Optional[pd.DataFrame] = None
        self.model_data_: Optional[pd.DataFrame] = None
        self.month_means_: Optional[pd.DataFrame] = None
        self.component_scaler_: Optional[StandardScaler] = None
        self.macro_scaler_: Optional[StandardScaler] = None
        self.component_pca_: Optional[object] = None
        self.macro_pca_: Optional[object] = None
        self.component_cols_: list[str] = []
        self.macro_cols_: list[str] = []
        self.models_: dict[int, object] = {}
        self.betas_: dict[int, np.ndarray] = {}

    @classmethod
    def _normalize_input(cls, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPolicyForecaster._normalize_input(df)

    @staticmethod
    def _available_nonconstant(data: pd.DataFrame, cols: Iterable[str]) -> list[str]:
        return [
            col
            for col in cols
            if col in data.columns
            and not data[col].isna().all()
            and data[col].nunique(dropna=True) > 1
        ]

    def _block_factor(self, data: pd.DataFrame, cols: list[str], block: str) -> np.ndarray:
        if not cols:
            return np.zeros(len(data), dtype=float)

        scaler = StandardScaler()
        values = scaler.fit_transform(data[cols].astype(float).values)
        if block == "component":
            self.component_scaler_ = scaler
        else:
            self.macro_scaler_ = scaler

        if len(cols) == 1:
            return values[:, 0]
        if self.spec.factor_method == "average":
            return values.mean(axis=1)

        if self.spec.factor_method == "weighted_pca":
            # Recent observations receive more influence, but weights are
            # deterministic within the training window and never use test data.
            weights = np.linspace(0.35, 1.0, len(values), dtype=float)
            weighted_values = values * np.sqrt(weights[:, None])
            pca = PCA(n_components=1)
            pca.fit(weighted_values)
            factor = values @ pca.components_[0]
        elif self.spec.factor_method == "sparse_pca":
            pca = SparsePCA(n_components=1, alpha=0.5, ridge_alpha=0.01, random_state=42, max_iter=300)
            factor = pca.fit_transform(values)[:, 0]
        else:
            pca = PCA(n_components=1)
            factor = pca.fit_transform(values)[:, 0]
        if block == "component":
            self.component_pca_ = pca
        else:
            self.macro_pca_ = pca
        return factor

    @staticmethod
    def _feature_row(values: np.ndarray, origin_idx: int, lags: int) -> np.ndarray:
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(values[origin_idx - lag + 1])
        return np.asarray(row, dtype=float)

    def _direct_design(self, block_data: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        values = block_data.loc[:, BLOCK_COLUMNS].values.astype(float)
        x_rows, y_rows = [], []
        last_origin = len(block_data) - horizon - 1
        for origin_idx in range(self.spec.lags - 1, last_origin + 1):
            x_rows.append(self._feature_row(values, origin_idx, self.spec.lags))
            y_rows.append(values[origin_idx + horizon, 0])
        return np.asarray(x_rows), np.asarray(y_rows)

    def _fit_model(self, x: np.ndarray, y: np.ndarray) -> object:
        if self.spec.estimator == "huber":
            model = HuberRegressor(
                alpha=0.0,
                epsilon=self.huber_epsilon,
                fit_intercept=False,
                max_iter=500,
            )
            model.fit(x, y)
            return model
        if self.spec.estimator == "ridge":
            model = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            model.fit(x, y)
            return model
        if self.spec.estimator == "quantile":
            model = QuantileRegressor(
                quantile=0.5,
                alpha=self.quantile_alpha,
                fit_intercept=False,
                solver="highs",
            )
            model.fit(x, y)
            return model
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        return beta

    @staticmethod
    def _predict_model(model: object, row: np.ndarray) -> float:
        if isinstance(model, np.ndarray):
            return float(row @ model)
        return float(model.predict(row.reshape(1, -1))[0])

    def fit(self, df: pd.DataFrame, target_col: str = "Все товары и услуги") -> "FactorBridgeForecaster":
        train = self._normalize_input(df)
        if self.spec.rolling_window is not None and len(train) > self.spec.rolling_window:
            train = train.tail(self.spec.rolling_window)
        if len(train) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных для FactorBridgeForecaster: {len(train)} < {self.MIN_TRAIN_SIZE}")

        info_cols = list(dict.fromkeys([*self.spec.component_cols, *self.spec.macro_cols]))
        model_data = train[[TARGET] + [col for col in info_cols if col in train.columns]].copy().ffill().bfill()

        self.month_means_ = None
        if self.spec.seasonal:
            self.month_means_ = model_data.groupby(model_data.index.month).mean()
            for month, means in self.month_means_.iterrows():
                mask = model_data.index.month == month
                cols = model_data.columns.intersection(means.index)
                model_data.loc[mask, cols] = model_data.loc[mask, cols] - means.loc[cols].values

        self.component_cols_ = self._available_nonconstant(model_data, self.spec.component_cols)
        self.macro_cols_ = self._available_nonconstant(model_data, self.spec.macro_cols)
        if not self.component_cols_:
            raise ValueError("Нет доступных компонентных рядов для FactorBridgeForecaster")

        block_data = pd.DataFrame(index=model_data.index)
        block_data["CPI"] = model_data[TARGET].astype(float)
        block_data["ComponentFactor"] = self._block_factor(model_data, self.component_cols_, "component")
        block_data["MacroFactor"] = self._block_factor(model_data, self.macro_cols_, "macro")
        block_data = block_data.dropna()

        self.models_.clear()
        self.betas_.clear()
        for horizon in range(1, self.spec.max_horizon + 1):
            x, y = self._direct_design(block_data, horizon)
            if len(x) < 24:
                continue
            try:
                self.models_[horizon] = self._fit_model(x, y)
            except Exception:
                self.models_[horizon] = np.linalg.lstsq(x, y, rcond=None)[0]

        if not self.models_:
            raise ValueError("Не удалось оценить ни одно direct horizon equation")

        self.train_ = train
        self.model_data_ = block_data
        self._last_train_date = pd.Timestamp(train.index.max())
        self._is_fitted = True
        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        self._check_fitted()
        if self.model_data_ is None:
            raise ValueError("FactorBridgeForecaster has no fitted block data")

        values = self.model_data_.loc[:, BLOCK_COLUMNS].values.astype(float)
        origin_idx = len(values) - 1
        if origin_idx < self.spec.lags - 1:
            raise ValueError("Недостаточно истории для factor bridge forecast")
        row = self._feature_row(values, origin_idx, self.spec.lags)

        path = []
        for step in range(1, horizon + 1):
            model = self.models_.get(step) or self.models_[max(self.models_)]
            pred = self._predict_model(model, row)
            if self.spec.seasonal and self.month_means_ is not None and self._last_train_date is not None:
                month = (self._last_train_date + pd.DateOffset(months=step)).month
                if month in self.month_means_.index:
                    pred += float(self.month_means_.loc[month, TARGET])
            path.append(pred)
        return np.asarray(path, dtype=float)

    def diagnostics(self) -> dict:
        component_exp = None
        macro_exp = None
        if self.component_pca_ is not None and hasattr(self.component_pca_, "explained_variance_ratio_"):
            component_exp = float(self.component_pca_.explained_variance_ratio_[0])
        if self.macro_pca_ is not None and hasattr(self.macro_pca_, "explained_variance_ratio_"):
            macro_exp = float(self.macro_pca_.explained_variance_ratio_[0])
        return {
            "model": self.name,
            "kind": "Block factor direct bridge",
            "lags": self.spec.lags,
            "estimator": self.spec.estimator,
            "seasonal": self.spec.seasonal,
            "factor_method": self.spec.factor_method,
            "component_cols": self.component_cols_,
            "macro_cols": self.macro_cols_,
            "rolling_window": self.spec.rolling_window,
            "component_factor_explained_variance": component_exp,
            "macro_factor_explained_variance": macro_exp,
            "max_horizon_fitted": max(self.models_) if self.models_ else None,
            "last_train_date": self._last_train_date.strftime("%Y-%m-%d") if self._last_train_date is not None else None,
        }

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> dict:
        cutoff = pd.Timestamp(target_date).to_period("M").to_timestamp() - pd.DateOffset(months=1)
        train = df.copy()
        if isinstance(train.index, pd.DatetimeIndex):
            train = train[pd.to_datetime(train.index).to_period("M").to_timestamp() <= cutoff]
        self.fit(train)
        return {
            "date": target_date,
            "prediction": float(self.forecast(1)[0] + 100),
            "model": self.name,
            "lags": self.spec.lags,
            "estimator": self.spec.estimator,
            "factor_method": self.spec.factor_method,
            "component_cols": self.component_cols_,
            "macro_cols": self.macro_cols_,
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        data = self._normalize_input(df)
        rows = []
        for target_date in data.loc[pd.Timestamp(start_date) :].index:
            cutoff = target_date - pd.DateOffset(months=1)
            train = data[data.index <= cutoff]
            if len(train) < self.MIN_TRAIN_SIZE:
                continue
            try:
                model = FactorBridgeForecaster(
                    lags=self.spec.lags,
                    estimator=self.spec.estimator,
                    seasonal=self.spec.seasonal,
                    factor_method=self.spec.factor_method,
                    component_cols=self.spec.component_cols,
                    macro_cols=self.spec.macro_cols,
                    max_horizon=self.spec.max_horizon,
                    rolling_window=self.spec.rolling_window,
                    ridge_alpha=self.ridge_alpha,
                    huber_epsilon=self.huber_epsilon,
                    quantile_alpha=self.quantile_alpha,
                )
                model.fit(train)
                pred = float(model.forecast(1)[0])
            except Exception:
                pred = np.nan
            actual = float(data.loc[target_date, TARGET])
            rows.append(
                {
                    "date": target_date,
                    "actual": actual,
                    "prediction": pred,
                    "error": actual - pred if not np.isnan(pred) else np.nan,
                }
            )
        return pd.DataFrame(rows)
