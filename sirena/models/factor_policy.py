"""
Mandatory factor-family policy model.

The model is intentionally conservative: factors are estimated from training data
only with PCA, then a small VAR is fitted on CPI plus latent factors. The selected
policy uses explicit train-only month seasonality and robust equation estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster
from .registry import ModelRegistry


TARGET = "CPI"
COMPONENTS = ["Food", "NonFood", "Services"]
MACRO = ["USD", "Ki_i", "Ruonia", "Ruonia_i", "Deposits", "RetailReal"]
AVAILABLE_INFO = COMPONENTS + MACRO
DEFAULT_INFO = COMPONENTS + ["USD", "Ki_i", "Ruonia"]


@dataclass
class _FactorSpec:
    n_factors: int = 2
    lags: int = 1
    seasonal: bool = True
    robust: bool = True
    info_cols: tuple[str, ...] = tuple(DEFAULT_INFO)


@ModelRegistry.register("factor_policy")
class FactorPolicyForecaster(BaseForecaster):
    """Leakage-safe factor model selected from the 2026 factor research track."""

    name = "factor_policy"
    MIN_TRAIN_SIZE = 60

    def __init__(
        self,
        n_factors: int = 2,
        lags: int = 1,
        seasonal: bool = True,
        robust: bool = True,
        info_cols: Optional[Iterable[str]] = None,
        huber_epsilon: float = 1.35,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spec = _FactorSpec(
            n_factors=n_factors,
            lags=lags,
            seasonal=seasonal,
            robust=robust,
            info_cols=tuple(info_cols) if info_cols is not None else tuple(DEFAULT_INFO),
        )
        self.huber_epsilon = huber_epsilon
        self.train_: Optional[pd.DataFrame] = None
        self.month_means_: Optional[pd.DataFrame] = None
        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        self.beta_: Optional[np.ndarray] = None
        self.var_columns_: Optional[list[str]] = None
        self.used_info_cols_: Optional[list[str]] = None

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _load_official(cls, cutoff: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        path = cls._project_root() / "data" / "inflation_data.csv"
        raw = pd.read_csv(path, sep=";", decimal=",")
        raw["Date"] = pd.to_datetime(raw["Date"], format="%d.%m.%Y", errors="coerce")
        raw["Date"] = raw["Date"].dt.to_period("M").dt.to_timestamp()
        raw = raw.set_index("Date").sort_index()

        out = pd.DataFrame(index=raw.index)
        out["CPI"] = pd.to_numeric(raw["mom"], errors="coerce") - 100
        out["Food"] = pd.to_numeric(raw["Prod"], errors="coerce") - 100
        out["NonFood"] = pd.to_numeric(raw["Nonprod"], errors="coerce") - 100
        out["Services"] = pd.to_numeric(raw["Serv"], errors="coerce") - 100
        out["USD"] = pd.to_numeric(raw["usd_nom_i"], errors="coerce") - 100
        out["Ki_i"] = pd.to_numeric(raw["Ki_i"], errors="coerce")
        out["Ruonia"] = pd.to_numeric(raw["Ruonia"], errors="coerce")
        out["Ruonia_i"] = pd.to_numeric(raw["Ruonia_i"], errors="coerce")
        out["Deposits"] = pd.to_numeric(raw["fl_dep"], errors="coerce") - 100
        out["RetailReal"] = pd.to_numeric(raw["all_real"], errors="coerce") - 100
        if cutoff is not None:
            out = out[out.index <= cutoff]
        return out.dropna(subset=[TARGET])

    @classmethod
    def _normalize_input(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("DataFrame пустой")

        work = df.copy()
        if not isinstance(work.index, pd.DatetimeIndex):
            if "Date" in work.columns:
                work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
                work = work.set_index("Date")
            else:
                raise ValueError("FactorPolicyForecaster requires DatetimeIndex or Date column")
        work.index = pd.to_datetime(work.index).to_period("M").to_timestamp()
        work = work.sort_index()

        mapping = {
            "Все товары и услуги": "CPI",
            "mom": "CPI",
            "Продовольственные товары": "Food",
            "Prod": "Food",
            "Непродовольственные товары": "NonFood",
            "Nonprod": "NonFood",
            "Услуги": "Services",
            "Serv": "Services",
            "usd_nom_i": "USD",
            "USD": "USD",
            "Ki_i": "Ki_i",
            "Ruonia": "Ruonia",
            "Ruonia_i": "Ruonia_i",
            "fl_dep": "Deposits",
            "all_real": "RetailReal",
        }
        normalized = pd.DataFrame(index=work.index)
        for source, target in mapping.items():
            if source in work.columns and target not in normalized.columns:
                values = pd.to_numeric(work[source], errors="coerce")
                if target in {"CPI", "Food", "NonFood", "Services", "USD", "Deposits", "RetailReal"}:
                    if values.mean(skipna=True) and values.mean(skipna=True) > 50:
                        values = values - 100
                normalized[target] = values

        cutoff = pd.Timestamp(work.index.max())
        official = cls._load_official(cutoff=cutoff)
        for col in [TARGET] + AVAILABLE_INFO:
            if col not in normalized.columns or normalized[col].isna().all():
                normalized[col] = official[col]
            else:
                normalized[col] = normalized[col].combine_first(official[col])

        return normalized.loc[normalized.index <= cutoff, [TARGET] + AVAILABLE_INFO].dropna(subset=[TARGET])

    @staticmethod
    def _make_design(values: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
        rows, targets = [], []
        for t in range(lags, len(values)):
            row = [1.0]
            for lag in range(1, lags + 1):
                row.extend(values[t - lag])
            rows.append(row)
            targets.append(values[t])
        return np.asarray(rows), np.asarray(targets)

    def _fit_equations(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        betas = []
        for col_idx in range(y.shape[1]):
            if self.spec.robust:
                try:
                    model = HuberRegressor(
                        alpha=0.0,
                        epsilon=self.huber_epsilon,
                        fit_intercept=False,
                        max_iter=400,
                    )
                    model.fit(x, y[:, col_idx])
                    betas.append(model.coef_)
                    continue
                except Exception:
                    pass
            betas.append(np.linalg.lstsq(x, y[:, col_idx], rcond=None)[0])
        return np.asarray(betas)

    def fit(self, df: pd.DataFrame, target_col: str = "Все товары и услуги") -> "FactorPolicyForecaster":
        train = self._normalize_input(df)
        if len(train) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных для FactorPolicyForecaster: {len(train)} < {self.MIN_TRAIN_SIZE}")

        used_info = [col for col in self.spec.info_cols if col in train.columns and not train[col].isna().all()]
        if len(used_info) < self.spec.n_factors:
            raise ValueError(
                f"Недостаточно информационных рядов для PCA: {len(used_info)} < {self.spec.n_factors}"
            )

        model_data = train[[TARGET] + used_info].copy().ffill().bfill()
        self.month_means_ = None
        if self.spec.seasonal:
            self.month_means_ = model_data.groupby(model_data.index.month).mean()
            for month, means in self.month_means_.iterrows():
                mask = model_data.index.month == month
                model_data.loc[mask, :] = model_data.loc[mask, :] - means.values

        x_info = model_data[used_info].astype(float)
        nonconstant = [col for col in used_info if x_info[col].nunique(dropna=True) > 1]
        if len(nonconstant) < self.spec.n_factors:
            raise ValueError(
                f"После удаления константных рядов осталось {len(nonconstant)} признаков < {self.spec.n_factors}"
            )
        x_info = x_info[nonconstant]

        self.scaler_ = StandardScaler()
        x_scaled = self.scaler_.fit_transform(x_info.values)
        self.pca_ = PCA(n_components=self.spec.n_factors)
        factors = self.pca_.fit_transform(x_scaled)

        factor_cols = [f"Factor_{i + 1}" for i in range(self.spec.n_factors)]
        var_data = pd.DataFrame(factors, index=model_data.index, columns=factor_cols)
        var_data.insert(0, TARGET, model_data[TARGET].astype(float).values)
        var_data = var_data.dropna()
        if len(var_data) < self.MIN_TRAIN_SIZE - 12:
            raise ValueError("Недостаточно данных после подготовки факторной VAR-системы")

        design_x, design_y = self._make_design(var_data.values.astype(float), self.spec.lags)
        if len(design_x) < 24:
            raise ValueError("Недостаточно строк для оценки факторной VAR-системы")

        self.beta_ = self._fit_equations(design_x, design_y)
        self.var_columns_ = list(var_data.columns)
        self.used_info_cols_ = nonconstant
        self.train_ = train
        self._last_train_date = pd.Timestamp(train.index.max())
        self._is_fitted = True
        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        self._check_fitted()
        if self.train_ is None or self.beta_ is None:
            raise ValueError("FactorPolicyForecaster has no fitted state")

        train = self.train_[[TARGET] + list(self.used_info_cols_ or [])].copy().ffill().bfill()
        if self.spec.seasonal and self.month_means_ is not None:
            for month, means in self.month_means_.iterrows():
                mask = train.index.month == month
                cols = train.columns.intersection(means.index)
                train.loc[mask, cols] = train.loc[mask, cols] - means.loc[cols].values

        x_info = train[list(self.used_info_cols_ or [])].astype(float).values
        factors = self.pca_.transform(self.scaler_.transform(x_info))
        var_hist = pd.DataFrame(factors, index=train.index, columns=self.var_columns_[1:])
        var_hist.insert(0, TARGET, train[TARGET].astype(float).values)
        history = [row for row in var_hist.values[-self.spec.lags :]]

        path = []
        for step in range(1, horizon + 1):
            row = [1.0]
            for lag in range(1, self.spec.lags + 1):
                row.extend(history[-lag])
            pred = (np.asarray([row]) @ self.beta_.T).ravel()
            history.append(pred)
            cpi = float(pred[0])
            if self.spec.seasonal and self.month_means_ is not None:
                month = (self._last_train_date + pd.DateOffset(months=step)).month
                if month in self.month_means_.index:
                    cpi += float(self.month_means_.loc[month, TARGET])
            path.append(cpi)
        return np.asarray(path)

    def diagnostics(self) -> dict:
        """Return compact model diagnostics for forecast cache/reporting."""
        explained = []
        if self.pca_ is not None:
            explained = [float(x) for x in self.pca_.explained_variance_ratio_]
        return {
            "model": self.name,
            "kind": "Robust seasonal FAVAR",
            "n_factors": self.spec.n_factors,
            "lags": self.spec.lags,
            "seasonal": self.spec.seasonal,
            "robust": self.spec.robust,
            "info_cols": list(self.used_info_cols_ or self.spec.info_cols),
            "pca_explained_variance_ratio": explained,
            "pca_explained_variance_sum": float(sum(explained)) if explained else None,
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
            "n_factors": self.spec.n_factors,
            "lags": self.spec.lags,
            "seasonal": self.spec.seasonal,
            "robust": self.spec.robust,
            "info_cols": list(self.used_info_cols_ or []),
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
                model = FactorPolicyForecaster(
                    n_factors=self.spec.n_factors,
                    lags=self.spec.lags,
                    seasonal=self.spec.seasonal,
                    robust=self.spec.robust,
                    info_cols=self.spec.info_cols,
                    huber_epsilon=self.huber_epsilon,
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
