"""
Diagnostics-aware stationary block FAVAR model.

This is the report-oriented factor model selected after the factor diagnostics
track. It keeps a compact VAR system on CPI plus two interpretable PCA blocks:
component inflation and stationary monetary conditions.
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
MONETARY = ["USD", "Ki_i", "d_spread_Ruonia_Ki"]


@dataclass
class _StationaryBlockSpec:
    lags: int = 1
    robust: bool = True
    seasonal: bool = True
    macro_cols: tuple[str, ...] = tuple(MONETARY)


@ModelRegistry.register("stationary_block_favar")
class StationaryBlockFAVARForecaster(BaseForecaster):
    """Leakage-safe stationary block FAVAR selected by econometric gates."""

    name = "stationary_block_favar"
    MIN_TRAIN_SIZE = 72

    def __init__(
        self,
        lags: int = 1,
        robust: bool = True,
        seasonal: bool = True,
        macro_cols: Optional[Iterable[str]] = None,
        huber_epsilon: float = 1.35,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spec = _StationaryBlockSpec(
            lags=lags,
            robust=robust,
            seasonal=seasonal,
            macro_cols=tuple(macro_cols) if macro_cols is not None else tuple(MONETARY),
        )
        self.huber_epsilon = huber_epsilon
        self.train_: Optional[pd.DataFrame] = None
        self.month_means_: Optional[pd.DataFrame] = None
        self.component_scaler_: Optional[StandardScaler] = None
        self.monetary_scaler_: Optional[StandardScaler] = None
        self.component_pca_: Optional[PCA] = None
        self.monetary_pca_: Optional[PCA] = None
        self.beta_: Optional[np.ndarray] = None
        self.var_columns_: list[str] = ["CPI", "ComponentFactor", "MonetaryFactor"]

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
        out["spread_Ruonia_Ki"] = out["Ruonia"] - out["Ki_i"]
        out["d_spread_Ruonia_Ki"] = out["spread_Ruonia_Ki"].diff()
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
                raise ValueError("StationaryBlockFAVARForecaster requires DatetimeIndex or Date column")
        work.index = pd.to_datetime(work.index).to_period("M").to_timestamp()
        work = work.sort_index()

        mapping = {
            "CPI": "CPI",
            "Все товары и услуги": "CPI",
            "mom": "CPI",
            "Food": "Food",
            "Продовольственные товары": "Food",
            "Prod": "Food",
            "NonFood": "NonFood",
            "Непродовольственные товары": "NonFood",
            "Nonprod": "NonFood",
            "Services": "Services",
            "Услуги": "Services",
            "Serv": "Services",
            "usd_nom_i": "USD",
            "USD": "USD",
            "Ki_i": "Ki_i",
            "Ruonia": "Ruonia",
            "d_spread_Ruonia_Ki": "d_spread_Ruonia_Ki",
        }
        normalized = pd.DataFrame(index=work.index)
        for source, target in mapping.items():
            if source in work.columns and target not in normalized.columns:
                values = pd.to_numeric(work[source], errors="coerce")
                if target in {"CPI", "Food", "NonFood", "Services", "USD"}:
                    if values.mean(skipna=True) and values.mean(skipna=True) > 50:
                        values = values - 100
                normalized[target] = values

        cutoff = pd.Timestamp(work.index.max())
        official = cls._load_official(cutoff=cutoff)
        for col in [TARGET] + COMPONENTS + ["USD", "Ki_i", "Ruonia"]:
            if col not in normalized.columns or normalized[col].isna().all():
                normalized[col] = official[col]
            else:
                normalized[col] = normalized[col].combine_first(official[col])

        normalized["spread_Ruonia_Ki"] = normalized["Ruonia"] - normalized["Ki_i"]
        spread_diff = normalized["spread_Ruonia_Ki"].diff()
        if "d_spread_Ruonia_Ki" in normalized.columns:
            normalized["d_spread_Ruonia_Ki"] = normalized["d_spread_Ruonia_Ki"].combine_first(spread_diff)
        else:
            normalized["d_spread_Ruonia_Ki"] = spread_diff
        cols = [TARGET] + COMPONENTS + ["USD", "Ki_i", "d_spread_Ruonia_Ki"]
        return normalized.loc[normalized.index <= cutoff, cols].dropna(subset=[TARGET])

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
                        max_iter=500,
                    )
                    model.fit(x, y[:, col_idx])
                    betas.append(model.coef_)
                    continue
                except Exception:
                    pass
            betas.append(np.linalg.lstsq(x, y[:, col_idx], rcond=None)[0])
        return np.asarray(betas)

    def _prepare_var_data(self, train: pd.DataFrame, fit_blocks: bool) -> pd.DataFrame:
        cols = [TARGET] + COMPONENTS + list(self.spec.macro_cols)
        model_data = train[cols].copy().ffill().bfill().dropna()
        if self.spec.seasonal:
            if fit_blocks:
                self.month_means_ = model_data.groupby(model_data.index.month).mean()
            if self.month_means_ is not None:
                for month, means in self.month_means_.iterrows():
                    mask = model_data.index.month == month
                    model_data.loc[mask, :] = model_data.loc[mask, :] - means.values

        component_values = model_data[COMPONENTS].values.astype(float)
        monetary_values = model_data[list(self.spec.macro_cols)].values.astype(float)
        if fit_blocks:
            self.component_scaler_ = StandardScaler()
            self.monetary_scaler_ = StandardScaler()
            self.component_pca_ = PCA(n_components=1)
            self.monetary_pca_ = PCA(n_components=1)
            comp_scaled = self.component_scaler_.fit_transform(component_values)
            mon_scaled = self.monetary_scaler_.fit_transform(monetary_values)
            comp_factor = self.component_pca_.fit_transform(comp_scaled)[:, 0]
            mon_factor = self.monetary_pca_.fit_transform(mon_scaled)[:, 0]
        else:
            comp_scaled = self.component_scaler_.transform(component_values)
            mon_scaled = self.monetary_scaler_.transform(monetary_values)
            comp_factor = self.component_pca_.transform(comp_scaled)[:, 0]
            mon_factor = self.monetary_pca_.transform(mon_scaled)[:, 0]

        return pd.DataFrame(
            {
                "CPI": model_data[TARGET].values.astype(float),
                "ComponentFactor": comp_factor,
                "MonetaryFactor": mon_factor,
            },
            index=model_data.index,
        )

    def fit(self, df: pd.DataFrame, target_col: str = "Все товары и услуги") -> "StationaryBlockFAVARForecaster":
        train = self._normalize_input(df)
        if len(train) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных для StationaryBlockFAVARForecaster: {len(train)} < {self.MIN_TRAIN_SIZE}")

        var_data = self._prepare_var_data(train, fit_blocks=True)
        design_x, design_y = self._make_design(var_data.values.astype(float), self.spec.lags)
        if len(design_x) < 24:
            raise ValueError("Недостаточно строк для оценки stationary block FAVAR")

        self.beta_ = self._fit_equations(design_x, design_y)
        self.train_ = train
        self._last_train_date = pd.Timestamp(train.index.max())
        self._is_fitted = True
        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        self._check_fitted()
        if self.train_ is None or self.beta_ is None:
            raise ValueError("StationaryBlockFAVARForecaster has no fitted state")

        var_data = self._prepare_var_data(self.train_, fit_blocks=False)
        history = [row for row in var_data.values[-self.spec.lags :]]
        path = []
        for step in range(1, horizon + 1):
            row = [1.0]
            for lag in range(1, self.spec.lags + 1):
                row.extend(history[-lag])
            pred_vec = (np.asarray([row]) @ self.beta_.T).ravel()
            history.append(pred_vec)
            cpi = float(pred_vec[0])
            if self.spec.seasonal and self.month_means_ is not None:
                month = (self._last_train_date + pd.DateOffset(months=step)).month
                cpi += float(self.month_means_.loc[month, TARGET])
            path.append(cpi)
        return np.asarray(path)

    def diagnostics(self) -> dict:
        comp_ev = None
        mon_ev = None
        comp_loadings = None
        mon_loadings = None
        if self.component_pca_ is not None:
            comp_ev = float(self.component_pca_.explained_variance_ratio_[0])
            comp_loadings = dict(zip(COMPONENTS, self.component_pca_.components_[0].astype(float)))
        if self.monetary_pca_ is not None:
            mon_ev = float(self.monetary_pca_.explained_variance_ratio_[0])
            mon_loadings = dict(zip(self.spec.macro_cols, self.monetary_pca_.components_[0].astype(float)))
        return {
            "model": self.name,
            "kind": "Diagnostics-aware stationary block FAVAR",
            "lags": self.spec.lags,
            "seasonal": self.spec.seasonal,
            "robust": self.spec.robust,
            "component_block": COMPONENTS,
            "monetary_block": list(self.spec.macro_cols),
            "component_explained_variance": comp_ev,
            "monetary_explained_variance": mon_ev,
            "component_loadings": comp_loadings,
            "monetary_loadings": mon_loadings,
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
            "seasonal": self.spec.seasonal,
            "robust": self.spec.robust,
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
                model = StationaryBlockFAVARForecaster(
                    lags=self.spec.lags,
                    robust=self.spec.robust,
                    seasonal=self.spec.seasonal,
                    macro_cols=self.spec.macro_cols,
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
