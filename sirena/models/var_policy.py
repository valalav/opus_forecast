"""
Mandatory VAR-family policy model.

Horizon policy:
- h=1: RegimeMacroVARX_l1 (normal regime: VARX with USD/Ruonia/Ki_i;
  shock regime: robust Huber VAR without macro exog).
- h>=2: deterministic SeasonalVAR_CPI_F_NF_S trajectory.

The point path is deterministic and uses no random noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from .base import BaseForecaster
from .registry import ModelRegistry


ENDOG = ["CPI", "Food", "NonFood", "Services"]
MACRO = ["USD", "Ruonia", "Ki_i"]


@dataclass
class _PolicyForecast:
    path: np.ndarray
    regime: str


@ModelRegistry.register("var_policy")
class VARPolicyForecaster(BaseForecaster):
    """Leakage-safe mandatory VAR-family policy model."""

    name = "var_policy"
    MIN_TRAIN_SIZE = 48

    def __init__(
        self,
        shock_abs_cpi_threshold: float = 1.0,
        shock_vol_threshold: float = 0.55,
        huber_epsilon: float = 1.35,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shock_abs_cpi_threshold = shock_abs_cpi_threshold
        self.shock_vol_threshold = shock_vol_threshold
        self.huber_epsilon = huber_epsilon
        self.train_: Optional[pd.DataFrame] = None
        self.last_regime_: Optional[str] = None

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _load_official(cls, cutoff: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        path = cls._project_root() / "data" / "inflation_data.csv"
        df = pd.read_csv(path, sep=";", decimal=",")
        df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
        df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
        df = df.set_index("Date").sort_index()

        out = pd.DataFrame(index=df.index)
        out["CPI"] = pd.to_numeric(df["mom"], errors="coerce") - 100
        out["Food"] = pd.to_numeric(df["Prod"], errors="coerce") - 100
        out["NonFood"] = pd.to_numeric(df["Nonprod"], errors="coerce") - 100
        out["Services"] = pd.to_numeric(df["Serv"], errors="coerce") - 100
        out["USD"] = pd.to_numeric(df["usd_nom_i"], errors="coerce") - 100
        out["Ki_i"] = pd.to_numeric(df["Ki_i"], errors="coerce")
        out["Ruonia"] = pd.to_numeric(df["Ruonia"], errors="coerce")
        if cutoff is not None:
            out = out[out.index <= cutoff]
        return out.dropna(subset=["CPI"])

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
                raise ValueError("VARPolicyForecaster requires DatetimeIndex or Date column")
        work.index = pd.to_datetime(work.index).to_period("M").to_timestamp()
        work = work.sort_index()

        renamed = pd.DataFrame(index=work.index)
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
            "Ruonia": "Ruonia",
            "Ki_i": "Ki_i",
        }
        for source, target in mapping.items():
            if source in work.columns and target not in renamed.columns:
                values = pd.to_numeric(work[source], errors="coerce")
                if target in {"CPI", "Food", "NonFood", "Services", "USD"} and values.mean(skipna=True) and values.mean(skipna=True) > 50:
                    values = values - 100
                renamed[target] = values

        cutoff = pd.Timestamp(work.index.max())
        official = cls._load_official(cutoff=cutoff)
        for col in ENDOG + MACRO:
            if col not in renamed.columns or renamed[col].isna().all():
                renamed[col] = official[col]
            else:
                renamed[col] = renamed[col].combine_first(official[col])

        return renamed.loc[renamed.index <= cutoff, ENDOG + MACRO].dropna(subset=["CPI"])

    @staticmethod
    def _design(data: np.ndarray, lags: int, exog: Optional[np.ndarray] = None):
        x_rows, y_rows = [], []
        for t in range(lags, len(data)):
            row = [1.0]
            for lag in range(1, lags + 1):
                row.extend(data[t - lag])
            if exog is not None:
                row.extend(exog[t])
            x_rows.append(row)
            y_rows.append(data[t])
        return np.asarray(x_rows), np.asarray(y_rows)

    def _fit_equations(self, x: np.ndarray, y: np.ndarray, estimator: str) -> np.ndarray:
        betas = []
        for j in range(y.shape[1]):
            if estimator == "huber":
                try:
                    model = HuberRegressor(
                        alpha=0.0,
                        epsilon=self.huber_epsilon,
                        max_iter=300,
                        fit_intercept=False,
                    )
                    model.fit(x, y[:, j])
                    betas.append(model.coef_)
                    continue
                except Exception:
                    pass
            betas.append(np.linalg.lstsq(x, y[:, j], rcond=None)[0])
        return np.asarray(betas)

    def _eq_var_path(
        self,
        train: pd.DataFrame,
        horizon: int,
        *,
        estimator: str = "ols",
        exog_cols: Optional[List[str]] = None,
        min_train: int = 40,
    ) -> np.ndarray:
        exog_cols = exog_cols or []
        data = train.loc[:, ENDOG + exog_cols].dropna()
        if len(data) < max(min_train, 25):
            return np.full(horizon, np.nan)

        y_data = data.loc[:, ENDOG].values.astype(float)
        x_exog = data.loc[:, exog_cols].values.astype(float) if exog_cols else None
        x, y = self._design(y_data, 1, x_exog)
        if len(x) < 13:
            return np.full(horizon, np.nan)

        beta = self._fit_equations(x, y, estimator)
        hist = [y_data[-1]]
        exog_future = data.iloc[-1][exog_cols].values.astype(float) if exog_cols else None
        path = []
        cpi_idx = ENDOG.index("CPI")
        for _ in range(horizon):
            row = [1.0]
            row.extend(hist[-1])
            if exog_cols:
                row.extend(exog_future)
            pred_vec = (np.asarray([row]) @ beta.T).ravel()
            hist.append(pred_vec)
            path.append(float(pred_vec[cpi_idx]))
        return np.asarray(path)

    @staticmethod
    def _seasonal_var_path(train: pd.DataFrame, horizon: int) -> np.ndarray:
        data = train.loc[:, ENDOG].dropna()
        if len(data) < 40:
            return np.full(horizon, np.nan)

        month_means = data.groupby(data.index.month).mean()
        resid = data.copy()
        for month in range(1, 13):
            mask = resid.index.month == month
            if month in month_means.index:
                resid.loc[mask, ENDOG] = data.loc[mask, ENDOG] - month_means.loc[month, ENDOG].values

        arr = resid.values.astype(float)
        x_rows, y_rows = [], []
        for t in range(1, len(arr)):
            x_rows.append([1.0, *arr[t - 1]])
            y_rows.append(arr[t])
        x = np.asarray(x_rows)
        y = np.asarray(y_rows)
        if len(x) < 13:
            return np.full(horizon, np.nan)

        beta = np.asarray([np.linalg.lstsq(x, y[:, j], rcond=None)[0] for j in range(y.shape[1])])
        hist = [arr[-1]]
        last_date = data.index.max()
        path = []
        cpi_idx = ENDOG.index("CPI")
        for step in range(1, horizon + 1):
            pred_resid = (np.asarray([[1.0, *hist[-1]]]) @ beta.T).ravel()
            hist.append(pred_resid)
            month = (last_date + pd.DateOffset(months=step)).month
            seasonal = month_means.loc[month, ENDOG].values if month in month_means.index else np.zeros(len(ENDOG))
            path.append(float((pred_resid + seasonal)[cpi_idx]))
        return np.asarray(path)

    def _is_shock_regime(self, train: pd.DataFrame) -> bool:
        cpi = train["CPI"].dropna()
        if len(cpi) < 12:
            return False
        last_abs = abs(float(cpi.iloc[-1]))
        trailing_std = float(cpi.iloc[-12:].std())
        return last_abs >= self.shock_abs_cpi_threshold or trailing_std >= self.shock_vol_threshold

    def _regime_macro_varx_path(self, train: pd.DataFrame, horizon: int) -> _PolicyForecast:
        if self._is_shock_regime(train):
            path = self._eq_var_path(train, horizon, estimator="huber", min_train=40)
            return _PolicyForecast(path=path, regime="shock_huber_var")
        path = self._eq_var_path(train, horizon, estimator="ols", exog_cols=MACRO, min_train=42)
        return _PolicyForecast(path=path, regime="normal_macro_varx")

    def fit(self, df: pd.DataFrame, target_col: str = "Все товары и услуги") -> "VARPolicyForecaster":
        train = self._normalize_input(df)
        if len(train.dropna(subset=ENDOG)) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных для VARPolicyForecaster: {len(train)} < {self.MIN_TRAIN_SIZE}")
        self.train_ = train
        self._last_train_date = pd.Timestamp(train.index.max())
        self._is_fitted = True
        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        self._check_fitted()
        if self.train_ is None:
            raise ValueError("VARPolicyForecaster has no fitted training data")
        if horizon < 1:
            return np.array([])

        seasonal_path = self._seasonal_var_path(self.train_, horizon)
        if horizon == 1:
            regime_result = self._regime_macro_varx_path(self.train_, 1)
            self.last_regime_ = regime_result.regime
            return regime_result.path

        path = seasonal_path.copy()
        regime_result = self._regime_macro_varx_path(self.train_, 1)
        if np.isfinite(regime_result.path[0]):
            path[0] = regime_result.path[0]
        self.last_regime_ = f"h1:{regime_result.regime};h2plus:seasonal_var"
        return path

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        data = self._normalize_input(df)
        rows = []
        for target_date in data[data.index >= pd.Timestamp(start_date)].index:
            train = data[data.index < target_date]
            if len(train) < self.MIN_TRAIN_SIZE:
                continue
            model = self.__class__(
                shock_abs_cpi_threshold=self.shock_abs_cpi_threshold,
                shock_vol_threshold=self.shock_vol_threshold,
                huber_epsilon=self.huber_epsilon,
            )
            model.fit(train)
            prediction = float(model.forecast(1)[0])
            actual = float(data.loc[target_date, "CPI"])
            rows.append(
                {
                    "date": target_date,
                    "actual": actual,
                    "prediction": prediction,
                    "error": actual - prediction,
                    "model": self.name,
                    "regime": model.last_regime_,
                }
            )
        return pd.DataFrame(rows)

    def diagnostics(self) -> Dict[str, object]:
        return {
            "model": self.name,
            "last_train_date": self._last_train_date.strftime("%Y-%m-%d") if self._last_train_date else None,
            "last_regime": self.last_regime_,
            "policy": "h1 RegimeMacroVARX; h2+ SeasonalVAR trajectory",
            "no_random_noise": True,
        }
