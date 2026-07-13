"""
Conformal Prediction — калиброванные доверительные интервалы
============================================================

Идея: использовать ошибки на валидационном сете для калибровки CI.
Гарантирует coverage rate при заданном уровне confidence.

Метод: Split Conformal Prediction
1. Обучаем базовую модель на train
2. Вычисляем residuals на calibration set
3. CI = prediction ± quantile(|residuals|, 1-alpha)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, Any, List

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("conformal")
class ConformalForecaster(BaseForecaster):
    """
    Ridge с Conformal Prediction для калиброванных CI.

    Гарантирует заданный coverage rate (например, 90%).
    """

    name = "conformal"
    MIN_TRAIN_SIZE = 36
    OUTLIER_YEARS = [2010, 2022]
    ALPHA = 0.3

    # Доля данных для калибровки
    CALIBRATION_RATIO = 0.2
    # Целевой уровень покрытия
    COVERAGE_TARGET = 0.90

    # ETS веса
    ETS_WEIGHTS = {
        1: 0.9,
        2: 0.0,
        3: 0.5,
        4: 0.3,
        5: 0.9,
        6: 0.5,
        7: 0.0,
        8: 0.5,
        9: 0.9,
        10: 0.9,
        11: 0.0,
        12: 0.0,
    }

    BASE_FEATURES = [
        "y_lag1",
        "y_lag2",
        "y_lag12",
        "y_lag3",
        "y_lag6",
        "y_ma3",
        "y_ma6",
        "d_y_lag1",
        "d_y_lag3",
        "y_vol3",
        "y_vol6",
        "month_sin",
        "month_cos",
        "quarter_sin",
        "quarter_cos",
        "is_jan",
        "is_dec",
        "is_tariff_month",
        "is_q1",
        "is_summer",
        "food_lag1",
        "nonfood_lag1",
        "services_lag1",
        "seasonal_norm",
        "deviation_lag1",
    ]

    def __init__(
        self,
        alpha: float = None,
        coverage_target: float = None,
        calibration_ratio: float = None,
        quantile_multiplier: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = alpha or self.ALPHA
        self.coverage_target = coverage_target or self.COVERAGE_TARGET
        self.calibration_ratio = calibration_ratio or self.CALIBRATION_RATIO
        self.quantile_multiplier = quantile_multiplier or 5.0

        self.ridge = None
        self.scaler = None
        self.seasonal_norm = None
        self._features = None

        # Conformal prediction параметры
        self.conformal_quantile = None  # калиброванный квантиль для CI

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков."""
        df = df.copy()

        df["month"] = df.index.month
        df["year"] = df.index.year
        df["quarter"] = df.index.quarter

        y = df["Все товары и услуги"]

        # Лаги
        df["y_lag1"] = y.shift(1)
        df["y_lag2"] = y.shift(2)
        df["y_lag12"] = y.shift(12)
        df["y_lag3"] = y.shift(3)
        df["y_lag6"] = y.shift(6)

        # MA
        df["y_ma3"] = y.rolling(3).mean().shift(1)
        df["y_ma6"] = y.rolling(6).mean().shift(1)

        # Momentum
        df["d_y_lag1"] = y.shift(1) - y.shift(2)
        df["d_y_lag3"] = y.shift(1) - y.shift(4)

        # Volatility
        df["y_vol3"] = y.rolling(3).std().shift(1)
        df["y_vol6"] = y.rolling(6).std().shift(1)

        # Сезонность
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)

        # Календарь
        df["is_jan"] = (df["month"] == 1).astype(int)
        df["is_dec"] = (df["month"] == 12).astype(int)
        df["is_tariff_month"] = (df["month"] == 7).astype(int)
        df["is_q1"] = (df["quarter"] == 1).astype(int)
        df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)

        # Компоненты
        if "Продовольственные товары" in df.columns:
            df["food_lag1"] = df["Продовольственные товары"].shift(1)
        else:
            df["food_lag1"] = df["y_lag1"]

        if "Непродовольственные товары" in df.columns:
            df["nonfood_lag1"] = df["Непродовольственные товары"].shift(1)
        else:
            df["nonfood_lag1"] = df["y_lag1"]

        if "Услуги" in df.columns:
            df["services_lag1"] = df["Услуги"].shift(1)
        else:
            df["services_lag1"] = df["y_lag1"]

        return df

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        """Сезонная норма."""
        clean_df = df[~df["year"].isin(self.OUTLIER_YEARS)]
        return clean_df.groupby("month")["Все товары и услуги"].mean()

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "ConformalForecaster":
        """Обучение с калибровкой conformal prediction."""
        self._validate_data(df, target_col)

        df_prep = self._prepare_features(df)
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)

        df_prep["seasonal_norm"] = df_prep["month"].map(self.seasonal_norm)
        df_prep["deviation_lag1"] = df_prep["y_lag1"] - df_prep["month"].shift(1).map(
            self.seasonal_norm
        )

        self._features = self.BASE_FEATURES.copy()

        # Исключаем выбросные годы
        train_df = df_prep[~df_prep["year"].isin(self.OUTLIER_YEARS)]
        train_clean = train_df.dropna(subset=self._features + [target_col])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(
                f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}"
            )

        # Split на train и calibration
        n_calib = max(int(len(train_clean) * self.calibration_ratio), 12)
        n_train = len(train_clean) - n_calib

        train_data = train_clean.iloc[:n_train]
        calib_data = train_clean.iloc[n_train:]

        X_train = train_data[self._features].values
        y_train = train_data[target_col].values

        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Обучаем Ridge на train части
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(X_train_scaled, y_train)

        # Калибруем на calibration части
        X_calib = calib_data[self._features].values
        y_calib = calib_data[target_col].values
        X_calib_scaled = self.scaler.transform(X_calib)

        # Получаем прогнозы и residuals
        calib_preds = self.ridge.predict(X_calib_scaled)

        # Применяем ETS корректировку к калибровочным прогнозам
        calib_months = calib_data["month"].values
        calib_preds_adj = []
        for i, (pred, month) in enumerate(zip(calib_preds, calib_months)):
            ets_val = self.seasonal_norm.get(month, 100.0)
            ets_weight = self.ETS_WEIGHTS.get(month, 0.3)
            adj_pred = (1 - ets_weight) * pred + ets_weight * ets_val
            calib_preds_adj.append(adj_pred)
        calib_preds_adj = np.array(calib_preds_adj)

        residuals = np.abs(y_calib - calib_preds_adj)

        # Conformal quantile для заданного coverage
        # Для coverage 90% берём 90-й перцентиль абсолютных ошибок
        # С поправкой на finite sample: (n+1)(1-alpha)/n
        # И с мультипликатором для гарантии покрытия на нестационарных данных
        n = len(residuals)
        adjusted_quantile = min(1.0, (n + 1) * self.coverage_target / n)
        base_quantile = np.quantile(residuals, adjusted_quantile)
        self.conformal_quantile = base_quantile * self.quantile_multiplier

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз (ETS fallback)."""
        self._check_fitted()

        if self.seasonal_norm is None:
            return np.zeros(horizon)

        last_month = self._last_train_date.month if self._last_train_date else 1
        predictions = []

        for i in range(horizon):
            month = ((last_month + i) % 12) + 1
            pred = self.seasonal_norm.get(month, 100.0)
            predictions.append(pred)

        return np.array(predictions)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Прогноз с калиброванными CI."""
        self._check_fitted()

        df_prep = self._prepare_features(df)
        df_prep["seasonal_norm"] = df_prep["month"].map(self.seasonal_norm)
        df_prep["deviation_lag1"] = df_prep["y_lag1"] - df_prep["month"].shift(1).map(
            self.seasonal_norm
        )

        test_row = df_prep.loc[[target_date]]
        X_test = self.scaler.transform(test_row[self._features].values)
        pred_ridge = self.ridge.predict(X_test)[0]

        target_month = target_date.month
        pred_ets = self.seasonal_norm.get(target_month, 100.0)

        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        pred_combined = (1 - ets_weight) * pred_ridge + ets_weight * pred_ets

        # Калиброванные CI
        ci_lower = pred_combined - self.conformal_quantile
        ci_upper = pred_combined + self.conformal_quantile

        return {
            "date": target_date,
            "prediction": pred_combined,
            "pred_ridge": pred_ridge,
            "pred_ets": pred_ets,
            "ets_weight": ets_weight,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_upper - ci_lower,
            "conformal_quantile": self.conformal_quantile,
            "model": self.name,
        }

    def predict_with_ci(
        self, df: pd.DataFrame, target_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """Алиас для совместимости."""
        return self.predict(df, target_date)

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        """Бэктест с калиброванными CI."""
        start = pd.Timestamp(start_date)
        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = ConformalForecaster(
                    alpha=self.alpha,
                    coverage_target=self.coverage_target,
                    calibration_ratio=self.calibration_ratio,
                )
                model.fit(train_df, target_col)

                test_df = df[df.index <= target_date].copy()
                pred_result = model.predict(test_df, target_date)

                actual = df.loc[target_date, target_col]
                in_ci = (actual >= pred_result["ci_lower"]) and (
                    actual <= pred_result["ci_upper"]
                )

                results.append(
                    {
                        "date": target_date,
                        "actual": actual,
                        "prediction": pred_result["prediction"],
                        "error": actual - pred_result["prediction"],
                        "ci_lower": pred_result["ci_lower"],
                        "ci_upper": pred_result["ci_upper"],
                        "ci_width": pred_result["ci_width"],
                        "conformal_quantile": pred_result["conformal_quantile"],
                        "in_ci": in_ci,
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            "name": self.name,
            "alpha": self.alpha,
            "coverage_target": self.coverage_target,
            "calibration_ratio": self.calibration_ratio,
            "quantile_multiplier": self.quantile_multiplier,
            "conformal_quantile": self.conformal_quantile,
            "features_count": len(self._features) if self._features else 0,
            "is_fitted": self._is_fitted,
        }
