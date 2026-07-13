"""
NGBoost — Natural Gradient Boosting с вероятностными прогнозами
===============================================================

Gradient Boosting с параметрическим распределением:
- Прогнозирует параметры Normal(μ, σ) для каждой точки
- Естественные доверительные интервалы
- Учитывает гетероскедастичность (разная σ в разные периоды)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, Any, List
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry

# Проверяем доступность NGBoost
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal

    NGBOOST_AVAILABLE = True
except ImportError:
    NGBOOST_AVAILABLE = False


@ModelRegistry.register("ngboost")
class NGBoostForecaster(BaseForecaster):
    """
    NGBoost с вероятностными прогнозами.

    Возвращает:
    - mean: среднее прогноза
    - std: стандартное отклонение
    - ci_lower, ci_upper: 90% доверительный интервал
    """

    name = "ngboost"
    MIN_TRAIN_SIZE = 36

    OUTLIER_YEARS = [2010, 2022]

    # Параметры NGBoost (консервативные для малых данных)
    N_ESTIMATORS = 200
    LEARNING_RATE = 0.05
    MINIBATCH_FRAC = 0.8

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
        n_estimators: int = None,
        learning_rate: float = None,
        minibatch_frac: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not NGBOOST_AVAILABLE:
            raise ImportError("NGBoost не установлен. Выполните: pip install ngboost")

        self.n_estimators = n_estimators or self.N_ESTIMATORS
        self.learning_rate = learning_rate or self.LEARNING_RATE
        self.minibatch_frac = minibatch_frac or self.MINIBATCH_FRAC

        self.model = None
        self.scaler = None
        self.seasonal_norm = None
        self._features = None

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
    ) -> "NGBoostForecaster":
        """Обучение NGBoost модели."""
        self._validate_data(df, target_col)

        df_prep = self._prepare_features(df)
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)

        df_prep["seasonal_norm"] = df_prep["month"].map(self.seasonal_norm)
        df_prep["deviation_lag1"] = df_prep["y_lag1"] - df_prep["month"].shift(1).map(
            self.seasonal_norm
        )

        self._features = self.BASE_FEATURES.copy()

        # Исключаем выбросы
        train_df = df_prep[~df_prep["year"].isin(self.OUTLIER_YEARS)]
        train_clean = train_df.dropna(subset=self._features + [target_col])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(
                f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}"
            )

        X = train_clean[self._features].values
        y = train_clean[target_col].values

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Обучаем NGBoost
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = NGBRegressor(
                Dist=Normal,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                minibatch_frac=self.minibatch_frac,
                verbose=False,
            )
            self.model.fit(X_scaled, y)

        self._is_fitted = True
        self._last_train_date = df.index.max()
        self._target_col = target_col
        self._train_df = df.copy()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз на горизонт через итеративный predict()."""
        self._check_fitted()

        # Используем iterative_forecast с сохранёнными данными
        if hasattr(self, "_train_df") and self._train_df is not None:
            target_col = getattr(self, "_target_col", "Все товары и услуги")
            return self.iterative_forecast(self._train_df, horizon, target_col)

        # Fallback на сезонную норму
        if self.seasonal_norm is None:
            return np.zeros(horizon)

        last_month = self._last_train_date.month if self._last_train_date else 1
        predictions = []

        for i in range(horizon):
            month = ((last_month + i) % 12) + 1
            pred = self.seasonal_norm.get(month, 100.0) - 100
            predictions.append(pred)

        return np.array(predictions)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Прогноз с доверительными интервалами."""
        self._check_fitted()

        df_prep = self._prepare_features(df)
        df_prep["seasonal_norm"] = df_prep["month"].map(self.seasonal_norm)
        df_prep["deviation_lag1"] = df_prep["y_lag1"] - df_prep["month"].shift(1).map(
            self.seasonal_norm
        )

        test_row = df_prep.loc[[target_date]]
        X_test = self.scaler.transform(test_row[self._features].values)

        # Получаем распределение прогноза
        dist = self.model.pred_dist(X_test)
        pred_mean = dist.mean()[0]
        pred_std = dist.std()[0]

        # 90% CI
        ci_lower = dist.ppf(0.05)[0]
        ci_upper = dist.ppf(0.95)[0]

        target_month = target_date.month
        pred_ets = self.seasonal_norm.get(target_month, 100.0)

        # Комбинируем с ETS
        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        pred_combined = (1 - ets_weight) * pred_mean + ets_weight * pred_ets

        # CI тоже корректируем
        ci_lower_adj = (1 - ets_weight) * ci_lower + ets_weight * pred_ets
        ci_upper_adj = (1 - ets_weight) * ci_upper + ets_weight * pred_ets

        return {
            "date": target_date,
            "prediction": pred_combined,
            "pred_ngboost": pred_mean,
            "pred_ets": pred_ets,
            "ets_weight": ets_weight,
            "std": pred_std,
            "ci_lower": ci_lower_adj,
            "ci_upper": ci_upper_adj,
            "ci_width": ci_upper_adj - ci_lower_adj,
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
        """Бэктестирование с CI."""
        start = pd.Timestamp(start_date)

        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = NGBoostForecaster(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    minibatch_frac=self.minibatch_frac,
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
                        "std": pred_result["std"],
                        "ci_lower": pred_result["ci_lower"],
                        "ci_upper": pred_result["ci_upper"],
                        "ci_width": pred_result["ci_width"],
                        "in_ci": in_ci,
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Важность признаков."""
        self._check_fitted()

        # NGBoost использует деревья, можно получить feature importance
        # feature_importances_ returns shape (n_dist_params, n_features)
        # Take mean across distribution parameters for overall importance
        feat_imp = self.model.feature_importances_
        if feat_imp.ndim == 2:
            # Average across distribution parameters (mean and std)
            feat_imp = feat_imp.mean(axis=0)

        importance = pd.DataFrame({"feature": self._features, "importance": feat_imp})
        return importance.sort_values("importance", ascending=False)

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            "name": self.name,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "minibatch_frac": self.minibatch_frac,
            "features_count": len(self._features) if self._features else 0,
            "is_fitted": self._is_fitted,
        }
