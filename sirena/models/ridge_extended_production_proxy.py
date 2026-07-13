"""
Ridge Extended Production Proxy
===============================

Производная модель поверх RidgeExtended:
- сохраняет расширенный ridge feature pipeline;
- добавляет локальные demand/services proxy features из `infostat.csv`;
- не изменяет исходный `ridge_extended.py`.
"""

from typing import Any, Dict, cast

import pandas as pd

from sirena.macro_features import PRODUCTION_FEATURES, add_production_features

from .registry import ModelRegistry
from .ridge_extended import RidgeExtendedForecaster


@ModelRegistry.register("ridge_extended_production_proxy")
class RidgeExtendedProductionProxyForecaster(RidgeExtendedForecaster):
    """RidgeExtended с локальными demand/services proxy features из infostat."""

    name = "ridge_extended_production_proxy"
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
    MACRO_FEATURES = [
        "ruonia_diff_lag1",
        "spread_lag4",
        "ki_diff_lag6",
        "ki_vol",
    ]
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
    OUTLIER_YEARS = [2010, 2022]

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any):
        super().__init__(**kwargs)
        self.alpha = self.alpha
        self.use_macro = self.use_macro
        self.data_dir = data_dir

    @property
    def FEATURES(self):
        if self._features is not None:
            return self._features
        return self.BASE_FEATURES + PRODUCTION_FEATURES

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return cast(pd.DataFrame, cast(Any, super())._prepare_features(df))

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        return cast(pd.Series, cast(Any, super())._compute_seasonal_norm(df))

    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return cast(pd.DataFrame, cast(Any, super())._add_macro_features(df))

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "RidgeExtendedProductionProxyForecaster":
        _ = self._validate_data(df, target_col)

        df_prep = self._prepare_features(df)
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)
        seasonal_map = cast(pd.Series, self.seasonal_norm).to_dict()

        month_series = cast(pd.Series, df_prep["month"])
        df_prep["seasonal_norm"] = month_series.apply(
            lambda m: seasonal_map.get(m, float("nan")) if pd.notna(m) else float("nan")
        )
        prev_month = month_series.shift(1)
        prev_seasonal = prev_month.apply(
            lambda m: seasonal_map.get(m, float("nan")) if pd.notna(m) else float("nan")
        )
        df_prep["deviation_lag1"] = cast(pd.Series, df_prep["y_lag1"]) - cast(
            pd.Series, prev_seasonal
        )

        df_prep = add_production_features(df_prep, data_dir=self.data_dir)

        self._features = self.BASE_FEATURES.copy()

        self._has_macro = False
        if self.use_macro and "Ki" in df.columns and "Ruonia" in df.columns:
            df_prep = self._add_macro_features(df_prep)
            available_macro = [f for f in self.MACRO_FEATURES if f in df_prep.columns]
            if available_macro:
                self._features.extend(available_macro)
                self._has_macro = True

        available_production = [f for f in PRODUCTION_FEATURES if f in df_prep.columns]
        if available_production:
            self._features.extend(available_production)

        train_df = cast(pd.DataFrame, df_prep[~df_prep["year"].isin(self.OUTLIER_YEARS)])
        subset_cols = list(self._features) + [target_col]
        train_clean = cast(pd.DataFrame, train_df.dropna(subset=subset_cols))

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")

        X = train_clean[self._features].values
        y = train_clean[target_col].values

        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import RobustScaler

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(X_scaled, y)

        self._is_fitted = True
        self._last_train_date = cast(pd.Timestamp, df.index.max())
        self._target_col = target_col
        self._train_df = df.copy()

        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        self._check_fitted()

        df_prep = self._prepare_features(df)
        seasonal_map = cast(pd.Series, self.seasonal_norm).to_dict()
        month_series = cast(pd.Series, df_prep["month"])
        df_prep["seasonal_norm"] = month_series.apply(
            lambda m: seasonal_map.get(m, float("nan")) if pd.notna(m) else float("nan")
        )
        prev_month = month_series.shift(1)
        prev_seasonal = prev_month.apply(
            lambda m: seasonal_map.get(m, float("nan")) if pd.notna(m) else float("nan")
        )
        df_prep["deviation_lag1"] = cast(pd.Series, df_prep["y_lag1"]) - cast(
            pd.Series, prev_seasonal
        )

        if self._has_macro:
            df_prep = self._add_macro_features(df_prep)

        df_prep = add_production_features(df_prep, data_dir=self.data_dir)

        for feature in self._features:
            if feature in df_prep.columns:
                feature_series = cast(pd.Series, df_prep[feature])
                if feature_series.isna().any():
                    fallback = (
                        float(feature_series.dropna().iloc[-1])
                        if not feature_series.dropna().empty
                        else 0.0
                    )
                    df_prep[feature] = feature_series.fillna(fallback)

        test_row = df_prep.loc[[target_date]]
        X_test = self.scaler.transform(test_row[self._features].values)
        pred_ridge = self.ridge.predict(X_test)[0]

        target_month = target_date.month
        pred_ets_raw = cast(pd.Series, self.seasonal_norm).get(target_month, 100.0)
        pred_ets = 100.0 if pred_ets_raw is None else float(pred_ets_raw)

        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        pred_combined = (1 - ets_weight) * pred_ridge + ets_weight * pred_ets

        available_production = [f for f in PRODUCTION_FEATURES if f in df_prep.columns]

        return {
            "date": target_date,
            "prediction": pred_combined,
            "pred_ridge": pred_ridge,
            "pred_ets": pred_ets,
            "ets_weight": ets_weight,
            "model": self.name,
            "has_macro": self._has_macro,
            "production_features": available_production,
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date)

        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()

            target_subset = cast(pd.DataFrame, train_df[[target_col]]).dropna()
            if len(target_subset) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = RidgeExtendedProductionProxyForecaster(
                    alpha=self.alpha,
                    use_macro=self.use_macro,
                    data_dir=self.data_dir,
                )
                model.fit(cast(pd.DataFrame, train_df), target_col)

                test_df = cast(pd.DataFrame, df[df.index <= target_date].copy())
                pred_result = model.predict(test_df, target_date)

                actual = df.loc[target_date, target_col]

                results.append(
                    {
                        "date": target_date,
                        "actual": actual,
                        "prediction": pred_result["prediction"],
                        "error": actual - pred_result["prediction"],
                        "pred_ridge": pred_result["pred_ridge"],
                        "pred_ets": pred_result["pred_ets"],
                        "has_macro": pred_result.get("has_macro", False),
                        "production_features_count": len(
                            pred_result.get("production_features", [])
                        ),
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(results)
