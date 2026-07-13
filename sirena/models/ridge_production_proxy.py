"""
Ridge Production Proxy
======================

Производная модель поверх RidgeShockDummies:
- сохраняет baseline ridge + shock dummies;
- добавляет локальные demand/services proxy features из `infostat.csv`;
- не изменяет исходный `ridge_shock_dummies.py`.
"""

from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd

from sirena.macro_features import PRODUCTION_FEATURES, add_production_features

from .registry import ModelRegistry
from .ridge_shock_dummies import RidgeShockDummiesForecaster


@ModelRegistry.register("ridge_production_proxy")
class RidgeProductionProxyForecaster(RidgeShockDummiesForecaster):
    """RidgeShockDummies с признаками спроса/услуг из infostat."""

    name = "ridge_production_proxy"
    BASE_FEATURES = [
        "y_lag1",
        "y_lag2",
        "y_lag12",
        "y_ma3",
        "month_sin",
        "month_cos",
        "food_lag1",
        "nonfood_lag1",
        "services_lag1",
        "seasonal_norm",
        "deviation_lag1",
    ]
    SHOCK_DUMMIES = [
        "is_shock_dec2014",
        "is_shock_jan2015",
        "is_tariff_jul2017",
        "is_shock_mar2022",
        "is_shock_apr2022",
        "is_shock_2022",
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

    def __init__(
        self,
        alpha: Optional[float] = None,
        use_macro: bool = True,
        use_2022_dummy: bool = True,
        data_dir: str = "data/raw",
        **kwargs,
    ):
        super().__init__(
            alpha=alpha,
            use_macro=use_macro,
            use_2022_dummy=use_2022_dummy,
            **kwargs,
        )
        self.alpha: float = self.alpha
        self.use_macro: bool = self.use_macro
        self.use_2022_dummy: bool = self.use_2022_dummy
        self.data_dir = data_dir

    def _add_shock_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        idx_series = pd.Series(pd.to_datetime(df.index), index=df.index)
        years = idx_series.dt.year
        months = idx_series.dt.month
        df["is_shock_dec2014"] = (
            (years == 2014) & (months == 12)
        ).astype(int)
        df["is_shock_jan2015"] = (
            (years == 2015) & (months == 1)
        ).astype(int)
        df["is_tariff_jul2017"] = (months == 7).astype(int)
        df["is_shock_mar2022"] = (
            (years == 2022) & (months == 3)
        ).astype(int)
        df["is_shock_apr2022"] = (
            (years == 2022) & (months == 4)
        ).astype(int)
        df["is_shock_2022"] = (years == 2022).astype(int)
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        idx_series = pd.Series(pd.to_datetime(df.index), index=df.index)
        month_series = pd.Series(idx_series.dt.month, index=df.index, dtype=float)
        df["month"] = month_series.astype(int)
        df["year"] = pd.Series(idx_series.dt.year, index=df.index, dtype=int)
        target_series = cast(pd.Series, df["Все товары и услуги"]).astype(float)
        target_series = pd.Series(target_series.to_numpy(), index=df.index, dtype=float)
        df["y_lag1"] = target_series.shift(1)
        df["y_lag2"] = target_series.shift(2)
        df["y_lag12"] = target_series.shift(12)
        rolling_target = pd.DataFrame({"target": target_series}, index=df.index).rolling(3).mean()
        df["y_ma3"] = cast(pd.Series, rolling_target["target"]).shift(1)
        df["month_sin"] = pd.Series(np.sin(2 * np.pi * month_series / 12), index=df.index)
        df["month_cos"] = pd.Series(np.cos(2 * np.pi * month_series / 12), index=df.index)

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

        return self._add_shock_dummies(df)

    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "Ki" not in df.columns or "Ruonia" not in df.columns:
            return df

        df["ruonia_diff"] = df["Ruonia"].diff()
        df["ruonia_diff_lag1"] = df["ruonia_diff"].shift(1)
        df["spread"] = df["Ki"] - df["Ruonia"]
        df["spread_lag4"] = df["spread"].shift(4)
        df["ki_diff"] = df["Ki"].diff()
        df["ki_diff_lag6"] = df["ki_diff"].shift(6)
        df["ki_vol"] = df["Ki"].rolling(6).std().shift(1)

        for col in self.MACRO_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        clean_df = df[df["year"] != 2022]
        seasonal_norm = clean_df.groupby("month")["Все товары и услуги"].mean()
        return cast(pd.Series, seasonal_norm)

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "RidgeProductionProxyForecaster":
        """Обучение модели с production proxy features."""
        _ = self._validate_data(df, target_col)

        df_prep = self._prepare_features(df)
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)
        seasonal_map: Dict[int, float] = {
            int(k): float(v) for k, v in self.seasonal_norm.to_dict().items()
        }
        df_prep["seasonal_norm"] = cast(pd.Series, df_prep["month"]).apply(
            lambda m: seasonal_map.get(int(m), float("nan")) if pd.notna(m) else float("nan")
        )
        prev_month = cast(pd.Series, df_prep["month"]).shift(1)
        prev_seasonal = prev_month.apply(
            lambda m: seasonal_map.get(int(m), float("nan")) if pd.notna(m) else float("nan")
        )
        df_prep["deviation_lag1"] = cast(pd.Series, df_prep["y_lag1"]) - cast(
            pd.Series, prev_seasonal
        )

        df_prep = add_production_features(df_prep, data_dir=self.data_dir)

        self._features = self.BASE_FEATURES.copy()

        if self.use_2022_dummy:
            self._features.extend(self.SHOCK_DUMMIES)
        else:
            self._features.extend(
                ["is_shock_dec2014", "is_shock_jan2015", "is_tariff_jul2017"]
            )

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

        if not self.use_2022_dummy:
            train_df = cast(pd.DataFrame, df_prep[df_prep["year"] != 2022].copy())
        else:
            train_df = cast(pd.DataFrame, df_prep.copy())

        train_clean = train_df.dropna(subset=self._features + [target_col])

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
        """Точечный прогноз с production proxy features."""
        self._check_fitted()

        df_prep = self._prepare_features(df)
        seasonal_map: Dict[int, float] = (
            {int(k): float(v) for k, v in self.seasonal_norm.to_dict().items()}
            if self.seasonal_norm is not None
            else {}
        )
        df_prep["seasonal_norm"] = cast(pd.Series, df_prep["month"]).apply(
            lambda m: seasonal_map.get(int(m), float("nan")) if pd.notna(m) else float("nan")
        )
        prev_month = cast(pd.Series, df_prep["month"]).shift(1)
        prev_seasonal = prev_month.apply(
            lambda m: seasonal_map.get(int(m), float("nan")) if pd.notna(m) else float("nan")
        )
        df_prep["deviation_lag1"] = cast(pd.Series, df_prep["y_lag1"]) - cast(
            pd.Series, prev_seasonal
        )
        df_prep = add_production_features(df_prep, data_dir=self.data_dir)

        if self._has_macro:
            df_prep = self._add_macro_features(df_prep)

        test_row = df_prep.loc[[target_date]]
        X_test = self.scaler.transform(test_row[self._features].values)
        pred_ridge = self.ridge.predict(X_test)[0]

        target_month = target_date.month
        raw_pred_ets = self.seasonal_norm.get(target_month, 100.0)
        pred_ets = 100.0 if raw_pred_ets is None else float(raw_pred_ets)
        ets_weight = float(self.ETS_WEIGHTS.get(target_month, 0.3))
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
            "use_2022_dummy": self.use_2022_dummy,
            "production_features": available_production,
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        """Бэктестирование производной модели с production proxy features."""
        start = pd.Timestamp(start_date)

        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            train_df = cast(pd.DataFrame, df[df.index < target_date].copy())

            if len(train_df[[target_col]].dropna()) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = RidgeProductionProxyForecaster(
                    alpha=self.alpha,
                    use_macro=self.use_macro,
                    use_2022_dummy=self.use_2022_dummy,
                    data_dir=self.data_dir,
                )
                model.fit(train_df, target_col)

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
                        "has_macro": pred_result.get("has_macro", False),
                        "use_2022_dummy": pred_result.get("use_2022_dummy", False),
                        "production_features_count": len(
                            pred_result.get("production_features", [])
                        ),
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(results)
