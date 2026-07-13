#!/usr/bin/env python3
"""
МИКРОКОМПОНЕНТНАЯ МОДЕЛЬ (Bottom-Up, Level 5)
=============================================
Прогнозирование 537 микрокомпонентов с агрегацией по весам.

Архитектура:
- Топ-100 по весу: индивидуальные Ridge модели (56% веса)
- Остальные: VotingRegressor (Ridge + Lasso) как baseline
- Волатильные товары (плодоовощи): расширенные признаки

Результаты:
- Охват: 99.4% весов ИПЦ
- 537 микрокомпонентов
- Агрегация: sum(weight_i * prediction_i) / total_weight
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler


class MicrocomponentForecaster:
    """
    Bottom-up forecaster using 537 microcomponents.

    Each microcomponent is forecasted individually with:
    - Ridge regression for top-100 by weight
    - VotingRegressor (Ridge+Lasso) for others
    - Extended features for volatile items (vegetables)
    - Seasonal adjustment based on historical aggregate patterns
    """

    name = "microcomponent"

    # Volatile items (vegetables) - need extended features
    VOLATILE_ITEMS = {
        435,
        382,
        506,
        279,
        305,
        333,
        342,
        755,
    }  # Огурцы, морковь, помидоры и др.

    # Historical seasonal adjustment (2019-2024, excluding 2022)
    # Calculated from aggregate microcomponent data
    SEASONAL_ADJ = {
        1: +0.30,  # Январь - высокая инфляция после НГ
        2: +0.13,  # Февраль
        3: +0.07,  # Март
        4: +0.14,  # Апрель
        5: -0.31,  # Май - начало сезона низкой инфляции
        6: -0.55,  # Июнь - низкая (свежие овощи)
        7: -0.20,  # Июль - тарифы ЖКХ, но овощи дешевеют
        8: -0.59,  # Август - минимум инфляции
        9: +0.36,  # Сентябрь - рост (конец сезона)
        10: +0.19,  # Октябрь
        11: +0.43,  # Ноябрь - высокая инфляция
        12: +0.12,  # Декабрь
    }

    def __init__(
        self,
        horizon=1,
        train_start="2016-01-01",
        random_state=42,
        top_n=100,
        use_extended_for_volatile=True,
        use_seasonal_adj=True,
    ):
        """
        Parameters
        ----------
        horizon : int
            Forecast horizon (1, 2, or 12)
        train_start : str
            Start date for training data
        random_state : int
            Random seed
        top_n : int
            Number of top microcomponents by weight to use individual Ridge models
        use_extended_for_volatile : bool
            Use extended features for volatile items
        use_seasonal_adj : bool
            Apply seasonal adjustment to aggregate forecast (default True)
        """
        self.horizon = horizon
        self.train_start = train_start
        self.random_state = random_state
        self.top_n = top_n
        self.use_extended_for_volatile = use_extended_for_volatile
        self.use_seasonal_adj = use_seasonal_adj

        self._is_fitted = False
        self.micro_models = {}  # {item_code: {'model': model, 'scaler': scaler, ...}}
        self.weights = {}
        self.top_items = set()

    def _load_data(self, data_dir):
        """Load microcomponent data and справочник."""
        # Historical MoM data
        micro_df = pd.read_csv(data_dir / "kbr_micro_full.csv", sep=",", decimal=".")
        # Day column has format MM/DD/YY HH:MM:SS
        micro_df["DateParsed"] = pd.to_datetime(
            micro_df["Day"].str.split(" ").str[0], format="%m/%d/%y", errors="coerce"
        )
        micro_df["Period"] = micro_df["DateParsed"].dt.to_period("M").dt.to_timestamp()
        micro_pivot = micro_df.pivot_table(
            index="Period", columns="Item_code", values="MoM", aggfunc="first"
        )
        micro_pivot = micro_pivot[~micro_pivot.index.duplicated(keep="last")]

        # Справочник with weights
        sprav = pd.read_csv(
            data_dir / "raw" / "micro_sprav.csv",
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )
        self.weights = dict(zip(sprav["Item_code"], sprav["Weight"]))
        self.item_names = dict(zip(sprav["Item_code"], sprav["Товар"]))
        self.item_subcomp = dict(zip(sprav["Item_code"], sprav["Субкомпонент"]))

        # Determine top-N by weight
        sorted_items = sorted(self.weights.items(), key=lambda x: -x[1])
        self.top_items = set([item for item, _ in sorted_items[: self.top_n]])

        # Filter to items in справочник with valid data
        valid_cols = [c for c in micro_pivot.columns if c in self.weights]
        micro_pivot = micro_pivot[valid_cols]

        # Convert MoM to changes
        micro_pivot = micro_pivot - 100

        return micro_pivot

    def _create_features(self, series, extended=False):
        """Create features for ML models."""
        df = pd.DataFrame({"y": series})

        # Basic lags
        for lag in [1, 2, 3, 6, 12]:
            df[f"L{lag}"] = df["y"].shift(lag)

        # Momentum
        df["D1"] = df["y"].diff(1)
        df["MA3"] = df["y"].rolling(3).mean()

        # Seasonality
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

        if extended:
            # Extended features for volatile items
            df["MA6"] = df["y"].rolling(6).mean()
            df["STD3"] = df["y"].rolling(3).std()
            df["STD6"] = df["y"].rolling(6).std()
            df["MAX3"] = df["y"].rolling(3).max()
            df["MIN3"] = df["y"].rolling(3).min()
            df["RANGE3"] = df["MAX3"] - df["MIN3"]

        return df

    def _fit_ridge(self, series, item_code):
        """Fit Ridge model for top items."""
        extended = self.use_extended_for_volatile and item_code in self.VOLATILE_ITEMS
        df = self._create_features(series, extended=extended)
        df["target"] = df["y"].shift(-self.horizon)
        df = df.dropna()

        if self.train_start:
            df = df[df.index >= pd.to_datetime(self.train_start)]

        if len(df) < 24:
            return None

        feature_cols = [c for c in df.columns if c not in ["target", "y"]]
        X = df[feature_cols].values
        y = df["target"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Higher regularization for volatile items
        alpha = 200.0 if item_code in self.VOLATILE_ITEMS else 100.0
        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_scaled, y)

        return {
            "type": "ridge",
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "extended": extended,
            "last_data": series.copy(),
        }

    def _fit_voting(self, series, item_code):
        """Fit VotingRegressor for other items."""
        df = self._create_features(series, extended=False)
        df["target"] = df["y"].shift(-self.horizon)
        df = df.dropna()

        if self.train_start:
            df = df[df.index >= pd.to_datetime(self.train_start)]

        if len(df) < 24:
            return None

        feature_cols = [c for c in df.columns if c not in ["target", "y"]]
        X = df[feature_cols].values
        y = df["target"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = VotingRegressor(
            [
                ("ridge", Ridge(alpha=100.0, random_state=self.random_state)),
                (
                    "lasso",
                    Lasso(alpha=0.1, random_state=self.random_state, max_iter=5000),
                ),
            ]
        )
        model.fit(X_scaled, y)

        return {
            "type": "voting",
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "extended": False,
            "last_data": series.copy(),
        }

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "MicrocomponentForecaster":
        """
        Fit models for all microcomponents.

        Parameters
        ----------
        df : pd.DataFrame
            Main inflation DataFrame (used for macro context)
        target_col : str
            Target column (ignored, using micro data)
        """
        data_dir = Path(__file__).parent.parent.parent / "data"
        micro_data = self._load_data(data_dir)
        self.macro_df = df.copy()

        fitted_count = 0
        top_count = 0

        for item_code in micro_data.columns:
            series = micro_data[item_code].dropna()

            if len(series) < 36:  # Need at least 3 years
                continue

            # Use Ridge for top items, Voting for others
            if item_code in self.top_items:
                result = self._fit_ridge(series, item_code)
                if result:
                    top_count += 1
            else:
                result = self._fit_voting(series, item_code)

            if result:
                self.micro_models[item_code] = result
                fitted_count += 1

        self._is_fitted = True
        print(
            f"MicrocomponentForecaster: fitted {fitted_count} models "
            f"({top_count} top Ridge, {fitted_count - top_count} Voting)"
        )

        return self

    def _predict_single(self, model_data, target_date):
        """Predict for a single microcomponent."""
        series = model_data["last_data"]
        extended = model_data["extended"]
        df = self._create_features(series, extended=extended)

        # Get last valid row for prediction
        pred_date = target_date - pd.DateOffset(months=self.horizon)
        if pred_date not in df.index:
            pred_date = df.index[-1]

        feature_cols = model_data["feature_cols"]
        X = df.loc[[pred_date], feature_cols].values

        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=0)

        X_scaled = model_data["scaler"].transform(X)
        return model_data["model"].predict(X_scaled)[0]

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Predict aggregated MoM for target date.

        Returns
        -------
        dict
            {'prediction': value} where value is MoM index (e.g., 100.5)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        predictions = {}

        for item_code, model_data in self.micro_models.items():
            try:
                pred = self._predict_single(model_data, target_date)
                predictions[item_code] = pred
            except Exception:
                continue

        if not predictions:
            return {"prediction": 100.0}

        # Weighted aggregation
        total_weight = sum(self.weights.get(c, 0) for c in predictions.keys())
        if total_weight == 0:
            return {"prediction": 100.0}

        agg_pred = sum(
            self.weights.get(c, 0) / total_weight * predictions[c]
            for c in predictions.keys()
        )

        return {"prediction": 100 + agg_pred}

    def forecast(self, horizon: Optional[int] = None) -> np.ndarray:
        """
        Generate forecast trajectory using iterative prediction.

        Each step updates the feature data with previous predictions
        to create a dynamic trajectory.

        Returns
        -------
        np.array
            Array of MoM changes (e.g., [0.3, 0.4, ...])
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        h = horizon or self.horizon
        forecasts = []
        last_date = self.macro_df.index[-1]

        # Create copies of model data for iterative updates
        updated_data = {}
        for item_code, model_data in self.micro_models.items():
            updated_data[item_code] = model_data["last_data"].copy()

        for i in range(h):
            target_date = last_date + pd.DateOffset(months=i + 1)
            predictions = {}

            for item_code, model_data in self.micro_models.items():
                try:
                    # Use updated series with previous predictions
                    series = updated_data[item_code]
                    extended = model_data["extended"]
                    df = self._create_features(series, extended=extended)

                    # Get last available row
                    pred_row = df.iloc[[-1]]
                    feature_cols = model_data["feature_cols"]
                    X = pred_row[feature_cols].values

                    if np.any(np.isnan(X)):
                        X = np.nan_to_num(X, nan=0)

                    X_scaled = model_data["scaler"].transform(X)
                    pred = model_data["model"].predict(X_scaled)[0]
                    predictions[item_code] = pred

                    # Update series with prediction for next iteration
                    updated_data[item_code] = pd.concat(
                        [series, pd.Series([pred], index=[target_date])]
                    )
                except Exception:
                    continue

            if not predictions:
                forecasts.append(0.0)
                continue

            # Weighted aggregation
            total_weight = sum(self.weights.get(c, 0) for c in predictions.keys())
            if total_weight == 0:
                forecasts.append(0.0)
                continue

            agg_pred = sum(
                self.weights.get(c, 0) / total_weight * predictions[c]
                for c in predictions.keys()
            )

            # Apply seasonal adjustment
            if self.use_seasonal_adj:
                month = target_date.month
                seasonal_adj = self.SEASONAL_ADJ.get(month, 0)
                agg_pred += seasonal_adj

            forecasts.append(agg_pred)

        return np.array(forecasts)

    def get_stats(self):
        """Get model statistics."""
        if not self._is_fitted:
            return {}

        stats = {
            "total_models": len(self.micro_models),
            "top_models": sum(
                1 for m in self.micro_models.values() if m["type"] == "ridge"
            ),
            "voting_models": sum(
                1 for m in self.micro_models.values() if m["type"] == "voting"
            ),
            "volatile_models": sum(
                1
                for k, m in self.micro_models.items()
                if k in self.VOLATILE_ITEMS and m["extended"]
            ),
            "total_weight": sum(
                self.weights.get(k, 0) for k in self.micro_models.keys()
            )
            * 100,
        }
        return stats

    def get_top_predictions(self, target_date, n=20):
        """Get predictions for top-N microcomponents by weight."""
        if not self._is_fitted:
            return pd.DataFrame()

        predictions = []
        for item_code in self.top_items:
            if item_code not in self.micro_models:
                continue

            try:
                pred = self._predict_single(self.micro_models[item_code], target_date)
                predictions.append(
                    {
                        "Item_code": item_code,
                        "Name": self.item_names.get(item_code, str(item_code))[:40],
                        "Weight": self.weights.get(item_code, 0) * 100,
                        "Prediction": pred,
                    }
                )
            except Exception:
                continue

        df = pd.DataFrame(predictions)
        return df.sort_values("Weight", ascending=False).head(n)
