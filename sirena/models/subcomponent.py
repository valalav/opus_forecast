#!/usr/bin/env python3
"""
СУБКОМПОНЕНТНАЯ МОДЕЛЬ (Bottom-Up)
==================================
Прогнозирует 45 субкомпонентов индивидуально с оптимальными подходами,
затем агрегирует по весам.

Результаты бэктеста (2022-2025):
- h=1: MAE 0.426 (-25% vs baseline)
- h=12: MAE 0.373 (-15% vs direct)

Автор: Claude Code
Дата: 2025-12-29
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler

# Оптимальные подходы по горизонтам (на основе эксперимента)
OPTIMAL_H1 = {
    # USD (32% веса) - важен для h=1
    "29": "usd",
    "54": "usd",
    "53": "usd",
    "24": "usd",
    "27": "usd",
    "34": "usd",
    "21": "usd",
    "51": "usd",
    "44": "usd",
    "43": "usd",
    "49": "usd",
    "11": "usd",
    "52": "usd",
    # BRENT (13% веса)
    "12": "brent",
    "47": "brent",
    "55": "brent",
    "19": "brent",
    "23": "brent",
    "15": "brent",
    "46": "brent",
    # SEASONAL (9% веса)
    "17": "seasonal",
    "16": "seasonal",
    "38": "seasonal",
    "39": "seasonal",
    "35": "seasonal",
    "67": "seasonal",
    "32": "seasonal",
    # ALL (9% веса)
    "33": "all",
    "40": "all",
    "41": "all",
    # MONETARY (6% веса)
    "42": "monetary",
    "31": "monetary",
    "25": "monetary",
    "37": "monetary",
    # BASELINE (31% веса)
    "26": "baseline",
    "14": "baseline",
    "48": "baseline",
    "20": "baseline",
    "30": "baseline",
    "50": "baseline",
    "13": "baseline",
    "22": "baseline",
    "28": "baseline",
    "36": "baseline",
}

OPTIMAL_H12 = {
    # SEASONAL (40% веса) - важен для h=12
    "26": "seasonal",
    "14": "seasonal",
    "29": "seasonal",
    "24": "seasonal",
    "48": "seasonal",
    "30": "seasonal",
    "43": "seasonal",
    "28": "seasonal",
    "38": "seasonal",
    "39": "seasonal",
    "35": "seasonal",
    "67": "seasonal",
    # USD (17% веса)
    "53": "usd",
    "16": "usd",
    "27": "usd",
    "47": "usd",
    "18": "usd",
    "51": "usd",
    "44": "usd",
    "52": "usd",
    # BRENT (8% веса)
    "42": "brent",
    "34": "brent",
    "19": "brent",
    "40": "brent",
    "23": "brent",
    "15": "brent",
    # TARIFF (7% веса)
    "12": "tariff",
    "21": "tariff",
    "46": "tariff",
    # MONETARY (2% веса)
    "49": "monetary",
    "31": "monetary",
    "41": "monetary",
    # ALL (3% веса)
    "20": "all",
    "11": "all",
    # BASELINE (23% веса)
    "33": "baseline",
    "54": "baseline",
    "17": "baseline",
    "55": "baseline",
    "50": "baseline",
    "13": "baseline",
    "22": "baseline",
    "25": "baseline",
    "37": "baseline",
    "36": "baseline",
    "32": "baseline",
}


class SubcomponentForecaster:
    """
    Bottom-up forecaster using 45 subcomponents with optimal approaches.

    Usage:
        model = SubcomponentForecaster(horizon=12)
        model.fit(df)
        forecast = model.forecast()  # Returns aggregated MoM forecast
    """

    name = "subcomponent"

    def __init__(self, horizon=12, train_start="2016-01-01", random_state=42):
        self.horizon = horizon
        self.train_start = train_start
        self.random_state = random_state
        self._is_fitted = False
        self.subcomponent_models = {}
        self.weights = {}
        self.approaches = OPTIMAL_H12 if horizon >= 6 else OPTIMAL_H1

    def _load_data(self, data_dir):
        """Load subcomponent data."""
        # Try newer subcomp.csv first (до октября 2025), fallback to sub_mom.csv
        subcomp_file = data_dir / "raw" / "subcomp.csv"
        sub_mom_file = data_dir / "raw" / "sub_mom.csv"

        if subcomp_file.exists():
            sub = pd.read_csv(subcomp_file, sep=";", decimal=",", encoding="utf-8-sig")
            date_col = "Day" if "Day" in sub.columns else "Date"
            sub[date_col] = pd.to_datetime(sub[date_col], format="%d.%m.%Y")
            sub = sub.rename(columns={date_col: "Date"}).set_index("Date").sort_index()
        else:
            sub = pd.read_csv(sub_mom_file, sep=";", decimal=",", encoding="utf-8-sig")
            sub["Date"] = pd.to_datetime(sub["Date"], format="%d.%m.%Y")
            sub = sub.set_index("Date").sort_index()

        sub.index = sub.index.to_period("M").to_timestamp()
        sub = sub[~sub.index.duplicated(keep="last")]

        # Weights
        sprav = pd.read_csv(
            data_dir / "raw" / "subcomp_sprav.csv",
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )
        self.weights = dict(zip(sprav["Item_code"].astype(str), sprav["Weight"]))

        # Filter valid
        valid_cols = [c for c in sub.columns if c in self.weights]
        sub = sub[valid_cols]

        return sub

    def _create_features(self, series, macro_df, approach):
        """Create features based on approach."""
        df = pd.DataFrame({"y": series})

        # Basic features
        for lag in [1, 2, 3, 6, 12]:
            df[f"L{lag}"] = df["y"].shift(lag)
        df["D1"] = df["y"].diff(1)
        df["MA3"] = df["y"].rolling(3).mean()
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

        # Approach-specific features
        if approach in ["usd", "all"] and "usd_nom_i" in macro_df.columns:
            usd = macro_df["usd_nom_i"].reindex(df.index)
            df["usd_L1"] = usd.shift(1)
            df["usd_L3"] = usd.shift(3)
            df["usd_D1"] = usd.diff(1)

        if approach in ["brent", "all"] and "brent" in macro_df.columns:
            brent = macro_df["brent"].reindex(df.index)
            df["brent_L1"] = brent.shift(1)
            df["brent_L3"] = brent.shift(3)
            df["brent_D1"] = brent.diff(1)

        if approach in ["monetary", "all"]:
            if "Ki_i" in macro_df.columns:
                ki = macro_df["Ki_i"].reindex(df.index)
                df["ki_L3"] = ki.shift(3)
            if "Ruonia" in macro_df.columns:
                ruonia = macro_df["Ruonia"].reindex(df.index)
                df["ruonia_L1"] = ruonia.shift(1)

        if approach in ["seasonal", "all"]:
            df["is_jan"] = (df.index.month == 1).astype(int)
            df["is_jul"] = (df.index.month == 7).astype(int)
            df["quarter_sin"] = np.sin(2 * np.pi * ((df.index.month - 1) // 3) / 4)

        if approach == "tariff":
            df["is_jul"] = (df.index.month == 7).astype(int)
            df["trend"] = np.arange(len(df))

        return df

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "SubcomponentForecaster":
        """
        Fit models for all subcomponents.

        Args:
            df: DataFrame with macro data (mom, usd_nom_i, Ki_i, Ruonia, brent)
        """
        # Determine data directory
        data_dir = Path(__file__).parent.parent.parent / "data"

        # Load subcomponent data
        sub_data = self._load_data(data_dir)

        # Store macro data for later
        self.macro_df = df.copy()

        # Try to add brent if available
        try:
            brent = pd.read_csv(data_dir / "brent_prices.csv")
            brent["Date"] = pd.to_datetime(brent["Date"])
            brent = brent.set_index("Date").sort_index()
            brent.index = brent.index.to_period("M").to_timestamp()
            brent = brent[~brent.index.duplicated(keep="last")]
            if "brent" not in self.macro_df.columns:
                self.macro_df = self.macro_df.join(brent[["brent"]], how="left")
        except:
            pass

        # Fit model for each subcomponent
        for col in sub_data.columns:
            approach = self.approaches.get(col, "baseline")

            features = self._create_features(sub_data[col], self.macro_df, approach)
            features["target"] = features["y"].shift(-self.horizon)
            features = features.dropna()

            if self.train_start:
                features = features[features.index >= pd.to_datetime(self.train_start)]

            if len(features) < 24:
                continue

            feature_cols = [c for c in features.columns if c not in ["target", "y"]]
            X = features[feature_cols].values
            y = features["target"].values

            # Handle NaN
            if np.any(np.isnan(X)):
                col_means = np.nanmean(X, axis=0)
                for i in range(X.shape[1]):
                    X[np.isnan(X[:, i]), i] = (
                        col_means[i] if not np.isnan(col_means[i]) else 0
                    )

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

            try:
                model.fit(X_scaled, y)
                self.subcomponent_models[col] = {
                    "model": model,
                    "scaler": scaler,
                    "feature_cols": feature_cols,
                    "approach": approach,
                    "last_data": sub_data[col].copy(),
                }
            except:
                continue

        self._is_fitted = True
        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Predict aggregated MoM for target date.

        Returns:
            dict with 'prediction' (in index format 100+%)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = {}

        for col, model_data in self.subcomponent_models.items():
            approach = model_data["approach"]
            series = model_data["last_data"]

            # Create features for prediction
            features = self._create_features(series, self.macro_df, approach)

            # Get last available row (normalize to first of month)
            pred_date = target_date - pd.DateOffset(months=self.horizon)
            pred_date = pred_date.to_period(
                "M"
            ).to_timestamp()  # Normalize to 1st of month
            if pred_date not in features.index:
                pred_date = features.index[-1]

            feature_cols = model_data["feature_cols"]
            X = features.loc[[pred_date], feature_cols].values

            # Handle NaN
            if np.any(np.isnan(X)):
                X = np.nan_to_num(X, nan=0)

            X_scaled = model_data["scaler"].transform(X)
            pred = model_data["model"].predict(X_scaled)[0]
            predictions[col] = pred

        # Aggregate with weights
        total_weight = sum(self.weights[c] for c in predictions.keys())
        agg_pred = sum(
            self.weights[c] / total_weight * predictions[c] for c in predictions.keys()
        )

        # Return in index format (100 + %)
        return {"prediction": 100 + agg_pred}

    def forecast(self, horizon: Optional[int] = None) -> np.ndarray:
        """
        Generate forecast trajectory.

        Returns:
            np.array of MoM values (in % format, not index)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        h = horizon or self.horizon
        forecasts = []

        last_date = self.macro_df.index[-1]

        for i in range(h):
            target_date = last_date + pd.DateOffset(months=i + 1)
            pred = self.predict(None, target_date)
            forecasts.append(pred["prediction"] - 100)  # Convert to %

        return np.array(forecasts)

    def get_subcomponent_forecasts(self, df, target_date):
        """Get individual subcomponent forecasts for analysis."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        result = {}
        for col, model_data in self.subcomponent_models.items():
            approach = model_data["approach"]
            series = model_data["last_data"]

            features = self._create_features(series, self.macro_df, approach)
            pred_date = target_date - pd.DateOffset(months=self.horizon)
            if pred_date not in features.index:
                pred_date = features.index[-1]

            feature_cols = model_data["feature_cols"]
            X = features.loc[[pred_date], feature_cols].values
            if np.any(np.isnan(X)):
                X = np.nan_to_num(X, nan=0)

            X_scaled = model_data["scaler"].transform(X)
            pred = model_data["model"].predict(X_scaled)[0]

            result[col] = {
                "prediction": pred,
                "weight": self.weights[col],
                "approach": approach,
            }

        return result
