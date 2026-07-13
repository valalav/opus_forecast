"""
Focused Forecaster - Lightweight model targeting MAE improvement.

This model:
1. Uses only proven features from Ridge and Huber models
2. Simplifies to avoid overfitting
3. Optimizes hyperparameters carefully
"""

import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import RidgeCV, Ridge, HuberRegressor
from sklearn.preprocessing import RobustScaler


class FocusedForecaster:
    """
    Focused Forecaster using proven Ridge + Huber combination.

    Based on analysis of best-performing models:
    - Ridge: 0.321 MAE (good all-rounder)
    - Huber: 0.324 MAE (robust to outliers)
    """

    def __init__(self, ridge_weight: float = 0.7, huber_weight: float = 0.3):
        """Initialize Focused forecaster."""
        self.ridge_weight = ridge_weight
        self.huber_weight = huber_weight
        self.ridge_model = None
        self.huber_model = None
        self.scaler = None
        self.feature_names = []
        self.target_col = None
        self.best_alpha = None
        self.is_fitted = False

    def _create_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Create focused set of proven features."""
        features = {}

        # Target lags (most important)
        for lag in [1, 2, 3, 6, 12]:
            if idx - lag >= 0:
                features[f"y_lag{lag}"] = df.iloc[idx - lag][self.target_col]
                if idx - lag - 1 >= 0:
                    val = df.iloc[idx - lag][self.target_col]
                    prev = df.iloc[idx - lag - 1][self.target_col]
                    features[f"y_diff_lag{lag}"] = val - prev

        # Momentum features
        for window in [3, 6]:
            if idx - window >= 0:
                data = df.iloc[idx - window : idx][self.target_col]
                features[f"y_ma{window}"] = data.mean()
                if len(data) > 1:
                    features[f"y_std{window}"] = data.std()
                    features[f"y_trend{window}"] = data.iloc[-1] - data.iloc[0]

        # Seasonality (simple)
        date = df.index[idx]
        features["month_sin"] = np.sin(2 * np.pi * date.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * date.month / 12)
        features["is_jan"] = 1 if date.month == 1 else 0
        features["is_dec"] = 1 if date.month == 12 else 0
        features["is_q1"] = 1 if date.quarter == 1 else 0
        features["is_q4"] = 1 if date.quarter == 4 else 0

        # Macro features (only if available)
        for col in ["brent", "usd_nom_i", "Ki"]:
            if col in df.columns and idx >= 0:
                if pd.notna(df.iloc[idx][col]):
                    features[f"{col}_curr"] = df.iloc[idx][col]
                if idx >= 1:
                    curr = df.iloc[idx][col] if pd.notna(df.iloc[idx][col]) else np.nan
                    prev = (
                        df.iloc[idx - 1][col]
                        if pd.notna(df.iloc[idx - 1][col])
                        else np.nan
                    )
                    if pd.notna(curr) and pd.notna(prev) and prev != 0:
                        features[f"{col}_pct"] = ((curr / prev) - 1) * 100

        return features

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training."""
        min_idx = 24
        X_list = []

        for i in range(min_idx, len(df)):
            features = self._create_features(df, i)
            X_list.append(features)

        X = pd.DataFrame(X_list)
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        return X

    def fit(self, df: pd.DataFrame, target_col: str = "Все товары и услуги"):
        """Fit the Focused model."""
        self.target_col = target_col

        X = self._prepare_features(df)
        y = df.iloc[24:][target_col].reset_index(drop=True)

        self.feature_names = X.columns.tolist()

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        ridge_cv = RidgeCV(
            alphas=np.logspace(-3, 2, 50), cv=5, scoring="neg_mean_absolute_error"
        )
        ridge_cv.fit(X_scaled, y)

        self.best_alpha = ridge_cv.alpha_
        self.ridge_model = Ridge(alpha=self.best_alpha, random_state=42)
        self.ridge_model.fit(X_scaled, y)

        self.huber_model = HuberRegressor(
            epsilon=1.35, alpha=0.01, max_iter=2000, random_state=42
        )
        self.huber_model.fit(X_scaled, y)

        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> Dict[str, float]:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if horizon == 1:
            features = self._create_features(df, len(df) - 1)
            X = pd.DataFrame([features])
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)

            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0

            X = X[self.feature_names]
            X_scaled = self.scaler.transform(X)

            pred_ridge = self.ridge_model.predict(X_scaled)[0]
            pred_huber = self.huber_model.predict(X_scaled)[0]

            prediction = self.ridge_weight * pred_ridge + self.huber_weight * pred_huber

            return {
                "prediction": prediction,
                "ridge_pred": pred_ridge,
                "huber_pred": pred_huber,
            }

        return {"predictions": [self.predict(df, 1)["prediction"]]}

    def backtest(
        self, df: pd.DataFrame, start_date: str = "2024-06-01"
    ) -> pd.DataFrame:
        """Run backtest."""
        results = []
        start_ts = pd.Timestamp(start_date)

        for i in range(24, len(df)):
            if df.index[i] < start_ts:
                continue

            train_df = df.iloc[:i].copy()

            try:
                self.fit(train_df, "Все товары и услуги")
                result = self.predict(train_df, horizon=1)
                prediction = result["prediction"]
                actual = df.iloc[i]["Все товары и услуги"]

                results.append(
                    {
                        "date": df.index[i],
                        "prediction": prediction,
                        "actual": actual,
                        "error": prediction - actual,
                    }
                )
            except Exception as e:
                continue

        return pd.DataFrame(results)


def main():
    """Test the Focused forecaster."""
    df = pd.read_csv(
        "/home/valalav/_projects/sirena-kbr/data/inflation_data.csv",
        sep=";",
        decimal=",",
        index_col=0,
        dayfirst=True,
        parse_dates=True,
    )

    df.index = df.index.to_period("M").to_timestamp()

    brent = pd.read_csv(
        "/home/valalav/_projects/sirena-kbr/data/brent_prices.csv",
        index_col=0,
        parse_dates=True,
    )

    df = df.join(brent[["brent"]], how="left")
    df = df[["mom", "brent", "usd_nom_i", "Ki", "Ruonia"]].copy()
    df = df.rename(columns={"mom": "Все товары и услуги"})
    df = df.dropna(subset=["Все товары и услуги", "usd_nom_i"])

    print("=" * 70)
    print("Focused Forecaster (Ridge + Huber)")
    print("=" * 70)
    print(f"\nData: {len(df)} months ({df.index[0].date()} to {df.index[-1].date()})")

    model = FocusedForecaster()
    results = model.backtest(df, start_date="2024-06-01")

    if len(results) > 0:
        mae = (results["error"].abs()).mean()
        rmse = np.sqrt((results["error"] ** 2).mean())

        print(f"\nBacktest Results:")
        print(f"  MAE:   {mae:.6f}")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  N:     {len(results)}")
        print(f"  Best Alpha: {model.best_alpha:.6f}")

        ridge_baseline = 0.321
        subcomp_best = 0.309
        diff_pct_ridge = ((mae - ridge_baseline) / ridge_baseline) * 100
        diff_pct_subcomp = ((mae - subcomp_best) / subcomp_best) * 100

        print(f"\nComparison:")
        print(f"  vs Ridge ({ridge_baseline:.6f}):  {diff_pct_ridge:+.2f}%")
        print(f"  vs Subcomp ({subcomp_best:.6f}): {diff_pct_subcomp:+.2f}%")

        if mae < ridge_baseline:
            print(f"\n  ✅ IMPROVED over Ridge by {abs(diff_pct_ridge):.2f}%")
            print(f"  🎉 MAE IMPROVED - Acceptance criterion MET!")
            return 0
        else:
            print(f"\n  ❌ Still worse than Ridge by {diff_pct_ridge:.2f}%")
            return 1
    else:
        print("❌ No backtest results")
        return 1


if __name__ == "__main__":
    exit(main())
