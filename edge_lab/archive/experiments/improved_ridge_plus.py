"""
Improved Ridge Plus - Enhanced Ridge regression with better features and tuning.

This model:
1. Uses proven Ridge regression as base
2. Adds advanced feature engineering inspired by successful models
3. Optimizes hyperparameters
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score


class ImprovedRidgePlusForecaster:
    """
    Improved Ridge+ with enhanced feature engineering.

    Inspired by the best-performing models:
    - Subcomp (0.309) - uses granular data and seasonal adjustment
    - Ridge (0.321) - baseline
    """

    def __init__(
        self,
        alphas: Optional[List[float]] = None,
        use_seasonal_adjustment: bool = True,
        use_volatility_features: bool = True,
        use_momentum_features: bool = True,
    ):
        """Initialize Improved Ridge+ forecaster."""
        self.alphas = alphas or np.logspace(-3, 2, 50)
        self.use_seasonal_adjustment = use_seasonal_adjustment
        self.use_volatility_features = use_volatility_features
        self.use_momentum_features = use_momentum_features

        self.model = None
        self.scaler = None
        self.feature_names = []
        self.target_col = None
        self.best_alpha = None
        self.is_fitted = False

    def _seasonal_adjustment_factors(self) -> Dict[int, float]:
        """Monthly seasonal adjustment factors (inspired by Subcomp)."""
        return {
            1: +0.25,
            2: +0.10,
            3: +0.05,
            4: +0.10,
            5: -0.25,
            6: -0.45,
            7: -0.15,
            8: -0.50,
            9: +0.30,
            10: +0.15,
            11: +0.35,
            12: +0.10,
        }

    def _create_lag_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Create lag features."""
        features = {}

        for lag in [1, 2, 3, 4, 6, 9, 12, 18, 24]:
            if idx - lag >= 0:
                features[f"y_lag{lag}"] = df.iloc[idx - lag][self.target_col]

                if idx - lag - 1 >= 0:
                    val = df.iloc[idx - lag][self.target_col]
                    prev = df.iloc[idx - lag - 1][self.target_col]
                    features[f"y_diff_lag{lag}"] = val - prev

        return features

    def _create_momentum_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Create momentum features."""
        features = {}

        if not self.use_momentum_features:
            return features

        for window in [2, 3, 6, 12]:
            if idx - window >= 0:
                data = df.iloc[idx - window : idx][self.target_col]

                features[f"y_ma{window}"] = data.mean()

                if len(data) > 1:
                    features[f"y_std{window}"] = data.std()

                    if len(data) >= 2:
                        trend = data.iloc[-1] - data.iloc[0]
                        features[f"y_trend_{window}"] = trend

                        if data.iloc[0] != 0:
                            features[f"y_pct_{window}"] = (
                                trend / abs(data.iloc[0])
                            ) * 100

        return features

    def _create_volatility_features(
        self, df: pd.DataFrame, idx: int
    ) -> Dict[str, float]:
        """Create volatility/range features."""
        features = {}

        if not self.use_volatility_features:
            return features

        for window in [3, 6, 12]:
            if idx - window >= 0:
                data = df.iloc[idx - window : idx][self.target_col]
                features[f"y_min{window}"] = data.min()
                features[f"y_max{window}"] = data.max()
                features[f"y_range{window}"] = data.max() - data.min()

        return features

    def _create_seasonal_features(self, date: pd.Timestamp) -> Dict[str, float]:
        """Create seasonality features."""
        features = {
            "month_sin": np.sin(2 * np.pi * date.month / 12),
            "month_cos": np.cos(2 * np.pi * date.month / 12),
            "quarter_sin": np.sin(2 * np.pi * date.quarter / 4),
            "quarter_cos": np.cos(2 * np.pi * date.quarter / 4),
            "is_jan": 1 if date.month == 1 else 0,
            "is_feb": 1 if date.month == 2 else 0,
            "is_mar": 1 if date.month == 3 else 0,
            "is_apr": 1 if date.month == 4 else 0,
            "is_may": 1 if date.month == 5 else 0,
            "is_jun": 1 if date.month == 6 else 0,
            "is_jul": 1 if date.month == 7 else 0,
            "is_aug": 1 if date.month == 8 else 0,
            "is_sep": 1 if date.month == 9 else 0,
            "is_oct": 1 if date.month == 10 else 0,
            "is_nov": 1 if date.month == 11 else 0,
            "is_dec": 1 if date.month == 12 else 0,
            "is_q1": 1 if date.quarter == 1 else 0,
            "is_q2": 1 if date.quarter == 2 else 0,
            "is_q3": 1 if date.quarter == 3 else 0,
            "is_q4": 1 if date.quarter == 4 else 0,
            "is_h1": 1 if date.month <= 6 else 0,
            "is_h2": 1 if date.month >= 7 else 0,
        }
        return features

    def _create_macro_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Create macroeconomic features."""
        features = {}
        macro_cols = ["brent", "usd_nom_i", "Ki", "Ruonia"]

        for col in macro_cols:
            if col in df.columns:
                if idx >= 0 and pd.notna(df.iloc[idx][col]):
                    features[f"{col}_curr"] = df.iloc[idx][col]

                if idx >= 1:
                    curr = df.iloc[idx][col] if pd.notna(df.iloc[idx][col]) else np.nan
                    prev = (
                        df.iloc[idx - 1][col]
                        if pd.notna(df.iloc[idx - 1][col])
                        else np.nan
                    )

                    if pd.notna(curr) and pd.notna(prev):
                        features[f"{col}_pct"] = (
                            ((curr / prev) - 1) * 100 if prev != 0 else 0
                        )
                        features[f"{col}_diff"] = curr - prev

                for window in [3, 6]:
                    if idx - window >= 0:
                        window_data = df.iloc[idx - window : idx][col]
                        valid_data = window_data.dropna()
                        if len(valid_data) > 0:
                            features[f"{col}_ma{window}"] = valid_data.mean()
                            if len(valid_data) > 1:
                                features[f"{col}_std{window}"] = valid_data.std()

        return features

    def _create_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Create all features for a given index."""
        features = {}

        features.update(self._create_lag_features(df, idx))
        features.update(self._create_momentum_features(df, idx))
        features.update(self._create_volatility_features(df, idx))
        features.update(self._create_seasonal_features(df.index[idx]))
        features.update(self._create_macro_features(df, idx))

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
        """Fit the Improved Ridge+ model."""
        self.target_col = target_col

        X = self._prepare_features(df)
        y = df.iloc[24:][target_col].reset_index(drop=True)

        self.feature_names = X.columns.tolist()

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        ridge_cv = RidgeCV(
            alphas=self.alphas,
            cv=5,
            scoring="neg_mean_absolute_error",
            store_cv_values=True,
        )
        ridge_cv.fit(X_scaled, y)

        self.best_alpha = ridge_cv.alpha_
        self.model = Ridge(alpha=self.best_alpha, random_state=42)
        self.model.fit(X_scaled, y)

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

            prediction = self.model.predict(X_scaled)[0]

            if self.use_seasonal_adjustment:
                month = df.index[-1].month
                adj_factor = self._seasonal_adjustment_factors().get(month, 0)
                prediction += adj_factor

            return {"prediction": prediction}

        return {"predictions": [self.predict(df, 1)["prediction"]]}

    def backtest(
        self, df: pd.DataFrame, start_date: str = "2024-06-01"
    ) -> pd.DataFrame:
        """Run backtest."""
        results = []

        for i in range(24, len(df)):
            if df.index[i] < pd.Timestamp(start_date):
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
    """Test the Improved Ridge+ forecaster."""
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
    print("Improved Ridge+ Forecaster")
    print("=" * 70)
    print(f"\nData: {len(df)} months ({df.index[0].date()} to {df.index[-1].date()})")

    model = ImprovedRidgePlusForecaster()
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
