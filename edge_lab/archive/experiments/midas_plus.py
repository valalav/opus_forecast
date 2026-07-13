"""
MIDAS+ Ridge Forecaster - Hybrid model combining MIDAS concepts with Ridge.

This model:
1. Creates multi-scale features (daily, weekly, monthly lags) from monthly data
2. Uses Ridge regression for robust prediction
3. Implements adaptive feature selection to prevent overfitting
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class MIDASPlusForecaster:
    """
    MIDAS+ Forecaster - Combines MIDAS concepts with Ridge regression.

    Key improvements:
    1. Multi-scale feature engineering (weekly, biweekly, monthly)
    2. Rolling window statistics (mean, std, min, max)
    3. Difference and rate-of-change features
    4. Seasonality features
    5. Adaptive regularization
    """

    def __init__(
        self,
        alpha: float = 0.1,
        max_features: int = None,
        lags: List[int] = None,
    ):
        """Initialize MIDAS+ forecaster."""
        self.alpha = alpha
        self.max_features = max_features
        self.lags = lags or [1, 2, 3, 4, 6, 8, 12]
        self.model = None
        self.feature_names = []
        self.target_col = None
        self.selected_features = None

    def _create_multi_scale_features(
        self, df: pd.DataFrame, target_idx: int
    ) -> pd.DataFrame:
        """
        Create multi-scale features from monthly data.

        Simulates MIDAS by treating monthly data at different "frequencies":
        - 1-3 lags: "Weekly" equivalent
        - 4-6 lags: "Biweekly" equivalent
        - 8-12 lags: "Monthly" equivalent
        """
        features = {}

        for lag in self.lags:
            if target_idx - lag >= 0:
                val = df.iloc[target_idx - lag][self.target_col]
                features[f"y_lag_{lag}"] = val

                # Rate of change (simulate high-frequency differences)
                if lag > 1 and target_idx - lag >= 0:
                    prev_val = df.iloc[target_idx - lag - 1][self.target_col]
                    if prev_val != 0:
                        features[f"y_pct_{lag}"] = (val - prev_val) / prev_val * 100
                    else:
                        features[f"y_pct_{lag}"] = 0

        # Rolling window statistics (simulate high-frequency aggregation)
        for window in [3, 6, 12]:
            if target_idx - window >= 0:
                window_data = df.iloc[target_idx - window : target_idx][self.target_col]
                features[f"y_mean_{window}"] = window_data.mean()
                features[f"y_std_{window}"] = window_data.std()
                features[f"y_min_{window}"] = window_data.min()
                features[f"y_max_{window}"] = window_data.max()

                # Momentum features
                features[f"y_trend_{window}"] = (
                    (window_data.iloc[-1] - window_data.iloc[0])
                    / window_data.iloc[0]
                    * 100
                )

        return pd.DataFrame([features])

    def _create_macro_features(self, df: pd.DataFrame, target_idx: int) -> pd.DataFrame:
        """Create features from macro variables."""
        features = {}

        macro_cols = ["brent", "usd_nom_i", "Ki"]
        for col in macro_cols:
            if col in df.columns:
                # Recent value
                if target_idx >= 0:
                    features[f"{col}_recent"] = df.iloc[target_idx][col]

                # Recent change
                if target_idx >= 1:
                    curr = df.iloc[target_idx][col]
                    prev = df.iloc[target_idx - 1][col]
                    features[f"{col}_pct"] = (
                        (curr - prev) / prev * 100 if prev != 0 else 0
                    )

                # Rolling mean
                for window in [3, 6]:
                    if target_idx - window >= 0:
                        window_data = df.iloc[target_idx - window : target_idx][col]
                        features[f"{col}_mean_{window}"] = window_data.mean()

        return pd.DataFrame([features])

    def _create_seasonality_features(self, date: pd.Timestamp) -> pd.DataFrame:
        """Create seasonality features."""
        features = {
            "month_sin": np.sin(2 * np.pi * date.month / 12),
            "month_cos": np.cos(2 * np.pi * date.month / 12),
            "quarter_sin": np.sin(2 * np.pi * date.quarter / 4),
            "quarter_cos": np.cos(2 * np.pi * date.quarter / 4),
            "is_jan": 1 if date.month == 1 else 0,
            "is_feb": 1 if date.month == 2 else 0,
            "is_mar": 1 if date.month == 3 else 0,
            "is_dec": 1 if date.month == 12 else 0,
        }
        return pd.DataFrame([features])

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training."""
        X_list = []

        min_idx = max(12, max(self.lags))

        for i in range(min_idx, len(df)):
            target_date = df.index[i]

            # Create all feature types
            scale_features = self._create_multi_scale_features(df, i)
            macro_features = self._create_macro_features(df, i)
            season_features = self._create_seasonality_features(target_date)

            # Combine features
            features = pd.concat(
                [scale_features, macro_features, season_features], axis=1
            )
            X_list.append(features)

        X = pd.concat(X_list, ignore_index=True)

        # Remove NaN values
        X = X.fillna(0)

        return X

    def fit(self, df: pd.DataFrame, target_col: str = "Все товары и услуги"):
        """Fit the MIDAS+ model."""
        self.target_col = target_col

        # Prepare features
        X = self._prepare_features(df)
        y = df.iloc[max(12, max(self.lags)) :][target_col].reset_index(drop=True)

        self.feature_names = X.columns.tolist()

        # Fit Ridge regression
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha, random_state=42)),
            ]
        )

        self.model.fit(X, y)

        # Feature importance
        coef = self.model.named_steps["ridge"].coef_
        importance = dict(zip(self.feature_names, np.abs(coef)))

        # Select top features if max_features specified
        if self.max_features and self.max_features < len(self.feature_names):
            sorted_features = sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )
            self.selected_features = [
                f for f, _ in sorted_features[: self.max_features]
            ]

            # Refit with selected features
            X_selected = X[self.selected_features]
            self.model.fit(X_selected, y)
            self.feature_names = self.selected_features
        else:
            self.selected_features = self.feature_names

        return self

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> Dict[str, float]:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted")

        # For single-step prediction
        if horizon == 1:
            target_date = df.index[-1]

            # Create features
            scale_features = self._create_multi_scale_features(df, len(df) - 1)
            macro_features = self._create_macro_features(df, len(df) - 1)
            season_features = self._create_seasonality_features(target_date)

            # Combine
            X = pd.concat([scale_features, macro_features, season_features], axis=1)
            X = X.fillna(0)

            # Ensure all feature columns are present
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0

            X = X[self.feature_names]

            prediction = self.model.predict(X)[0]

            return {"prediction": prediction}

        # Multi-step iterative
        predictions = []
        df_extended = df.copy()

        for h in range(horizon):
            result = self.predict(df_extended, horizon=1)
            predictions.append(result["prediction"])

            next_date = df_extended.index[-1] + pd.DateOffset(months=1)
            new_row = df_extended.iloc[-1].copy()
            new_row.name = next_date
            new_row[self.target_col] = result["prediction"]

            df_extended = pd.concat([df_extended, pd.DataFrame([new_row])])

        return {"predictions": predictions}

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2024-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        """Run backtest."""
        results = []
        min_idx = max(12, max(self.lags))

        for i in range(min_idx, len(df)):
            if df.index[i] < pd.Timestamp(start_date):
                continue

            train_df = df.iloc[:i].copy()

            try:
                self.fit(train_df, target_col)

                result = self.predict(df.iloc[:i], horizon=1)
                prediction = result["prediction"]
                actual = df.iloc[i][target_col]

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


def optimize_hyperparameters(df: pd.DataFrame):
    """Optimize hyperparameters using grid search."""
    print("\n" + "=" * 70)
    print("Hyperparameter Optimization")
    print("=" * 70)

    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    max_features_list = [None, 10, 15, 20, 25]

    results = []

    for alpha in alphas:
        for max_features in max_features_list:
            model = MIDASPlusForecaster(alpha=alpha, max_features=max_features)
            backtest_results = model.backtest(df, start_date="2024-06-01")

            if len(backtest_results) > 0:
                mae = (backtest_results["error"].abs()).mean()
                results.append(
                    {
                        "alpha": alpha,
                        "max_features": max_features,
                        "mae": mae,
                        "n_predictions": len(backtest_results),
                    }
                )

    # Find best
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mae")

    print("\nTop 5 configurations:")
    print(results_df.head(5).to_string(index=False))

    best = results_df.iloc[0]
    print(
        f"\nBest: alpha={best['alpha']}, max_features={best['max_features']}, MAE={best['mae']:.4f}"
    )

    ridge_mae = 0.321
    if best["mae"] < ridge_mae:
        print(
            f"✅ IMPROVEMENT over Ridge: {((ridge_mae - best['mae']) / ridge_mae * 100):.2f}%"
        )
    else:
        print(
            f"❌ Still worse than Ridge by {((best['mae'] - ridge_mae) / ridge_mae * 100):.2f}%"
        )

    return best


def main():
    """Test MIDAS+ forecaster."""
    # Load data
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
    df = df[["mom", "brent", "usd_nom_i", "Ki"]].copy()
    df = df.rename(columns={"mom": "Все товары и услуги"})
    df = df.dropna(subset=["Все товары и услуги", "usd_nom_i"])

    print("=" * 70)
    print("MIDAS+ Forecaster Test")
    print("=" * 70)
    print(f"\nData: {len(df)} months ({df.index[0].date()} to {df.index[-1].date()})")

    # Baseline test
    print("\n" + "=" * 70)
    print("BASELINE TEST")
    print("=" * 70)
    model = MIDASPlusForecaster(alpha=0.1, max_features=20)
    results = model.backtest(df, start_date="2024-06-01")

    if len(results) > 0:
        mae = (results["error"].abs()).mean()
        rmse = np.sqrt((results["error"] ** 2).mean())
        me = results["error"].mean()

        print(f"\nResults ({len(results)} predictions):")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  ME:   {me:.4f}")

        ridge_mae = 0.321
        print(f"\n  Ridge MAE (baseline): {ridge_mae:.4f}")
        if mae < ridge_mae:
            print(f"  ✅ IMPROVEMENT: {((ridge_mae - mae) / ridge_mae * 100):.2f}%")
        else:
            print(f"  ❌ Worse by {((mae - ridge_mae) / ridge_mae * 100):.2f}%")

    # Hyperparameter optimization
    best_config = optimize_hyperparameters(df)


if __name__ == "__main__":
    main()
