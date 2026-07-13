"""
Minimalist Forecaster - Simple Ridge with only proven essential features.

Philosophy: More features ≠ better performance.
Focus on what actually works for this specific forecasting problem.
"""

import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import RobustScaler


class MinimalistForecaster:
    """
    Minimalist Forecaster using only essential features.

    Proven approach:
    - Target lags (autocorrelation)
    - Momentum (recent trend)
    - Simple seasonality
    """

    def __init__(self):
        """Initialize Minimalist forecaster."""
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.target_col = None
        self.best_alpha = None
        self.is_fitted = False

    def _create_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Create only essential features."""
        features = {}

        # Most important: target lags (autocorrelation)
        for lag in [1, 2, 3, 6, 12]:
            if idx - lag >= 0:
                features[f"y_lag{lag}"] = df.iloc[idx - lag][self.target_col]

        # Simple momentum (recent trend)
        if idx - 3 >= 0:
            recent = df.iloc[idx - 3 : idx][self.target_col]
            features["y_ma3"] = recent.mean()
            features["y_trend3"] = recent.iloc[-1] - recent.iloc[0]

        # Very simple seasonality
        month = df.index[idx].month
        features["is_q1"] = 1 if month in [1, 2, 3] else 0
        features["is_q4"] = 1 if month in [10, 11, 12] else 0

        # Only most important macro variables
        if idx >= 0 and "usd_nom_i" in df.columns:
            if pd.notna(df.iloc[idx]["usd_nom_i"]):
                features["usd_curr"] = df.iloc[idx]["usd_nom_i"]

        return features

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training."""
        min_idx = 12
        X_list = []

        for i in range(min_idx, len(df)):
            features = self._create_features(df, i)
            X_list.append(features)

        X = pd.DataFrame(X_list)
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        return X

    def fit(self, df: pd.DataFrame, target_col: str = "Все товары и услуги"):
        """Fit the Minimalist model."""
        self.target_col = target_col

        X = self._prepare_features(df)
        y = df.iloc[12:][target_col].reset_index(drop=True)

        self.feature_names = X.columns.tolist()

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        ridge_cv = RidgeCV(
            alphas=np.logspace(-4, 1, 100), cv=5, scoring="neg_mean_absolute_error"
        )
        ridge_cv.fit(X_scaled, y)

        self.best_alpha = ridge_cv.alpha_
        self.model = Ridge(alpha=self.best_alpha)
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

            return {"prediction": prediction}

        return {"predictions": [self.predict(df, 1)["prediction"]]}

    def backtest(
        self, df: pd.DataFrame, start_date: str = "2024-06-01"
    ) -> pd.DataFrame:
        """Run backtest."""
        results = []
        start_ts = pd.Timestamp(start_date)

        for i in range(12, len(df)):
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
    """Test the Minimalist forecaster."""
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
    print("Minimalist Forecaster")
    print("=" * 70)
    print(f"\nData: {len(df)} months ({df.index[0].date()} to {df.index[-1].date()})")
    print(
        f"Features: {['Target lags (1,2,3,6,12)', '3-month momentum', 'Quarter seasonality', 'USD rate']}"
    )

    model = MinimalistForecaster()
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
