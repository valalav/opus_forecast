"""
Optimized Ridge ETS - Ridge with ETS combination and outlier handling.

Based on successful Ridge Forecaster (MAE 0.321).
Improvements:
1. CV-optimized alpha (vs fixed 0.3)
2. Excludes outlier years
3. Uses proven feature set
"""

import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import RobustScaler


class OptimizedRidgeETSForecaster:
    """
    Optimized Ridge ETS Forecaster.

    Key improvements over standard Ridge:
    - CV-optimized alpha (vs fixed 0.3)
    - Outlier year exclusion
    - Proven feature set
    """

    name = "optimized_ridge_ets"

    # Outlier years
    OUTLIER_YEARS = [2022, 2010]

    # ETS weights (from Ridge Forecaster)
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

    def __init__(self, alpha: float = None, ets_weights: Dict[int, float] = None):
        """Initialize Optimized Ridge ETS forecaster."""
        self.alpha = alpha
        self.ets_weights = ets_weights if ets_weights else self.ETS_WEIGHTS
        self.ridge_model = None
        self.scaler = None
        self.feature_names = []
        self.target_col = None
        self.best_alpha = None
        self.seasonal_norm = None
        self.is_fitted = False

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        """Compute seasonal norm excluding outlier years."""
        df = df.copy()
        df["year"] = df.index.year
        df["month"] = df.index.month

        clean_df = df[~df["year"].isin(self.OUTLIER_YEARS)]
        return clean_df.groupby("month")[self.target_col].mean()

    def _create_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Create features based on Ridge Forecaster."""
        features = {}

        # Target lags (autocorrelation)
        for lag in [1, 2, 12]:
            if idx - lag >= 0:
                features[f"y_lag{lag}"] = df.iloc[idx - lag][self.target_col]

        # Moving average
        if idx - 3 >= 0:
            data = df.iloc[idx - 3 : idx][self.target_col]
            features["y_ma3"] = data.mean()

        # Seasonality
        date = df.index[idx]
        features["month_sin"] = np.sin(2 * np.pi * date.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * date.month / 12)

        # Deviation from seasonal norm
        if self.seasonal_norm is not None:
            features["seasonal_norm"] = self.seasonal_norm.get(date.month, 100.0)
            if idx >= 1:
                features["deviation_lag1"] = df.iloc[idx - 1][
                    self.target_col
                ] - self.seasonal_norm.get(df.index[idx - 1].month, 100.0)

        # Macro features (if available)
        for col in ["Ki", "Ruonia"]:
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
                        features[f"{col}_diff"] = curr - prev

        # Spread (Ki - Ruonia)
        if "Ki" in df.columns and "Ruonia" in df.columns:
            if idx >= 0:
                ki = df.iloc[idx]["Ki"] if pd.notna(df.iloc[idx]["Ki"]) else np.nan
                ruonia = (
                    df.iloc[idx]["Ruonia"]
                    if pd.notna(df.iloc[idx]["Ruonia"])
                    else np.nan
                )
                if pd.notna(ki) and pd.notna(ruonia):
                    features["spread"] = ki - ruonia

        return features

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training."""
        df = df.copy()
        df["year"] = df.index.year
        df["month"] = df.index.month

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
        """Fit the Optimized Ridge ETS model."""
        self.target_col = target_col

        # Compute seasonal norm (excluding outlier years)
        self.seasonal_norm = self._compute_seasonal_norm(df)

        # Prepare features
        X = self._prepare_features(df)
        y = df.iloc[24:][target_col].reset_index(drop=True)

        self.feature_names = X.columns.tolist()

        # Exclude outlier years from training
        df_year = df.copy()
        df_year["year"] = df_year.index.year
        train_mask = ~df_year.iloc[24:]["year"].isin(self.OUTLIER_YEARS).values

        X_train = X[train_mask].copy()
        y_train = y[train_mask].copy()

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # CV-optimized alpha
        if self.alpha is None:
            ridge_cv = RidgeCV(
                alphas=np.logspace(-3, 1, 100), cv=5, scoring="neg_mean_absolute_error"
            )
            ridge_cv.fit(X_scaled, y_train)
            self.best_alpha = ridge_cv.alpha_
            self.ridge_model = Ridge(alpha=self.best_alpha)
        else:
            self.best_alpha = self.alpha
            self.ridge_model = Ridge(alpha=self.alpha)

        self.ridge_model.fit(X_scaled, y_train)

        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> Dict[str, float]:
        """Make predictions with ETS combination."""
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

            # ETS prediction (seasonal norm)
            target_month = df.index[-1].month
            pred_ets = self.seasonal_norm.get(target_month, 100.0)

            # Combine Ridge and ETS
            ets_weight = self.ets_weights.get(target_month, 0.3)
            pred_combined = (1 - ets_weight) * pred_ridge + ets_weight * pred_ets

            return {
                "prediction": pred_combined,
                "pred_ridge": pred_ridge,
                "pred_ets": pred_ets,
                "ets_weight": ets_weight,
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
                        "pred_ridge": result["pred_ridge"],
                        "pred_ets": result["pred_ets"],
                        "ets_weight": result["ets_weight"],
                    }
                )
            except Exception as e:
                continue

        return pd.DataFrame(results)


def main():
    """Test the Optimized Ridge ETS forecaster."""
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
    print("Optimized Ridge ETS Forecaster")
    print("=" * 70)
    print(f"\nData: {len(df)} months ({df.index[0].date()} to {df.index[-1].date()})")
    print(f"Features: Target lags (1,2,12), 3-month MA, Seasonality,")
    print(f"          Deviation from norm, Macro (Ki, Ruonia, Spread)")
    print(f"Outlier years excluded: {OptimizedRidgeETSForecaster.OUTLIER_YEARS}")

    model = OptimizedRidgeETSForecaster()
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
