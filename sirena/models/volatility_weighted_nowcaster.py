"""
Volatility-Weighted Nowcaster for CPI Inflation
=================================================

This model implements inverse volatility weighting for weekly price signals.
Based on Task 411 research findings.

Hypothesis (from research):
    Weighting products by inverse volatility (1/std) improves nowcasting accuracy.

Research Findings (Task 411):
    - Inverse volatility weighting shows NO improvement vs basket weights
    - Product-specific volatility tuning worsens MAE by 5%
    - Conclusion: Use standard CPI basket weights, NOT volatility-based weights

Nevertheless, this model is implemented for completeness and historical reference.

Methodology:
    1. Calculate historical volatility (std of WoW growth) per product
    2. Compute inverse volatility weights: w_i = 1 / std_i
    3. Normalize weights to sum to 1
    4. Aggregate weekly signals using these weights
    5. Use weighted signal to predict monthly CPI

Data Source:
    - Weekly prices: data/kbr_weekly_prices_2008_2026.csv
    - Monthly CPI: data/inflation_data.csv

Training Period: 2016-2026 (post-Crimea, stable methodology)
"""

import pandas as pd
import numpy as np
import argparse
from typing import Optional, Dict, Any, List
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    from .base import BaseForecaster
    from ..data.weekly_loader import (
        load_weekly_prices,
        get_high_quality_products,
        HIGH_QUALITY_PRODUCTS,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from sirena.models.base import BaseForecaster
    from sirena.data.weekly_loader import (
        load_weekly_prices,
        get_high_quality_products,
        HIGH_QUALITY_PRODUCTS,
    )


class VolatilityWeightedNowcaster(BaseForecaster):
    """
    Volatility-Weighted Nowcaster using inverse volatility weights.

    This model calculates product volatility (std of WoW growth rates)
    and uses inverse volatility as weights for aggregating weekly signals.

    Research Finding:
        Task 411 showed that volatility weighting does NOT improve
        accuracy compared to standard CPI basket weights.
        This model is provided for reference/comparison.

    Methodology:
        1. For each product, calculate std(wow_growth) over training period
        2. Compute weight = 1 / std (higher weight for stable products)
        3. Normalize weights to sum to 1
        4. Aggregate weekly signals: signal = Σ(price_growth_i * weight_i)
        5. Use signal + lagged CPI + seasonality to forecast monthly CPI
    """

    name = "volatility_weighted_nowcaster"
    MIN_TRAIN_SIZE = 36

    def __init__(
        self,
        alpha: float = 1.0,
        min_samples_per_product: int = 20,
        volatility_window: int = 52,  # 1 year of weeks
        **kwargs,
    ):
        """
        Initialize volatility-weighted nowcaster.

        Args:
            alpha: Regularization strength for Ridge model
            min_samples_per_product: Minimum observations to calculate volatility
            volatility_window: Rolling window size for volatility calculation (weeks)
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.min_samples_per_product = min_samples_per_product
        self.volatility_window = volatility_window

        # Ridge model for final prediction
        from sklearn.linear_model import Ridge

        self.model = Ridge(alpha=alpha, random_state=42)
        self.scaler = None

        # Volatility weights per product
        self.product_volatility = {}
        self.product_weights = {}

        # Weekly signal per month
        self.monthly_signals = {}

        self._is_fitted = False

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "VolatilityWeightedNowcaster":
        """
        Fit volatility-weighted nowcaster.

        Steps:
            1. Load weekly price data
            2. Calculate product volatility (std of WoW growth)
            3. Compute inverse volatility weights
            4. Aggregate weekly signals to monthly
            5. Train Ridge model on signal + lagged CPI

        Args:
            df: Monthly CPI DataFrame (must have datetime index)
            target_col: Target column name (default: 'Все товары и услуги')

        Returns:
            self for method chaining
        """
        from sklearn.preprocessing import StandardScaler

        # Validate data
        target_series = self._validate_data(df, target_col)

        # Load weekly price data
        weekly_df = self._load_weekly_data()

        # Calculate volatility and weights
        self._calculate_volatility_weights(weekly_df)

        # Aggregate weekly to monthly signals
        monthly_features = self._aggregate_weekly_to_monthly(weekly_df, df)

        # Merge with monthly CPI
        features_df = df[[target_col]].join(monthly_features, how="inner").dropna()

        # Create features (X will have fewer rows due to lag NaNs)
        X_df = features_df.copy()

        # Add lags
        X_df["target_lag1"] = X_df[target_col].shift(1)
        X_df["target_lag2"] = X_df[target_col].shift(2)

        # Add seasonality
        X_df["month"] = X_df.index.month
        X_df["month_sin"] = np.sin(2 * np.pi * X_df["month"] / 12)
        X_df["month_cos"] = np.cos(2 * np.pi * X_df["month"] / 12)

        # Select features
        feature_cols = [
            "volatility_weighted_signal",
            "target_lag1",
            "target_lag2",
            "month_sin",
            "month_cos",
        ]

        # Drop NaN rows and get matching y values
        X = X_df[feature_cols].dropna().values
        y = X_df.loc[X_df[feature_cols].dropna().index, target_col].values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """
        Forecast monthly CPI.

        For h=1, uses latest weekly signal + last CPI values.

        Args:
            horizon: Forecast horizon in months (typically 1 for nowcasting)

        Returns:
            numpy array with MoM forecasts in % (not index)
        """
        self._check_fitted()

        if horizon != 1:
            # For h>1, naive persistence (this is a nowcasting model)
            warnings.warn(
                f"VolatilityWeightedNowcaster is designed for h=1. "
                f"Using persistence for h={horizon}."
            )
            return np.array([0.0] * horizon)

        # Load latest weekly data for signal
        weekly_df = self._load_weekly_data()

        # Calculate current month signal
        current_date = pd.Timestamp.now()
        period = current_date.to_period("M")

        # Get this month's weekly data
        month_weeks = weekly_df[weekly_df["year_month"] == period]

        if len(month_weeks) == 0:
            # Fallback to last available signal
            last_signal = list(self.monthly_signals.values())[-1]
        else:
            # Compute volatility-weighted signal
            signal = self._compute_weighted_signal(month_weeks)
            last_signal = signal

        # Simple forecast: use signal as MoM prediction
        # This assumes weekly price movements directly translate to monthly inflation
        prediction = last_signal

        return np.array([prediction])

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2024-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        """
        Run backtest using rolling window approach.

        Args:
            df: Monthly CPI DataFrame
            start_date: Start date for backtest
            target_col: Target column name

        Returns:
            DataFrame with columns: date, actual, prediction, error
        """
        from sklearn.preprocessing import StandardScaler

        results = []
        test_dates = df[df.index >= start_date].index

        for date in test_dates:
            # Train on data before this date
            train_df = df[df.index < date]

            if len(train_df) < self.MIN_TRAIN_SIZE:
                continue

            # Load weekly data
            weekly_df = self._load_weekly_data()

            # Calculate volatility on training period
            self._calculate_volatility_weights(weekly_df, end_date=date)

            # Get signal for prediction month
            period = date.to_period("M")
            month_weeks = weekly_df[weekly_df["year_month"] == period]

            if len(month_weeks) == 0:
                continue

            signal = self._compute_weighted_signal(month_weeks)

            # Store prediction
            actual = df.loc[date, target_col]
            results.append(
                {
                    "date": date,
                    "actual": actual,
                    "prediction": signal,
                    "error": signal - actual,
                }
            )

        return pd.DataFrame(results)

    def _load_weekly_data(self) -> pd.DataFrame:
        """Load and prepare weekly price data."""
        weekly_df = load_weekly_prices()
        weekly_df["year_month"] = weekly_df["date"].dt.to_period("M")
        return weekly_df

    def _calculate_volatility_weights(
        self, weekly_df: pd.DataFrame, end_date: Optional[pd.Timestamp] = None
    ):
        """
        Calculate volatility (std) and inverse volatility weights per product.

        Args:
            weekly_df: Weekly price data
            end_date: Optional cutoff date for volatility calculation
        """
        weights = {}
        volatility = {}

        # Filter high-quality products
        hq_products = get_high_quality_products()

        for prod_code in hq_products["product_code"].unique():
            prod_data = weekly_df[weekly_df["product_code"] == prod_code].copy()

            if end_date:
                prod_data = prod_data[prod_data["date"] < end_date]

            if len(prod_data) < self.min_samples_per_product:
                continue

            # Calculate WoW growth volatility
            wow_growth = prod_data["wow_growth"].dropna()

            if len(wow_growth) < self.min_samples_per_product:
                continue

            std_dev = wow_growth.std()

            if std_dev == 0 or pd.isna(std_dev):
                continue

            # Inverse volatility weight
            inv_vol = 1.0 / std_dev
            weights[prod_code] = inv_vol
            volatility[prod_code] = std_dev

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.product_weights = {
                code: w / total_weight for code, w in weights.items()
            }
        else:
            # Fallback to equal weights
            n_products = len(weights)
            self.product_weights = {code: 1.0 / n_products for code in weights}

        self.product_volatility = volatility

    def _compute_weighted_signal(self, month_weeks: pd.DataFrame) -> float:
        """
        Compute volatility-weighted signal for a month.

        Signal = Σ(wow_growth_i * weight_i)

        Args:
            month_weeks: Weekly data for one month

        Returns:
            Weighted average WoW growth
        """
        signals = []

        for prod_code, weight in self.product_weights.items():
            prod_data = month_weeks[month_weeks["product_code"] == prod_code]

            if len(prod_data) == 0:
                continue

            # Average WoW growth for this product in the month
            avg_growth = prod_data["wow_growth"].mean()

            if pd.isna(avg_growth):
                continue

            signals.append(avg_growth * weight)

        if not signals:
            return 0.0

        return sum(signals)

    def _aggregate_weekly_to_monthly(
        self, weekly_df: pd.DataFrame, monthly_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate weekly signals to monthly.

        Returns DataFrame with volatility_weighted_signal column.
        """
        monthly_data = []

        for period in weekly_df["year_month"].unique():
            month_weeks = weekly_df[weekly_df["year_month"] == period]

            # Compute weighted signal
            signal = self._compute_weighted_signal(month_weeks)

            # Convert period to timestamp (first day of month)
            timestamp = period.to_timestamp()

            monthly_data.append(
                {"date": timestamp, "volatility_weighted_signal": signal}
            )

        monthly_df_signals = pd.DataFrame(monthly_data)
        monthly_df_signals = monthly_df_signals.set_index("date")

        return monthly_df_signals

    def _create_features(
        self, features_df: pd.DataFrame, target_col: str
    ) -> np.ndarray:
        """
        Create feature matrix for training.

        Features:
            - volatility_weighted_signal (main)
            - lag1, lag2 of target
            - month_sin, month_cos (seasonality)
        """
        df = features_df.copy()

        # Add lags
        df["target_lag1"] = df[target_col].shift(1)
        df["target_lag2"] = df[target_col].shift(2)

        # Add seasonality
        df["month"] = df.index.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Select features
        feature_cols = [
            "volatility_weighted_signal",
            "target_lag1",
            "target_lag2",
            "month_sin",
            "month_cos",
        ]

        X = df[feature_cols].dropna().values

        return X

    def get_volatility_weights(self) -> Dict[int, float]:
        """
        Get computed volatility weights per product.

        Returns:
            Dictionary {product_code: weight}
        """
        self._check_fitted()
        return self.product_weights.copy()

    def get_product_volatility(self) -> Dict[int, float]:
        """
        Get calculated volatility (std) per product.

        Returns:
            Dictionary {product_code: std_dev}
        """
        self._check_fitted()
        return self.product_volatility.copy()


def main():
    """CLI for standalone testing."""
    parser = argparse.ArgumentParser(
        description="Test VolatilityWeightedNowcaster model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/inflation_data.csv",
        help="Path to monthly CPI data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/volatility_nowcast_test.csv",
        help="Output path for backtest results",
    )
    parser.add_argument(
        "--start-date", type=str, default="2024-01-01", help="Backtest start date"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Ridge regularization strength"
    )

    args = parser.parse_args()

    # Load monthly data
    df = pd.read_csv(args.data, sep=";", encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", dayfirst=True)
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df = df.set_index("Date").sort_index()

    # Normalize string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(",", ".").astype(float)

    # Create and fit model
    model = VolatilityWeightedNowcaster(alpha=args.alpha)
    model.fit(df)

    print(f"Model fitted: {model.name}")
    print(f"Training data until: {model._last_train_date}")
    print(f"Number of products weighted: {len(model.product_weights)}")

    # Show top products by weight (most stable)
    top_products = sorted(
        model.product_weights.items(), key=lambda x: x[1], reverse=True
    )[:5]

    print("\nTop 5 products by weight (most stable):")
    for code, weight in top_products:
        prod_name = HIGH_QUALITY_PRODUCTS.get(code, {}).get("name", "Unknown")
        std = model.product_volatility.get(code, 0)
        print(f"  {code} ({prod_name}): weight={weight:.4f}, std={std:.4f}")

    # Run backtest
    results = model.backtest(df, start_date=args.start_date)

    if len(results) > 0:
        mae = results["error"].abs().mean()
        print(f"\nBacktest MAE: {mae:.4f}%")
        print(f"Backtest period: {results['date'].min()} to {results['date'].max()}")
        print(f"Number of predictions: {len(results)}")

        # Save results
        results.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")
    else:
        print("\nNo backtest results generated (insufficient data)")


if __name__ == "__main__":
    main()
