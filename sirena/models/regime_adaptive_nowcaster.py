"""
Regime-Adaptive Nowcaster for CPI Inflation
==========================================

This model implements regime-switching for weekly price signals.
Based on Task 414 (Weekly: Regime Weights) research findings.

Hypothesis (from research):
    Different economic regimes (shock/normal/high_inflation) require
    different product weightings for accurate nowcasting.

Research Findings (Task 414):
    - Regime-specific product weights differ significantly from fixed weights
    - Shock regime: Equal weighting (all products ~11%) works best
    - Normal regime: Product-specific weights based on historical performance
    - High inflation regime: Some products get higher weights (oil, eggs, etc.)

Regime Detection:
    - Uses detect_regime() from regime_detector.py
    - SHOCK: |ΔRuonia| > 0.5 or |ΔKi| > 0.5
    - HIGH_INFLATION: ΔInflation_YoY > 1.5 pp
    - NORMAL: otherwise

Data Source:
    - Weekly prices: data/kbr_weekly_prices_2008_2026.csv
    - Monthly CPI: data/inflation_data.csv
    - Regime weights: data/weekly_regime_weights.csv (from Task 414)
    - Macro data: data/inflation_data.csv (Ki, Ruonia, etc.)

Methodology:
    1. Load weekly price data
    2. Load regime-specific product weights from Task 414 backtest
    3. Detect current macro regime using Ki/Ruonia/inflation data
    4. Aggregate weekly signals using regime-specific weights
    5. Forecast monthly CPI using weighted signal + lagged CPI
"""

import pandas as pd
import numpy as np
import argparse
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    from .base import BaseForecaster
    from ..data.weekly_loader import load_weekly_prices
    from .regime_detector import detect_regime, MacroRegime
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from sirena.models.base import BaseForecaster
    from sirena.data.weekly_loader import load_weekly_prices
    from sirena.models.regime_detector import detect_regime, MacroRegime


class RegimeAdaptiveNowcaster(BaseForecaster):
    """
    Regime-Adaptive Nowcaster using regime-specific product weights.

    This model detects the current macroeconomic regime and uses
    regime-specific product weights for aggregating weekly price signals.

    Regime Detection:
        - SHOCK: Rapid rate changes (|ΔKi| > 0.5 or |ΔRuonia| > 0.5)
        - HIGH_INFLATION: Inflation acceleration (ΔYoY > 1.5 pp)
        - NORMAL: Stable conditions

    Regime-Specific Weights:
        - Shock: Equal weighting (all products ~11%)
        - Normal: Optimized weights from historical backtest
        - High Inflation: Weights focused on volatile products

    Methodology:
        1. Detect current regime using macro indicators (Ki, Ruonia, inflation)
        2. Select regime-specific product weights
        3. Aggregate weekly price signals using these weights
        4. Forecast monthly CPI using signal + lagged CPI + seasonality

    Example:
        >>> model = RegimeAdaptiveNowcaster()
        >>> model.fit(df_with_macro)
        >>> # Detect current regime
        >>> regime = model.detect_current_regime()
        >>> print(f"Current regime: {regime}")
        >>> # Forecast
        >>> fc = model.forecast(horizon=1)
    """

    name = "regime_adaptive_nowcaster"
    MIN_TRAIN_SIZE = 36

    def __init__(
        self,
        alpha: float = 1.0,
        regime_weights_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize regime-adaptive nowcaster.

        Args:
            alpha: Regularization strength for Ridge model
            regime_weights_path: Path to CSV with regime-specific weights
                (default: data/weekly_regime_weights.csv from Task 414)
        """
        super().__init__(**kwargs)
        self.alpha = alpha

        # Path to regime-specific weights (from Task 414)
        self.regime_weights_path = regime_weights_path or str(
            Path(__file__).parent.parent.parent / "data" / "weekly_regime_weights.csv"
        )

        # Ridge model for final prediction
        from sklearn.linear_model import Ridge

        self.model = Ridge(alpha=alpha, random_state=42)
        self.scaler = None

        # Regime-specific product weights {regime: {product_code: weight}}
        self.regime_weights: Dict[str, Dict[int, float]] = {}

        # Current regime
        self._current_regime: str = "normal"
        self._regime_diagnostics: Dict = {}

        # Weekly signal per month
        self.monthly_signals = {}

        self._is_fitted = False

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "RegimeAdaptiveNowcaster":
        """
        Fit regime-adaptive nowcaster.

        Steps:
            1. Load regime-specific weights from Task 414 backtest
            2. Load weekly price data
            3. Detect current regime using macro indicators
            4. Aggregate weekly signals to monthly using regime weights
            5. Train Ridge model on signal + lagged CPI

        Args:
            df: Monthly CPI DataFrame with macro indicators (Ki, Ruonia, etc.)
                Must have datetime index
            target_col: Target column name (default: 'Все товары и услуги')

        Returns:
            self for method chaining
        """
        from sklearn.preprocessing import StandardScaler

        # Validate data
        target_series = self._validate_data(df, target_col)

        # Load regime-specific weights
        self._load_regime_weights()

        # Detect current regime
        self._current_regime, self._regime_diagnostics = self.detect_current_regime(df)

        # Load weekly price data
        weekly_df = self._load_weekly_data()

        # Aggregate weekly to monthly signals using regime-specific weights
        monthly_features = self._aggregate_weekly_to_monthly(weekly_df, df)

        # Merge with monthly CPI
        features_df = df[[target_col]].join(monthly_features, how="inner").dropna()

        # Create features
        X = self._create_features(features_df, target_col)
        y = features_df[target_col].values

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

        For h=1, uses latest weekly signal with regime-specific weights.

        Args:
            horizon: Forecast horizon in months (typically 1 for nowcasting)

        Returns:
            numpy array with MoM forecasts in % (not index)
        """
        self._check_fitted()

        if horizon != 1:
            # For h>1, naive persistence (this is a nowcasting model)
            warnings.warn(
                f"RegimeAdaptiveNowcaster is designed for h=1. "
                f"Using persistence for h={horizon}."
            )
            return np.array([0.0] * horizon)

        # Load latest weekly data for signal
        weekly_df = self._load_weekly_data()

        # Calculate current month signal with regime-specific weights
        current_date = pd.Timestamp.now()
        period = current_date.to_period("M")

        # Get this month's weekly data
        month_weeks = weekly_df[weekly_df["year_month"] == period]

        if len(month_weeks) == 0:
            # Fallback to last available signal
            last_signal = list(self.monthly_signals.values())[-1]
        else:
            # Compute regime-weighted signal
            signal = self._compute_regime_weighted_signal(month_weeks)
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
            df: Monthly CPI DataFrame with macro indicators
            start_date: Start date for backtest
            target_col: Target column name

        Returns:
            DataFrame with columns: date, actual, prediction, error, regime
        """
        from sklearn.preprocessing import StandardScaler

        results = []
        test_dates = df[df.index >= start_date].index

        for date in test_dates:
            # Train on data before this date
            train_df = df[df.index < date]

            if len(train_df) < self.MIN_TRAIN_SIZE:
                continue

            # Create model for this fold
            model_fold = RegimeAdaptiveNowcaster(
                alpha=self.alpha, regime_weights_path=self.regime_weights_path
            )
            model_fold.fit(train_df, target_col)

            # Get signal for prediction month
            period = date.to_period("M")
            weekly_df = self._load_weekly_data()
            month_weeks = weekly_df[weekly_df["year_month"] == period]

            if len(month_weeks) == 0:
                continue

            signal = model_fold._compute_regime_weighted_signal(month_weeks)

            # Store prediction
            actual = df.loc[date, target_col]
            results.append(
                {
                    "date": date,
                    "actual": actual,
                    "prediction": signal,
                    "error": signal - actual,
                    "regime": model_fold._current_regime,
                }
            )

        return pd.DataFrame(results)

    def detect_current_regime(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[str, Dict]:
        """
        Detect current macroeconomic regime.

        Uses detect_regime() from regime_detector.py which analyzes:
        - Ki (key rate) changes
        - Ruonia changes
        - Inflation YoY acceleration

        Args:
            df: DataFrame with macro indicators (Ki, Ruonia, inflation)
                If None, uses data from fit()

        Returns:
            Tuple[str, Dict]: (regime_name, diagnostics_dict)

            regime_name: One of 'normal', 'shock', 'high_inflation'
            diagnostics: Dictionary with diagnostic values (ki_change, ruonia_change, etc.)
        """
        if df is None:
            if hasattr(self, "_df") and self._df is not None:
                df = self._df
            else:
                raise ValueError(
                    "No data available for regime detection. Provide df parameter or call fit() first."
                )

        # Use the regime detector from regime_detector.py
        # Convert from MacroRegime enum to string
        macro_regime, diagnostics = detect_regime(
            df,
            ki_col="Ki",
            ruonia_col="Ruonia",
            mom_col="mom",
        )

        regime_name = macro_regime.value
        return regime_name, diagnostics

    def _load_regime_weights(self):
        """Load regime-specific product weights from Task 414 backtest results."""
        try:
            weights_df = pd.read_csv(self.regime_weights_path)

            # Group by regime
            for regime in weights_df["Regime"].unique():
                regime_data = weights_df[weights_df["Regime"] == regime]
                weights_dict = {}
                for _, row in regime_data.iterrows():
                    weights_dict[row["Product_code"]] = row["Weight"]
                self.regime_weights[regime] = weights_dict

        except Exception as e:
            warnings.warn(
                f"Failed to load regime weights: {e}. Using default equal weights."
            )
            # Fallback to equal weights
            self.regime_weights = {
                "normal": {},
                "shock": {},
                "high_inflation": {},
            }

    def _load_weekly_data(self) -> pd.DataFrame:
        """Load and prepare weekly price data."""
        weekly_df = load_weekly_prices()
        weekly_df["year_month"] = weekly_df["date"].dt.to_period("M")
        return weekly_df

    def _compute_regime_weighted_signal(self, month_weeks: pd.DataFrame) -> float:
        """
        Compute regime-weighted signal for a month.

        Signal = Σ(wow_growth_i * weight_i)
        where weight_i depends on current regime.

        Args:
            month_weeks: Weekly data for one month

        Returns:
            Weighted average WoW growth using regime-specific weights
        """
        signals = []

        # Get weights for current regime
        current_weights = self.regime_weights.get(
            self._current_regime, self.regime_weights.get("normal", {})
        )

        if not current_weights:
            # Fallback: equal weights for all products
            products = month_weeks["product_code"].unique()
            n_products = len(products)
            current_weights = {code: 1.0 / n_products for code in products}

        for prod_code, weight in current_weights.items():
            prod_data = month_weeks[month_weeks["product_code"] == prod_code]

            if len(prod_data) == 0:
                continue

            # Average WoW growth for this product in month
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
        Aggregate weekly signals to monthly using regime-specific weights.

        Returns DataFrame with regime_weighted_signal column.
        """
        monthly_data = []

        for period in weekly_df["year_month"].unique():
            month_weeks = weekly_df[weekly_df["year_month"] == period]

            # Compute weighted signal
            signal = self._compute_regime_weighted_signal(month_weeks)

            # Convert period to timestamp (first day of month)
            timestamp = period.to_timestamp()

            monthly_data.append({"date": timestamp, "regime_weighted_signal": signal})

        monthly_df_signals = pd.DataFrame(monthly_data)
        monthly_df_signals = monthly_df_signals.set_index("date")

        return monthly_df_signals

    def _create_features(
        self, features_df: pd.DataFrame, target_col: str
    ) -> np.ndarray:
        """
        Create feature matrix for training.

        Features:
            - regime_weighted_signal (main)
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
            "regime_weighted_signal",
            "target_lag1",
            "target_lag2",
            "month_sin",
            "month_cos",
        ]

        X = df[feature_cols].dropna().values

        return X

    def get_regime_weights(self) -> Dict[str, Dict[int, float]]:
        """
        Get regime-specific product weights.

        Returns:
            Dictionary {regime: {product_code: weight}}
        """
        self._check_fitted()
        return self.regime_weights.copy()

    @property
    def current_regime(self) -> str:
        """Get current detected regime."""
        return self._current_regime

    def get_regime_diagnostics(self) -> Dict:
        """
        Get regime detection diagnostics.

        Returns:
            Dictionary with diagnostic values (ki_change, ruonia_change, yoy_change, etc.)
        """
        self._check_fitted()
        return self._regime_diagnostics.copy()


def main():
    """CLI for standalone testing."""
    parser = argparse.ArgumentParser(description="Test RegimeAdaptiveNowcaster model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/inflation_data.csv",
        help="Path to monthly CPI data with macro indicators",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/regime_adaptive_nowcast_test.csv",
        help="Output path for backtest results",
    )
    parser.add_argument(
        "--start-date", type=str, default="2024-01-01", help="Backtest start date"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Ridge regularization strength"
    )
    parser.add_argument(
        "--regime-weights",
        type=str,
        default=None,
        help="Path to regime-specific weights CSV",
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
    model = RegimeAdaptiveNowcaster(
        alpha=args.alpha, regime_weights_path=args.regime_weights
    )
    model.fit(df)

    print(f"Model fitted: {model.name}")
    print(f"Training data until: {model._last_train_date}")
    print(f"Current regime: {model.current_regime}")
    print(f"Regime diagnostics: {model.get_regime_diagnostics()}")

    # Show regime-specific weights
    print("\nRegime-specific weights summary:")
    for regime, weights in model.regime_weights.items():
        print(f"  {regime}: {len(weights)} products")

    # Run backtest
    results = model.backtest(df, start_date=args.start_date)

    if len(results) > 0:
        mae = results["error"].abs().mean()
        print(f"\nBacktest MAE: {mae:.4f}%")
        print(f"Backtest period: {results['date'].min()} to {results['date'].max()}")
        print(f"Number of predictions: {len(results)}")

        # Show performance by regime
        print("\nPerformance by regime:")
        regime_perf = results.groupby("regime")["error"].agg(
            ["count", lambda x: x.abs().mean()]
        )
        regime_perf.columns = ["count", "mae"]
        print(regime_perf)

        # Save results
        results.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")
    else:
        print("\nNo backtest results generated (insufficient data)")


if __name__ == "__main__":
    main()
