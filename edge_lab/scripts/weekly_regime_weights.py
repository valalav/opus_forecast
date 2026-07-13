#!/usr/bin/env python3
"""
Regime-Dependent Weekly Weights for CPI Nowcasting
===================================================

Tests the hypothesis that different product weights work better in different economic regimes.

Methodology:
1. Detect regime (shock/normal/high_inflation) using RegimeDetector
2. Optimize weights separately for each regime
3. Create adaptive nowcaster that switches weights
4. Backtest vs fixed weights
5. Document regime-specific patterns

Outputs:
- data/weekly_regime_weights.csv - Optimal weights per regime
- data/weekly_regime_backtest.csv - Backtest comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple
from scipy.optimize import minimize
import argparse

warnings.filterwarnings("ignore")

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sirena.data.weekly_loader import (
        load_weekly_prices,
        calculate_product_volatility,
        HIGH_QUALITY_PRODUCTS,
        aggregate_to_monthly,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sirena" / "data"))
    from weekly_loader import (
        load_weekly_prices,
        calculate_product_volatility,
        HIGH_QUALITY_PRODUCTS,
        aggregate_to_monthly,
    )

try:
    from agents.regime_detector import RegimeDetector, RegimeType
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))
    from regime_detector import RegimeDetector, RegimeType


class RegimeWeighter:
    """
    Computes optimal product weights for different market regimes.

    Analyzes how each product's predictive power varies across regimes
    and calculates optimal weights for each regime type.
    """

    def __init__(
        self,
        shock_threshold: float = 0.5,
        inflation_threshold: float = 1.5,
        lookback_weeks: int = 52,
    ):
        """
        Initialize RegimeWeighter.

        Args:
            shock_threshold: Threshold for rate shock detection (pp)
            inflation_threshold: Threshold for high inflation (pp)
            lookback_weeks: Lookback window for weight calculation
        """
        self.shock_threshold = shock_threshold
        self.inflation_threshold = inflation_threshold
        self.lookback_weeks = lookback_weeks
        self.regime_detector = RegimeDetector(
            shock_threshold=shock_threshold,
            inflation_shock_threshold=inflation_threshold,
        )
        self.regime_weights: Dict[str, Tuple[np.ndarray, List[int]]] = {}
        self.regime_history: List[str] = []

    def fit(
        self,
        merged_df: pd.DataFrame,
    ) -> "RegimeWeighter":
        """
        Fit regime weighter on historical data.

        Args:
            merged_df: Merged DataFrame with product columns, inflation, and regime

        Returns:
            Self (fitted RegimeWeighter)
        """
        print("[RegimeWeighter] Computing optimal regime-specific weights...")

        # Optimize weights for each regime
        print("[RegimeWeighter] Calculating optimal weights per regime...")

        for regime_type in RegimeType:
            regime_name = regime_type.value
            print(f"  Optimizing for {regime_name}...")

            weights, mae, codes = optimize_weights_for_regime(merged_df, regime_name)
            self.regime_weights[regime_name] = (weights, codes)
            print(f"    Optimal MAE: {mae:.4f}%")

        return self

    def predict_weights(self, regime: str) -> Tuple[np.ndarray, List[int]]:
        """
        Get optimal weights for a given regime.

        Args:
            regime: Regime type ('normal', 'shock', 'high_inflation')

        Returns:
            Tuple of (weights array, product_codes list)
        """
        if regime not in self.regime_weights:
            regime = "normal"

        return self.regime_weights.get(regime, (np.array([]), []))

    def compute_weights(
        self, regime: str, df: pd.DataFrame
    ) -> Tuple[np.ndarray, float, List[int]]:
        """
        Compute optimal weights for a specific regime.

        Args:
            regime: Regime type ('normal', 'shock', 'high_inflation')
            df: Dataset with product columns and inflation target

        Returns:
            Tuple of (weights, optimal_MAE, product_codes)
        """
        weights, mae, codes = optimize_weights_for_regime(df, regime)
        return weights, mae, codes

    def get_regime_weights_df(self) -> pd.DataFrame:
        """
        Get regime weights as DataFrame.

        Returns:
            DataFrame with columns: Regime, Product_code, Product_name, Weight
        """
        rows = []
        for regime, (weights, codes) in self.regime_weights.items():
            for idx, product_code in enumerate(codes):
                if idx < len(weights):
                    product_name = HIGH_QUALITY_PRODUCTS.get(product_code, {}).get(
                        "name", f"Product_{product_code}"
                    )
                    rows.append(
                        {
                            "Regime": regime,
                            "Product_code": product_code,
                            "Product_name": product_name,
                            "Weight": weights[idx],
                        }
                    )

        return pd.DataFrame(rows)

    def save_weights(self, output_path: Path):
        """
        Save regime-specific weights to CSV.

        Args:
            output_path: Path to output CSV file
        """
        df_weights = self.get_regime_weights_df()
        df_weights.to_csv(output_path, index=False)
        print(f"[RegimeWeighter] Weights saved to: {output_path}")
        print(f"  Total rows: {len(df_weights)}")


def load_monthly_cpi() -> pd.DataFrame:
    """Load monthly CPI inflation data with regime indicators."""
    base_paths = [
        Path.cwd() / "data" / "inflation_data.csv",
        Path.cwd().parent / "data" / "inflation_data.csv",
        Path(__file__).parent.parent / "data" / "inflation_data.csv",
    ]

    for path in base_paths:
        if path.exists():
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
            df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", dayfirst=True)
            df = df.set_index("Date").sort_index()
            df.index = df.index.normalize()

            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

            if "mom" in df.columns:
                df["inflation"] = df["mom"] - 100
            else:
                first_numeric = df.select_dtypes(include=[np.number]).columns[0]
                df["inflation"] = df[first_numeric].pct_change() * 100

            return df[["inflation"]].dropna()

    raise FileNotFoundError("Monthly CPI data not found")


def load_enhanced_cpi() -> pd.DataFrame:
    """Load enhanced CPI data with macro indicators for regime detection."""
    base_paths = [
        Path.cwd() / "data" / "enhanced_inflation_data.csv",
        Path(__file__).parent.parent / "data" / "enhanced_inflation_data.csv",
    ]

    for path in base_paths:
        if path.exists():
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            df.index = df.index.normalize()

            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

            if "mom" in df.columns:
                df["inflation"] = df["mom"] - 100

            return df.dropna(subset=["inflation"])

    raise FileNotFoundError("Enhanced CPI data not found")


def detect_regimes_for_months(cpi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect regimes for each month in CPI data.

    Args:
        cpi_df: DataFrame with Date index and inflation, Ki, Ruonia columns

    Returns:
        DataFrame with regime column added
    """
    detector = RegimeDetector()

    regimes = []
    confidences = []

    for date in cpi_df.index:
        result = detector.detect(cpi_df, date)
        regimes.append(result.regime.value)
        confidences.append(result.confidence)

    cpi_df = cpi_df.copy()
    cpi_df["regime"] = regimes
    cpi_df["regime_confidence"] = confidences

    return cpi_df


def prepare_weekly_monthly_dataset(
    weekly_df: pd.DataFrame, monthly_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare merged dataset with weekly signals and monthly CPI with regimes.

    Args:
        weekly_df: Weekly price data
        monthly_df: Monthly CPI with regime labels

    Returns:
        DataFrame with monthly aggregated weekly signals
    """
    high_quality_codes = list(HIGH_QUALITY_PRODUCTS.keys())
    filtered = weekly_df[weekly_df["product_code"].isin(high_quality_codes)].copy()

    filtered["year_month"] = filtered["date"].dt.to_period("M")

    monthly_signals = (
        filtered.groupby(["year_month", "product_code"])["wow_growth"]
        .mean()
        .reset_index()
    )
    monthly_signals["Date"] = monthly_signals["year_month"].dt.to_timestamp()

    pivot_signals = monthly_signals.pivot(
        index="Date", columns="product_code", values="wow_growth"
    )

    pivot_signals = pivot_signals.sort_index()

    result = monthly_df.join(pivot_signals, how="inner")

    return result


def optimize_weights_for_regime(
    df: pd.DataFrame, regime: str
) -> Tuple[np.ndarray, float, List[int]]:
    """
    Optimize product weights for a specific regime.

    Args:
        df: Dataset with product columns and inflation target
        regime: Regime to optimize for

    Returns:
        Tuple of (weights, optimal_MAE, product_codes)
    """
    regime_data = df[df["regime"] == regime].copy()

    product_cols = [c for c in regime_data.columns if c in HIGH_QUALITY_PRODUCTS]
    product_codes = [int(c) for c in product_cols]

    if len(regime_data) < 10:
        return (
            np.full(len(product_codes), 1.0 / len(product_codes))
            if product_codes
            else np.array([]),
            float("inf"),
            product_codes,
        )

    X = regime_data[product_cols].values
    y = regime_data["inflation"].values

    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]

    if len(X) == 0 or len(product_cols) == 0:
        return np.array([]), float("inf"), product_codes

    def objective(weights):
        signal = np.dot(X, weights)
        mae = np.mean(np.abs(signal - y))
        return mae

    n_products = len(product_cols)
    bounds = [(0.0, 1.0) for _ in range(n_products)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    initial_weights = np.ones(n_products) / n_products

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500},
    )

    if result.success:
        return result.x, result.fun, product_codes
    else:
        return initial_weights, objective(initial_weights), product_codes


def compute_adaptive_prediction(
    df: pd.DataFrame, regime_weights_data: Dict[str, Tuple[np.ndarray, List[int]]]
) -> pd.Series:
    """
    Compute predictions using adaptive weights based on regime.

    Args:
        df: Dataset with product columns and regime
        regime_weights_data: Dict mapping regime -> (weights, product_codes)

    Returns:
        Series with adaptive predictions
    """
    product_cols = [c for c in df.columns if c in HIGH_QUALITY_PRODUCTS]
    X = df[product_cols].values

    predictions = np.zeros(len(df))

    for idx in range(len(df)):
        regime = df.iloc[idx]["regime"]
        weights_info = regime_weights_data.get(regime)

        if weights_info is None:
            weights_info = regime_weights_data.get("normal")

        if weights_info is not None:
            weights, codes = weights_info

            if len(weights) == 0:
                continue

            code_to_idx = {code: i for i, code in enumerate(codes)}
            row_signal = X[idx, :]

            valid_products = []
            valid_weights = []

            for col_idx, product_code in enumerate(product_cols):
                if product_code in code_to_idx:
                    weight_idx = code_to_idx[product_code]
                    if not np.isnan(row_signal[col_idx]) and not np.isnan(
                        weights[weight_idx]
                    ):
                        valid_products.append(row_signal[col_idx])
                        valid_weights.append(weights[weight_idx])

            if valid_products and sum(valid_weights) > 0:
                valid_weights = np.array(valid_weights) / sum(valid_weights)
                predictions[idx] = np.dot(valid_products, valid_weights)

    return pd.Series(predictions, index=df.index)


def compute_fixed_prediction(
    df: pd.DataFrame, weights_info: Tuple[np.ndarray, List[int]]
) -> pd.Series:
    """
    Compute predictions using fixed weights.

    Args:
        df: Dataset with product columns
        weights_info: Tuple of (weights, product_codes)

    Returns:
        Series with fixed predictions
    """
    weights, codes = weights_info
    product_cols = [c for c in df.columns if c in HIGH_QUALITY_PRODUCTS]
    X = df[product_cols].values

    predictions = np.zeros(len(df))

    if len(weights) == 0:
        return pd.Series(predictions, index=df.index)

    code_to_idx = {code: i for i, code in enumerate(codes)}

    for idx in range(len(df)):
        row_signal = X[idx, :]

        valid_products = []
        valid_weights = []

        for col_idx, product_code in enumerate(product_cols):
            if product_code in code_to_idx:
                weight_idx = code_to_idx[product_code]
                if not np.isnan(row_signal[col_idx]) and not np.isnan(
                    weights[weight_idx]
                ):
                    valid_products.append(row_signal[col_idx])
                    valid_weights.append(weights[weight_idx])

        if valid_products and sum(valid_weights) > 0:
            valid_weights = np.array(valid_weights) / sum(valid_weights)
            predictions[idx] = np.dot(valid_products, valid_weights)

    return pd.Series(predictions, index=df.index)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Regime-Dependent Weekly Weights for CPI Nowcasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/weekly_regime_weights.py
  python3 scripts/weekly_regime_weights.py --input_file data/weekly_prices.csv --output_dir data/

Outputs:
  - data/weekly_regime_weights.csv      - Optimal weights per regime
  - data/weekly_regime_backtest.csv    - Backtest comparison
        """,
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/weekly_prices.csv",
        help="Path to weekly prices CSV file (default: data/weekly_prices.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/",
        help="Output directory for results (default: data/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Regime-Dependent Weekly Weights for CPI Nowcasting")
    print("=" * 70)
    print(f"\nInput file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")

    print("\n📂 Loading data...")
    weekly_df = load_weekly_prices()
    print(f"   Loaded {len(weekly_df):,} weekly price observations")

    monthly_cpi = load_monthly_cpi()
    print(f"   Loaded {len(monthly_cpi):,} monthly CPI observations")

    enhanced_cpi = load_enhanced_cpi()
    print(f"   Loaded enhanced CPI with macro indicators")

    print("\n🔍 Detecting regimes...")
    monthly_cpi_with_regime = detect_regimes_for_months(enhanced_cpi)

    regime_counts = monthly_cpi_with_regime["regime"].value_counts()
    print(f"   Regime distribution:")
    for regime, count in regime_counts.items():
        print(
            f"     - {regime}: {count} months ({count / len(monthly_cpi_with_regime) * 100:.1f}%)"
        )

    print("\n🔗 Preparing weekly-monthly merged dataset...")
    merged_df = prepare_weekly_monthly_dataset(weekly_df, monthly_cpi_with_regime)
    print(f"   Merged dataset: {len(merged_df)} months")

    product_cols = [c for c in merged_df.columns if c in HIGH_QUALITY_PRODUCTS]
    print(f"   Products: {len(product_cols)}")

    print("\n⚙️  Optimizing weights per regime...")
    regime_weights_data = {}
    regime_maes = {}

    for regime in RegimeType:
        regime_name = regime.value
        print(f"   Optimizing for {regime_name}...")

        weights, mae, codes = optimize_weights_for_regime(merged_df, regime_name)
        regime_weights_data[regime_name] = (weights, codes)
        regime_maes[regime_name] = mae

        print(f"     Optimal MAE: {mae:.4f}%")

    print("\n💾 Saving regime weights...")
    weights_data = []
    for regime, (weights, codes) in regime_weights_data.items():
        for idx, product_code in enumerate(codes):
            if product_code in HIGH_QUALITY_PRODUCTS:
                product_name = HIGH_QUALITY_PRODUCTS[product_code]["name"]
                weights_data.append(
                    {
                        "Regime": regime,
                        "Product_code": product_code,
                        "Product_name": product_name,
                        "Weight": weights[idx],
                    }
                )

    weights_df = pd.DataFrame(weights_data)
    weights_path = Path(args.output_dir) / "weekly_regime_weights.csv"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_df.to_csv(weights_path, index=False)
    print(f"   Saved to {weights_path}")

    print("\n🧪 Running backtest...")
    split_idx = int(len(merged_df) * 0.7)
    train_df = merged_df.iloc[:split_idx]
    test_df = merged_df.iloc[split_idx:]

    print(
        f"   Training period: {train_df.index[0].date()} to {train_df.index[-1].date()}"
    )
    print(f"   Test period: {test_df.index[0].date()} to {test_df.index[-1].date()}")

    train_codes = [c for c in product_cols if isinstance(c, int)]
    baseline_weights = (np.ones(len(train_codes)) / len(train_codes), train_codes)

    fixed_train_pred = compute_fixed_prediction(train_df, baseline_weights)
    adaptive_train_pred = compute_adaptive_prediction(train_df, regime_weights_data)

    fixed_test_pred = compute_fixed_prediction(test_df, baseline_weights)
    adaptive_test_pred = compute_adaptive_prediction(test_df, regime_weights_data)

    fixed_train_mae = np.mean(np.abs(fixed_train_pred - train_df["inflation"]))
    adaptive_train_mae = np.mean(np.abs(adaptive_train_pred - train_df["inflation"]))

    fixed_test_mae = np.mean(np.abs(fixed_test_pred - test_df["inflation"]))
    adaptive_test_mae = np.mean(np.abs(adaptive_test_pred - test_df["inflation"]))

    print(f"\n   Training MAE:")
    print(f"     Fixed weights: {fixed_train_mae:.4f}%")
    print(f"     Adaptive weights: {adaptive_train_mae:.4f}%")
    print(
        f"     Improvement: {(fixed_train_mae - adaptive_train_mae) / fixed_train_mae * 100:.2f}%"
    )

    print(f"\n   Test MAE:")
    print(f"     Fixed weights: {fixed_test_mae:.4f}%")
    print(f"     Adaptive weights: {adaptive_test_mae:.4f}%")
    print(
        f"     Improvement: {(fixed_test_mae - adaptive_test_mae) / fixed_test_mae * 100:.2f}%"
    )

    backtest_data = {
        "Date": test_df.index,
        "Regime": test_df["regime"].values,
        "Fixed_pred": fixed_test_pred.values,
        "Adaptive_pred": adaptive_test_pred.values,
        "Actual": test_df["inflation"].values,
    }

    backtest_df = pd.DataFrame(backtest_data)
    backtest_path = Path(args.output_dir) / "weekly_regime_backtest.csv"
    backtest_path.parent.mkdir(parents=True, exist_ok=True)
    backtest_df.to_csv(backtest_path, index=False)
    print(f"\n💾 Backtest saved to {backtest_path}")

    mae_by_regime = backtest_df.groupby("Regime").apply(
        lambda g: pd.Series(
            {
                "Fixed_MAE": np.mean(np.abs(g["Fixed_pred"] - g["Actual"])),
                "Adaptive_MAE": np.mean(np.abs(g["Adaptive_pred"] - g["Actual"])),
                "Improvement_pct": (
                    np.mean(np.abs(g["Fixed_pred"] - g["Actual"]))
                    - np.mean(np.abs(g["Adaptive_pred"] - g["Actual"]))
                )
                / np.mean(np.abs(g["Fixed_pred"] - g["Actual"]))
                * 100,
                "n_obs": len(g),
            }
        )
    )

    print("\n📊 MAE by Regime (Test Period):")
    print(mae_by_regime.to_string())

    print("\n🎯 Key Findings:")
    print(
        f"   - Adaptive weights {'IMPROVED' if adaptive_test_mae < fixed_test_mae else 'DID NOT IMPROVE'} overall accuracy"
    )
    if adaptive_test_mae < fixed_test_mae:
        print(
            f"   - Test MAE reduction: {(fixed_test_mae - adaptive_test_mae):.4f}% points"
        )

    best_regime = mae_by_regime["Improvement_pct"].idxmax()
    worst_regime = mae_by_regime["Improvement_pct"].idxmin()
    print(
        f"   - Best performing regime: {best_regime} ({mae_by_regime.loc[best_regime, 'Improvement_pct']:.1f}% improvement)"
    )
    print(
        f"   - Worst performing regime: {worst_regime} ({mae_by_regime.loc[worst_regime, 'Improvement_pct']:.1f}% improvement)"
    )

    summary = {
        "train_fixed_mae": float(fixed_train_mae),
        "train_adaptive_mae": float(adaptive_train_mae),
        "test_fixed_mae": float(fixed_test_mae),
        "test_adaptive_mae": float(adaptive_test_mae),
        "test_improvement_pct": float(
            (fixed_test_mae - adaptive_test_mae) / fixed_test_mae * 100
        ),
        "best_regime": str(best_regime),
        "worst_regime": str(worst_regime),
    }

    print(f"\n📋 Summary:")
    for k, v in summary.items():
        print(f"   {k}: {v}")

    return summary


if __name__ == "__main__":
    summary = main()
    print("\n✅ Analysis complete!")
