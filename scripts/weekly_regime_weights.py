#!/usr/bin/env python3
"""
Regime-Dependent Weights for Weekly Nowcasting.
Tooling script with RegimeWeighter class and argparse support.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from sirena.models.regime_detector import detect_regime, MacroRegime

HIGH_QUALITY_PRODUCTS = {
    111: "Говядина",
    114: "Куры",
    701: "Масло сливочное",
    1501: "Яйца",
    1601: "Сахар",
    2501: "Картофель",
    2601: "Капуста",
    3201: "Лук репчатый",
    3301: "Яблоки",
    3501: "Виноград",
    4001: "Свекла",
    4501: "Огурцы и помидоры",
    4801: "Масло подсолнечное",
    5001: "Сметана",
    5401: "Сыры",
    5601: "Молоко питьевое",
    5801: "Хлеб из пшеничной муки",
    6201: "Хлеб из ржаной муки",
    6401: "Мука пшеничная",
    7802: "Бензин АИ-92",
    7901: "Бензин АИ-95",
}


class RegimeWeighter:
    """Class for regime-dependent weight optimization for weekly nowcasting."""

    def __init__(self, input_file=None, output_dir=None):
        """
        Initialize RegimeWeighter.

        Args:
            input_file: Path to weekly prices CSV (default: data/weekly_prices.csv)
            output_dir: Path to output directory (default: data/)
        """
        if input_file is None:
            input_file = Path(__file__).parent.parent / "data" / "weekly_prices.csv"
        else:
            input_file = Path(input_file)

        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "data"
        else:
            output_dir = Path(output_dir)

        self.input_file = input_file
        self.output_dir = output_dir
        self.weekly_prices = None
        self.monthly_cpi = None
        self.weekly_cpi_df = None
        self.monthly_prices = None
        self.regime_weights = {}
        self.product_cols = []
        self.metrics = {}

    def load_weekly_prices(self):
        """Load weekly price data."""
        if not self.input_file.exists():
            # Fallback to known file
            self.input_file = (
                Path(__file__).parent.parent
                / "data"
                / "kbr_weekly_prices_2008_2026.csv"
            )

        self.weekly_prices = pd.read_csv(self.input_file)
        self.weekly_prices["date"] = pd.to_datetime(self.weekly_prices["date"])
        return self.weekly_prices

    def load_monthly_data(self):
        """Load monthly CPI and macro data."""
        data_dir = Path(__file__).parent.parent / "data"
        df = pd.read_csv(data_dir / "inflation_data.csv", sep=";", decimal=",")
        for col in df.columns:
            if col != "Date" and df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
        df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
        self.monthly_cpi = df.set_index("Date").sort_index()
        if "Ki_i" in self.monthly_cpi.columns:
            self.monthly_cpi["Ki"] = self.monthly_cpi["Ki_i"] / 100
        if "Ruonia" in self.monthly_cpi.columns:
            self.monthly_cpi["Ruonia"] = self.monthly_cpi["Ruonia_i"] / 100
        return self.monthly_cpi

    def load_weekly_cpi(self):
        """Load weekly CPI data."""
        data_dir = Path(__file__).parent.parent / "data"
        df = pd.read_csv(data_dir / "kbr_weekly_cpi_2008_2026.csv")
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M")
        monthly = df.groupby("month").agg({"weekly_inflation_pct": "sum"}).reset_index()
        monthly["date"] = monthly["month"].dt.to_timestamp()
        self.weekly_cpi_df = monthly.set_index("date")
        return self.weekly_cpi_df

    def aggregate_weekly_to_monthly(self, weekly_df):
        """Aggregate weekly prices to monthly by product."""
        monthly_data = []
        for product_code in HIGH_QUALITY_PRODUCTS.keys():
            product_df = weekly_df[weekly_df["product_code"] == product_code].copy()
            if len(product_df) < 10:
                continue
            product_df["month"] = product_df["date"].dt.to_period("M")
            monthly = (
                product_df.groupby("month").agg({"wow_growth": "mean"}).reset_index()
            )
            monthly["date"] = monthly["month"].dt.to_timestamp()
            monthly["product_code"] = product_code
            monthly["product_name"] = HIGH_QUALITY_PRODUCTS[product_code]
            monthly_data.append(monthly)
        if not monthly_data:
            return pd.DataFrame()
        result = pd.concat(monthly_data, ignore_index=True)
        return result.pivot(index="date", columns="product_code", values="wow_growth")

    def get_regime_for_month(self, monthly_cpi, date):
        """Get regime for a given month."""
        month_start = date.replace(day=1)
        if month_start not in monthly_cpi.index:
            available_dates = monthly_cpi.index[monthly_cpi.index <= month_start]
            if len(available_dates) == 0:
                return MacroRegime.NORMAL
            month_start = available_dates[-1]
        df_slice = monthly_cpi.loc[:month_start]
        if len(df_slice) < 12:
            return MacroRegime.NORMAL
        regime, _ = detect_regime(df_slice)
        return regime

    def compute_weights(self, X, y, n_products, regime_name):
        """Compute optimal weights for a specific regime using MAE minimization."""
        n_samples = len(y)
        if n_samples < 5:
            return np.ones(n_products) / n_products

        def objective(weights):
            weighted_pred = X @ weights
            mae = np.mean(np.abs(weighted_pred - y))
            return mae

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_products)]
        x0 = np.ones(n_products) / n_products

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "disp": False},
        )

        if result.success:
            return result.x
        else:
            print(
                f"Warning: Optimization failed for {regime_name}, using equal weights"
            )
            return np.ones(n_products) / n_products

    def optimize_weights_for_regime(self, X, y, n_products, regime_name):
        """Optimize weights for a specific regime using MAE minimization."""
        return self.compute_weights(X, y, n_products, regime_name)

    def backtest_regime_adaptive(self):
        """Run backtest with regime-adaptive weights."""
        if self.monthly_prices is None:
            raise ValueError("Monthly prices not loaded. Call prepare_data() first.")

        start_date = pd.Timestamp("2016-01-01")
        end_date = pd.Timestamp("2022-12-31")
        train_mask = (self.monthly_prices.index >= start_date) & (
            self.monthly_prices.index <= end_date
        )
        train_dates = self.monthly_prices.index[train_mask]

        test_start = pd.Timestamp("2023-01-01")
        test_end = pd.Timestamp("2025-12-31")
        test_mask = (self.monthly_prices.index >= test_start) & (
            self.monthly_prices.index <= test_end
        )
        test_dates = self.monthly_prices.index[test_mask]

        print(
            f"Training period: {start_date} to {end_date} ({len(train_dates)} months)"
        )
        print(f"Test period: {test_start} to {test_end} ({len(test_dates)} months)")

        valid_products = []
        for col in self.monthly_prices.columns:
            missing_pct = self.monthly_prices[col].isna().sum() / len(
                self.monthly_prices
            )
            if missing_pct < 0.3 and col in HIGH_QUALITY_PRODUCTS:
                valid_products.append(col)

        print(f"Valid products for optimization: {len(valid_products)}")

        if len(valid_products) == 0:
            raise ValueError("No valid products found")

        self.product_cols = valid_products

        common_train_dates = train_dates.intersection(self.weekly_cpi_df.index)
        common_test_dates = test_dates.intersection(self.weekly_cpi_df.index)

        print(f"Common train dates: {len(common_train_dates)}")
        print(f"Common test dates: {len(common_test_dates)}")

        X_train = self.monthly_prices.loc[common_train_dates, self.product_cols].values
        y_train = self.weekly_cpi_df.loc[
            common_train_dates, "weekly_inflation_pct"
        ].values

        valid_idx = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
        train_dates_clean = common_train_dates[valid_idx]

        print(f"Training samples after cleaning: {len(X_train)}")

        regimes = []
        for date in train_dates_clean:
            regime = self.get_regime_for_month(self.monthly_cpi, date)
            regimes.append(regime)

        regimes = np.array(regimes)

        print("\nOptimizing fixed weights...")
        fixed_weights = self.compute_weights(
            X_train, y_train, len(self.product_cols), "fixed"
        )

        for regime in [
            MacroRegime.NORMAL,
            MacroRegime.SHOCK,
            MacroRegime.HIGH_INFLATION,
        ]:
            regime_mask = regimes == regime
            if regime_mask.sum() >= 3:
                X_regime = X_train[regime_mask]
                y_regime = y_train[regime_mask]
                weights = self.compute_weights(
                    X_regime, y_regime, len(self.product_cols), regime.value
                )
                self.regime_weights[regime.value] = weights
                print(
                    f"  {regime.value}: {len(X_regime)} samples, top 3 weights: {np.argsort(weights)[-3:]}"
                )
            else:
                print(
                    f"  {regime.value}: insufficient data ({regime_mask.sum()} samples), using equal weights"
                )
                self.regime_weights[regime.value] = np.ones(
                    len(self.product_cols)
                ) / len(self.product_cols)

        print("\nRunning backtest...")
        results = []

        for date in common_test_dates:
            regime = self.get_regime_for_month(self.monthly_cpi, date)

            if date not in self.monthly_prices.index:
                continue

            X_row = self.monthly_prices.loc[[date], self.product_cols].values

            if np.isnan(X_row).any():
                continue

            try:
                actual = self.weekly_cpi_df.loc[date, "weekly_inflation_pct"]
            except KeyError:
                continue

            fixed_pred = (X_row @ fixed_weights)[0]
            adaptive_weights = self.regime_weights.get(regime.value, fixed_weights)
            adaptive_pred = (X_row @ adaptive_weights)[0]

            results.append(
                {
                    "Date": date,
                    "Regime": regime.value,
                    "Fixed_pred": fixed_pred,
                    "Adaptive_pred": adaptive_pred,
                    "Actual": actual,
                }
            )

        if len(results) == 0:
            raise ValueError("No test results generated")

        results_df = pd.DataFrame(results)

        fixed_mae = np.mean(np.abs(results_df["Fixed_pred"] - results_df["Actual"]))
        adaptive_mae = np.mean(
            np.abs(results_df["Adaptive_pred"] - results_df["Actual"])
        )
        improvement = (fixed_mae - adaptive_mae) / fixed_mae * 100

        print(f"\n=== BACKTEST RESULTS ===")
        print(f"Fixed Weights MAE: {fixed_mae:.4f}%")
        print(f"Adaptive Weights MAE: {adaptive_mae:.4f}%")
        print(f"Improvement: {improvement:.2f}%")

        print(f"\n=== MAE BY REGIME ===")
        for regime in ["normal", "shock", "high_inflation"]:
            regime_mask = results_df["Regime"] == regime
            if regime_mask.sum() > 0:
                regime_fixed_mae = np.mean(
                    np.abs(
                        results_df[regime_mask]["Fixed_pred"]
                        - results_df[regime_mask]["Actual"]
                    )
                )
                regime_adaptive_mae = np.mean(
                    np.abs(
                        results_df[regime_mask]["Adaptive_pred"]
                        - results_df[regime_mask]["Actual"]
                    )
                )
                regime_improvement = (
                    (regime_fixed_mae - regime_adaptive_mae) / regime_fixed_mae * 100
                    if regime_fixed_mae > 0
                    else 0
                )
                print(
                    f"{regime}: Fixed={regime_fixed_mae:.4f}%, Adaptive={regime_adaptive_mae:.4f}%, Improvement={regime_improvement:.2f}% (n={regime_mask.sum()})"
                )

        print(f"\n=== TRAINING METRICS ===")
        train_fixed_pred = X_train @ fixed_weights
        train_adaptive_pred = np.zeros_like(y_train)
        for i, regime in enumerate(regimes):
            train_adaptive_pred[i] = X_train[i] @ self.regime_weights[regime.value]

        train_fixed_mae = np.mean(np.abs(train_fixed_pred - y_train))
        train_adaptive_mae = np.mean(np.abs(train_adaptive_pred - y_train))
        train_improvement = (
            (train_fixed_mae - train_adaptive_mae) / train_fixed_mae * 100
        )

        print(f"Training Fixed MAE: {train_fixed_mae:.4f}%")
        print(f"Training Adaptive MAE: {train_adaptive_mae:.4f}%")
        print(f"Training Improvement: {train_improvement:.2f}%")

        self.metrics = {
            "test_fixed_mae": fixed_mae,
            "test_adaptive_mae": adaptive_mae,
            "test_improvement": improvement,
            "train_fixed_mae": train_fixed_mae,
            "train_adaptive_mae": train_adaptive_mae,
            "train_improvement": train_improvement,
        }

        return results_df

    def prepare_data(self):
        """Load and prepare all required data."""
        print("Loading data...")
        self.load_weekly_prices()
        print(f"   Weekly prices: {len(self.weekly_prices)} rows")

        self.load_monthly_data()
        print(
            f"   Monthly CPI: {self.monthly_cpi.index.min()} to {self.monthly_cpi.index.max()}"
        )

        self.load_weekly_cpi()
        print(f"   Weekly CPI: {len(self.weekly_cpi_df)} rows")

        print("\nAggregating weekly prices to monthly...")
        self.monthly_prices = self.aggregate_weekly_to_monthly(self.weekly_prices)
        print(f"   Monthly prices: {self.monthly_prices.shape}")
        print(
            f"   Date range: {self.monthly_prices.index.min()} to {self.monthly_prices.index.max()}"
        )

    def run(self):
        """Run full analysis pipeline."""
        print("=" * 70)
        print("Regime-Dependent Weights for Weekly Nowcasting")
        print("=" * 70)

        self.prepare_data()

        print("\nRunning regime-dependent weight optimization...")
        results_df = self.backtest_regime_adaptive()

        print("\nSaving results...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results_path = self.output_dir / "weekly_regime_backtest.csv"
        results_df.to_csv(results_path, index=False)
        print(f"   Backtest results: {results_path}")

        weights_path = self.output_dir / "weekly_regime_weights.csv"
        weights_data = []
        for regime, weights in self.regime_weights.items():
            for i, product_code in enumerate(self.product_cols):
                weights_data.append(
                    {
                        "Regime": regime,
                        "Product_code": product_code,
                        "Product_name": HIGH_QUALITY_PRODUCTS.get(
                            product_code, "Unknown"
                        ),
                        "Weight": weights[i],
                    }
                )

        weights_df = pd.DataFrame(weights_data)
        weights_df.to_csv(weights_path, index=False)
        print(f"   Regime weights: {weights_path}")

        print("\nTop products by regime (weights):")
        for regime in ["normal", "shock", "high_inflation"]:
            regime_data = weights_df[weights_df["Regime"] == regime].sort_values(
                "Weight", ascending=False
            )
            print(f"\n   {regime.upper()} REGIME:")
            for _, row in regime_data.head(5).iterrows():
                print(f"      {row['Product_name']}: {row['Weight'] * 100:.1f}%")

        print("\n" + "=" * 70)
        print("✅ COMPLETE")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Test MAE improvement: {self.metrics['test_improvement']:.2f}%")
        print(f"  Training MAE improvement: {self.metrics['train_improvement']:.2f}%")
        print(f"  Regimes analyzed: {len(self.regime_weights)}")

        return results_df, weights_df


def main():
    """Main entry point with argparse support."""
    parser = argparse.ArgumentParser(
        description="Regime-Dependent Weights for Weekly Nowcasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/weekly_regime_weights.py
  python3 scripts/weekly_regime_weights.py --input_file data/custom_weekly.csv --output_dir data/results/
        """,
    )

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
        help="Path to output directory (default: data/)",
    )

    args = parser.parse_args()

    # Create and run RegimeWeighter
    weighter = RegimeWeighter(input_file=args.input_file, output_dir=args.output_dir)
    weighter.run()


if __name__ == "__main__":
    main()
