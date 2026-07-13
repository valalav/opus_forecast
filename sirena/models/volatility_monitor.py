"""
Volatility Monitor for Weekly Price Anomaly Detection
=======================================================

Monitors weekly price changes for anomalies and unusual volatility patterns.
Provides alerts for significant price movements that may impact CPI forecasts.

Key functionality:
- Calculate rolling volatility per product
- Detect anomalous price changes (>2σ from historical norm)
- Generate alerts for unusual market activity
- Track seasonal volatility patterns

Training period: 2016-2026 (post-Crimea, stable methodology)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

try:
    from ..data.weekly_loader import (
        load_weekly_prices,
        HIGH_QUALITY_PRODUCTS,
        calculate_product_volatility,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from sirena.data.weekly_loader import (
        load_weekly_prices,
        HIGH_QUALITY_PRODUCTS,
        calculate_product_volatility,
    )


class VolatilityMonitor:
    """
    Monitors weekly price volatility and detects anomalies.

    Features:
    - Rolling volatility calculation per product
    - Z-score based anomaly detection
    - Seasonal pattern detection
    - Real-time alerting

    Alert Levels:
    - WARNING: 2-3 standard deviations
    - CRITICAL: >3 standard deviations
    """

    ALERT_LEVELS = {
        "normal": (0, 2),
        "warning": (2, 3),
        "critical": (3, float("inf")),
    }

    def __init__(
        self,
        lookback_weeks: int = 52,
        warning_threshold: float = 1.5,
        critical_threshold: float = 2.5,
    ):
        """
        Initialize volatility monitor.

        Args:
            lookback_weeks: Weeks of history for volatility calculation
            warning_threshold: Z-score threshold for warnings (optimal: 1.5 based on F1 tuning)
            critical_threshold: Z-score threshold for critical alerts (optimal: 2.5)
        """
        self.lookback_weeks = lookback_weeks
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self.volatility_stats = {}
        self.alerts = []
        self._is_initialized = False

    def _calculate_rolling_stats(
        self,
        weekly_df: pd.DataFrame,
        product_code: int,
    ) -> Dict[str, float]:
        """
        Calculate rolling statistics for a product.

        Args:
            weekly_df: Weekly prices DataFrame
            product_code: Product to analyze

        Returns:
            Dict with mean, std, min, max, current, z_score
        """
        product_df = weekly_df[weekly_df["product_code"] == product_code].copy()

        if len(product_df) < 10:
            return None

        product_df = product_df.sort_values("date")

        # Historical stats (excluding last 4 weeks for anomaly detection)
        historical = product_df.iloc[:-4] if len(product_df) > 4 else product_df

        mean_wow = historical["wow_growth"].mean()
        std_wow = historical["wow_growth"].std()

        # Recent stats
        recent = product_df.tail(4)
        current_wow = recent["wow_growth"].mean()

        # Z-score
        z_score = (current_wow - mean_wow) / std_wow if std_wow > 0 else 0.0

        return {
            "mean": mean_wow,
            "std": std_wow,
            "min": historical["wow_growth"].min(),
            "max": historical["wow_growth"].max(),
            "current": current_wow,
            "z_score": z_score,
            "last_date": product_df["date"].max(),
        }

    def initialize(
        self,
        start_date: str = "2016-01-01",
    ) -> "VolatilityMonitor":
        """
        Initialize volatility baselines for all products.

        Args:
            start_date: Start date for historical data

        Returns:
            self for method chaining
        """
        weekly_df = load_weekly_prices(start_date=start_date)

        for code in weekly_df["product_code"].unique():
            stats = self._calculate_rolling_stats(weekly_df, code)
            if stats:
                product_name = weekly_df[weekly_df["product_code"] == code][
                    "product_name"
                ].iloc[0]
                self.volatility_stats[code] = {
                    "name": product_name,
                    **stats,
                }

        self._is_initialized = True
        return self

    def check_anomalies(
        self,
        lookback_weeks: int = 4,
    ) -> List[Dict]:
        """
        Check for current anomalies across all products.

        Args:
            lookback_weeks: Recent weeks to analyze

        Returns:
            List of anomalies with details
        """
        if not self._is_initialized:
            self.initialize()

        anomalies = []

        weekly_df = load_weekly_prices()
        cutoff = weekly_df["date"].max() - pd.Timedelta(weeks=lookback_weeks)
        recent = weekly_df[weekly_df["date"] >= cutoff]

        for code in recent["product_code"].unique():
            if code not in self.volatility_stats:
                continue

            baseline = self.volatility_stats[code]
            product_recent = recent[recent["product_code"] == code]

            if len(product_recent) == 0:
                continue

            # Calculate current average growth
            current_growth = product_recent["wow_growth"].mean()
            z_score = (
                (current_growth - baseline["mean"]) / baseline["std"]
                if baseline["std"] > 0
                else 0
            )

            # Check alert level
            abs_z = abs(z_score)

            if abs_z >= self.critical_threshold:
                level = "critical"
            elif abs_z >= self.warning_threshold:
                level = "warning"
            else:
                continue

            anomalies.append(
                {
                    "product_code": code,
                    "product_name": baseline["name"],
                    "current_growth": round(current_growth, 2),
                    "historical_mean": round(baseline["mean"], 2),
                    "z_score": round(z_score, 2),
                    "level": level,
                    "direction": "up" if z_score > 0 else "down",
                    "last_date": str(product_recent["date"].max().date()),
                }
            )

        # Sort by absolute z-score
        anomalies = sorted(anomalies, key=lambda x: abs(x["z_score"]), reverse=True)
        self.alerts = anomalies

        return anomalies

    def get_volatility_summary(self) -> pd.DataFrame:
        """
        Get summary of volatility across all products.

        Returns:
            DataFrame with volatility stats per product
        """
        if not self._is_initialized:
            self.initialize()

        rows = []
        for code, stats in self.volatility_stats.items():
            rows.append(
                {
                    "product_code": code,
                    "product_name": stats["name"],
                    "mean_wow": round(stats["mean"], 3),
                    "std_wow": round(stats["std"], 3),
                    "current_wow": round(stats["current"], 3),
                    "z_score": round(stats["z_score"], 2),
                    "volatility_tier": self._classify_volatility(stats["std"]),
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("std_wow", ascending=False)

        return df

    def _classify_volatility(self, std: float) -> str:
        """Classify product by volatility."""
        if std < 2.0:
            return "stable"
        elif std < 5.0:
            return "medium"
        elif std < 10.0:
            return "volatile"
        else:
            return "ultra_volatile"

    def get_seasonal_patterns(
        self,
        product_code: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Analyze seasonal volatility patterns.

        Args:
            product_code: Specific product to analyze (None = all high-quality)

        Returns:
            DataFrame with seasonal patterns by month
        """
        weekly_df = load_weekly_prices()

        if product_code:
            weekly_df = weekly_df[weekly_df["product_code"] == product_code]
        else:
            # Use high-quality products only
            hq_codes = list(HIGH_QUALITY_PRODUCTS.keys())
            weekly_df = weekly_df[weekly_df["product_code"].isin(hq_codes)]

        weekly_df["month"] = weekly_df["date"].dt.month

        # Calculate stats by month
        monthly_stats = (
            weekly_df.groupby("month")
            .agg(
                {
                    "wow_growth": ["mean", "std", "min", "max", "count"],
                }
            )
            .round(3)
        )

        monthly_stats.columns = ["mean", "std", "min", "max", "count"]

        # Add month names
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        monthly_stats["month_name"] = [month_names[i - 1] for i in monthly_stats.index]

        return monthly_stats

    def generate_report(self) -> str:
        """
        Generate volatility monitoring report.

        Returns:
            Markdown formatted report
        """
        anomalies = self.check_anomalies()
        summary = self.get_volatility_summary()

        report = []
        report.append("# Weekly Price Volatility Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Alert summary
        critical = [a for a in anomalies if a["level"] == "critical"]
        warnings = [a for a in anomalies if a["level"] == "warning"]

        report.append(f"\n## Alerts")
        report.append(f"- Critical: {len(critical)}")
        report.append(f"- Warning: {len(warnings)}")

        if critical:
            report.append("\n### Critical Alerts")
            for a in critical:
                report.append(
                    f"- **{a['product_name'][:40]}**: "
                    f"z={a['z_score']:+.1f} ({a['direction']}) "
                    f"[{a['current_growth']:+.2f}% vs hist {a['historical_mean']:.2f}%]"
                )

        if warnings:
            report.append("\n### Warnings")
            for a in warnings[:10]:  # Top 10 warnings
                report.append(
                    f"- {a['product_name'][:40]}: "
                    f"z={a['z_score']:+.1f} ({a['direction']})"
                )

        # Volatility distribution
        report.append("\n## Volatility Distribution")
        tier_counts = summary["volatility_tier"].value_counts()
        for tier, count in tier_counts.items():
            report.append(f"- {tier}: {count} products")

        # Top volatile products
        report.append("\n## Most Volatile Products (Top 10)")
        for _, row in summary.head(10).iterrows():
            report.append(
                f"- {row['product_name'][:40]}: "
                f"std={row['std_wow']:.2f}%, current={row['current_wow']:+.2f}%"
            )

        return "\n".join(report)


def run_monitoring():
    """Run volatility monitoring."""
    print("=" * 60)
    print("Weekly Price Volatility Monitor")
    print("=" * 60)

    monitor = VolatilityMonitor()
    monitor.initialize()

    # Check anomalies
    anomalies = monitor.check_anomalies()

    critical = [a for a in anomalies if a["level"] == "critical"]
    warnings = [a for a in anomalies if a["level"] == "warning"]

    print(f"\nAlerts: {len(critical)} critical, {len(warnings)} warning")

    if critical:
        print("\n🚨 Critical Alerts:")
        for a in critical[:5]:
            print(
                f"  {a['product_name'][:40]:<40} | "
                f"z={a['z_score']:+.1f} | "
                f"current={a['current_growth']:+.2f}% | "
                f"hist={a['historical_mean']:.2f}%"
            )

    if warnings:
        print("\n⚠️ Warnings:")
        for a in warnings[:5]:
            print(f"  {a['product_name'][:40]:<40} | z={a['z_score']:+.1f}")

    # Volatility summary
    summary = monitor.get_volatility_summary()
    print("\nVolatility Distribution:")
    tier_counts = summary["volatility_tier"].value_counts()
    for tier in ["stable", "medium", "volatile", "ultra_volatile"]:
        count = tier_counts.get(tier, 0)
        print(f"  {tier:<15}: {count} products")

    # Seasonal patterns
    print("\nSeasonal Patterns (High-Quality Products):")
    seasonal = monitor.get_seasonal_patterns()
    print(seasonal[["month_name", "mean", "std"]].to_string())

    return monitor


if __name__ == "__main__":
    run_monitoring()
