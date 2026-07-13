"""
Leading Indicator Detection for CPI Nowcasting
================================================

Identifies products that lead CPI changes by 2-4 weeks using Granger causality
tests and correlation analysis.

Key functionality:
- Granger causality tests: weekly product prices → monthly CPI
- Lead-lag correlation analysis
- Real-time signal generation from leading products

Training period: 2016-2026 (post-Crimea, stable methodology)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import warnings
from scipy import stats

warnings.filterwarnings("ignore")

try:
    from ..data.weekly_loader import (
        load_weekly_prices,
        HIGH_QUALITY_PRODUCTS,
        get_high_quality_products,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from sirena.data.weekly_loader import (
        load_weekly_prices,
        HIGH_QUALITY_PRODUCTS,
        get_high_quality_products,
    )


def granger_causality_test(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 4,
) -> Dict[str, float]:
    """
    Simplified Granger causality test.

    Tests if past values of x help predict y beyond past values of y alone.

    Args:
        x: Potential leading indicator series
        y: Target series (CPI)
        max_lag: Maximum lag to test

    Returns:
        Dict with best lag and p-value
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < max_lag + 10:
        return {'best_lag': None, 'p_value': 1.0, 'f_stat': 0.0}

    best_lag = 1
    best_improvement = 0.0
    best_f_stat = 0.0

    for lag in range(1, max_lag + 1):
        # Create lagged features
        n = len(y) - lag

        # Restricted model: y ~ y_lags only
        y_lags = np.column_stack([y[lag-i-1:-i-1] for i in range(min(lag, 3))])
        y_target = y[lag:]

        # Unrestricted model: y ~ y_lags + x_lags
        x_lags = np.column_stack([x[lag-i-1:-i-1] for i in range(min(lag, 3))])
        combined = np.hstack([y_lags, x_lags])

        # Fit models
        model_r = Ridge(alpha=1.0)
        model_r.fit(y_lags, y_target)
        sse_r = np.sum((y_target - model_r.predict(y_lags)) ** 2)

        model_u = Ridge(alpha=1.0)
        model_u.fit(combined, y_target)
        sse_u = np.sum((y_target - model_u.predict(combined)) ** 2)

        # F-statistic
        df1 = x_lags.shape[1]  # Added parameters
        df2 = n - combined.shape[1]  # Degrees of freedom

        if sse_u > 0 and df2 > 0:
            f_stat = ((sse_r - sse_u) / df1) / (sse_u / df2)
            improvement = (sse_r - sse_u) / sse_r if sse_r > 0 else 0

            if improvement > best_improvement:
                best_improvement = improvement
                best_lag = lag
                best_f_stat = f_stat

    # Calculate p-value
    if best_f_stat > 0:
        p_value = 1 - stats.f.cdf(best_f_stat, df1, df2)
    else:
        p_value = 1.0

    return {
        'best_lag': best_lag,
        'p_value': p_value,
        'f_stat': best_f_stat,
        'improvement': best_improvement,
    }


def calculate_lead_correlations(
    weekly_df: pd.DataFrame,
    monthly_cpi: pd.Series,
    product_code: int,
    max_lag_weeks: int = 8,
) -> Dict[str, float]:
    """
    Calculate correlations at different lead times.

    Args:
        weekly_df: Weekly prices DataFrame
        monthly_cpi: Monthly CPI series
        product_code: Product to analyze
        max_lag_weeks: Maximum lag in weeks to test

    Returns:
        Dict with correlation at each lag
    """
    # Filter to product
    product_df = weekly_df[weekly_df['product_code'] == product_code].copy()

    if len(product_df) < 20:
        return {'best_lag': None, 'best_corr': 0.0}

    # Aggregate to weekly MoM growth
    product_df = product_df.set_index('date')['wow_growth']

    # Resample monthly CPI to weekly for comparison
    monthly_cpi = monthly_cpi.resample('W').ffill()

    correlations = {}
    best_lag = 0
    best_corr = 0.0

    for lag in range(1, max_lag_weeks + 1):
        # Shift product prices forward (they lead CPI)
        shifted = product_df.shift(lag)

        # Align with CPI
        aligned = pd.concat([shifted, monthly_cpi], axis=1, join='inner')
        aligned.columns = ['product', 'cpi']
        aligned = aligned.dropna()

        if len(aligned) > 10:
            corr = aligned['product'].corr(aligned['cpi'])
            correlations[f'lag_{lag}w'] = corr

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

    return {
        'best_lag': best_lag,
        'best_corr': best_corr,
        'correlations': correlations,
    }


class LeadingIndicatorDetector:
    """
    Detects products that lead CPI changes.

    Methodology:
    1. For each product, compute weekly MoM growth
    2. Test Granger causality: product_growth → CPI
    3. Identify products with significant leading relationship
    4. Create real-time signals from leading products
    """

    def __init__(
        self,
        min_data_points: int = 50,
        significance_level: float = 0.10,
    ):
        """
        Initialize detector.

        Args:
            min_data_points: Minimum observations required for testing
            significance_level: P-value threshold for significance
        """
        self.min_data_points = min_data_points
        self.significance_level = significance_level
        self.leading_products = {}
        self._is_analyzed = False

    def _load_monthly_cpi(self) -> pd.Series:
        """Load monthly CPI series."""
        base_paths = [
            Path.cwd() / 'data' / 'inflation_data.csv',
            Path.cwd().parent / 'data' / 'inflation_data.csv',
            Path(__file__).parent.parent.parent / 'data' / 'inflation_data.csv',
        ]

        for path in base_paths:
            if path.exists():
                df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
                df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', dayfirst=True)
                df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
                df = df.set_index('Date').sort_index()

                # Convert mom to percentage
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].str.replace(',', '.').astype(float)

                return df['mom'] - 100  # Convert to percentage

        raise FileNotFoundError("Monthly CPI data not found")

    def analyze(
        self,
        start_date: str = '2016-01-01',
    ) -> pd.DataFrame:
        """
        Analyze all products for leading indicator potential.

        Args:
            start_date: Start date for analysis

        Returns:
            DataFrame with products ranked by leading indicator strength
        """
        # Load data
        weekly_df = load_weekly_prices(start_date=start_date)
        monthly_cpi = self._load_monthly_cpi()

        # Aggregate weekly prices to monthly
        weekly_df['year_month'] = weekly_df['date'].dt.to_period('M')

        results = []

        for code in weekly_df['product_code'].unique():
            product_df = weekly_df[weekly_df['product_code'] == code]
            product_name = product_df['product_name'].iloc[0]

            # Aggregate to monthly
            monthly_product = product_df.groupby('year_month')['wow_growth'].sum()
            monthly_product.index = monthly_product.index.to_timestamp()

            # Align with CPI
            aligned = pd.concat([monthly_product, monthly_cpi], axis=1, join='inner')
            aligned.columns = ['product', 'cpi']
            aligned = aligned.dropna()

            if len(aligned) < self.min_data_points:
                continue

            # Granger causality test
            granger = granger_causality_test(
                aligned['product'].values,
                aligned['cpi'].values,
                max_lag=4,
            )

            # Lead correlations
            lead_corrs = {}
            for lag in range(1, 5):
                shifted = aligned['product'].shift(lag)
                corr_data = pd.concat([shifted, aligned['cpi']], axis=1).dropna()
                if len(corr_data) > 10:
                    lead_corrs[lag] = corr_data.iloc[:, 0].corr(corr_data.iloc[:, 1])
                else:
                    lead_corrs[lag] = 0.0

            best_lag = max(lead_corrs, key=lambda x: abs(lead_corrs.get(x, 0)))
            best_corr = lead_corrs.get(best_lag, 0.0)

            results.append({
                'product_code': code,
                'product_name': product_name,
                'granger_lag': granger['best_lag'],
                'granger_pvalue': granger['p_value'],
                'granger_improvement': granger['improvement'],
                'lead_corr_1m': lead_corrs.get(1, 0.0),
                'lead_corr_2m': lead_corrs.get(2, 0.0),
                'best_lead_lag': best_lag,
                'best_lead_corr': best_corr,
                'n_observations': len(aligned),
            })

        # Create DataFrame and rank
        df = pd.DataFrame(results)
        df['is_significant'] = df['granger_pvalue'] < self.significance_level
        df['leading_score'] = (
            (1 - df['granger_pvalue']) * 0.5 +
            df['best_lead_corr'].abs() * 0.5
        )
        df = df.sort_values('leading_score', ascending=False)

        # Store significant leading products
        self.leading_products = {
            row['product_code']: {
                'name': row['product_name'],
                'lag': row['best_lead_lag'],
                'correlation': row['best_lead_corr'],
                'p_value': row['granger_pvalue'],
            }
            for _, row in df[df['is_significant']].iterrows()
        }

        self._is_analyzed = True
        return df

    def get_current_signal(
        self,
        lookback_weeks: int = 4,
    ) -> Dict[str, float]:
        """
        Get current leading indicator signal.

        Args:
            lookback_weeks: Weeks of data to use for signal

        Returns:
            Dict with signal value and components
        """
        if not self._is_analyzed:
            self.analyze()

        if not self.leading_products:
            return {'signal': 0.0, 'components': {}, 'n_products': 0}

        # Load recent weekly data
        weekly_df = load_weekly_prices()
        cutoff = weekly_df['date'].max() - pd.Timedelta(weeks=lookback_weeks)
        recent = weekly_df[weekly_df['date'] >= cutoff]

        signals = {}
        weights = {}

        for code, info in self.leading_products.items():
            product_df = recent[recent['product_code'] == code]

            if len(product_df) > 0:
                # Average WoW growth as signal
                signal = product_df['wow_growth'].mean()
                signals[code] = signal

                # Weight by inverse p-value (more significant = higher weight)
                weights[code] = 1 / (info['p_value'] + 0.01)

        if not signals:
            return {'signal': 0.0, 'components': {}, 'n_products': 0}

        # Weighted average signal
        total_weight = sum(weights.values())
        weighted_signal = sum(s * weights[c] / total_weight for c, s in signals.items())

        return {
            'signal': weighted_signal,
            'components': {
                self.leading_products[c]['name']: s
                for c, s in signals.items()
            },
            'n_products': len(signals),
            'confidence': 'high' if len(signals) >= 3 else 'medium',
        }

    def get_top_indicators(self, n: int = 10) -> List[Dict]:
        """
        Get top N leading indicators.

        Args:
            n: Number of indicators to return

        Returns:
            List of top indicators with their properties
        """
        if not self._is_analyzed:
            df = self.analyze()
        else:
            # Re-analyze to get fresh data
            df = self.analyze()

        top = df.head(n)

        return [
            {
                'code': row['product_code'],
                'name': row['product_name'],
                'lead_lag': row['best_lead_lag'],
                'correlation': round(row['best_lead_corr'], 3),
                'p_value': round(row['granger_pvalue'], 4),
                'is_significant': row['is_significant'],
            }
            for _, row in top.iterrows()
        ]


def run_analysis():
    """Run leading indicator analysis."""
    print("=" * 60)
    print("Leading Indicator Analysis")
    print("=" * 60)

    detector = LeadingIndicatorDetector(significance_level=0.10)
    results = detector.analyze()

    print(f"\nTotal products analyzed: {len(results)}")
    print(f"Significant leading indicators: {results['is_significant'].sum()}")

    print("\nTop 10 Leading Indicators:")
    print("-" * 60)

    for indicator in detector.get_top_indicators(10):
        status = "✓" if indicator['is_significant'] else " "
        print(f"[{status}] {indicator['name'][:40]:<40} | "
              f"lag={indicator['lead_lag']}m | "
              f"r={indicator['correlation']:+.3f} | "
              f"p={indicator['p_value']:.4f}")

    # Get current signal
    signal = detector.get_current_signal()
    print(f"\nCurrent Leading Signal: {signal['signal']:.3f}%")
    print(f"Products used: {signal['n_products']}")
    print(f"Confidence: {signal['confidence']}")

    return results


if __name__ == '__main__':
    run_analysis()
