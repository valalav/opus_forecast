"""
Weekly Price Data Loader for CPI Nowcasting
=============================================

Loads and preprocesses weekly price data from Rosstat.
Classifies products by quality tier and maps to CPI basket weights.

Data source: data/kbr_weekly_prices_2008_2026.csv
Training period: 2016-2026 (post-Crimea, stable methodology)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from functools import lru_cache


# High-quality products with <5% missing data
HIGH_QUALITY_PRODUCTS = {
    111: {
        "name": "Говядина (кроме бескостного мяса)",
        "weight": 0.0158,
        "category": "food",
    },
    114: {"name": "Куры охлажденные и мороженые", "weight": 0.0095, "category": "food"},
    202: {"name": "Сосиски, сардельки", "weight": 0.005, "category": "food"},
    204: {
        "name": "Колбаса полукопченая и варено-копченая",
        "weight": 0.005,
        "category": "food",
    },
    411: {"name": "Рыба мороженая неразделанная", "weight": 0.005, "category": "food"},
    701: {"name": "Масло сливочное", "weight": 0.0088, "category": "food"},
    1001: {"name": "Маргарин", "weight": 0.002, "category": "food"},
    1102: {"name": "Сметана", "weight": 0.0074, "category": "food"},
    1501: {"name": "Яйца куриные", "weight": 0.006, "category": "food"},
    1601: {"name": "Сахар-песок", "weight": 0.004, "category": "food"},
    1701: {"name": "Печенье", "weight": 0.003, "category": "food"},
    1903: {"name": "Чай черный байховый", "weight": 0.002, "category": "food"},
    2002: {"name": "Соль поваренная пищевая", "weight": 0.001, "category": "food"},
    2101: {"name": "Мука пшеничная", "weight": 0.003, "category": "food"},
    2201: {"name": "Хлеб из ржаной муки", "weight": 0.005, "category": "food"},
    2301: {"name": "Рис шлифованный", "weight": 0.003, "category": "food"},
    2303: {"name": "Пшено", "weight": 0.001, "category": "food"},
    2401: {"name": "Вермишель", "weight": 0.002, "category": "food"},
    2501: {"name": "Картофель", "weight": 0.004, "category": "food"},
    2601: {"name": "Капуста белокочанная свежая", "weight": 0.002, "category": "food"},
    7802: {"name": "Бензин АИ-92", "weight": 0.0101, "category": "nonfood"},
    7805: {"name": "Бензин АИ-95", "weight": 0.0139, "category": "nonfood"},
}

# Medium quality products (5-30% missing)
MEDIUM_QUALITY_PRODUCTS = {
    116: {"name": "Баранина", "weight": 0.002, "category": "food"},
    801: {"name": "Масло подсолнечное", "weight": 0.003, "category": "food"},
    1201: {
        "name": "Сыры сычужные твердые и мягкие",
        "weight": 0.005,
        "category": "food",
    },
    7800: {"name": "Бензин автомобильный", "weight": 0.024, "category": "nonfood"},
}

# Volatile products requiring special handling
VOLATILE_PRODUCTS = {
    2621: {"name": "Огурцы свежие", "category": "food", "seasonal": True},
    2623: {"name": "Помидоры свежие", "category": "food", "seasonal": True},
    2711: {"name": "Бананы", "category": "food", "seasonal": False},
}

# CPI Component weights for aggregation
COMPONENT_WEIGHTS = {
    "food": 0.3948,
    "nonfood": 0.3653,
    "services": 0.2399,
}


def get_data_path() -> Path:
    """Get path to weekly prices data file."""
    base_paths = [
        Path.cwd() / "data" / "kbr_weekly_prices_2008_2026.csv",
        Path.cwd().parent / "data" / "kbr_weekly_prices_2008_2026.csv",
        Path(__file__).parent.parent.parent
        / "data"
        / "kbr_weekly_prices_2008_2026.csv",
    ]

    for path in base_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Weekly prices data not found. Expected at: data/kbr_weekly_prices_2008_2026.csv"
    )


@lru_cache(maxsize=1)
def load_weekly_prices(
    start_date: str = "2016-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load weekly prices data.

    Args:
        start_date: Start date for data (default: 2016-01-01)
        end_date: End date for data (default: None = use all available)

    Returns:
        DataFrame with columns: date, product_code, product_name, price,
                               price_prev_week, wow_growth
    """
    data_path = get_data_path()
    df = pd.read_csv(data_path)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])

    # Filter by date range
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    # Add derived columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week
    df["year_month"] = df["date"].dt.to_period("M")

    return df.sort_values(["date", "product_code"]).reset_index(drop=True)


def get_product_quality_tier(product_code: int) -> str:
    """
    Get quality tier for a product.

    Returns:
        'high', 'medium', 'volatile', or 'low'
    """
    if product_code in HIGH_QUALITY_PRODUCTS:
        return "high"
    elif product_code in MEDIUM_QUALITY_PRODUCTS:
        return "medium"
    elif product_code in VOLATILE_PRODUCTS:
        return "volatile"
    return "low"


def get_high_quality_products() -> pd.DataFrame:
    """
    Load only high-quality products for nowcasting.

    Returns:
        DataFrame filtered to high-quality products only
    """
    df = load_weekly_prices()
    high_quality_codes = list(HIGH_QUALITY_PRODUCTS.keys())
    return df[df["product_code"].isin(high_quality_codes)]


def calculate_product_volatility(
    df: Optional[pd.DataFrame] = None,
    lookback_years: int = 3,
) -> pd.DataFrame:
    """
    Calculate historical volatility for each product.

    Args:
        df: Weekly prices DataFrame (if None, loads from file)
        lookback_years: Years of history to use for volatility calculation

    Returns:
        DataFrame with columns: product_code, product_name, volatility,
                               mean_wow, tier
    """
    if df is None:
        df = load_weekly_prices()

    # Filter to recent years for volatility calculation
    cutoff = df["date"].max() - pd.DateOffset(years=lookback_years)
    recent = df[df["date"] >= cutoff]

    # Calculate volatility stats per product
    stats = (
        recent.groupby(["product_code", "product_name"])["wow_growth"]
        .agg(
            [
                ("volatility", "std"),
                ("mean_wow", "mean"),
                ("count", "count"),
                ("min_wow", "min"),
                ("max_wow", "max"),
            ]
        )
        .reset_index()
    )

    # Classify by volatility tier
    def classify_volatility(row):
        vol = row["volatility"]
        if pd.isna(vol):
            return "unknown"
        elif vol < 2.0:
            return "stable"
        elif vol < 5.0:
            return "medium"
        elif vol < 10.0:
            return "volatile"
        else:
            return "ultra_volatile"

    stats["tier"] = stats.apply(classify_volatility, axis=1)
    stats["quality"] = stats["product_code"].apply(get_product_quality_tier)

    return stats.sort_values("volatility")


def aggregate_to_monthly(
    df: Optional[pd.DataFrame] = None,
    products: Optional[List[int]] = None,
    method: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate weekly prices to monthly frequency.

    Args:
        df: Weekly prices DataFrame (if None, loads high-quality products)
        products: List of product codes to include (if None, use high-quality)
        method: Aggregation method ('mean', 'last', 'first')

    Returns:
        DataFrame with monthly aggregated data per product
    """
    if df is None:
        df = get_high_quality_products()

    if products:
        df = df[df["product_code"].isin(products)]

    # Group by month and product
    agg_funcs = {
        "price": method,
        "wow_growth": "sum",  # Sum weekly growth to get monthly growth
    }

    monthly = (
        df.groupby(["year_month", "product_code", "product_name"])
        .agg(agg_funcs)
        .reset_index()
    )

    # Convert period to datetime
    monthly["date"] = monthly["year_month"].dt.to_timestamp()

    return monthly.sort_values(["date", "product_code"])


def compute_basket_signal(
    df: Optional[pd.DataFrame] = None,
    target_month: Optional[pd.Timestamp] = None,
    use_weights: bool = True,
    weighting: str = "basket",
) -> Dict[str, float]:
    """
    Compute weighted basket signal for nowcasting.

    Args:
        df: Weekly prices DataFrame (if None, loads from file)
        target_month: Month to compute signal for (if None, use current)
        use_weights: [DEPRECATED] Whether to weight by CPI basket weights.
                    Use weighting='basket' instead.
        weighting: Weighting method:
                 - 'basket': Use CPI basket weights (default)
                 - 'volatility': Use inverse volatility weights (1/std)
                 - 'equal': Equal weights for all products

    Returns:
        Dict with:
            - signal: Weighted average wow_growth
            - food_signal: Food component signal
            - nonfood_signal: Non-food component signal
            - n_products: Number of products used
            - coverage: Data coverage percentage
    """
    if df is None:
        df = get_high_quality_products()

    if target_month is None:
        target_month = df["date"].max()

    # Filter to target month
    month_start = pd.Timestamp(target_month).replace(day=1)
    month_end = month_start + pd.DateOffset(months=1)
    month_data = df[(df["date"] >= month_start) & (df["date"] < month_end)]

    if len(month_data) == 0:
        return {
            "signal": np.nan,
            "food_signal": np.nan,
            "nonfood_signal": np.nan,
            "n_products": 0,
            "coverage": 0.0,
        }

    # Calculate signals per product
    product_signals = month_data.groupby("product_code")["wow_growth"].mean()

    # Determine weighting scheme
    if not use_weights and weighting == "basket":
        weighting = (
            "equal"  # Backward compatibility: use_weights=False -> equal weights
        )

    if weighting == "basket":
        # Use CPI basket weights
        weighted_sum = 0.0
        weight_sum = 0.0
        food_sum = 0.0
        food_weight = 0.0
        nonfood_sum = 0.0
        nonfood_weight = 0.0

        for code, signal in product_signals.items():
            if pd.isna(signal):
                continue

            info = HIGH_QUALITY_PRODUCTS.get(code, {})
            weight = info.get("weight", 0.01)
            category = info.get("category", "nonfood")

            weighted_sum += signal * weight
            weight_sum += weight

            if category == "food":
                food_sum += signal * weight
                food_weight += weight
            else:
                nonfood_sum += signal * weight
                nonfood_weight += weight

        overall_signal = weighted_sum / weight_sum if weight_sum > 0 else np.nan
        food_signal = food_sum / food_weight if food_weight > 0 else np.nan
        nonfood_signal = nonfood_sum / nonfood_weight if nonfood_weight > 0 else np.nan

    elif weighting == "volatility":
        # Use inverse volatility weights: w = 1/std
        vol_stats = calculate_product_volatility(df=df, lookback_years=3)

        weighted_sum = 0.0
        weight_sum = 0.0
        food_sum = 0.0
        food_weight = 0.0
        nonfood_sum = 0.0
        nonfood_weight = 0.0

        for code, signal in product_signals.items():
            if pd.isna(signal):
                continue

            # Get volatility for this product
            vol_row = vol_stats[vol_stats["product_code"] == code]
            if len(vol_row) == 0:
                continue
            vol = vol_row.iloc[0]["volatility"]

            # Handle zero/negative volatility
            if pd.isna(vol) or vol <= 0:
                continue

            # Inverse volatility weight: w = 1/std
            inv_vol = 1.0 / vol

            # Get category for component breakdown
            info = HIGH_QUALITY_PRODUCTS.get(code, {})
            category = info.get("category", "nonfood")

            weighted_sum += signal * inv_vol
            weight_sum += inv_vol

            if category == "food":
                food_sum += signal * inv_vol
                food_weight += inv_vol
            else:
                nonfood_sum += signal * inv_vol
                nonfood_weight += inv_vol

        overall_signal = weighted_sum / weight_sum if weight_sum > 0 else np.nan
        food_signal = food_sum / food_weight if food_weight > 0 else np.nan
        nonfood_signal = nonfood_sum / nonfood_weight if nonfood_weight > 0 else np.nan

    elif weighting == "equal":
        # Equal weights for all products
        overall_signal = product_signals.mean()
        food_codes = [
            k for k, v in HIGH_QUALITY_PRODUCTS.items() if v.get("category") == "food"
        ]
        nonfood_codes = [
            k
            for k, v in HIGH_QUALITY_PRODUCTS.items()
            if v.get("category") == "nonfood"
        ]
        food_signal = product_signals[product_signals.index.isin(food_codes)].mean()
        nonfood_signal = product_signals[
            product_signals.index.isin(nonfood_codes)
        ].mean()

    else:
        raise ValueError(
            f"Unknown weighting method: {weighting}. Use 'basket', 'volatility', or 'equal'."
        )

    # Calculate coverage
    expected_products = len(HIGH_QUALITY_PRODUCTS)
    actual_products = len(product_signals.dropna())
    coverage = actual_products / expected_products if expected_products > 0 else 0.0

    return {
        "signal": overall_signal,
        "food_signal": food_signal,
        "nonfood_signal": nonfood_signal,
        "n_products": actual_products,
        "coverage": coverage,
    }


def get_weekly_trend(
    product_code: int,
    n_weeks: int = 12,
) -> pd.DataFrame:
    """
    Get recent weekly trend for a specific product.

    Args:
        product_code: Product code to analyze
        n_weeks: Number of recent weeks to return

    Returns:
        DataFrame with weekly price data
    """
    df = load_weekly_prices()
    product_df = df[df["product_code"] == product_code].copy()

    if len(product_df) == 0:
        return pd.DataFrame()

    # Get most recent weeks
    recent = product_df.nlargest(n_weeks, "date")

    return recent.sort_values("date")


def detect_anomalies(
    df: Optional[pd.DataFrame] = None,
    threshold_std: float = 2.0,
) -> pd.DataFrame:
    """
    Detect anomalous price changes.

    Args:
        df: Weekly prices DataFrame (if None, loads from file)
        threshold_std: Number of standard deviations for anomaly threshold

    Returns:
        DataFrame with anomalous observations
    """
    if df is None:
        df = load_weekly_prices()

    # Calculate historical stats per product
    stats = df.groupby("product_code")["wow_growth"].agg(["mean", "std"])

    # Merge stats back
    df = df.merge(stats, left_on="product_code", right_index=True, how="left")

    # Calculate z-scores
    df["z_score"] = (df["wow_growth"] - df["mean"]) / df["std"]

    # Filter anomalies
    anomalies = df[df["z_score"].abs() > threshold_std].copy()
    anomalies = anomalies.sort_values("date", ascending=False)

    return anomalies[
        ["date", "product_code", "product_name", "price", "wow_growth", "z_score"]
    ]


def get_summary_stats() -> Dict:
    """
    Get summary statistics for the weekly dataset.

    Returns:
        Dict with dataset statistics
    """
    df = load_weekly_prices()

    return {
        "total_rows": len(df),
        "products": df["product_code"].nunique(),
        "date_range": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        },
        "high_quality_products": len(HIGH_QUALITY_PRODUCTS),
        "medium_quality_products": len(MEDIUM_QUALITY_PRODUCTS),
        "missing_rate": df["wow_growth"].isna().mean(),
    }


if __name__ == "__main__":
    # Test loading
    print("Loading weekly prices...")
    df = load_weekly_prices()
    print(f"Loaded {len(df):,} rows")

    # Summary stats
    stats = get_summary_stats()
    print(f"\nSummary:")
    print(
        f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}"
    )
    print(f"  Products: {stats['products']}")
    print(f"  High quality: {stats['high_quality_products']}")

    # Compute current signal
    signal = compute_basket_signal()
    print(f"\nCurrent month signal:")
    print(f"  Overall: {signal['signal']:.3f}%")
    print(f"  Food: {signal['food_signal']:.3f}%")
    print(f"  Non-food: {signal['nonfood_signal']:.3f}%")
    print(f"  Coverage: {signal['coverage'] * 100:.1f}%")

    # Check for recent anomalies
    anomalies = detect_anomalies()
    recent_anomalies = anomalies[
        anomalies["date"] >= df["date"].max() - pd.DateOffset(weeks=4)
    ]
    print(f"\nRecent anomalies (last 4 weeks): {len(recent_anomalies)}")
    if len(recent_anomalies) > 0:
        print(recent_anomalies.head())
