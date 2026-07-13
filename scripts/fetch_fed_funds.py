"""
Fetch Federal Funds Rate History (via Treasury Yields as Proxy)
========================================================================

The Federal Funds Rate is the target rate set by the Federal Reserve.
This script fetches 5-year Treasury yields (^FVX) as a proxy for Fed policy.

Note: This uses Treasury yields as a proxy since the actual Fed Funds Rate
requires FRED API (pandas-datareader or fredapi).

Usage:
    python3 scripts/fetch_fed_funds.py
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def fetch_fed_funds_proxy(
    start_date: str = "2010-01-01",
    end_date: str = None,
    cache_file: str = "data/fed_funds_rate.csv",
) -> pd.DataFrame:
    """
    Fetch Federal Funds Rate proxy (5-year Treasury yield ^FVX).

    Args:
        start_date: Start date for data
        end_date: End date (None = today)
        cache_file: Path to cache file

    Returns:
        DataFrame with columns: Date (index), Rate (proxy rate)
    """
    cache_path = Path(cache_file)

    # Try loading from cache
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["Date"], index_col="Date")
            if len(df) > 0:
                last_date = df.index.max()
                if (pd.Timestamp.now() - last_date).days < 7:
                    print(f"Loaded from cache: {cache_file}")
                    return df
        except Exception as e:
            print(f"Cache load failed: {e}")

    # Fetch from Yahoo Finance
    try:
        import yfinance as yf

        ticker = "^FVX"  # 5-year Treasury yield (proxy for Fed policy)
        print(f"Fetching {ticker} from Yahoo Finance...")

        fed_data = yf.download(
            ticker, start=start_date, end=end_date, interval="1mo", progress=False
        )

        if len(fed_data) == 0:
            raise ValueError("No data retrieved from Yahoo Finance")

        # Extract Close price and convert to percentage
        df = pd.DataFrame()
        df["Rate"] = fed_data["Close"].resample("MS").last()
        df.index.name = "Date"

        # Drop NaN values
        df = df.dropna()

        if len(df) == 0:
            raise ValueError("All data is NaN after cleaning")

        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file)
        print(f"Saved to cache: {cache_file}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Total rows: {len(df)}")

        return df

    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")
    except Exception as e:
        raise ValueError(f"Error fetching Fed Funds proxy: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Federal Funds Rate history (via Treasury yield proxy)"
    )
    parser.add_argument(
        "--start", default="2010-01-01", help="Start date (default: 2010-01-01)"
    )
    parser.add_argument("--end", default=None, help="End date (default: today)")
    parser.add_argument(
        "--output",
        default="data/fed_funds_rate.csv",
        help="Output file path (default: data/fed_funds_rate.csv)",
    )

    args = parser.parse_args()

    try:
        df = fetch_fed_funds_proxy(
            start_date=args.start, end_date=args.end, cache_file=args.output
        )

        print(f"\n✓ Successfully fetched {len(df)} data points")
        print(f"✓ Output: {args.output}")
        print(f"✓ Columns: {df.columns.tolist()}")
        print(f"✓ Date range: {df.index.min()} to {df.index.max()}")

    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
