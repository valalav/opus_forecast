#!/usr/bin/env python3
"""
Test for Task 120: Deep Dive - GRP Forecasts

Verifies:
1. Forecast horizon extracted (e.g. 2026-2027)
2. Data aligned with standard monthly grid
"""

import pandas as pd
from pathlib import Path


def test_grp_forecast_extraction():
    """Test that GRP forecast data is properly extracted."""

    # Check file exists
    output_path = Path("data/kbr_grp_forecast.csv")
    assert output_path.exists(), "Output file does not exist"

    # Load data
    df = pd.read_csv(output_path)

    # Test 1: Forecast horizon extracted
    forecasts = df[df["is_forecast"] == True]
    forecast_years = sorted(forecasts["year"].unique())

    assert len(forecast_years) >= 1, "No forecast years found"
    assert 2026 in forecast_years, f"2026 not in forecast years: {forecast_years}"

    print(f"✓ Forecast horizon extracted: {forecast_years}")

    # Test 2: Data aligned with standard monthly grid
    assert len(df) > 0, "No data extracted"

    # Check date format (YYYY-MM-DD)
    dates = pd.to_datetime(df["date"])
    assert (dates.dt.day == 1).all(), "Not all dates are first of month"

    # Check date format string
    assert df["date"].str.match(r"\d{4}-\d{2}-\d{2}").all(), (
        "Date format not YYYY-MM-DD"
    )

    print(f"✓ Data aligned with standard monthly grid")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    # Additional validation
    assert "grp_index_base_sa" in df.columns, "GRP index column missing"
    assert df["grp_index_base_sa"].notna().all(), "Missing values in GRP index"

    print(f"✓ Data validation passed")
    print(f"  - Historical rows: {len(df[df['is_forecast'] == False])}")
    print(f"  - Forecast rows: {len(forecasts)}")

    return True


if __name__ == "__main__":
    result = test_grp_forecast_extraction()
    if result:
        print("\n✓ All tests passed!")
        exit(0)
    else:
        print("\n✗ Tests failed!")
        exit(1)
