#!/usr/bin/env python3
"""
Nowcast auto-update: reads latest weekly Rosstat data, calculates current month
MoM estimate, and compares with the forecast from forecasts/2026_mom_forecast_v1.md.

Usage:
  python3 scripts/nowcast_update.py
  python3 scripts/nowcast_update.py --month 3  # Force specific month

Output:
  - Prints nowcast vs forecast comparison
  - Updates forecasts/nowcast_log.csv with each run
"""
import csv
import sys
import os
from datetime import datetime, date
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
LOG_FILE = PROJECT / "forecasts" / "nowcast_log.csv"

# User's forecast (MoM index format)
FORECAST_2026 = {
    1: 101.60, 2: 101.10, 3: 100.75, 4: 100.45, 5: 100.20, 6: 100.10,
    7: 99.85, 8: 99.75, 9: 100.50, 10: 100.80, 11: 100.20, 12: 100.25,
}
FORECAST_2027 = {
    1: 100.90, 2: 100.55, 3: 100.40, 4: 100.35, 5: 100.15, 6: 100.05,
    7: 100.00, 8: 99.90, 9: 100.50, 10: 100.65, 11: 100.20, 12: 100.30,
}


def find_weekly_csv():
    """Find the latest weekly price comparison CSV."""
    candidates = sorted(DATA_DIR.glob("Сравнение еженедельных цен*.csv"), reverse=True)
    if not candidates:
        print("ERROR: No weekly price CSV found in data/")
        sys.exit(1)
    return candidates[0]


def load_weekly_data(csv_path):
    """Load weekly YoY and reconstruct MoM from weekly CPI estimates."""
    # Determine encoding and delimiter  
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline()
    
    sep = ";" if ";" in first_line else ","
    
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=sep)
        headers = reader.fieldnames
        
        # Find date columns (they look like "DD.MM.YYYY" or similar)
        date_cols = []
        for h in headers:
            try:
                d = datetime.strptime(h.strip(), "%d.%m.%Y")
                date_cols.append((h, d.date()))
            except ValueError:
                pass
        
        if not date_cols:
            print(f"WARNING: No date columns found in {csv_path}")
            return None, None
        
        for row in reader:
            name_col = headers[0]
            commodity = row[name_col].strip()
            if commodity in ["Все товары и услуги", "Индекс потребительских цен"]:
                weekly_vals = {}
                for col_name, d in date_cols:
                    val_str = row.get(col_name, "").strip().replace(",", ".")
                    try:
                        weekly_vals[d] = float(val_str)
                    except (ValueError, TypeError):
                        pass
                return weekly_vals, date_cols
    
    return None, None


def estimate_monthly_mom(weekly_vals, target_year, target_month):
    """
    Estimate MoM from weekly cumulative CPI data.
    Weekly values are typically cumulative within the month.
    """
    month_vals = {d: v for d, v in weekly_vals.items()
                  if d.year == target_year and d.month == target_month}
    
    if not month_vals:
        return None, 0
    
    # Sort and take the latest week's value as best estimate
    sorted_dates = sorted(month_vals.keys())
    latest_val = month_vals[sorted_dates[-1]]
    n_weeks = len(sorted_dates)
    
    # If we don't have full month, extrapolate
    # Assume roughly 4 weeks per month
    if n_weeks < 4:
        # Linear extrapolation: scale up proportionally
        total_weeks = 4
        per_week_rate = (latest_val - 100.0) / n_weeks if n_weeks > 0 else 0
        estimate = 100.0 + per_week_rate * total_weeks
    else:
        estimate = latest_val
    
    return estimate, n_weeks


def main():
    target_month = None
    if "--month" in sys.argv:
        idx = sys.argv.index("--month")
        target_month = int(sys.argv[idx + 1])
    
    now = datetime.now()
    year = now.year
    month = target_month or now.month
    
    print(f"=" * 60)
    print(f"NOWCAST UPDATE — {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Target: {year}-{month:02d}")
    print(f"=" * 60)
    
    # Load weekly data
    csv_path = find_weekly_csv()
    print(f"\nWeekly data: {csv_path.name}")
    
    weekly_vals, date_cols = load_weekly_data(csv_path)
    if weekly_vals is None:
        print("ERROR: Could not parse weekly data")
        sys.exit(1)
    
    print(f"  Dates available: {len(weekly_vals)}")
    available_dates = sorted(weekly_vals.keys())
    print(f"  Range: {available_dates[0]} — {available_dates[-1]}")
    
    # Estimate current month MoM
    estimate, n_weeks = estimate_monthly_mom(weekly_vals, year, month)
    
    if estimate is None:
        print(f"\n  No data for {year}-{month:02d}")
        return
    
    # Get forecast
    if year == 2026:
        forecast = FORECAST_2026.get(month, None)
    elif year == 2027:
        forecast = FORECAST_2027.get(month, None)
    else:
        forecast = None
    
    # Display results
    print(f"\n  Nowcast estimate:  {estimate:.2f} ({estimate - 100:+.2f}%)")
    print(f"  Weeks available:  {n_weeks}/4")
    
    if forecast:
        diff = estimate - forecast
        status = "✅ ON TRACK" if abs(diff) < 0.15 else ("⚠️ ABOVE" if diff > 0 else "⚠️ BELOW")
        print(f"  Forecast:         {forecast:.2f} ({forecast - 100:+.2f}%)")
        print(f"  Difference:       {diff:+.2f} п.п.")
        print(f"  Status:           {status}")
    
    # Log
    log_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["timestamp", "year", "month", "nowcast", "forecast", "diff", "n_weeks"])
        writer.writerow([
            now.strftime("%Y-%m-%d %H:%M"),
            year, month,
            f"{estimate:.2f}",
            f"{forecast:.2f}" if forecast else "",
            f"{estimate - forecast:.2f}" if forecast else "",
            n_weeks
        ])
    print(f"\n  Logged to: {LOG_FILE}")


if __name__ == "__main__":
    main()
