#!/usr/bin/env python3
"""Update Nowcast for February 2026 — Pure Python (no pandas)."""
import csv
import json
import math
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "Сравнение еженедельных цен_01.csv"
FORECASTS_FILE = ROOT / "data" / "precomputed_forecasts.json"

# KBR weights
W_PROD, W_NONPROD, W_SERV = 0.3986, 0.3638, 0.2376
W_TOTAL = W_PROD + W_NONPROD + W_SERV

print("=" * 60)
print("NOWCAST UPDATE — February 2026 (Pure Python)")
print("=" * 60)

# Step 1: Read only February 2026 rows
print("\n[1] Reading CSV...")
weeks = {}  # {date_str: {component: [changes]}}
seen = set()  # (date, item_num) dedup

with open(DATA_FILE, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        date_str = row.get('Date', '').strip()
        if not date_str:
            continue
        # Only February 2026
        parts = date_str.split('.')
        if len(parts) != 3:
            continue
        if parts[2] != '2026' or parts[1] != '02':
            continue

        item_num = row.get('№', '').strip()
        key = (date_str, item_num)
        if key in seen:
            continue
        seen.add(key)

        comp = row.get('Справка_нед.Компоненты', '').strip()
        change_str = row.get('Изменение цен, в % к предыдущей неделе ', '').strip()
        if not change_str:
            change_str = row.get('Изменение цен, в % к предыдущей неделе', '').strip()
        change_str = change_str.replace(',', '.')
        try:
            change_val = float(change_str)
        except (ValueError, TypeError):
            continue

        if date_str not in weeks:
            weeks[date_str] = {'Продовольственные товары': [], 'Непродовольственные товары': [], 'Услуги': []}

        if 'Непродовольств' in comp:
            weeks[date_str]['Непродовольственные товары'].append(change_val)
        elif 'Продовольств' in comp:
            weeks[date_str]['Продовольственные товары'].append(change_val)
        elif 'Услуги' in comp:
            weeks[date_str]['Услуги'].append(change_val)

print(f"  Found {len(weeks)} February weeks")

# Sort by date
sorted_dates = sorted(weeks.keys(), key=lambda x: (int(x.split('.')[2]), int(x.split('.')[1]), int(x.split('.')[0])))

# Step 2: Calculate weekly weighted indices
print("\n[2] Weekly indices...")
weekly_indices = []

for date_str in sorted_dates:
    w = weeks[date_str]
    prod_vals = w['Продовольственные товары']
    nonprod_vals = w['Непродовольственные товары']
    serv_vals = w['Услуги']

    prod_mean = sum(prod_vals) / len(prod_vals) if prod_vals else 100.0
    nonprod_mean = sum(nonprod_vals) / len(nonprod_vals) if nonprod_vals else 100.0
    serv_mean = sum(serv_vals) / len(serv_vals) if serv_vals else 100.0

    weighted_idx = (prod_mean * W_PROD + nonprod_mean * W_NONPROD + serv_mean * W_SERV) / W_TOTAL

    print(f"\n  Week {date_str} ({len(prod_vals)}P / {len(nonprod_vals)}N / {len(serv_vals)}S):")
    print(f"    Prod:    {prod_mean:.4f} ({prod_mean - 100:+.3f}%)")
    print(f"    Nonprod: {nonprod_mean:.4f} ({nonprod_mean - 100:+.3f}%)")
    print(f"    Serv:    {serv_mean:.4f} ({serv_mean - 100:+.3f}%)")
    print(f"    Weighted:{weighted_idx:.4f} ({weighted_idx - 100:+.3f}%)")

    weekly_indices.append(weighted_idx)

# Step 3: Cumulative
print(f"\n[3] Cumulative...")
cumulative = 1.0
for idx in weekly_indices:
    cumulative *= (idx / 100)
cumulative_change = (cumulative - 1) * 100
print(f"  Chain ({len(weekly_indices)} weeks): {cumulative_change:+.4f}%")

# Extrapolate remaining
avg_change = sum(idx - 100 for idx in weekly_indices) / len(weekly_indices)
remaining = max(0, 4 - len(weekly_indices))
decay = 0.6
extrapolated = avg_change * decay * remaining
weekly_signal = cumulative_change + extrapolated
if remaining > 0:
    print(f"  Extrapolation (+{remaining} weeks): {extrapolated:+.4f}%")
print(f"  Weekly signal: {weekly_signal:+.4f}%")

# Step 4: Combine with ensemble
print(f"\n[4] Combining...")
with open(FORECASTS_FILE, 'r', encoding='utf-8') as f:
    forecasts = json.load(f)

skip = {'Nowcast', 'Micro', 'Ensemble'}
h1_vals = []
for k, v in forecasts['forecasts'].items():
    if k in skip:
        continue
    if v and len(v) > 0 and v[0] is not None:
        h1_vals.append(v[0])

ensemble_h1 = sum(h1_vals) / len(h1_vals) if h1_vals else 0.5

weight_map = {1: (0.60, 0.40), 2: (0.70, 0.30), 3: (0.80, 0.20), 4: (0.90, 0.10)}
w_weekly, w_model = weight_map.get(len(weekly_indices), (0.70, 0.30))
nowcast_val = w_weekly * weekly_signal + w_model * ensemble_h1

print(f"  Ensemble h=1: {ensemble_h1:+.3f}% ({len(h1_vals)} models)")
print(f"  Weights: {int(w_weekly*100)}% weekly / {int(w_model*100)}% model")
print(f"\n  ★ NOWCAST Февраль 2026: {nowcast_val:+.4f}%")

# Step 5: Save
print(f"\n[5] Saving...")
horizon = forecasts.get('horizon', 12)
nowcast_list = [None] * horizon
nowcast_list[0] = round(nowcast_val, 6)
forecasts['forecasts']['Nowcast'] = nowcast_list
forecasts['nowcast_updated'] = datetime.now().isoformat()
forecasts['nowcast_weeks'] = len(weekly_indices)
forecasts['nowcast_dates'] = sorted_dates

with open(FORECASTS_FILE, 'w', encoding='utf-8') as f:
    json.dump(forecasts, f, indent=2, ensure_ascii=False, default=str)

print(f"  ✓ Saved to {FORECASTS_FILE}")
print(f"\n{'='*60}")
print(f"DONE: Nowcast = {nowcast_val:+.4f}% ({len(weekly_indices)} weeks)")
print(f"{'='*60}")
