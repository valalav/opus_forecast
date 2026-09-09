#!/usr/bin/env python3
"""Append verified operational weekly prices without shifting existing dates.

Use --source-start to declare an audited release window when older source
vintages disagree. All overlaps within that window must agree. Historical date
migration is deliberately excluded: it is a separate, reviewed one-time task.
The accounting-month overrides belong to weekly_bridge; observed dates stay put.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sirena.data.weekly_bridge import load_semicolon_weekly_prices

CANONICAL = ROOT / 'data/kbr_weekly_prices_2008_2026.csv'
FRESH = ROOT / 'data/Сравнение еженедельных цен_01.csv'
AGGREGATE_ROWS = {'Продовольственные товары', 'Непродовольственные товары', 'Услуги'}
KEY = ['date', 'product_name']

def append_verified(history, fresh):
    """Return only an append; conflict, duplicate or missing boundary fails closed."""
    history, fresh = history.copy(), fresh.copy()
    for frame in (history, fresh):
        frame['date'] = pd.to_datetime(frame['date'], errors='raise')
        if frame.empty or frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
            raise ValueError('Empty input, invalid key or duplicate weekly key')
        if np.isinf(frame.price).any() or (frame.price <= 0).any():
            raise ValueError('Weekly observed prices must be positive or missing')
    if history.groupby('product_name').product_code.nunique().max() > 1:
        raise ValueError('Ambiguous historical product-name/code mapping')
    overlap = history.merge(fresh, on=KEY, suffixes=('_history', '_fresh'))
    if overlap.empty:
        raise ValueError('No common observations to verify source date alignment')
    conflicts = overlap[~np.isclose(overlap.price_history, overlap.price_fresh, rtol=0, atol=1e-8, equal_nan=True)]
    if not conflicts.empty:
        first = conflicts.iloc[0]
        raise ValueError(f'Overlapping prices disagree: {len(conflicts)}; {first.date.date()} {first.product_name}')
    end = history.date.max()
    add = fresh[fresh.date > end].copy()
    if add.empty:
        return history, {'added_rows': 0, 'overlap_rows': len(overlap), 'last': str(end.date())}
    mapping = history.groupby('product_name').product_code.first().to_dict()
    unknown = sorted(set(add.product_name) - set(mapping))
    if unknown:
        raise ValueError(f'Explicit product-code mapping required: {unknown}')
    # Every appended item must have the same last historical boundary in source.
    needed = history[history.product_name.isin(add.product_name)].sort_values('date').groupby('product_name').tail(1)
    boundary = needed.merge(fresh[KEY], on=KEY, how='left', indicator=True)
    if (boundary['_merge'] != 'both').any():
        raise ValueError('Source misses a last-observed item boundary')
    add['product_code'] = add.product_name.map(mapping)
    add = add[['date', 'product_code', 'product_name', 'price']].sort_values(['product_code', 'date'])
    anchors = needed[['date', 'product_code', 'product_name', 'price']]
    chain = pd.concat([anchors, add], ignore_index=True).sort_values(['product_code', 'date'])
    chain['price_prev_week'] = chain.groupby('product_code').price.shift()
    lag_date = chain.groupby('product_code').date.shift()
    # A missing observation is not a zero or a one-week change across a gap.
    consecutive = (chain.date - lag_date).dt.days == 7
    chain['price_prev_week'] = chain.price_prev_week.where(consecutive)
    chain['wow_growth'] = (chain.price / chain.price_prev_week - 1) * 100
    add = chain[chain.date > end][list(history.columns)]
    result = pd.concat([history, add], ignore_index=True)
    return result, {'added_rows':len(add), 'added_dates':sorted(add.date.dt.strftime('%Y-%m-%d').unique()), 'overlap_rows':len(overlap),'last':str(result.date.max().date())}

def build(dry_run=False, canonical=CANONICAL, fresh=FRESH, output=None, source_start=None):
    canonical, fresh = Path(canonical), Path(fresh)
    output = Path(output) if output else canonical
    original = canonical.read_bytes()
    incoming_hash = hashlib.sha256(fresh.read_bytes()).hexdigest()
    history = pd.read_csv(canonical, parse_dates=['date'])
    incoming = load_semicolon_weekly_prices(fresh)
    if hashlib.sha256(fresh.read_bytes()).hexdigest() != incoming_hash:
        raise ValueError('Operational source changed while reading')
    incoming = incoming[~incoming.product_name.isin(AGGREGATE_ROWS)]
    if source_start:
        incoming = incoming[incoming.date >= pd.Timestamp(source_start)]
    result, audit = append_verified(history, incoming)
    audit.update({'canonical_sha256':hashlib.sha256(original).hexdigest(), 'fresh_sha256':incoming_hash,'source_start':source_start, 'dry_run':dry_run})
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if dry_run: return 0
    if not audit['added_rows']:
        if output != canonical: output.write_bytes(original)
        return 0
    if canonical.read_bytes() != original:
        raise ValueError('Canonical source changed during refresh')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output == canonical:
        backup=canonical.with_name(canonical.name+'.before_refresh_'+audit['canonical_sha256'][:12])
        if not backup.exists(): backup.write_bytes(original)
    temp=output.with_suffix(output.suffix+'.tmp')
    appended = result.iloc[len(history):].to_csv(index=False, header=False, date_format='%Y-%m-%d').encode('utf-8')
    # Keep every historical byte; do not round/reformat already verified prices.
    payload = original + (b'' if original.endswith(b'\n') else b'\n') + appended
    temp.write_bytes(payload)
    if canonical.read_bytes() != original or hashlib.sha256(fresh.read_bytes()).hexdigest() != incoming_hash:
        temp.unlink(missing_ok=True)
        raise ValueError('Input changed before weekly publication')
    temp.replace(output)
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--canonical',type=Path,default=CANONICAL)
    p.add_argument('--fresh',type=Path,default=FRESH)
    p.add_argument('--output',type=Path)
    p.add_argument('--source-start',help='Explicit audited source window start, e.g. 2026-01-01')
    return build(**vars(p.parse_args()))
if __name__=='__main__': raise SystemExit(main())
