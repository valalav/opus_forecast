# Weekly Laspeyres Nowcast Prototype

Date: 2026-06-25

Target month: June 2026.

## What Was Built

A prototype weighted weekly nowcast was added under:

```text
experiments/weekly_laspeyres_nowcast/run_weekly_laspeyres_nowcast.py
```

It uses:

- fresh weekly price data:
  `data/Сравнение еженедельных цен_01.csv`;
- item weights and names:
  `data/micro_sprav.csv`;
- the existing loader:
  `sirena/data/weekly_bridge.py`.

Outputs were saved under:

```text
archive/results/weekly_laspeyres_nowcast_20260625/
```

## Output Files

- `weekly_laspeyres_nowcast.csv` - weekly chain summary.
- `weekly_laspeyres_contributions.csv` - item-level contributions.
- `weekly_laspeyres_matches.csv` - weekly item to CPI weight matching table.
- `weekly_laspeyres_summary.md` - compact run summary.

## Matching Coverage

The prototype matched 108 of 116 unique weekly items after exact, fuzzy and
manual alias matching.

Matched total CPI weight in `micro_sprav.csv`: about `0.350`.

Manual aliases were added only for obvious cases such as:

- gasoline AI-98;
- selected medicine names with dosages omitted in the weekly file;
- electricity as a 744-747 weighted aggregate;
- housing rent in state/municipal housing;
- dental filling;
- children's sneakers.

The aggregate weekly item `Бензин автомобильный` is intentionally not matched
because AI-92, AI-95 and AI-98 are already represented separately.

## June 2026 Results

Latest weekly date: `2026-06-22`.

Weekly chain through June 22:

- observed matched-basket signal: `+1.673` p.p.;
- headline partial contribution: `+0.578` p.p.;
- component-scaled signal: `+2.124` p.p.

Month-end price bridge from `2026-05-25` to `2026-06-22`:

- observed matched-basket price index: `101.758`;
- headline partial index: `100.610`;
- component-scaled index: `102.218`.

## Interpretation

The safest diagnostic number is the partial headline contribution:

```text
June 2026 weighted weekly pressure from matched items: about +0.58 to +0.61 p.p.
```

This assumes unmatched CPI weights are unchanged, so it is conservative if
unmatched items moved in the same direction.

The observed matched-basket signal is much higher because it is normalized only
over the matched weekly basket. It should not be read as headline CPI.

The component-scaled signal is likely too high for June 2026 because gasoline
has an unusually large movement and the matched non-food weekly basket is not a
representative sample of the whole non-food component.

## Main Drivers

Largest upward weekly-chain contributions:

- AI-95 gasoline, 2026-06-22: about `+0.186` p.p.;
- AI-92 gasoline, 2026-06-22: about `+0.140` p.p.;
- AI-95 gasoline, 2026-06-15: about `+0.071` p.p.;
- potatoes and cucumbers also contributed materially.

Largest upward month-end bridge contributions:

- AI-95 gasoline: about `+0.316` p.p.;
- AI-92 gasoline: about `+0.193` p.p.;
- onions: about `+0.100` p.p.;
- potatoes: about `+0.086` p.p.;
- diesel fuel: about `+0.043` p.p.

## Forecast Implication

The prototype supports the earlier judgement that June 2026 is materially above
the old safe area near `100.3`.

Given matched-item weighted pressure around `+0.6` p.p. and non-weekly basket
uncertainty, this method is consistent with a June headline nowcast near the
upper part of the previous operational range, not with a calm `100.3` scenario.

It should not replace the final forecast yet. The next step is a cutoff-safe
backtest against official monthly CPI facts and comparison with current
`weekly_bridge_v1`.

## Verification

Ran:

```text
python3 experiments/weekly_laspeyres_nowcast/run_weekly_laspeyres_nowcast.py --month 2026-06
python3 -m py_compile experiments/weekly_laspeyres_nowcast/run_weekly_laspeyres_nowcast.py
```

No production forecast cache was changed.
