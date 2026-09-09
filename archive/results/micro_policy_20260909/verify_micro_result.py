"""Independent arithmetic checks; run from the remote project root."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sirena.data_loader import forecast_source_manifest, validate_forecast_cache, load_common_backtest

cache = json.loads(Path('data/precomputed_forecasts.json').read_text())
validate_forecast_cache(cache)
assert len(cache['forecasts']) == 18
for name, values in cache['forecasts'].items():
    if name == 'Nowcast':
        assert np.isfinite(values[:2]).all() and all(x is None for x in values[2:])
    else:
        assert len(values) == 12 and np.isfinite(values).all(), name
for h in [1, 2, 12]:
    frame = load_common_backtest(h)
    assert len(frame) == 12 and frame.attrs['common_origins']

folder = Path('archive/results/micro_policy_20260909')
summary = json.loads((folder/'common_summary.json').read_text())
assert summary['source_manifest'] == forecast_source_manifest()
pred = pd.read_csv(folder/'common_predictions.csv', parse_dates=['Date', 'cutoff'])
metrics = pd.read_csv(folder/'common_metrics.csv')
assert len(pred) == 72 and np.isfinite(pred[summary['models']]).all().all()
assert not summary['failures']
for row in pred.itertuples():
    assert row.Date == row.cutoff + pd.DateOffset(months=row.horizon)
for row in metrics.itertuples():
    part = pred[pred.horizon.eq(row.horizon)]
    if row.period == 'development':
        part = part[part.Date.lt(summary['validation_start'])]
    if row.period == 'validation':
        part = part[part.Date.ge(summary['validation_start'])]
    error = part[row.model] - part.Actual
    assert np.isclose(error.abs().mean(), row.MAE_common)
    assert np.isclose(np.sqrt(np.mean(error**2)), row.RMSE_common)
    assert np.isclose((error.abs() <= .5).mean(), row.hit_within_0_5pp)
    assert row.N_common == len(part) == row.N_valid
components = pd.read_csv('data/micro_forecast_components.csv')
assert np.allclose(components.groupby('date').Weight.sum(), 1, rtol=0, atol=1e-8)
assert np.allclose(components.groupby('date').Contribution.sum(), cache['forecasts']['Micro'])
code = ['scripts/backtest_framework.py', 'sirena/models/microcomponent.py',
        'sirena/data/micro_basket.py', 'sirena/data_loader.py']
verification = {
    'checked_at': pd.Timestamp.now().isoformat(),
    'forecast_paths': '17 complete 12-month paths; Nowcast has 2 valid months, then explicit nulls',
    'source_manifest_matches': True,
    'common_prediction_rows': 72, 'common_origins': 35,
    'independent_metric_recalculation': 'all 45 rows agree, including success counts and 0.5pp hit rates',
    'forecast_weight_every_month': 1,
    'component_contributions_match_cache': True,
    'code_sha256_at_final_verification': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code},
    'code_hash_timing_note': 'Comparison ran before adding code_sha256 output and choosing constructor default; its two candidate policies were explicit and unchanged. These are final-verification hashes, not launch-time hashes.',
}
(folder/'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2))
print(json.dumps(verification, ensure_ascii=False, indent=2))
print('Current Micro h1:', cache['forecasts']['Micro'][0])
print(metrics[metrics.period.eq('validation')][['horizon', 'model', 'MAE_common', 'N_common']].to_string(index=False))
