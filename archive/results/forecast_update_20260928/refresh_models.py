"""Refresh the existing ensemble for the 16-month dated publication."""
from pathlib import Path
import importlib.util
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    spec = importlib.util.spec_from_file_location('precompute', ROOT/'scripts/precompute_forecasts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.compute_all_forecasts(horizon=16)
    # Do not publish the August 31 operational signal as a fresh September nowcast.
    last_week = result.get('diagnostics', {}).get('weekly_bridge', {}).get('data_date_range', {}).get('end')
    if last_week is None or pd.Timestamp(last_week) < pd.Timestamp('2026-09-14'):
        result['forecasts']['Nowcast'] = None
        result['model_status']['Nowcast'] = {'status': 'unavailable',
            'reason': f'Publication 2026-09-28: latest local weekly source {last_week}; September refresh required. Excluded from policy and Ensemble.'}
        result['diagnostics']['weekly_bridge']['publication_warning'] = result['model_status']['Nowcast']['reason']
    module.save_results(result, ROOT/'data/precomputed_forecasts.json')


if __name__ == '__main__':
    main()
