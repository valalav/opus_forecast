"""Immutable matched-origin experiments used by the existing BacktestRunner."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd

from sirena.data_loader import forecast_source_manifest, DataFreshnessError


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def code_manifest(root):
    paths = list((root/'sirena').rglob('*.py')) + list((root/'scripts').glob('*.py'))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def validate_protocol(protocol):
    required = {'run_id', 'targets', 'horizons', 'models', 'parameters', 'seed',
                'evidence_level', 'fact_policy', 'admission', 'budget_seconds'}
    if required - protocol.keys():
        raise ValueError(f'Missing protocol fields: {sorted(required-protocol.keys())}')
    if protocol['evidence_level'] != 'C':
        raise ValueError('This revised-vintage evaluator supports C diagnostics only')
    if protocol['fact_policy'] != 'current_revised_raw':
        raise ValueError('Historical first releases unavailable: use current_revised_raw')
    targets = pd.DatetimeIndex(pd.to_datetime(protocol['targets']))
    if (targets.empty or targets.has_duplicates or not targets.is_monotonic_increasing
            or not targets.equals(targets.to_period('M').to_timestamp())):
        raise ValueError('Targets must be unique sorted month starts')
    horizons = protocol['horizons']
    if not horizons or any(type(h) is not int or h < 1 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError('Unique positive integer horizons required')
    names = protocol['models']
    if not names or len(set(names)) != len(names):
        raise ValueError('Unique model names required')
    if protocol['budget_seconds'] <= 0:
        raise ValueError('Positive compute budget required')
    return targets


def matched_metrics(rows, model_names):
    """Coverage stays visible; pairwise comparisons do not delete other failures."""
    frame = pd.DataFrame(rows)
    output, pairs = [], []
    for horizon, part in frame.groupby('horizon', sort=True):
        wide = part.pivot(index='target_month', columns='model', values='prediction')
        wide = wide.reindex(columns=model_names).astype(float)
        actual = part.groupby('target_month').actual.first().reindex(wide.index)
        common = np.isfinite(wide).all(axis=1) & np.isfinite(actual)
        for name in model_names:
            valid = np.isfinite(wide[name]) & np.isfinite(actual)
            errors = wide.loc[common, name] - actual[common]
            own = wide.loc[valid, name] - actual[valid]
            output.append({'horizon': int(horizon), 'model': name,
                'N_planned': len(wide), 'N_valid': int(valid.sum()), 'N_common': int(common.sum()),
                'success_rate': float(valid.mean()),
                'MAE_common': float(errors.abs().mean()) if len(errors) else None,
                'RMSE_common': float(np.sqrt((errors**2).mean())) if len(errors) else None,
                'bias_common': float(errors.mean()) if len(errors) else None,
                'hit_within_0_5pp': float((errors.abs() <= .5).mean()) if len(errors) else None,
                'MAE_available_diagnostic': float(own.abs().mean()) if len(own) else None,
                'max_abs_error_available': float(own.abs().max()) if len(own) else None})
        for name in model_names:
            for baseline in model_names:
                if name == baseline:
                    continue
                mask = np.isfinite(wide[name]) & np.isfinite(wide[baseline]) & np.isfinite(actual)
                candidate_error = wide.loc[mask, name] - actual[mask]
                base_error = wide.loc[mask, baseline] - actual[mask]
                for target in wide.index[mask]:
                    pairs.append({'horizon': int(horizon), 'model': name, 'baseline': baseline,
                        'target_month': target, 'N_pair': int(mask.sum()), 'N_planned': len(wide),
                        'candidate_error': float(candidate_error.loc[target]),
                        'baseline_error': float(base_error.loc[target]),
                        'absolute_loss_difference': float(abs(candidate_error.loc[target])-abs(base_error.loc[target]))})
    return pd.DataFrame(output), pd.DataFrame(pairs)


def run_registered_experiment(runner, protocol, forecasters):
    """One full path per origin/model; adapters reuse live fit/forecast implementations.

    An adapter returns {'path': array, 'coverage': dict, optional 'groups': frame}.
    Its train input contains no future rows. Internally loaded data must enforce
    the same cutoff, independently tested in the model's own contract tests.
    """
    targets = validate_protocol(protocol)
    names = protocol['models']
    if set(names) != set(forecasters):
        raise ValueError('Protocol and adapter set differ')
    root = Path(__file__).resolve().parent.parent
    output = runner.output_dir
    # BacktestRunner creates output_dir, so reserve via exclusive protocol creation.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'Experiment directory must be empty: {output}')
    with (output/'protocol.json').open('x', encoding='utf-8') as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2, allow_nan=False)
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).isoformat()
    sources = forecast_source_manifest()
    code = code_manifest(root)
    fingerprint = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    manifest = {'retrieved_at': stamp, 'sources': sources, 'code': code,
                'code_sha256': fingerprint, 'evidence_level': 'C',
                'published_at': None, 'historical_vintages_available': False,
                'fact_vintage_id': sources['inflation_data.csv'],
                'fact_policy': protocol['fact_policy']}
    write_json(output/'input_manifest.json', manifest)
    rows, coverage, failures, contributions = [], [], [], []
    budget_exhausted = False
    try:
        runner._prepare_data()
        data = runner.df_ridge
        if not targets.isin(data.index).all():
            raise ValueError('Target facts missing from current RAW data')
        target_facts = pd.to_numeric(data.loc[targets, 'Все товары и услуги'], errors='coerce')
        missing_facts = targets[~np.isfinite(target_facts.to_numpy(dtype=float))]
        if len(missing_facts):
            raise ValueError(f'Non-finite target facts: {list(missing_facts.strftime("%Y-%m"))}')
        schedule = {}
        for h in protocol['horizons']:
            for target in targets:
                schedule.setdefault(target-pd.DateOffset(months=h), []).append((h, target))
        for pos, (cutoff, requests) in enumerate(sorted(schedule.items()), 1):
            train = data.loc[:cutoff].copy()
            if train.empty or train.index.max() != cutoff:
                raise ValueError(f'Missing origin {cutoff}')
            maximum = max(h for h, _ in requests)
            context = {'cutoff': cutoff, 'horizon': maximum, 'cache': {}, 'root': root}
            for name in names:
                begin = time.monotonic()
                reason, bundle = None, {}
                try:
                    if time.monotonic()-started > protocol['budget_seconds']:
                        budget_exhausted = True
                        raise TimeoutError('Registered experiment compute budget exhausted')
                    bundle = forecasters[name](train.copy(), maximum, context)
                    if bundle.get('units') != 'mom_percent':
                        raise ValueError('Adapter must declare output units=mom_percent')
                    path = np.asarray(bundle['path'], dtype=float)
                    if path.shape != (maximum,) or not np.isfinite(path).all():
                        raise ValueError('Incomplete/non-finite path; no implicit units conversion')
                except Exception as exc:
                    reason = f'{type(exc).__name__}: {exc}'
                    path = np.full(maximum, np.nan)
                    failures.append({'model': name, 'observation_cutoff': str(cutoff.date()), 'reason': reason,
                                     'failure_kind': 'budget_not_evaluated' if budget_exhausted else 'model_failure'})
                details = bundle.get('coverage', {})
                coverage.append({'model': name, 'observation_cutoff': str(cutoff.date()),
                                 'status': 'unavailable' if reason else 'available',
                                 'reason': reason, **details})
                groups = bundle.get('contributions') if reason is None else None
                if groups is not None:
                    for record in groups:
                        contributions.append({'run_id': protocol['run_id'], 'status': 'available', 'model': name, 'observation_cutoff': str(cutoff.date()), **record})
                for h, target in requests:
                    rows.append({'run_id': protocol['run_id'], 'model': name,
                        'model_version': fingerprint, 'calculated_at': stamp,
                        'origin_asof': None, 'observation_cutoff': str(cutoff.date()),
                        'target_month': str(target.date()), 'horizon': h,
                        'representation': 'raw', 'units': 'mom_percent', 'evidence_level': 'C',
                        'basket_vintage_id': details.get('weight_vintage'),
                        'last_required_observation': str(cutoff.date()),
                        'prediction': float(path[h-1]) if np.isfinite(path[h-1]) else None,
                        'actual': float(data.at[target, 'Все товары и услуги']-100),
                        'fact_vintage_id': sources['inflation_data.csv'],
                        'fact_policy': protocol['fact_policy'], 'nature': 'model',
                        'status': 'unavailable' if reason else 'available', 'reason': reason,
                        'fit_and_path_seconds': time.monotonic()-begin})
            print(f'Registered origin {pos}/{len(schedule)} {cutoff:%Y-%m}', flush=True)
        if forecast_source_manifest() != sources or code_manifest(root) != code:
            raise DataFreshnessError('Inputs/code changed during experiment; result is invalid')
        metrics, pairs = matched_metrics(rows, names)
        metrics['fact_vintage_id'] = sources['inflation_data.csv']
        metrics['evidence_level'] = 'C'
        pd.DataFrame(rows).to_csv(output/'forecast_rows.csv', index=False)
        metrics.to_csv(output/'metrics.csv', index=False)
        pairs.to_csv(output/'paired_losses.csv', index=False)
        pd.DataFrame(coverage).to_csv(output/'coverage.csv', index=False)
        pd.DataFrame(contributions, columns=['run_id','status','model','observation_cutoff','target_month','group','prediction','weight','contribution']).to_csv(output/'contributions.csv', index=False)
        write_json(output/'failures.json', failures)
        budget_exhausted = budget_exhausted or time.monotonic()-started > protocol['budget_seconds']
        minimum_common = int(metrics.N_common.min())
        state = {'status': ('BUDGET_EXHAUSTED' if budget_exhausted else
                            'COMPLETED_NO_COMMON_SAMPLE' if minimum_common == 0 else
                            'COMPLETED_WITH_FAILURES' if failures else 'COMPLETED'),
                 'min_common_sample': minimum_common, 'budget_exhausted': budget_exhausted,
                 'complete_common_sample': bool((metrics.N_common == metrics.N_planned).all()),
                 'screening_eligible': not budget_exhausted and bool((metrics.N_common == metrics.N_planned).all()),
                 'rows': len(rows), 'origins': len(schedule), 'failures': len(failures),
                 'seconds': time.monotonic()-started, 'evidence_level': 'C',
                 'promotion': 'not decided; historical diagnostic only'}
        write_json(output/'execution.json', state)
        (output/'decision.md').write_text(
            '# Experiment awaiting adjudication\n\nHistorical revised-vintage diagnostic (C). '
            f'Execution status: {state["status"]}. Minimum common sample: {minimum_common}. '
            'All planned models/targets and failures are retained. No production promotion.\n', encoding='utf-8')
        return pd.DataFrame(rows), metrics, state
    except Exception as exc:
        write_json(output/'execution.json', {'status': 'INVALID', 'reason': f'{type(exc).__name__}: {exc}'})
        raise
