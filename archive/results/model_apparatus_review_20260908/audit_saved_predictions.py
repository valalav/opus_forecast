"""Read-only audit of saved forecasts; does not refit or select production models."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-root', type=Path, default=ROOT / 'inputs')
parser.add_argument('--output-dir', type=Path, default=ROOT / 'evidence')
args = parser.parse_args()
INPUTS = args.input_root
OUT = args.output_dir
OUT.mkdir(exist_ok=True)

def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None

def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def stats(rows, name):
    valid = [r for r in rows if number(r.get(name)) is not None]
    errors = [float(r[name]) - float(r['Actual']) for r in valid]
    if not errors:
        return {'model': name, 'n': 0, 'total': len(rows)}
    return {'model': name, 'n': len(errors), 'total': len(rows),
            'mae': sum(abs(e) for e in errors) / len(errors),
            'rmse': math.sqrt(sum(e*e for e in errors) / len(errors)),
            'bias_forecast_minus_actual': sum(errors) / len(errors),
            'violations': sum(abs(e) > .5 for e in errors),
            'within_05_pct': 100*sum(abs(e) <= .5 for e in errors)/len(errors),
            'dates': [r['Date'] for r in valid]}

tree = ast.parse((INPUTS/'scripts/precompute_forecasts.py').read_text(encoding='utf-8'))
weights = next(ast.literal_eval(n.value) for n in ast.walk(tree)
               if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'weights' for t in n.targets))
aliases = {'RidgeShockDummies':'Ridge_Shock', 'NGBoostShock':'NGBoost_Shock',
           'RidgeExtended':'Ridge_Ext'}
summary = {'production_nominal_weight_sum':sum(weights.values()), 'horizons':{}}
all_metrics = []
for h in [1,2,12]:
    rows = read_rows(INPUTS/f'archive/results/backtest_h{h}_predictions.csv')
    stored = {r['Model']:r for r in read_rows(INPUTS/f'archive/results/backtest_h{h}_metrics.csv')}
    names = [k for k in rows[0] if k not in ['Date','Actual']]
    mismatches = []
    metrics = []
    for name in names:
        s = stats(rows, name)
        s['horizon_label'] = h
        if s['n']:
            original = stored.get(name, {})
            for key, oldkey in [('mae','MAE'), ('within_05_pct','Coverage_50pct')]:
                if oldkey in original and abs(s[key]-float(original[oldkey])) > 1e-8:
                    mismatches.append({'model':name, 'metric':key, 'stored':float(original[oldkey]), 'recomputed':s[key]})
        metrics.append(s)
        all_metrics.append(s)
    for r in rows:
        available = [(float(r[aliases.get(k,k)]), w) for k,w in weights.items()
                     if number(r.get(aliases.get(k,k))) is not None]
        r['production_weights_on_saved_columns'] = sum(v*w for v,w in available)/sum(w for v,w in available)
    full = [s for s in metrics if s['n'] == len(rows)]
    summary['horizons'][str(h)] = {
        'period':[rows[0]['Date'],rows[-1]['Date']], 'total':len(rows),
        'ranked_complete': sorted(full,key=lambda s:s['mae']),
        'incomplete':[s for s in metrics if s['n'] < len(rows)],
        'stored_metrics_mismatches': mismatches,
        'production_weights_replay':stats(rows,'production_weights_on_saved_columns'),
        'replay_caveat':'Arithmetic replay only: saved model forecasts may use different inputs/settings than live models; not production backtest evidence.'}
    with (OUT/f'replay_h{h}.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');writer.writeheader();writer.writerows(rows)
source_paths = [INPUTS/'scripts/precompute_forecasts.py'] + [
    INPUTS/f'archive/results/backtest_h{h}_{kind}.csv'
    for h in [1,2,12] for kind in ['predictions','metrics']]
summary['source_hashes']={str(p.relative_to(INPUTS)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in source_paths}
(OUT/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'model_metrics.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['horizon_label','model','n','total','mae','rmse','bias_forecast_minus_actual','violations','within_05_pct','dates'],lineterminator='\n')
    writer.writeheader();writer.writerows(all_metrics)
for h,d in summary['horizons'].items():
    print('HORIZON',h,'period',d['period'],'N',d['total'])
    print('Best full-coverage:',[(r['model'],round(r['mae'],4),r['violations']) for r in d['ranked_complete'][:6]])
    print('Incomplete:',[(r['model'],r['n'],round(r.get('mae',0),4)) for r in d['incomplete']])
    print('Stored metric mismatches:',d['stored_metrics_mismatches'])
    print('Saved Ensemble:',next(r for r in d['ranked_complete'] if r['model']=='Ensemble'))
    print('Production-weights arithmetic replay:',d['production_weights_replay'])
print('Production weights sum:',summary['production_nominal_weight_sum'])
