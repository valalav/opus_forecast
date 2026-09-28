"""Independent arithmetic review and current diagnostic paths; no tuning."""
from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from sirena.data_loader import load_model_data, forecast_source_manifest
from sirena.evaluation import code_manifest
from sirena.experiment_models import m1_adapters, p3_adapters

OUT = Path(__file__).resolve().parent
RUN = ROOT / 'archive/results/model_development/20260928_p3_01'


def main():
    protocol = json.loads((RUN/'protocol.json').read_text())
    rows = pd.read_csv(RUN/'forecast_rows.csv')
    metrics = pd.read_csv(RUN/'metrics.csv')
    assert len(rows) == len(protocol['targets'])*len(protocol['models'])*3
    assert not rows.duplicated(['model', 'target_month', 'horizon']).any()
    assert rows.prediction.notna().all(), 'Do not silently drop failed predictions'
    assert set(rows.status) == {'available'}
    for (model, h), part in rows.groupby(['model', 'horizon']):
        err = part.prediction-part.actual
        saved = metrics.loc[(metrics.model == model)&(metrics.horizon == h)].iloc[0]
        for key, value in {'MAE_common': abs(err).mean(), 'RMSE_common': np.sqrt((err**2).mean()),
                           'bias_common': err.mean(), 'hit_within_0_5pp': (abs(err) <= .5).mean()}.items():
            assert np.isclose(saved[key], value, atol=1e-12), (model,h,key)
    contributions = pd.read_csv(RUN/'contributions.csv')
    keys = ['model','observation_cutoff','target_month']
    summed = contributions.groupby(keys).agg(weight=('weight','sum'), contribution=('contribution','sum'), n=('group','size'))
    assert (summed.n == 45).all()
    np.testing.assert_allclose(summed.weight, 1., atol=1e-10)
    matched = rows.merge(summed.reset_index(), on=keys)
    np.testing.assert_allclose(matched.prediction, matched.contribution, atol=1e-10)
    comparisons = [('P3PartialPooling','P3IndependentRidge'),('P3PoolingLagTrend','P3PartialPooling')]
    comparisons += [(name,baseline) for name in protocol['models'] if name.startswith('P3') for baseline in ['Ridge','Huber']]
    checks = []
    for candidate, baseline in comparisons:
        a = metrics[(metrics.model==candidate)&(metrics.horizon==1)].iloc[0]
        b = metrics[(metrics.model==baseline)&(metrics.horizon==1)].iloc[0]
        gain = float(b.MAE_common-a.MAE_common)
        guards = bool(a.RMSE_common<=b.RMSE_common and abs(a.bias_common)<=abs(b.bias_common) and a.hit_within_0_5pp>=b.hit_within_0_5pp)
        checks.append(dict(candidate=candidate, baseline=baseline, mae_gain_pp=gain, guards_pass=guards,
                           screening_pass=gain>=.025 and guards))
    pd.DataFrame(checks).to_csv(OUT/'screening.csv',index=False)
    data = load_model_data('raw',include_macro=True)
    context = {'cutoff': data.index.max(), 'cache': {}}
    paths = []
    adapters = {**m1_adapters(), **p3_adapters()}
    for name in protocol['models']:
        bundle = adapters[name](data,16,context)
        for date,pred in zip(pd.date_range(data.index.max()+pd.offsets.MonthBegin(),periods=16,freq='MS'),bundle['path']):
            paths.append(dict(model=name,target_month=str(date.date()),prediction=float(pred)))
        if name.startswith('P3'):
            variant=protocol['parameters'][name]['variant']
            model=context['cache']['p3:'+variant]
            (OUT/(name+'_diagnostics.json')).write_text(json.dumps(model.diagnostics,ensure_ascii=False,indent=2,default=str)+'\n')
    pathframe=pd.DataFrame(paths)
    pathframe.to_csv(OUT/'current_paths.csv',index=False)
    source_diagnostics=[]
    for code in model.group_codes:
        y=model.basket['history'][code].loc['2016-01-01':data.index.max()].dropna()
        median=y.median();mad=(y-median).abs().median()
        robust_z=(y-median)/(1.4826*mad) if mad>0 else y*float('nan')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            adf_p=adfuller(y,regression='c',autolag='AIC')[1]
            kpss_p=kpss(y,regression='c',nlags='auto')[1]
        seasonal=y.groupby(y.index.month).median()
        source_diagnostics.append(dict(group=code,family=model.group_family[code],n=len(y),mean=y.mean(),
            std=y.std(),minimum=y.min(),maximum=y.max(),mad=mad,robust_outliers_gt4=int((abs(robust_z)>4).sum()),
            largest_abs_month=str(y.abs().idxmax().date()),adf_p=adf_p,kpss_p=kpss_p,
            ljung_box_12_p=float(acorr_ljungbox(y,lags=[12],return_df=True).lb_pvalue.iloc[0]),
            calendar_median_amplitude=float(seasonal.max()-seasonal.min())))
    pd.DataFrame(source_diagnostics).to_csv(OUT/'source_group_diagnostics.csv',index=False)
    diagnostics=[]
    for (model,origin),part in contributions.groupby(['model','observation_cutoff']):
        series=part.groupby('target_month').contribution.sum().sort_index()
        if len(series)>=12:
            diagnostics.append(dict(model=model,origin=origin,n=len(series),minimum=series.min(),maximum=series.max(),
                                    standard_deviation=series.std(),max_monthly_jump=series.diff().abs().max(),
                                    compound_12m_pct=(np.prod(1+series.iloc[:12]/100)-1)*100))
    pd.DataFrame(diagnostics).to_csv(OUT/'historical_path_diagnostics.csv',index=False)
    manifest=json.loads((RUN/'input_manifest.json').read_text())
    assert manifest['sources'] == forecast_source_manifest(), 'Source changed after run'
    assert manifest['code'] == code_manifest(ROOT), 'Code changed after run'
    (OUT/'verification.json').write_text(json.dumps(dict(status='PASS',rows=len(rows),matched_contribution_rows=len(matched),
        independent_metric_recalculation=True,full_weight_mass=True,no_missing_predictions=True,
        source_manifest=forecast_source_manifest(),code_manifest=code_manifest(ROOT),
        current_cutoff=str(data.index.max().date()), evidence_level='C',production_changed=False),indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'PASS','rows':len(rows),'contribution_checks':len(matched),'screening':checks}))


if __name__ == '__main__':
    main()
