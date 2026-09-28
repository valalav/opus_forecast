"""Recompute paired weekly metrics, preserve missing stages and source hashes."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from sirena.evaluation import code_manifest
from sirena.data_loader import forecast_source_manifest
RUN=ROOT/'archive/results/model_development/20260928_p4_01'

protocol=json.loads((RUN/'protocol.json').read_text())
rows=pd.read_csv(RUN/'forecast_rows.csv')
saved=pd.read_csv(RUN/'metrics.csv')
assert len(rows)==len(protocol['targets'])*4*3
assert not rows.duplicated(['target_month','stage','model']).any()
assert protocol['code_hashes']==code_manifest(ROOT)
assert protocol['source_manifest']==forecast_source_manifest()
result=[]
for stage,part in rows.groupby('stage'):
    wide=part.pivot(index='target_month',columns='model',values='prediction')
    actual=part.groupby('target_month').actual.first()
    wide=wide.dropna()
    errors=wide.sub(actual.loc[wide.index],axis=0)
    for name in protocol['model_names']:
        e=errors[name]
        metric=saved[(saved.model==name)&(saved.stage==stage)].iloc[0]
        assert metric.N_common==len(e)
        assert metric.N_valid==part.loc[part.model==name,'prediction'].notna().sum()
        for key,value in {'MAE_common':e.abs().mean(),'RMSE_common':np.sqrt((e**2).mean()),
                          'bias_common':e.mean(),'hit_within_0_5pp_common':(e.abs()<=.5).mean()}.items():
            assert np.isclose(metric[key],value,atol=1e-12),(stage,name,key)
    for base in ['Ridge','CalendarStageCorrection']:
        a=errors.WeeklyNewsBridge;b=errors[base]
        gain=float(b.abs().mean()-a.abs().mean())
        guards=bool(np.mean(a*a)<=np.mean(b*b) and abs(a.mean())<=abs(b.mean()) and (abs(a)<=.5).mean()>=(abs(b)<=.5).mean())
        result.append(dict(stage=stage,baseline=base,N_pair=len(a),N_planned=len(protocol['targets']),
                           mae_gain_pp=gain,guards_pass=guards,complete_pair=len(a)==len(protocol['targets']),
                           screening_pass=gain>=.025 and guards and len(a)==len(protocol['targets'])))
pd.DataFrame(result).to_csv(HERE/'p4_screening.csv',index=False)
summary=dict(status='PASS',experimental_execution=json.loads((RUN/'summary.json').read_text())['status'],
             rows=len(rows),unavailable=int(rows.prediction.isna().sum()),unique_planned_slots=True,
             independent_metrics=True,code_sources_unchanged=True,production_promotion=False)
(HERE/'p4_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
