"""Development-only parity check against the repository's Python estimators."""
from pathlib import Path
import json
import subprocess
import sys
import time
import warnings
import argparse
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(REPO))
from sirena.models.ridge import RidgeForecaster
from sirena.models.huber import HuberForecaster
from sklearn.linear_model._huber import _huber_loss_and_gradient
from scipy.optimize import minimize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--converged-reference',action='store_true')
    args=parser.parse_args()
    package = json.loads((PROJECT/'data/regional_indices.json').read_text())
    binary = PROJECT/'core/target/release/regional-inflation-core'
    results=[]
    for code in [7, 16, 77]:
        region=next(r for r in package['regions'] if r['code']==code)
        for frequency in ['raw','sa']:
            rows=region['rows' if frequency=='raw' else 'sa_rows']
            frame=pd.DataFrame(rows).rename(columns={'y':'Все товары и услуги','food':'Продовольственные товары',
                'nonfood':'Непродовольственные товары','services':'Услуги','ki':'Ki','ruonia':'Ruonia'})
            frame['date']=pd.to_datetime(frame.date);frame=frame.set_index('date')
            for cutoff in ['2024-12-01','2026-08-01']:
                for model,cls in [('ridge',RidgeForecaster),('huber',HuberForecaster)]:
                    start=time.monotonic()
                    req={'schema_version':1,'model':model,'frequency':frequency,'rows':rows,
                         'config':{'cutoff':cutoff,'horizon':12,'use_macro':True}}
                    proc=subprocess.run([str(binary)],input=json.dumps(req),capture_output=True,text=True,check=True)
                    response=json.loads(proc.stdout)
                    with warnings.catch_warnings(record=True) as warns:
                        warnings.simplefilter('always')
                        fitted=cls().fit(frame.loc[:cutoff])
                        reference_info={}
                        if model=='huber':
                            train=frame.loc[:cutoff]
                            features=fitted._prepare_features(train)
                            features['seasonal_norm']=features['month'].map(fitted.seasonal_norm)
                            features['deviation_lag1']=features.y_lag1-features.month.shift(1).map(fitted.seasonal_norm)
                            features=fitted._add_macro_features(features).dropna(subset=fitted._features+['Все товары и услуги'])
                            X=fitted.scaler.transform(features[fitted._features].values)
                            y=features['Все товары и услуги'].values
                            theta=np.r_[fitted.model.coef_,fitted.model.intercept_,fitted.model.scale_]
                            objective,gradient=_huber_loss_and_gradient(theta,X,y,1.35,.3,np.ones(len(y)))
                            reference_info={'legacy_iterations':int(fitted.model.n_iter_), 'legacy_objective':float(objective),
                                            'legacy_gradient_max':float(max(abs(gradient)))}
                            if args.converged_reference:
                                optimized=minimize(_huber_loss_and_gradient,theta,args=(X,y,1.35,.3,np.ones(len(y))),
                                    jac=True,method='L-BFGS-B',bounds=[(None,None)]*(len(theta)-1)+[(np.finfo(float).eps*10,None)],
                                    options={'maxiter':20000,'gtol':1e-8,'ftol':1e-14,'maxls':100})
                                assert optimized.success,optimized.message
                                fitted.model.coef_=optimized.x[:-2];fitted.model.intercept_=optimized.x[-2];fitted.model.scale_=optimized.x[-1]
                                reference_info.update(objective=float(optimized.fun),gradient_max=float(max(abs(optimized.jac))),iterations=int(optimized.nit))
                        reference=fitted.forecast(12)
                    if response.get('ok'):
                        actual=np.array([s['all'] for s in response['forecast']['steps']])
                        maximum=float(np.max(np.abs(actual-reference)))
                    else:
                        actual=np.full(12,np.nan);maximum=None
                    tolerance=1e-7 if model=='ridge' else 1e-3
                    for h in [1,2,12]:
                        results.append({'region':code,'frequency':frequency,'cutoff':cutoff,'model':model,'horizon':h,
                            'python':float(reference[h-1]),'rust':float(actual[h-1]) if np.isfinite(actual[h-1]) else None,
                            'abs_difference':float(abs(actual[h-1]-reference[h-1])) if np.isfinite(actual[h-1]) else None,
                            'max_path_difference':maximum,'tolerance':tolerance,'pass':maximum is not None and maximum<=tolerance,
                            'error':response.get('error'),'python_warnings':[str(w.message) for w in warns],
                            'rust_fit':response.get('fit'),'reference_info':reference_info,'seconds':time.monotonic()-start})
    out=PROJECT/'verification';out.mkdir(exist_ok=True)
    filename='converged_reference.json' if args.converged_reference else 'reference_parity.json'
    (out/filename).write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    failures=[r for r in results if not r['pass']]
    print(json.dumps({'checks':len(results),'failed':len(failures),'max_difference':{
        name:max((r['max_path_difference'] or 0) for r in results if r['model']==name) for name in ['ridge','huber']},
        'failure_examples':[{k:r[k] for k in ['region','frequency','cutoff','model','max_path_difference','error']} for r in failures[::3]][:6]},ensure_ascii=False))
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
