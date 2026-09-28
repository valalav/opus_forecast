"""Evaluate a supplied sourced calendar with the existing native core backtest path."""
import argparse,copy,hashlib,json,math,statistics,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('calendar',type=Path);args=p.parse_args()
calendar=json.loads(args.calendar.read_text());package=json.loads((ROOT/'data/regional_indices.json').read_text())
region=next(r for r in package['regions'] if str(r['code'])==calendar['region_code']);rows=region['rows']
binary=ROOT/'core/target/release/regional-inflation-core'
results={};requests={}
for model in ['ridge','huber']:
    for mode in ['off','current','lags3']:
        key=f'{model}_{mode}'
        req={'schema_version':1,'region_code':calendar['region_code'],'model':model,'frequency':'raw','rows':rows,'tariff_calendar':calendar,'config':{'horizon':12,'tariff_features':mode,'seasonality_mode':'legacy','outlier_mode':'none'},'backtests':{'targets':[r['date'] for r in rows[-36:]],'horizons':[1,2,12]}}
        requests[key]=req
        process=subprocess.run([str(binary)],input=json.dumps(req),capture_output=True,text=True)
        out=json.loads(process.stdout);results[key]=out
        print(json.dumps({'case':key,'ok':out.get('ok'),'metrics':out.get('backtest',{}).get('metrics'),'error':out.get('error')},ensure_ascii=False))
node="""const fs=require('fs'),vm=require('vm');for(const f of ['tariff-calendar.js','comparison.js','tariff-comparison.js'])vm.runInThisContext(fs.readFileSync(process.argv[1]+'/web/'+f,'utf8'));const x=JSON.parse(fs.readFileSync(0,'utf8'));const out={};for(const model of ['ridge','huber']){const base=x.results[model+'_off'];if(!base.ok)continue;out[model]={};for(const mode of ['current','lags3']){const r=x.results[model+'_'+mode];out[model][mode]=r.ok?{paired:RegionalComparison.paired(base,r),groups:TariffComparison.grouped(r.backtest.rows,x.calendar)}:{error:r.error};}const expert={backtest:{rows:TariffComparison.expertRows(base.backtest.rows,x.calendar)}};out[model].expert={paired:RegionalComparison.paired(base,expert),groups:TariffComparison.grouped(expert.backtest.rows,x.calendar)};}process.stdout.write(JSON.stringify(out));"""
paired=json.loads(subprocess.run(['node','-e',node,str(ROOT)],input=json.dumps({'results':results,'calendar':calendar}),capture_output=True,text=True,check=True).stdout)
# Correct jointly over all tested models, specifications and horizons.
tests=[v for bymodel in paired.values() for spec in bymodel.values() for v in spec.get('paired',{}).values() if v.get('p_value') is not None]
tests.sort(key=lambda x:x['p_value']);last=0
for i,t in enumerate(tests):last=max(last,min(1,(len(tests)-i)*t['p_value']));t['p_value_holm_all_specs']=last
counts={key:{'ok':out.get('ok'),'unavailable_reasons':list(dict.fromkeys(r.get('reason') for r in out.get('backtest',{}).get('rows',[]) if r.get('status')=='unavailable'))[:8]} for key,out in results.items()}
# Source-series diagnostics are descriptive; no promotion from MAE alone.
diagnostics={}
try:
    import numpy as np
    from statsmodels.tsa.stattools import adfuller,kpss,acf
    from statsmodels.stats.diagnostic import acorr_ljungbox,het_arch
    from scipy.stats import jarque_bera
    y=np.array([r['y']-100 for r in rows]);diagnostics['source']={'n':len(y),'mean_mom':float(y.mean()),'std_mom':float(y.std()),'adf_p':float(adfuller(y,autolag='AIC')[1]),'kpss_p':float(kpss(y,regression='c',nlags='auto')[1]),'acf_lag12':float(acf(y,nlags=12)[12]),'monthly_means':{str(m):float(np.mean([r['y']-100 for r in rows if int(r['date'][5:7])==m])) for m in range(1,13)}}
    for key,out in results.items():
        for h in [1,2,12]:
            err=np.array([r['error'] for r in out.get('backtest',{}).get('rows',[]) if r.get('status')=='ok' and r['horizon']==h])
            if len(err)>=16:diagnostics[f'{key}_h{h}']={'n':len(err),'ljung_box_lag6_p':float(acorr_ljungbox(err,lags=[6],return_df=True).iloc[0]['lb_pvalue']),'arch_lm_lag3_p':float(het_arch(err,nlags=3)[1]),'jarque_bera_p':float(jarque_bera(err).pvalue),'first_half_mae':float(np.abs(err[:len(err)//2]).mean()),'second_half_mae':float(np.abs(err[len(err)//2:]).mean()),'note':'Valid errors only; calendar gaps may limit serial diagnostics.'}
except ImportError as e:diagnostics['unavailable']=str(e)
trajectories={}
for key,request in requests.items():
    trial=copy.deepcopy(request);trial.pop('backtests');trial['config']['cutoff']='2025-12-01'
    out=json.loads(subprocess.run([str(binary)],input=json.dumps(trial),capture_output=True,text=True).stdout)
    values=[step['all'] for step in out.get('forecast',{}).get('steps',[])]
    trajectories[key]={'ok':out.get('ok'),'origin':'2025-12-01','error':out.get('error'),'steps':out.get('forecast',{}).get('steps',[]),'std_mom':statistics.pstdev(values) if values else None,'max_abs_monthly_change':max(abs(b-a) for a,b in zip(values,values[1:])) if len(values)>1 else None,'min_mom':min(values) if values else None,'max_mom':max(values) if values else None}
report={'status':'EXPERIMENTAL_NOT_PROMOTED','calendar_path':str(args.calendar),'calendar_sha256':hashlib.sha256(args.calendar.read_bytes()).hexdigest(),'series_id':calendar['series_id'],'region':{'code':region['code'],'name':region['name']},'target_period':[rows[-36]['date'],rows[-1]['date']],'matched_comparisons':paired,'coverage':counts,'diagnostics':diagnostics,'trajectories':trajectories,'data_sha256':hashlib.sha256((ROOT/'data/regional_indices.json').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'limitations':['Current revised CPI and macro package, not complete historical publication vintages.','Calendar is a published payment-cap proxy, not observed regional utility inflation; its definition and geography must be read in the source audit.','No automatic hyperparameter tuning; fixed model defaults. Missing tariff records remain unavailable. Training rows differ when calendar history has gaps; changes cannot be attributed only to tariff features.','No sourced historical weight/baseline => expert comparison cannot be evaluated.','Source diagnostics are descriptive, not full regressors/structural-break diagnostics; no model promotion. KPSS p values are tabulation bounds (0.1 means p>=0.1, 0.01 means p<=0.01).'],'results':results,'requests':requests}
(ROOT/'verification/tariff_calendar_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'artifact':'verification/tariff_calendar_evaluation.json','status':report['status']}))
