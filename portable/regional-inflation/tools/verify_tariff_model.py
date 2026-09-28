"""Core integration, origin safety and feature isolation on a declared synthetic calendar.
CPI observations are real; tariff assumptions below are deliberately synthetic, NOT research evidence.
"""
import copy, hashlib, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'core/target/release/regional-inflation-core'
checks=[]
def check(name,value):
    checks.append({'name':name,'pass':bool(value)})
    assert value,name
package=json.loads((ROOT/'data/regional_indices.json').read_text())
region=next(x for x in package['regions'] if str(x['code'])=='7')
rows=region['rows']
calendar={'schema_version':1,'region_code':'7','series_id':'SYNTHETIC_CONTRACT_TEST_NOT_ECONOMIC_DATA','records':[]}
for year in range(2015,2029):
    for month in range(1,13):
        calendar['records'].append({'date':f'{year}-{month:02d}-01','rate':8.0 if month==7 else 0.0,'known_at':f'{year-1}-11-01','source':'synthetic:contract-test','kind':'assumption','weight':5.0,'baseline':6.0 if month==7 else 0.0})
req={'schema_version':1,'region_code':'7','model':'ridge','frequency':'raw','rows':rows,'config':{'horizon':12,'tariff_features':'off','cutoff':'2024-08-01'},'tariff_calendar':calendar}
def run(request):
    p=subprocess.run([str(BINARY)],input=json.dumps(request),text=True,capture_output=True)
    try:return json.loads(p.stdout)
    except Exception:raise RuntimeError(p.stderr or p.stdout)
baseline=run(req);check('baseline succeeds',baseline.get('ok'))
no_calendar=copy.deepcopy(req);no_calendar.pop('tariff_calendar');check('off preserves exact baseline',baseline==run(no_calendar))
outputs={}
for mode in ['current','lags3']:
    local=copy.deepcopy(req);local['config']['tariff_features']=mode
    # Dec previous year plan does not cover Jan next year from Aug origin: explicit future unknown must fail.
    missing=run(local);check(mode+' unknown next-year plan fails',not missing.get('ok'))
    # Scenario is made known before origin solely to test mechanics, not measure economic accuracy.
    for r in local['tariff_calendar']['records']:
        if r['date'][:4]=='2025':r['known_at']='2024-07-01'
    out=run(local);check(mode+' forecast succeeds',out.get('ok'));outputs[mode]=out
    features=out['fit']['features'];check(mode+' contains tariff features',any('tariff' in f for f in features))
    future=copy.deepcopy(local)
    for r in future['rows']:
        if r['date']>'2024-08-01':r['y']+=20;r['food']+=30;r['ki']=99
    # Later published revisions to both historical and future events must not alter an older origin.
    future['tariff_calendar']['records'] += [{**r,'rate':40,'known_at':'2026-09-01'} for r in calendar['records'] if r['date'][:4] in ('2023','2024','2025')]
    check(mode+' no leakage from future facts or calendar revisions',out==run(future))
    for h in [1,2,12]:
        shorter=copy.deepcopy(local);shorter['config']['horizon']=h
        a=run(shorter);check(mode+f' horizon {h} finite path',a.get('ok') and len(a['forecast']['steps'])==h)
    bt=copy.deepcopy(req);bt['config']['tariff_features']=mode;bt['config'].pop('cutoff');bt['backtests']={'targets':[r['date'] for r in rows[-36:]],'horizons':[1,2,12]}
    result=run(bt);check(mode+' backtest survives unavailable live forecast',result.get('ok') and len(result['backtest']['rows'])==108)
    check(mode+' missing schedules are visible',any(r['status']=='unavailable' for r in result['backtest']['rows']))
    outputs[mode+'_backtest']=result
# Live information date can be later than the latest monthly CPI observation.
live=copy.deepcopy(req);live['config'].update({'tariff_features':'current','tariff_as_of':'2025-01-15'})
check('later live tariff information allowed',run(live).get('ok'))
changed=copy.deepcopy(live);changed['tariff_calendar']['records'].append({'date':'2025-07-01','rate':40,'known_at':'2025-01-16','source':'synthetic:future-revision','kind':'assumption'})
check('exact live information day cutoff',run(live)==run(changed))
original=copy.deepcopy(live);original['config'].pop('tariff_as_of');original['backtests']={'targets':['2024-10-01','2025-01-01'],'horizons':[1,2,12]}
with_today=copy.deepcopy(original);with_today['config']['tariff_as_of']='2026-09-28'
check('live as-of override ignored in historical backtests',run(original)['backtest']==run(with_today)['backtest'])
wrong=copy.deepcopy(req);wrong['config']['tariff_features']='current';wrong['region_code']='77';check('region mismatch blocked',not run(wrong).get('ok'))
sa=copy.deepcopy(req);sa['config']['tariff_features']='current';sa['frequency']='sa';check('SA blocked',not run(sa).get('ok'))
report={'status':'PASS','checks':checks,'purpose':'SYNTHETIC TARIFF CALENDAR CONTRACT TEST ONLY, not economic performance evidence','binary_sha256':hashlib.sha256(BINARY.read_bytes()).hexdigest(),'outputs':outputs}
(ROOT/'verification/tariff_model_contract.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks)}))
