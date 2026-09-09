"""Independent M1 audit; stdout JSON only, reads immutable experiment artifacts."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np

ROOT=Path('/home/valalav/_projects/sirena-kbr')
RUN=ROOT/'archive/results/model_development/20260909_m1_02'
OLD=ROOT/'archive/results/model_development/20260909_m1_01'
PRIOR=ROOT/'archive/results/micro_policy_20260909/common_predictions.csv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    execution=json.loads((RUN/'execution.json').read_text())
    assert execution['status']=='COMPLETED',execution
    watched=list(RUN.glob('*'))+list(OLD.glob('*'))+[PRIOR]
    watched=[p for p in watched if p.is_file()]
    before={str(p.relative_to(ROOT)):sha(p) for p in watched}
    protocol=json.loads((RUN/'protocol.json').read_text())
    threshold=protocol['admission']['min_mae_gain_pp'];assert threshold==.025
    f=pd.read_csv(RUN/'forecast_rows.csv');old=pd.read_csv(OLD/'forecast_rows.csv')
    targets=protocol['targets'];models=protocol['models'];horizons=protocol['horizons']
    keys=['model','observation_cutoff','target_month','horizon']
    assert len(f)==len(targets)*len(models)*len(horizons)==432
    assert not f.duplicated(keys).any()
    assert (f.status=='available').all()
    assert np.isfinite(f[['prediction','actual']]).all().all()
    assert f.origin_asof.isna().all() and (f.evidence_level=='C').all()
    assert set(f.target_month)==set(targets)
    for r in f.itertuples():
        origin=pd.Period(r.observation_cutoff,freq='M');target=pd.Period(r.target_month,freq='M')
        assert target.ordinal-origin.ordinal==r.horizon
    pairs=f.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(pairs)==432
    changes=pairs.assign(abs_prediction_change=lambda x:(x.prediction_new-x.prediction_old).abs()).groupby('model').abs_prediction_change.max().to_dict()
    assert max(changes.values())<1e-12
    prior=pd.read_csv(PRIOR)
    baseline=[]
    for model,column in [('Ridge','Ridge'),('Huber','Huber'),('Micro','MicroParentSeasonal')]:
        x=f[f.model==model].merge(prior,left_on=['target_month','observation_cutoff','horizon'],right_on=['Date','cutoff','horizon'],validate='one_to_one')
        difference=float((x.prediction-x[column]).abs().max());assert len(x)==72 and difference<1e-12
        assert float((x.actual-x.Actual).abs().max())<1e-12
        baseline.append({'model':model,'matched_rows':len(x),'max_abs_difference':difference})
    reported=pd.read_csv(RUN/'metrics.csv').set_index(['horizon','model'])
    calculated=[];metric_diffs=[]
    for (h,model),group in f.groupby(['horizon','model']):
        assert set(group.target_month)==set(targets) and len(group)==24
        errors=(group.prediction-group.actual).to_numpy()
        m={'horizon':int(h),'model':model,'N':len(errors),'MAE':float(np.abs(errors).mean()),'RMSE':float(np.sqrt((errors**2).mean())),'bias':float(errors.mean()),'hit':float((np.abs(errors)<=.5).mean())}
        calculated.append(m)
        r=reported.loc[(h,model)]
        assert (r[['N_planned','N_valid','N_common']]==24).all()
        for key,saved in [('MAE','MAE_common'),('RMSE','RMSE_common'),('bias','bias_common'),('hit','hit_within_0_5pp')]:
            delta=abs(m[key]-r[saved]);assert delta<1e-12;metric_diffs.append(delta)
    contributions=pd.read_csv(RUN/'contributions.csv')
    gkeys=['observation_cutoff','target_month','group']
    a=contributions[contributions.model=='GroupsFixed'];b=contributions[contributions.model=='GroupsDated']
    p=a.merge(b,on=gkeys,suffixes=('_fixed','_dated'),validate='one_to_one')
    assert len(p)==len(a)==len(b)
    group_prediction_delta=float((p.prediction_fixed-p.prediction_dated).abs().max())
    assert group_prediction_delta<1e-12
    assert np.isfinite(contributions[['prediction','weight','contribution']]).all().all()
    assert (contributions.weight>=0).all()
    identity_delta=float((contributions.prediction*contributions.weight-contributions.contribution).abs().max());assert identity_delta<1e-12
    sums=contributions.groupby(['model','observation_cutoff','target_month']).agg(weight=('weight','sum'),prediction=('contribution','sum'),groups=('group','nunique')).reset_index()
    mass_delta=float((sums.weight-1).abs().max());assert mass_delta<1e-12
    x=f[f.model.isin(['GroupsFixed','GroupsDated'])].merge(sums,on=['model','observation_cutoff','target_month'],suffixes=('_row','_sum'),validate='many_to_one')
    assert len(x)==144
    total_delta=float((x.prediction_row-x.prediction_sum).abs().max());assert total_delta<1e-12
    metrics={(x['horizon'],x['model']):x for x in calculated}
    comparisons=[]
    for h in horizons:
        x=f[f.horizon==h].pivot(index='target_month',columns='model',values='prediction').sort_index()
        actual=f[f.horizon==h].groupby('target_month').actual.first().reindex(x.index).to_numpy()
        losses=np.abs(x.GroupsFixed.to_numpy()-actual)-np.abs(x.GroupsDated.to_numpy()-actual)
        n=len(losses);block=12 if h==12 else 3;draws=2000;rng=np.random.default_rng(42)
        starts=rng.integers(0,n,size=(draws,int(np.ceil(n/block))))
        indices=((starts[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:n]
        sampled=losses[indices].mean(axis=1);low,high=np.quantile(sampled,[.025,.975])
        ma=metrics[(h,'GroupsFixed')];mb=metrics[(h,'GroupsDated')]
        gain=float(losses.mean());guards={'minimum_gain':gain>=threshold,'rmse_not_worse':mb['RMSE']<=ma['RMSE']+1e-12,'abs_bias_not_worse':abs(mb['bias'])<=abs(ma['bias'])+1e-12,'hit_not_worse':mb['hit']>=ma['hit']-1e-12}
        comparisons.append({'horizon':h,'N':n,'gain_fixed_loss_minus_dated_loss_pp':gain,'mae_fixed':ma['MAE'],'mae_dated':mb['MAE'],'rmse_fixed':ma['RMSE'],'rmse_dated':mb['RMSE'],'bias_fixed':ma['bias'],'bias_dated':mb['bias'],'hit_fixed':ma['hit'],'hit_dated':mb['hit'],'ci95_exploratory':[float(low),float(high)],'circular_block_length':block,'draws':draws,'seed':42,'guards':guards,'screening_verdict':'NO_GAIN' if not all(guards.values()) else 'INCONCLUSIVE','prospective_validation':False})
    reconstruction=pd.read_csv('/tmp/sirena_aggregation_diagnostic.csv')
    re={'N':len(reconstruction),'fixed_MAE':float((reconstruction.fixed45-reconstruction.Actual).abs().mean()),'dated_MAE':float((reconstruction.updated45-reconstruction.Actual).abs().mean()),'fixed_max_abs':float((reconstruction.fixed45-reconstruction.Actual).abs().max()),'dated_max_abs':float((reconstruction.updated45-reconstruction.Actual).abs().max()),'sha256':sha(Path('/tmp/sirena_aggregation_diagnostic.csv'))}
    assert re['N']==35 and abs(re['fixed_MAE']-.043321)<1e-6 and abs(re['dated_MAE']-.017230)<1e-6
    after={str(p.relative_to(ROOT)):sha(p) for p in watched};assert before==after
    result={'status':'PASS_INDEPENDENT_CHECKS','execution':execution,'protocol_sha256':sha(RUN/'protocol.json'),'primary_screening_verdict':next(c['screening_verdict'] for c in comparisons if c['horizon']==1),'metrics':calculated,'metric_recompute_max_delta':max(metric_diffs),'new_vs_01_max_prediction_change':changes,'baseline_vs_prior':baseline,'group_forecast_identity_max_delta':group_prediction_delta,'contribution_identity_max_delta':identity_delta,'full_weight_max_delta':mass_delta,'group_sum_forecast_max_delta':total_delta,'group_counts_min_max':[int(sums.groups.min()),int(sums.groups.max())],'paired_comparisons':comparisons,'fact_reconstruction':re,'immutable_inputs_verified':True,'input_hashes':before,'limitations':['Only 24 previously inspected targets; bootstrap intervals exploratory, not confirmatory.','Current revised RAW facts and weights; C evidence, no historical release vintages.','Group December price base assumed rather than observed official detailed base.','Cross-year basket freeze is explicit approximation.','Ensemble excluded: no verified matching live origin-aware adapter.','Factual reconstruction improvement is not forecast improvement.']}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
