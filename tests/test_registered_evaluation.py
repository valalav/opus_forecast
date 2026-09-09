import numpy as np
import pandas as pd
import pytest
from sirena.evaluation import matched_metrics, validate_protocol, run_registered_experiment


def protocol():
    return {'run_id':'test','targets':['2025-01-01','2025-02-01'], 'horizons':[1,2],
        'models':['ok','broken'], 'parameters':{}, 'seed':42,'evidence_level':'C',
        'fact_policy':'current_revised_raw','admission':{},'budget_seconds':30}


def test_no_successful_origin_filter_and_null_common_metrics():
    rows=[]
    for d in ['2025-01-01','2025-02-01']:
        rows.extend([{'target_month':d,'horizon':1,'model':'ok','prediction':1.,'actual':2.},
                     {'target_month':d,'horizon':1,'model':'broken','prediction':None,'actual':2.}])
    metrics,pairs=matched_metrics(rows,['ok','broken'])
    assert (metrics.N_planned==2).all()
    assert (metrics.N_common==0).all()
    assert metrics.MAE_common.isna().all()
    assert pairs.empty
    assert metrics.set_index('model').loc['ok','N_valid']==2


def test_asof_claim_and_duplicate_schedule_rejected():
    p=protocol(); p['evidence_level']='A'
    with pytest.raises(ValueError,match='C diagnostics'):
        validate_protocol(p)
    p=protocol(); p['targets']=['2025-01-01']*2
    with pytest.raises(ValueError,match='unique'):
        validate_protocol(p)


def test_paths_called_once_per_origin_and_failures_kept(tmp_path, monkeypatch):
    import sirena.evaluation as module
    monkeypatch.setattr(module,'forecast_source_manifest',lambda:{'inflation_data.csv':'fake_hash'})
    monkeypatch.setattr(module,'code_manifest',lambda root:{})
    class Runner:
        output_dir=tmp_path
        def _prepare_data(self):
            self.df_ridge=pd.DataFrame({'Все товары и услуги':100.3},index=pd.date_range('2023-01-01','2025-02-01',freq='MS'))
    calls=[]
    def good(train,horizon,context):
        assert train.index.max()==context['cutoff']
        calls.append(context['cutoff'])
        return {'units':'mom_percent','path':np.arange(1,horizon+1,dtype=float)}
    def bad(*args):
        raise RuntimeError('deliberate missing source')
    rows,metrics,state=run_registered_experiment(Runner(),protocol(),{'ok':good,'broken':bad})
    assert len(calls)==3 and len(rows)==8
    assert (rows[rows.model.eq('ok')].prediction==rows[rows.model.eq('ok')].horizon).all()
    assert rows[rows.model.eq('broken')].reason.str.contains('deliberate').all()
    assert state['status']=='COMPLETED_NO_COMMON_SAMPLE'
    assert rows.origin_asof.isna().all()
    with pytest.raises(FileExistsError):
        run_registered_experiment(Runner(),protocol(),{'ok':good,'broken':bad})


def test_source_revision_invalidates_whole_run(tmp_path,monkeypatch):
    import sirena.evaluation as module
    count=[0]
    def manifest():
        count[0]+=1
        return {'inflation_data.csv':str(count[0])}
    monkeypatch.setattr(module,'forecast_source_manifest',manifest)
    monkeypatch.setattr(module,'code_manifest',lambda root:{})
    class Runner:
        output_dir=tmp_path
        def _prepare_data(self):
            self.df_ridge=pd.DataFrame({'Все товары и услуги':100.3},index=pd.date_range('2023-01-01','2025-02-01',freq='MS'))
    p=protocol(); p['models']=['ok']
    with pytest.raises(ValueError,match='changed'):
        run_registered_experiment(Runner(),p,{'ok':lambda train,h,c:{'units':'mom_percent','path':np.zeros(h)}})
    assert not (tmp_path/'metrics.csv').exists()


def test_missing_target_fact_fails_before_model_call(tmp_path,monkeypatch):
    import sirena.evaluation as module
    monkeypatch.setattr(module,'forecast_source_manifest',lambda:{'inflation_data.csv':'fake'})
    monkeypatch.setattr(module,'code_manifest',lambda root:{})
    class Runner:
        output_dir=tmp_path
        def _prepare_data(self):
            self.df_ridge=pd.DataFrame({'Все товары и услуги':100.3},index=pd.date_range('2023-01-01','2025-02-01',freq='MS'))
            self.df_ridge.loc['2025-02-01','Все товары и услуги']=np.nan
    def forbidden(*args):
        pytest.fail('No model should run before target-fact validation')
    with pytest.raises(ValueError,match='Non-finite target facts.*2025-02'):
        run_registered_experiment(Runner(),protocol(),{'ok':forbidden,'broken':forbidden})
    assert not (tmp_path/'forecast_rows.csv').exists()


@pytest.mark.parametrize('units,short', [('mom_index_100',False),('mom_percent',True)])
def test_rejected_adapter_cannot_publish_contributions(tmp_path,monkeypatch,units,short):
    import sirena.evaluation as module
    monkeypatch.setattr(module,'forecast_source_manifest',lambda:{'inflation_data.csv':'fake'})
    monkeypatch.setattr(module,'code_manifest',lambda root:{})
    class Runner:
        output_dir=tmp_path
        def _prepare_data(self):
            self.df_ridge=pd.DataFrame({'Все товары и услуги':100.3},index=pd.date_range('2023-01-01','2025-02-01',freq='MS'))
    def invalid(train,h,c):
        return {'units':units,'path':np.zeros(h-1 if short else h),'contributions':[{'group':1,'contribution':9.}]}
    p=protocol();p['models']=['bad']
    rows,metrics,state=run_registered_experiment(Runner(),p,{'bad':invalid})
    assert state['status']=='COMPLETED_NO_COMMON_SAMPLE'
    assert not state['screening_eligible']
    assert rows.prediction.isna().all()
    assert pd.read_csv(tmp_path/'contributions.csv').empty


def test_budget_preserves_planned_rows_with_explicit_unscored_status(tmp_path,monkeypatch):
    import sirena.evaluation as module
    monkeypatch.setattr(module,'forecast_source_manifest',lambda:{'inflation_data.csv':'fake'})
    monkeypatch.setattr(module,'code_manifest',lambda root:{})
    clock=[0.]
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    class Runner:
        output_dir=tmp_path
        def _prepare_data(self):
            self.df_ridge=pd.DataFrame({'Все товары и услуги':100.3},index=pd.date_range('2023-01-01','2025-02-01',freq='MS'))
    calls=[]
    def slow(train,h,c):
        calls.append(1);clock[0]+=100.
        return {'units':'mom_percent','path':np.zeros(h)}
    rows,metrics,state=run_registered_experiment(Runner(),protocol(),{'ok':slow,'broken':slow})
    assert len(calls)==1 and len(rows)==8
    assert state['status']=='BUDGET_EXHAUSTED' and not state['screening_eligible']
    import json
    assert {f['failure_kind'] for f in json.loads((tmp_path/'failures.json').read_text())}=={'budget_not_evaluated'}
