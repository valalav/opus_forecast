import hashlib
import json
import pandas as pd
import pytest
from sirena.data.access_refresh import refresh_access
from sirena.models.microcomponent import MicrocomponentForecaster


@pytest.fixture
def release(tmp_path):
    stage, data = tmp_path/'release', tmp_path/'data'
    (stage/'extracts').mkdir(parents=True)
    (data/'raw').mkdir(parents=True)
    database = stage/'db_cpi_store.accdb'
    database.write_bytes(b'fixture database identity')
    digest = hashlib.sha256(database.read_bytes()).hexdigest()
    (stage/'manifest.json').write_text(json.dumps({'database_sha256':digest,'public_url':'https://example.invalid'}))
    rows = []
    for month in [6,7]:
        for code, value in [(1,100.82),(2,101.62),(3,100.59),(4,100.02),(11,100.1),(100,100.2)]:
            rows.append({'Day':f'{month:02}/01/26 00:00:00','Region_code':7,'Item_code':code,'MoM':value,'YoY':105.})
    indices = pd.DataFrame(rows)
    weights = pd.DataFrame([{'Day':'01/01/26 00:00:00','Region_code':r,'Item_code':c,'Weight_vertical':v}
                           for r in [7,8] for c,v in [(100,.9),(101,.1)]])
    frames = {'data_indices':indices,'data_seasonalised':indices[indices.Day.str.startswith('06')],
              'weights':weights[weights.Region_code.eq(7)],'weights_all':weights,
              'items_names':pd.DataFrame({'Item_code':[1,2,3,4,11,100,101],
                 'Item_name':['total','nonfood','food','services','sub','observed','missing'],
                 'Item_rosstat':[1,7,6,9000,11,100,101]}),
              'items_structure':pd.DataFrame({'Item_code':[100,101],'Item_type':[5,5],'Item_on':[1,1],
                 'Component':[3,4],'Subcomponent':[11,11],'Primary':[1,0]})}
    hashes = {}
    for name, frame in frames.items():
        path = stage/'extracts'/(name+'.csv')
        frame.to_csv(path,index=False)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (stage/'extract_manifest.json').write_text(json.dumps({'database_sha256':digest,'tables':hashes}))
    pd.DataFrame({'Date':['30.06.2026','31.07.2026'],'mom':[100.82]*2,'Nonprod':[101.62]*2,
                  'Prod':[100.59]*2,'Serv':[100.02]*2}).to_csv(data/'inflation_data.csv',sep=';',decimal=',',index=False)
    for name, label in [('subcomp','Day'),('sub_mom','Date')]:
        (data/'raw'/(name+'.csv')).write_text(label+';11\n')
    (data/'sa_fl.csv').write_bytes(b'newer independent SA fixture')
    return stage, data


def test_preview_is_read_only_and_reports_source_gaps(release):
    stage,data=release
    before={str(p):p.read_bytes() for p in data.rglob('*') if p.is_file()}
    result=refresh_access(stage,data)
    assert result['basket_observed_weight']==pytest.approx(.9)
    assert result['missing_items'][0]['item_code']==101
    assert result['aggregate_crosscheck']['2']['max_abs_difference']==pytest.approx(0)
    assert result['access_sa_last_observation']=='2026-06-01'
    assert before=={str(p):p.read_bytes() for p in data.rglob('*') if p.is_file()}


def test_apply_preserves_sa_all_regions_and_required_empty_item(release):
    stage,data=release
    refresh_access(stage,data,apply=True)
    assert (data/'sa_fl.csv').read_bytes()==b'newer independent SA fixture'
    assert set(pd.read_csv(data/'access_weights.csv').Region_code)=={7,8}
    model=MicrocomponentForecaster()
    loaded=model._load_data(data)
    assert set(loaded.columns)=={100,101}
    assert loaded[101].isna().all()


def test_wrong_aggregate_mapping_fails_before_writes(release):
    stage,data=release
    frame=pd.read_csv(data/'inflation_data.csv',sep=';',decimal=',')
    frame[['Prod','Nonprod']]=frame[['Nonprod','Prod']].to_numpy()
    frame.to_csv(data/'inflation_data.csv',sep=';',decimal=',',index=False)
    with pytest.raises(ValueError,match='Aggregate CPI mismatch'):
        refresh_access(stage,data,apply=True)
    assert not (data/'micro_sprav.csv').exists()


def test_staged_extract_tamper_rejected(release):
    stage,data=release
    with (stage/'extracts/data_indices.csv').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError,match='Extract hash mismatch'):
        refresh_access(stage,data,apply=True)
    assert not (data/'micro_sprav.csv').exists()


def test_no_current_cutoff_relabelling(release):
    stage,data=release
    path=data/'inflation_data.csv'
    path.write_text(path.read_text().replace('31.07.2026','31.08.2026'))
    with pytest.raises(ValueError,match='same cutoff'):
        refresh_access(stage,data)


@pytest.mark.parametrize('horizon',[1,2,3,6,12])
def test_dashboard_selects_same_horizon_as_production_cache(horizon):
    from sirena.data_loader import cached_forecast_point
    payload={'last_data_date':'2026-07-01',
             'forecast_dates':pd.date_range('2026-08-01',periods=12,freq='MS').astype(str).tolist(),
             'forecasts':{'Ridge_ProdProxy':[float(x) for x in range(12)],
                          'NGBoostShock':[float(x+20) for x in range(12)],'Micro':None}}
    target=pd.Timestamp('2026-07-01')+pd.DateOffset(months=horizon)
    assert cached_forecast_point(payload,'Ridge_ProdProxy',target,'2026-07-01')==horizon-1
    assert cached_forecast_point(payload,'NGBoost_Shock',target,'2026-07-01')==horizon+19
    assert pd.isna(cached_forecast_point(payload,'Micro',target,'2026-07-01'))


def test_dashboard_rejects_relabelled_cache_cutoff():
    from sirena.data_loader import cached_forecast_point,DataFreshnessError
    with pytest.raises(DataFreshnessError,match='cutoff'):
        cached_forecast_point({'last_data_date':'2026-05-01'},'Ridge','2026-08-01','2026-07-01')
