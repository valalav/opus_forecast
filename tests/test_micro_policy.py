import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from sirena.data.micro_basket import load_micro_basket
from sirena.data_loader import DataFreshnessError
from sirena.models.microcomponent import MicrocomponentForecaster


@pytest.fixture
def inputs(tmp_path):
    dates=pd.date_range('2019-01-01','2026-07-01',freq='MS')
    rows=[]
    for t,date in enumerate(dates):
        base=.15+.2*np.sin(2*np.pi*date.month/12)+.01*np.cos(t)
        for code,offset in [(1,0),(11,.1),(12,-.1),(100,.2),(101,1.),(102,-.2),(103,.3)]:
            value=100+base+offset
            if code==101 and date<pd.Timestamp('2025-01-01'):
                continue
            if code==102 and date==pd.Timestamp('2025-07-01'):
                continue
            rows.append({'Item_code':code,'Region_code':7,'Day':date.strftime('%m/%d/%y 00:00:00'),'MoM':value})
    pd.DataFrame(rows).to_csv(tmp_path/'kbr_indices.csv',index=False)
    structure=pd.DataFrame({'Item_code':[1,11,12,100,101,102,103],
        'Item_type':[1,3,3,5,5,5,7],'Subcomponent':[1,11,12,11,11,12,11],
        'Item_on':[1,1,1,1,1,1,0]})
    structure.to_csv(tmp_path/'items_structure.csv',index=False)
    weights=[]
    for year,leaves in [(2024,{100:.5,103:.05,102:.4}), (2025,{100:.55,101:.05,102:.4}),
                         (2026,{100:.1,101:.5,102:.4})]:
        for code,weight in {11:.6,12:.4,**leaves}.items():
            weights.append({'Item_code':code,'Region_code':7,'Day':f'01/01/{year%100} 00:00:00','Weight_vertical':weight})
    pd.DataFrame(weights).to_csv(tmp_path/'access_weights.csv',index=False)
    pd.DataFrame({'Item_code':structure.Item_code,'Item_name':structure.Item_code.astype(str)}).to_csv(tmp_path/'items_names.csv',index=False)
    df=pd.DataFrame({'Все товары и услуги':100.2},index=dates)
    return tmp_path,df


def test_dated_basket_retains_retired_leaves_and_residual(inputs):
    root,df=inputs
    old=load_micro_basket(root,'2024-12-01')
    assert 103 in old['leaves'].index and 101 not in old['leaves'].index
    assert old['residual'].sum()==pytest.approx(.05)
    assert old['weight_vintage']==pd.Timestamp('2024-01-01')
    new=load_micro_basket(root,'2025-07-01')
    assert 101 in new['leaves'].index and 103 not in new['leaves'].index
    assert new['residual'].sum()==pytest.approx(0)


def test_missing_and_short_items_preserve_total_weight_and_observations(inputs):
    root,df=inputs
    before=hashlib.sha256((root/'kbr_indices.csv').read_bytes()).hexdigest()
    model=MicrocomponentForecaster(data_dir=root).fit(df.loc[:'2025-07-01'])
    path=model.forecast(12)
    assert np.isfinite(path).all()
    assert model.fallbacks[102]['reason']=='missing_current_observation'
    assert model.fallbacks[101]['reason']=='short_history'
    assert model.fallbacks[101]['offset']>0
    assert model.coverage['native_model_weight']==pytest.approx(.55)
    assert model.coverage['fallback_item_weight']==pytest.approx(.45)
    assert model.coverage['forecast_weight']==pytest.approx(1)
    bymonth=model.last_item_forecasts.groupby('date').agg(weight=('Weight','sum'),forecast=('Contribution','sum'))
    assert np.allclose(bymonth.weight,1)
    assert np.allclose(bymonth.forecast,path)
    assert hashlib.sha256((root/'kbr_indices.csv').read_bytes()).hexdigest()==before


def test_forecast_and_predict_share_exact_steps_without_mutating_fitted_state(inputs):
    root,df=inputs
    train=df.loc[:'2025-06-01']
    model=MicrocomponentForecaster(horizon=12,data_dir=root).fit(train)
    path=model.forecast(12)
    for h in [1,2,3,6,12]:
        result=model.predict(train,train.index.max()+pd.DateOffset(months=h))
        assert result['prediction']-100==pytest.approx(path[h-1])
    assert np.allclose(model.forecast(12),path)
    with pytest.raises(ValueError,match='follow'):
        model.predict(train,train.index.max())


def test_post_cutoff_observations_and_future_weights_cannot_change_forecast(inputs):
    root,df=inputs
    cutoff=pd.Timestamp('2025-07-01')
    first=MicrocomponentForecaster(data_dir=root).fit(df.loc[:cutoff]).forecast(12)
    data=pd.read_csv(root/'kbr_indices.csv')
    dates=pd.to_datetime(data.Day,format='%m/%d/%y %H:%M:%S')
    data.loc[dates.gt(cutoff),'MoM']=800
    data.to_csv(root/'kbr_indices.csv',index=False)
    weights=pd.read_csv(root/'access_weights.csv')
    weights.loc[weights.Day.str.contains('/26 '),'Weight_vertical']=42.
    weights.to_csv(root/'access_weights.csv',index=False)
    model=MicrocomponentForecaster(data_dir=root).fit(df.loc[:cutoff])
    assert np.allclose(first,model.forecast(12))
    assert model.coverage['weight_vintage']=='2025-01-01'


def test_missing_parent_is_still_a_hard_failure(inputs):
    root,df=inputs
    data=pd.read_csv(root/'kbr_indices.csv')
    data=data[~(data.Item_code.eq(12)&data.Day.eq('07/01/25 00:00:00'))]
    data.to_csv(root/'kbr_indices.csv',index=False)
    with pytest.raises(DataFreshnessError,match='2025-07'):
        MicrocomponentForecaster(data_dir=root).fit(df.loc[:'2025-07-01'])


def test_overlap_is_rejected_instead_of_normalized_away(inputs):
    root,df=inputs
    weights=pd.read_csv(root/'access_weights.csv')
    weights.loc[weights.Item_code.eq(100),'Weight_vertical']=.9
    weights.to_csv(root/'access_weights.csv',index=False)
    with pytest.raises(DataFreshnessError,match='Overlapping'):
        load_micro_basket(root,'2025-07-01')


@pytest.mark.parametrize('extended',[False,True])
def test_fast_recursive_feature_row_matches_training_features(extended):
    model=MicrocomponentForecaster()
    series=pd.Series(np.linspace(-1,2,40)**2,index=pd.date_range('2020-01-01',periods=40,freq='MS'))
    frame=model._create_features(series,extended)
    columns=[c for c in frame if c!='y']
    fast=model._feature_row(series.values,series.index[-1],columns)
    assert np.allclose(fast,frame[columns].iloc[[-1]].values)


def test_common_comparison_uses_holdout_and_rejects_source_revision(tmp_path, monkeypatch):
    import sirena.data_loader as loader
    summary = {'source_manifest': {'source':'hash'}, 'horizons':[1,2,12],
               'validation_start':'2025-08-01', 'selected_on_development_h1':'MicroParentSeasonal',
               'models':['Ridge','Huber','Ridge_ProdProxy','MicroParentRidge','MicroParentSeasonal']}
    (tmp_path/'common_summary.json').write_text(json.dumps(summary))
    rows=[]
    for date,h in [('2025-07-01',1),('2025-08-01',1),('2025-09-01',1),('2025-08-01',2)]:
        row={'Date':date,'cutoff':'2025-06-01','horizon':h,'Actual':.1}
        row.update({name:.2 for name in summary['models']})
        if date=='2025-09-01': row['MicroParentRidge']=np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path/'common_predictions.csv',index=False)
    monkeypatch.setattr(loader,'forecast_source_manifest',lambda data_dir=None: {'source':'hash'})
    result=loader.load_common_backtest(1,tmp_path)
    assert list(result.Date)==[pd.Timestamp('2025-08-01')]
    assert list(result.columns)==['Date','Actual','Ridge','Huber','Ridge_ProdProxy','Micro']
    assert result.attrs['common_origins']
    assert loader.load_common_backtest(3,tmp_path) is None
    monkeypatch.setattr(loader,'forecast_source_manifest',lambda data_dir=None: {'source':'revised'})
    assert loader.load_common_backtest(1,tmp_path) is None
