import pandas as pd
import pytest
from scripts.rebuild_weekly_canonical import append_verified, build

def history():
    return pd.DataFrame({'date':pd.to_datetime(['2026-08-03','2026-08-10']), 'product_code':[111,111], 'product_name':['item','item'], 'price':[100.,101.], 'price_prev_week':[float('nan'),100.], 'wow_growth':[float('nan'),1.]})

def fresh():
    return pd.DataFrame({'date':pd.to_datetime(['2026-08-10','2026-08-17','2026-08-24','2026-08-31']), 'product_name':['item']*4, 'price':[101.,102.,103.,104.], 'accounting_month':pd.to_datetime(['2026-08-01']*3+['2026-09-01'])})

def test_append_retains_observed_dates_and_history():
    old=history();new,audit=append_verified(old,fresh())
    pd.testing.assert_frame_equal(new.iloc[:len(old)],old)
    assert new.date.max()==pd.Timestamp('2026-08-31')
    assert audit['added_rows']==3
    assert new.iloc[-1].wow_growth==pytest.approx((104/103-1)*100)
    assert 'accounting_month' not in new

def test_repeated_run_is_byte_identical(tmp_path,monkeypatch):
    import scripts.rebuild_weekly_canonical as m
    path=tmp_path/'weekly.csv';history().to_csv(path,index=False)
    source=tmp_path/'source';source.write_text('fixture')
    monkeypatch.setattr(m,'load_semicolon_weekly_prices',lambda _:fresh())
    original=path.read_bytes()
    build(canonical=path,fresh=source);a=path.read_bytes()
    assert a.startswith(original)
    build(canonical=path,fresh=source);assert path.read_bytes()==a
    build(canonical=path,fresh=source);assert path.read_bytes()==a

def test_overlap_conflict_rejected():
    f=fresh();f.loc[0,'price']=999
    with pytest.raises(ValueError,match='Overlapping prices disagree'):append_verified(history(),f)

def test_missing_week_is_not_one_week_growth():
    f=fresh().drop(index=1);new,_=append_verified(history(),f)
    assert pd.isna(new[new.date==pd.Timestamp('2026-08-24')].iloc[0].wow_growth)

def test_duplicate_and_unknown_product_rejected():
    with pytest.raises(ValueError,match='duplicate'):append_verified(history(),pd.concat([fresh(),fresh().iloc[:1]]))
    f=fresh();f.loc[3,'product_name']='new item'
    with pytest.raises(ValueError,match='mapping'):append_verified(history(),f)

def test_original_missing_observations_preserved():
    old=history();old.loc[0,'price']=float('nan')
    new,_=append_verified(old,fresh())
    assert pd.isna(new.loc[0,'price'])
    pd.testing.assert_frame_equal(new.iloc[:len(old)],old)


def test_operational_source_race_rejected(tmp_path,monkeypatch):
    import scripts.rebuild_weekly_canonical as m
    path=tmp_path/'weekly.csv';history().to_csv(path,index=False);old=path.read_bytes()
    source=tmp_path/'source';source.write_text('v1')
    def raced(_):
        source.write_text('v2');return fresh()
    monkeypatch.setattr(m,'load_semicolon_weekly_prices',raced)
    with pytest.raises(ValueError,match='changed while reading'):
        build(canonical=path,fresh=source)
    assert path.read_bytes()==old
