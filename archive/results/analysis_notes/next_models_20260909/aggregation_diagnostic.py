"""Read-only diagnostic: fixed versus price-updated annual group weights."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sirena.data.micro_basket import load_micro_basket

root=Path('/home/valalav/_projects/sirena-kbr')
summary=json.loads((root/'archive/results/micro_policy_20260909/common_summary.json').read_text())
dates=sorted({r['cutoff'] for r in summary['coverage']})
rows=[]
for date in dates:
    date=pd.Timestamp(date)
    basket=load_micro_basket(root/'data',date)
    history=basket['history']
    group_weights=basket['groups'].Weight_vertical
    codes=group_weights.index
    previous=history.loc[f'{date.year}-01-01':date-pd.DateOffset(months=1),codes]
    if date.month==1: previous=previous.iloc[:0]
    assert len(previous)==date.month-1
    assert np.isfinite(previous).all().all()
    level=(1+previous/100).prod(axis=0)
    updated=group_weights*level
    updated=updated/updated.sum()
    actual=history.at[date,1]
    fixed=float((group_weights*history.loc[date,codes]).sum())
    chained=float((updated*history.loc[date,codes]).sum())
    rows.append({'Date':str(date.date()),'Actual':actual,'fixed45':fixed,'updated45':chained,
                 'fixed_error':fixed-actual,'updated_error':chained-actual})
frame=pd.DataFrame(rows)
frame.to_csv('/tmp/sirena_aggregation_diagnostic.csv',index=False)
print('35-origin FACT reconstruction only; not a forecast or causal attribution')
print(frame[['fixed_error','updated_error']].agg(['mean',lambda s:s.abs().mean(),lambda s:s.abs().max()]).to_string())
print(frame.tail(3).to_string(index=False))
