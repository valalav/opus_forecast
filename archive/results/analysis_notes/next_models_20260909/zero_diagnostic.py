"""Describe unchanged item-months in the CURRENT basket; not a historical test."""
import json
from pathlib import Path
import numpy as np
from sirena.data.micro_basket import load_micro_basket

root=Path('/home/valalav/_projects/sirena-kbr')
basket=load_micro_basket(root/'data','2026-07-01')
history=basket['history'].loc['2023-08-01':'2026-07-01',basket['leaves'].index]
rows=[]
for code,label in [(3,'food'),(2,'nonfood'),(4,'services')]:
    items=basket['leaves'].index[basket['leaves'].Component.eq(code)]
    data=history[items]
    finite=np.isfinite(data)
    unchanged=data.abs().le(.005)&finite
    rows.append({'component':label,'items':len(items),
                 'observed_item_months':int(finite.sum().sum()),
                 'near_zero_item_months':int(unchanged.sum().sum()),
                 'near_zero_share':float(unchanged.sum().sum()/finite.sum().sum())})
payload={'period':'2023-08--2026-07',
         'basket':'2026-01 vintage; descriptive current-basket only, not a backtest',
         'near_zero_definition':'abs(index-100)<=0.005 percentage point',
         'data':'data/kbr_indices.csv','groups':rows}
Path('/tmp/sirena_zero_diagnostic.json').write_text(json.dumps(payload,indent=2))
print(json.dumps(payload,indent=2))
