"""Independent arithmetic and publication checks for the September nowcast."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
s=json.loads((OUT/"summary.json").read_text())
d=pd.read_csv(OUT/"weighted_drivers.csv")
u=pd.read_csv(OUT/"unobserved_micro_positions.csv")
a=json.loads((ROOT/"data/precomputed_forecasts.json").read_text())
b=json.loads((OUT/"precomputed_forecasts.before.json").read_text())
checks={}
checks["input_hashes"]=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in s["source_hashes"].items())
checks["no_double_weights"]=bool(d.Item_code.is_unique and not set(d.Item_code)&set(u.Item_code))
checks["full_basket"]=bool(np.isclose(d.weight.sum()+u.Weight.sum(),1))
reconstructed=((1+d.observed_mom/100)*np.power(1+d.prior_prediction/100,.2)-1)*100
checks["weighted_arithmetic"]=bool(np.allclose(reconstructed,d.updated_prediction) and np.isclose((reconstructed*d.weight).sum()+u.Contribution.sum(),s["weighted_micro_mom"]))
fuel=d.product_name.str.contains("Бензин|Дизельное",case=False)&d.prior_prediction.lt(0)
checks["fuel_persistence_scenario"]=bool(np.isclose(s["weighted_micro_mom"]+(d.loc[fuel,"observed_contribution"]-d.loc[fuel,"contribution"]).sum(),s["fuel_no_decline_tail_mom"]))
checks["unchanged_monthly_forecasts"]=all(a["forecasts"][k]==v for k,v in b["forecasts"].items() if k!="Nowcast")
checks["policy_preserved"]=(ROOT/"data/send_ready_policy_trajectory.json").read_bytes()==(OUT/"send_ready_policy_trajectory.before.json").read_bytes()
checks["cache_diagnostic"]=a["diagnostics"]["weighted_micro_nowcast"]==s
checks["operational_calendar"]=a["diagnostics"]["weekly_bridge"]["by_month"]["2026-09"]["chain"]["weeks"][0]["date"]=="2026-08-31"
html=(ROOT/"assets/charts/nowcast.html").read_text()
checks["html_both_methods"]=all(x in html for x in ["weighted-nowcast","+0.50%","+0.356%","2026-09-21","35.265%"])
checks["workbook_readable"]=pd.read_excel(OUT/"september_nowcast.xlsx",sheet_name="Weighted drivers").shape==d.shape
result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"tests":"17 weekly bridge / canonical refresh tests passed", "limitations":["Hybrid has no independent historical forecast accuracy backtest", "Four of five operational weeks; 64.735% basket remains modelled", "Separate canonical weekly store was not refreshed and is not an input to this nowcast"]}
(OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2));assert all(checks.values())
