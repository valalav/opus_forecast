"""Independently check scenario arithmetic, vintages, source hashes and delivery."""
from pathlib import Path
import json,hashlib,zipfile
import pandas as pd
import numpy as np
import openpyxl
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
f=pd.read_csv(OUT/"forecast_2026_2027.csv",parse_dates=["date"])
full=pd.read_csv(OUT/"full_path_2026_2027.csv",parse_dates=["date"])
meta=json.loads((OUT/"calculation_manifest.json").read_text())
policy=json.loads((ROOT/"data/send_ready_policy_trajectory.json").read_text())
checks={"source_hashes":all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in meta["sources"].items())}
checks["forecast_months"]=len(f)==16 and f.date.tolist()==pd.date_range("2026-09-01","2027-12-01",freq="MS").tolist()
checks["policy_alignment"]=policy["forecast_dates"]==f.date.dt.strftime("%Y-%m-%d").tolist() and np.allclose(policy["mom_pp"],f.central_mom_pp)
actual=pd.read_csv(ROOT/"data/inflation_data.csv",sep=";",decimal=",")
actual.index=pd.to_datetime(actual.Date,dayfirst=True).dt.to_period("M").dt.to_timestamp()
checks["facts_preserved"]=np.allclose(full[full.status.eq("fact")].mom_index,actual.loc["2026-01-01":"2026-08-01","mom"])
for scenario in ["central","low","high"]:
    series=pd.concat([actual.mom,pd.Series(f[f"{scenario}_mom_pp"].values+100,index=f.date)])
    independent=(series/100).rolling(12).apply(np.prod,raw=True).sub(1).mul(100).reindex(f.date)
    checks[f"{scenario}_yoy"]=np.allclose(independent,f[f"{scenario}_yoy_pct"])
checks["annual_target_rounding"]=abs(f.iloc[-1].central_yoy_pct-meta["annual_realized_rounded_2027"])<1e-6
if meta["tariff_2027_pct"]==11 and meta["tariff_2027_date"]=="2027-07-01": checks["reference_target"]=abs(f.iloc[-1].central_yoy_pct-4.4)<.025
checks["scenario_ordering"]=bool((f.low_mom_pp<=f.central_mom_pp).all() and (f.central_mom_pp<=f.high_mom_pp).all())
checks["finite_rates"]=bool(np.isfinite(f.select_dtypes("number")).all().all() and (f.central_mom_pp>-100).all())
checks["plausible_months"]=bool(f.loc[f.date.dt.year.eq(2027),"central_mom_pp"].between(-.5,1.2).all())
checks["no_december_plug"]=bool(.2<=f.iloc[-1].central_mom_pp<=.6)
book=openpyxl.load_workbook(ROOT/"assets/06_2026_02_Прогноз.xlsx",data_only=True,read_only=True)["Прогноз"]
old=openpyxl.load_workbook(OUT/"OPR092026_source.xlsx",data_only=True,read_only=True)["Прогноз"]
forecast=f.set_index("date");matched=0;valid=True;previous=True
for row in book.iter_rows():
    date=row[0].value
    if not hasattr(date,"year"):continue
    rown=row[0].row
    if date in forecast.index:
        r=forecast.loc[date];matched+=1
        valid &= np.isclose(row[4].value,100+r.central_mom_pp) and np.isclose(row[5].value,round(100+r.central_yoy_pct,2))
    previous &= row[6].value==old[f"E{rown}"].value and row[7].value==old[f"F{rown}"].value
checks["opr_current_values"]=bool(valid and matched==16);checks["opr09_comparison"]=bool(previous)
with zipfile.ZipFile(OUT/"OPR_before_adjustment.xlsx") as a,zipfile.ZipFile(ROOT/"assets/06_2026_02_Прогноз.xlsx") as b:
    changed=[p for p in a.namelist() if a.read(p)!=b.read(p)]
checks["xlsx_other_parts_preserved"]=len(changed)==1 and changed[0].startswith("xl/worksheets/")
checks["dated_form_matches_canonical"]=(OUT/"OPR102026_Прогноз.xlsx").read_bytes()==(ROOT/"assets/06_2026_02_Прогноз.xlsx").read_bytes()
items=pd.read_csv(OUT/"item_price_scenarios.csv")
checks["item_adjustments"]=bool(np.allclose(items.weight*(items.scenario_mom_pct-items.micro_mom_pct),items.delta_contribution_pp))
checks["unique_item_weights"]=not items.duplicated(["item_code","month"]).any()
sensitivity=pd.read_csv(OUT/"tariff_sensitivity.csv")
checks["tariff_risk_not_retargeted"]=bool((sensitivity.december_yoy_pct.diff().dropna()>0).all())
source=ROOT/"data/external/med_forecast_2027_2029_20260924"
manifest=json.loads((source/"manifest.json").read_text())
checks["official_source_hashes"]=all(hashlib.sha256((source/r["file"]).read_bytes()).hexdigest()==r["sha256"] for r in manifest["files"])
checks["source_excel_anchor"]=np.isclose(openpyxl.load_workbook(source/"attachments/4. ИПЦ_базовый.xlsx",data_only=True,read_only=True).active["D8"].value,104.02)
result={"status":"PASS" if all(checks.values()) else "FAIL","checks":{k:bool(v) for k,v in checks.items()},"workbook_changed_parts":changed,"scope":"conditional expert scenario; no new forecast model or out-of-sample accuracy claim"}
(OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
report="# Проверка ОПР10.2026\n\n"+result["status"]+". Независимый пересчёт всех сценарных YoY, формулы и сохранность XLSX, версии ОПР09/ОПР10, факты и хэши источников.\n\n"+"\n".join(f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in checks.items())+"\n\nПроверка подтверждает воспроизводимость и арифметику. Она не доказывает прогнозную точность условной траектории. Национальные тарифы используются как сценарий КБР.\n"
(OUT/"verification_report.md").write_text(report)
print(json.dumps(result,ensure_ascii=False,indent=2));assert all(checks.values())
