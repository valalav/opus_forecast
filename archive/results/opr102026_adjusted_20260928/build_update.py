"""Conditional OPR10 scenario, reusing the existing loaders and publication schema.
Not a new statistical model: the 2027 annual anchor is an explicit user scenario.
"""
from pathlib import Path
import json, hashlib, sys, shutil
import numpy as np
import pandas as pd
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from sirena.data_loader import load_model_data
from sirena.data.weekly_bridge import load_semicolon_weekly_prices
SOURCE=ROOT/"data/external/med_forecast_2027_2029_20260924"

def item_scenarios():
    w=load_semicolon_weekly_prices(ROOT/"data/Сравнение еженедельных цен_01.csv")
    assert w.date.max()==pd.Timestamp("2026-09-21")
    mapping=pd.read_csv(ROOT/"archive/results/september_nowcast_20260928/item_mapping.csv")
    micro=pd.read_csv(ROOT/"data/micro_forecast_components.csv")
    subset=mapping[mapping.matched & mapping.product_name.str.contains("Картофель|Яблоки|Огурцы|Помидоры|Лук|Морковь|Капуста|Бананы|Яйца")]
    rows=[]
    for r in subset.itertuples():
        d=w[w.product_name.eq(r.product_name)].sort_values("date").copy()
        d["year"]=d.date.dt.year;d["month"]=d.date.dt.month
        panel=d.groupby(["year","month"]).tail(1).pivot(index="year",columns="month",values="price")
        panel=panel[panel.index<2026]
        m=micro[micro.Item_code.eq(int(float(r.item_code)))].set_index("date").Prediction
        prior=m.loc[["2026-10-01","2026-11-01"]].values
        cumulative=np.cumprod(1+prior/100)
        analog=[(panel[month]/panel[9]).dropna() for month in [10,11]]
        assert all(len(a)==3 for a in analog)
        # Equal log-weight between existing Micro and the three-year analogue median.
        # This is a disclosed scenario shrinkage, not a fitted optimal estimator.
        ratios=np.array([a.median() for a in analog])
        blended=np.sqrt(cumulative*ratios)
        new_mom=100*(blended/np.r_[1,blended[:-1]]-1)
        last=float(d.iloc[-1].price)
        sept_end=last*(1+m["2026-09-01"]/100)**.2
        for k,month in enumerate([10,11]):
            rows.append(dict(item_code=int(float(r.item_code)),name=r.product_name,weight=r.weight,
                month=f"2026-{month:02d}-01",last_observed_price=last,conditional_september_end_price=sept_end,
                micro_mom_pct=prior[k],scenario_mom_pct=new_mom[k],
                delta_contribution_pp=r.weight*(new_mom[k]-prior[k]),
                conditional_price=sept_end*blended[k],analog_years="2023;2024;2025",
                analog_ratio_median=ratios[k],reference_price_p25=sept_end*analog[k].quantile(.25),
                reference_price_p75=sept_end*analog[k].quantile(.75),
                below_reference_p25=bool(blended[k]<analog[k].quantile(.25)),above_reference_p75=bool(blended[k]>analog[k].quantile(.75))))
    table=pd.DataFrame(rows);table.to_csv(OUT/"item_price_scenarios.csv",index=False)
    return table

def main():
    for source,name in [(ROOT/"data/send_ready_policy_trajectory.json","policy.before.json"),(ROOT/"assets/06_2026_02_Прогноз.xlsx","OPR_before_adjustment.xlsx")]:
        if not (OUT/name).exists():shutil.copy2(source,OUT/name)
    raw=load_model_data("raw",include_macro=True)
    assert raw.index.max()==pd.Timestamp("2026-08-01")
    observed=raw["Все товары и услуги"]-100
    cache=json.loads((ROOT/"data/precomputed_forecasts.json").read_text())
    dates=pd.date_range("2026-09-01","2027-12-01",freq="MS")
    assert cache["forecast_dates"]==dates.strftime("%Y-%m-%d").tolist()
    baseline=np.array(cache["forecasts"]["Ensemble"])
    old=pd.read_csv(OUT/"opr092026_reference.csv",parse_dates=["date"]).set_index("date")
    recent=pd.read_csv(ROOT/"archive/results/forecast_update_20260928/forecast_2026_2027.csv",parse_dates=["date"]).set_index("date")
    weights=pd.read_csv(ROOT/"data/access_weights.csv")
    utility_weight=float(weights.loc[weights.Region_code.eq(7)&weights.Item_code.eq(279)&weights.Day.eq("01/01/26 00:00:00"),"Weight_vertical"].iloc[0])
    ix=pd.read_csv(ROOT/"data/kbr_indices.csv",parse_dates=["Date"])
    utility=ix[ix.Item_code.eq(279)].set_index("Date").MoM-100
    h=utility[(utility.index.year>=2016)&(utility.index.year<=2025)]
    embedded_july=float(h[h.index.month==7].median())
    embedded_oct=float(h[h.index.month==10].median())
    tariff=pd.read_csv(SOURCE/"tariff_calendar.csv")
    rate26=float(tariff.loc[tariff.code.eq("utilities_total")&tariff.year.eq(2026),"rate_pct"].iloc[0])
    rate27=float(tariff.loc[tariff.code.eq("utilities_total")&tariff.year.eq(2027),"rate_pct"].iloc[0])
    event26=pd.Timestamp(tariff.loc[tariff.code.eq("utilities_total")&tariff.year.eq(2026),"effective_date"].iloc[0])
    event27=pd.Timestamp(tariff.loc[tariff.code.eq("utilities_total")&tariff.year.eq(2027),"effective_date"].iloc[0])
    assert event26 in dates and event27 in dates and event27.year==2027
    event26_idx=dates.get_loc(event26);event27_idx=event27.month-1
    items=item_scenarios()
    adjustment=items.groupby("month").delta_contribution_pp.sum()
    item_delta=np.r_[0,adjustment["2026-10-01"],adjustment["2026-11-01"],np.zeros(13)]
    direct=np.zeros(16);direct[event26_idx]=utility_weight*(rate26-float(h[h.index.month==event26.month].median()))
    # One total fuel/logistics/rail cost budget. Do not add the Micro fuel-floor
    # sensitivity again: Micro is NOT in Ensemble and no exact attribution exists.
    cost=np.r_[0,.15,.10,.06,np.zeros(12)]
    central=np.round(baseline+direct+cost+item_delta,2);central[0]=.50
    # Remove the approximate utility effect before estimating a seasonal shape.
    history=pd.DataFrame({"mom":observed,"utility":utility.reindex(observed.index)})
    history["year"]=history.index.year;history["month"]=history.index.month
    history=history[history.year.isin([2016,2017,2018,2019,2023,2024,2025])].copy()
    assert history.shape[0]==84 and not history.isna().any().any()
    history["nonutility_contribution"]=history.mom-utility_weight*history.utility
    history["demeaned"]=history.nonutility_contribution-history.groupby("year").nonutility_contribution.transform("mean")
    hist_shape=history.groupby("month").demeaned.median().values
    hist_shape-=hist_shape.mean()
    prior=old.loc[pd.date_range("2027-01-01","2027-12-01",freq="MS"),"opr09_mom_pp"].values
    old_shape=prior-prior.mean()
    model_shape=baseline[4:].copy();model_shape[6]-=utility_weight*embedded_july;model_shape-=model_shape.mean()
    shape=.30*old_shape+.50*hist_shape+.20*model_shape
    persistence=np.linspace(.06,-.06,12)
    core_shape=shape+persistence
    def conditional_path(target,rate,event_index=6):
        reference=core_shape.copy();reference[event_index]+=utility_weight*rate
        shift=brentq(lambda c:np.prod(1+(reference+c)/100)-(1+target/100),-2,2)
        return reference+shift,shift
    target27=4.4
    # Freeze the reference calibration when the tariff calendar is edited.
    # A higher tariff must raise CPI, not silently force lower other prices.
    reference_rate27=11.0
    continuous,shift=conditional_path(target27,reference_rate27)
    continuous[6]-=utility_weight*reference_rate27
    continuous[event27_idx]+=utility_weight*rate27
    central[4:]=np.round(continuous,2)
    low=central.copy();high=central.copy()
    low[:4]=[.25,1.12,.65,.71];high[:4]=[.75,1.67,1.20,1.21]
    low[4:]=np.round(conditional_path(3.8,9,event27_idx)[0],2)
    high[4:]=np.round(conditional_path(5.2,13,event27_idx)[0],2)
    f=pd.DataFrame(dict(date=dates,baseline_mom_pp=baseline,opr09_mom_pp=old.reindex(dates).opr09_mom_pp.values,
        previous_28sep_mom_pp=recent.reindex(dates).central_mom_pp.values,utility_adjustment_pp=direct,
        incremental_cost_pp=cost,item_adjustment_pp=item_delta,central_mom_pp=central,low_mom_pp=low,high_mom_pp=high))
    f["utility_direct_pp"]=0.;f.loc[event26_idx,"utility_direct_pp"]=utility_weight*rate26;f.loc[4+event27_idx,"utility_direct_pp"]=utility_weight*rate27
    f["target_conditioning_adjustment_pp"]=0.;f.loc[4:,"target_conditioning_adjustment_pp"]=central[4:]-baseline[4:]
    for scenario in ["central","low","high"]:
        path=pd.concat([observed,pd.Series(f[f"{scenario}_mom_pp"].values,index=dates)])
        yoy=(1+path/100).rolling(12).apply(np.prod,raw=True).sub(1).mul(100)
        f[f"{scenario}_yoy_pct"]=yoy.reindex(dates).values
    f["central_mom_index"]=100+central
    f["opr09_yoy_original_pct"]=old.reindex(dates).opr09_yoy_pct.values
    old_rebased=pd.concat([observed,pd.Series(f.opr09_mom_pp.values,index=dates)])
    f["opr09_yoy_with_current_facts_pct"]=(1+old_rebased/100).rolling(12).apply(np.prod,raw=True).sub(1).mul(100).reindex(dates).values
    f.to_csv(OUT/"forecast_2026_2027.csv",index=False,float_format="%.8f")
    actual=observed[observed.index.year==2026]
    export=pd.concat([pd.DataFrame(dict(date=actual.index,mom_pp=actual.values,status="fact")),pd.DataFrame(dict(date=dates,mom_pp=central,status="forecast"))],ignore_index=True)
    all_path=pd.concat([observed,pd.Series(central,index=dates)])
    export["mom_index"]=export.mom_pp+100
    export["yoy_pct"]=export.date.map((1+all_path/100).rolling(12).apply(np.prod,raw=True).sub(1).mul(100))
    export.to_csv(OUT/"full_path_2026_2027.csv",index=False,float_format="%.8f")
    decomposition=pd.DataFrame(dict(month=range(1,13),old_opr_shape=old_shape,historical_shape=hist_shape,model_shape=model_shape,
        combined_shape=shape,persistence=persistence,common_calibration=shift,utility_direct=np.eye(1,12,event27_idx)[0]*utility_weight*rate27,
        unrounded_mom=continuous,central_mom=central[4:]))
    decomposition.to_csv(OUT/"2027_conditioning.csv",index=False)
    # Freeze underlying prices to show tariff risk; do NOT recalibrate to target.
    sensitivity=[]
    for rate in [9,11,13]:
        path=central[4:].copy();path[event27_idx]+=utility_weight*(rate-rate27)
        sensitivity.append(dict(july_tariff_pct=rate,event_month=event27.strftime("%Y-%m"),july_mom_pct=path[6],event_month_mom_pct=path[event27_idx],december_yoy_pct=100*(np.prod(1+path/100)-1),other_prices="held fixed"))
    sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(OUT/"tariff_sensitivity.csv",index=False)
    with pd.ExcelWriter(OUT/"forecast_calculations.xlsx",engine="openpyxl") as writer:
        for name,table in [("Факт и прогноз",export),("ОПР09 и новый вариант",f),("Товарные уровни",items),("Калибровка 2027",decomposition),("Тарифный календарь",tariff),("Риск тарифов",sensitivity),("РФ ориентиры",pd.read_csv(SOURCE/"national_cpi_anchors.csv"))]:table.to_excel(writer,sheet_name=name,index=False)
        for sheet in writer.book:
            sheet.freeze_panes="B2";sheet.auto_filter.ref=sheet.dimensions
            for col in sheet.columns:sheet.column_dimensions[col[0].column_letter].width=min(38,max(16,len(str(col[0].value))+2))
    sources=["data/inflation_data.csv","data/kbr_indices.csv","data/access_weights.csv","data/precomputed_forecasts.json","data/micro_forecast_components.csv","data/Сравнение еженедельных цен_01.csv",str((SOURCE/"manifest.json").relative_to(ROOT)),str((SOURCE/"tariff_calendar.csv").relative_to(ROOT)),str((OUT/"OPR092026_source.xlsx").relative_to(ROOT))]
    meta=dict(as_of="2026-09-28",cutoff="2026-08",weekly_cutoff="2026-09-21",prior_opr="092026",current_opr="102026",prior_git_commit="ed25ddf663d982795f16458c4418ddb367fcb33a",
        utility_weight=utility_weight,embedded_july_rate=embedded_july,embedded_october_rate=embedded_oct,tariff_2026_pct=rate26,tariff_2027_pct=rate27,
        regional_tariff_status="national scenario; KBR order not verified",tariff_2026_date=str(event26.date()),tariff_2027_date=str(event27.date()),annual_target_2027=target27,annual_realized_rounded_2027=float(f.iloc[-1].central_yoy_pct),
        annual_anchor_status="conditional user scenario, not an estimated forecast or guarantee",reference_tariff_for_calibration_pct=reference_rate27,seasonal_weights=dict(opr09=.30,normal_year_history=.50,ensemble=.20),
        seasonal_history_years=[2016,2017,2018,2019,2023,2024,2025],model_status=cache["model_status"],sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    (OUT/"calculation_manifest.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n")
    policy=dict(generated_at="2026-09-28",latest_official_month="2026-08-01",forecast_dates=dates.strftime("%Y-%m-%d").tolist(),mom_pp=central.tolist(),scenario="opr102026_conditional_target_2027",source_artifact=str((OUT/"REPORT.md").relative_to(ROOT)),description="OPR09 comparison; September nowcast; item price scenarios; MED tariff calendar; conditional 2027 anchor 4.4%.")
    (ROOT/"data/send_ready_policy_trajectory.json").write_text(json.dumps(policy,ensure_ascii=False,indent=2)+"\n")
    make_chart(f)
    print(f[["date","central_mom_pp","central_yoy_pct","opr09_mom_pp"]].round(3).to_string(index=False))

def make_chart(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for ax,suffix,label in [(axes[0],"mom_pp","% к предыдущему месяцу"),(axes[1],"yoy_pct","% к тому же месяцу прошлого года")]:
        ax.fill_between(f.date,f[f"low_{suffix}"],f[f"high_{suffix}"],alpha=.2,color="#197b75",label="Условные сценарии")
        ax.plot(f.date,f[f"central_{suffix}"],marker="o",color="#197b75",label="ОПР10: скорректированный")
        old=f.opr09_mom_pp if suffix=="mom_pp" else f.opr09_yoy_with_current_facts_pct
        ax.plot(f.date,old,linestyle="--",color="#8a6a9c",label="ОПР09: с актуальными фактами")
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[1].axhline(4,color="#999999",linestyle=":",label="Цель 4%")
    axes[0].set_title("КБР · ОПР10.2026 · условный выход к 4,4% в 2027 году")
    axes[0].legend(fontsize=8);fig.autofmt_xdate();fig.tight_layout();fig.savefig(OUT/"forecast.png",dpi=170);plt.close(fig)
    fig=go.Figure()
    for col,name in [("central_mom_pp","ОПР10"),("opr09_mom_pp","ОПР09"),("previous_28sep_mom_pp","Первый вариант 28.09"),("baseline_mom_pp","Модельный Ensemble")]:fig.add_scatter(x=f.date,y=f[col],mode="lines+markers",name=name)
    fig.update_layout(title="КБР: сравнение траекторий; 2027 — условный сценарий",yaxis_title="% м/м",template="plotly_white")
    fig.write_html(OUT/"forecast.html",include_plotlyjs=True)

if __name__=="__main__":main()
