"""Dated September diagnostic: audited weekly indices + existing Micro basket."""
from pathlib import Path
import sys, importlib.util, json, hashlib
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sirena.data.weekly_bridge import load_semicolon_weekly_prices
spec = importlib.util.spec_from_file_location("weighted", ROOT / "experiments/weekly_laspeyres_nowcast/run_weekly_laspeyres_nowcast.py")
weighted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weighted)

def main():
    source = ROOT / "data/Сравнение еженедельных цен_01.csv"
    w = load_semicolon_weekly_prices(source)
    assert not w.duplicated(["date", "product_name"]).any()
    assert w.date.max() == pd.Timestamp("2026-09-21")
    window = w[w.date.between("2026-08-24", "2026-09-21")].copy()
    observed = window[window.accounting_month.astype(str).str.startswith("2026-09")]
    assert observed.date.nunique() == 4 and observed.groupby("date").size().eq(109).all()
    weights = weighted.load_weights(ROOT / "data/micro_sprav.csv")
    matches = weighted.build_item_matches(window, weights)
    # Reject a fuzzy collision: UHT is not the pasteurized milk monthly position.
    uht = matches.product_name.str.contains("ультрапастеризованное", case=False)
    matches.loc[uht, ["matched", "match_method", "item_code", "weight"]] = [False, "excluded_duplicate_milk", np.nan, np.nan]
    tv = matches.product_name.eq("Телевизор, шт.")
    tv_weight = weights[weights.Item_code.astype(str).eq("629")].iloc[0]
    matches.loc[tv, ["matched", "match_method", "item_code", "weight", "sprav_name"]] = [True, "reviewed_alias", 629, tv_weight.weight, tv_weight["Товар"]]
    sea = matches.product_name.str.contains("Черноморское", case=False)
    assert str(matches.loc[sea, "item_code"].iloc[0]) in ("1083", "1083.0")
    matches.loc[sea, "match_method"] = "reviewed_alias_geographic_scope"
    micro = pd.read_csv(ROOT / "data/micro_forecast_components.csv")
    micro = micro[micro.date.eq("2026-09-01")].copy().set_index("Item_code")
    assert micro.index.is_unique and np.isclose(micro.Weight.sum(), 1)
    factors = observed.groupby("item_key").change_index.agg(lambda s: (s / 100).prod())
    counts = observed.groupby("item_key").size()
    rows, used = [], set()
    for row in matches[matches.matched].itertuples():
        assert counts[row.item_key] == 4
        codes = [int(float(c)) for c in str(row.item_code).split("+")]
        assert not used.intersection(codes), (row.product_name, codes)
        used.update(codes)
        assert np.isclose(micro.loc[codes].Weight.sum(), row.weight)
        for code in codes:
            m = micro.loc[code]
            factor = factors[row.item_key]
            updated = 100 * (factor * (1 + m.Prediction / 100) ** .2 - 1)
            rows.append(dict(Item_code=code, product_name=row.product_name, component=row.component,
                weight=m.Weight, observed_mom=100*(factor-1), observed_contribution=m.Weight*100*(factor-1),
                prior_prediction=m.Prediction, updated_prediction=updated, contribution=m.Weight*updated,
                match_method=row.match_method))
    drivers = pd.DataFrame(rows)
    missing = micro.loc[~micro.index.isin(used)].copy()
    residual = float(missing.Contribution.sum())
    estimate = float(drivers.contribution.sum() + residual)
    flat = float(drivers.observed_contribution.sum() + residual)
    fuel_falling = drivers.product_name.str.contains("Бензин|Дизельное", case=False) & drivers.prior_prediction.lt(0)
    fuel_floor = estimate + float((drivers.loc[fuel_falling, "observed_contribution"] - drivers.loc[fuel_falling, "contribution"]).sum())
    coverage = float(drivers.weight.sum())
    cache = json.loads((ROOT / "data/precomputed_forecasts.json").read_text())
    bridge = cache["diagnostics"]["weekly_bridge"]["by_month"]["2026-09"]
    # Compare the actual archived input, not a reconstructed older export.
    journal = ROOT / "archive/results/forecast_journal"
    manifest = json.loads((journal / "20260928T085006562414Z_f9854aa128cc43778115ecc147e6c665/manifest.json").read_text())
    old_path = journal / manifest["inputs"][source.name]["object_path"]
    old = load_semicolon_weekly_prices(old_path)
    common = old.merge(w, on=["date", "product_name"], suffixes=("_old", "_new"))
    audit = dict(new_rows=len(w), old_valid_rows=len(old), common_rows=len(common),
        new_only=len(w)-len(common), old_only=len(old)-len(common),
        index_revisions=int((~np.isclose(common.change_index_old,common.change_index_new,equal_nan=True)).sum()),
        price_revisions=int((~np.isclose(common.price_old,common.price_new,equal_nan=True)).sum()),
        component_revisions=int(common.component_old.ne(common.component_new).sum()))
    summary = dict(month="2026-09", last_week="2026-09-21", weeks_observed=4, weeks_expected=5,
        central_mom=.50, scenario_low=.25, scenario_high=.75, scenario_interval_is_statistical=False,
        weighted_micro_mom=estimate, flat_remaining_week_mom=flat, fuel_no_decline_tail_mom=fuel_floor, covered_weight=coverage,
        matched_weekly_positions=int(matches.matched.sum()), matched_monthly_positions=len(used),
        observed_contribution_pp=float(drivers.observed_contribution.sum()),
        unobserved_model_contribution_pp=residual,
        standard_bridge_nowcast=cache["forecasts"]["Nowcast"][0],
        ensemble_mom=cache["forecasts"]["Ensemble"][0], prior_micro_mom=float(micro.Contribution.sum()),
        status="experimental_diagnostic_not_backtested", report="archive/results/september_nowcast_20260928/REPORT.md")
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, ROOT/"data/micro_sprav.csv",ROOT/"data/micro_forecast_components.csv",ROOT/"data/weekly_accounting_month_overrides.csv",ROOT/"data/inflation_data.csv"]}
    summary["source_hashes"] = hashes
    matches.to_csv(OUT / "item_mapping.csv", index=False)
    drivers.to_csv(OUT / "weighted_drivers.csv", index=False)
    missing.to_csv(OUT / "unobserved_micro_positions.csv")
    (OUT / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    (OUT / "source_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
    with pd.ExcelWriter(OUT / "september_nowcast.xlsx") as x:
        pd.DataFrame([summary]).drop(columns=["source_hashes"]).to_excel(x,sheet_name="Summary",index=False)
        drivers.to_excel(x,sheet_name="Weighted drivers",index=False)
        matches.to_excel(x,sheet_name="Mapping audit",index=False)
        missing.to_excel(x,sheet_name="Unobserved model")
        pd.DataFrame(bridge["chain"]["weeks"]).drop(columns=["components"]).to_excel(x,sheet_name="Weekly bridge",index=False)
        pd.DataFrame([audit]).to_excel(x,sheet_name="Source audit",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = drivers.groupby("product_name").observed_contribution.sum().sort_values()
    selected = pd.concat([top.head(5),top.tail(5)]).sort_values()
    fig, ax = plt.subplots(figsize=(11,6))
    ax.barh(selected.index, selected.values, color=["#197c70" if v<0 else "#c25a35" for v in selected])
    ax.axvline(0,color="#333333",linewidth=.7)
    ax.set_xlabel("Приближённый вклад в ИПЦ, п.п.; четыре учтённые недели")
    ax.set_title("Сентябрь: бензин и яйца против снижения цен на плодоовощи")
    fig.tight_layout();fig.savefig(OUT/"weekly_drivers.png",dpi=160);plt.close(fig)
    print(json.dumps({"summary":summary,"audit":audit},ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
