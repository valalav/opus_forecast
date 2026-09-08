#!/usr/bin/env python3
"""Independently check the August refresh against raw weekly observations."""
import hashlib
import json
import math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
cache = json.loads((ROOT / "data/precomputed_forecasts.json").read_text())
before = json.loads((OUT / "precomputed_before.json").read_text())
weekly = pd.read_csv(ROOT / "data/Сравнение еженедельных цен_01.csv",
                     sep=";", encoding="utf-8-sig")
weekly.columns = weekly.columns.str.strip()
weekly["date"] = pd.to_datetime(weekly["Name"], format="%d.%m.%Y", errors="coerce")
weekly["index"] = pd.to_numeric(weekly["Изменение цен, в % к предыдущей неделе"]
                                .astype(str).str.replace(",", "."), errors="coerce")
weekly["component"] = weekly["Компонент"].str.strip()
weights = {"Продовольственные товары": .3948,
           "Непродовольственные товары": .3653, "Услуги": .2399}
dates = ["2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24"]
raw_indices = []
for date in dates:
    rows = weekly[weekly.date == pd.Timestamp(date)].drop_duplicates("Наименование")
    assert len(rows) == 107, (date, len(rows))
    assert rows.groupby("component").size().to_dict() == {
        "Продовольственные товары": 44, "Непродовольственные товары": 43, "Услуги": 20}
    means = rows.groupby("component")["index"].mean()
    raw_indices.append(sum(means[k] * v for k, v in weights.items()))
raw_chain = (math.prod(i / 100 for i in raw_indices) - 1) * 100
aug = cache["diagnostics"]["weekly_bridge"]["by_month"]["2026-08"]
sep = cache["diagnostics"]["weekly_bridge"]["by_month"]["2026-09"]
chain = aug["chain"]
assert [w["date"] for w in chain["weeks"]] == dates
assert chain["weeks_count"] == chain["weeks_expected"] == 4
assert chain["remaining_weeks"] == 0
assert abs(raw_chain - chain["mom"]) < 1e-6
assert chain["extrapolated_mom"] == chain["mom"]
assert [w["date"] for w in sep["chain"]["weeks"]] == ["2026-08-31"]
assert aug["first_next_week_vs_month_end"] is not None
blend = aug["nowcast_blend"]
assert blend["weekly_weight"] == .9 and blend["model_weight"] == .1
calculated = .9 * raw_chain + .1 * blend["model_proxy"]
idx = cache["forecast_dates"].index("2026-08-01")
actual = cache["forecasts"]["Nowcast"][idx]
assert abs(actual - calculated) < 1e-6
assert math.isfinite(actual)
assert cache["last_data_date"] == "2026-07-01"
policy = json.loads((ROOT / "data/send_ready_policy_trajectory.json").read_text())
assert policy["operational_anchor_month"] == "2026-08-01"
assert policy["operational_anchor_mom_pp"] == 0.
assert policy["mom_pp"][policy["forecast_dates"].index("2026-09-01")] == .5
for line in (OUT / "source_hashes.sha256").read_text().splitlines():
    expected, path = line.split("  ", 1)
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
html = (ROOT / "assets/charts/nowcast.html").read_text()
assert "Январь" not in html
for month, value in zip(cache["forecast_dates"], cache["forecasts"]["Nowcast"]):
    if value is not None:
        assert f'id="month-{month[:7]}"' in html
        assert f"{value:+.3f}% м/м" in html
assert "nowcast.html" in (ROOT / "assets/charts/index.html").read_text()
old = before["diagnostics"]["weekly_bridge"]["by_month"]["2026-08"]
old_val = before["forecasts"]["Nowcast"][idx]
calendar_effect = .9 * (chain["extrapolated_mom"] - old["chain"]["extrapolated_mom"])
model_effect = .1 * (blend["model_proxy"] - old["nowcast_blend"]["model_proxy"])
assert abs((actual - old_val) - (calendar_effect + model_effect)) < 1e-10
summary = {
    "status": "PASS",
    "target_month": "2026-08",
    "generated_at": cache["generated_at"],
    "latest_weekly_date": weekly.date.max().strftime("%Y-%m-%d"),
    "august_weekly_dates": dates,
    "raw_chain_mom": raw_chain,
    "price_level_bridge_mom": aug["month_end"]["mom"],
    "model_proxy_mom": blend["model_proxy"],
    "nowcast_mom": actual,
    "nowcast_index": 100 + actual,
    "previous_nowcast_mom": old_val,
    "change_pp": actual - old_val,
    "calendar_effect_pp": calendar_effect,
    "model_recompute_effect_pp": model_effect,
    "ensemble_mom": cache["forecasts"]["Ensemble"][idx],
    "september_aux_nowcast_mom": cache["forecasts"]["Nowcast"][idx + 1],
    "september_weekly_mom": sep["chain"]["mom"],
    "source_hashes_unchanged": True,
    "html_values_match_cache": True,
    "live_dashboard_verification": "Unavailable: no listener on localhost:8503",
}
(OUT / "verification.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
