#!/usr/bin/env python3
"""Build the canonical KBR 45-component mapping table.

The mapping connects the external Khabarovsk/Omsk 45-component code style with
the SIRENA-KBR subcomponent structure, current regional weights, and item-level
micro positions used for scenario diagnostics.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "kbr_45_component_mapping"

SUBCOMP_SPRAV = ROOT / "data" / "raw" / "subcomp_sprav.csv"
MICRO_SPRAV = ROOT / "data" / "micro_sprav.csv"
REGION_LONG = ROOT / "data" / "external" / "micro_cpi_region_export" / "region_cpi_long.csv"


EXTERNAL_VAR_BY_CODE = {
    13: "n_galant",
    15: "n_instr",
    17: "n_avto",
    20: "n_mebel",
    21: "n_med",
    23: "n_meh",
    25: "n_mou_sr",
    27: "n_obuv",
    29: "n_odegd",
    30: "n_parf",
    31: "n_comp",
    32: "n_pech_iz",
    37: "n_sr_sv",
    38: "n_str",
    40: "n_tabac",
    41: "n_tv",
    42: "n_topl",
    43: "n_tric",
    51: "n_electro",
    54: "n_proch",
    11: "p_alco",
    16: "p_sweet",
    18: "p_mak_kr",
    19: "p_butter",
    24: "p_milk",
    26: "p_meat",
    28: "p_ob_pit",
    33: "p_ovosh",
    34: "p_fish",
    36: "p_sugar",
    39: "p_cheese",
    49: "p_bread",
    50: "p_tea",
    52: "p_egg",
    53: "p_proch",
    12: "u_bit",
    14: "u_gkh",
    22: "u_med",
    35: "u_san",
    44: "u_obr",
    46: "u_cult",
    47: "u_transp",
    48: "u_sv",
    55: "u_proch",
    67: "u_z_tour",
}


SCENARIO_TAGS = {
    11: "food;alcohol;excise_regulated",
    12: "services;household_services;labor_cost_sensitive",
    13: "nonfood;import_sensitive",
    14: "services;regulated;tariff;gkh;july_october_scenario",
    15: "nonfood;durable;import_sensitive",
    16: "food;core_food",
    17: "nonfood;durable;credit_sensitive;import_sensitive",
    18: "food;core_food",
    19: "food;core_food;volatile",
    20: "nonfood;durable;credit_sensitive",
    21: "nonfood;medicine;regulated_partial",
    22: "services;medical;labor_cost_sensitive",
    23: "nonfood;import_sensitive",
    24: "food;core_food",
    25: "nonfood;core_nonfood",
    26: "food;core_food",
    27: "nonfood;import_sensitive",
    28: "food;services_like;demand_sensitive",
    29: "nonfood;import_sensitive",
    30: "nonfood;import_sensitive",
    31: "nonfood;durable;import_sensitive",
    32: "nonfood;core_nonfood",
    33: "food;volatile;fruit_vegetables;seasonal;weather_sensitive",
    34: "food;core_food",
    35: "services;seasonal;sanatorium",
    36: "food;volatile;sugar",
    37: "nonfood;import_sensitive;electronics_related",
    38: "nonfood;construction;credit_sensitive",
    39: "food;core_food",
    40: "nonfood;tobacco;excise_regulated",
    41: "nonfood;durable;import_sensitive",
    42: "nonfood;fuel;volatile;gasoline;weekly_nowcast",
    43: "nonfood;import_sensitive",
    44: "services;regulated_partial;education;seasonal_september",
    46: "services;culture;demand_sensitive",
    47: "services;transport;regulated_partial;fuel_sensitive",
    48: "services;telecom;regulated_partial",
    49: "food;core_food",
    50: "food;core_food;import_sensitive",
    51: "nonfood;durable;credit_sensitive;import_sensitive",
    52: "food;volatile;eggs",
    53: "food;core_food",
    54: "nonfood;core_nonfood",
    55: "services;other_services;demand_sensitive",
    67: "services;tourism;volatile;fx_sensitive",
}


PARENT_BY_COMPONENT = {
    "Продовольственные товары": "Prod",
    "Непродовольственные товары": "Nonprod",
    "Услуги": "Serv",
}


SERVICE_RULES: list[tuple[int, str, list[str]]] = [
    (
        14,
        "u_gkh",
        [
            "газ",
            "водоснаб",
            "водоотвед",
            "отоплен",
            "электроэнерг",
            "коммуналь",
            "капитальн",
            "содержание",
            "ремонт жилья",
            "эксплуатации домов",
            "жилищн",
            "аренда",
            "наем жил",
            "общежит",
        ],
    ),
    (
        48,
        "u_sv",
        [
            "интернет",
            "сотов",
            "телефон",
            "радиосвяз",
            "телевизионн",
            "антенн",
            "письм",
            "посылк",
            "абонент",
            "видеосервис",
        ],
    ),
    (
        47,
        "u_transp",
        [
            "проезд",
            "такси",
            "автобус",
            "троллейбус",
            "транспорт",
        ],
    ),
    (
        22,
        "u_med",
        [
            "врач",
            "стоматолог",
            "кариес",
            "лечен",
            "анализ",
            "исследован",
            "фгдс",
            "мрт",
            "томограф",
            "стационар",
            "массаж",
            "привив",
            "осмотр",
            "ультразвук",
            "коронк",
            "протез",
            "удаление зуб",
            "сидел",
        ],
    ),
    (35, "u_san", ["санатор", "дом отдыха", "пансионат"]),
    (
        44,
        "u_obr",
        [
            "обуч",
            "занят",
            "ясли-сад",
            "дошколь",
            "курсы",
            "вождени",
            "образователь",
        ],
    ),
    (46, "u_cult", ["театр", "кино", "музе", "выстав", "культур"]),
    (
        67,
        "u_z_tour",
        [
            "зарубеж",
            "поездки в",
            "поездка в",
            "егип",
            "турц",
            "оаэ",
            "беларус",
            "закавказ",
            "азии",
        ],
    ),
    (
        12,
        "u_bit",
        [
            "стриж",
            "маникюр",
            "мойка",
            "шиномонтаж",
            "ремонт",
            "стирк",
            "химчист",
            "фотограф",
            "бане",
            "натяжн",
            "окон",
            "обой",
            "плитк",
            "набоек",
            "ателье",
            "регулировка",
            "торжеств",
            "ксерокоп",
            "гроб",
            "могил",
            "общественным туалетом",
            "аренды автомобилей",
            "элементов питания",
        ],
    ),
]


def norm(text: object) -> str:
    value = "" if pd.isna(text) else str(text).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", value).strip()


def assign_service_code(item_name: str) -> tuple[int, str, str]:
    name = norm(item_name)
    for code, external_var, markers in SERVICE_RULES:
        if any(marker in name for marker in markers):
            return code, external_var, "service_name_rule"
    return 55, "u_proch", "service_residual_rule"


def load_latest_region_weights() -> pd.DataFrame:
    if not REGION_LONG.exists():
        return pd.DataFrame(columns=["subcomponent_code", "latest_region_weight", "latest_region_date"])

    region = pd.read_csv(REGION_LONG, parse_dates=["date"])
    latest_date = region["date"].max()
    latest = region[region["date"] == latest_date].copy()
    latest["subcomponent_code"] = latest["item_code"].astype(int)
    latest = latest[latest["subcomponent_code"].isin(EXTERNAL_VAR_BY_CODE)]
    return latest[["subcomponent_code", "weight_vertical"]].rename(
        columns={"weight_vertical": "latest_region_weight"}
    ).assign(latest_region_date=latest_date.date().isoformat())


def build_item_mapping(subcomp: pd.DataFrame, micro: pd.DataFrame) -> pd.DataFrame:
    name_to_code = dict(zip(subcomp["subcomponent_name"].map(norm), subcomp["subcomponent_code"]))
    rows = []

    for row in micro.itertuples(index=False):
        component = getattr(row, "Компонент")
        subcomponent = getattr(row, "Субкомпонент")
        item_name = getattr(row, "Товар")
        item_code = int(getattr(row, "Item_code"))
        weight = float(getattr(row, "Weight"))

        if component == "Услуги":
            code, external_var, source = assign_service_code(item_name)
        else:
            code = name_to_code.get(norm(subcomponent))
            external_var = EXTERNAL_VAR_BY_CODE.get(code)
            source = "micro_sprav_subcomponent_exact" if code is not None else "unmapped"

        rows.append(
            {
                "item_code": item_code,
                "item_name": item_name,
                "item_weight": weight,
                "source_component": component,
                "source_subcomponent": "" if pd.isna(subcomponent) else subcomponent,
                "subcomponent_code": code,
                "external_var": external_var,
                "mapping_source": source,
            }
        )

    return pd.DataFrame(rows)


def pipe_join(values: pd.Series) -> str:
    return "|".join(str(v) for v in values.dropna().astype(int).tolist())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subcomp = pd.read_csv(SUBCOMP_SPRAV, sep=";", decimal=",", encoding="utf-8-sig").rename(
        columns={
            "Item_code": "subcomponent_code",
            "Товар": "subcomponent_name",
            "Компонент": "component_ru",
            "Weight": "canonical_weight",
        }
    )
    subcomp["parent_component"] = subcomp["component_ru"].map(PARENT_BY_COMPONENT)
    subcomp["external_var"] = subcomp["subcomponent_code"].map(EXTERNAL_VAR_BY_CODE)
    subcomp["scenario_tags"] = subcomp["subcomponent_code"].map(SCENARIO_TAGS)

    latest_weights = load_latest_region_weights()
    mapping = subcomp.merge(latest_weights, on="subcomponent_code", how="left")

    micro = pd.read_csv(MICRO_SPRAV, sep=";", encoding="utf-8-sig")
    item_mapping = build_item_mapping(mapping, micro)

    item_summary = (
        item_mapping.dropna(subset=["subcomponent_code"])
        .assign(subcomponent_code=lambda df: df["subcomponent_code"].astype(int))
        .groupby("subcomponent_code")
        .agg(
            micro_item_count=("item_code", "size"),
            micro_weight_sum=("item_weight", "sum"),
            micro_item_codes=("item_code", pipe_join),
        )
        .reset_index()
    )

    mapping = mapping.merge(item_summary, on="subcomponent_code", how="left")
    mapping["micro_item_count"] = mapping["micro_item_count"].fillna(0).astype(int)
    mapping["micro_weight_sum"] = mapping["micro_weight_sum"].fillna(0.0)
    mapping["micro_item_codes"] = mapping["micro_item_codes"].fillna("")
    mapping["weight_gap_latest_vs_canonical"] = (
        mapping["latest_region_weight"] - mapping["canonical_weight"]
    )
    mapping["mapping_status"] = "mapped"
    mapping.loc[mapping["external_var"].isna(), "mapping_status"] = "missing_external_var"
    mapping.loc[mapping["micro_item_count"] == 0, "mapping_status"] = (
        mapping["mapping_status"] + ";no_micro_items"
    )

    item_mapping.to_csv(OUT_DIR / "kbr_45_micro_item_mapping.csv", index=False, encoding="utf-8")
    mapping.to_csv(OUT_DIR / "kbr_45_component_mapping.csv", index=False, encoding="utf-8")

    summary = (
        mapping.groupby("parent_component")
        .agg(
            components=("subcomponent_code", "size"),
            canonical_weight=("canonical_weight", "sum"),
            latest_region_weight=("latest_region_weight", "sum"),
            micro_weight_sum=("micro_weight_sum", "sum"),
            micro_item_count=("micro_item_count", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "kbr_45_component_mapping_summary.csv", index=False, encoding="utf-8")

    unmapped_items = item_mapping[item_mapping["subcomponent_code"].isna()].copy()
    unmapped_items.to_csv(OUT_DIR / "kbr_45_unmapped_micro_items.csv", index=False, encoding="utf-8")

    top_components = mapping.sort_values("canonical_weight", ascending=False).head(12)
    no_micro = mapping[mapping["micro_item_count"] == 0]
    latest_date = mapping["latest_region_date"].dropna().iloc[0] if mapping["latest_region_date"].notna().any() else "n/a"

    report = f"""# KBR 45-Component Mapping

Date: 2026-06-25

Purpose: canonical bridge between the external Khabarovsk/Omsk 45-component
model style and SIRENA-KBR data.

## Inputs

- SIRENA 45-component weights: `data/raw/subcomp_sprav.csv`
- KBR item-level weights: `data/micro_sprav.csv`
- Current official regional long export: `data/external/micro_cpi_region_export/region_cpi_long.csv`

## Outputs

- Component mapping: `experiments/kbr_45_component_mapping/kbr_45_component_mapping.csv`
- Micro item mapping: `experiments/kbr_45_component_mapping/kbr_45_micro_item_mapping.csv`
- Parent summary: `experiments/kbr_45_component_mapping/kbr_45_component_mapping_summary.csv`
- Unmapped item diagnostics: `experiments/kbr_45_component_mapping/kbr_45_unmapped_micro_items.csv`

## Coverage

- 45 components mapped to external variable names: `{mapping["external_var"].notna().sum()}` / `{len(mapping)}`
- Canonical 45-component weight sum: `{mapping["canonical_weight"].sum():.5f}`
- Latest regional 45-component weight sum ({latest_date}): `{mapping["latest_region_weight"].sum():.5f}`
- Micro-sprav item weight assigned to the 45 layer: `{item_mapping["item_weight"].sum() - unmapped_items["item_weight"].sum():.5f}`
- Unmapped micro-sprav item weight: `{unmapped_items["item_weight"].sum():.5f}`

## Parent Components

{summary.to_markdown(index=False, floatfmt=".5f")}

## Largest Components

{top_components[["subcomponent_code", "external_var", "subcomponent_name", "parent_component", "canonical_weight", "latest_region_weight", "micro_weight_sum", "scenario_tags"]].to_markdown(index=False, floatfmt=".5f")}

## Diagnostics

- Components without micro items: `{len(no_micro)}`
- Service micro allocation is inferred by item-name rules because
  `data/micro_sprav.csv` does not contain service subcomponent labels.
- The mapping is suitable for scenario diagnostics and a KBR45 prototype, but
  it is not yet a promoted forecasting model.

## Immediate Use

1. Use `scenario_tags` to apply transparent expert overrides to fuel, ЖКУ,
   плодоовощи, eggs, sugar, tourism, education and transport.
2. Use `latest_region_weight` for current headline contribution calculations
   when the official long export is available.
3. Use `canonical_weight` to stay compatible with existing
   `SubcomponentForecaster` and historical `data/raw/subcomp.csv` structure.
4. Build the first `KBR45_ARIMA` or scenario-aggregation prototype on top of
   this mapping, not on a new private spreadsheet.
"""
    (OUT_DIR / "kbr_45_component_mapping_report.md").write_text(report, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
