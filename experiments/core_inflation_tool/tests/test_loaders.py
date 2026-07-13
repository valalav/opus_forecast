from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from core_inflation.loaders import (  # noqa: E402
    LoaderError,
    load_component_basket,
    load_headline_cpi,
    load_long_item_indices,
    load_weights_table,
    load_wide_component_indices,
    read_csv_readonly,
)


def test_read_csv_readonly_parses_semicolon_and_comma_decimal(tmp_path: Path) -> None:
    source = tmp_path / "semicolon.csv"
    source.write_text("Date;value;name\n31.01.2026;101,25;A\n", encoding="utf-8")

    frame = read_csv_readonly(source)

    assert list(frame.columns) == ["Date", "value", "name"]
    assert frame.loc[0, "value"] == pytest.approx(101.25)
    assert frame.loc[0, "name"] == "A"


def test_read_csv_readonly_falls_back_when_sniffer_cannot_detect_separator(tmp_path: Path) -> None:
    source = tmp_path / "wide.csv"
    source.write_text("Код;Товар;2026-01;2026-02\n1;А;100,1;100,2\n", encoding="utf-8")

    frame = read_csv_readonly(source)

    assert list(frame.columns) == ["Код", "Товар", "2026-01", "2026-02"]
    assert frame.loc[0, "2026-02"] == pytest.approx(100.2)


def test_load_headline_cpi_normalizes_date_and_headline_name(tmp_path: Path) -> None:
    source = tmp_path / "inflation_data.csv"
    source.write_text(
        "Date;mom;Prod\n31.01.2026;100,83;101,36\n",
        encoding="utf-8",
    )

    frame = load_headline_cpi(source)

    assert frame.loc[0, "date"] == pd.Timestamp("2026-01-31")
    assert frame.loc[0, "headline_index"] == pytest.approx(100.83)
    assert frame.loc[0, "Prod"] == pytest.approx(101.36)


def test_load_wide_component_indices_melts_month_columns(tmp_path: Path) -> None:
    source = tmp_path / "mom_sa_kbr.csv"
    source.write_text(
        "Код;Товар;2026-01;2026-02\n"
        "1;Все товары и услуги;100,83;100,39\n"
        "3;Продовольственные товары;101,36;100,68\n",
        encoding="utf-8",
    )

    frame = load_wide_component_indices(source)

    assert list(frame.columns) == ["date", "item_code", "item_name", "index_value"]
    assert len(frame) == 4
    assert frame.loc[0, "date"] == pd.Timestamp("2026-01-01")
    assert frame.loc[0, "item_code"] == 1
    assert frame.loc[0, "index_value"] == pytest.approx(100.83)
    assert frame.loc[3, "item_name"] == "Продовольственные товары"


def test_load_wide_component_indices_strips_utf8_bom_from_code_column(tmp_path: Path) -> None:
    source = tmp_path / "mom_sa_kbr.csv"
    source.write_text(
        "\ufeffКод;Товар;2026-01\n"
        "1;Все товары и услуги;100,83\n",
        encoding="utf-8",
    )

    frame = load_wide_component_indices(source)

    assert frame.loc[0, "item_code"] == 1
    assert frame.loc[0, "index_value"] == pytest.approx(100.83)


def test_load_long_item_indices_keeps_mom_and_yoy_separate(tmp_path: Path) -> None:
    source = tmp_path / "kbr_indices.csv"
    source.write_text(
        "Code,Day,Region_code,Item_code,MoM,YoY,Calc,Date\n"
        "1,01/01/26 00:00:00,7,3,101.36,109.12,0,2026-01-01\n",
        encoding="utf-8",
    )

    frame = load_long_item_indices(source)

    assert list(frame.columns) == [
        "date",
        "region_code",
        "item_code",
        "mom_index",
        "yoy_index",
    ]
    assert frame.loc[0, "mom_index"] == pytest.approx(101.36)
    assert frame.loc[0, "yoy_index"] == pytest.approx(109.12)


def test_load_weights_table_parses_comma_decimal_weight(tmp_path: Path) -> None:
    source = tmp_path / "micro_sprav.csv"
    source.write_text(
        "Item_code;Товар;Компонент;Weight\n"
        "195;Говядина;Продовольственные товары;0,01634\n",
        encoding="utf-8",
    )

    frame = load_weights_table(source)

    assert frame.loc[0, "item_code"] == 195
    assert frame.loc[0, "weight"] == pytest.approx(0.01634)



def test_load_component_basket_preserves_leaf_groups(tmp_path: Path) -> None:
    source = tmp_path / "micro_sprav.csv"
    source.write_text(
        "Item_code;Товар;Компонент;Субкомпонент;Weight\n"
        "131;Бензин АИ-95;Непродовольственные товары;Топливо моторное;0,01543\n",
        encoding="utf-8",
    )

    frame = load_component_basket(source)

    assert frame.loc[0, "item_code"] == 131
    assert frame.loc[0, "item_name"] == "Бензин АИ-95"
    assert frame.loc[0, "subcomponent_group"] == "Топливо моторное"


def test_wide_loader_requires_month_columns(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    source.write_text("Код;Товар;not_month\n1;A;100,0\n", encoding="utf-8")

    with pytest.raises(LoaderError, match="no YYYY-MM month columns"):
        load_wide_component_indices(source)
