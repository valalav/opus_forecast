from pathlib import Path

import pytest

from sirena.data.weekly_bridge import (
    classify_component,
    compute_weekly_bridge_for_months,
    compute_weekly_bridge_nowcast,
    load_semicolon_weekly_prices,
    weekly_model_blend_weights,
)


def _write_weekly_fixture(path: Path) -> None:
    rows = [
        "Name;Наименование ;Средние цены, рублей ;Изменение цен, в % к предыдущей неделе ;Компонент;№",
        "30.03.2026;Хлеб;100,0;100,0;Продовольственные товары;1",
        "30.03.2026;Телевизор;200,0;100,0;Непродовольственные товары;2",
        "30.03.2026;Стрижка;300,0;100,0;Услуги;3",
        "06.04.2026;Хлеб;101,0;101,0;Продовольственные товары;1",
        "06.04.2026;Телевизор;198,0;99,0;Непродовольственные товары;2",
        "06.04.2026;Стрижка;300,0;100,0;Услуги;3",
        "06.04.2026;Хлеб;101,0;101,0;Продовольственные товары;1",
        "13.04.2026;Хлеб;102,0;101,0;Продовольственные товары;1",
        "13.04.2026;Телевизор;196,0;99,0;Непродовольственные товары;2",
        "13.04.2026;Стрижка;303,0;101,0;Услуги;3",
        "04.05.2026;Хлеб;103,0;101,0;Продовольственные товары;1",
        "04.05.2026;Телевизор;194,0;99,0;Непродовольственные товары;2",
        "04.05.2026;Стрижка;306,0;101,0;Услуги;3",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def test_load_semicolon_weekly_prices_dedupes_and_classifies(tmp_path):
    data_path = tmp_path / "weekly.csv"
    _write_weekly_fixture(data_path)

    df = load_semicolon_weekly_prices(data_path)

    assert df.attrs["raw_rows"] == 13
    assert df.attrs["deduped_rows"] == 12
    assert df.attrs["duplicates_removed"] == 1
    assert set(df["component"]) == {"food", "nonfood", "services"}


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Непродовольственные товары", "nonfood"),
        ("Продовольственные товары", "food"),
        ("Услуги", "services"),
    ],
)
def test_classify_component(label, expected):
    assert classify_component(label) == expected


def test_compute_weekly_bridge_nowcast_chain_and_month_end(tmp_path):
    data_path = tmp_path / "weekly.csv"
    _write_weekly_fixture(data_path)
    weights = {"food": 0.4, "nonfood": 0.35, "services": 0.25}

    result = compute_weekly_bridge_nowcast("2026-04", data_path=data_path, weights=weights)

    assert result["status"] == "ok"
    assert result["chain"]["weeks_count"] == 2
    assert result["chain"]["remaining_weeks"] == 2
    assert result["chain"]["mom"] == pytest.approx(0.35015)
    assert result["chain"]["extrapolated_mom"] == pytest.approx(0.56015)
    assert result["month_end"]["matched_items"] == 3
    assert result["month_end"]["mom"] == pytest.approx(0.35)
    assert result["first_next_week_vs_month_end"]["mom"] == pytest.approx(0.282539)


def test_compute_weekly_bridge_for_months_schema(tmp_path):
    data_path = tmp_path / "weekly.csv"
    _write_weekly_fixture(data_path)

    result = compute_weekly_bridge_for_months(["2026-04", "2026-05"], data_path=data_path)

    assert result["method"] == "weekly_bridge_v1"
    assert result["duplicates_removed"] == 1
    assert result["data_date_range"] == {"start": "2026-03-30", "end": "2026-05-04"}
    assert "2026-04" in result["by_month"]
    assert "2026-05" in result["by_month"]


def test_weekly_model_blend_weights():
    assert weekly_model_blend_weights(1) == (0.60, 0.40)
    assert weekly_model_blend_weights(4) == (0.90, 0.10)
    assert weekly_model_blend_weights(6) == (0.70, 0.30)
