from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.freeze_analysis import (
    AGGREGATE_BUCKET_CODES,
    EXCLUSION_RULES,
    apply_exclusions,
    build_level5_scope,
    is_education_item,
)


def test_documented_exclusions_match_repo_note():
    assert set(EXCLUSION_RULES) == {
        162,
        400,
        416,
        444,
        445,
        446,
        449,
        545,
        589,
        590,
        664,
        676,
        679,
        681,
        969,
    }

    assert set(AGGREGATE_BUCKET_CODES) == {
        1,
        2,
        4,
        6,
        7,
        8,
        9,
        14,
        53,
        54,
        55,
        436,
        510,
    }


def test_level5_scope_uses_micro_codes_and_keeps_education_separate():
    monthly = pd.DataFrame(
        {
            "Region_code": [7, 7, 7, 7],
            "Item_code": [1, 162, 211, 700],
            "Date": pd.to_datetime(["2026-01-01"] * 4),
            "Year": [2026, 2026, 2026, 2026],
            "MoM": [101.0, 100.0, 100.0, 99.5],
        }
    )
    weights = pd.DataFrame(
        {
            "Year": [2026, 2026, 2026, 2026],
            "Item_code": [1, 162, 211, 700],
            "Weight_vertical": [1.0, 0.02, 0.01, 0.03],
        }
    )
    item_names = pd.DataFrame(
        {
            "Item_code": [1, 162, 211, 700],
            "Item_name": [
                "Все товары и услуги",
                "Взносы на капитальный ремонт, м2",
                "Дополнительные занятия в государственных и муниципальных общеобразовательных организациях очной формы обучения, академический час",
                "Рыба условная, кг",
            ],
        }
    )
    micro_sprav = pd.DataFrame(
        {
            "Item_code": [162, 211, 700],
            "Micro_name": [
                "Взносы на капитальный ремонт, м2",
                "Дополнительные занятия ...",
                "Рыба условная, кг",
            ],
            "Component": ["Услуги", "Услуги", "Продовольственные товары"],
            "Subcomponent": ["", "", "Рыбопродукты"],
        }
    )

    level5 = build_level5_scope(monthly, weights, item_names, micro_sprav)

    assert set(level5["Item_code"]) == {162, 211, 700}
    assert 1 not in set(level5["Item_code"])

    included, excluded = apply_exclusions(level5)
    included = included.copy()
    included["Is_education"] = is_education_item(included)

    assert set(excluded["Item_code"]) == {162}
    assert set(included["Item_code"]) == {211, 700}
    assert included.loc[included["Item_code"] == 211, "Is_education"].iloc[0]
    assert not included.loc[included["Item_code"] == 700, "Is_education"].iloc[0]


def test_education_tag_is_limited_to_education_services():
    sample = pd.DataFrame(
        {
            "Item_name": [
                "Услуги высшего образования",
                "Тетрадь школьная, шт.",
                "Экскурсия автобусная, час",
                "Начальный курс обучения вождению легкового автомобиля, курс",
            ],
            "Micro_name": ["", "", "", ""],
            "Component": ["Услуги", "Непродовольственные товары", "Услуги", "Услуги"],
            "Subcomponent": ["", "", "", ""],
        }
    )

    flags = is_education_item(sample)

    assert flags.tolist() == [True, False, False, True]
