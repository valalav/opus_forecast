import csv
import importlib.util
from datetime import datetime
from pathlib import Path

import openpyxl
import pytest


SCRIPT = Path(__file__).parents[1] / "tools" / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
prepare_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_data)


def make_source(path: Path, *, duplicate: bool = False) -> None:
    workbook = openpyxl.Workbook()
    db = workbook.active
    db.title = "db"
    db.append(["Day", "item", "region", "mom", "mom_sa"])
    for item, mom, sa in [(1, 100.7, 100.5), (2, 101.0, 100.9), (3, 100.2, 100.1), (4, 100.4, 100.3)]:
        db.append([datetime(2026, 8, 1), item, 7, mom, sa])
    if duplicate:
        db.append([datetime(2026, 8, 1), 1, 7, 100.7, 100.5])
    names = workbook.create_sheet("momsa_regions")
    names.append(["", "", ""])
    names.append(["", "", ""])
    names.append(["", "", ""])
    names.append(["", "Код", "Регион"])
    names.append([None, 7, "Кабардино-Балкарская Республика"])
    workbook.save(path)


def make_macros(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["Date", "Ki", "Ruonia"])
        writer.writerow(["31.08.2026", "14,00", "13,68"])


def test_build_keeps_raw_and_sa_separate_and_joins_month_level_macros(tmp_path):
    source = tmp_path / "regional.xlsx"
    macros = tmp_path / "inflation_data.csv"
    make_source(source)
    make_macros(macros)

    package = prepare_data.build(source, macros)

    assert package["schema_version"] == 1
    assert package["coverage"]["release_latest"] == "2026-08-01"
    assert len(package["regions"]) == 1
    region = package["regions"][0]
    assert (region["code"], region["kind"]) == (7, "subject")
    assert region["rows"] == [
        {"date": "2026-08-01", "y": 100.7, "nonfood": 101.0, "food": 100.2, "services": 100.4, "ki": 14.0, "ruonia": 13.68}
    ]
    assert region["sa_rows"][0]["y"] == 100.5
    assert package["macro_rows"] == [{"date": "2026-08-01", "ki": 14.0, "ruonia": 13.68}]


def test_build_rejects_duplicate_observation(tmp_path):
    source = tmp_path / "regional.xlsx"
    macros = tmp_path / "inflation_data.csv"
    make_source(source, duplicate=True)
    make_macros(macros)

    with pytest.raises(ValueError, match="Duplicate raw observation"):
        prepare_data.build(source, macros)
