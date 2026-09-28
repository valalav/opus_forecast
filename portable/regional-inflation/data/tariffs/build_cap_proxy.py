"""Build a monthly change-of-published-cap proxy from verified KBR schedules.

This is not actual CPI or observed tariff inflation. It transforms legal
maximum household-payment indices into a separate, explicitly named plan
feature and leaves unsupported years absent.
"""

import csv
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "kbr_nalchik_payment_caps.csv"
OUTPUT = ROOT / "kbr_nalchik_monthly_payment_cap_proxy.json"
SERIES_ID = "monthly_change_of_nalchik_payment_cap_proxy_pct"


def month_starts(start: date, end: date):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield date(year, month, 1)
        year, month = (year + (month == 12), 1 if month == 12 else month + 1)


def load_periods():
    with INPUT.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    periods = []
    for row in rows:
        periods.append(
            {
                "start": date.fromisoformat(row["period_start"]),
                "end": date.fromisoformat(row["period_end"]),
                "cap": float(row["cap_pct"]),
                "known_at": row["known_at"],
                "source": row["publication_source"],
            }
        )
    return rows[0]["region_code"], periods


def build():
    region_code, periods = load_periods()
    records = []
    by_year = {}
    for period in periods:
        by_year.setdefault(period["start"].year, []).append(period)

    for year, year_periods in sorted(by_year.items()):
        year_periods = sorted(year_periods, key=lambda item: item["start"])
        if year_periods[0]["start"] != date(year, 1, 1) or year_periods[-1]["end"] != date(year, 12, 31):
            raise ValueError(f"cap periods do not cover the full published calendar year {year}")
        for prior, current in zip(year_periods, year_periods[1:]):
            if (current["start"] - prior["end"]).days != 1:
                raise ValueError(f"gap or overlap in published cap intervals for {year}")
        previous_cap = None
        for period in year_periods:
            for month in month_starts(period["start"], period["end"]):
                if month.year != year:
                    raise ValueError(f"cap interval crosses calendar year: {period}")
                if previous_cap is None:
                    previous_cap = 0.0  # prescribed annual baseline: prior December
                rate = ((1 + period["cap"] / 100) / (1 + previous_cap / 100) - 1) * 100
                records.append(
                    {
                        "date": month.isoformat(),
                        "rate": round(rate, 6),
                        "known_at": period["known_at"],
                        "source": period["source"],
                        "kind": "plan",
                    }
                )
                previous_cap = period["cap"]

    payload = {
        "schema_version": 1,
        "region_code": region_code,
        "series_id": SERIES_ID,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
