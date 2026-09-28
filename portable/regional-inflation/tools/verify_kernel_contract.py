#!/usr/bin/env python3
"""Run independent contract checks against the native regional-inflation CLI.

This is a development-only verifier. It writes a concise JSON evidence report
and deliberately does not modify the Rust kernel or claim model superiority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(REPO))
INVOKE_LOG: list[dict[str, Any]] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def working_tree_hashes() -> dict[str, str]:
    paths = [
        PROJECT / "core" / "src" / "lib.rs",
        PROJECT / "core" / "src" / "main.rs",
        PROJECT / "core" / "Cargo.toml",
        PROJECT / "core" / "Cargo.lock",
        REPO / "sirena" / "models" / "ridge.py",
        Path(__file__).resolve(),
    ]
    return {str(path.relative_to(REPO)): sha256(path) for path in paths}


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def invoke(binary: Path, request: dict[str, Any], label: str) -> dict[str, Any]:
    stdin = json.dumps(request, separators=(",", ":"))
    result = subprocess.run(
        [str(binary)],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"native CLI exited {result.returncode}: {result.stderr[-500:]}")
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"native CLI returned invalid JSON: {result.stdout[:300]!r}") from exc
    INVOKE_LOG.append({
        "label": label,
        "request_sha256": hashlib.sha256(stdin.encode("utf-8")).hexdigest(),
        "request_rows": len(request.get("rows", [])),
        "frequency": request.get("frequency"),
        "cutoff": request.get("config", {}).get("cutoff"),
        "horizon": request.get("config", {}).get("horizon"),
        "return_code": result.returncode,
        "response_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "ok": response.get("ok"),
        "error_code": response.get("error", {}).get("code") if isinstance(response.get("error"), dict) else None,
        "stderr": result.stderr[-500:] if result.stderr else None,
    })
    return response


def month_key(value: str) -> tuple[int, int]:
    return tuple(map(int, value[:7].split("-")))


def add_month(value: str, count: int) -> str:
    year, month = month_key(value)
    offset = year * 12 + month - 1 + count
    return f"{offset // 12:04d}-{offset % 12 + 1:02d}-01"


def assert_ok(response: dict[str, Any], label: str) -> None:
    if response.get("ok") is not True:
        raise AssertionError(f"{label} failed: {response.get('error')}")


def assert_error(response: dict[str, Any], label: str) -> dict[str, Any]:
    if response.get("ok") is not False or not isinstance(response.get("error"), dict):
        raise AssertionError(f"{label} should return an explicit ok=false error: {response}")
    return response["error"]


def same_forecast(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("forecast") == right.get("forecast") and left.get("fit") == right.get("fit")


def base_request(rows: list[dict[str, Any]], frequency: str, cutoff: str, horizon: int = 3) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": "ridge",
        "frequency": frequency,
        "rows": rows,
        "config": {"cutoff": cutoff, "horizon": horizon, "use_macro": True},
    }


def run(binary: Path, package_path: Path) -> dict[str, Any]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    region = next(item for item in package["regions"] if item["code"] == 7)
    raw_rows = region["rows"]
    sa_rows = region["sa_rows"]
    cutoff = "2025-06-01"
    raw_request = base_request(raw_rows, "raw", cutoff)
    raw_response = invoke(binary, raw_request, "raw_cutoff_forecast")
    assert_ok(raw_response, "RAW ridge forecast")

    # Future facts and future macro observations must be outside the fit window.
    mutated = json.loads(json.dumps(raw_request))
    for row in mutated["rows"]:
        if row["date"] > cutoff:
            row["y"] += 17.0
            row["food"] -= 21.0
            row["nonfood"] += 13.0
            row["services"] -= 9.0
            row["ki"] = 1_000.0
            row["ruonia"] = -500.0
    mutated_response = invoke(binary, mutated, "future_mutation_forecast")
    assert_ok(mutated_response, "future-mutation forecast")
    if not same_forecast(raw_response, mutated_response):
        raise AssertionError("future CPI/macro values changed a forecast fitted at 2025-06")

    sa_request = base_request(sa_rows, "sa", cutoff)
    sa_response = invoke(binary, sa_request, "sa_cutoff_forecast")
    assert_ok(sa_response, "SA ridge forecast")
    if raw_rows is sa_rows or raw_rows[-1]["y"] == sa_rows[-1]["y"]:
        raise AssertionError("RAW and SA source histories were not distinct")
    if raw_response["forecast"]["steps"] == sa_response["forecast"]["steps"]:
        raise AssertionError("RAW and SA histories unexpectedly produced identical forecast paths")

    no_macros = json.loads(json.dumps(raw_request))
    for row in no_macros["rows"]:
        row["ki"] = None
        row["ruonia"] = None
    macro_error = assert_error(invoke(binary, no_macros, "all_macro_values_none"), "all-macro-None request")
    if macro_error.get("code") == "internal_error":
        raise AssertionError("all-macro-None request reached an internal panic instead of a controlled error")

    gap = json.loads(json.dumps(raw_request))
    gap["rows"].pop(60)
    gap_error = assert_error(invoke(binary, gap, "missing_month_gap"), "monthly-gap request")
    if "gap" not in gap_error.get("message", "").lower():
        raise AssertionError(f"monthly gap returned an unrelated error: {gap_error}")

    invalid_horizon = json.loads(json.dumps(raw_request))
    invalid_horizon["config"]["horizon"] = 25
    horizon_error = assert_error(invoke(binary, invalid_horizon, "invalid_horizon_25"), "invalid-horizon request")

    # Independently recompute every planned 1/2/12-month pair and metric.
    cutoffs = ["2024-12-01", "2025-06-01", "2026-07-01"]
    horizons = [1, 2, 12]
    backtest_request = base_request(raw_rows, "raw", "2026-08-01", horizon=1)
    backtest_request["backtests"] = {"cutoffs": cutoffs, "horizons": horizons}
    backtest = invoke(binary, backtest_request, "backtest_horizons_1_2_12")
    assert_ok(backtest, "rolling backtest")
    rows_by_date = {row["date"]: row for row in raw_rows}
    result_rows = backtest["backtest"]["rows"]
    if len(result_rows) != 9:
        raise AssertionError(f"expected 9 planned backtest slots, got {len(result_rows)}")
    recomputed: dict[str, dict[str, Any]] = {}
    exact_pairs = []
    for horizon in horizons:
        matches = [row for row in result_rows if row["horizon"] == horizon]
        if len(matches) != 3:
            raise AssertionError(f"horizon {horizon} did not retain all 3 cutoff slots")
        errors = []
        valid = 0
        for cutoff_date, result in zip(cutoffs, matches):
            target_date = add_month(cutoff_date, horizon)
            if result["cutoff"] != cutoff_date or result["target_date"] != target_date:
                raise AssertionError(f"misaligned backtest slot: {result}")
            if target_date not in rows_by_date:
                if result["status"] != "unavailable":
                    raise AssertionError(f"missing actual {target_date} was treated as valid")
                exact_pairs.append({"horizon": horizon, "cutoff": cutoff_date, "target_date": target_date, "status": "unavailable"})
                continue
            actual = rows_by_date[target_date]["y"] - 100.0
            if result["status"] != "ok" or not math.isclose(result["actual"], actual, rel_tol=0, abs_tol=1e-12):
                raise AssertionError(f"actual pair mismatch at {target_date}: {result.get('actual')} != {actual}")
            errors.append(result["forecast"] - actual)
            valid += 1
            exact_pairs.append({"horizon": horizon, "cutoff": cutoff_date, "target_date": target_date, "status": "ok", "actual": actual})
        if valid != len(errors):
            raise AssertionError("backtest valid-pair count mismatch")
        observed = backtest["backtest"]["metrics"][str(horizon)]
        expected = {
            "n_planned": 3,
            "n_valid": valid,
            "mae": sum(abs(error) for error in errors) / valid if valid else None,
            "rmse": math.sqrt(sum(error * error for error in errors) / valid) if valid else None,
            "bias": sum(errors) / valid if valid else None,
            "hit_within_0_5": sum(abs(error) <= 0.5 for error in errors) / valid if valid else None,
        }
        for key, value in expected.items():
            got = observed.get(key)
            if isinstance(value, float):
                if not isinstance(got, (int, float)) or not math.isclose(got, value, rel_tol=0, abs_tol=1e-12):
                    raise AssertionError(f"h={horizon} metric {key} mismatch: {got} != {value}")
            elif got != value:
                raise AssertionError(f"h={horizon} metric {key} mismatch: {got} != {value}")
        recomputed[str(horizon)] = expected

    unavailable = [row for row in result_rows if row["status"] == "unavailable"]
    if len(unavailable) != 2 or {row["target_date"] for row in unavailable} != {"2026-09-01", "2027-07-01"}:
        raise AssertionError(f"expected 2 unavailable targets after the supplied history, got {unavailable}")

    # Regression check: a historical fit at the same cutoff matches the existing Python Ridge.
    import pandas as pd
    from sirena.models.ridge import RidgeForecaster

    frame = pd.DataFrame(raw_rows).rename(
        columns={"y": "Все товары и услуги", "food": "Продовольственные товары", "nonfood": "Непродовольственные товары", "services": "Услуги", "ki": "Ki", "ruonia": "Ruonia"}
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date")
    reference_model = RidgeForecaster().fit(frame.loc[:cutoff])
    reference_path = reference_model.forecast(3)
    native_path = [step["all"] for step in raw_response["forecast"]["steps"]]
    reference_delta = max(abs(float(a) - float(b)) for a, b in zip(native_path, reference_path))
    if reference_delta > 1e-7:
        raise AssertionError(f"historical Ridge feature/forecast regression delta {reference_delta} > 1e-7")

    return {
        "status": "PASS",
        "contract": "native CLI behavioral checks; no model superiority claim",
        "dataset": {"file": package_path.name, "sha256": sha256(package_path), "region_code": 7, "region": region["name"], "raw_rows": len(raw_rows), "sa_rows": len(sa_rows), "source_latest": package["coverage"]["release_latest"]},
        "fixtures": {
            "region_code": 7,
            "raw_rows_sha256": canonical_sha(raw_rows),
            "sa_rows_sha256": canonical_sha(sa_rows),
            "future_mutation": "for dates after 2025-06-01, add 17 to y, subtract 21 from food, add 13 to nonfood, subtract 9 from services, set Ki=1000 and Ruonia=-500",
            "backtest_cutoffs": ["2024-12-01", "2025-06-01", "2026-07-01"],
            "backtest_horizons": [1, 2, 12],
        },
        "binary": {"path": binary.name, "sha256": sha256(binary)},
        "working_tree_source_hashes": working_tree_hashes(),
        "provenance_note": "The tested executable SHA-256 identifies the exact native binary. Working-tree source hashes identify current source files; they do not prove that this binary was built from those exact hashes.",
        "native_cli_log": INVOKE_LOG,
        "checks": {
            "future_cpi_and_macro_mutation_is_cutoff_safe": {"pass": True, "cutoff": cutoff},
            "raw_and_sa_histories_are_separate": {"pass": True, "raw_latest_y": raw_rows[-1]["y"], "sa_latest_y": sa_rows[-1]["y"]},
            "all_macro_none_returns_explicit_error": {"pass": True, "error_code": macro_error.get("code"), "message": macro_error.get("message")},
            "monthly_gap_rejected": {"pass": True, "error_code": gap_error.get("code"), "message": gap_error.get("message")},
            "invalid_horizon_rejected": {"pass": True, "error_code": horizon_error.get("code"), "message": horizon_error.get("message")},
            "backtest_pairs_and_metrics_recomputed": {"pass": True, "planned_slots": len(result_rows), "valid_slots": len(result_rows) - len(unavailable), "unavailable_targets": [row["target_date"] for row in unavailable], "metrics": recomputed, "pairs": exact_pairs},
            "historical_ridge_regression": {"pass": True, "cutoff": cutoff, "horizon": 3, "max_abs_difference_vs_python": reference_delta, "tolerance": 1e-7},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=PROJECT / "core" / "target" / "release" / "regional-inflation-core")
    parser.add_argument("--package", type=Path, default=PROJECT / "data" / "regional_indices.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "verification" / "kernel_contract.json")
    args = parser.parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"native CLI not found: {args.binary}; build core first")
    try:
        report = run(args.binary, args.package)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "contract": "native CLI behavioral checks; no model superiority claim",
            "binary": {"path": args.binary.name, "sha256": sha256(args.binary)},
            "dataset": {"file": args.package.name, "sha256": sha256(args.package)},
            "working_tree_source_hashes": working_tree_hashes(),
            "native_cli_log": INVOKE_LOG,
            "provenance_note": "The tested executable SHA-256 identifies the exact native binary. Working-tree source hashes identify current source files; they do not prove that this binary was built from those exact hashes.",
            "failure": f"{type(exc).__name__}: {exc}",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"status": report["status"], "checks": list(report.get("checks", {})), "failure": report.get("failure"), "output": str(args.output)}
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
