#!/usr/bin/env python3
"""Diagnose known regional Huber nonconvergence without changing model defaults."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def invoke(binary: Path, request: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [str(binary)],
        input=json.dumps(request, ensure_ascii=False, separators=(",", ":")),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"native CLI exited {result.returncode}: {result.stderr[-400:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"native CLI returned invalid JSON: {result.stdout[:250]!r}") from exc


def diagnostic_run(binary: Path, region: dict[str, Any], frequency: str, max_iter: int, tol: float) -> dict[str, Any]:
    rows = region["rows" if frequency == "raw" else "sa_rows"]
    request = {
        "schema_version": 1,
        "model": "huber",
        "frequency": frequency,
        "rows": rows,
        "config": {
            "horizon": 12,
            "use_macro": True,
            "max_iter": max_iter,
            "tol": tol,
            # This flag only exposes fit diagnostics; nonconverged paths remain diagnostic.
            "allow_nonconverged": True,
        },
    }
    response = invoke(binary, request)
    if response.get("ok") is not True:
        return {
            "region_code": region["code"],
            "region": region["name"],
            "frequency": frequency,
            "max_iter": max_iter,
            "tol": tol,
            "status": "controlled_error",
            "error": response.get("error"),
        }
    fit = response.get("fit", {})
    converged = fit.get("converged") is True
    steps = response.get("forecast", {}).get("steps", [])
    values = [step.get("all") for step in steps if isinstance(step, dict)]
    return {
        "region_code": region["code"],
        "region": region["name"],
        "frequency": frequency,
        "rows": len(rows),
        "origin": rows[-1]["date"],
        "max_iter": max_iter,
        "tol": tol,
        "status": "converged" if converged else "nonconverged_diagnostic_only",
        "converged": converged,
        "iterations": fit.get("iterations"),
        "objective": fit.get("objective"),
        "gradient_inf_norm": fit.get("gradient_inf_norm"),
        "scale": fit.get("scale"),
        "n_train": fit.get("n_train"),
        "forecast_steps": len(steps),
        "forecast_finite": len(values) == 12 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values),
        "forecast_first_last_diagnostic_only": [values[0], values[-1]] if len(values) == 12 else None,
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    converged = [row for row in runs if row.get("converged") is True]
    failed = [row for row in runs if row.get("status") != "converged"]
    return {
        "runs": len(runs),
        "converged": len(converged),
        "not_converged_or_error": len(failed),
        "iterations": {
            "min": min((r["iterations"] for r in runs if isinstance(r.get("iterations"), int)), default=None),
            "max": max((r["iterations"] for r in runs if isinstance(r.get("iterations"), int)), default=None),
        },
        "max_gradient_inf_norm": max((r["gradient_inf_norm"] for r in runs if isinstance(r.get("gradient_inf_norm"), (int, float))), default=None),
        "max_objective": max((r["objective"] for r in runs if isinstance(r.get("objective"), (int, float))), default=None),
    }


def run(binary: Path, package_path: Path, baseline_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    package = json.loads(package_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures = baseline.get("controlled_failures", [])
    if not failures:
        raise ValueError("No controlled Huber failures found in all_regions.json")
    regions = {region["code"]: region for region in package["regions"]}
    cases = sorted({(item["region_code"], item["frequency"]) for item in failures})
    if len(cases) != 27:
        raise ValueError(f"Expected 27 known Huber failure cases, found {len(cases)}")

    baseline_diagnostics = []
    iter_2000 = []
    iter_10000 = []
    for code, frequency in cases:
        region = regions[code]
        baseline_diagnostics.append(diagnostic_run(binary, region, frequency, 500, 1e-5))
        iter_2000.append(diagnostic_run(binary, region, frequency, 2_000, 1e-5))
        iter_10000.append(diagnostic_run(binary, region, frequency, 10_000, 1e-5))

    still_nonconverged = [
        case for case, result in zip(cases, iter_10000)
        if result.get("converged") is not True
    ]
    tolerance_500: list[dict[str, Any]] = []
    tolerance_2000: list[dict[str, Any]] = []
    if still_nonconverged:
        for code, frequency in still_nonconverged:
            region = regions[code]
            tolerance_500.append(diagnostic_run(binary, region, frequency, 500, 1e-4))
            tolerance_2000.append(diagnostic_run(binary, region, frequency, 2_000, 1e-4))

    iteration_comparison = []
    for baseline_fit, at_2000, at_10000 in zip(baseline_diagnostics, iter_2000, iter_10000):
        objectives = [baseline_fit.get("objective"), at_2000.get("objective"), at_10000.get("objective")]
        gradients = [baseline_fit.get("gradient_inf_norm"), at_2000.get("gradient_inf_norm"), at_10000.get("gradient_inf_norm")]
        finite_objectives = [x for x in objectives if isinstance(x, (int, float))]
        finite_gradients = [x for x in gradients if isinstance(x, (int, float))]
        iteration_comparison.append({
            "region_code": baseline_fit["region_code"],
            "frequency": baseline_fit["frequency"],
            "objectives_500_2000_10000": objectives,
            "gradient_inf_norms_500_2000_10000": gradients,
            "objective_max_abs_delta": max(finite_objectives) - min(finite_objectives) if finite_objectives else None,
            "gradient_max_abs_delta": max(finite_gradients) - min(finite_gradients) if finite_gradients else None,
            "fit_metrics_unchanged_with_more_iterations": len(finite_objectives) == 3 and len(finite_gradients) == 3 and len(set(finite_objectives)) == 1 and len(set(finite_gradients)) == 1,
        })
    unchanged_fits = sum(item["fit_metrics_unchanged_with_more_iterations"] for item in iteration_comparison)

    source_files = [
        PROJECT / "core" / "src" / "lib.rs",
        PROJECT / "core" / "src" / "main.rs",
        PROJECT / "core" / "Cargo.toml",
        PROJECT / "core" / "Cargo.lock",
    ]
    return {
        "status": "COMPLETE" if not any(r.get("status") == "controlled_error" for rows in (baseline_diagnostics, iter_2000, iter_10000, tolerance_500, tolerance_2000) for r in rows) else "PARTIAL",
        "purpose": "Huber convergence sensitivity only; does not change defaults or promote nonconverged forecasts",
        "method": {
            "known_cases": len(cases),
            "frequency_and_rows": "each case uses the same RAW or SA history from regional_indices.json, macro enabled, horizon 12, and default model exclusions",
            "allow_nonconverged": "true only to expose fit objective, gradient norm and scale; any nonconverged forecast path is diagnostic only",
            "baseline_default": {"max_iter": 500, "tol": 1e-5},
            "requested_sensitivities": [
                {"max_iter": 2_000, "tol": 1e-5},
                {"max_iter": 10_000, "tol": 1e-5},
                {"max_iter": 500, "tol": 1e-4, "run_when": "only cases still nonconverged at max_iter=10000,tol=1e-5"},
                {"max_iter": 2_000, "tol": 1e-4, "run_when": "only cases still nonconverged at max_iter=10000,tol=1e-5"},
            ],
        },
        "source": {
            "dataset": {"file": package_path.name, "sha256": sha256(package_path), "regional_source": package.get("source"), "latest": package.get("coverage", {}).get("release_latest")},
            "baseline_report": {"file": baseline_path.name, "sha256": sha256(baseline_path)},
            "binary": {"file": binary.name, "sha256": sha256(binary)},
            "working_tree_hashes": {str(path.relative_to(REPO)): sha256(path) for path in source_files},
            "provenance_note": "Binary hash identifies the exact tested executable. Source hashes identify current files and do not prove the binary was built from those exact hashes.",
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "summaries": {
            "baseline_500_tol_1e-5": summarize(baseline_diagnostics),
            "max_iter_2000_tol_1e-5": summarize(iter_2000),
            "max_iter_10000_tol_1e-5": summarize(iter_10000),
            "max_iter_500_tol_1e-4": summarize(tolerance_500),
            "max_iter_2000_tol_1e-4": summarize(tolerance_2000),
        },
        "iteration_sensitivity": {
            "cases_with_identical_objective_and_gradient_at_500_2000_10000": unchanged_fits,
            "cases": iteration_comparison,
            "interpretation": "For cases with identical fit metrics as the iteration cap rises, raising max_iter alone did not move the optimizer to a different fit. A relaxed tolerance may accept a fit sooner, but only the converged flag indicates whether it met that configured threshold.",
        },
        "remaining_at_10000_tol_1e-5": [{"region_code": code, "frequency": frequency} for code, frequency in still_nonconverged],
        "baseline_failure_messages": [
            {"region_code": row["region_code"], "frequency": row["frequency"], "message": row.get("error", {}).get("message")}
            for row in failures
        ],
        "runs": {
            "baseline_500_tol_1e-5": baseline_diagnostics,
            "max_iter_2000_tol_1e-5": iter_2000,
            "max_iter_10000_tol_1e-5": iter_10000,
            "max_iter_500_tol_1e-4": tolerance_500,
            "max_iter_2000_tol_1e-4": tolerance_2000,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=PROJECT / "core" / "target" / "release" / "regional-inflation-core")
    parser.add_argument("--package", type=Path, default=PROJECT / "data" / "regional_indices.json")
    parser.add_argument("--baseline", type=Path, default=PROJECT / "verification" / "all_regions.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "verification" / "huber_convergence_sensitivity.json")
    args = parser.parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"native CLI not found: {args.binary}")
    report = run(args.binary, args.package, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"status": report["status"], "elapsed_seconds": report["elapsed_seconds"], "summaries": report["summaries"], "remaining_at_10000": len(report["remaining_at_10000_tol_1e-5"]), "output": str(args.output)}
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
