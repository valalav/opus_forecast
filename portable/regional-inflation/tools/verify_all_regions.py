#!/usr/bin/env python3
"""Run a native 12-month forecast smoke test for every bundled region series."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def smoke(binary: Path, package: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    successes: list[dict[str, Any]] = []
    controlled_failures: list[dict[str, Any]] = []
    unexpected_failures: list[dict[str, Any]] = []
    total = 0

    for region in package["regions"]:
        for frequency, row_key in (("raw", "rows"), ("sa", "sa_rows")):
            rows = region[row_key]
            for model in ("ridge", "huber"):
                total += 1
                request = {
                    "schema_version": 1,
                    "model": model,
                    "frequency": frequency,
                    "rows": rows,
                    "config": {"horizon": 12, "use_macro": True},
                }
                encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
                case = {
                    "region_code": region["code"],
                    "region": region["name"],
                    "kind": region["kind"],
                    "frequency": frequency,
                    "model": model,
                    "n_rows": len(rows),
                    "origin": rows[-1]["date"] if rows else None,
                }
                try:
                    result = subprocess.run(
                        [str(binary)],
                        input=encoded,
                        text=True,
                        capture_output=True,
                        timeout=120,
                        check=False,
                    )
                    case["return_code"] = result.returncode
                    if result.returncode != 0:
                        case["stderr"] = result.stderr[-500:]
                        unexpected_failures.append({**case, "failure": "native process returned nonzero"})
                        continue
                    response = json.loads(result.stdout)
                except subprocess.TimeoutExpired:
                    unexpected_failures.append({**case, "failure": "native process timed out after 120s"})
                    continue
                except (json.JSONDecodeError, OSError) as exc:
                    unexpected_failures.append({**case, "failure": f"{type(exc).__name__}: {exc}"})
                    continue

                if response.get("ok") is not True:
                    error = response.get("error")
                    if isinstance(error, dict):
                        controlled_failures.append({**case, "error": error})
                    else:
                        unexpected_failures.append({**case, "failure": "ok=false without a structured error", "response": response})
                    continue

                steps = response.get("forecast", {}).get("steps", [])
                values = [step.get("all") for step in steps if isinstance(step, dict)]
                if len(steps) != 12 or len(values) != 12 or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                    unexpected_failures.append({**case, "failure": "successful response did not contain 12 finite forecast values", "step_count": len(steps)})
                    continue
                fit = response.get("fit", {})
                successes.append({
                    **case,
                    "status": "ok",
                    "forecast_steps": len(steps),
                    "n_train": fit.get("n_train"),
                    "converged": fit.get("converged"),
                    "iterations": fit.get("iterations"),
                    "objective": fit.get("objective"),
                    "first_forecast": values[0],
                    "last_forecast": values[-1],
                })

    elapsed = time.monotonic() - started
    status = "PASS" if not controlled_failures and not unexpected_failures else "PARTIAL" if not unexpected_failures else "FAIL"
    model_frequency_counts = {}
    for model in ("ridge", "huber"):
        for frequency in ("raw", "sa"):
            key = f"{model}/{frequency}"
            model_frequency_counts[key] = {
                "planned": len(package["regions"]),
                "successes": sum(x["model"] == model and x["frequency"] == frequency for x in successes),
                "controlled_failures": sum(x["model"] == model and x["frequency"] == frequency for x in controlled_failures),
                "unexpected_failures": sum(x["model"] == model and x["frequency"] == frequency for x in unexpected_failures),
            }
    failure_groups: dict[str, int] = {}
    for failure in controlled_failures:
        error = failure["error"]
        message = error.get("message", "")
        group_message = "Huber optimizer nonconverged after max iterations" if "Huber optimizer status=nonconverged" in message else message
        key = f"{failure['model']}/{failure['frequency']}:{error.get('code')}:{group_message}"
        failure_groups[key] = failure_groups.get(key, 0) + 1
    core_paths = [
        PROJECT / "core" / "src" / "lib.rs",
        PROJECT / "core" / "src" / "main.rs",
        PROJECT / "core" / "Cargo.toml",
        PROJECT / "core" / "Cargo.lock",
    ]
    return {
        "status": status,
        "contract": "12-step native CLI smoke coverage; no model superiority claim",
        "coverage": {"regions": len(package["regions"]), "frequencies": ["raw", "sa"], "models": ["ridge", "huber"], "horizon": 12, "planned_runs": total},
        "dataset": {"file": "regional_indices.json", "sha256": sha256(PROJECT / "data" / "regional_indices.json"), "source": package.get("source"), "source_latest": package.get("coverage", {}).get("release_latest")},
        "binary": {"file": binary.name, "sha256": sha256(binary)},
        "working_tree_source_hashes": {str(path.relative_to(REPO)): sha256(path) for path in core_paths},
        "provenance_note": "The executable hash identifies the exact tested binary; source hashes identify current working-tree files and do not prove which source revision produced the binary.",
        "elapsed_seconds": round(elapsed, 3),
        "counts": {"successes": len(successes), "controlled_failures": len(controlled_failures), "unexpected_failures": len(unexpected_failures)},
        "model_frequency_counts": model_frequency_counts,
        "controlled_failure_groups": failure_groups,
        "successes": successes,
        "controlled_failures": controlled_failures,
        "unexpected_failures": unexpected_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=PROJECT / "core" / "target" / "release" / "regional-inflation-core")
    parser.add_argument("--package", type=Path, default=PROJECT / "data" / "regional_indices.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "verification" / "all_regions.json")
    args = parser.parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"native CLI not found: {args.binary}; build core first")
    package = json.loads(args.package.read_text(encoding="utf-8"))
    report = smoke(args.binary, package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"status": report["status"], "counts": report["counts"], "elapsed_seconds": report["elapsed_seconds"], "output": str(args.output)}
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
