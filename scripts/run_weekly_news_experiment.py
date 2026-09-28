#!/usr/bin/env python3
"""Run an isolated staged weekly-news adjustment experiment (evidence level C)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sirena.data.weekly_bridge import load_semicolon_weekly_prices
from sirena.data_loader import forecast_source_manifest, load_model_data
from sirena.evaluation import code_manifest
from sirena.models.ridge import RidgeForecaster
from sirena.models.weekly_news_bridge import (
    ALPHA,
    MIN_TRAIN_MONTHS,
    STAGES,
    PooledWeeklyNewsBridge,
    WeeklyBridgeDataError,
    build_stage_surprises,
    calendar_stage_features,
    observed_prefix,
)


TARGET_COLUMN = "Все товары и услуги"
EVALUATION_END = pd.Timestamp("2026-08-01")
AUXILIARY_MONTH = pd.Timestamp("2026-09-01")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ridge_forecast(monthly: pd.DataFrame, target: pd.Timestamp, cache: dict) -> float:
    """Re-fit the project Ridge at target-1 for each origin; memoized by origin."""
    if target in cache:
        return cache[target]
    cutoff = target - pd.DateOffset(months=1)
    train = monthly.loc[:cutoff].copy()
    if train.empty or train.index.max() != cutoff:
        raise ValueError(f"Ridge monthly origin missing at {cutoff:%Y-%m}")
    target_row = pd.DataFrame(index=pd.DatetimeIndex([target], name=monthly.index.name), columns=monthly.columns, dtype=float)
    prediction_frame = pd.concat([train, target_row]).sort_index()
    model = RidgeForecaster(use_macro=False)
    model.fit(train, TARGET_COLUMN)
    prediction = float(model.predict(prediction_frame, target)["prediction"] - 100.0)
    if not np.isfinite(prediction):
        raise ValueError(f"non-finite Ridge forecast for {target:%Y-%m}")
    cache[target] = prediction
    return prediction


def make_prefix_cache(weekly: pd.DataFrame, months: list[pd.Timestamp]) -> tuple[dict, dict, dict]:
    prefixes, observed_dates, errors = {}, {}, {}
    for month in months:
        for stage in STAGES:
            try:
                prefix, observation_date = observed_prefix(weekly, month, stage)
                prefixes[(month, stage)] = prefix
                observed_dates[(month, stage)] = observation_date
            except WeeklyBridgeDataError as exc:
                errors[(month, stage)] = str(exc)
    return prefixes, observed_dates, errors


def staged_feature_rows(monthly: pd.DataFrame, weekly: pd.DataFrame):
    official_months = list(pd.DatetimeIndex(monthly.index).to_period("M").to_timestamp())
    weekly_months = sorted(set(pd.to_datetime(weekly["accounting_month"]).dt.to_period("M").dt.to_timestamp()))
    months = [m for m in weekly_months if m in set(official_months)]
    prefixes, observed_dates, prefix_errors = make_prefix_cache(weekly, months)
    rows, failures = [], []
    for pos, month in enumerate(months):
        prior = [p for p in months[:pos] if p in official_months]
        for stage in STAGES:
            current = prefixes.get((month, stage))
            if current is None:
                failures.append({"target_month": month.strftime("%Y-%m"), "stage": stage,
                                 "status": "unavailable", "reason": prefix_errors.get((month, stage), "weekly prefix unavailable")})
                continue
            surprises = []
            expected_counts = []
            valid = True
            for component in ("food", "nonfood", "services"):
                history = [prefixes[(p, stage)][component] for p in prior if (p, stage) in prefixes]
                history = history[-12:]
                expected_counts.append(len(history))
                if len(history) < 6:
                    valid = False
                    break
                surprises.append(current[component] - float(np.median(history)))
            if not valid:
                failures.append({"target_month": month.strftime("%Y-%m"), "stage": stage,
                                 "status": "unavailable", "reason": f"insufficient same-stage history; counts={expected_counts}"})
                continue
            rows.append({
                "target_month": month,
                "stage": stage,
                "observation_cutoff": observed_dates[(month, stage)],
                "surprise_food": surprises[0],
                "surprise_nonfood": surprises[1],
                "surprise_services": surprises[2],
            })
    return pd.DataFrame(rows), failures, official_months


def fit_adjustments(train_rows: pd.DataFrame, target_row: pd.Series) -> tuple[float, float]:
    adapter = PooledWeeklyNewsBridge(alpha=ALPHA).fit(train_rows)
    full, calendar = adapter.predict(
        target_row.target_month, int(target_row.stage),
        [target_row.surprise_food, target_row.surprise_nonfood, target_row.surprise_services],
    )
    return full, calendar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None, help="New, unused run directory; default is a UTC-stamped run path")
    parser.add_argument("--weekly-path", default=None)
    args = parser.parse_args()

    monthly = load_model_data("raw", include_macro=False)
    weekly = load_semicolon_weekly_prices(args.weekly_path)
    weekly["accounting_month"] = pd.to_datetime(weekly["accounting_month"], errors="raise")
    if TARGET_COLUMN not in monthly:
        raise ValueError(f"RAW monthly source missing {TARGET_COLUMN!r}")
    weekly_path = Path(weekly.attrs["source_file"])
    overrides_path = Path(weekly.attrs["accounting_overrides_file"]) if weekly.attrs.get("accounting_overrides_file") else None
    monthly_path = ROOT / "data" / "inflation_data.csv"
    run_id = datetime.now(timezone.utc).strftime("weekly_news_%Y%m%dT%H%M%SZ")
    output = Path(args.output_dir) if args.output_dir else ROOT / "experiments" / "weekly_news_bridge" / "runs" / run_id
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"refusing to reuse existing experiment output: {output}")
    output.mkdir(parents=True)

    feature_rows, feature_failures, official_months = staged_feature_rows(monthly, weekly)
    supported = []
    feature_months = set(feature_rows.target_month)
    for month in official_months:
        if month > EVALUATION_END:
            continue
        prior_months = {m for m in feature_months if m < month}
        if len(prior_months) >= MIN_TRAIN_MONTHS:
            supported.append(month)
            break
    if not supported:
        raise ValueError("no target has 18 prior completed months with supported stage features")
    target_start = supported[0]
    targets = [m for m in official_months if target_start <= m <= EVALUATION_END]
    if not targets or targets[-1] != EVALUATION_END:
        raise ValueError(f"official RAW facts do not reach frozen evaluation end {EVALUATION_END:%Y-%m}")

    source_hashes = {"inflation_data.csv": sha256(monthly_path), weekly_path.name: sha256(weekly_path)}
    if overrides_path:
        source_hashes[overrides_path.name] = sha256(overrides_path)
    run_code_manifest = code_manifest(ROOT)
    input_manifest = forecast_source_manifest()
    protocol = {
        "run_id": run_id,
        "model_names": ["Ridge", "CalendarStageCorrection", "WeeklyNewsBridge"],
        "target_start": target_start.strftime("%Y-%m"),
        "target_end": EVALUATION_END.strftime("%Y-%m"),
        "targets": [m.strftime("%Y-%m") for m in targets],
        "stages": list(STAGES),
        "training_cutoff_rule": "target_month - one calendar month; monthly RAW only",
        "observation_cutoff_rule": "actual kth accounting-month weekly date from source file; not a release timestamp",
        "features": ["3 equal-item-weight weekly prefix surprises", "month sin/cos", "stage 1..3 indicators; stage 4 reference"],
        "expected_prefix": "median of up to 12 strictly prior completed monthly prefixes at the same stage; minimum 6",
        "training": {"pooled_stages": True, "minimum_distinct_completed_months": MIN_TRAIN_MONTHS, "alpha": ALPHA, "scaler": "StandardScaler fit on training rows only"},
        "weekly_index_policy": "published change_index means; equal weight within component; no current CPI weights; exact aggregate gasoline row excluded",
        "base_model": "RidgeForecaster(use_macro=False), re-fit at every historical target origin and cached by target month",
        "response": "official RAW target MoM minus origin-specific monthly Ridge forecast",
        "fact_policy": "current_revised_raw",
        "evidence_level": "C",
        "source_hashes": source_hashes,
        "source_manifest": input_manifest,
        "code_hashes": run_code_manifest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plan = pd.MultiIndex.from_product([targets, STAGES], names=["target_month", "stage"]).to_frame(index=False)
    plan = plan.merge(feature_rows, how="left", on=["target_month", "stage"])
    plan["training_cutoff"] = plan.target_month - pd.DateOffset(months=1)
    plan["target_fact_available"] = plan.target_month.isin(official_months)
    plan["status"] = np.where(plan["surprise_food"].notna(), "planned", "unavailable")
    failure_reasons = {(pd.Timestamp(row["target_month"]), int(row["stage"])): row["reason"] for row in feature_failures}
    plan["reason"] = ["" if status == "planned" else failure_reasons.get((m, int(s)), "weekly feature unavailable")
                       for m, s, status in zip(plan.target_month, plan.stage, plan.status)]
    plan.to_csv(output / "planned_rows.csv", index=False)
    (output / "feature_failures.json").write_text(json.dumps(feature_failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    base_cache: dict[pd.Timestamp, float] = {}
    rows, failures = [], []
    # Target dates are frozen before fitting. Each target uses only preceding completed months.
    for month in targets:
        target_stage_rows = plan.loc[plan.target_month.eq(month)].sort_values("stage")
        train_months = set(feature_rows.loc[feature_rows.target_month.lt(month), "target_month"])
        historical = feature_rows.loc[feature_rows.target_month.isin(train_months)].copy()
        residual_by_month = {}
        for past_month in sorted(train_months):
            try:
                base_pred = ridge_forecast(monthly, past_month, base_cache)
                actual = float(monthly.at[past_month, TARGET_COLUMN] - 100.0)
                if not np.isfinite(actual):
                    raise ValueError("non-finite official RAW target fact")
                residual_by_month[past_month] = actual - base_pred
            except Exception as exc:
                failures.append({"target_month": past_month.strftime("%Y-%m"), "stage": None, "status": "unavailable", "reason": f"base residual: {type(exc).__name__}: {exc}"})
        historical = historical.loc[historical.target_month.isin(residual_by_month)].copy()
        historical["base_residual"] = historical.target_month.map(residual_by_month)
        try:
            current_base = ridge_forecast(monthly, month, base_cache)
            actual = float(monthly.at[month, TARGET_COLUMN] - 100.0)
            if not np.isfinite(actual):
                raise ValueError("non-finite official RAW target fact")
        except Exception as exc:
            for _, target_row in target_stage_rows.iterrows():
                failures.append({"target_month": month.strftime("%Y-%m"), "stage": int(target_row.stage), "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        for _, target_row in target_stage_rows.iterrows():
            try:
                common = {"run_id": run_id, "target_month": month.strftime("%Y-%m"), "stage": int(target_row.stage),
                          "training_cutoff": (month - pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
                          "observation_cutoff": (pd.Timestamp(target_row.observation_cutoff).strftime("%Y-%m-%d") if pd.notna(target_row.observation_cutoff) else None),
                          "actual": actual, "evidence_level": "C"}
                rows.append({**common, "model": "Ridge", "prediction": current_base, "error": current_base - actual, "status": "available", "reason": ""})
                if historical.target_month.nunique() < MIN_TRAIN_MONTHS:
                    raise ValueError("fewer than 18 valid prior residual months")
                adapter = PooledWeeklyNewsBridge(alpha=ALPHA).fit(historical)
                calendar_adjustment = adapter.predict_calendar(month, int(target_row.stage))
                rows.append({**common, "model": "CalendarStageCorrection", "prediction": current_base + calendar_adjustment,
                             "error": current_base + calendar_adjustment - actual, "status": "available", "reason": ""})
                if pd.notna(target_row.surprise_food) and pd.notna(target_row.surprise_nonfood) and pd.notna(target_row.surprise_services):
                    full_adjustment, _ = adapter.predict(month, int(target_row.stage),
                        [target_row.surprise_food, target_row.surprise_nonfood, target_row.surprise_services])
                    rows.append({**common, "model": "WeeklyNewsBridge", "prediction": current_base + full_adjustment,
                                 "error": current_base + full_adjustment - actual, "status": "available", "reason": ""})
                else:
                    reason = str(target_row.reason or "weekly feature unavailable")
                    failures.append({"target_month": month.strftime("%Y-%m"), "stage": int(target_row.stage), "status": "unavailable", "reason": reason})
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                failures.append({"target_month": month.strftime("%Y-%m"), "stage": int(target_row.stage), "status": "unavailable", "reason": reason})
                if not any(r.get("target_month") == month.strftime("%Y-%m") and r.get("stage") == int(target_row.stage) and r.get("model") == "Ridge" for r in rows):
                    common = {"run_id": run_id, "target_month": month.strftime("%Y-%m"), "stage": int(target_row.stage),
                              "training_cutoff": (month - pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
                              "observation_cutoff": (pd.Timestamp(target_row.observation_cutoff).strftime("%Y-%m-%d") if pd.notna(target_row.observation_cutoff) else None),
                              "actual": actual, "evidence_level": "C"}
                    rows.append({**common, "model": "Ridge", "prediction": current_base, "error": current_base - actual, "status": "available", "reason": ""})

    predictions = pd.DataFrame(rows)
    # Preserve all planned stage/model slots, including failures before model fitting.
    known = {(r["target_month"], int(r["stage"]), r["model"]) for r in rows}
    failure_lookup = {}
    for failure in failures:
        if failure.get("stage") is not None:
            failure_lookup[(failure["target_month"], int(failure["stage"]))] = failure.get("reason", "unavailable")
    for _, planned in plan.iterrows():
        month_text = planned.target_month.strftime("%Y-%m")
        for name in ("Ridge", "CalendarStageCorrection", "WeeklyNewsBridge"):
            if (month_text, int(planned.stage), name) in known:
                continue
            reason = failure_lookup.get((month_text, int(planned.stage)), str(planned.reason or "model unavailable"))
            rows.append({"run_id": run_id, "model": name, "target_month": month_text, "stage": int(planned.stage),
                         "training_cutoff": planned.training_cutoff.strftime("%Y-%m-%d"),
                         "observation_cutoff": pd.Timestamp(planned.observation_cutoff).strftime("%Y-%m-%d") if pd.notna(planned.observation_cutoff) else None,
                         "prediction": np.nan, "actual": float(monthly.at[planned.target_month, TARGET_COLUMN] - 100.0),
                         "error": np.nan, "evidence_level": "C", "status": "unavailable", "reason": reason})
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output / "forecast_rows.csv", index=False)
    metrics = []
    for stage in STAGES:
        stage_part = predictions.loc[predictions.stage.eq(stage)]
        common_targets = stage_part.pivot(index="target_month", columns="model", values="prediction").dropna().index
        for model in ("Ridge", "CalendarStageCorrection", "WeeklyNewsBridge"):
            part = stage_part.loc[stage_part.model.eq(model)]
            valid = part.loc[np.isfinite(pd.to_numeric(part.error, errors="coerce")), "error"].to_numpy(float)
            common = part.loc[part.target_month.isin(common_targets) & np.isfinite(pd.to_numeric(part.error, errors="coerce")), "error"].to_numpy(float)
            metrics.append({"model": model, "stage": int(stage), "N_planned": len(targets), "N_valid": len(valid), "N_common": len(common),
                            "MAE_common": float(np.mean(np.abs(common))) if len(common) else None,
                            "RMSE_common": float(np.sqrt(np.mean(common ** 2))) if len(common) else None,
                            "bias_common": float(np.mean(common)) if len(common) else None,
                            "hit_within_0_5pp_common": float(np.mean(np.abs(common) <= 0.5)) if len(common) else None,
                            "MAE_available": float(np.mean(np.abs(valid))) if len(valid) else None,
                            "evidence_level": "C"})
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)

    # September is an auxiliary current-month projection only: no target fact or production write.
    if AUXILIARY_MONTH in set(pd.to_datetime(weekly.accounting_month).dt.to_period("M").dt.to_timestamp()):
        try:
            completed = [m for m in official_months if m < AUXILIARY_MONTH]
            surprise, observed_at = build_stage_surprises(weekly, completed, AUXILIARY_MONTH, 4)
            train_months = set(feature_rows.loc[feature_rows.target_month.lt(AUXILIARY_MONTH), "target_month"])
            historical = feature_rows.loc[feature_rows.target_month.isin(train_months)].copy()
            residuals = {}
            for past_month in sorted(train_months):
                residuals[past_month] = float(monthly.at[past_month, TARGET_COLUMN] - 100.0) - ridge_forecast(monthly, past_month, base_cache)
            historical = historical.loc[historical.target_month.isin(residuals)].copy()
            historical["base_residual"] = historical.target_month.map(residuals)
            target_row = pd.Series({"target_month": AUXILIARY_MONTH, "stage": 4,
                                    "surprise_food": surprise[0], "surprise_nonfood": surprise[1], "surprise_services": surprise[2]})
            adjustment, _ = fit_adjustments(historical, target_row)
            base = ridge_forecast(monthly, AUXILIARY_MONTH, base_cache)
            pd.DataFrame([{"model": "WeeklyNewsBridge", "target_month": "2026-09", "stage": 4,
                           "training_cutoff": "2026-08-01", "observation_cutoff": observed_at.strftime("%Y-%m-%d"),
                           "prediction": base + adjustment, "base_ridge": base, "status": "auxiliary_prediction",
                           "evidence_level": "C"}]).to_csv(output / "september_auxiliary_prediction.csv", index=False)
        except Exception as exc:
            failures.append({"target_month": "2026-09", "stage": 4, "status": "unavailable", "reason": f"auxiliary: {type(exc).__name__}: {exc}"})

    end_hashes = {"inflation_data.csv": sha256(monthly_path), weekly_path.name: sha256(weekly_path)}
    if overrides_path:
        end_hashes[overrides_path.name] = sha256(overrides_path)
    invalid_reasons = []
    if end_hashes != source_hashes:
        invalid_reasons.append("source inputs changed during evaluation")
    if forecast_source_manifest() != input_manifest:
        invalid_reasons.append("forecast source manifest changed during evaluation")
    if code_manifest(ROOT) != run_code_manifest:
        invalid_reasons.append("sirena/scripts code changed during evaluation")
    if invalid_reasons:
        failures.extend({"target_month": None, "stage": None, "status": "invalid", "reason": reason} for reason in invalid_reasons)
    deduplicated = {}
    for failure in failures:
        key = (failure.get("target_month"), failure.get("stage"), failure.get("status"), failure.get("reason"))
        deduplicated[key] = failure
    failures = list(deduplicated.values())
    (output / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    planned_models = len(targets) * len(STAGES) * 3
    available_models = int(predictions.status.eq("available").sum())
    incomplete = available_models < planned_models
    summary = {"run_id": run_id, "output_dir": str(output), "status": "INVALID" if invalid_reasons else ("INCOMPLETE" if incomplete else "COMPLETED"),
               "planned_targets": len(targets), "planned_target_stage_rows": int(len(plan)),
               "planned_model_rows": planned_models, "available_model_rows": available_models,
               "failure_count": len(failures), "invalid_reasons": invalid_reasons,
               "interpretation": "revised-vintage retrospective; no real weekly release vintages; diagnostic only"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
