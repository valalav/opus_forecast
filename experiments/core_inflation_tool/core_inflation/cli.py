"""CLI for the isolated experimental core inflation tool."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .contributions import ContributionColumns, build_contribution_table, build_series_from_contributions
from .longrun import add_stable_rate_metrics, build_longrun_metrics, causal_robust_filter, render_dynamics_report
from .report import render_jump_report


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, result)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, value = raw_line.strip().partition(":")
        if not separator:
            raise ValueError(f"unsupported YAML line: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return result


def load_config(path: Path) -> dict[str, Any]:
    """Load JSON/YAML config without making PyYAML mandatory for tests."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ImportError:
        return _parse_simple_yaml(text)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _resolve_input_path(path_value: str | os.PathLike[str], base_dir: Path) -> Path:
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates = [base_dir / path, Path.cwd() / path, EXPERIMENT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _monthly_basket_coverage(
    components: pd.DataFrame,
    weights: pd.DataFrame,
    eligible_codes: set[int],
) -> pd.Series:
    annual_weight = (
        weights.loc[weights["item_code"].astype(int).isin(eligible_codes)]
        .groupby("weight_year")["weight"]
        .sum()
    )
    observed = components.dropna(subset=["mom", "weight"]).groupby("date")["weight"].sum()
    expected = pd.Series(
        [annual_weight.get(pd.Timestamp(date).year, float("nan")) for date in observed.index],
        index=observed.index,
        dtype=float,
    )
    return observed / expected


def _components_from_repo_inputs(config: dict[str, Any], config_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    from .diagnostics import basket_coverage_diagnostic, mom_yoy_contamination, range_diagnostic
    from .indicators import index_to_growth
    from .loaders import load_component_basket, load_indices, load_weights

    inputs = config.get("inputs", {})
    required = ["indices", "weights", "basket"]
    missing = [name for name in required if name not in inputs]
    if missing:
        raise ValueError(f"config inputs missing required keys: {', '.join(missing)}")

    region_code = int(config.get("region_code", 7))
    indices = load_indices(_resolve_input_path(inputs["indices"], config_dir), region_code=region_code)
    weights = load_weights(
        _resolve_input_path(inputs["weights"], config_dir),
        region_code=region_code,
        weight_column=str(config.get("weight_column", "Weight_gross")),
    )
    basket = load_component_basket(_resolve_input_path(inputs["basket"], config_dir))
    eligible_codes = set(basket["item_code"].astype(int))

    indices["weight_year"] = pd.to_datetime(indices["date"]).dt.year
    merged = indices.merge(weights, on=["weight_year", "item_code"], how="inner")
    if merged.empty:
        raise ValueError("no overlapping item/date rows between indices and weights")
    merged["mom"] = index_to_growth(merged["mom_index"])

    headline = merged.loc[merged["item_code"].astype(int).eq(1), ["date", "mom"]].copy()
    headline_frame = (
        headline.rename(columns={"mom": "headline_mom"}).drop_duplicates("date") if not headline.empty else None
    )

    components = merged.merge(basket, on="item_code", how="inner")
    excluded_groups = set(config.get("exclude_subcomponents", []))
    excluded_codes = {int(code) for code in config.get("exclude_item_codes", [])}
    components["component"] = components["item_name"].fillna(components["item_code"].astype(str))
    components["excluded"] = components["subcomponent_group"].isin(excluded_groups) | components[
        "item_code"
    ].astype(int).isin(excluded_codes)

    diagnostics = [
        mom_yoy_contamination(components["mom_index"], components["yoy_index"]),
        range_diagnostic(components["mom"]),
        {
            "check": "robust_tail_treatment",
            "status": "pass",
            "message": "raw outliers retained for audit; exclusion core uses weighted 5%/95% winsorization",
        },
        {
            "check": "component_hierarchy",
            "status": "pass",
            "message": f"canonical leaf basket contains {len(eligible_codes)} item codes",
        },
        basket_coverage_diagnostic(
            _monthly_basket_coverage(components, weights, eligible_codes),
            minimum=float(config.get("minimum_weight_coverage", 0.98)),
        ),
    ]

    component_frame = components[["date", "component", "mom", "weight", "excluded"]].copy()
    dropped = int(component_frame[["mom", "weight"]].isna().any(axis=1).sum())
    if dropped:
        diagnostics.append(
            {
                "check": "missing_component_values",
                "status": "warning",
                "message": f"dropped {dropped} canonical component rows with missing MoM or weight",
            }
        )
        component_frame = component_frame.dropna(subset=["mom", "weight"])
    component_frame["date"] = pd.to_datetime(component_frame["date"]).dt.strftime("%Y-%m-%d")
    if headline_frame is not None:
        headline_frame["date"] = pd.to_datetime(headline_frame["date"]).dt.strftime("%Y-%m-%d")

    return component_frame, headline_frame, pd.DataFrame(diagnostics)


def _prefix_diagnostics(diagnostics: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = diagnostics.copy()
    if not out.empty and "check" in out.columns:
        out["check"] = prefix + out["check"].astype(str)
    return out


def _sa_components_from_repo_inputs(
    config: dict[str, Any], config_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame] | None:
    from .diagnostics import basket_coverage_diagnostic, range_diagnostic
    from .indicators import index_to_growth
    from .loaders import load_weights, load_wide_component_indices

    inputs = config.get("inputs", {})
    if "sa_indices" not in inputs:
        return None

    region_code = int(config.get("region_code", 7))
    sa = load_wide_component_indices(_resolve_input_path(inputs["sa_indices"], config_dir))
    weights = load_weights(
        _resolve_input_path(inputs["weights"], config_dir),
        region_code=region_code,
        weight_column=str(config.get("weight_column", "Weight_gross")),
    )
    sa["weight_year"] = pd.to_datetime(sa["date"]).dt.year
    merged = sa.merge(weights, on=["weight_year", "item_code"], how="inner")
    diagnostics = [
        {
            "check": "mom_yoy_distinct",
            "status": "expected_skip",
            "message": "SA source has MoM indices only; MoM/YoY contamination check is not applicable",
        }
    ]
    if merged.empty:
        diagnostics.append(
            {
                "check": "component_coverage",
                "status": "fail",
                "message": "no overlapping item/date rows between SA indices and weights",
            }
        )
        return pd.DataFrame(columns=["date", "component", "mom", "weight", "excluded"]), None, _prefix_diagnostics(
            pd.DataFrame(diagnostics), "sa_"
        )

    merged["mom"] = index_to_growth(merged["index_value"])
    headline = merged.loc[merged["item_code"].astype(int).eq(1), ["date", "mom"]].copy()
    headline_frame = (
        headline.rename(columns={"mom": "headline_mom"}).drop_duplicates("date") if not headline.empty else None
    )

    eligible_codes = {int(code) for code in config.get("sa_component_item_codes", [])}
    if not eligible_codes:
        raise ValueError("config sa_component_item_codes must define one non-overlapping SA hierarchy level")
    components = merged.loc[merged["item_code"].astype(int).isin(eligible_codes)].copy()
    excluded_codes = {int(code) for code in config.get("sa_exclude_item_codes", [])}
    components["component"] = components["item_name"].fillna(components["item_code"].astype(str))
    components["excluded"] = components["item_code"].astype(int).isin(excluded_codes)

    diagnostics.extend(
        [
            range_diagnostic(components["mom"]),
            {
                "check": "robust_tail_treatment",
                "status": "pass",
                "message": "raw SA outliers retained for audit; exclusion core uses weighted 5%/95% winsorization",
            },
            {
                "check": "component_hierarchy",
                "status": "pass" if components["item_code"].nunique() == len(eligible_codes) else "fail",
                "message": (
                    f"configured SA level has {len(eligible_codes)} codes; "
                    f"{components['item_code'].nunique()} are available"
                ),
            },
            basket_coverage_diagnostic(
                _monthly_basket_coverage(components, weights, eligible_codes),
                minimum=float(config.get("minimum_weight_coverage", 0.98)),
            ),
        ]
    )

    component_frame = components[["date", "component", "mom", "weight", "excluded"]].copy()
    dropped = int(component_frame[["mom", "weight"]].isna().any(axis=1).sum())
    if dropped:
        diagnostics.append(
            {
                "check": "missing_component_values",
                "status": "warning",
                "message": f"dropped {dropped} canonical SA component rows with missing MoM or weight",
            }
        )
        component_frame = component_frame.dropna(subset=["mom", "weight"])
    component_frame["date"] = pd.to_datetime(component_frame["date"]).dt.strftime("%Y-%m-%d")
    if headline_frame is not None:
        headline_frame["date"] = pd.to_datetime(headline_frame["date"]).dt.strftime("%Y-%m-%d")

    return component_frame, headline_frame, _prefix_diagnostics(pd.DataFrame(diagnostics), "sa_")


def _diagnostics_from_series(series: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    numeric = series.select_dtypes(include="number")
    value_columns = [
        column
        for column in [
            "headline_mom",
            "exclusion_core_mom",
            "trimmed_mean_mom",
            "weighted_median_mom",
            "stable_core_mom",
            "headline_sa_mom",
            "exclusion_core_sa_mom",
            "trimmed_mean_sa_mom",
            "weighted_median_sa_mom",
            "stable_core_sa_mom",
        ]
        if column in series.columns
    ]
    internal_missing = 0
    for column in value_columns:
        valid = series[column].notna()
        if not valid.any():
            continue
        window = series.loc[valid.idxmax() : valid[::-1].idxmax(), column]
        internal_missing += int(window.isna().sum())
    if internal_missing:
        rows.append(
            {
                "check": "series_internal_no_nan",
                "status": "fail",
                "message": f"series contains {internal_missing} internal NaN values inside indicator coverage windows",
            }
        )
    else:
        rows.append(
            {
                "check": "series_internal_no_nan",
                "status": "pass",
                "message": "indicator coverage windows contain no internal NaN values",
            }
        )
    combined_missing = int(numeric[value_columns].isna().sum().sum()) if value_columns else 0
    if combined_missing:
        rows.append(
            {
                "check": "series_combined_coverage",
                "status": "warning",
                "message": f"combined ordinary/SA panel contains {combined_missing} edge NaN cells from unequal source coverage",
            }
        )
    else:
        rows.append(
            {
                "check": "series_combined_coverage",
                "status": "pass",
                "message": "combined ordinary/SA panel has complete coverage",
            }
        )

    max_abs = float(numeric.abs().max().max()) if not numeric.empty else 0.0
    status = "warning" if max_abs > 20 else "pass"
    rows.append(
        {
            "check": "series_reasonable_range",
            "status": status,
            "message": f"maximum absolute numeric value is {max_abs:.3f}",
        }
    )

    excluded_share = contributions.groupby("date").first()["excluded_weight_sum"] / contributions.groupby("date").first()[
        "total_weight_sum"
    ]
    max_excluded = float(excluded_share.max()) if not excluded_share.empty else 0.0
    rows.append(
        {
            "check": "excluded_weight_share",
            "status": "warning" if max_excluded > 0.5 else "pass",
            "message": f"maximum excluded weight share is {max_excluded:.3f}",
        }
    )
    return pd.DataFrame(rows)


def _weighted_trimmed(values: pd.DataFrame, trim_lower: float, trim_upper: float) -> float:
    from .indicators import weighted_trimmed_mean

    result = weighted_trimmed_mean(
        values["mom"],
        values["weight"],
        trim_lower=trim_lower,
        trim_upper=trim_upper,
    )
    return float(result.value)


def _weighted_median(values: pd.DataFrame) -> float:
    from .indicators import weighted_median

    result = weighted_median(values["mom"], values["weight"])
    return float(result.value)


def _add_distribution_indicators(
    series: pd.DataFrame,
    components: pd.DataFrame,
    trim_lower: float,
    trim_upper: float,
    winsor_lower: float,
    winsor_upper: float,
) -> pd.DataFrame:
    from .indicators import weighted_winsorized_mean

    rows = []
    for date, group in components.groupby("date"):
        included = group.loc[~group["excluded"]]
        rows.append(
            {
                "date": date,
                "exclusion_core_mom": weighted_winsorized_mean(
                    included["mom"],
                    included["weight"],
                    lower=winsor_lower,
                    upper=winsor_upper,
                ),
                "trimmed_mean_mom": _weighted_trimmed(group, trim_lower, trim_upper),
                "weighted_median_mom": _weighted_median(group),
            }
        )
    indicators = pd.DataFrame(rows)
    return series.merge(indicators, on="date", how="left")


def _load_diagnostics(config: dict[str, Any], base_dir: Path, series: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
    diagnostics_path = config.get("input", {}).get("diagnostics_csv")
    if diagnostics_path:
        diagnostics = _read_csv(_resolve_input_path(diagnostics_path, base_dir))
    else:
        diagnostics = _diagnostics_from_series(series, contributions)
    for column in ["check", "status", "message"]:
        if column not in diagnostics.columns:
            raise ValueError(f"diagnostics must include column {column}")
    return diagnostics[["check", "status", "message"]]


def _add_stable_core(series: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = series.sort_values("date").copy()
    smoothing = config.get("smoothing", {})
    alpha = float(smoothing.get("alpha", 0.35))
    max_innovation_pp = float(smoothing.get("max_innovation_pp", 1.0))
    for suffix in ("", "_sa"):
        exclusion = f"exclusion_core{suffix}_mom"
        trimmed = f"trimmed_mean{suffix}_mom"
        signal = f"stable_core{suffix}_signal_mom"
        stable = f"stable_core{suffix}_mom"
        if {exclusion, trimmed}.issubset(out.columns):
            out[signal] = out[[exclusion, trimmed]].mean(axis=1)
            out[stable] = causal_robust_filter(
                out[signal],
                alpha=alpha,
                max_innovation_pp=max_innovation_pp,
            )
    return out


def _build_indicator_series(
    components: pd.DataFrame,
    headline_frame: pd.DataFrame | None,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contributions = build_contribution_table(components, ContributionColumns())
    series = build_series_from_contributions(contributions).rename(
        columns={"exclusion_core_mom": "exclusion_core_raw_mom"}
    )
    if headline_frame is not None:
        series = series.drop(columns=["headline_mom"], errors="ignore").merge(headline_frame, on="date", how="left")
    series = _add_distribution_indicators(
        series,
        contributions[["date", "mom", "weight", "excluded"]],
        float(config.get("trim_lower", config.get("trim", {}).get("lower", 0.10))),
        float(config.get("trim_upper", config.get("trim", {}).get("upper", 0.10))),
        float(config.get("winsor", {}).get("lower", 0.05)),
        float(config.get("winsor", {}).get("upper", 0.05)),
    )
    if "headline_mom" in series:
        series["headline_core_gap"] = series["headline_mom"] - series["exclusion_core_mom"]
    series = _add_stable_core(series, config)
    return series, contributions


def _rename_sa_series(series: pd.DataFrame) -> pd.DataFrame:
    return series.rename(
        columns={
            "headline_mom": "headline_sa_mom",
            "exclusion_core_raw_mom": "exclusion_core_sa_raw_mom",
            "exclusion_core_mom": "exclusion_core_sa_mom",
            "headline_core_gap": "headline_core_sa_gap",
            "total_weight_sum": "sa_total_weight_sum",
            "included_weight_sum": "sa_included_weight_sum",
            "excluded_weight_sum": "sa_excluded_weight_sum",
            "trimmed_mean_mom": "trimmed_mean_sa_mom",
            "weighted_median_mom": "weighted_median_sa_mom",
            "stable_core_signal_mom": "stable_core_sa_signal_mom",
            "stable_core_mom": "stable_core_sa_mom",
        }
    )


def run(
    config_path: Path,
    output_path: Path,
    *,
    allow_output_outside_experiment: bool = False,
) -> dict[str, Path]:
    """Run the Worker C artifact writer."""

    output_path = output_path.resolve()
    if not allow_output_outside_experiment and not _is_relative_to(output_path, OUTPUT_ROOT):
        raise ValueError(f"output path must be under {OUTPUT_ROOT}")

    config = load_config(config_path)
    base_dir = config_path.resolve().parent
    input_config = config.get("input", {})
    columns_config = config.get("columns", {})
    report_config = config.get("report", {})
    precomputed_diagnostics: pd.DataFrame | None = None
    headline_frame: pd.DataFrame | None = None
    sa_bundle: tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame] | None = None

    components_path = input_config.get("components_csv")
    if components_path:
        components = _read_csv(_resolve_input_path(components_path, base_dir))
        column_names = ContributionColumns(
            date=columns_config.get("date", "date"),
            component=columns_config.get("component", "component"),
            mom=columns_config.get("mom", "mom"),
            weight=columns_config.get("weight", "weight"),
            excluded=columns_config.get("excluded", "excluded"),
        )
    elif config.get("inputs"):
        components, headline_frame, precomputed_diagnostics = _components_from_repo_inputs(config, base_dir)
        sa_bundle = _sa_components_from_repo_inputs(config, base_dir)
        column_names = ContributionColumns()
    else:
        raise ValueError("config input.components_csv or repo-style inputs are required")
    if components_path:
        contributions = build_contribution_table(components, column_names)
        series = build_series_from_contributions(contributions).rename(
            columns={"exclusion_core_mom": "exclusion_core_raw_mom"}
        )
        series = _add_distribution_indicators(
            series,
            contributions[["date", "mom", "weight", "excluded"]],
            float(config.get("trim_lower", config.get("trim", {}).get("lower", 0.10))),
            float(config.get("trim_upper", config.get("trim", {}).get("upper", 0.10))),
            float(config.get("winsor", {}).get("lower", 0.05)),
            float(config.get("winsor", {}).get("upper", 0.05)),
        )
        series = _add_stable_core(series, config)
    else:
        series, contributions = _build_indicator_series(components, headline_frame, config)

    sa_contributions = pd.DataFrame()
    if sa_bundle is not None:
        sa_components, sa_headline_frame, sa_diagnostics = sa_bundle
        if not sa_components.empty:
            sa_series, sa_contributions = _build_indicator_series(sa_components, sa_headline_frame, config)
            sa_series = _rename_sa_series(sa_series)
            series = series.merge(sa_series, on="date", how="outer").sort_values("date")
        precomputed_diagnostics = pd.concat([precomputed_diagnostics, sa_diagnostics], ignore_index=True) if precomputed_diagnostics is not None else sa_diagnostics

    if precomputed_diagnostics is None:
        diagnostics = _load_diagnostics(config, base_dir, series, contributions)
    else:
        diagnostics = pd.concat(
            [precomputed_diagnostics, pd.DataFrame(_diagnostics_from_series(series, contributions))],
            ignore_index=True,
        )
        diagnostics = diagnostics[["check", "status", "message"]]
    series["diagnostic_status"] = "fail" if diagnostics["status"].astype(str).str.lower().eq("fail").any() else "pass"
    series = add_stable_rate_metrics(series)
    metrics = build_longrun_metrics(series)

    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "series": output_path / "core_inflation_series.csv",
        "diagnostics": output_path / "core_inflation_diagnostics.csv",
        "contributions": output_path / "core_inflation_contributions.csv",
        "sa_contributions": output_path / "core_inflation_sa_contributions.csv",
        "longrun_metrics": output_path / "core_inflation_longrun_metrics.csv",
        "jump_report": output_path / "core_inflation_jump_report.md",
        "dynamics_report": output_path / "core_inflation_dynamics_report.md",
        "config": output_path / "core_inflation_config_used.json",
    }
    series.to_csv(paths["series"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    contributions.to_csv(paths["contributions"], index=False)
    sa_contributions.to_csv(paths["sa_contributions"], index=False)
    metrics.to_csv(paths["longrun_metrics"], index=False)
    paths["jump_report"].write_text(
        render_jump_report(
            series,
            diagnostics,
            contributions,
            jump_threshold=float(report_config.get("jump_threshold", config.get("jump_threshold_pp", 0.5))),
            lookback_months=int(report_config.get("lookback_months", 36)),
            top_n=int(report_config.get("top_n", 10)),
        ),
        encoding="utf-8",
    )
    smoothing_config = config.get("smoothing", {})
    winsor_config = config.get("winsor", {})
    paths["dynamics_report"].write_text(
        render_dynamics_report(
            series,
            metrics,
            diagnostics,
            smoothing_alpha=float(smoothing_config.get("alpha", 0.35)),
            max_innovation_pp=float(smoothing_config.get("max_innovation_pp", 1.0)),
            winsor_lower=float(winsor_config.get("lower", 0.05)),
            winsor_upper=float(winsor_config.get("upper", 0.05)),
        ),
        encoding="utf-8",
    )
    paths["config"].write_text(
        json.dumps({"config_path": str(config_path), "config": config}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build experimental core inflation analysis artifacts.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    allow_external = os.environ.get("CORE_INFLATION_ALLOW_EXTERNAL_OUTPUT") == "1"
    paths = run(args.config, args.output, allow_output_outside_experiment=allow_external)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
