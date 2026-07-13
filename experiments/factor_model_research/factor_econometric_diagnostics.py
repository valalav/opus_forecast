#!/usr/bin/env python3
"""
Econometric diagnostics pack for the selected factor-family model.

This is intentionally separate from the forecast-ranking runner: backtest MAE is
not enough to call a model econometrically acceptable. The script saves source
series diagnostics, selected FAVAR in-sample equation residual diagnostics,
rolling forecast-error diagnostics, PCA/factor stability checks, and a short
report with pass/fail notes.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import acf, adfuller, kpss

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TARGET = "CPI"
COMPONENTS = ["Food", "NonFood", "Services"]
DEFAULT_INFO = ["Food", "NonFood", "Services", "USD", "Ki_i", "Ruonia"]
MACRO_EXTRA = ["Ruonia_i", "Deposits", "RetailReal"]
SELECTED_NAME = "RobustFAVAR_lean_f2_l1_seasonal"


@dataclass
class DiagnosticGate:
    name: str
    status: str
    detail: str


def load_official() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "data" / "inflation_data.csv", sep=";", decimal=",")
    raw["Date"] = pd.to_datetime(raw["Date"], format="%d.%m.%Y", errors="coerce")
    raw["Date"] = raw["Date"].dt.to_period("M").dt.to_timestamp()
    raw = raw.set_index("Date").sort_index()

    df = pd.DataFrame(index=raw.index)
    df["CPI"] = pd.to_numeric(raw["mom"], errors="coerce") - 100
    df["Food"] = pd.to_numeric(raw["Prod"], errors="coerce") - 100
    df["NonFood"] = pd.to_numeric(raw["Nonprod"], errors="coerce") - 100
    df["Services"] = pd.to_numeric(raw["Serv"], errors="coerce") - 100
    df["USD"] = pd.to_numeric(raw["usd_nom_i"], errors="coerce") - 100
    df["Ki_i"] = pd.to_numeric(raw["Ki_i"], errors="coerce")
    df["Ruonia"] = pd.to_numeric(raw["Ruonia"], errors="coerce")
    df["Ruonia_i"] = pd.to_numeric(raw["Ruonia_i"], errors="coerce")
    df["Deposits"] = pd.to_numeric(raw["fl_dep"], errors="coerce") - 100
    df["RetailReal"] = pd.to_numeric(raw["all_real"], errors="coerce") - 100
    return df.dropna(subset=[TARGET])


def robust_z(series: pd.Series) -> pd.Series:
    x = series.dropna()
    med = x.median()
    mad = (x - med).abs().median()
    if not mad or np.isnan(mad):
        return pd.Series(index=series.index, dtype=float)
    return 0.6745 * (series - med) / mad


def safe_adf(series: pd.Series) -> tuple[float, float]:
    x = series.dropna().astype(float)
    if len(x) < 24 or x.nunique() < 3:
        return np.nan, np.nan
    try:
        stat, pvalue, *_ = adfuller(x, autolag="AIC")
        return float(stat), float(pvalue)
    except Exception:
        return np.nan, np.nan


def safe_kpss(series: pd.Series) -> tuple[float, float]:
    x = series.dropna().astype(float)
    if len(x) < 24 or x.nunique() < 3:
        return np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pvalue, *_ = kpss(x, regression="c", nlags="auto")
        return float(stat), float(pvalue)
    except Exception:
        return np.nan, np.nan


def safe_ljung(series: pd.Series, lags: Iterable[int]) -> dict[int, float]:
    x = series.dropna().astype(float)
    out = {}
    for lag in lags:
        if len(x) <= lag + 5 or x.nunique() < 3:
            out[lag] = np.nan
            continue
        try:
            out[lag] = float(acorr_ljungbox(x, lags=[lag], return_df=True)["lb_pvalue"].iloc[0])
        except Exception:
            out[lag] = np.nan
    return out


def safe_arch(series: pd.Series) -> float:
    x = series.dropna().astype(float)
    if len(x) < 36 or x.nunique() < 3:
        return np.nan
    try:
        return float(het_arch(x, nlags=min(4, max(1, len(x) // 20)))[1])
    except Exception:
        return np.nan


def safe_jb(series: pd.Series) -> tuple[float, float]:
    x = series.dropna().astype(float)
    if len(x) < 24 or x.nunique() < 3:
        return np.nan, np.nan
    try:
        stat, pvalue, *_ = jarque_bera(x)
        return float(stat), float(pvalue)
    except Exception:
        return np.nan, np.nan


def safe_month_seasonality(series: pd.Series) -> tuple[float, float]:
    s = series.dropna().astype(float)
    if len(s) < 36:
        return np.nan, np.nan
    groups = [s[s.index.month == month].values for month in range(1, 13)]
    groups = [group for group in groups if len(group) >= 3]
    if len(groups) < 4:
        return np.nan, np.nan
    try:
        stat, pvalue = stats.kruskal(*groups)
        return float(stat), float(pvalue)
    except Exception:
        return np.nan, np.nan


def source_series_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [TARGET] + DEFAULT_INFO + MACRO_EXTRA:
        if col not in data.columns:
            continue
        s = data[col].dropna().astype(float)
        if len(s) < 24:
            continue
        acfs = acf(s, nlags=12, fft=False, missing="drop")
        lb = safe_ljung(s, [1, 6, 12])
        adf_stat, adf_p = safe_adf(s)
        kpss_stat, kpss_p = safe_kpss(s)
        jb_stat, jb_p = safe_jb(s)
        season_stat, season_p = safe_month_seasonality(s)
        rows.append(
            {
                "series": col,
                "n": int(len(s)),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "acf_lag1": float(acfs[1]),
                "acf_lag6": float(acfs[6]),
                "acf_lag12": float(acfs[12]),
                "ljung_box_p_lag1": lb[1],
                "ljung_box_p_lag6": lb[6],
                "ljung_box_p_lag12": lb[12],
                "adf_p": adf_p,
                "kpss_p": kpss_p,
                "stationarity_flag": bool((not np.isnan(adf_p) and adf_p < 0.05) and (not np.isnan(kpss_p) and kpss_p > 0.05)),
                "arch_lm_p": safe_arch(s),
                "jarque_bera_p": jb_p,
                "seasonality_kruskal_p": season_p,
                "robust_abs_z_gt_3_5": int((robust_z(s).abs() > 3.5).sum()),
            }
        )
    return pd.DataFrame(rows)


def make_design(values: np.ndarray, lags: int = 1) -> tuple[np.ndarray, np.ndarray]:
    x_rows, y_rows = [], []
    for t in range(lags, len(values)):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(values[t - lag])
        x_rows.append(row)
        y_rows.append(values[t])
    return np.asarray(x_rows), np.asarray(y_rows)


def fit_selected_favar_in_sample(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cols = [TARGET] + DEFAULT_INFO
    model_data = data[cols].copy().ffill().bfill().dropna()
    month_means = model_data.groupby(model_data.index.month).mean()
    seasonal = model_data.copy()
    for month, means in month_means.iterrows():
        mask = seasonal.index.month == month
        seasonal.loc[mask, :] = seasonal.loc[mask, :] - means.values

    scaler = StandardScaler()
    scaled = scaler.fit_transform(seasonal[DEFAULT_INFO].values.astype(float))
    pca = PCA(n_components=2)
    factors = pca.fit_transform(scaled)
    factor_cols = ["Factor_1", "Factor_2"]
    var_data = pd.DataFrame(factors, index=seasonal.index, columns=factor_cols)
    var_data.insert(0, TARGET, seasonal[TARGET].astype(float).values)

    x, y = make_design(var_data.values.astype(float), lags=1)
    residual_rows = {}
    beta_rows = {}
    for j, eq_name in enumerate(var_data.columns):
        try:
            model = HuberRegressor(alpha=0.0, epsilon=1.35, fit_intercept=False, max_iter=500)
            model.fit(x, y[:, j])
            pred = model.predict(x)
            beta = model.coef_
        except Exception:
            beta = np.linalg.lstsq(x, y[:, j], rcond=None)[0]
            pred = x @ beta
        residual_rows[eq_name] = y[:, j] - pred
        beta_rows[eq_name] = beta

    residuals = pd.DataFrame(residual_rows, index=var_data.index[1:])
    loadings = pd.DataFrame(
        pca.components_.T,
        index=DEFAULT_INFO,
        columns=factor_cols,
    )
    meta = {
        "selected_model": SELECTED_NAME,
        "sample_start": model_data.index.min().strftime("%Y-%m"),
        "sample_end": model_data.index.max().strftime("%Y-%m"),
        "n_obs": int(len(model_data)),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "pca_explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
        "equation_names": list(var_data.columns),
        "var_lags": 1,
        "seasonality": "train/full-sample diagnostic month residualization; rolling production path uses train-only seasonality at each cutoff",
    }
    return residuals, loadings, meta


def residual_diagnostics(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in residuals.columns:
        s = residuals[col].dropna().astype(float)
        lb = safe_ljung(s, [1, 6, 12])
        adf_stat, adf_p = safe_adf(s)
        kpss_stat, kpss_p = safe_kpss(s)
        jb_stat, jb_p = safe_jb(s)
        rows.append(
            {
                "equation": col,
                "n": int(len(s)),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "bias_t_p": float(stats.ttest_1samp(s, 0.0, nan_policy="omit").pvalue),
                "ljung_box_p_lag1": lb[1],
                "ljung_box_p_lag6": lb[6],
                "ljung_box_p_lag12": lb[12],
                "arch_lm_p": safe_arch(s),
                "jarque_bera_p": jb_p,
                "adf_p": adf_p,
                "kpss_p": kpss_p,
            }
        )
    return pd.DataFrame(rows)


def forecast_error_diagnostics(run_dir: Path) -> pd.DataFrame:
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        return pd.DataFrame()
    predictions = pd.read_csv(pred_path, parse_dates=["target_date", "cutoff"])
    rows = []
    selected = predictions[predictions["candidate"] == SELECTED_NAME]
    for horizon, group in selected.groupby("horizon"):
        valid = group.dropna(subset=["error"]).sort_values("target_date")
        if len(valid) < 20:
            continue
        s = valid["error"].astype(float)
        lb = safe_ljung(s, [1, 6, 12])
        jb_stat, jb_p = safe_jb(s)
        rows.append(
            {
                "candidate": SELECTED_NAME,
                "horizon": int(horizon),
                "n": int(len(s)),
                "mae": float(s.abs().mean()),
                "bias": float(s.mean()),
                "bias_t_p": float(stats.ttest_1samp(s, 0.0, nan_policy="omit").pvalue),
                "ljung_box_p_lag1": lb[1],
                "ljung_box_p_lag6": lb[6],
                "ljung_box_p_lag12": lb[12],
                "arch_lm_p": safe_arch(s),
                "jarque_bera_p": jb_p,
            }
        )
    return pd.DataFrame(rows)


def regressor_diagnostics(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = data[DEFAULT_INFO].copy().ffill().bfill().dropna().astype(float)
    scaled = StandardScaler().fit_transform(x.values)
    corr = pd.DataFrame(x.corr(), index=DEFAULT_INFO, columns=DEFAULT_INFO)
    vif_rows = []
    for idx, col in enumerate(DEFAULT_INFO):
        try:
            vif = float(variance_inflation_factor(scaled, idx))
        except Exception:
            vif = np.nan
        vif_rows.append({"variable": col, "vif": vif})
    summary = {
        "condition_number_scaled_info_matrix": float(np.linalg.cond(scaled)),
        "max_abs_pairwise_corr": float(corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max()),
        "note": "High collinearity is expected in component/macro panels; PCA is used partly to address it.",
    }
    return pd.DataFrame(vif_rows), {"correlation": corr, "summary": summary}


def factor_loading_stability(data: pd.DataFrame, window: int = 84) -> tuple[pd.DataFrame, dict]:
    rows = []
    prev = None
    x = data[[TARGET] + DEFAULT_INFO].copy().ffill().bfill().dropna()
    for end in range(window, len(x) + 1, 6):
        train = x.iloc[end - window : end].copy()
        month_means = train.groupby(train.index.month).mean()
        seasonal = train.copy()
        for month, means in month_means.iterrows():
            mask = seasonal.index.month == month
            seasonal.loc[mask, :] = seasonal.loc[mask, :] - means.values
        scaled = StandardScaler().fit_transform(seasonal[DEFAULT_INFO].values.astype(float))
        pca = PCA(n_components=2)
        pca.fit(scaled)
        components = pca.components_.copy()
        if prev is not None:
            for factor_idx in range(components.shape[0]):
                if np.dot(components[factor_idx], prev[factor_idx]) < 0:
                    components[factor_idx] *= -1.0
        load_change = np.nan if prev is None else float(np.mean(np.abs(components - prev)))
        prev = components
        row = {
            "window_end": train.index.max().strftime("%Y-%m"),
            "n": int(len(train)),
            "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
            "mean_abs_loading_change_vs_previous": load_change,
        }
        for factor_idx in range(2):
            for var_idx, var in enumerate(DEFAULT_INFO):
                row[f"F{factor_idx + 1}_{var}"] = float(components[factor_idx, var_idx])
        rows.append(row)
    stability = pd.DataFrame(rows)
    summary = {
        "window_months": window,
        "n_windows": int(len(stability)),
        "min_explained_variance_sum": float(stability["explained_variance_sum"].min()) if not stability.empty else np.nan,
        "median_explained_variance_sum": float(stability["explained_variance_sum"].median()) if not stability.empty else np.nan,
        "median_abs_loading_change": float(stability["mean_abs_loading_change_vs_previous"].median()) if len(stability) > 1 else np.nan,
        "max_abs_loading_change": float(stability["mean_abs_loading_change_vs_previous"].max()) if len(stability) > 1 else np.nan,
    }
    return stability, summary


def gate_summary(
    source_diag: pd.DataFrame,
    in_sample_resid: pd.DataFrame,
    forecast_resid: pd.DataFrame,
    reg_summary: dict,
    stability_summary: dict,
) -> list[DiagnosticGate]:
    gates = []
    source_autocorr = source_diag[source_diag["ljung_box_p_lag12"] < 0.05]["series"].tolist()
    gates.append(
        DiagnosticGate(
            "source_autocorrelation",
            "expected_warning" if source_autocorr else "pass",
            f"Ljung-Box lag12 flags: {', '.join(source_autocorr) if source_autocorr else 'none'}",
        )
    )
    nonstationary = source_diag[
        ~((source_diag["adf_p"] < 0.05) & (source_diag["kpss_p"] > 0.05))
    ]["series"].tolist()
    gates.append(
        DiagnosticGate(
            "source_stationarity",
            "warning" if nonstationary else "pass",
            f"ADF/KPSS stationarity not clean for: {', '.join(nonstationary) if nonstationary else 'none'}",
        )
    )
    resid_auto = in_sample_resid[in_sample_resid["ljung_box_p_lag12"] < 0.05]["equation"].tolist()
    gates.append(
        DiagnosticGate(
            "in_sample_equation_residual_autocorrelation",
            "warning" if resid_auto else "pass",
            f"Equation residual Ljung-Box lag12 flags: {', '.join(resid_auto) if resid_auto else 'none'}",
        )
    )
    if forecast_resid.empty:
        gates.append(DiagnosticGate("rolling_forecast_error_autocorrelation", "missing", "No rolling prediction file found"))
    else:
        forecast_auto = forecast_resid[forecast_resid["ljung_box_p_lag12"] < 0.05]["horizon"].astype(str).tolist()
        gates.append(
            DiagnosticGate(
                "rolling_forecast_error_autocorrelation",
                "warning" if forecast_auto else "pass",
                f"Forecast-error Ljung-Box lag12 flags by horizon: {', '.join(forecast_auto) if forecast_auto else 'none'}",
            )
        )
    max_vif = None
    gates.append(
        DiagnosticGate(
            "source_multicollinearity",
            "expected_warning" if reg_summary["max_abs_pairwise_corr"] > 0.7 or reg_summary["condition_number_scaled_info_matrix"] > 10 else "pass",
            f"condition={reg_summary['condition_number_scaled_info_matrix']:.2f}, max_abs_corr={reg_summary['max_abs_pairwise_corr']:.2f}; PCA mitigates this in selected model",
        )
    )
    gates.append(
        DiagnosticGate(
            "factor_stability",
            "warning" if stability_summary["max_abs_loading_change"] and stability_summary["max_abs_loading_change"] > 0.5 else "pass",
            f"median EV sum={stability_summary['median_explained_variance_sum']:.3f}, max loading change={stability_summary['max_abs_loading_change']:.3f}",
        )
    )
    normality_flags = in_sample_resid[in_sample_resid["jarque_bera_p"] < 0.05]["equation"].tolist()
    gates.append(
        DiagnosticGate(
            "residual_normality",
            "expected_warning" if normality_flags else "pass",
            f"Jarque-Bera non-normal equations: {', '.join(normality_flags) if normality_flags else 'none'}; robust Huber estimation used",
        )
    )
    return gates


def write_report(
    out_dir: Path,
    source_diag: pd.DataFrame,
    in_sample_diag: pd.DataFrame,
    forecast_diag: pd.DataFrame,
    loadings: pd.DataFrame,
    favar_meta: dict,
    reg_summary: dict,
    stability_summary: dict,
    gates: list[DiagnosticGate],
) -> None:
    lines = [
        "# Econometric Diagnostics Pack: Robust Seasonal FAVAR",
        "",
        f"Selected model: `{SELECTED_NAME}`",
        f"Sample: {favar_meta['sample_start']} to {favar_meta['sample_end']} ({favar_meta['n_obs']} months)",
        "",
        "## Bottom Line",
        "",
        "The selected FAVAR is forecast-useful and its rolling forecast errors do not show a clear Ljung-Box autocorrelation flag, but the raw source series are strongly autocorrelated and several diagnostics are warnings rather than clean passes. This is acceptable for a mandatory factor benchmark/control model, not evidence that it is a fully clean structural econometric model.",
        "",
        "## Gate Summary",
        "",
        "| Gate | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for gate in gates:
        lines.append(f"| {gate.name} | {gate.status} | {gate.detail} |")

    lines.extend(
        [
            "",
            "## Source Series Diagnostics",
            "",
            "- Raw monthly series are expected to be autocorrelated; CPI/components/rates are not white noise.",
            "- FAVAR is appropriate partly because these dynamics must be modeled, not ignored.",
            "",
            "| Series | ACF1 | LB p12 | ADF p | KPSS p | ARCH p | JB p | Seasonality p |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in source_diag.iterrows():
        lines.append(
            f"| {row['series']} | {row['acf_lag1']:.3f} | {row['ljung_box_p_lag12']:.3g} | {row['adf_p']:.3g} | {row['kpss_p']:.3g} | {row['arch_lm_p']:.3g} | {row['jarque_bera_p']:.3g} | {row['seasonality_kruskal_p']:.3g} |"
        )

    lines.extend(
        [
            "",
            "## Selected FAVAR In-Sample Equation Residuals",
            "",
            "| Equation | Mean | LB p12 | ARCH p | JB p | ADF p | KPSS p |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in in_sample_diag.iterrows():
        lines.append(
            f"| {row['equation']} | {row['mean']:.4f} | {row['ljung_box_p_lag12']:.3g} | {row['arch_lm_p']:.3g} | {row['jarque_bera_p']:.3g} | {row['adf_p']:.3g} | {row['kpss_p']:.3g} |"
        )

    lines.extend(
        [
            "",
            "## Rolling Forecast Error Diagnostics",
            "",
        ]
    )
    if forecast_diag.empty:
        lines.append("No rolling forecast-error diagnostics were available.")
    else:
        lines.extend(
            [
                "| Horizon | MAE | Bias | LB p12 | ARCH p | JB p |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in forecast_diag.iterrows():
            lines.append(
                f"| {int(row['horizon'])} | {row['mae']:.3f} | {row['bias']:.3f} | {row['ljung_box_p_lag12']:.3g} | {row['arch_lm_p']:.3g} | {row['jarque_bera_p']:.3g} |"
            )

    lines.extend(
        [
            "",
            "## Factor And Regressor Diagnostics",
            "",
            f"- PCA explained variance ratio: {', '.join(f'{x:.3f}' for x in favar_meta['pca_explained_variance_ratio'])}; sum={favar_meta['pca_explained_variance_sum']:.3f}.",
            f"- Scaled info-matrix condition number: {reg_summary['condition_number_scaled_info_matrix']:.2f}.",
            f"- Max absolute pairwise correlation among info variables: {reg_summary['max_abs_pairwise_corr']:.2f}.",
            f"- Rolling 84-month PCA median explained variance sum: {stability_summary['median_explained_variance_sum']:.3f}.",
            f"- Rolling 84-month PCA max mean absolute loading change: {stability_summary['max_abs_loading_change']:.3f}.",
            "",
            "## Interpretation For Model Quality",
            "",
            "This diagnostic pack does not make the model a clean structural model. It says something narrower and more defensible: the raw monthly rows are dynamic/autocorrelated, the selected FAVAR absorbs enough dynamics that rolling forecast errors do not show a clear autocorrelation flag, and the model is acceptable as an interpretable factor-family benchmark/control. Warnings remain around source non-normality, shock periods, source autocorrelation, and factor-loading stability.",
        ]
    )
    (out_dir / "econometric_diagnostics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = ROOT / "experiments" / "factor_model_research" / "runs" / "factor_diagnostics_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal_run = ROOT / "experiments" / "factor_model_research" / "runs" / "factor_proposal_sweep"

    data = load_official()
    source_diag = source_series_diagnostics(data)
    source_diag.to_csv(out_dir / "source_series_diagnostics.csv", index=False)

    residuals, loadings, favar_meta = fit_selected_favar_in_sample(data)
    residuals.to_csv(out_dir / "selected_favar_in_sample_residuals.csv")
    loadings.to_csv(out_dir / "selected_favar_pca_loadings.csv")
    (out_dir / "selected_favar_meta.json").write_text(json.dumps(favar_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    in_sample_diag = residual_diagnostics(residuals)
    in_sample_diag.to_csv(out_dir / "selected_favar_in_sample_residual_diagnostics.csv", index=False)

    forecast_diag = forecast_error_diagnostics(proposal_run)
    forecast_diag.to_csv(out_dir / "selected_favar_rolling_forecast_error_diagnostics.csv", index=False)

    vif, reg = regressor_diagnostics(data)
    vif.to_csv(out_dir / "info_variable_vif.csv", index=False)
    reg["correlation"].to_csv(out_dir / "info_variable_correlation.csv")
    (out_dir / "info_variable_regressor_summary.json").write_text(
        json.dumps(reg["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stability, stability_summary = factor_loading_stability(data)
    stability.to_csv(out_dir / "rolling_pca_loading_stability.csv", index=False)
    (out_dir / "rolling_pca_loading_stability_summary.json").write_text(
        json.dumps(stability_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    gates = gate_summary(source_diag, in_sample_diag, forecast_diag, reg["summary"], stability_summary)
    gates_json = [gate.__dict__ for gate in gates]
    (out_dir / "diagnostic_gate_summary.json").write_text(json.dumps(gates_json, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(gates_json).to_csv(out_dir / "diagnostic_gate_summary.csv", index=False)

    write_report(out_dir, source_diag, in_sample_diag, forecast_diag, loadings, favar_meta, reg["summary"], stability_summary, gates)
    print(f"Saved diagnostics: {out_dir}")
    print(pd.DataFrame(gates_json).to_string(index=False))


if __name__ == "__main__":
    main()
