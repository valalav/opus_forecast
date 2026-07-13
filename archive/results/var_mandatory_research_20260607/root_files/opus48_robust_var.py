#!/usr/bin/env python3
"""
Opus 4.8 — Robust / outlier-aware VAR-family search.

Question: can proper outlier/shock handling improve the mandatory VAR benchmark
(PlainVAR_BIC == VAR(1) on CPI,Food,NonFood,Services) while staying leakage-free?

All robust candidates are VAR-family (dynamics = lagged endogenous), official data only,
train-only transformations (except an explicitly pre-declared deterministic dummy calendar).

Robust candidates:
  1. HuberVAR_l1          : equation-by-equation HuberRegressor on lagged endog. Robust loss,
                            fully train-only, future-blind. (Direction 4)
  2. WinsorizedVAR_l1     : winsorize each training series at train-only expanding 2.5/97.5%
                            quantiles, then OLS VAR(1). Target left untransformed. (Direction 3)
  3. PulseOutlierVAR_l1   : fit VAR(1); flag train months with robust |z|>3.5 (MAD) in any
                            equation; refit VARX-OLS with pulse dummies; future pulse=0. (Direction 2)
  4. InterventionVAR_l1   : OLS VAR(1) + PRE-DECLARED deterministic dummies (sanctions 2022-03..08,
                            COVID 2020-04..05, July tariff). Future dummy from the calendar. (Direction 1)
  5. RegimeMacroVARX_l1   : cutoff-only regime; normal -> VARX(USD,Ruonia,Ki, last-exog);
                            shock -> HuberVAR(1) (robust, no macro). (Direction 6)
  6. RobustFAVAR_f2_l1    : train-only robust(winsorized) standardized PCA factors + VAR(1). (Direction 7)

Baselines: PlainVAR_BIC(==plain_var_tc_l1), FAVAR_f2_l1, VARX_last_exog_l1, ArchivedBVAR, RW, SeasonalNaive.

Research-only. Writes only under runs/opus48_robust_var_*. No production changes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

RESEARCH_DIR = Path(__file__).resolve().parent
ROOT = RESEARCH_DIR.parents[1]
sys.path.append(str(RESEARCH_DIR))
sys.path.append(str(ROOT))

from run_var_sa_backtests import (  # noqa: E402
    Candidate, forecast_var, forecast_bvar, load_official_data,
    _seasonal_means, _future_seasonal, OFFICIAL_COMPONENT_WEIGHTS,
)
from statsmodels.tsa.api import VAR  # noqa: E402

warnings.filterwarnings("ignore")

ENDOG = ["CPI", "Food", "NonFood", "Services"]
W = OFFICIAL_COMPONENT_WEIGHTS  # Food/NonFood/Services weights


# -------------------- pre-declared dummy calendar -------------------------
def declared_dummies(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic, declared BEFORE evaluation; function of the date only."""
    idx = pd.DatetimeIndex(dates)
    df = pd.DataFrame(index=idx)
    sanctions = pd.DatetimeIndex(pd.date_range("2022-03-01", "2022-08-01", freq="MS"))
    covid = pd.DatetimeIndex(["2020-04-01", "2020-05-01"])
    df["d_sanctions"] = idx.isin(sanctions).astype(float)
    df["d_covid"] = idx.isin(covid).astype(float)
    df["d_july"] = (idx.month == 7).astype(float)
    return df


# -------------------- equation-by-equation VAR engine ---------------------
def _design(data: np.ndarray, lags: int, exog: np.ndarray | None):
    """Return X (with intercept, lagged endog, contemporaneous exog) and Y rows."""
    n, k = data.shape
    X, Y = [], []
    for t in range(lags, n):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(data[t - lag])
        if exog is not None:
            row.extend(exog[t])
        X.append(row)
        Y.append(data[t])
    return np.asarray(X), np.asarray(Y)


def _fit_eqs(X, Y, estimator):
    k = Y.shape[1]
    betas = []
    for j in range(k):
        if estimator == "huber":
            try:
                m = HuberRegressor(alpha=0.0, epsilon=1.35, max_iter=300, fit_intercept=False)
                m.fit(X, Y[:, j])
                betas.append(m.coef_)
            except Exception:
                betas.append(np.linalg.lstsq(X, Y[:, j], rcond=None)[0])
        else:
            betas.append(np.linalg.lstsq(X, Y[:, j], rcond=None)[0])
    return np.array(betas)  # (k, p)


def eqvar_forecast(data_df, horizon, endog, lags, estimator="ols",
                   exog_df=None, exog_future_fn=None, target="CPI", min_train=40):
    data = data_df.loc[:, endog].dropna()
    if len(data) < max(min_train, lags + 24):
        return np.nan
    idx = data.index
    arr = data.values.astype(float)
    exog = None
    n_exog = 0
    if exog_df is not None:
        exog = exog_df.reindex(idx).fillna(0.0).values.astype(float)
        n_exog = exog.shape[1]
    X, Y = _design(arr, lags, exog)
    if len(X) < lags + 12:
        return np.nan
    B = _fit_eqs(X, Y, estimator)  # (k, 1 + k*lags + n_exog)

    hist = list(arr[-lags:])  # last `lags` rows
    last_date = idx[-1]
    pred_vec = None
    for step in range(1, horizon + 1):
        fdate = last_date + pd.DateOffset(months=step)
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(hist[-lag])
        if n_exog:
            if exog_future_fn is not None:
                row.extend(exog_future_fn(fdate))
            else:
                row.extend([0.0] * n_exog)
        pred_vec = np.array([row]) @ B.T  # (1,k)
        pred_vec = pred_vec.ravel()
        hist.append(pred_vec)
    if target == "bottom_up":
        d = dict(zip(endog, pred_vec))
        return sum(d[n] * W[n] for n in ("Food", "NonFood", "Services"))
    return float(pred_vec[endog.index("CPI")])


# -------------------- baselines -------------------------------------------
def predict_plain_bic(train, horizon):
    data = train.loc[:, ENDOG].dropna()
    if len(data) < 40:
        return np.nan
    try:
        lags = int(getattr(VAR(data).select_order(maxlags=6), "bic", 1)) or 1
    except Exception:
        lags = 1
    lags = max(1, min(6, lags))
    return forecast_var(data, horizon, Candidate("p", "VAR", "official", tuple(ENDOG),
                        {"lags": lags, "target": "CPI"}, forecast_var, ""), 0)


def predict_archived_bvar(train, horizon, draws=120):
    cand = Candidate("abv", "BVAR", "official", ("CPI", "Food", "USD", "Ruonia"),
                     {"lags": 4, "lambda1": 1.0, "lambda2": 0.5, "lambda3": 1.0,
                      "target": "CPI", "n_draws": draws}, forecast_bvar, "")
    return forecast_bvar(train, horizon, cand, 12345)


def predict_rw(train, horizon):
    s = train["CPI"].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def predict_seasonal_naive(train, horizon):
    mm = _seasonal_means(train, ENDOG, None)
    tm = (train.index.max() + pd.DateOffset(months=horizon)).month
    return float(_future_seasonal(mm, ENDOG, tm)[0])


def _favar_frame(train, factors_n, robust):
    factor_cols = ["Food", "NonFood", "Services", "USD", "Ruonia", "Ki_i"]
    data = train.loc[:, ["CPI"] + factor_cols].dropna()
    if len(data) < 48:
        return None
    z = data.loc[:, factor_cols].values.astype(float)
    if robust:
        lo = np.percentile(z, 2.5, axis=0)
        hi = np.percentile(z, 97.5, axis=0)
        z = np.clip(z, lo, hi)
        center = np.median(z, axis=0)
        scale = 1.4826 * np.median(np.abs(z - center), axis=0)
        scale[scale < 1e-8] = 1.0
        zs = (z - center) / scale
    else:
        mean = z.mean(axis=0); std = z.std(axis=0); std[std < 1e-8] = 1.0
        zs = (z - mean) / std
    _, _, vt = np.linalg.svd(zs, full_matrices=False)
    scores = zs @ vt[:factors_n].T
    fav = pd.DataFrame(index=data.index, data={"CPI": data["CPI"].values})
    for i in range(factors_n):
        fav[f"F{i+1}"] = scores[:, i]
    return fav


def predict_favar(train, horizon, factors_n=2, lags=1, robust=False):
    fav = _favar_frame(train, factors_n, robust)
    if fav is None:
        return np.nan
    cols = list(fav.columns)
    return eqvar_forecast(fav, horizon, cols, lags, "ols", target="CPI", min_train=48)


def predict_varx_last_exog(train, horizon, lags=1):
    endog = ENDOG
    exog = ["USD", "Ruonia", "Ki_i"]
    data = train.loc[:, endog + exog].dropna()
    if len(data) < 42:
        return np.nan
    last_exog = data.iloc[-1][exog].values.astype(float)
    exog_df = data.loc[:, exog]
    return eqvar_forecast(data.loc[:, endog].join(exog_df), horizon, endog, lags, "ols",
                          exog_df=exog_df, exog_future_fn=lambda d: last_exog, target="CPI", min_train=42)


# -------------------- robust candidates -----------------------------------
def predict_huber_var(train, horizon, lags=1):
    return eqvar_forecast(train, horizon, ENDOG, lags, "huber", target="CPI")


def predict_winsorized_var(train, horizon, lags=1, ql=2.5, qh=97.5):
    data = train.loc[:, ENDOG].dropna().copy()
    if len(data) < 40:
        return np.nan
    for c in ENDOG:
        lo, hi = np.percentile(data[c].values, [ql, qh])
        data[c] = data[c].clip(lo, hi)
    return eqvar_forecast(data, horizon, ENDOG, lags, "ols", target="CPI")


def detect_pulses(train, lags=1, z_thr=3.5):
    """Train-only additive outlier months via robust z of VAR(1) residuals."""
    data = train.loc[:, ENDOG].dropna()
    if len(data) < 40:
        return []
    arr = data.values.astype(float)
    X, Y = _design(arr, lags, None)
    B = _fit_eqs(X, Y, "ols")
    resid = Y - X @ B.T  # (n-lags, k)
    flagged = set()
    for j in range(resid.shape[1]):
        r = resid[:, j]
        med = np.median(r)
        mad = 1.4826 * np.median(np.abs(r - med))
        if mad < 1e-8:
            continue
        z = np.abs(r - med) / mad
        for i, zi in enumerate(z):
            if zi > z_thr:
                flagged.add(data.index[lags + i])
    return sorted(flagged)


def predict_pulse_var(train, horizon, lags=1, z_thr=3.5):
    pulses = detect_pulses(train, lags, z_thr)
    if not pulses:
        return predict_plain_bic(train, horizon), pulses
    data = train.loc[:, ENDOG].dropna()
    pulse_df = pd.DataFrame(index=data.index)
    for k, pdate in enumerate(pulses):
        pulse_df[f"p{k}"] = (data.index == pdate).astype(float)
    # future pulse dummies = 0 (transitory)
    pred = eqvar_forecast(data.join(pulse_df), horizon, ENDOG, lags, "ols",
                          exog_df=pulse_df, exog_future_fn=lambda d: [0.0] * len(pulses),
                          target="CPI")
    return pred, pulses


def predict_intervention_var(train, horizon, lags=1):
    data = train.loc[:, ENDOG].dropna()
    if len(data) < 40:
        return np.nan
    dd = declared_dummies(data.index)
    def fut(d):
        return declared_dummies(pd.DatetimeIndex([d])).iloc[0].values.astype(float)
    return eqvar_forecast(data.join(dd), horizon, ENDOG, lags, "ols",
                          exog_df=dd, exog_future_fn=fut, target="CPI")


def regime_is_shock(train):
    cpi = train["CPI"].dropna()
    if len(cpi) < 24:
        return False
    return bool(abs(float(cpi.iloc[-1])) >= 1.0 or float(cpi.iloc[-12:].std()) >= 0.55)


def predict_regime_macro_varx(train, horizon):
    if regime_is_shock(train):
        return predict_huber_var(train, horizon, 1), "shock_huber"
    p = predict_varx_last_exog(train, horizon, 1)
    return p, "normal_varx"


# -------------------- windows / metrics -----------------------------------
def build_windows():
    def rng(s, n): return pd.date_range(s, periods=n, freq="MS")
    return {
        "2018_2019": rng("2018-01-01", 24), "2020_2021": rng("2020-01-01", 24),
        "2022_shock": rng("2022-01-01", 12), "2023": rng("2023-01-01", 12),
        "2024_2025q1": rng("2024-01-01", 15), "selection_2025-04_2026-03": rng("2025-04-01", 12),
    }


def metrics_from(g):
    v = g.dropna(subset=["abs_error"]); n = len(v)
    if n == 0:
        return dict(n=0, MAE=np.nan, RMSE=np.nan, Bias=np.nan, MaxAbs=np.nan, KPI_Viol=np.nan, Coverage=np.nan)
    return dict(n=n, MAE=v["abs_error"].mean(), RMSE=np.sqrt((v["error"]**2).mean()),
                Bias=v["error"].mean(), MaxAbs=v["abs_error"].max(),
                KPI_Viol=int((v["abs_error"] > 0.5).sum()),
                Coverage=float((v["abs_error"] <= 0.5).mean() * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="opus48_robust_var_main")
    ap.add_argument("--horizons", default="1,2,12")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    out_dir = RESEARCH_DIR / "runs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    official = load_official_data()
    windows = build_windows()

    pred_rows, outlier_rows, regime_rows = [], [], []
    t0 = time.time()
    for hz in horizons:
        for wname, wdates in windows.items():
            for td in wdates:
                if td not in official.index:
                    continue
                actual = float(official.loc[td, "CPI"])
                cutoff = td - pd.DateOffset(months=hz)
                tf = official[official.index <= cutoff].copy()
                if len(tf) < 40:
                    continue
                common = dict(horizon=hz, window=wname, target_date=td, actual=actual, cutoff=cutoff, train_n=len(tf))

                p_pulse, pulses = predict_pulse_var(tf, hz)
                p_regime, regime = predict_regime_macro_varx(tf, hz)
                preds = {
                    "PlainVAR_BIC": predict_plain_bic(tf, hz),
                    "HuberVAR_l1": predict_huber_var(tf, hz, 1),
                    "WinsorizedVAR_l1": predict_winsorized_var(tf, hz, 1),
                    "PulseOutlierVAR_l1": p_pulse,
                    "InterventionVAR_l1": predict_intervention_var(tf, hz, 1),
                    "RegimeMacroVARX_l1": p_regime,
                    "RobustFAVAR_f2_l1": predict_favar(tf, hz, 2, 1, robust=True),
                    "FAVAR_f2_l1": predict_favar(tf, hz, 2, 1, robust=False),
                    "VARX_last_exog_l1": predict_varx_last_exog(tf, hz, 1),
                    "ArchivedBVAR": predict_archived_bvar(tf, hz),
                    "RW": predict_rw(tf, hz),
                    "SeasonalNaive": predict_seasonal_naive(tf, hz),
                }
                for mname, pred in preds.items():
                    err = actual - pred if np.isfinite(pred) else np.nan
                    pred_rows.append({**common, "model": mname, "prediction": pred, "error": err,
                                      "abs_error": abs(err) if np.isfinite(err) else np.nan})
                if hz == 1:
                    outlier_rows.append({**common, "n_pulses": len(pulses),
                                         "pulse_dates": ";".join(str(p.date()) for p in pulses)})
                    regime_rows.append({**common, "regime": regime})
        print(f"[h={hz}] done {time.time()-t0:.1f}s", flush=True)

    predictions = pd.DataFrame(pred_rows)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(outlier_rows).to_csv(out_dir / "outlier_log.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(out_dir / "selection_log.csv", index=False)

    # metrics: per window + aggregates required by task
    def agg(df, label, rows):
        for (hz, mname), g in df.groupby(["horizon", "model"]):
            rows.append({"horizon": hz, "window": label, "model": mname, **metrics_from(g)})
    mrows = []
    for (hz, wname, mname), g in predictions.groupby(["horizon", "window", "model"]):
        mrows.append({"horizon": hz, "window": wname, "model": mname, **metrics_from(g)})
    agg(predictions, "AGG_all", mrows)
    agg(predictions[predictions.window != "selection_2025-04_2026-03"], "AGG_out_of_selection", mrows)
    agg(predictions[~predictions.window.isin(["selection_2025-04_2026-03", "2022_shock"])], "AGG_oos_nonshock", mrows)
    metrics = pd.DataFrame(mrows).sort_values(["horizon", "window", "MAE"])
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    comp = metrics[metrics.window.str.startswith("AGG") | (metrics.window == "2022_shock")]
    comp = comp[comp.horizon == 1].pivot_table(index="model", columns="window", values="MAE").round(4)
    comp.to_csv(out_dir / "comparison.csv")

    config = {
        "audit": "opus48 robust/outlier-aware VAR-family",
        "robust_candidates": ["HuberVAR_l1", "WinsorizedVAR_l1", "PulseOutlierVAR_l1",
                              "InterventionVAR_l1", "RegimeMacroVARX_l1", "RobustFAVAR_f2_l1"],
        "baselines": ["PlainVAR_BIC", "FAVAR_f2_l1", "VARX_last_exog_l1", "ArchivedBVAR", "RW", "SeasonalNaive"],
        "declared_dummies": {"sanctions": "2022-03..2022-08", "covid": "2020-04..2020-05", "july_tariff": "month==7"},
        "pulse_detection": {"method": "robust z (MAD) of VAR(1) residuals, any equation", "threshold": 3.5,
                            "future_pulse": 0},
        "regime_rule": "shock if |last CPI|>=1.0 or trailing12 std>=0.55 (cutoff-only)",
        "horizons": horizons,
        "windows": {k: [str(v[0].date()), str(v[-1].date())] for k, v in windows.items()},
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    pd.set_option("display.width", 260)
    print("\n=== h=1 MAE: AGG + 2022_shock ===")
    print(comp.to_string())
    print("\nSaved to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
