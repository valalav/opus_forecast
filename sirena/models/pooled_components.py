"""Experimental pooled monthly models for the 45 CPI subcomponents.

All variants consume the same cutoff-vintage basket and the same complete
training panel. Coefficients are estimated jointly as one panel problem; the
independent baseline is a block-diagonal ridge system, not hundreds of
separately tuned estimators.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sirena.data.micro_basket import load_micro_basket
from sirena.data.aggregation import aggregate_group_forecasts
from sirena.models.base import BaseForecaster


P3_PARAMETERS = {
    "train_start": "2016-01-01",
    "lags": (1, 2, 3, 6, 12),
    "feature_ids": ("lag1", "lag2", "lag3", "lag6", "lag12", "month_sin", "month_cos", "lagged_common_trend"),
    "group_family_source": "items_structure.Item_type=3.Component (2/3/4)",
    "seasonality": "target-month sin/cos",
    "ridge_alpha": 25.0,
    "pooling_penalty": 100.0,
    "trend_penalty": 25.0,
    "trend": "cutoff-vintage weighted 45-group mean at forecast origin (lag 1 to target)",
    "aggregation": "fixed",
    "cross_year_policy": "freeze",
    "minimum_panel_rows": 24,
}

_VARIANTS = {"independent_ridge", "partial_pooling", "partial_pooling_trend"}
_COMPONENT_LABELS = {
    2: "nonfood",
    3: "food",
    4: "services",
}
_FEATURES = ("lag1", "lag2", "lag3", "lag6", "lag12", "month_sin", "month_cos")


class PooledComponentsForecaster(BaseForecaster):
    """Forecast 45 monthly RAW parent groups with fixed P3 ablations."""

    name = "pooled_components"
    input_representation = "raw"

    def __init__(
        self,
        variant="partial_pooling",
        data_dir=None,
        train_start=P3_PARAMETERS["train_start"],
        ridge_alpha=P3_PARAMETERS["ridge_alpha"],
        pooling_penalty=P3_PARAMETERS["pooling_penalty"],
        trend_penalty=P3_PARAMETERS["trend_penalty"],
    ):
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {sorted(_VARIANTS)}")
        self.variant = variant
        self.data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[2] / "data"
        self.train_start = pd.Timestamp(train_start)
        self.ridge_alpha = float(ridge_alpha)
        self.pooling_penalty = float(pooling_penalty)
        self.trend_penalty = float(trend_penalty)
        self._is_fitted = False
        self.diagnostics = {}
        self.basket = None

    def fit(self, df, target_col="Все товары и услуги"):
        """Fit through ``df``'s final month; no post-cutoff data is consulted."""
        self._is_fitted = False
        self.coef_ = None
        self.diagnostics = {}
        if df is None or df.empty:
            raise ValueError("PooledComponents requires a nonempty cutoff frame")
        supplied = df.attrs.get("input_contract", {}).get("representation")
        if supplied == "sa":
            raise ValueError("PooledComponents requires homogeneous RAW inputs")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("PooledComponents requires a DatetimeIndex")

        cutoff = pd.Timestamp(df.index.max()).to_period("M").to_timestamp()
        self.cutoff = cutoff
        self._last_train_date = cutoff
        self.basket = load_micro_basket(self.data_dir, cutoff)
        history = self.basket["history"].loc[:cutoff].copy().asfreq("MS")
        groups = self.basket["groups"].sort_index()
        codes = [int(code) for code in groups.index]
        if len(codes) != 45 or history[codes].shape[1] != 45:
            raise ValueError(f"Expected the validated 45-group partition, got {len(codes)}")
        self.group_codes = codes
        self.group_weights = groups.Weight_vertical.astype(float).reindex(codes)
        self.group_family = self._load_group_families(codes)
        if not np.isclose(self.group_weights.sum(), 1.0, atol=1e-8, rtol=0):
            raise ValueError("Cutoff group weights do not sum to one")

        y = history[codes].apply(pd.to_numeric, errors="coerce")
        y = y.loc[y.index >= self.train_start]
        # One common complete-case panel keeps all variants and groups aligned.
        train_rows = []
        for target_date in y.index:
            origin = target_date - pd.DateOffset(months=1)
            if origin < self.train_start or target_date > cutoff:
                continue
            lag_dates = [origin - pd.DateOffset(months=lag - 1) for lag in P3_PARAMETERS["lags"]]
            required_dates = [target_date, *lag_dates]
            if any(date not in y.index for date in required_dates):
                continue
            block = y.reindex(required_dates)
            if not np.isfinite(block.to_numpy(dtype=float)).all():
                continue
            for code in codes:
                vals = [float(y.at[date, code]) for date in lag_dates]
                target = float(y.at[target_date, code])
                month = target_date.month
                row = vals + [np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]
                trend = float(y.loc[origin, codes].to_numpy(dtype=float) @ self.group_weights.to_numpy())
                train_rows.append((target_date, code, row, trend, target))
        if not train_rows:
            raise ValueError("No complete pooled-component training rows")

        dates = pd.DatetimeIndex(sorted({r[0] for r in train_rows}))
        if len(dates) < P3_PARAMETERS["minimum_panel_rows"]:
            raise ValueError(f"Only {len(dates)} complete panel rows; need at least {P3_PARAMETERS['minimum_panel_rows']}")
        features = np.asarray([r[2] for r in train_rows], dtype=float)
        trend_values = np.asarray([r[3] for r in train_rows], dtype=float)
        targets = np.asarray([r[4] for r in train_rows], dtype=float)
        row_codes = np.asarray([r[1] for r in train_rows], dtype=int)
        row_dates = pd.DatetimeIndex([r[0] for r in train_rows])

        # Train-only pooled scaling. Seasonal terms remain on their natural scale.
        self.feature_mean_ = features[:, :5].mean(axis=0)
        self.feature_scale_ = features[:, :5].std(axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-10] = 1.0
        features[:, :5] = (features[:, :5] - self.feature_mean_) / self.feature_scale_
        self.trend_mean_ = float(trend_values.mean())
        self.trend_scale_ = float(trend_values.std())
        if self.trend_scale_ < 1e-10:
            self.trend_scale_ = 1.0
        trend_values = (trend_values - self.trend_mean_) / self.trend_scale_

        self._feature_names = list(_FEATURES)
        diagnostic_feature_names = list(self._feature_names)
        if self.variant == "partial_pooling_trend":
            diagnostic_feature_names.append("lagged_common_trend")
        self._fit_panel(features, trend_values, targets, row_codes)

        fitted = self._predict_panel(features, trend_values, row_codes)
        residuals = targets - fitted
        records = {}
        for code in codes:
            mask = row_codes == code
            records[code] = {
                "nobs": int(mask.sum()),
                "train_start": str(row_dates[mask].min().date()),
                "train_end": str(row_dates[mask].max().date()),
                "target_mean": float(targets[mask].mean()),
                "fitted_mean": float(fitted[mask].mean()),
                "residual_mean": float(residuals[mask].mean()),
                "residual_std": float(residuals[mask].std(ddof=1)),
                "residual_acf1": self._acf1(residuals[mask]),
                "family": self.group_family[code],
                "coefficients": {"intercept": float(self._group_intercept(code)), **{
                    name: float(value) for name, value in zip(diagnostic_feature_names, self._group_coefficients(code))
                }},
            }
        design = self._design_matrix(features, trend_values, row_codes)
        gram_eigenvalues = np.linalg.eigvalsh(self._penalized_gram)
        positive = gram_eigenvalues[gram_eigenvalues > 1e-12]
        self.diagnostics = {
            "variant": self.variant,
            "cutoff": str(cutoff.date()),
            "n_panel_months": len(dates),
            "n_panel_rows": len(train_rows),
            "groups": records,
            "feature_names": diagnostic_feature_names,
            "penalized_gram_condition_number": float(positive[-1] / positive[0]) if len(positive) else float("inf"),
            "fit_predict_max_abs_diff": float(np.max(np.abs(design @ self.coef_ - fitted))),
            "parameters": dict(P3_PARAMETERS),
            "aggregation": "fixed",
            "weight_vintage": str(pd.Timestamp(self.basket["weight_vintage"]).date()),
        }
        self._is_fitted = True
        return self

    def _load_group_families(self, codes):
        structure = pd.read_csv(self.data_dir / "items_structure.csv")
        groups = structure.loc[structure.Item_type.eq(3), ["Item_code", "Component"]]
        mapping = {int(row.Item_code): int(row.Component) for row in groups.itertuples(index=False)}
        if any(code not in mapping or mapping[code] not in _COMPONENT_LABELS for code in codes):
            raise ValueError("Dated group structure lacks a supported parent Component")
        return {code: _COMPONENT_LABELS[mapping[code]] for code in codes}

    def _fit_panel(self, features, trend, targets, row_codes):
        codes = self.group_codes
        p = len(self._feature_names)
        xbase = features
        self._n_features = p
        if self.variant == "independent_ridge":
            # A single block-diagonal least-squares system is exactly 45 independent ridges.
            design = np.zeros((len(targets), len(codes) * (p + 1)), dtype=float)
            for i, code in enumerate(codes):
                mask = row_codes == code
                start = i * (p + 1)
                design[mask, start] = 1.0
                design[mask, start + 1:start + p + 1] = xbase[mask]
            penalty = np.eye(design.shape[1]) * self.ridge_alpha
            for i in range(len(codes)):
                penalty[i * (p + 1), i * (p + 1)] = 0.0
            self.coef_ = self._ridge_solve(design, targets, penalty)
            self._independent_design = design
            self._design = design
            return

        families = sorted(set(self.group_family.values()))
        fam_index = {fam: i for i, fam in enumerate(families)}
        # Family mean parameters plus zero-sum group deviations make the
        # decomposition identifiable. Orthonormal null-space bases preserve
        # the stated L2 deviation penalty exactly.
        from scipy.linalg import null_space
        dev_bases = {}
        group_positions = {code: i for i, code in enumerate(codes)}
        for fam in families:
            members = [code for code in codes if self.group_family[code] == fam]
            dev_bases[fam] = (members, null_space(np.ones((1, len(members)))))

        n_family = len(families)
        common_cols = n_family * (p + 1)
        deviation_cols = sum(len(members) - 1 for members, _ in dev_bases.values()) * (p + 1)
        trend_col = common_cols + deviation_cols if self.variant == "partial_pooling_trend" else None
        total_cols = common_cols + deviation_cols + (1 if trend_col is not None else 0)
        design = np.zeros((len(targets), total_cols), dtype=float)
        penalty = np.zeros((total_cols, total_cols), dtype=float)
        dev_starts = {}
        col = common_cols
        for fam in families:
            members, basis = dev_bases[fam]
            dev_starts[fam] = col
            col += basis.shape[1] * (p + 1)
        for r, code in enumerate(row_codes):
            fam = self.group_family[int(code)]
            fi = fam_index[fam]
            family_start = fi * (p + 1)
            design[r, family_start] = 1.0
            design[r, family_start + 1:family_start + p + 1] = xbase[r]
            members, basis = dev_bases[fam]
            local = members.index(int(code))
            basis_row = basis[local]
            start = dev_starts[fam]
            for j, weight in enumerate(basis_row):
                block_start = start + j * (p + 1)
                design[r, block_start] = weight
                design[r, block_start + 1:block_start + p + 1] = weight * xbase[r]
        # Family intercepts are unpenalized; all slope means use ridge alpha.
        for fi in range(n_family):
            start = fi * (p + 1)
            penalty[start + 1:start + p + 1, start + 1:start + p + 1] += np.eye(p) * self.ridge_alpha
        for fam in families:
            members, basis = dev_bases[fam]
            start = dev_starts[fam]
            # deviation intercepts are penalized too, with sum(delta)=0 by construction
            lam = self.pooling_penalty
            for j in range(basis.shape[1]):
                block_start = start + j * (p + 1)
                penalty[block_start:block_start + p + 1, block_start:block_start + p + 1] += np.eye(p + 1) * lam
        if trend_col is not None:
            design[:, trend_col] = trend
            penalty[trend_col, trend_col] = self.trend_penalty
        self.coef_ = self._ridge_solve(design, targets, penalty)
        self._design = design
        self._families = families
        self._family_index = fam_index
        self._dev_bases = dev_bases
        self._dev_starts = dev_starts
        self._trend_col = trend_col

    def _ridge_solve(self, design, target, penalty):
        lhs = design.T @ design + penalty
        self._penalized_gram = lhs
        rhs = design.T @ target
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    def _predict_panel(self, features, trend, row_codes):
        if self.variant == "independent_ridge":
            p = self._n_features
            out = np.empty(len(row_codes), dtype=float)
            for r, code in enumerate(row_codes):
                start = self.group_codes.index(int(code)) * (p + 1)
                out[r] = self.coef_[start] + features[r] @ self.coef_[start + 1:start + p + 1]
            return out
        xbase = features
        out = np.empty(len(row_codes), dtype=float)
        for r, code in enumerate(row_codes):
            fam = self.group_family[int(code)]
            start = self._family_index[fam] * (self._n_features + 1)
            beta = self.coef_[start:start + self._n_features + 1].copy()
            members, basis = self._dev_bases[fam]
            local = members.index(int(code))
            dev_start = self._dev_starts[fam]
            for j, weight in enumerate(basis[local]):
                block_start = dev_start + j * (self._n_features + 1)
                beta += weight * self.coef_[block_start:block_start + self._n_features + 1]
            out[r] = beta[0] + xbase[r] @ beta[1:]
            if self._trend_col is not None:
                out[r] += trend[r] * self.coef_[self._trend_col]
        return out

    def _group_coefficients(self, code):
        p = self._n_features
        if self.variant == "independent_ridge":
            idx = self.group_codes.index(code) * (p + 1)
            return self.coef_[idx + 1:idx + p + 1]
        fam = self.group_family[code]
        start = self._family_index[fam] * (p + 1)
        beta = self.coef_[start + 1:start + p + 1].copy()
        members, basis = self._dev_bases[fam]
        dev_start = self._dev_starts[fam]
        for j, weight in enumerate(basis[members.index(code)]):
            block_start = dev_start + j * (p + 1)
            beta += weight * self.coef_[block_start + 1:block_start + p + 1]
        if self._trend_col is not None:
            beta = np.append(beta, self.coef_[self._trend_col])
        return beta

    def _group_intercept(self, code):
        p = self._n_features
        if self.variant == "independent_ridge":
            return float(self.coef_[self.group_codes.index(code) * (p + 1)])
        fam = self.group_family[code]
        start = self._family_index[fam] * (p + 1)
        intercept = float(self.coef_[start])
        members, basis = self._dev_bases[fam]
        dev_start = self._dev_starts[fam]
        for j, weight in enumerate(basis[members.index(code)]):
            intercept += float(weight * self.coef_[dev_start + j * (p + 1)])
        return intercept

    def _design_matrix(self, features, trend, row_codes):
        if self.variant == "independent_ridge":
            return self._independent_design
        return self._design

    @staticmethod
    def _acf1(values):
        if len(values) < 3 or np.std(values) < 1e-12:
            return 0.0
        return float(np.corrcoef(values[:-1], values[1:])[0, 1])

    def forecast_groups(self, horizon=1):
        self._check_fitted()
        h = int(horizon)
        if h < 1:
            raise ValueError("Positive forecast horizon required")
        history = self.basket["history"].loc[:self.cutoff, self.group_codes]
        if not np.isfinite(history.tail(12).to_numpy(dtype=float)).all():
            raise ValueError("Missing required 12-month group history at forecast origin")
        values = {code: list(history[code].tail(12).astype(float)) for code in self.group_codes}
        output = []
        weights = self.group_weights.to_numpy(dtype=float)
        for step in range(1, h + 1):
            target_date = self.cutoff + pd.DateOffset(months=step)
            x = np.empty((len(self.group_codes), len(_FEATURES)), dtype=float)
            for i, code in enumerate(self.group_codes):
                seq = values[code]
                x[i, :5] = [seq[-lag] for lag in (1, 2, 3, 6, 12)]
                x[i, 5] = np.sin(2 * np.pi * target_date.month / 12)
                x[i, 6] = np.cos(2 * np.pi * target_date.month / 12)
            x[:, :5] = (x[:, :5] - self.feature_mean_) / self.feature_scale_
            trend_value = float(np.asarray([values[code][-1] for code in self.group_codes]) @ weights)
            trend_scaled = (trend_value - self.trend_mean_) / self.trend_scale_
            codes = np.asarray(self.group_codes, dtype=int)
            preds = self._predict_panel(x, np.full(len(codes), trend_scaled), codes)
            if not np.isfinite(preds).all():
                raise ValueError(f"Non-finite group forecast at {target_date:%Y-%m}")
            output.append(pd.Series(preds, index=self.group_codes, name=target_date))
            for code, pred in zip(self.group_codes, preds):
                values[code].append(float(pred))
        return pd.DataFrame(output)

    def forecast(self, horizon=1):
        groups = self.forecast_groups(horizon)
        result = aggregate_group_forecasts(
            self.basket,
            groups,
            origin=self.cutoff,
            method="fixed",
            cross_year_policy="freeze",
            units="mom_percent",
        )
        self.last_group_forecasts = groups
        self.last_aggregation = result
        return result.total.to_numpy(dtype=float)

    def backtest(self, df, start_date="2019-01-01", target_col="Все товары и услуги"):
        """Use the registered matched-experiment runner for cutoff-safe comparisons."""
        raise NotImplementedError("P3 comparisons use sirena.evaluation.run_registered_experiment")


class P3IndependentRidgeForecaster(PooledComponentsForecaster):
    name = "P3IndependentRidge"

    def __init__(self, **kwargs):
        super().__init__(variant="independent_ridge", **kwargs)


class P3PartialPoolingForecaster(PooledComponentsForecaster):
    name = "P3PartialPooling"

    def __init__(self, **kwargs):
        super().__init__(variant="partial_pooling", **kwargs)


class P3PoolingLagTrendForecaster(PooledComponentsForecaster):
    name = "P3PoolingLagTrend"

    def __init__(self, **kwargs):
        super().__init__(variant="partial_pooling_trend", **kwargs)
