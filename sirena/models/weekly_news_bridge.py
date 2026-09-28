"""Experimental regularized weekly-news update for the monthly Ridge forecast.

This module deliberately stays outside model registration and production forecast
paths. Weekly indices are equal-weight sample signals, never CPI estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


COMPONENTS = ("food", "nonfood", "services")
STAGES = (1, 2, 3, 4)
MIN_EXPECTATION_MONTHS = 6
EXPECTATION_LOOKBACK = 12
MIN_TRAIN_MONTHS = 18
ALPHA = 10.0


class WeeklyBridgeDataError(ValueError):
    """Raised when an observation needed for a weekly stage is unavailable."""


def _month_start(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def _weekly_month(weekly: pd.DataFrame) -> pd.Series:
    if "accounting_month" in weekly.columns:
        return pd.to_datetime(weekly["accounting_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if "date" not in weekly:
        raise WeeklyBridgeDataError("weekly input has no date column")
    return pd.to_datetime(weekly["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()


def _is_gasoline_aggregate(names: pd.Series) -> pd.Series:
    # The aggregate coexists with grade-level gasoline rows in the export.
    return names.fillna("").astype(str).str.strip().str.casefold().eq("бензин автомобильный")


def observed_prefix(
    weekly: pd.DataFrame,
    target_month: str | pd.Timestamp,
    stage: int,
    observation_cutoff: str | pd.Timestamp | None = None,
) -> tuple[dict[str, float], pd.Timestamp]:
    """Return three equal-item-weight cumulative prefixes through the kth week.

    ``change_index`` is the published weekly index (around 100); no current
    household-basket weights are applied. The observation cutoff is the actual
    date in the weekly file and is not interpreted as a release timestamp.
    """
    if stage not in STAGES:
        raise WeeklyBridgeDataError(f"invalid stage {stage}; expected 1..4")
    required = {"date", "component", "change_index"}
    if not required.issubset(weekly.columns):
        raise WeeklyBridgeDataError(f"weekly columns missing: {sorted(required - set(weekly.columns))}")
    frame = weekly.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["component"] = frame["component"].astype(str).str.strip().str.casefold()
    frame["change_index"] = pd.to_numeric(frame["change_index"], errors="coerce")
    month = _month_start(target_month)
    frame = frame.loc[_weekly_month(frame).eq(month) & frame["date"].notna()]
    if observation_cutoff is not None:
        frame = frame.loc[frame["date"] <= pd.Timestamp(observation_cutoff)]
    dates = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
    if len(dates) < stage:
        raise WeeklyBridgeDataError(f"{month:%Y-%m} stage {stage} has only {len(dates)} observed weekly dates")
    selected_dates = dates[:stage]
    frame = frame.loc[frame["date"].isin(selected_dates)]
    if "product_name" in frame:
        frame = frame.loc[~_is_gasoline_aggregate(frame["product_name"])]

    prefixes: dict[str, float] = {}
    for component in COMPONENTS:
        part = frame.loc[frame["component"].eq(component)]
        means = []
        for date in selected_dates:
            vals = part.loc[part["date"].eq(date), "change_index"].dropna()
            if vals.empty:
                raise WeeklyBridgeDataError(f"{month:%Y-%m} stage {stage}: missing {component} index on {date:%Y-%m-%d}")
            mean_index = float(vals.mean())
            if not np.isfinite(mean_index) or mean_index <= 0:
                raise WeeklyBridgeDataError(f"invalid {component} weekly index on {date:%Y-%m-%d}")
            means.append(mean_index)
        prefixes[component] = float((np.prod(np.asarray(means) / 100.0) - 1.0) * 100.0)
    return prefixes, selected_dates[-1]


def calendar_stage_features(target_month: str | pd.Timestamp, stage: int) -> np.ndarray:
    """Two annual Fourier terms and three indicators (stage 4 is reference)."""
    if stage not in STAGES:
        raise WeeklyBridgeDataError(f"invalid stage {stage}; expected 1..4")
    month = _month_start(target_month).month
    phase = 2.0 * np.pi * (month - 1) / 12.0
    return np.asarray([np.sin(phase), np.cos(phase), *(1.0 if stage == s else 0.0 for s in (1, 2, 3))], dtype=float)


def build_stage_surprises(
    weekly: pd.DataFrame,
    completed_months: Iterable[str | pd.Timestamp],
    target_month: str | pd.Timestamp,
    stage: int,
    observation_cutoff: str | pd.Timestamp | None = None,
) -> tuple[np.ndarray, pd.Timestamp]:
    """Build observed minus trailing same-stage median prefix surprises.

    Only completed months strictly before the target can contribute to the
    expected prefix. The last twelve valid completed months are considered;
    each component needs at least six values.
    """
    target = _month_start(target_month)
    history = sorted({_month_start(m) for m in completed_months if _month_start(m) < target})
    current, observed_at = observed_prefix(weekly, target, stage, observation_cutoff)
    expected: dict[str, float] = {}
    for component in COMPONENTS:
        values = []
        for month in history:
            try:
                prefix, _ = observed_prefix(weekly, month, stage)
            except WeeklyBridgeDataError:
                continue
            values.append(prefix[component])
        values = values[-EXPECTATION_LOOKBACK:]
        if len(values) < MIN_EXPECTATION_MONTHS:
            raise WeeklyBridgeDataError(
                f"{target:%Y-%m} stage {stage}: {component} expected prefix has {len(values)} months; need {MIN_EXPECTATION_MONTHS}"
            )
        expected[component] = float(np.median(values))
    surprises = np.asarray([current[c] - expected[c] for c in COMPONENTS], dtype=float)
    if not np.isfinite(surprises).all():
        raise WeeklyBridgeDataError(f"non-finite weekly surprises for {target:%Y-%m} stage {stage}")
    return surprises, observed_at


@dataclass
class PooledWeeklyNewsBridge:
    """Train-only standardized Ridge adjustment and calendar-only control."""

    alpha: float = ALPHA
    scaler: StandardScaler | None = None
    full_model: Ridge | None = None
    calendar_scaler: StandardScaler | None = None
    calendar_model: Ridge | None = None
    training_months: int = 0

    def fit(self, rows: pd.DataFrame) -> "PooledWeeklyNewsBridge":
        # A failed refit must not leave a stale, apparently usable estimator.
        self.scaler = None
        self.full_model = None
        self.calendar_scaler = None
        self.calendar_model = None
        self.training_months = 0
        required = {"target_month", "stage", "base_residual", "surprise_food", "surprise_nonfood", "surprise_services"}
        if not required.issubset(rows.columns):
            raise ValueError(f"training rows missing: {sorted(required - set(rows.columns))}")
        if rows["target_month"].nunique() < MIN_TRAIN_MONTHS:
            raise ValueError(f"need {MIN_TRAIN_MONTHS} distinct completed training months")
        if not rows["stage"].isin(STAGES).all():
            raise ValueError("training rows contain invalid stages")
        surprises = rows[["surprise_food", "surprise_nonfood", "surprise_services"]].to_numpy(float)
        calendar = np.vstack([calendar_stage_features(m, int(s)) for m, s in zip(rows.target_month, rows.stage)])
        y = rows["base_residual"].to_numpy(float)
        X = np.column_stack([surprises, calendar])
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("non-finite training values; weekly input is unavailable")
        self.scaler = StandardScaler().fit(X)
        self.full_model = Ridge(alpha=self.alpha).fit(self.scaler.transform(X), y)
        self.calendar_scaler = StandardScaler().fit(calendar)
        self.calendar_model = Ridge(alpha=self.alpha).fit(self.calendar_scaler.transform(calendar), y)
        self.training_months = int(rows["target_month"].nunique())
        return self

    def predict(self, target_month: str | pd.Timestamp, stage: int, surprises: Iterable[float]) -> tuple[float, float]:
        if self.scaler is None or self.full_model is None or self.calendar_scaler is None or self.calendar_model is None:
            raise ValueError("adapter is not fitted")
        surprise_values = np.asarray(list(surprises), dtype=float)
        if surprise_values.shape != (3,) or not np.isfinite(surprise_values).all():
            raise ValueError("three finite component surprises are required")
        calendar = calendar_stage_features(target_month, stage).reshape(1, -1)
        full_x = np.concatenate([surprise_values, calendar.ravel()]).reshape(1, -1)
        full_adjustment = float(self.full_model.predict(self.scaler.transform(full_x))[0])
        calendar_adjustment = float(self.calendar_model.predict(self.calendar_scaler.transform(calendar))[0])
        if not np.isfinite([full_adjustment, calendar_adjustment]).all():
            raise ValueError("non-finite weekly bridge adjustment")
        return full_adjustment, calendar_adjustment

    def predict_calendar(self, target_month: str | pd.Timestamp, stage: int) -> float:
        """Predict the matched calendar/stage control even when target weeks are absent."""
        if self.calendar_scaler is None or self.calendar_model is None:
            raise ValueError("adapter is not fitted")
        calendar = calendar_stage_features(target_month, stage).reshape(1, -1)
        value = float(self.calendar_model.predict(self.calendar_scaler.transform(calendar))[0])
        if not np.isfinite(value):
            raise ValueError("non-finite calendar/stage adjustment")
        return value
