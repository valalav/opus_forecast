"""Complete-partition CPI aggregation with origin-safe price-updated shares.

Inputs use RAW monthly changes in percent (0.5 means an index of 100.5), and
annual expenditure weights expressed as fractions summing to one. This is
an aggregation helper, not an official CPI reconstruction or forecast model.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from sirena.data_loader import DataFreshnessError


@dataclass(frozen=True)
class AggregationResult:
    """Rows are target months; weights are shares before that month's change."""

    weights: pd.DataFrame
    contributions: pd.DataFrame
    total: pd.Series
    origin: pd.Timestamp
    weight_vintage: pd.Timestamp
    method: str
    cross_year_policy: str
    frozen_basket: bool
    price_reference_period: pd.Timestamp
    price_reference_evidence: str


def _month(value, label):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.to_period('M').to_timestamp():
        raise DataFreshnessError(f'{label} must be a timezone-naive month-start date')
    return stamp


def _monthly_frame(frame, label):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise DataFreshnessError(f'{label} requires a DataFrame with a DatetimeIndex')
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise DataFreshnessError(f'{label} has duplicate months or codes')
    if not frame.index.is_monotonic_increasing:
        raise DataFreshnessError(f'{label} months must be ordered')
    for value in frame.index:
        _month(value, label)


def _numeric(values, label):
    try:
        array = values.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataFreshnessError(f'{label} must be numeric') from exc
    if not np.isfinite(array).all():
        raise DataFreshnessError(f'{label} contains missing or non-finite values')
    return array


def aggregate_group_forecasts(
    basket: Mapping,
    forecasts: pd.DataFrame,
    *,
    origin,
    method: str = 'dated',
    cross_year_policy: str = 'error',
    units: str = 'mom_percent',
    price_reference_period=None,
) -> AggregationResult:
    """Aggregate a full forecast path using an existing ``load_micro_basket`` dict.

    ``forecasts`` must contain exactly the basket's parent groups, starting one
    month after ``origin`` with no gaps. ``dated`` multiplies annual weights by
    group price relatives from the preceding December to the origin, then by
    forecast changes recursively. Future observations in basket history are
    never read. ``fixed`` uses the same annual weights at every horizon.

    The default price reference is preceding December, an explicit methodology
    assumption, not verified Access field metadata. Callers may provide a
    documented price_reference_period; the result preserves this distinction.

    The basket must have a January annual vintage available by the origin.
    Crossing its calendar year requires explicit ``cross_year_policy='freeze'``:
    the existing basket and price chain continue, without a January reset or
    invented future expenditure weights. A newer published basket must be
    selected by the caller for a new origin. The helper does not infer release
    dates: ``load_micro_basket`` alone supplies an observation-cutoff vintage,
    not proof of historical publication availability.

    Normalization is over the entire validated partition. Missing forecasts,
    weights, history months, or codes fail instead of reallocating their mass.
    Values returned are monthly percentage-point contributions and MoM percent.
    Neither the basket nor the forecast frame is modified.
    """
    if units != 'mom_percent':
        raise DataFreshnessError('Aggregation units must be mom_percent (not index=100)')
    if method not in {'dated', 'fixed'}:
        raise ValueError('method must be dated or fixed')
    if cross_year_policy not in {'error', 'freeze'}:
        raise ValueError('cross_year_policy must be error or freeze')
    origin = _month(origin, 'origin')
    vintage = _month(basket['weight_vintage'], 'weight_vintage')
    if vintage.month != 1 or vintage > origin:
        raise DataFreshnessError('A January annual weight vintage available by origin is required')
    groups = basket['groups']
    if not isinstance(groups, pd.DataFrame) or 'Weight_vertical' not in groups:
        raise DataFreshnessError('Basket requires groups with Weight_vertical')
    if groups.empty or groups.index.has_duplicates or groups.index.isna().any():
        raise DataFreshnessError('Basket group codes must be nonempty, unique and nonmissing')
    base = _numeric(groups['Weight_vertical'], 'annual weights')
    if (base <= 0).any() or not np.isclose(base.sum(), 1.0, atol=1e-8, rtol=0):
        raise DataFreshnessError('Positive annual weights must form a full partition summing to one')
    codes = groups.index
    _monthly_frame(forecasts, 'forecasts')
    if forecasts.empty or len(forecasts.columns) != len(codes) or set(forecasts.columns) != set(codes):
        raise DataFreshnessError('Forecast codes must cover exactly the full basket partition')
    expected = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=len(forecasts), freq='MS')
    if not forecasts.index.equals(expected):
        raise DataFreshnessError('Forecast months must start after origin and be consecutive')
    changes = _numeric(forecasts.loc[:, codes], 'forecasts')
    if (changes <= -100).any():
        raise DataFreshnessError('MoM percent changes must exceed -100 to keep positive price levels')
    frozen = forecasts.index[-1].year > vintage.year
    if frozen and cross_year_policy != 'freeze':
        raise DataFreshnessError('Unknown new-year weights: explicitly select cross_year_policy=freeze')

    inferred = price_reference_period is None
    reference = (_month(f'{vintage.year-1}-12-01', 'price_reference_period') if inferred
                 else _month(price_reference_period, 'price_reference_period'))
    if reference > origin:
        raise DataFreshnessError('Price reference period cannot follow origin')
    evidence = 'assumed_previous_december_not_source_verified' if inferred else 'explicit_caller_reference'
    levels = np.ones(len(codes))
    if method == 'dated':
        history = basket['history']
        _monthly_frame(history, 'history')
        if not set(codes).issubset(history.columns):
            raise DataFreshnessError('History lacks a weighted group code')
        required = pd.date_range(reference + pd.offsets.MonthBegin(1), origin, freq='MS')
        if not required.isin(history.index).all():
            raise DataFreshnessError('History lacks a required month in the annual price chain')
        observed = _numeric(history.loc[required, codes], 'observed price chain')
        if (observed <= -100).any():
            raise DataFreshnessError('Observed MoM percent changes must exceed -100')
        with np.errstate(over='ignore', invalid='ignore'):
            levels = np.prod(1 + observed / 100, axis=0)

    weight_rows = []
    for row in changes:
        with np.errstate(over='ignore', invalid='ignore'):
            mass = base * levels if method == 'dated' else base.copy()
        if not np.isfinite(mass).all() or (mass <= 0).any() or not np.isfinite(mass.sum()):
            raise DataFreshnessError('Non-finite or non-positive cumulative price mass')
        # This normalization is valid only because all group forecasts passed.
        weight_rows.append(mass / mass.sum())
        if method == 'dated':
            with np.errstate(over='ignore', invalid='ignore'):
                levels = levels * (1 + row / 100)

    weights = pd.DataFrame(weight_rows, index=forecasts.index.copy(), columns=codes.copy())
    contributions = weights * forecasts.loc[:, codes].astype(float)
    if not np.isfinite(contributions.to_numpy()).all():
        raise DataFreshnessError('Non-finite aggregated contributions')
    total = contributions.sum(axis=1).rename('mom_percent')
    return AggregationResult(weights, contributions, total, origin, vintage,
                             method, cross_year_policy, frozen, reference, evidence)
