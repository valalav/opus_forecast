"""Dated, complete CPI weight partition for the Micro forecasting model."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from sirena.data_loader import DataFreshnessError, require_observations


@lru_cache(maxsize=4)
def _read_sources(signatures):
    paths = [Path(entry[0]) for entry in signatures]
    indices, weights, structure, names = [pd.read_csv(p) for p in paths]
    if set(indices.Region_code) != {7}:
        raise DataFreshnessError('Micro indices must contain only KBR region 7')
    indices['Date'] = pd.to_datetime(indices.Day, format='%m/%d/%y %H:%M:%S', errors='raise')
    weights = weights[weights.Region_code.eq(7)].copy()
    weights['Date'] = pd.to_datetime(weights.Day, format='%m/%d/%y %H:%M:%S', errors='raise')
    if indices.duplicated(['Date','Item_code']).any() or weights.duplicated(['Date','Item_code']).any():
        raise DataFreshnessError('Duplicate micro date/item or weight vintage/item')
    pivot = indices.pivot(index='Date', columns='Item_code', values='MoM').sort_index() - 100
    return pivot, weights, structure, names


def load_micro_basket(data_dir, cutoff):
    """Use weights available at cutoff; preserve uncovered parent mass explicitly.

    Type 7 records are retired leaves. The current Item_on flag must not erase
    their historical annual weights. Classification is a revised vintage.
    """
    root = Path(data_dir)
    cutoff = pd.Timestamp(cutoff).to_period('M').to_timestamp()
    paths = [root/n for n in ['kbr_indices.csv','access_weights.csv','items_structure.csv','items_names.csv']]
    signatures = tuple((str(p.resolve()), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
    pivot, weights, structure, names = _read_sources(signatures)
    weights = weights[weights.Date.le(cutoff)]
    if weights.empty:
        raise DataFreshnessError(f'No weight vintage available through {cutoff:%Y-%m}')
    vintage = weights.Date.max()
    weights = weights[weights.Date.eq(vintage)].merge(structure, on='Item_code', validate='one_to_one')
    if not np.isfinite(weights.Weight_vertical).all() or (weights.Weight_vertical < 0).any():
        raise DataFreshnessError('Invalid basket weights')
    groups = weights[weights.Item_type.eq(3) & weights.Weight_vertical.gt(0)].set_index('Item_code')
    if not np.isclose(groups.Weight_vertical.sum(), 1., atol=1e-8, rtol=0):
        raise DataFreshnessError('Subcomponent weights do not form a full CPI partition')
    leaves = weights[weights.Item_type.isin([5,7]) & weights.Weight_vertical.gt(0)].set_index('Item_code')
    if not leaves.Subcomponent.isin(groups.index).all():
        raise DataFreshnessError('Weighted leaf lacks a parent in the dated partition')
    used = leaves.groupby('Subcomponent').Weight_vertical.sum().reindex(groups.index, fill_value=0)
    residual = groups.Weight_vertical - used
    if (residual < -1e-8).any():
        raise DataFreshnessError(f'Overlapping leaf weights: {residual[residual < -1e-8].to_dict()}')
    residual = residual.clip(lower=0)
    history = pivot.loc[:cutoff].copy().asfreq('MS')
    # Group observations are required; missing leaf observations are modeled,
    # never overwritten or silently forward-filled in the official dataset.
    require_observations(history, groups.index, cutoff, 'kbr_indices: parent subcomponents', trailing_months=12)
    return {'history':history, 'leaves':leaves, 'groups':groups,
            'residual':residual, 'weight_vintage':vintage,
            'names':names.set_index('Item_code').Item_name.to_dict()}
