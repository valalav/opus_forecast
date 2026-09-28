from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sirena.experiment_models import (
    M1_PARAMETERS,
    P3_MODEL_VARIANTS,
    P3_PARAMETERS,
    m1_adapters,
    p3_adapters,
)


def test_p3_adapter_catalog_matches_locked_variants_and_keeps_m1():
    assert set(p3_adapters()) == set(P3_MODEL_VARIANTS) == set(P3_PARAMETERS)
    assert set(m1_adapters()) == set(M1_PARAMETERS)
    for name, params in P3_PARAMETERS.items():
        assert params['variant'] == P3_MODEL_VARIANTS[name]
        assert params['aggregation'] == 'fixed'
        assert params['cross_year_policy'] == 'freeze'
        assert params['basket_groups'] == 45
        assert params['output_units'] == 'mom_percent'


def test_p3_adapters_use_same_fixed_aggregation_and_emit_group_contributions(monkeypatch):
    import sirena.data.aggregation as aggregation
    import sirena.models.pooled_components as pooled

    group_codes = list(range(45))
    target = pd.Timestamp('2025-01-01')
    calls = []

    class FakeForecaster:
        def __init__(self, variant):
            self.variant = variant
            self.basket = {
                'groups': pd.DataFrame(index=group_codes),
                'weight_vintage': pd.Timestamp('2024-01-01'),
            }
            self.group_family = {code: 'family' for code in group_codes}

        def fit(self, train, target_col):
            assert target_col == 'Все товары и услуги'
            return self

        def forecast_groups(self, horizon):
            return pd.DataFrame(
                np.zeros((horizon, 45)),
                index=pd.date_range(target, periods=horizon, freq='MS'),
                columns=group_codes,
            )

    def aggregate(basket, groups, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            total=pd.Series(np.zeros(len(groups)), index=groups.index),
            weights=pd.DataFrame(1 / 45, index=groups.index, columns=groups.columns),
            contributions=pd.DataFrame(0.0, index=groups.index, columns=groups.columns),
            price_reference_period=pd.Timestamp('2024-12-01'),
            price_reference_evidence='test fixture',
        )

    monkeypatch.setattr(pooled, 'PooledComponentsForecaster', FakeForecaster)
    monkeypatch.setattr(aggregation, 'aggregate_group_forecasts', aggregate)
    adapters = p3_adapters()
    train = pd.DataFrame({'Все товары и услуги': [100.0]}, index=[pd.Timestamp('2024-12-01')])
    context = {'cutoff': pd.Timestamp('2024-12-01'), 'cache': {}}

    for name, adapter in adapters.items():
        result = adapter(train, 1, context)
        assert result['units'] == 'mom_percent'
        assert result['path'].shape == (1,)
        assert len(result['contributions']) == 45
        assert result['coverage']['aggregation'] == 'fixed'
    assert calls == [
        {'origin': pd.Timestamp('2024-12-01'), 'method': 'fixed', 'cross_year_policy': 'freeze'}
    ] * 3


def test_p3_adapter_fails_on_incomplete_group_basket(monkeypatch):
    import sirena.models.pooled_components as pooled

    class IncompleteForecaster:
        def __init__(self, variant):
            self.basket = {'groups': pd.DataFrame(index=range(45))}

        def fit(self, train, target_col):
            return self

        def forecast_groups(self, horizon):
            return pd.DataFrame(
                np.zeros((horizon, 44)),
                index=pd.date_range('2025-01-01', periods=horizon, freq='MS'),
                columns=range(44),
            )

    monkeypatch.setattr(pooled, 'PooledComponentsForecaster', IncompleteForecaster)
    adapter = p3_adapters()['P3IndependentRidge']
    with pytest.raises(ValueError, match='complete common 45-group basket'):
        adapter(
            pd.DataFrame({'Все товары и услуги': [100.0]}, index=[pd.Timestamp('2024-12-01')]),
            1,
            {'cutoff': pd.Timestamp('2024-12-01'), 'cache': {}},
        )
