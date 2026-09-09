"""Arithmetic and information-boundary tests for complete CPI aggregation."""
import numpy as np
import pandas as pd
import pytest

from sirena.data.aggregation import aggregate_group_forecasts
from sirena.data_loader import DataFreshnessError


def basket(origin='2026-02-01', vintage='2026-01-01'):
    return {
        'weight_vintage': pd.Timestamp(vintage),
        'groups': pd.DataFrame({'Weight_vertical': [.6, .4]}, index=[11, 22]),
        'history': pd.DataFrame(0., index=pd.date_range(vintage, origin, freq='MS'), columns=[11, 22]),
    }


def path(origin='2026-02-01', values=((10., 0.), (0., 20.))):
    return pd.DataFrame(values, index=pd.date_range(pd.Timestamp(origin) + pd.offsets.MonthBegin(),
                                                  periods=len(values), freq='MS'), columns=[11, 22])


def test_exact_dated_arithmetic_and_contribution_identity():
    data = basket()
    data['history'].loc['2026-01-01'] = [10., -10.]
    data['history'].loc['2026-02-01'] = [0., 10.]
    result = aggregate_group_forecasts(data, path(), origin='2026-02-01')
    # At March opening: .6 * 1.1 versus .4 * .9 * 1.1.
    first = np.array([.66, .396]) / 1.056
    second_mass = np.array([.66 * 1.1, .396])
    np.testing.assert_allclose(result.weights.iloc[0], first)
    np.testing.assert_allclose(result.weights.iloc[1], second_mass / second_mass.sum())
    np.testing.assert_allclose(result.total, [first[0] * 10, second_mass[1] / second_mass.sum() * 20])
    np.testing.assert_allclose(result.contributions.sum(axis=1), result.total)
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.)


def test_fixed_baseline_and_permuted_codes():
    result = aggregate_group_forecasts(basket(), path()[[22, 11]], origin='2026-02-01', method='fixed')
    np.testing.assert_allclose(result.weights, [[.6, .4], [.6, .4]])
    np.testing.assert_allclose(result.total, [6., 8.])


def test_future_actual_values_are_ignored_without_mutating_inputs():
    data = basket()
    data['history'] = pd.concat([data['history'], path(values=((9999., np.nan), (np.nan, -100.)))])
    original_history = data['history'].copy(deep=True)
    forecast = path()
    original_forecast = forecast.copy(deep=True)
    actual = aggregate_group_forecasts(data, forecast, origin='2026-02-01')
    expected = aggregate_group_forecasts(basket(), forecast, origin='2026-02-01')
    pd.testing.assert_frame_equal(actual.weights, expected.weights)
    pd.testing.assert_frame_equal(data['history'], original_history)
    pd.testing.assert_frame_equal(forecast, original_forecast)


def test_year_crossing_requires_explicit_freeze_and_keeps_price_chain():
    data = basket(origin='2026-11-01')
    data['history'].iloc[0] = [100., 0.]
    forecast = path(origin='2026-11-01', values=((100., 0.), (0., 10.)))
    with pytest.raises(DataFreshnessError, match='new-year weights'):
        aggregate_group_forecasts(data, forecast, origin='2026-11-01')
    result = aggregate_group_forecasts(data, forecast, origin='2026-11-01', cross_year_policy='freeze')
    assert result.frozen_basket
    assert result.weight_vintage == pd.Timestamp('2026-01-01')
    np.testing.assert_allclose(result.weights.iloc[0], [.75, .25])
    np.testing.assert_allclose(result.weights.iloc[1], [6/7, 1/7])
    assert result.total.iloc[1] == pytest.approx(10/7)


def test_old_vintage_chain_requires_all_previous_year_months():
    data = basket(origin='2027-01-01', vintage='2026-01-01')
    data['history'].loc['2026-01-01'] = [100., 0.]
    forecast = path(origin='2027-01-01')
    result = aggregate_group_forecasts(data, forecast, origin='2027-01-01', cross_year_policy='freeze')
    np.testing.assert_allclose(result.weights.iloc[0], [.75, .25])
    data['history'] = data['history'].drop(pd.Timestamp('2026-06-01'))
    with pytest.raises(DataFreshnessError, match='required month'):
        aggregate_group_forecasts(data, forecast, origin='2027-01-01', cross_year_policy='freeze')


@pytest.mark.parametrize('method', ['fixed', 'dated'])
@pytest.mark.parametrize('fault', ['missing_code', 'extra_code', 'nan', 'infinity', 'zero_price', 'gap', 'duplicate'])
def test_invalid_forecast_partition_never_renormalizes(method, fault):
    forecast = path()
    if fault == 'missing_code':
        forecast = forecast.drop(columns=22)
    elif fault == 'extra_code':
        forecast[33] = 0.
    elif fault == 'nan':
        forecast.iloc[0, 1] = np.nan
    elif fault == 'infinity':
        forecast.iloc[0, 1] = np.inf
    elif fault == 'zero_price':
        forecast.iloc[0, 1] = -100.
    elif fault == 'gap':
        forecast.index = pd.to_datetime(['2026-03-01', '2026-05-01'])
    elif fault == 'duplicate':
        forecast.columns = [11, 11]
    with pytest.raises(DataFreshnessError):
        aggregate_group_forecasts(basket(), forecast, origin='2026-02-01', method=method)


@pytest.mark.parametrize('weights', [[.6, .3], [.6, np.nan], [1.1, -.1], [60., 40.], [1., 0.]])
def test_invalid_weight_mass_rejected(weights):
    data = basket()
    data['groups']['Weight_vertical'] = weights
    with pytest.raises(DataFreshnessError):
        aggregate_group_forecasts(data, path(), origin='2026-02-01')


@pytest.mark.parametrize('fault', ['missing_month', 'missing_code', 'nan', 'zero_price', 'duplicate'])
def test_broken_historical_price_chain_rejected(fault):
    data = basket()
    if fault == 'missing_month':
        data['history'] = data['history'].iloc[1:]
    elif fault == 'missing_code':
        data['history'] = data['history'].drop(columns=22)
    elif fault == 'nan':
        data['history'].iloc[0, 0] = np.nan
    elif fault == 'zero_price':
        data['history'].iloc[0, 0] = -100.
    else:
        data['history'].index = pd.to_datetime(['2026-01-01', '2026-01-01'])
    with pytest.raises(DataFreshnessError):
        aggregate_group_forecasts(data, path(), origin='2026-02-01')


def test_dates_and_units_are_explicit():
    for origin in ['2026-02-28', '2026-02-01 12:00', '2026-02-01T00:00Z']:
        with pytest.raises(DataFreshnessError, match='month-start'):
            aggregate_group_forecasts(basket(), path(), origin=origin)
    with pytest.raises(DataFreshnessError, match='units'):
        aggregate_group_forecasts(basket(), path(), origin='2026-02-01', units='index_100')
    for vintage in ['2027-01-01', '2026-02-01']:
        data = basket()
        data['weight_vintage'] = vintage
        with pytest.raises(DataFreshnessError, match='January annual'):
            aggregate_group_forecasts(data, path(), origin='2026-02-01')


def test_january_forecast_uses_new_vintage_january_change_for_february():
    data = basket(origin='2027-01-01', vintage='2027-01-01')
    data['history'].iloc[0] = [100., 0.]
    result = aggregate_group_forecasts(data, path(origin='2027-01-01'), origin='2027-01-01')
    assert not result.frozen_basket
    np.testing.assert_allclose(result.weights.iloc[0], [.75, .25])


def test_explicit_price_reference_is_separate_from_annual_vintage():
    data=basket(); data['history'].loc['2026-01-01']=[10.,0.]
    explicit=aggregate_group_forecasts(data,path(),origin='2026-02-01',price_reference_period='2026-01-01')
    inferred=aggregate_group_forecasts(data,path(),origin='2026-02-01')
    assert explicit.total.iloc[0]==pytest.approx(6.)
    assert inferred.total.iloc[0]>explicit.total.iloc[0]
    assert explicit.price_reference_evidence=='explicit_caller_reference'
    assert inferred.price_reference_evidence=='assumed_previous_december_not_source_verified'
    with pytest.raises(DataFreshnessError,match='cannot follow'):
        aggregate_group_forecasts(data,path(),origin='2026-02-01',price_reference_period='2026-03-01')
