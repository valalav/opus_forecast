"""Small adapters of existing model paths for registered experiments."""
from pathlib import Path
import numpy as np
import pandas as pd


def micro_at_origin(train, context):
    from sirena.models.microcomponent import MicrocomponentForecaster
    if 'micro' not in context['cache']:
        context['cache']['micro'] = MicrocomponentForecaster(
            horizon=1, use_seasonal_adj=False, fallback_policy='parent_seasonal').fit(train)
    return context['cache']['micro']


def independent_groups(train, horizon, context):
    """Existing Micro parent Ridge, identical in both aggregation variants."""
    if 'group_paths' not in context['cache']:
        model = micro_at_origin(train, context)
        dates = pd.date_range(context['cutoff']+pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
        expected = pd.date_range(end=context['cutoff'], periods=13, freq='MS')
        for code, parent in model.parent_models.items():
            tail = parent['history'].tail(13)
            if not tail.index.equals(expected) or not np.isfinite(tail).all():
                raise ValueError(f'Parent {code}: incomplete calendar or shifted cutoff')
        histories = {c: list(v['history'].tail(13)) for c, v in model.parent_models.items()}
        paths, fallback_codes = [], set()
        for date in dates:
            row = {}
            for code, fitted in model.parent_models.items():
                if fitted['fit'] is None:
                    value = histories[code][-12]
                    fallback_codes.add(code)
                else:
                    value = model._model_step(fitted['fit'], histories[code], date-pd.offsets.MonthBegin(1))
                if not np.isfinite(value):
                    raise ValueError(f'Invalid parent forecast {code}')
                histories[code].append(float(value))
                row[code] = float(value)
            paths.append(row)
        context['cache']['group_paths'] = pd.DataFrame(paths, index=dates)
        context['cache']['group_fallbacks'] = sorted(fallback_codes)
    return context['cache']['group_paths']


def group_aggregate_adapter(method):
    def calculate(train, horizon, context):
        from sirena.data.aggregation import aggregate_group_forecasts
        model = micro_at_origin(train, context)
        groups = independent_groups(train, horizon, context)
        result = aggregate_group_forecasts(model.basket, groups, origin=context['cutoff'],
                    method=method, cross_year_policy='freeze')
        records = [{'target_month': str(date.date()), 'group': int(code),
                    'prediction': float(groups.at[date, code]),
                    'weight': float(result.weights.at[date, code]),
                    'contribution': float(result.contributions.at[date, code])}
                   for date in groups.index for code in groups.columns]
        return {'units': 'mom_percent', 'path': result.total.to_numpy(), 'coverage': {
            'weight_vintage': str(model.basket['weight_vintage'].date()),
            'observed_weight': 1., 'forecast_weight': 1.,
            'group_fallbacks': str(context['cache']['group_fallbacks']),
            'aggregation': method, 'cross_year_policy': 'freeze',
            'price_reference_period': str(result.price_reference_period.date()),
            'price_reference_evidence': result.price_reference_evidence}, 'contributions': records}
    return calculate


def m1_adapters():
    from sirena.models.ridge import RidgeForecaster
    from sirena.models.huber import HuberForecaster

    def fit_adapter(factory):
        def calculate(train, horizon, context):
            return {'units': 'mom_percent', 'path': factory().fit(train).forecast(horizon=horizon)}
        return calculate

    def naive(train, horizon, context):
        history = list(train['Все товары и услуги']-100)
        if len(history) < 12 or not np.isfinite(history[-12:]).all():
            raise ValueError('SeasonalNaive needs last 12 complete calendar months')
        result = []
        for _ in range(horizon):
            result.append(history[-12])
            history.append(result[-1])
        return {'units': 'mom_percent', 'path': np.asarray(result)}

    def micro(train, horizon, context):
        model = micro_at_origin(train, context)
        return {'units': 'mom_percent', 'path': model.forecast(horizon), 'coverage': model.coverage}

    return {'Ridge': fit_adapter(RidgeForecaster), 'Huber': fit_adapter(HuberForecaster),
            'SeasonalNaive': naive, 'Micro': micro,
            'GroupsFixed': group_aggregate_adapter('fixed'),
            'GroupsDated': group_aggregate_adapter('dated')}


M1_PARAMETERS = {
    'Ridge': {'implementation': 'RidgeForecaster defaults at code hash'},
    'Huber': {'implementation': 'HuberForecaster defaults at code hash'},
    'SeasonalNaive': {'lag_months': 12},
    'Micro': {'horizon': 1, 'use_seasonal_adj': False, 'fallback_policy': 'parent_seasonal'},
    'GroupsFixed': {'group_predictor': 'existing Micro parent Ridge', 'aggregation': 'fixed', 'cross_year_policy': 'freeze'},
    'GroupsDated': {'group_predictor': 'existing Micro parent Ridge', 'aggregation': 'dated', 'cross_year_policy': 'freeze'},
}


P3_MODEL_VARIANTS = {
    'P3IndependentRidge': 'independent_ridge',
    'P3PartialPooling': 'partial_pooling',
    'P3PoolingLagTrend': 'partial_pooling_trend',
}


def p3_adapters():
    """Build experimental P3 adapters using one common fixed-basket contract.

    Group models expose their 45 parent paths; this adapter applies the same
    fixed aggregation used by GroupsFixed so all ablations share basket,
    weights, and cutoff semantics. The pooled model stays outside production.
    """
    from sirena.models.pooled_components import PooledComponentsForecaster
    from sirena.data.aggregation import aggregate_group_forecasts

    def make_adapter(variant):
        def calculate(train, horizon, context):
            cache_key = f'p3:{variant}'
            if cache_key not in context['cache']:
                model = PooledComponentsForecaster(variant=variant).fit(
                    train, target_col='Все товары и услуги')
                context['cache'][cache_key] = model
            model = context['cache'][cache_key]
            groups = model.forecast_groups(horizon)
            basket = model.basket
            if len(basket['groups']) != 45 or set(groups.columns) != set(basket['groups'].index):
                raise ValueError('P3 requires the complete common 45-group basket')
            result = aggregate_group_forecasts(
                basket, groups, origin=context['cutoff'], method='fixed',
                cross_year_policy='freeze')
            records = [
                {'target_month': str(date.date()), 'group': int(code),
                 'prediction': float(groups.at[date, code]),
                 'weight': float(result.weights.at[date, code]),
                 'contribution': float(result.contributions.at[date, code])}
                for date in groups.index for code in groups.columns
            ]
            return {
                'units': 'mom_percent', 'path': result.total.to_numpy(),
                'coverage': {
                    'weight_vintage': str(basket['weight_vintage'].date()),
                    'observed_weight': 1., 'forecast_weight': 1.,
                    'aggregation': 'fixed', 'cross_year_policy': 'freeze',
                    'group_family': str(model.group_family),
                    'price_reference_period': str(result.price_reference_period.date()),
                    'price_reference_evidence': result.price_reference_evidence,
                },
                'contributions': records,
            }
        return calculate

    return {name: make_adapter(variant) for name, variant in P3_MODEL_VARIANTS.items()}


# This manifest is compared literally with the protocol before any evaluation.
# Constructor and aggregation values are frozen here so protocol edits cannot
# silently change the experiment implementation.
from sirena.models.pooled_components import P3_PARAMETERS as _P3_CORE_PARAMETERS

P3_PARAMETERS = {
    name: {
        'variant': variant,
        **_P3_CORE_PARAMETERS,
        'lags': list(_P3_CORE_PARAMETERS['lags']),
        'feature_ids': list(_P3_CORE_PARAMETERS['feature_ids']),
        'basket_groups': 45,
        'target_col': 'Все товары и услуги',
        'output_units': 'mom_percent',
    }
    for name, variant in P3_MODEL_VARIANTS.items()
}
