"""Regression for observed same-input NGBoost/Ensemble drift."""
import numpy as np
import pandas as pd
import pytest
pytest.importorskip('ngboost')
from sirena.models.ngboost_model import NGBoostForecaster
from sirena.models.ngboost_shock import NGBoostShockForecaster


@pytest.mark.parametrize('cls', [NGBoostForecaster, NGBoostShockForecaster])
def test_forecast_independent_of_global_random_state(cls):
    dates = pd.date_range('2016-01-01', periods=96, freq='MS')
    t = np.arange(len(dates))
    values = 100 + .3 + .2*np.sin(2*np.pi*t/12) + .07*np.cos(t*.8)
    df = pd.DataFrame({'Все товары и услуги': values,
                       'Продовольственные товары': values+.2,
                       'Непродовольственные товары': values-.1,
                       'Услуги': values+.1}, index=dates)
    previous = np.random.get_state()
    try:
        np.random.seed(11)
        first = cls(n_estimators=30).fit(df).forecast(12)
        np.random.seed(927)
        np.random.normal(size=1000)
        second = cls(n_estimators=30).fit(df).forecast(12)
    finally:
        np.random.set_state(previous)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize('cls', [NGBoostForecaster, NGBoostShockForecaster])
@pytest.mark.parametrize('seed', [None, -1, 2**32, 1.5])
def test_invalid_seed_rejected(cls, seed):
    with pytest.raises(ValueError, match='random_state'):
        cls(random_state=seed)
