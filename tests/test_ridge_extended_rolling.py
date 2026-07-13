from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from sirena.models import ModelRegistry
from sirena.models.ridge_extended_rolling import RidgeExtendedRollingForecaster


class RidgeExtendedRollingLike(Protocol):
    seasonality_window: int
    use_macro: bool
    name: str
    ridge: Any
    scaler: Any
    seasonal_norm: pd.Series | None
    _features: list[str] | None

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "RidgeExtendedRollingLike": ...

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, Any]: ...

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame: ...

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series: ...


def build_model(**kwargs: object) -> Any:
    return RidgeExtendedRollingForecaster(**kwargs)


def sample_data() -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=80, freq="MS")
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(80) * 0.25,
            "Продовольственные товары": 100.6 + np.random.randn(80) * 0.3,
            "Непродовольственные товары": 100.3 + np.random.randn(80) * 0.2,
            "Услуги": 100.4 + np.random.randn(80) * 0.25,
            "Ki": 9.5 + np.random.randn(80) * 0.5,
            "Ruonia": 8.0 + np.random.randn(80) * 0.4,
        },
        index=dates,
    )
    return data


def test_model_import_and_registry() -> None:
    model = RidgeExtendedRollingForecaster()
    assert model is not None

    registry_model = ModelRegistry.get("ridge_extended_rolling_24m")
    assert isinstance(registry_model, RidgeExtendedRollingForecaster)


def test_model_parameters() -> None:
    model = cast(RidgeExtendedRollingLike, build_model(use_macro=True, seasonality_window=24))
    assert model.seasonality_window == 24
    assert model.use_macro is True
    assert model.name == "ridge_extended_rolling_24m"


def test_fit_predict_backtest() -> None:
    data = sample_data()
    model = cast(RidgeExtendedRollingLike, build_model(use_macro=True, seasonality_window=24))
    model.fit(data, "Все товары и услуги")

    assert model.ridge is not None
    assert model.scaler is not None
    assert model.seasonal_norm is not None
    assert model._features is not None

    target_date = cast(pd.Timestamp, data.index.to_list()[-1])
    result = model.predict(data, target_date)
    assert "prediction" in result
    assert "pred_ridge" in result
    assert "pred_ets" in result
    assert "seasonality_window" in result
    assert result["seasonality_window"] == 24

    backtest_results = model.backtest(data, start_date="2023-01-01")
    assert isinstance(backtest_results, pd.DataFrame)
    if not backtest_results.empty:
        assert "prediction" in backtest_results.columns
        assert "seasonality_window" in backtest_results.columns


def test_rolling_seasonal_norm_uses_recent_window() -> None:
    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    values = np.full(72, 100.0)

    date_series = pd.Series(dates, index=dates)
    january_mask = cast(pd.Series, date_series.dt.month == 1).to_numpy(dtype=bool)
    values[january_mask] = [101.0, 101.0, 101.0, 110.0, 110.0, 110.0]

    df = pd.DataFrame(
        {
            "Все товары и услуги": values,
            "Продовольственные товары": values,
            "Непродовольственные товары": values,
            "Услуги": values,
            "Ki": np.full(72, 10.0),
            "Ruonia": np.full(72, 8.0),
        },
        index=dates,
    )

    model = cast(RidgeExtendedRollingLike, build_model(use_macro=False, seasonality_window=24))
    prepared = cast(pd.DataFrame, model._prepare_features(df))
    seasonal_norm = model._compute_seasonal_norm(prepared)

    jan_value = float(seasonal_norm.loc[1])
    assert jan_value > 105.0
