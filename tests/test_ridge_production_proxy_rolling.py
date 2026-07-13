"""Focused tests for RidgeProductionProxyRollingForecaster."""

from pathlib import Path
import sys
from typing import Any, Protocol, cast
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.ridge_production_proxy_rolling import RidgeProductionProxyRollingForecaster


class RollingProdProxyLike(Protocol):
    seasonality_window: int
    data_dir: str
    name: str
    ridge: Any
    scaler: Any
    seasonal_norm: pd.Series | None
    _features: list[str]

    def fit(self, df: pd.DataFrame, target_col: str = ...) -> None: ...
    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, Any]: ...
    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = ...,
        target_col: str = ...,
    ) -> pd.DataFrame: ...
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series: ...


def build_model(**kwargs: object) -> RollingProdProxyLike:
    return cast(RollingProdProxyLike, RidgeProductionProxyRollingForecaster(**kwargs))


@pytest.fixture
def sample_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=72, freq="MS")
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + rng.normal(0, 0.25, 72),
            "Продовольственные товары": 100.6 + rng.normal(0, 0.35, 72),
            "Непродовольственные товары": 100.3 + rng.normal(0, 0.2, 72),
            "Услуги": 100.4 + rng.normal(0, 0.3, 72),
            "Ki": 12 + rng.normal(0, 0.4, 72),
            "Ruonia": 11 + rng.normal(0, 0.35, 72),
        },
        index=dates,
    )
    return data


@pytest.fixture
def temp_infostat_dir(tmp_path: Path) -> str:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    dates = pd.date_range("2019-01-01", periods=96, freq="MS")
    infostat = pd.DataFrame(
        {
            "Date": dates.strftime("%d.%m.%Y"),
            "Torg": np.linspace(100, 120, len(dates)),
            "pp": np.linspace(95, 110, len(dates)),
        }
    )
    infostat.to_csv(raw_dir / "infostat.csv", sep=";", decimal=",", encoding="utf-8-sig", index=False)
    return str(raw_dir)


def test_model_import_and_registry() -> None:
    from sirena.models import ModelRegistry, RidgeProductionProxyRollingForecaster

    model = RidgeProductionProxyRollingForecaster()
    assert isinstance(model, RidgeProductionProxyRollingForecaster)
    assert model is not None
    registry_model = ModelRegistry.get("ridge_production_proxy_rolling_24m")
    assert isinstance(registry_model, RidgeProductionProxyRollingForecaster)


def test_model_parameters(temp_infostat_dir: str) -> None:
    model = cast(
        RollingProdProxyLike,
        build_model(
            data_dir=temp_infostat_dir,
            seasonality_window=24,
        ),
    )
    assert model.seasonality_window == 24
    assert model.data_dir == temp_infostat_dir
    assert model.name == "ridge_production_proxy_rolling_24m"


def test_fit_predict_backtest(sample_data: pd.DataFrame, temp_infostat_dir: str) -> None:
    from sirena.macro_features import PRODUCTION_FEATURES

    model = cast(
        RollingProdProxyLike,
        build_model(
            use_macro=True,
            use_2022_dummy=False,
            data_dir=temp_infostat_dir,
            seasonality_window=24,
        ),
    )
    model.fit(sample_data, "Все товары и услуги")

    assert model.ridge is not None
    assert model.scaler is not None
    assert model.seasonal_norm is not None
    assert any(feature in model._features for feature in PRODUCTION_FEATURES)

    target_date = sample_data.index.to_list()[-1]
    prediction = model.predict(sample_data, target_date)
    assert "prediction" in prediction
    assert "production_features" in prediction
    assert prediction["seasonality_window"] == 24

    results = model.backtest(sample_data, start_date="2024-01-01")
    assert isinstance(results, pd.DataFrame)
    if not results.empty:
        assert "prediction" in results.columns
        assert "seasonality_window" in results.columns
        assert (results["seasonality_window"] == 24).all()


def test_rolling_seasonal_norm_uses_recent_window(temp_infostat_dir: str) -> None:
    dates = pd.date_range("2020-01-01", periods=72, freq="MS")
    values = np.full(72, 100.0)
    idx = pd.Series(dates, index=dates)
    months = idx.dt.month.to_numpy()
    years = idx.dt.year.to_numpy()
    jan_mask = months == 1
    values[jan_mask & (years <= 2022)] = 110.0
    values[jan_mask & (years >= 2023)] = 130.0
    df = pd.DataFrame(
        {
            "Все товары и услуги": values,
            "Продовольственные товары": values,
            "Непродовольственные товары": values,
            "Услуги": values,
            "Ki": np.full(72, 12.0),
            "Ruonia": np.full(72, 11.0),
        },
        index=dates,
    )

    model = cast(
        RollingProdProxyLike,
        build_model(
            data_dir=temp_infostat_dir,
            seasonality_window=24,
        ),
    )
    prepared = model._prepare_features(df)
    seasonal_norm = model._compute_seasonal_norm(prepared)
    january_norm = float(seasonal_norm.loc[1])

    assert january_norm > 120.0
    assert january_norm < 131.0
