from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pandas as pd
import pytest

from sirena.models import ModelRegistry
from sirena.models.ridge_extended_production_proxy import (
    RidgeExtendedProductionProxyForecaster,
)

if TYPE_CHECKING:
    from sirena.models.ridge_extended_production_proxy import (
        RidgeExtendedProductionProxyForecaster as RidgeExtendedProductionProxyType,
    )


class RidgeExtendedProdProxyLike(Protocol):
    alpha: float
    use_macro: bool
    data_dir: str
    name: str
    ridge: Any
    scaler: Any
    seasonal_norm: Any
    _features: list[str] | None

    def fit(self, df: pd.DataFrame, target_col: str = ...) -> Any: ...
    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, Any]: ...
    def backtest(self, df: pd.DataFrame, start_date: str = ..., target_col: str = ...) -> pd.DataFrame: ...


def build_model(**kwargs: object) -> Any:
    return RidgeExtendedProductionProxyForecaster(**kwargs)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=96, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(96) * 0.3,
            "Продовольственные товары": 100.6 + np.random.randn(96) * 0.4,
            "Непродовольственные товары": 100.3 + np.random.randn(96) * 0.2,
            "Услуги": 100.4 + np.random.randn(96) * 0.3,
            "Ki": 12 + np.random.randn(96) * 0.4,
            "Ruonia": 11 + np.random.randn(96) * 0.3,
        },
        index=dates,
    )
    return data


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> str:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    infostat_dates = pd.date_range("2017-01-01", periods=96, freq="MS")
    infostat_df = pd.DataFrame(
        {
            "Date": infostat_dates.strftime("%d.%m.%Y"),
            "Torg": np.linspace(100, 130, len(infostat_dates)),
            "pp": np.linspace(95, 120, len(infostat_dates)),
        }
    )
    infostat_df.to_csv(
        raw_dir / "infostat.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )
    return str(raw_dir)


def test_model_import_and_registry() -> None:
    registry_model = ModelRegistry.get("ridge_extended_production_proxy")
    assert isinstance(registry_model, RidgeExtendedProductionProxyForecaster)


def test_model_parameters(temp_data_dir: str) -> None:
    model = cast(
        RidgeExtendedProdProxyLike,
        build_model(data_dir=temp_data_dir, use_macro=True),
    )
    assert model.data_dir == temp_data_dir
    assert model.use_macro is True
    assert model.name == "ridge_extended_production_proxy"


def test_fit_predict_backtest(sample_data: pd.DataFrame, temp_data_dir: str) -> None:
    model = cast(
        RidgeExtendedProdProxyLike,
        build_model(data_dir=temp_data_dir, use_macro=True),
    )
    model.fit(sample_data, "Все товары и услуги")

    assert model.ridge is not None
    assert model.scaler is not None
    assert model.seasonal_norm is not None
    assert model._features is not None
    assert "torg_lag3" in model._features
    assert "pp_lag3" in model._features

    target_date = cast(pd.Timestamp, sample_data.index.to_list()[-1])
    result = model.predict(sample_data, target_date)
    assert "prediction" in result
    assert "pred_ridge" in result
    assert "pred_ets" in result
    assert "production_features" in result
    assert "torg_lag3" in result["production_features"]

    bt = model.backtest(sample_data, start_date="2024-01-01")
    assert isinstance(bt, pd.DataFrame)
    if not bt.empty:
        assert "prediction" in bt.columns
        assert "production_features_count" in bt.columns


def test_production_features_present(sample_data: pd.DataFrame, temp_data_dir: str) -> None:
    model = cast(
        RidgeExtendedProdProxyLike,
        build_model(data_dir=temp_data_dir, use_macro=True),
    )
    model.fit(sample_data, "Все товары и услуги")

    assert model._features is not None
    for feature in ["torg_lag3", "torg_lag6", "pp_lag3", "pp_diff_lag3"]:
        assert feature in model._features
