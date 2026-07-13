from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import pytest

from sirena.models import ModelRegistry
from sirena.models.huber_production_proxy import HuberProductionProxyForecaster


class HuberProdProxyLike(Protocol):
    epsilon: float
    alpha: float
    max_iter: int
    use_macro: bool
    data_dir: str
    model: Any
    scaler: Any
    seasonal_norm: Any
    _features: list[str] | None

    def fit(self, df: pd.DataFrame, target_col: str = ...) -> Any: ...
    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, Any]: ...
    def backtest(self, df: pd.DataFrame, start_date: str = ..., target_col: str = ...) -> pd.DataFrame: ...


def build_model(**kwargs: object) -> Any:
    return HuberProductionProxyForecaster(**kwargs)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(72) * 0.25,
            "Продовольственные товары": 100.7 + np.random.randn(72) * 0.30,
            "Непродовольственные товары": 100.3 + np.random.randn(72) * 0.20,
            "Услуги": 100.4 + np.random.randn(72) * 0.25,
            "Ki": 8.5 + np.random.randn(72) * 0.3,
            "Ruonia": 7.9 + np.random.randn(72) * 0.25,
        },
        index=dates,
    )


@pytest.fixture
def temp_data_dir(tmp_path: Path, sample_data: pd.DataFrame) -> str:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    infostat_dates = pd.date_range(sample_data.index.min(), periods=96, freq="MS")
    infostat_df = pd.DataFrame(
        {
            "Date": infostat_dates.strftime("%d.%m.%Y"),
            "Torg": np.linspace(95.0, 115.0, len(infostat_dates)),
            "pp": np.linspace(97.0, 112.0, len(infostat_dates)),
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
    model = build_model()
    assert isinstance(model, HuberProductionProxyForecaster)

    registry_model = ModelRegistry.get("huber_production_proxy")
    assert isinstance(registry_model, HuberProductionProxyForecaster)


def test_model_parameters(temp_data_dir: str) -> None:
    model = cast(
        HuberProdProxyLike,
        build_model(epsilon=1.5, alpha=0.4, use_macro=True, data_dir=temp_data_dir),
    )
    assert model.epsilon == 1.5
    assert model.alpha == 0.4
    assert model.use_macro is True
    assert model.data_dir == temp_data_dir


def test_fit_predict_backtest(sample_data: pd.DataFrame, temp_data_dir: str) -> None:
    model = cast(
        HuberProdProxyLike,
        build_model(data_dir=temp_data_dir, use_macro=True),
    )

    model.fit(sample_data, "Все товары и услуги")

    assert model.model is not None
    assert model.scaler is not None
    assert model.seasonal_norm is not None
    assert model._features is not None
    assert "torg_lag3" in model._features
    assert "pp_lag3" in model._features

    target_date = cast(pd.Timestamp, sample_data.index.to_list()[-1])
    result = model.predict(sample_data, target_date)
    assert "prediction" in result
    assert "pred_huber" in result
    assert "production_features" in result
    assert "torg_lag3" in result["production_features"]

    backtest_results = model.backtest(sample_data, start_date="2022-01-01")
    assert isinstance(backtest_results, pd.DataFrame)
    if not backtest_results.empty:
        assert "prediction" in backtest_results.columns
        assert "production_features_count" in backtest_results.columns


def test_production_features_present(sample_data: pd.DataFrame, temp_data_dir: str) -> None:
    model = cast(
        HuberProdProxyLike,
        build_model(data_dir=temp_data_dir, use_macro=True),
    )
    model.fit(sample_data, "Все товары и услуги")
    assert model._features is not None
    assert any(feature.startswith("torg_") for feature in model._features)
    assert any(feature.startswith("pp_") for feature in model._features)
