import numpy as np
import pandas as pd
import pytest

from sirena.data_loader import DataFreshnessError
from sirena.models.pooled_components import (
    P3IndependentRidgeForecaster,
    P3PartialPoolingForecaster,
    P3PoolingLagTrendForecaster,
    PooledComponentsForecaster,
)


def make_inputs(root, *, end="2024-12-01", future_weight_shift=0.0, missing=None):
    root.mkdir(parents=True, exist_ok=True)
    codes = list(range(11, 56))
    dates = pd.date_range("2016-01-01", end, freq="MS")
    structure = pd.DataFrame({
        "Item_code": codes,
        "Item_type": 3,
        "Superior": [3 if i < 15 else 2 if i < 35 else 4 for i in range(45)],
        "Primary": 0,
        "Component": [3 if i < 15 else 2 if i < 35 else 4 for i in range(45)],
        "Subcomponent": codes,
        "Item_on": 1,
    })
    structure.to_csv(root / "items_structure.csv", index=False)
    pd.DataFrame({"Item_code": codes, "Item_name": [f"group {c}" for c in codes]}).to_csv(root / "items_names.csv", index=False)
    rows = []
    for j, code in enumerate(codes):
        for t, date in enumerate(dates):
            if missing == (code, date):
                continue
            value = 100.0 + 0.15 * np.sin(t / 3.0 + j / 7.0) + 0.02 * np.cos(t / 5.0)
            rows.append({"Region_code": 7, "Day": date.strftime("%m/%d/%y 00:00:00"), "Item_code": code, "MoM": value})
    pd.DataFrame(rows).to_csv(root / "kbr_indices.csv", index=False)
    vintage = pd.Timestamp("2023-01-01")
    weight_rows = []
    for date, shift in [(vintage, 0.0), (pd.Timestamp("2025-01-01"), future_weight_shift)]:
        weights = np.full(45, 1.0 / 45.0)
        weights[0] += shift
        weights[1] -= shift
        for code, weight in zip(codes, weights):
            weight_rows.append({"Region_code": 7, "Day": date.strftime("%m/%d/%y 00:00:00"), "Item_code": code, "Weight_vertical": weight})
    pd.DataFrame(weight_rows).to_csv(root / "access_weights.csv", index=False)
    return pd.date_range("2016-01-01", "2024-12-01", freq="MS"), codes


def input_frame(dates):
    frame = pd.DataFrame({"Все товары и услуги": np.linspace(0.1, 0.3, len(dates))}, index=dates)
    frame.attrs["input_contract"] = {"representation": "raw"}
    return frame


def test_future_observations_and_weights_do_not_change_origin_forecast(tmp_path):
    dates, _ = make_inputs(tmp_path / "base")
    future_dates, _ = make_inputs(tmp_path / "mutated", end="2025-06-01", future_weight_shift=0.25)
    cutoff = pd.Timestamp("2024-12-01")
    baseline = PooledComponentsForecaster("partial_pooling", data_dir=tmp_path / "base").fit(input_frame(dates))
    mutated = PooledComponentsForecaster("partial_pooling", data_dir=tmp_path / "mutated").fit(input_frame(dates))
    np.testing.assert_allclose(baseline.forecast(2), mutated.forecast(2), atol=1e-10)
    assert baseline.cutoff == mutated.cutoff == cutoff
    assert baseline.basket["weight_vintage"] == mutated.basket["weight_vintage"]


def test_all_three_variants_use_same_target_basket_and_full_weight(tmp_path):
    dates, codes = make_inputs(tmp_path / "data")
    forecasts = []
    models = [
        P3IndependentRidgeForecaster(data_dir=tmp_path / "data"),
        P3PartialPoolingForecaster(data_dir=tmp_path / "data"),
        P3PoolingLagTrendForecaster(data_dir=tmp_path / "data"),
    ]
    for model in models:
        model.fit(input_frame(dates))
        groups = model.forecast_groups(2)
        assert list(groups.columns) == codes
        assert len(groups) == 2
        forecasts.append(groups)
        total = model.forecast(2)
        assert np.isfinite(total).all()
        assert model.diagnostics["fit_predict_max_abs_diff"] < 1e-10
        assert np.allclose(model.last_aggregation.weights.sum(axis=1), 1.0)
        assert model.diagnostics["n_panel_months"] >= 24
    for groups in forecasts[1:]:
        assert groups.index.equals(forecasts[0].index)
        assert groups.columns.equals(forecasts[0].columns)
    assert [m.group_family[codes[0]] for m in models] == ["food"] * 3


def test_forecast_features_use_origin_values_as_lag1(tmp_path):
    dates, codes = make_inputs(tmp_path / "data")
    model = P3IndependentRidgeForecaster(data_dir=tmp_path / "data").fit(input_frame(dates))
    assert model.diagnostics["feature_names"][0] == "lag1"
    # Force a transparent coefficient vector: forecast equals standardized origin lag 1.
    model.coef_[:] = 0.0
    p = model._n_features
    for i in range(45):
        model.coef_[i * (p + 1) + 1] = 1.0
    observed = model.basket["history"].loc[model.cutoff, codes].to_numpy(dtype=float)
    actual = model.forecast_groups(1).iloc[0].to_numpy()
    expected = (observed - model.feature_mean_[0]) / model.feature_scale_[0]
    np.testing.assert_allclose(actual, expected)


def test_refit_resets_cutoff_and_predictions_are_deterministic(tmp_path):
    dates, _ = make_inputs(tmp_path / "data", end="2025-06-01")
    model = PooledComponentsForecaster("partial_pooling", data_dir=tmp_path / "data")
    model.fit(input_frame(dates[dates <= "2024-12-01"]))
    first = model.forecast(2)
    np.testing.assert_allclose(first, model.forecast(2))
    model.fit(input_frame(dates[dates <= "2023-12-01"]))
    assert model.cutoff == pd.Timestamp("2023-12-01")
    assert model.diagnostics["cutoff"] == "2023-12-01"


def test_missing_required_group_observation_fails_without_zero_fill(tmp_path):
    cutoff = pd.Timestamp("2024-12-01")
    missing = (11, cutoff - pd.DateOffset(months=1))
    dates, _ = make_inputs(tmp_path / "data", missing=missing)
    model = PooledComponentsForecaster("partial_pooling", data_dir=tmp_path / "data")
    with pytest.raises(DataFreshnessError):
        model.fit(input_frame(dates))
    assert not model.is_fitted
