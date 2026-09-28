import numpy as np
import pandas as pd
import pytest

from sirena.models import weekly_news_bridge as bridge


def _weekly(months=8):
    dates = pd.date_range("2024-01-01", periods=months, freq="MS")
    rows = []
    for month_pos, month in enumerate(dates):
        for week in range(4):
            date = month + pd.Timedelta(days=week * 7)
            for component_pos, component in enumerate(bridge.COMPONENTS):
                rows.append({"date": date, "accounting_month": month, "component": component,
                             "product_name": f"item-{component}-{week}",
                             "change_index": 100 + month_pos + week + component_pos})
        rows.append({"date": month, "accounting_month": month, "component": "nonfood",
                     "product_name": "Бензин автомобильный", "change_index": 10000.0})
    return pd.DataFrame(rows)


def test_prefix_uses_equal_week_items_and_excludes_gasoline_aggregate():
    weekly = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "accounting_month": pd.Timestamp("2024-01-01"),
         "component": component, "product_name": name, "change_index": value}
        for component, name, value in [
            ("food", "a", 102.0), ("food", "b", 104.0),
            ("nonfood", "Бензин автомобильный", 10000.0), ("nonfood", "Бензин АИ-92", 110.0),
            ("services", "c", 99.0),
        ]
    ])
    prefix, _ = bridge.observed_prefix(weekly, "2024-01", 1)
    assert prefix["food"] == pytest.approx(3.0)
    assert prefix["nonfood"] == pytest.approx(10.0)
    assert prefix["services"] == pytest.approx(-1.0)


def test_actual_stage_cutoff_ignores_later_weeks_and_future_months():
    weekly = _weekly()
    expected = [pd.Timestamp(f"2024-{m:02d}-01") for m in range(1, 8)]
    original, cutoff = bridge.build_stage_surprises(weekly, expected, "2024-08", 2,
                                                     "2024-08-08")
    altered = weekly.copy()
    mask = altered.date.gt(cutoff)
    altered.loc[mask, "change_index"] = 900.0
    changed, changed_cutoff = bridge.build_stage_surprises(altered, expected, "2024-08", 2,
                                                            "2024-08-08")
    assert changed_cutoff == cutoff
    np.testing.assert_allclose(changed, original)


def test_calendar_features_encode_stage_and_month():
    one = bridge.calendar_stage_features("2024-01", 1)
    four = bridge.calendar_stage_features("2024-01", 4)
    assert one.shape == (5,)
    assert one[0] == pytest.approx(0.0)
    assert one[1] == pytest.approx(1.0)
    assert one[2:5].tolist() == [1.0, 0.0, 0.0]
    assert four[2:5].tolist() == [0.0, 0.0, 0.0]


def test_missing_component_week_and_invalid_stage_fail_visibly():
    weekly = _weekly()
    missing = weekly.loc[~((weekly.accounting_month == pd.Timestamp("2024-08-01")) &
                           (weekly.date == pd.Timestamp("2024-08-08")) &
                           (weekly.component == "services"))]
    with pytest.raises(bridge.WeeklyBridgeDataError, match="missing services"):
        bridge.observed_prefix(missing, "2024-08", 2)
    with pytest.raises(bridge.WeeklyBridgeDataError, match="invalid stage"):
        bridge.observed_prefix(weekly, "2024-08", 5)


def test_expected_prefix_uses_only_strictly_prior_months():
    weekly = _weekly()
    completed = [pd.Timestamp(f"2024-{m:02d}-01") for m in range(1, 8)]
    baseline, _ = bridge.build_stage_surprises(weekly, completed, "2024-08", 1)
    future_changed = weekly.copy()
    future_changed.loc[future_changed.accounting_month.eq(pd.Timestamp("2024-08-01")), "change_index"] += 1000
    after, _ = bridge.build_stage_surprises(future_changed, completed, "2024-08", 1)
    assert after[0] - baseline[0] == pytest.approx(1000.0)
    # Adding later completed months cannot change the expectation for August.
    later = _weekly(months=10)
    later_result, _ = bridge.build_stage_surprises(later, completed + [pd.Timestamp("2024-09-01")], "2024-08", 1)
    np.testing.assert_allclose(later_result, baseline)


def test_adapter_requires_eighteen_distinct_months_and_resets_after_failed_refit():
    rows = pd.DataFrame({
        "target_month": pd.date_range("2023-01-01", periods=18, freq="MS").repeat(4),
        "stage": list(bridge.STAGES) * 18,
        "base_residual": np.linspace(-0.3, 0.3, 72),
        "surprise_food": np.linspace(-1, 1, 72),
        "surprise_nonfood": np.linspace(1, -1, 72),
        "surprise_services": np.zeros(72),
    })
    model = bridge.PooledWeeklyNewsBridge().fit(rows)
    full, calendar = model.predict("2024-07", 2, [0.1, 0.2, -0.1])
    assert np.isfinite([full, calendar]).all()
    with pytest.raises(ValueError, match="18 distinct"):
        model.fit(rows.iloc[:16])
    with pytest.raises(ValueError, match="not fitted"):
        model.predict("2024-07", 2, [0.1, 0.2, -0.1])


def test_runner_ridge_origin_ignores_future_monthly_facts(monkeypatch):
    from scripts import run_weekly_news_experiment as runner

    seen = []

    class FakeRidge:
        def __init__(self, use_macro=False):
            pass

        def fit(self, train, target):
            seen.append(("fit", train.index.max(), float(train[target].iloc[-1])))

        def predict(self, frame, target):
            seen.append(("predict", frame.index.max(), float(frame.loc[target, TARGET_COLUMN])))
            return {"prediction": 100.25}

    TARGET_COLUMN = runner.TARGET_COLUMN
    monkeypatch.setattr(runner, "RidgeForecaster", FakeRidge)
    months = pd.date_range("2024-01-01", periods=6, freq="MS")
    monthly = pd.DataFrame({TARGET_COLUMN: 100 + np.arange(6, dtype=float)}, index=months)
    first = runner.ridge_forecast(monthly, months[4], {})
    monthly.loc[months[5], TARGET_COLUMN] = 9999.0
    second = runner.ridge_forecast(monthly, months[4], {})
    assert first == second == pytest.approx(0.25)
    assert seen[0][1] == months[3]
    assert seen[1][1] == months[4]
    assert np.isnan(seen[1][2])
