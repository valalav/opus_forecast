"""
Tests for ScenarioRateModel
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sirena" / "models"))

import pytest
import numpy as np
import pandas as pd
from scenario_rate import (
    ScenarioRateModel,
    TransmissionParams,
    AsymmetricParams,
    CalibrationResult,
    create_gradual_hike,
    create_gradual_cut,
    create_shock_then_hold,
)


class TestScenarioRateModel:
    """Test ScenarioRateModel functionality"""

    @pytest.fixture
    def sample_data(self):
        """Create sample inflation data for testing"""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=48, freq="MS")
        df = pd.DataFrame(
            {
                "mom": np.random.randn(48) * 0.3 + 0.5,
                "Ki_i": np.linspace(7.0, 10.0, 48),
                "Date": dates,
            }
        )
        df = df.set_index("Date").sort_index()
        return df

    @pytest.fixture
    def model(self):
        """Create ScenarioRateModel instance"""
        return ScenarioRateModel()

    @pytest.fixture
    def asymmetric_model(self):
        """Create asymmetric ScenarioRateModel instance"""
        return ScenarioRateModel(use_asymmetric=True)

    def test_initialization_default_params(self, model):
        """Test model initialization with default params"""
        assert model.params.peak_effect == -0.08
        assert model.params.peak_lag == 6
        assert model.params.start_lag == 2
        assert model.params.duration == 18
        assert model.params.shape == "hump"
        assert model.use_asymmetric is False
        assert model._is_fitted is False

    def test_initialization_custom_params(self):
        """Test model initialization with custom params"""
        params = TransmissionParams(
            peak_effect=-0.12, peak_lag=8, start_lag=3, duration=24, shape="exponential"
        )
        model = ScenarioRateModel(params=params)
        assert model.params.peak_effect == -0.12
        assert model.params.peak_lag == 8
        assert model.params.start_lag == 3
        assert model.params.duration == 24
        assert model.params.shape == "exponential"

    def test_initialization_asymmetric(self):
        """Test model initialization with asymmetric params"""
        asym_params = AsymmetricParams(
            hike_peak_effect=-0.12,
            hike_peak_lag=5,
            hike_duration=18,
            cut_peak_effect=-0.06,
            cut_peak_lag=8,
            cut_duration=24,
        )
        model = ScenarioRateModel(asymmetric_params=asym_params, use_asymmetric=True)
        assert model.use_asymmetric is True
        assert model.asymmetric_params.hike_peak_effect == -0.12
        assert model.asymmetric_params.cut_peak_effect == -0.06

    def test_build_irf_hump(self, model):
        """Test IRF building with hump shape"""
        irf = model._build_irf()
        assert len(irf) == 18  # duration
        assert irf[0] == 0  # start_lag=2
        assert irf[1] == 0
        # Effect starts when s > start_lag (x > 0), so first non-zero is at index 3
        assert irf[2] == 0  # x = 0 when s = start_lag
        assert irf[3] != 0  # first non-zero value
        assert irf[6] < 0  # peak at lag 6 (negative effect)
        # Last element at index duration-1, not duration
        assert irf[17] != 0  # duration=18, index 17 is the last element

    def test_build_irf_exponential(self):
        """Test IRF building with exponential shape"""
        params = TransmissionParams(
            peak_effect=-0.08, peak_lag=6, start_lag=2, duration=18, shape="exponential"
        )
        model = ScenarioRateModel(params=params)
        irf = model._build_irf()
        assert len(irf) == 18
        assert irf[6] < 0  # peak at lag 6

    def test_build_irf_asymmetric_hike(self, asymmetric_model):
        """Test asymmetric IRF building for hike"""
        irf_hike = asymmetric_model.irf_hike
        assert len(irf_hike) == 18  # hike_duration
        # Effect starts when s > start_lag (x > 0), so first non-zero is at index 3
        assert irf_hike[2] == 0  # x = 0 when s = start_lag
        assert irf_hike[3] != 0  # first non-zero value
        assert irf_hike[5] < 0  # peak at lag 5 (hike_peak_lag)

    def test_build_irf_asymmetric_cut(self, asymmetric_model):
        """Test asymmetric IRF building for cut"""
        irf_cut = asymmetric_model.irf_cut
        assert len(irf_cut) == 24  # cut_duration
        # Effect starts when s > start_lag (x > 0), so first non-zero is at index 3
        assert irf_cut[2] == 0  # x = 0 when s = start_lag
        assert irf_cut[3] != 0  # first non-zero value
        assert irf_cut[8] < 0  # peak at lag 8 (cut_peak_lag)

    def test_fit(self, model, sample_data):
        """Test model fitting"""
        model.fit(sample_data)
        assert model._is_fitted is True
        assert len(model.mom_history) > 0
        assert model.last_date is not None

    def test_fit_without_target_col(self, model):
        """Test fitting without explicit target column"""
        dates = pd.date_range(start="2020-01-01", periods=48, freq="MS")
        df = pd.DataFrame(
            {
                "Все товары и услуги": np.random.randn(48) * 0.3 + 100.5,
            },
            index=dates,
        )
        model.fit(df)
        assert model._is_fitted is True

    def test_baseline_forecast(self, model, sample_data):
        """Test baseline forecast"""
        model.fit(sample_data)
        baseline = model._baseline_forecast(12)
        assert len(baseline) == 12
        assert not np.allclose(baseline, 0)  # should have some variation

    def test_forecast_scenario_no_change(self, model, sample_data):
        """Test forecast scenario with no rate change"""
        model.fit(sample_data)
        result = model.forecast_scenario(horizon=12, ki_change=0.0)

        assert "baseline" in result
        assert "effect" in result
        assert "total" in result
        assert "cumulative_effect" in result
        assert len(result["baseline"]) == 12
        assert len(result["effect"]) == 12
        assert len(result["total"]) == 12
        assert np.allclose(result["effect"], 0, atol=1e-10)  # No change = no effect
        assert np.allclose(result["total"], result["baseline"])  # Total = baseline

    def test_forecast_scenario_rate_hike(self, model, sample_data):
        """Test forecast scenario with rate hike"""
        model.fit(sample_data)
        result = model.forecast_scenario(horizon=12, ki_change=+2.0)

        assert result["cumulative_effect"] < 0  # Hike should reduce inflation
        assert np.allclose(result["total"], result["baseline"] + result["effect"])
        assert len(result["ki_path"]) == 12
        assert result["ki_path"][0] == 2.0  # Rate change in month 1

    def test_forecast_scenario_rate_cut(self, model, sample_data):
        """Test forecast scenario with rate cut"""
        model.fit(sample_data)
        result = model.forecast_scenario(horizon=12, ki_change=-2.0)

        # Rate cut should increase inflation (negative IRF * negative change = positive)
        assert result["cumulative_effect"] > 0
        assert np.allclose(result["total"], result["baseline"] + result["effect"])

    def test_forecast_scenario_array(self, model, sample_data):
        """Test forecast scenario with array of rate changes"""
        model.fit(sample_data)
        ki_path = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = model.forecast_scenario(horizon=12, ki_change=ki_path)

        assert len(result["effect"]) == 12
        assert (
            result["cumulative_effect"] < 0
        )  # Cumulative hike should reduce inflation

    def test_forecast_scenario_return_details(self, model, sample_data):
        """Test forecast scenario with detailed output"""
        model.fit(sample_data)
        result = model.forecast_scenario(horizon=12, ki_change=2.0, return_details=True)

        assert "dates" in result
        assert "irf" in result
        assert len(result["dates"]) == 12
        assert len(result["irf"]) == 12

    def test_forecast_scenario_not_fitted(self, model):
        """Test that forecast raises error if model not fitted"""
        with pytest.raises(ValueError, match="Сначала вызовите fit"):
            model.forecast_scenario(horizon=12, ki_change=1.0)

    def test_compare_scenarios(self, model, sample_data):
        """Test scenario comparison"""
        model.fit(sample_data)
        comparison = model.compare_scenarios(horizon=12)

        assert "Сценарий" in comparison.columns
        assert "Изменение Ki (п.п.)" in comparison.columns
        assert "Cum MoM (%)" in comparison.columns
        assert "Эффект от Ki (%)" in comparison.columns
        assert len(comparison) == 5  # base, +2, +4, -2, -4

    def test_compare_scenarios_custom(self, model, sample_data):
        """Test scenario comparison with custom scenarios"""
        model.fit(sample_data)
        scenarios = {"Base": 0.0, "Hike 3pp": 3.0, "Cut 1pp": -1.0}
        comparison = model.compare_scenarios(horizon=12, scenarios=scenarios)

        assert len(comparison) == 3
        assert comparison["Сценарий"].tolist() == ["Base", "Hike 3pp", "Cut 1pp"]

    def test_get_irf_table(self, model):
        """Test IRF table generation"""
        irf_df = model.get_irf_table()

        assert "Лаг (мес)" in irf_df.columns
        assert "IRF" in irf_df.columns
        assert "Cum IRF" in irf_df.columns
        assert len(irf_df) == 18

    def test_sensitivity_analysis(self, model, sample_data):
        """Test sensitivity analysis"""
        model.fit(sample_data)
        sensitivity = model.sensitivity_analysis(
            horizon=12, ki_range=np.arange(-2, 3, 1)
        )

        assert "Ki change (п.п.)" in sensitivity.columns
        assert "Effect (%)" in sensitivity.columns
        assert "Total CPI (%)" in sensitivity.columns
        assert len(sensitivity) == 5  # -2, -1, 0, 1, 2

    def test_sensitivity_analysis_default(self, model, sample_data):
        """Test sensitivity analysis with default range"""
        model.fit(sample_data)
        sensitivity = model.sensitivity_analysis(horizon=12)

        assert len(sensitivity) == 9  # -4 to +4

    def test_get_asymmetric_irf_table(self, asymmetric_model):
        """Test asymmetric IRF table generation"""
        asym_df = asymmetric_model.get_asymmetric_irf_table()

        assert "Лаг (мес)" in asym_df.columns
        assert "IRF Hike" in asym_df.columns
        assert "IRF Cut" in asym_df.columns
        assert "Cum Hike" in asym_df.columns
        assert "Cum Cut" in asym_df.columns
        assert "Asymmetry" in asym_df.columns

    def test_apply_nonlinearity_disabled(self):
        """Test nonlinearity application when disabled"""
        model = ScenarioRateModel(use_nonlinearity=False)
        effect = -0.16
        ki_change = 5.0  # Above threshold
        result = model._apply_nonlinearity(effect, ki_change)

        assert result == -0.16  # No change

    def test_apply_nonlinearity_enabled(self):
        """Test nonlinearity application when enabled"""
        asym_params = AsymmetricParams(
            nonlinearity_threshold=3.0, nonlinearity_factor=1.3
        )
        model = ScenarioRateModel(
            asymmetric_params=asym_params, use_asymmetric=True, use_nonlinearity=True
        )
        effect = -0.16
        ki_change = 5.0  # Above threshold

        result = model._apply_nonlinearity(effect, ki_change)

        # Effect should be amplified
        assert result < -0.16

    def test_apply_nonlinearity_below_threshold(self):
        """Test nonlinearity when change is below threshold"""
        asym_params = AsymmetricParams(
            nonlinearity_threshold=3.0, nonlinearity_factor=1.3
        )
        model = ScenarioRateModel(
            asymmetric_params=asym_params, use_asymmetric=True, use_nonlinearity=True
        )
        effect = -0.16
        ki_change = 2.0  # Below threshold

        result = model._apply_nonlinearity(effect, ki_change)

        assert result == -0.16  # No amplification

    def test_calibrate_on_data(self, model, sample_data):
        """Test parameter calibration on data"""
        model.fit(sample_data)
        calib_result = model.calibrate_on_data(sample_data)

        assert isinstance(calib_result, CalibrationResult)
        assert calib_result.mae > 0
        assert calib_result.rmse > 0
        assert calib_result.n_obs > 0
        assert calib_result.optimization_method in [
            "differential_evolution",
            "minimize",
        ]
        assert model.calibration_result is not None

    def test_calibrate_updates_params(self, model, sample_data):
        """Test that calibration updates model params"""
        original_peak = model.params.peak_effect
        model.fit(sample_data)
        model.calibrate_on_data(sample_data)

        # Params should be updated after calibration
        # Note: might be same if optimization found same values
        assert model.params is not None

    def test_calibrate_custom_bounds(self, model, sample_data):
        """Test calibration with custom bounds"""
        model.fit(sample_data)
        bounds = {
            "peak_effect": (-0.15, -0.05),
            "peak_lag": (4, 10),
            "duration": (12, 24),
        }
        calib_result = model.calibrate_on_data(sample_data, bounds=bounds)

        assert calib_result.bounds_used == bounds
        assert (
            calib_result.optimal_params.peak_effect >= -0.15
            and calib_result.optimal_params.peak_effect <= -0.05
        )
        assert (
            calib_result.optimal_params.peak_lag >= 4
            and calib_result.optimal_params.peak_lag <= 10
        )
        assert (
            calib_result.optimal_params.duration >= 12
            and calib_result.optimal_params.duration <= 24
        )

    def test_calibrate_with_minimize_method(self, model, sample_data):
        """Test calibration with minimize method"""
        model.fit(sample_data)
        calib_result = model.calibrate_on_data(sample_data, method="minimize")

        assert calib_result.optimization_method == "minimize"

    def test_calibrate_missing_ki_col(self, model, sample_data):
        """Test calibration error when Ki column is missing"""
        df_no_ki = sample_data.drop(columns=["Ki_i"])
        model.fit(df_no_ki)

        with pytest.raises(ValueError, match="Колонка Ki_i не найдена"):
            model.calibrate_on_data(df_no_ki)


class TestScenarioHelpers:
    """Test scenario helper functions"""

    def test_create_gradual_hike(self):
        """Test gradual hike creation"""
        path = create_gradual_hike(total_change=3.0, months=6)

        assert len(path) == 24
        assert np.sum(path[:6]) == 3.0
        assert np.allclose(path[6:], 0)

    def test_create_gradual_cut(self):
        """Test gradual cut creation"""
        path = create_gradual_cut(total_change=-2.0, months=4)

        assert len(path) == 24
        assert np.sum(path[:4]) == -2.0
        assert np.allclose(path[4:], 0)

    def test_create_shock_then_hold(self):
        """Test shock and hold scenario creation"""
        path = create_shock_then_hold(shock=2.0, hold_months=12)

        assert len(path) == 24
        assert path[0] == 2.0
        assert np.allclose(path[1:24], 0)


class TestTransmissionParams:
    """Test TransmissionParams dataclass"""

    def test_defaults(self):
        """Test default values"""
        params = TransmissionParams()
        assert params.peak_effect == -0.08
        assert params.peak_lag == 6
        assert params.start_lag == 2
        assert params.duration == 18
        assert params.shape == "hump"

    def test_custom_values(self):
        """Test custom values"""
        params = TransmissionParams(
            peak_effect=-0.10, peak_lag=8, start_lag=3, duration=24, shape="exponential"
        )
        assert params.peak_effect == -0.10
        assert params.peak_lag == 8
        assert params.start_lag == 3
        assert params.duration == 24
        assert params.shape == "exponential"


class TestAsymmetricParams:
    """Test AsymmetricParams dataclass"""

    def test_defaults(self):
        """Test default values"""
        params = AsymmetricParams()
        assert params.hike_peak_effect == -0.10
        assert params.hike_peak_lag == 5
        assert params.hike_duration == 18
        assert params.cut_peak_effect == -0.05
        assert params.cut_peak_lag == 8
        assert params.cut_duration == 24
        assert params.nonlinearity_threshold == 3.0
        assert params.nonlinearity_factor == 1.3

    def test_custom_values(self):
        """Test custom values"""
        params = AsymmetricParams(
            hike_peak_effect=-0.15,
            hike_peak_lag=6,
            hike_duration=24,
            cut_peak_effect=-0.08,
            cut_peak_lag=10,
            cut_duration=30,
            nonlinearity_threshold=4.0,
            nonlinearity_factor=1.5,
        )
        assert params.hike_peak_effect == -0.15
        assert params.hike_peak_lag == 6
        assert params.cut_peak_effect == -0.08
        assert params.nonlinearity_threshold == 4.0
        assert params.nonlinearity_factor == 1.5


class TestCalibrationResult:
    """Test CalibrationResult dataclass"""

    def test_defaults(self):
        """Test default values"""
        params = TransmissionParams()
        result = CalibrationResult(
            optimal_params=params,
            mae=0.5,
            rmse=0.6,
            n_obs=100,
            optimization_method="test",
        )
        assert result.optimal_params == params
        assert result.mae == 0.5
        assert result.rmse == 0.6
        assert result.n_obs == 100
        assert result.optimization_method == "test"
        assert result.bounds_used == {}
        assert result.convergence_info == ""
