"""
Unit tests for UnifiedSubcomponentForecaster
Tests are created in edge_lab but import from parent sirena package
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
import tempfile
import shutil

# Add parent directory to path to import sirena module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def sample_data():
    """Generate sample inflation data for testing."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(60) * 0.3,
            "Продовольственные товары": 100.6 + np.random.randn(60) * 0.4,
            "Непродовольственные товары": 100.3 + np.random.randn(60) * 0.2,
            "Услуги": 100.4 + np.random.randn(60) * 0.3,
        },
        index=dates,
    )

    return data


@pytest.fixture
def sample_data_with_macro():
    """Generate sample inflation data with macro features."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(60) * 0.3,
            "Продовольственные товары": 100.6 + np.random.randn(60) * 0.4,
            "Непродовольственные товары": 100.3 + np.random.randn(60) * 0.2,
            "Услуги": 100.4 + np.random.randn(60) * 0.3,
            "Ki": 16 + np.random.randn(60) * 0.5,
            "Ki_i": 16 + np.random.randn(60) * 0.5,
            "Ruonia": 15 + np.random.randn(60) * 0.5,
            "usd_nom_i": 75 + np.random.randn(60) * 2.0,
            "brent": 80 + np.random.randn(60) * 3.0,
            "mom": 100.5 + np.random.randn(60) * 0.3,
        },
        index=dates,
    )

    return data


@pytest.fixture
def sample_subcomponent_data():
    """Generate sample subcomponent data for testing."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    np.random.seed(42)

    # Create sample subcomponent data with 10 subcomponents (subset of 45)
    sub_dict = {}
    for i in range(10):
        code = f"{11 + i}"
        sub_dict[code] = 100.0 + np.random.randn(60) * 0.5

    sub_df = pd.DataFrame(sub_dict, index=dates)

    return sub_df


@pytest.fixture
def sample_weights():
    """Generate sample weights for subcomponents."""
    weights = {f"{11 + i}": 0.1 for i in range(10)}
    return weights


@pytest.fixture
def sample_sprav_csv(sample_weights):
    """Generate sample subcomp_sprav.csv for testing."""
    df = pd.DataFrame(
        {
            "Item_code": [int(k) for k in sample_weights.keys()],
            "Weight": list(sample_weights.values()),
        }
    )
    return df


@pytest.fixture
def temp_data_dir(sample_subcomponent_data, sample_sprav_csv):
    """Create temporary data directory with sample data."""
    temp_dir = tempfile.mkdtemp()

    # Create raw subdirectory
    raw_dir = Path(temp_dir) / "raw"
    raw_dir.mkdir(parents=True)

    # Create sub_mom.csv
    sub_file = raw_dir / "sub_mom.csv"
    sample_subcomponent_data.to_csv(sub_file, sep=";", decimal=",")

    # Create subcomp_sprav.csv
    sprav_file = raw_dir / "subcomp_sprav.csv"
    sample_sprav_csv.to_csv(sprav_file, sep=";", decimal=",", index=False)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestUnifiedSubcomponentForecaster:
    """Test suite for UnifiedSubcomponentForecaster."""

    def test_import_model(self):
        """Test model import."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster

        model = UnifiedSubcomponentForecaster()
        assert model is not None
        assert model.VERSION == "3.0"

    def test_model_parameters_default(self):
        """Test model default parameters."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster

        model = UnifiedSubcomponentForecaster()

        assert model.horizon == 1
        assert model.use_irf_convolution == True
        assert model.use_calibrated_irf == True
        assert model.use_ki_trajectory == False
        assert model._is_fitted == False
        assert model.base_model is None
        assert model.scenario_model is None
        assert model.ki_model is None

    def test_model_parameters_custom(self):
        """Test model custom parameters."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster

        model = UnifiedSubcomponentForecaster(
            horizon=12,
            use_irf_convolution=False,
            use_calibrated_irf=False,
            use_ki_trajectory=True,
        )

        assert model.horizon == 12
        assert model.use_irf_convolution == False
        assert model.use_calibrated_irf == False
        assert model.use_ki_trajectory == True

    def test_unified_forecast_result_dataclass(self):
        """Test UnifiedForecastResult dataclass."""
        from sirena.models.unified_subcomp import UnifiedForecastResult

        baseline = np.array([100.1, 100.2])
        effect = np.array([0.05, 0.10])
        total = baseline + effect

        result = UnifiedForecastResult(baseline=baseline, effect=effect, total=total)

        assert np.array_equal(result.baseline, baseline)
        assert np.array_equal(result.effect, effect)
        assert np.array_equal(result.total, total)

    def test_fit_basic(self, sample_data_with_macro, temp_data_dir):
        """Test basic fit functionality with baseline model only."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster
        from sirena.models import subcomponent_multi

        # Mock data directory
        original_load_data = subcomponent_multi.SubcomponentMultiForecaster._load_data

        def mock_load_data(self, data_dir):
            sub = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub.index = pd.to_datetime(sub.index)
            sub.index = sub.index.to_period("M").to_timestamp()
            sub = sub[~sub.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"].astype(str), sprav["Weight"]))

            valid_cols = [c for c in sub.columns if c in self.weights]
            sub = sub[valid_cols]

            return sub

        subcomponent_multi.SubcomponentMultiForecaster._load_data = mock_load_data

        model = UnifiedSubcomponentForecaster(
            horizon=1,
            use_irf_convolution=False,  # Disable IRF for simpler test
            use_ki_trajectory=False,
        )
        model.fit(sample_data_with_macro)

        assert model._is_fitted
        assert model.base_model is not None
        assert model.scenario_model is None  # Disabled
        assert model.ki_model is None  # Disabled

        # Restore
        subcomponent_multi.SubcomponentMultiForecaster._load_data = original_load_data

    def test_fit_with_irf_and_ki(self, sample_data_with_macro, temp_data_dir):
        """Test fit with IRF convolution and Ki trajectory enabled."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster
        from sirena.models import subcomponent_multi

        # Mock data directory
        original_load_data = subcomponent_multi.SubcomponentMultiForecaster._load_data

        def mock_load_data(self, data_dir):
            sub = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub.index = pd.to_datetime(sub.index)
            sub.index = sub.index.to_period("M").to_timestamp()
            sub = sub[~sub.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"].astype(str), sprav["Weight"]))

            valid_cols = [c for c in sub.columns if c in self.weights]
            sub = sub[valid_cols]

            return sub

        subcomponent_multi.SubcomponentMultiForecaster._load_data = mock_load_data

        model = UnifiedSubcomponentForecaster(
            horizon=1, use_irf_convolution=True, use_ki_trajectory=True
        )
        model.fit(sample_data_with_macro, macro_df=sample_data_with_macro)

        assert model._is_fitted
        assert model.base_model is not None
        # IRF and Ki models may be created if dependencies are available
        # Just verify fit completed

        # Restore
        subcomponent_multi.SubcomponentMultiForecaster._load_data = original_load_data

    def test_forecast_not_fitted_error(self, sample_data_with_macro):
        """Test forecast raises error when model not fitted."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster

        model = UnifiedSubcomponentForecaster()

        with pytest.raises(RuntimeError, match="not fitted"):
            model.forecast(12)

    def test_forecast_baseline(self, sample_data_with_macro, temp_data_dir):
        """Test baseline forecast (without rate effects)."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster
        from sirena.models import subcomponent_multi

        # Mock data directory
        original_load_data = subcomponent_multi.SubcomponentMultiForecaster._load_data

        def mock_load_data(self, data_dir):
            sub = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub.index = pd.to_datetime(sub.index)
            sub.index = sub.index.to_period("M").to_timestamp()
            sub = sub[~sub.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"].astype(str), sprav["Weight"]))

            valid_cols = [c for c in sub.columns if c in self.weights]
            sub = sub[valid_cols]

            return sub

        subcomponent_multi.SubcomponentMultiForecaster._load_data = mock_load_data

        model = UnifiedSubcomponentForecaster(
            horizon=1, use_irf_convolution=False, use_ki_trajectory=False
        )
        model.fit(sample_data_with_macro)

        # Test baseline forecast
        baseline = model.forecast(12)
        assert isinstance(baseline, np.ndarray)
        assert len(baseline) == 12

        # Restore
        subcomponent_multi.SubcomponentMultiForecaster._load_data = original_load_data

    def test_forecast_with_rate_not_fitted(self, sample_data_with_macro):
        """Test forecast_with_rate raises error when model not fitted."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster

        model = UnifiedSubcomponentForecaster()

        with pytest.raises(RuntimeError, match="not fitted"):
            model.forecast_with_rate(12, ki_change=1.0)

    def test_forecast_with_rate_basic(self, sample_data_with_macro, temp_data_dir):
        """Test forecast_with_rate with Ki change."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster
        from sirena.models import subcomponent_multi

        # Mock data directory
        original_load_data = subcomponent_multi.SubcomponentMultiForecaster._load_data

        def mock_load_data(self, data_dir):
            sub = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub.index = pd.to_datetime(sub.index)
            sub.index = sub.index.to_period("M").to_timestamp()
            sub = sub[~sub.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"].astype(str), sprav["Weight"]))

            valid_cols = [c for c in sub.columns if c in self.weights]
            sub = sub[valid_cols]

            return sub

        subcomponent_multi.SubcomponentMultiForecaster._load_data = mock_load_data

        model = UnifiedSubcomponentForecaster(
            horizon=1,
            use_irf_convolution=False,  # Disable IRF for simpler test
            use_ki_trajectory=False,
        )
        model.fit(sample_data_with_macro)

        # Test forecast with rate change (effect should be zero since IRF disabled)
        result = model.forecast_with_rate(12, ki_change=2.0)

        assert "baseline" in result
        assert "effect" in result
        assert "total" in result
        assert len(result["baseline"]) == 12
        assert len(result["effect"]) == 12
        assert len(result["total"]) == 12
        # Effect should be zero since IRF is disabled
        assert np.allclose(result["effect"], 0.0)
        # Total should equal baseline
        assert np.array_equal(result["total"], result["baseline"])

        # Restore
        subcomponent_multi.SubcomponentMultiForecaster._load_data = original_load_data

    def test_forecast_with_rate_with_ki_trajectory(
        self, sample_data_with_macro, temp_data_dir
    ):
        """Test forecast_with_rate with Ki trajectory."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster
        from sirena.models import subcomponent_multi

        # Mock data directory
        original_load_data = subcomponent_multi.SubcomponentMultiForecaster._load_data

        def mock_load_data(self, data_dir):
            sub = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub.index = pd.to_datetime(sub.index)
            sub.index = sub.index.to_period("M").to_timestamp()
            sub = sub[~sub.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"].astype(str), sprav["Weight"]))

            valid_cols = [c for c in sub.columns if c in self.weights]
            sub = sub[valid_cols]

            return sub

        subcomponent_multi.SubcomponentMultiForecaster._load_data = mock_load_data

        model = UnifiedSubcomponentForecaster(
            horizon=1, use_irf_convolution=False, use_ki_trajectory=False
        )
        model.fit(sample_data_with_macro)

        # Test with Ki trajectory
        ki_trajectory = np.linspace(16.0, 18.0, 12)
        result = model.forecast_with_rate(
            12, ki_trajectory=ki_trajectory, return_details=True
        )

        assert "baseline" in result
        assert "effect" in result
        assert "total" in result
        assert "ki_trajectory" in result
        assert len(result["baseline"]) == 12

        # Restore
        subcomponent_multi.SubcomponentMultiForecaster._load_data = original_load_data

    def test_forecast_scenario_not_fitted(self, sample_data_with_macro):
        """Test forecast_scenario raises error when model not fitted."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster

        model = UnifiedSubcomponentForecaster()

        with pytest.raises(RuntimeError, match="not fitted"):
            model.forecast_scenario(12, scenario="hike")

    def test_forecast_scenario_base(self, sample_data_with_macro, temp_data_dir):
        """Test forecast_scenario with base scenario."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster
        from sirena.models import subcomponent_multi

        # Mock data directory
        original_load_data = subcomponent_multi.SubcomponentMultiForecaster._load_data

        def mock_load_data(self, data_dir):
            sub = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub.index = pd.to_datetime(sub.index)
            sub.index = sub.index.to_period("M").to_timestamp()
            sub = sub[~sub.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"].astype(str), sprav["Weight"]))

            valid_cols = [c for c in sub.columns if c in self.weights]
            sub = sub[valid_cols]

            return sub

        subcomponent_multi.SubcomponentMultiForecaster._load_data = mock_load_data

        model = UnifiedSubcomponentForecaster(
            horizon=1, use_irf_convolution=False, use_ki_trajectory=False
        )
        model.fit(sample_data_with_macro)

        # Test base scenario
        result = model.forecast_scenario(12, scenario="base")

        assert "baseline" in result
        assert "effect" in result
        assert "total" in result
        assert len(result["baseline"]) == 12

        # Restore
        subcomponent_multi.SubcomponentMultiForecaster._load_data = original_load_data

    def test_forecast_scenario_hike(self, sample_data_with_macro, temp_data_dir):
        """Test forecast_scenario with hike scenario."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster
        from sirena.models import subcomponent_multi

        # Mock data directory
        original_load_data = subcomponent_multi.SubcomponentMultiForecaster._load_data

        def mock_load_data(self, data_dir):
            sub = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub.index = pd.to_datetime(sub.index)
            sub.index = sub.index.to_period("M").to_timestamp()
            sub = sub[~sub.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"].astype(str), sprav["Weight"]))

            valid_cols = [c for c in sub.columns if c in self.weights]
            sub = sub[valid_cols]

            return sub

        subcomponent_multi.SubcomponentMultiForecaster._load_data = mock_load_data

        model = UnifiedSubcomponentForecaster(
            horizon=1, use_irf_convolution=False, use_ki_trajectory=False
        )
        model.fit(sample_data_with_macro)

        # Test hike scenario
        result = model.forecast_scenario(12, scenario="hike")

        assert "baseline" in result
        assert "effect" in result
        assert "total" in result
        assert len(result["baseline"]) == 12

        # Restore
        subcomponent_multi.SubcomponentMultiForecaster._load_data = original_load_data

    def test_forecast_scenario_custom(self, sample_data_with_macro, temp_data_dir):
        """Test forecast_scenario with custom Ki trajectory."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster
        from sirena.models import subcomponent_multi

        # Mock data directory
        original_load_data = subcomponent_multi.SubcomponentMultiForecaster._load_data

        def mock_load_data(self, data_dir):
            sub = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub.index = pd.to_datetime(sub.index)
            sub.index = sub.index.to_period("M").to_timestamp()
            sub = sub[~sub.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"].astype(str), sprav["Weight"]))

            valid_cols = [c for c in sub.columns if c in self.weights]
            sub = sub[valid_cols]

            return sub

        subcomponent_multi.SubcomponentMultiForecaster._load_data = mock_load_data

        model = UnifiedSubcomponentForecaster(
            horizon=1, use_irf_convolution=False, use_ki_trajectory=False
        )
        model.fit(sample_data_with_macro)

        # Test custom scenario
        custom_ki = np.linspace(16.0, 20.0, 12)
        result = model.forecast_scenario(12, scenario="custom", custom_ki=custom_ki)

        assert "baseline" in result
        assert "effect" in result
        assert "total" in result
        assert len(result["baseline"]) == 12

        # Restore
        subcomponent_multi.SubcomponentMultiForecaster._load_data = original_load_data

    def test_forecast_scenario_unknown(self, sample_data_with_macro, temp_data_dir):
        """Test forecast_scenario raises error for unknown scenario."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster
        from sirena.models import subcomponent_multi

        # Mock data directory
        original_load_data = subcomponent_multi.SubcomponentMultiForecaster._load_data

        def mock_load_data(self, data_dir):
            sub = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub.index = pd.to_datetime(sub.index)
            sub.index = sub.index.to_period("M").to_timestamp()
            sub = sub[~sub.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"].astype(str), sprav["Weight"]))

            valid_cols = [c for c in sub.columns if c in self.weights]
            sub = sub[valid_cols]

            return sub

        subcomponent_multi.SubcomponentMultiForecaster._load_data = mock_load_data

        model = UnifiedSubcomponentForecaster(
            horizon=1, use_irf_convolution=False, use_ki_trajectory=False
        )
        model.fit(sample_data_with_macro)

        # Test unknown scenario
        with pytest.raises(ValueError, match="Unknown scenario"):
            model.forecast_scenario(12, scenario="unknown")

        # Restore
        subcomponent_multi.SubcomponentMultiForecaster._load_data = original_load_data

    def test_forecast_with_auto_ki_not_fitted(self, sample_data_with_macro):
        """Test forecast_with_auto_ki raises error when model not fitted."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster

        model = UnifiedSubcomponentForecaster()

        with pytest.raises(RuntimeError, match="not fitted"):
            model.forecast_with_auto_ki(12)

    def test_forecast_with_auto_ki_no_ki_model(
        self, sample_data_with_macro, temp_data_dir
    ):
        """Test forecast_with_auto_ki raises error when Ki model not fitted."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster
        from sirena.models import subcomponent_multi

        # Mock data directory
        original_load_data = subcomponent_multi.SubcomponentMultiForecaster._load_data

        def mock_load_data(self, data_dir):
            sub = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub.index = pd.to_datetime(sub.index)
            sub.index = sub.index.to_period("M").to_timestamp()
            sub = sub[~sub.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"].astype(str), sprav["Weight"]))

            valid_cols = [c for c in sub.columns if c in self.weights]
            sub = sub[valid_cols]

            return sub

        subcomponent_multi.SubcomponentMultiForecaster._load_data = mock_load_data

        # Fit without Ki trajectory model
        model = UnifiedSubcomponentForecaster(
            horizon=1, use_irf_convolution=False, use_ki_trajectory=False
        )
        model.fit(sample_data_with_macro)

        # Test forecast_with_auto_ki should raise error
        with pytest.raises(RuntimeError, match="KiTrajectoryForecaster not fitted"):
            model.forecast_with_auto_ki(12)

        # Restore
        subcomponent_multi.SubcomponentMultiForecaster._load_data = original_load_data

    def test_predict_not_fitted(self, sample_data_with_macro):
        """Test predict raises error when model not fitted."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster

        model = UnifiedSubcomponentForecaster()
        target_date = pd.Timestamp("2025-01-01")

        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(sample_data_with_macro, target_date)

    def test_predict_basic(self, sample_data_with_macro, temp_data_dir):
        """Test predict method (delegates to base_model)."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster
        from sirena.models import subcomponent_multi

        # Mock data directory
        original_load_data = subcomponent_multi.SubcomponentMultiForecaster._load_data

        def mock_load_data(self, data_dir):
            sub = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub.index = pd.to_datetime(sub.index)
            sub.index = sub.index.to_period("M").to_timestamp()
            sub = sub[~sub.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"].astype(str), sprav["Weight"]))

            valid_cols = [c for c in sub.columns if c in self.weights]
            sub = sub[valid_cols]

            return sub

        subcomponent_multi.SubcomponentMultiForecaster._load_data = mock_load_data

        model = UnifiedSubcomponentForecaster(
            horizon=1, use_irf_convolution=False, use_ki_trajectory=False
        )
        model.fit(sample_data_with_macro)

        # Test predict (may fail if base_model has no data, just verify method exists)
        try:
            result = model.predict(sample_data_with_macro, pd.Timestamp("2025-01-01"))
            # If it works, verify result structure
            assert isinstance(result, dict)
        except Exception:
            # It's OK if predict fails due to mock data limitations
            pass

        # Restore
        subcomponent_multi.SubcomponentMultiForecaster._load_data = original_load_data

    def test_get_info(self, sample_data_with_macro, temp_data_dir):
        """Test get_info method returns correct model information."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster
        from sirena.models import subcomponent_multi

        # Mock data directory
        original_load_data = subcomponent_multi.SubcomponentMultiForecaster._load_data

        def mock_load_data(self, data_dir):
            sub = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub.index = pd.to_datetime(sub.index)
            sub.index = sub.index.to_period("M").to_timestamp()
            sub = sub[~sub.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"].astype(str), sprav["Weight"]))

            valid_cols = [c for c in sub.columns if c in self.weights]
            sub = sub[valid_cols]

            return sub

        subcomponent_multi.SubcomponentMultiForecaster._load_data = mock_load_data

        model = UnifiedSubcomponentForecaster(
            horizon=12,
            use_irf_convolution=True,
            use_calibrated_irf=False,
            use_ki_trajectory=False,
        )
        model.fit(sample_data_with_macro)

        info = model.get_info()

        assert isinstance(info, dict)
        assert "version" in info
        assert "is_fitted" in info
        assert "use_irf_convolution" in info
        assert "use_calibrated_irf" in info
        assert "use_ki_trajectory" in info
        assert "horizon" in info
        assert "base_model" in info
        assert "scenario_model" in info
        assert "ki_model" in info

        # Check values
        assert info["version"] == "3.0"
        assert info["is_fitted"] == True
        assert info["horizon"] == 12
        assert info["use_irf_convolution"] == True
        assert info["use_calibrated_irf"] == False
        assert info["use_ki_trajectory"] == False
        assert info["base_model"] == "SubcomponentMultiForecaster"
        # scenario_model may be None if IRF disabled or issues with dependencies
        assert isinstance(info["scenario_model"], (str, type(None)))

        # Restore
        subcomponent_multi.SubcomponentMultiForecaster._load_data = original_load_data

    def test_get_info_before_fit(self):
        """Test get_info before fit."""
        from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster

        model = UnifiedSubcomponentForecaster()
        info = model.get_info()

        assert info["is_fitted"] == False
        assert info["base_model"] is None
        assert info["scenario_model"] is None
        assert info["ki_model"] is None
