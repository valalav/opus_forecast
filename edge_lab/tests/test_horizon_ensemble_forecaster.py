"""
Unit tests for HorizonEnsembleForecaster
Tests are created in edge_lab but import from parent sirena package
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
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
            "Ruonia": 15 + np.random.randn(60) * 0.5,
            "usd_nom_i": 75 + np.random.randn(60) * 2.0,
            "brent": 80 + np.random.randn(60) * 3.0,
        },
        index=dates,
    )

    return data


@pytest.fixture
def sample_micro_data():
    """Generate sample microcomponent data for testing."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    np.random.seed(42)

    # Create sample microcomponent data with 10 microcomponents
    micro_dict = {}
    for i in range(10):
        code = f"100{i}"
        micro_dict[code] = 100.0 + np.random.randn(60) * 0.5

    micro_df = pd.DataFrame(micro_dict, index=dates)

    return micro_df


@pytest.fixture
def temp_data_dir(sample_micro_data):
    """Create temporary data directory with micro data."""
    temp_dir = tempfile.mkdtemp()

    # Create raw subdirectory
    raw_dir = Path(temp_dir) / "raw"
    raw_dir.mkdir(parents=True)

    # Create kbr_micro_full.csv
    micro_file = raw_dir / "kbr_micro_full.csv"
    sample_micro_data.to_csv(micro_file)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestHorizonEnsembleForecaster:
    """Test suite for HorizonEnsembleForecaster."""

    def test_import_model(self):
        """Test model import."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster()
        assert model is not None
        assert model.name == "horizon_ensemble"

    def test_model_parameters(self):
        """Test model default parameters."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster()

        assert model.horizon == 1
        assert model.train_start == "2016-01-01"
        assert model.random_state == 42
        assert model._is_fitted == False
        assert model.huber_model is None
        assert model.micro_model is None

    def test_weights_for_h1(self):
        """Test weights for horizon 1."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster(horizon=1)

        assert model.weights["huber"] == 0.80
        assert model.weights["micro"] == 0.20
        assert abs(model.weights["huber"] + model.weights["micro"] - 1.0) < 1e-6

    def test_weights_for_h2(self):
        """Test weights for horizon 2."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster(horizon=2)

        assert model.weights["huber"] == 0.75
        assert model.weights["micro"] == 0.25

    def test_weights_for_h3(self):
        """Test weights for horizon 3."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster(horizon=3)

        assert model.weights["huber"] == 0.65
        assert model.weights["micro"] == 0.35

    def test_weights_for_h6(self):
        """Test weights for horizon 6."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster(horizon=6)

        assert model.weights["huber"] == 0.50
        assert model.weights["micro"] == 0.50

    def test_weights_for_h12(self):
        """Test weights for horizon 12."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster(horizon=12)

        assert model.weights["huber"] == 0.30
        assert model.weights["micro"] == 0.70

    def test_weights_for_other_horizons(self):
        """Test weights interpolation for other horizons."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        # h=4: interpolate between h=3 and h=6
        model4 = HorizonEnsembleForecaster(horizon=4)
        expected_huber4 = 0.65 + (0.50 - 0.65) * (1 / 3)
        assert abs(model4.weights["huber"] - expected_huber4) < 0.01

        # h=5: interpolate between h=3 and h=6
        model5 = HorizonEnsembleForecaster(horizon=5)
        expected_huber5 = 0.65 + (0.50 - 0.65) * (2 / 3)
        assert abs(model5.weights["huber"] - expected_huber5) < 0.01

    def test_weights_below_min_horizon(self):
        """Test weights for horizon below minimum."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        # h=0 should use h=1 weights
        model0 = HorizonEnsembleForecaster(horizon=0)

        assert model0.weights["huber"] == 0.80
        assert model0.weights["micro"] == 0.20

    def test_weights_above_max_horizon(self):
        """Test weights for horizon above maximum."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        # h=24 should use h=12 weights
        model24 = HorizonEnsembleForecaster(horizon=24)

        assert model24.weights["huber"] == 0.30
        assert model24.weights["micro"] == 0.70

    def test_custom_horizon(self):
        """Test custom horizon parameter."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster(horizon=6)
        assert model.horizon == 6
        assert model.weights["huber"] == 0.50

    def test_custom_train_start(self):
        """Test custom train_start parameter."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster(train_start="2020-01-01")
        assert model.train_start == "2020-01-01"

    def test_custom_random_state(self):
        """Test custom random_state parameter."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster(random_state=123)
        assert model.random_state == 123

    def test_fit_basic(self, sample_data_with_macro, temp_data_dir):
        """Test basic fit functionality."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster
        from sirena.models import horizon_ensemble

        # Mock data directory for micro model
        original_getattr = horizon_ensemble.__dict__.get("__getattr__", None)

        def mock_import_micro():
            # Create a dummy micro model
            class DummyMicroModel:
                def __init__(
                    self, horizon=1, train_start="2016-01-01", random_state=42
                ):
                    self.horizon = horizon
                    self.train_start = train_start
                    self.random_state = random_state
                    self._is_fitted = False

                def fit(self, df, target_col="Все товары и услуги"):
                    self._is_fitted = True
                    self.macro_df = df.copy()
                    return self

                def predict(self, df, target_date):
                    if not self._is_fitted:
                        raise ValueError("not fitted")
                    # Return dummy prediction
                    return {"prediction": 100.5}

            return DummyMicroModel

        # Temporarily replace MicrocomponentForecaster
        original_micro = horizon_ensemble.MicrocomponentForecaster
        horizon_ensemble.MicrocomponentForecaster = mock_import_micro()

        model = HorizonEnsembleForecaster(horizon=1)
        model.fit(sample_data_with_macro)

        assert model._is_fitted
        assert model.huber_model is not None
        assert model.micro_model is not None
        assert model.huber_model._is_fitted
        assert model.micro_model._is_fitted
        assert model.macro_df is not None

        # Restore
        horizon_ensemble.MicrocomponentForecaster = original_micro

    def test_fit_not_fitted_error(self, sample_data_with_macro):
        """Test predict raises error when model not fitted."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster()
        target_date = pd.Timestamp("2025-01-01")

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(sample_data_with_macro, target_date)

    def test_forecast_not_fitted_error(self, sample_data_with_macro):
        """Test forecast raises error when model not fitted."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster()

        with pytest.raises(ValueError, match="not fitted"):
            model.forecast()

    def test_predict_basic(self, sample_data_with_macro, temp_data_dir):
        """Test basic predict functionality."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster
        from sirena.models import horizon_ensemble

        # Create dummy micro model
        class DummyMicroModel:
            def __init__(self, horizon=1, train_start="2016-01-01", random_state=42):
                self.horizon = horizon
                self.train_start = train_start
                self.random_state = random_state
                self._is_fitted = False

            def fit(self, df, target_col="Все товары и услуги"):
                self._is_fitted = True
                self.macro_df = df.copy()
                return self

            def predict(self, df, target_date):
                if not self._is_fitted:
                    raise ValueError("not fitted")
                return {"prediction": 100.3}  # Dummy prediction

        original_micro = horizon_ensemble.MicrocomponentForecaster
        horizon_ensemble.MicrocomponentForecaster = DummyMicroModel

        model = HorizonEnsembleForecaster(horizon=1)
        model.fit(sample_data_with_macro)

        target_date = sample_data_with_macro.index[-1]
        result = model.predict(sample_data_with_macro, target_date)

        assert "prediction" in result
        assert "huber_pred" in result
        assert "micro_pred" in result
        assert "weights" in result
        assert result["prediction"] is not None
        assert result["weights"]["huber"] == 0.80
        assert result["weights"]["micro"] == 0.20

        # Restore
        horizon_ensemble.MicrocomponentForecaster = original_micro

    def test_predict_range(self, sample_data_with_macro, temp_data_dir):
        """Test predict returns reasonable values."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster
        from sirena.models import horizon_ensemble

        class DummyMicroModel:
            def __init__(self, horizon=1, train_start="2016-01-01", random_state=42):
                self._is_fitted = False

            def fit(self, df, target_col="Все товары и услуги"):
                self._is_fitted = True
                return self

            def predict(self, df, target_date):
                return {"prediction": 100.5}

        original_micro = horizon_ensemble.MicrocomponentForecaster
        horizon_ensemble.MicrocomponentForecaster = DummyMicroModel

        model = HorizonEnsembleForecaster(horizon=1)
        model.fit(sample_data_with_macro)

        target_date = sample_data_with_macro.index[-1]
        result = model.predict(sample_data_with_macro, target_date)

        assert 95 < result["prediction"] < 105

        # Restore
        horizon_ensemble.MicrocomponentForecaster = original_micro

    def test_predict_huber_fallback(self, sample_data_with_macro):
        """Test predict when micro model fails."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster
        from sirena.models import horizon_ensemble

        # Create failing micro model
        class FailingMicroModel:
            def __init__(self, horizon=1, train_start="2016-01-01", random_state=42):
                self._is_fitted = False

            def fit(self, df, target_col="Все товары и услуги"):
                self._is_fitted = True
                return self

            def predict(self, df, target_date):
                raise Exception("Micro model failed!")

        original_micro = horizon_ensemble.MicrocomponentForecaster
        horizon_ensemble.MicrocomponentForecaster = FailingMicroModel

        model = HorizonEnsembleForecaster(horizon=1)
        model.fit(sample_data_with_macro)

        target_date = sample_data_with_macro.index[-1]
        result = model.predict(sample_data_with_macro, target_date)

        # Should fall back to Huber only
        assert result["huber_pred"] is not None
        assert result["micro_pred"] is None
        assert result["prediction"] is not None

        # Restore
        horizon_ensemble.MicrocomponentForecaster = original_micro

    def test_predict_both_fail(self, sample_data_with_macro):
        """Test predict when both models fail."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster
        from sirena.models import horizon_ensemble

        # Create failing micro model
        class FailingMicroModel:
            def __init__(self, horizon=1, train_start="2016-01-01", random_state=42):
                self._is_fitted = False

            def fit(self, df, target_col="Все товары и услуги"):
                self._is_fitted = True
                return self

            def predict(self, df, target_date):
                raise Exception("Micro model failed!")

        # Mock Huber to also fail
        original_micro = horizon_ensemble.MicrocomponentForecaster
        original_huber = horizon_ensemble.HuberForecaster

        class FailingHuberModel:
            def __init__(self, horizon=1, train_start="2016-01-01", random_state=42):
                self._is_fitted = False

            def fit(self, df, target_col="Все товары и услуги"):
                self._is_fitted = True
                return self

            def predict(self, df, target_date):
                raise Exception("Huber model failed!")

        horizon_ensemble.MicrocomponentForecaster = FailingMicroModel
        horizon_ensemble.HuberForecaster = FailingHuberModel

        model = HorizonEnsembleForecaster(horizon=1)
        model.fit(sample_data_with_macro)

        target_date = sample_data_with_macro.index[-1]
        result = model.predict(sample_data_with_macro, target_date)

        # Should return 0.0 + 100 = 100
        assert result["prediction"] == 100.0
        assert result["huber_pred"] is None
        assert result["micro_pred"] is None

        # Restore
        horizon_ensemble.MicrocomponentForecaster = original_micro
        horizon_ensemble.HuberForecaster = original_huber

    def test_forecast_basic(self, sample_data_with_macro):
        """Test basic forecast functionality."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster
        from sirena.models import horizon_ensemble

        class DummyMicroModel:
            def __init__(self, horizon=1, train_start="2016-01-01", random_state=42):
                self._is_fitted = False

            def fit(self, df, target_col="Все товары и услуги"):
                self._is_fitted = True
                return self

            def predict(self, df, target_date):
                return {"prediction": 100.5}

        original_micro = horizon_ensemble.MicrocomponentForecaster
        horizon_ensemble.MicrocomponentForecaster = DummyMicroModel

        model = HorizonEnsembleForecaster(horizon=6)
        model.fit(sample_data_with_macro)

        forecast = model.forecast(horizon=6)

        assert len(forecast) == 6
        assert all(isinstance(v, (int, float)) for v in forecast)

        # Restore
        horizon_ensemble.MicrocomponentForecaster = original_micro

    def test_forecast_different_horizons(self, sample_data_with_macro):
        """Test forecast with different horizons."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster
        from sirena.models import horizon_ensemble

        class DummyMicroModel:
            def __init__(self, horizon=1, train_start="2016-01-01", random_state=42):
                self._is_fitted = False

            def fit(self, df, target_col="Все товары и услуги"):
                self._is_fitted = True
                return self

            def predict(self, df, target_date):
                return {"prediction": 100.5}

        original_micro = horizon_ensemble.MicrocomponentForecaster
        horizon_ensemble.MicrocomponentForecaster = DummyMicroModel

        model = HorizonEnsembleForecaster(horizon=1)
        model.fit(sample_data_with_macro)

        for h in [1, 3, 6, 12]:
            forecast = model.forecast(horizon=h)
            assert len(forecast) == h

        # Restore
        horizon_ensemble.MicrocomponentForecaster = original_micro

    def test_forecast_uses_default_horizon(self, sample_data_with_macro):
        """Test forecast uses default horizon if not specified."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster
        from sirena.models import horizon_ensemble

        class DummyMicroModel:
            def __init__(self, horizon=1, train_start="2016-01-01", random_state=42):
                self._is_fitted = False

            def fit(self, df, target_col="Все товары и услуги"):
                self._is_fitted = True
                return self

            def predict(self, df, target_date):
                return {"prediction": 100.5}

        original_micro = horizon_ensemble.MicrocomponentForecaster
        horizon_ensemble.MicrocomponentForecaster = DummyMicroModel

        model = HorizonEnsembleForecaster(horizon=6)
        model.fit(sample_data_with_macro)

        forecast = model.forecast()  # No horizon specified

        assert len(forecast) == 6

        # Restore
        horizon_ensemble.MicrocomponentForecaster = original_micro

    def test_get_model_contributions(self, sample_data_with_macro):
        """Test get_model_contributions method."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster
        from sirena.models import horizon_ensemble

        class DummyMicroModel:
            def __init__(self, horizon=1, train_start="2016-01-01", random_state=42):
                self._is_fitted = False

            def fit(self, df, target_col="Все товары и услуги"):
                self._is_fitted = True
                return self

            def predict(self, df, target_date):
                return {"prediction": 100.3}

        original_micro = horizon_ensemble.MicrocomponentForecaster
        horizon_ensemble.MicrocomponentForecaster = DummyMicroModel

        model = HorizonEnsembleForecaster(horizon=1)
        model.fit(sample_data_with_macro)

        target_date = sample_data_with_macro.index[-1]
        contributions = model.get_model_contributions(target_date)

        assert "ensemble" in contributions
        assert "huber" in contributions
        assert "micro" in contributions

        # Check Huber contribution structure
        assert "prediction" in contributions["huber"]
        assert "weight" in contributions["huber"]
        assert "contribution" in contributions["huber"]

        # Check Micro contribution structure
        assert "prediction" in contributions["micro"]
        assert "weight" in contributions["micro"]
        assert "contribution" in contributions["micro"]

        # Restore
        horizon_ensemble.MicrocomponentForecaster = original_micro

    def test_get_model_contributions_not_fitted(self, sample_data_with_macro):
        """Test get_model_contributions when not fitted."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster()
        target_date = pd.Timestamp("2025-01-01")

        contributions = model.get_model_contributions(target_date)

        assert contributions == {}

    def test_get_model_contributions_values(self, sample_data_with_macro):
        """Test get_model_contributions returns correct values."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster
        from sirena.models import horizon_ensemble

        class DummyMicroModel:
            def __init__(self, horizon=1, train_start="2016-01-01", random_state=42):
                self._is_fitted = False

            def fit(self, df, target_col="Все товары и услуги"):
                self._is_fitted = True
                return self

            def predict(self, df, target_date):
                return {"prediction": 100.5}

        original_micro = horizon_ensemble.MicrocomponentForecaster
        horizon_ensemble.MicrocomponentForecaster = DummyMicroModel

        model = HorizonEnsembleForecaster(horizon=1)
        model.fit(sample_data_with_macro)

        target_date = sample_data_with_macro.index[-1]
        contributions = model.get_model_contributions(target_date)

        # Weights should sum to 1
        w_huber = contributions["huber"]["weight"]
        w_micro = contributions["micro"]["weight"]
        assert abs(w_huber + w_micro - 1.0) < 1e-6

        # For h=1, Huber weight is 0.80
        assert w_huber == 0.80
        assert w_micro == 0.20

        # Restore
        horizon_ensemble.MicrocomponentForecaster = original_micro

    def test_repr(self):
        """Test model string representation."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        model = HorizonEnsembleForecaster()
        repr_str = str(model)

        # Check that repr contains relevant info
        assert "HorizonEnsembleForecaster" in repr_str

    def test_weights_class_attribute(self):
        """Test WEIGHTS class attribute."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        # Check all predefined horizons are present
        assert 1 in HorizonEnsembleForecaster.WEIGHTS
        assert 2 in HorizonEnsembleForecaster.WEIGHTS
        assert 3 in HorizonEnsembleForecaster.WEIGHTS
        assert 6 in HorizonEnsembleForecaster.WEIGHTS
        assert 12 in HorizonEnsembleForecaster.WEIGHTS

        # Check weights sum to 1
        for h, w in HorizonEnsembleForecaster.WEIGHTS.items():
            assert abs(w["huber"] + w["micro"] - 1.0) < 1e-6

    def test_weight_interpolation_consistency(self):
        """Test weight interpolation is consistent."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

        # Test that weights decrease for Huber as horizon increases
        h1_model = HorizonEnsembleForecaster(horizon=1)
        h12_model = HorizonEnsembleForecaster(horizon=12)

        assert h1_model.weights["huber"] > h12_model.weights["huber"]
        assert h1_model.weights["micro"] < h12_model.weights["micro"]

        # h=1: Huber 80%, h=12: Huber 30%
        assert h1_model.weights["huber"] == 0.80
        assert h12_model.weights["huber"] == 0.30

    def test_forecast_with_trajectory_weights(self, sample_data_with_macro):
        """Test forecast uses step-specific weights in trajectory."""
        from sirena.models.horizon_ensemble import HorizonEnsembleForecaster
        from sirena.models import horizon_ensemble

        # Track which weights were used
        used_weights = []

        class DummyMicroModel:
            def __init__(self, horizon=1, train_start="2016-01-01", random_state=42):
                self._is_fitted = False

            def fit(self, df, target_col="Все товары и услуги"):
                self._is_fitted = True
                return self

            def predict(self, df, target_date):
                return {"prediction": 100.5}

        original_micro = horizon_ensemble.MicrocomponentForecaster

        # Monkey-patch _get_weights to track calls
        original_get_weights = HorizonEnsembleForecaster._get_weights

        def tracked_get_weights(self, horizon):
            used_weights.append(horizon)
            return original_get_weights(self, horizon)

        HorizonEnsembleForecaster._get_weights = tracked_get_weights
        horizon_ensemble.MicrocomponentForecaster = DummyMicroModel

        model = HorizonEnsembleForecaster(horizon=6)
        model.fit(sample_data_with_macro)

        forecast = model.forecast(horizon=6)

        # _get_weights is called during init (for h=6) and for each step (1-6)
        # So we expect 7 calls: [6, 1, 2, 3, 4, 5, 6]
        assert len(used_weights) == 7
        # First call is init with h=6, then 6 steps
        assert used_weights[0] == 6
        assert used_weights[1:] == [1, 2, 3, 4, 5, 6]

        # Restore
        HorizonEnsembleForecaster._get_weights = original_get_weights
        horizon_ensemble.MicrocomponentForecaster = original_micro
