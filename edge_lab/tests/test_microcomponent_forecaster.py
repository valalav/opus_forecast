"""
Unit tests for MicrocomponentForecaster
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
def sample_micro_data():
    """Generate sample microcomponent data for testing."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    np.random.seed(42)

    # Create sample microcomponent data with 20 microcomponents (subset of 537)
    micro_dict = {}
    for i in range(20):
        code = 300 + i  # Some realistic item codes
        micro_dict[code] = 100.0 + np.random.randn(60) * (0.5 + i * 0.05)

    micro_df = pd.DataFrame(micro_dict, index=dates)
    return micro_df


@pytest.fixture
def sample_micro_csv(sample_micro_data):
    """Generate sample kbr_micro_full.csv for testing."""
    # Convert to long format
    long_data = []
    for code in sample_micro_data.columns:
        for date in sample_micro_data.index:
            long_data.append(
                {
                    "Day": date.strftime("%m/%d/%y 00:00:00"),
                    "Item_code": code,
                    "MoM": sample_micro_data.loc[date, code],
                }
            )

    df = pd.DataFrame(long_data)
    return df


@pytest.fixture
def sample_sprav_csv():
    """Generate sample micro_sprav.csv for testing."""
    # Create 20 microcomponents with different weights
    data = []
    for i in range(20):
        code = 300 + i
        data.append(
            {
                "Item_code": code,
                "Товар": f"Товар {code}",
                "Субкомпонент": 20 + i % 3,  # Spread across subcomponents
                "Weight": 0.05 - i * 0.001,  # Descending weights
            }
        )

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def temp_data_dir(sample_micro_csv, sample_sprav_csv):
    """Create temporary data directory with sample data."""
    temp_dir = tempfile.mkdtemp()

    # Create raw subdirectory
    raw_dir = Path(temp_dir) / "raw"
    raw_dir.mkdir(parents=True)

    # Create kbr_micro_full.csv
    micro_file = Path(temp_dir) / "kbr_micro_full.csv"
    sample_micro_csv.to_csv(micro_file, sep=",", decimal=".", index=False)

    # Create micro_sprav.csv
    sprav_file = raw_dir / "micro_sprav.csv"
    sample_sprav_csv.to_csv(
        sprav_file, sep=";", decimal=",", encoding="utf-8-sig", index=False
    )

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestMicrocomponentForecaster:
    """Test suite for MicrocomponentForecaster."""

    def test_import_model(self):
        """Test model import."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster()
        assert model is not None
        assert model.name == "microcomponent"

    def test_model_parameters(self):
        """Test model default parameters."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster()

        assert model.horizon == 1
        assert model.train_start == "2016-01-01"
        assert model.random_state == 42
        assert model.top_n == 100
        assert model.use_extended_for_volatile == True
        assert model.use_seasonal_adj == True
        assert model._is_fitted == False
        assert len(model.micro_models) == 0
        assert len(model.weights) == 0
        assert len(model.top_items) == 0

    def test_volatile_items(self):
        """Test volatile items set."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        assert len(MicrocomponentForecaster.VOLATILE_ITEMS) > 0
        assert 435 in MicrocomponentForecaster.VOLATILE_ITEMS
        assert 382 in MicrocomponentForecaster.VOLATILE_ITEMS

    def test_seasonal_adj(self):
        """Test seasonal adjustment dictionary."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        assert len(MicrocomponentForecaster.SEASONAL_ADJ) == 12
        for month in range(1, 13):
            assert month in MicrocomponentForecaster.SEASONAL_ADJ
            assert isinstance(
                MicrocomponentForecaster.SEASONAL_ADJ[month], (int, float)
            )

    def test_custom_horizon(self):
        """Test custom horizon parameter."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster(horizon=12)
        assert model.horizon == 12

        model2 = MicrocomponentForecaster(horizon=2)
        assert model2.horizon == 2

    def test_custom_train_start(self):
        """Test custom train_start parameter."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster(train_start="2020-01-01")
        assert model.train_start == "2020-01-01"

    def test_custom_top_n(self):
        """Test custom top_n parameter."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster(top_n=50)
        assert model.top_n == 50

    def test_use_extended_for_volatile_false(self):
        """Test use_extended_for_volatile parameter."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster(use_extended_for_volatile=False)
        assert model.use_extended_for_volatile == False

    def test_use_seasonal_adj_false(self):
        """Test use_seasonal_adj parameter."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster(use_seasonal_adj=False)
        assert model.use_seasonal_adj == False

    def test_create_features_basic(self):
        """Test basic feature creation."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster()

        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        series = pd.Series(100.0 + np.random.randn(60) * 0.5, index=dates)

        features = model._create_features(series, extended=False)

        # Check basic features
        assert "y" in features.columns
        assert "L1" in features.columns
        assert "L2" in features.columns
        assert "L3" in features.columns
        assert "L6" in features.columns
        assert "L12" in features.columns
        assert "D1" in features.columns
        assert "MA3" in features.columns
        assert "month_sin" in features.columns
        assert "month_cos" in features.columns

        # Check extended features NOT present
        assert "MA6" not in features.columns
        assert "STD3" not in features.columns
        assert "STD6" not in features.columns
        assert "MAX3" not in features.columns
        assert "MIN3" not in features.columns
        assert "RANGE3" not in features.columns

    def test_create_features_extended(self):
        """Test extended feature creation for volatile items."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster()

        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        series = pd.Series(100.0 + np.random.randn(60) * 0.5, index=dates)

        features = model._create_features(series, extended=True)

        # Check all extended features present
        assert "MA6" in features.columns
        assert "STD3" in features.columns
        assert "STD6" in features.columns
        assert "MAX3" in features.columns
        assert "MIN3" in features.columns
        assert "RANGE3" in features.columns

    def test_fit_basic(self, sample_data, temp_data_dir):
        """Test basic fit functionality."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        # Mock data directory
        original_load_data = MicrocomponentForecaster._load_data

        def mock_load_data(self, data_dir):
            # Use temp_data_dir directly
            micro_df = pd.read_csv(
                Path(temp_data_dir) / "kbr_micro_full.csv", sep=",", decimal="."
            )
            micro_df["DateParsed"] = pd.to_datetime(
                micro_df["Day"].str.split(" ").str[0],
                format="%m/%d/%y",
                errors="coerce",
            )
            micro_df["Period"] = (
                micro_df["DateParsed"].dt.to_period("M").dt.to_timestamp()
            )
            micro_pivot = micro_df.pivot_table(
                index="Period", columns="Item_code", values="MoM", aggfunc="first"
            )
            micro_pivot = micro_pivot[~micro_pivot.index.duplicated(keep="last")]

            # Load weights
            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "micro_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"], sprav["Weight"]))
            self.item_names = dict(zip(sprav["Item_code"], sprav["Товар"]))
            self.item_subcomp = dict(zip(sprav["Item_code"], sprav["Субкомпонент"]))

            # Determine top-N by weight
            sorted_items = sorted(self.weights.items(), key=lambda x: -x[1])
            self.top_items = set([item for item, _ in sorted_items[: self.top_n]])

            # Filter to items in справочник with valid data
            valid_cols = [c for c in micro_pivot.columns if c in self.weights]
            micro_pivot = micro_pivot[valid_cols]

            # Convert MoM to changes
            micro_pivot = micro_pivot - 100

            return micro_pivot

        MicrocomponentForecaster._load_data = mock_load_data

        try:
            model = MicrocomponentForecaster(top_n=10)
            model.fit(sample_data)

            assert model._is_fitted
            assert len(model.weights) > 0
            assert len(model.micro_models) > 0
            assert len(model.top_items) > 0
        finally:
            MicrocomponentForecaster._load_data = original_load_data

    def test_predict_not_fitted_error(self, sample_data):
        """Test predict raises error when model not fitted."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster()
        target_date = pd.Timestamp("2025-01-01")

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(sample_data, target_date)

    def test_forecast_not_fitted_error(self, sample_data):
        """Test forecast raises error when model not fitted."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster()

        with pytest.raises(ValueError, match="not fitted"):
            model.forecast()

    def test_predict_basic(self, sample_data, temp_data_dir):
        """Test basic predict functionality."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        # Mock data directory
        original_load_data = MicrocomponentForecaster._load_data

        def mock_load_data(self, data_dir):
            micro_df = pd.read_csv(
                Path(temp_data_dir) / "kbr_micro_full.csv", sep=",", decimal="."
            )
            micro_df["DateParsed"] = pd.to_datetime(
                micro_df["Day"].str.split(" ").str[0],
                format="%m/%d/%y",
                errors="coerce",
            )
            micro_df["Period"] = (
                micro_df["DateParsed"].dt.to_period("M").dt.to_timestamp()
            )
            micro_pivot = micro_df.pivot_table(
                index="Period", columns="Item_code", values="MoM", aggfunc="first"
            )
            micro_pivot = micro_pivot[~micro_pivot.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "micro_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"], sprav["Weight"]))
            self.item_names = dict(zip(sprav["Item_code"], sprav["Товар"]))
            self.item_subcomp = dict(zip(sprav["Item_code"], sprav["Субкомпонент"]))

            sorted_items = sorted(self.weights.items(), key=lambda x: -x[1])
            self.top_items = set([item for item, _ in sorted_items[: self.top_n]])

            valid_cols = [c for c in micro_pivot.columns if c in self.weights]
            micro_pivot = micro_pivot[valid_cols]

            micro_pivot = micro_pivot - 100

            return micro_pivot

        MicrocomponentForecaster._load_data = mock_load_data

        try:
            model = MicrocomponentForecaster(top_n=10)
            model.fit(sample_data)

            target_date = pd.Timestamp("2025-01-01")
            result = model.predict(sample_data, target_date)

            assert "prediction" in result
            assert isinstance(result["prediction"], (int, float))
            # Prediction should be around 100 (MoM index)
            assert 90 < result["prediction"] < 110
        finally:
            MicrocomponentForecaster._load_data = original_load_data

    def test_forecast_basic(self, sample_data, temp_data_dir):
        """Test basic forecast functionality."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        # Mock data directory
        original_load_data = MicrocomponentForecaster._load_data

        def mock_load_data(self, data_dir):
            micro_df = pd.read_csv(
                Path(temp_data_dir) / "kbr_micro_full.csv", sep=",", decimal="."
            )
            micro_df["DateParsed"] = pd.to_datetime(
                micro_df["Day"].str.split(" ").str[0],
                format="%m/%d/%y",
                errors="coerce",
            )
            micro_df["Period"] = (
                micro_df["DateParsed"].dt.to_period("M").dt.to_timestamp()
            )
            micro_pivot = micro_df.pivot_table(
                index="Period", columns="Item_code", values="MoM", aggfunc="first"
            )
            micro_pivot = micro_pivot[~micro_pivot.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "micro_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"], sprav["Weight"]))
            self.item_names = dict(zip(sprav["Item_code"], sprav["Товар"]))
            self.item_subcomp = dict(zip(sprav["Item_code"], sprav["Субкомпонент"]))

            sorted_items = sorted(self.weights.items(), key=lambda x: -x[1])
            self.top_items = set([item for item, _ in sorted_items[: self.top_n]])

            valid_cols = [c for c in micro_pivot.columns if c in self.weights]
            micro_pivot = micro_pivot[valid_cols]

            micro_pivot = micro_pivot - 100

            return micro_pivot

        MicrocomponentForecaster._load_data = mock_load_data

        try:
            model = MicrocomponentForecaster(top_n=10, horizon=3)
            model.fit(sample_data)

            forecast = model.forecast()

            assert isinstance(forecast, np.ndarray)
            assert len(forecast) == 3
            # Each forecast is a MoM change
            for val in forecast:
                assert isinstance(val, (int, float))
        finally:
            MicrocomponentForecaster._load_data = original_load_data

    def test_forecast_custom_horizon(self, sample_data, temp_data_dir):
        """Test forecast with custom horizon."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        # Mock data directory
        original_load_data = MicrocomponentForecaster._load_data

        def mock_load_data(self, data_dir):
            micro_df = pd.read_csv(
                Path(temp_data_dir) / "kbr_micro_full.csv", sep=",", decimal="."
            )
            micro_df["DateParsed"] = pd.to_datetime(
                micro_df["Day"].str.split(" ").str[0],
                format="%m/%d/%y",
                errors="coerce",
            )
            micro_df["Period"] = (
                micro_df["DateParsed"].dt.to_period("M").dt.to_timestamp()
            )
            micro_pivot = micro_df.pivot_table(
                index="Period", columns="Item_code", values="MoM", aggfunc="first"
            )
            micro_pivot = micro_pivot[~micro_pivot.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "micro_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"], sprav["Weight"]))
            self.item_names = dict(zip(sprav["Item_code"], sprav["Товар"]))
            self.item_subcomp = dict(zip(sprav["Item_code"], sprav["Субкомпонент"]))

            sorted_items = sorted(self.weights.items(), key=lambda x: -x[1])
            self.top_items = set([item for item, _ in sorted_items[: self.top_n]])

            valid_cols = [c for c in micro_pivot.columns if c in self.weights]
            micro_pivot = micro_pivot[valid_cols]

            micro_pivot = micro_pivot - 100

            return micro_pivot

        MicrocomponentForecaster._load_data = mock_load_data

        try:
            model = MicrocomponentForecaster(top_n=10, horizon=6)
            model.fit(sample_data)

            forecast = model.forecast(horizon=12)

            assert len(forecast) == 12
        finally:
            MicrocomponentForecaster._load_data = original_load_data

    def test_get_stats_not_fitted(self):
        """Test get_stats returns empty dict when not fitted."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster()
        stats = model.get_stats()

        assert stats == {}

    def test_get_stats(self, sample_data, temp_data_dir):
        """Test get_stats after fit."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        # Mock data directory
        original_load_data = MicrocomponentForecaster._load_data

        def mock_load_data(self, data_dir):
            micro_df = pd.read_csv(
                Path(temp_data_dir) / "kbr_micro_full.csv", sep=",", decimal="."
            )
            micro_df["DateParsed"] = pd.to_datetime(
                micro_df["Day"].str.split(" ").str[0],
                format="%m/%d/%y",
                errors="coerce",
            )
            micro_df["Period"] = (
                micro_df["DateParsed"].dt.to_period("M").dt.to_timestamp()
            )
            micro_pivot = micro_df.pivot_table(
                index="Period", columns="Item_code", values="MoM", aggfunc="first"
            )
            micro_pivot = micro_pivot[~micro_pivot.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "micro_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"], sprav["Weight"]))
            self.item_names = dict(zip(sprav["Item_code"], sprav["Товар"]))
            self.item_subcomp = dict(zip(sprav["Item_code"], sprav["Субкомпонент"]))

            sorted_items = sorted(self.weights.items(), key=lambda x: -x[1])
            self.top_items = set([item for item, _ in sorted_items[: self.top_n]])

            valid_cols = [c for c in micro_pivot.columns if c in self.weights]
            micro_pivot = micro_pivot[valid_cols]

            micro_pivot = micro_pivot - 100

            return micro_pivot

        MicrocomponentForecaster._load_data = mock_load_data

        try:
            model = MicrocomponentForecaster(top_n=10)
            model.fit(sample_data)

            stats = model.get_stats()

            assert "total_models" in stats
            assert "top_models" in stats
            assert "voting_models" in stats
            assert "volatile_models" in stats
            assert "total_weight" in stats

            assert stats["total_models"] > 0
            assert stats["top_models"] >= 0
            assert stats["voting_models"] >= 0
            assert stats["volatile_models"] >= 0
            assert stats["total_weight"] > 0
        finally:
            MicrocomponentForecaster._load_data = original_load_data

    def test_get_top_predictions_not_fitted(self):
        """Test get_top_predictions returns empty DataFrame when not fitted."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster()
        target_date = pd.Timestamp("2025-01-01")

        result = model.get_top_predictions(target_date)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_get_top_predictions(self, sample_data, temp_data_dir):
        """Test get_top_predictions after fit."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        # Mock data directory
        original_load_data = MicrocomponentForecaster._load_data

        def mock_load_data(self, data_dir):
            micro_df = pd.read_csv(
                Path(temp_data_dir) / "kbr_micro_full.csv", sep=",", decimal="."
            )
            micro_df["DateParsed"] = pd.to_datetime(
                micro_df["Day"].str.split(" ").str[0],
                format="%m/%d/%y",
                errors="coerce",
            )
            micro_df["Period"] = (
                micro_df["DateParsed"].dt.to_period("M").dt.to_timestamp()
            )
            micro_pivot = micro_df.pivot_table(
                index="Period", columns="Item_code", values="MoM", aggfunc="first"
            )
            micro_pivot = micro_pivot[~micro_pivot.index.duplicated(keep="last")]

            sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "micro_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            self.weights = dict(zip(sprav["Item_code"], sprav["Weight"]))
            self.item_names = dict(zip(sprav["Item_code"], sprav["Товар"]))
            self.item_subcomp = dict(zip(sprav["Item_code"], sprav["Субкомпонент"]))

            sorted_items = sorted(self.weights.items(), key=lambda x: -x[1])
            self.top_items = set([item for item, _ in sorted_items[: self.top_n]])

            valid_cols = [c for c in micro_pivot.columns if c in self.weights]
            micro_pivot = micro_pivot[valid_cols]

            micro_pivot = micro_pivot - 100

            return micro_pivot

        MicrocomponentForecaster._load_data = mock_load_data

        try:
            model = MicrocomponentForecaster(top_n=10)
            model.fit(sample_data)

            target_date = pd.Timestamp("2025-01-01")
            result = model.get_top_predictions(target_date, n=5)

            assert isinstance(result, pd.DataFrame)
            assert len(result) <= 5
            if len(result) > 0:
                assert "Item_code" in result.columns
                assert "Name" in result.columns
                assert "Weight" in result.columns
                assert "Prediction" in result.columns
        finally:
            MicrocomponentForecaster._load_data = original_load_data

    def test_repr(self):
        """Test model string representation."""
        from sirena.models.microcomponent import MicrocomponentForecaster

        model = MicrocomponentForecaster()
        repr_str = str(model)

        assert "MicrocomponentForecaster" in repr_str
