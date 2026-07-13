"""
Unit tests for HierarchicalMicroForecaster
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
from unittest.mock import patch, MagicMock

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

    # Create sample microcomponent data with 15 microcomponents
    micro_dict = {}
    for i in range(15):
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
def sample_micro_sprav_csv():
    """Generate sample micro_sprav.csv for testing."""
    # Create 15 microcomponents mapping to 3 subcomponents
    data = []
    subcomp_map = {0: 20, 1: 21, 2: 22}  # Map to 3 subcomponents
    comp_map = {
        20: "Продовольственные товары",
        21: "Непродовольственные товары",
        22: "Услуги",
    }

    for i in range(15):
        code = 300 + i
        subcomp = subcomp_map[i % 3]
        data.append(
            {
                "Item_code": code,
                "Товар": f"Товар {code}",
                "Субкомпонент": subcomp,
                "Компонент": comp_map[subcomp],
                "Weight": 0.05 - i * 0.001,  # Descending weights
            }
        )

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def sample_subcomp_data():
    """Generate sample subcomponent data for testing (in change format)."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    np.random.seed(42)

    # 3 subcomponents: 20, 21, 22 (values are already in change format, e.g., 0.5 for +0.5%)
    data = pd.DataFrame(
        {
            "20": np.random.randn(60) * 0.4,
            "21": np.random.randn(60) * 0.3,
            "22": np.random.randn(60) * 0.3,
        },
        index=dates,
    )

    return data


@pytest.fixture
def sample_subcomp_csv(sample_subcomp_data):
    """Generate sample sub_mom.csv for testing (in MoM change format)."""
    # Convert to Rosstat format
    df = sample_subcomp_data.copy()
    df["Date"] = df.index.strftime("%d.%m.%Y")
    df = df.reset_index(drop=True)

    # Values are already MoM indices, keep them as is
    df = df[["Date", "20", "21", "22"]]
    return df


@pytest.fixture
def sample_subcomp_sprav_csv():
    """Generate sample subcomp_sprav.csv for testing."""
    data = [
        {
            "Item_code": "20",
            "Товар": "Субкомпонент 20",
            "Компонент": "Продовольственные товары",
            "Weight": 0.395,
        },
        {
            "Item_code": "21",
            "Товар": "Субкомпонент 21",
            "Компонент": "Непродовольственные товары",
            "Weight": 0.365,
        },
        {
            "Item_code": "22",
            "Товар": "Субкомпонент 22",
            "Компонент": "Услуги",
            "Weight": 0.240,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir(
    sample_micro_csv,
    sample_subcomp_csv,
    sample_micro_sprav_csv,
    sample_subcomp_sprav_csv,
):
    """Create temporary data directory with all sample data."""
    temp_dir = tempfile.mkdtemp()

    # Create raw subdirectory
    raw_dir = Path(temp_dir) / "raw"
    raw_dir.mkdir(parents=True)

    # Create kbr_micro_full.csv
    micro_file = Path(temp_dir) / "kbr_micro_full.csv"
    sample_micro_csv.to_csv(micro_file, sep=",", decimal=".", index=False)

    # Create sub_mom.csv
    subcomp_file = Path(temp_dir) / "raw" / "sub_mom.csv"
    sample_subcomp_csv.to_csv(
        subcomp_file, sep=";", decimal=",", encoding="utf-8-sig", index=False
    )

    # Create micro_sprav.csv
    micro_sprav_file = raw_dir / "micro_sprav.csv"
    sample_micro_sprav_csv.to_csv(
        micro_sprav_file, sep=";", decimal=",", encoding="utf-8-sig", index=False
    )

    # Create subcomp_sprav.csv
    subcomp_sprav_file = raw_dir / "subcomp_sprav.csv"
    sample_subcomp_sprav_csv.to_csv(
        subcomp_sprav_file, sep=";", decimal=",", encoding="utf-8-sig", index=False
    )

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestHierarchicalMicroForecaster:
    """Test suite for HierarchicalMicroForecaster."""

    def test_import_model(self):
        """Test model import."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        model = HierarchicalMicroForecaster()
        assert model is not None
        assert model.name == "hierarchical_micro"

    def test_model_parameters(self):
        """Test model default parameters."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        model = HierarchicalMicroForecaster()

        assert model.region_code == 7
        assert model.horizon == 1
        assert model.train_start == "2016-01-01"
        assert isinstance(model.use_prophet, bool)  # Depends on Prophet availability
        assert model.random_state == 42
        assert model._is_fitted == False
        assert len(model.micro_models) == 0
        assert len(model.subcomp_models) == 0
        assert len(model.micro_weights) == 0
        assert len(model.subcomp_weights) == 0

    def test_prophet_subcomponents(self):
        """Test prophet subcomponents dictionary."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        assert len(HierarchicalMicroForecaster.PROPHET_SUBCOMPONENTS) > 0
        assert "14" in HierarchicalMicroForecaster.PROPHET_SUBCOMPONENTS  # ЖКХ
        assert "44" in HierarchicalMicroForecaster.PROPHET_SUBCOMPONENTS  # Образование

    def test_volatile_micro(self):
        """Test volatile microcomponents set."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        assert len(HierarchicalMicroForecaster.VOLATILE_MICRO) > 0
        assert 435 in HierarchicalMicroForecaster.VOLATILE_MICRO  # Огурцы
        assert 506 in HierarchicalMicroForecaster.VOLATILE_MICRO  # Помидоры

    def test_components_weights(self):
        """Test component weights dictionary."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        assert len(HierarchicalMicroForecaster.COMPONENTS) == 3
        assert "Продовольственные товары" in HierarchicalMicroForecaster.COMPONENTS
        assert "Непродовольственные товары" in HierarchicalMicroForecaster.COMPONENTS
        assert "Услуги" in HierarchicalMicroForecaster.COMPONENTS

        # Check weights sum to approximately 1.0
        total_weight = sum(
            HierarchicalMicroForecaster.COMPONENTS[comp]["weight"]
            for comp in HierarchicalMicroForecaster.COMPONENTS
        )
        assert abs(total_weight - 1.0) < 0.01

    def test_custom_region_code(self):
        """Test custom region_code parameter."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        model = HierarchicalMicroForecaster(region_code=10)
        assert model.region_code == 10

        model2 = HierarchicalMicroForecaster(region_code=77)  # Москва
        assert model2.region_code == 77

    def test_custom_horizon(self):
        """Test custom horizon parameter."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        model = HierarchicalMicroForecaster(horizon=12)
        assert model.horizon == 12

        model2 = HierarchicalMicroForecaster(horizon=2)
        assert model2.horizon == 2

    def test_custom_train_start(self):
        """Test custom train_start parameter."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        model = HierarchicalMicroForecaster(train_start="2020-01-01")
        assert model.train_start == "2020-01-01"

    def test_use_prophet_for_services(self):
        """Test use_prophet_for_services parameter."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        # This depends on whether Prophet is installed
        model = HierarchicalMicroForecaster(use_prophet_for_services=True)
        # use_prophet will be True only if Prophet is actually available
        assert isinstance(model.use_prophet, bool)

    def test_create_features_basic(self):
        """Test basic feature creation."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        model = HierarchicalMicroForecaster()

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
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        model = HierarchicalMicroForecaster()

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
        assert "YoY" in features.columns
        assert "is_summer" in features.columns
        assert "is_winter" in features.columns

    def test_fit_basic(self, sample_data, temp_data_dir):
        """Test basic fit functionality with mocked data directory."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        # Mock data directory to use temp_data_dir
        original_load_hierarchy = HierarchicalMicroForecaster._load_hierarchy
        original_load_micro_data = HierarchicalMicroForecaster._load_micro_data
        original_load_subcomp_data = HierarchicalMicroForecaster._load_subcomp_data

        def mock_load_hierarchy(self, data_dir):
            micro_sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "micro_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )

            for _, row in micro_sprav.iterrows():
                item_code = row["Item_code"]
                self.micro_weights[item_code] = row["Weight"]
                self.micro_names[item_code] = row["Товар"]
                self.micro_to_subcomp[item_code] = str(row["Субкомпонент"])
                comp = row["Компонент"]
                if pd.notna(row["Субкомпонент"]):
                    self.subcomp_to_comp[str(row["Субкомпонент"])] = comp

            subcomp_sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )

            for _, row in subcomp_sprav.iterrows():
                code = str(row["Item_code"])
                self.subcomp_weights[code] = row["Weight"]
                self.subcomp_names[code] = row["Товар"]
                if code not in self.subcomp_to_comp:
                    self.subcomp_to_comp[code] = row["Компонент"]

        def mock_load_micro_data(self, data_dir):
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
            micro_pivot = micro_pivot.sort_index()
            micro_pivot = micro_pivot[~micro_pivot.index.duplicated(keep="last")]
            return micro_pivot - 100

        def mock_load_subcomp_data(self, data_dir):
            sub_df = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub_df["Date"] = pd.to_datetime(sub_df["Date"], format="%d.%m.%Y")
            sub_df = sub_df.set_index("Date").sort_index()
            sub_df.index = sub_df.index.to_period("M").to_timestamp()
            sub_df = sub_df[~sub_df.index.duplicated(keep="last")]
            return sub_df

        HierarchicalMicroForecaster._load_hierarchy = mock_load_hierarchy
        HierarchicalMicroForecaster._load_micro_data = mock_load_micro_data
        HierarchicalMicroForecaster._load_subcomp_data = mock_load_subcomp_data

        try:
            model = HierarchicalMicroForecaster()
            model.fit(sample_data)

            assert model._is_fitted
            assert len(model.micro_weights) > 0
            assert len(model.subcomp_weights) > 0
            assert len(model.micro_models) > 0
            assert len(model.subcomp_models) > 0
        finally:
            HierarchicalMicroForecaster._load_hierarchy = original_load_hierarchy
            HierarchicalMicroForecaster._load_micro_data = original_load_micro_data
            HierarchicalMicroForecaster._load_subcomp_data = original_load_subcomp_data

    def test_predict_not_fitted_error(self, sample_data):
        """Test predict raises error when model not fitted."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        model = HierarchicalMicroForecaster()
        target_date = pd.Timestamp("2025-01-01")

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(sample_data, target_date)

    def test_forecast_not_fitted_error(self, sample_data):
        """Test forecast raises error when model not fitted."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        model = HierarchicalMicroForecaster()

        with pytest.raises(ValueError, match="not fitted"):
            model.forecast()

    def test_predict_basic(self, sample_data, temp_data_dir):
        """Test basic predict functionality with mocked data directory."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        # Mock data directory
        original_load_hierarchy = HierarchicalMicroForecaster._load_hierarchy
        original_load_micro_data = HierarchicalMicroForecaster._load_micro_data
        original_load_subcomp_data = HierarchicalMicroForecaster._load_subcomp_data

        def mock_load_hierarchy(self, data_dir):
            micro_sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "micro_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )

            for _, row in micro_sprav.iterrows():
                item_code = row["Item_code"]
                self.micro_weights[item_code] = row["Weight"]
                self.micro_names[item_code] = row["Товар"]
                self.micro_to_subcomp[item_code] = str(row["Субкомпонент"])
                comp = row["Компонент"]
                if pd.notna(row["Субкомпонент"]):
                    self.subcomp_to_comp[str(row["Субкомпонент"])] = comp

            subcomp_sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )

            for _, row in subcomp_sprav.iterrows():
                code = str(row["Item_code"])
                self.subcomp_weights[code] = row["Weight"]
                self.subcomp_names[code] = row["Товар"]
                if code not in self.subcomp_to_comp:
                    self.subcomp_to_comp[code] = row["Компонент"]

        def mock_load_micro_data(self, data_dir):
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
            micro_pivot = micro_pivot.sort_index()
            micro_pivot = micro_pivot[~micro_pivot.index.duplicated(keep="last")]
            return micro_pivot - 100

        def mock_load_subcomp_data(self, data_dir):
            sub_df = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub_df["Date"] = pd.to_datetime(sub_df["Date"], format="%d.%m.%Y")
            sub_df = sub_df.set_index("Date").sort_index()
            sub_df.index = sub_df.index.to_period("M").to_timestamp()
            sub_df = sub_df[~sub_df.index.duplicated(keep="last")]
            return sub_df

        HierarchicalMicroForecaster._load_hierarchy = mock_load_hierarchy
        HierarchicalMicroForecaster._load_micro_data = mock_load_micro_data
        HierarchicalMicroForecaster._load_subcomp_data = mock_load_subcomp_data

        try:
            model = HierarchicalMicroForecaster()
            model.fit(sample_data)

            target_date = pd.Timestamp("2025-01-01")
            result = model.predict(None, target_date)

            assert "prediction" in result
            assert isinstance(result["prediction"], (int, float))
            # Prediction should be around 100 (MoM index)
            assert 90 < result["prediction"] < 110
            assert "components" in result
            assert "subcomponents" in result
            assert "coverage" in result
        finally:
            HierarchicalMicroForecaster._load_hierarchy = original_load_hierarchy
            HierarchicalMicroForecaster._load_micro_data = original_load_micro_data
            HierarchicalMicroForecaster._load_subcomp_data = original_load_subcomp_data

    def test_forecast_basic(self, sample_data, temp_data_dir):
        """Test basic forecast functionality with mocked data directory."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        # Mock data directory
        original_load_hierarchy = HierarchicalMicroForecaster._load_hierarchy
        original_load_micro_data = HierarchicalMicroForecaster._load_micro_data
        original_load_subcomp_data = HierarchicalMicroForecaster._load_subcomp_data

        def mock_load_hierarchy(self, data_dir):
            micro_sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "micro_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )

            for _, row in micro_sprav.iterrows():
                item_code = row["Item_code"]
                self.micro_weights[item_code] = row["Weight"]
                self.micro_names[item_code] = row["Товар"]
                self.micro_to_subcomp[item_code] = str(row["Субкомпонент"])
                comp = row["Компонент"]
                if pd.notna(row["Субкомпонент"]):
                    self.subcomp_to_comp[str(row["Субкомпонент"])] = comp

            subcomp_sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )

            for _, row in subcomp_sprav.iterrows():
                code = str(row["Item_code"])
                self.subcomp_weights[code] = row["Weight"]
                self.subcomp_names[code] = row["Товар"]
                if code not in self.subcomp_to_comp:
                    self.subcomp_to_comp[code] = row["Компонент"]

        def mock_load_micro_data(self, data_dir):
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
            micro_pivot = micro_pivot.sort_index()
            micro_pivot = micro_pivot[~micro_pivot.index.duplicated(keep="last")]
            return micro_pivot - 100

        def mock_load_subcomp_data(self, data_dir):
            sub_df = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub_df["Date"] = pd.to_datetime(sub_df["Date"], format="%d.%m.%Y")
            sub_df = sub_df.set_index("Date").sort_index()
            sub_df.index = sub_df.index.to_period("M").to_timestamp()
            sub_df = sub_df[~sub_df.index.duplicated(keep="last")]
            return sub_df

        HierarchicalMicroForecaster._load_hierarchy = mock_load_hierarchy
        HierarchicalMicroForecaster._load_micro_data = mock_load_micro_data
        HierarchicalMicroForecaster._load_subcomp_data = mock_load_subcomp_data

        try:
            model = HierarchicalMicroForecaster(horizon=3)
            model.fit(sample_data)

            forecast = model.forecast()

            assert isinstance(forecast, np.ndarray)
            assert len(forecast) == 3
            # Each forecast is a MoM change
            for val in forecast:
                assert isinstance(val, (int, float))
        finally:
            HierarchicalMicroForecaster._load_hierarchy = original_load_hierarchy
            HierarchicalMicroForecaster._load_micro_data = original_load_micro_data
            HierarchicalMicroForecaster._load_subcomp_data = original_load_subcomp_data

    def test_get_detailed_forecast(self, sample_data, temp_data_dir):
        """Test get_detailed_forecast method."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        # Mock data directory
        original_load_hierarchy = HierarchicalMicroForecaster._load_hierarchy
        original_load_micro_data = HierarchicalMicroForecaster._load_micro_data
        original_load_subcomp_data = HierarchicalMicroForecaster._load_subcomp_data

        def mock_load_hierarchy(self, data_dir):
            micro_sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "micro_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )

            for _, row in micro_sprav.iterrows():
                item_code = row["Item_code"]
                self.micro_weights[item_code] = row["Weight"]
                self.micro_names[item_code] = row["Товар"]
                self.micro_to_subcomp[item_code] = str(row["Субкомпонент"])
                comp = row["Компонент"]
                if pd.notna(row["Субкомпонент"]):
                    self.subcomp_to_comp[str(row["Субкомпонент"])] = comp

            subcomp_sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )

            for _, row in subcomp_sprav.iterrows():
                code = str(row["Item_code"])
                self.subcomp_weights[code] = row["Weight"]
                self.subcomp_names[code] = row["Товар"]
                if code not in self.subcomp_to_comp:
                    self.subcomp_to_comp[code] = row["Компонент"]

        def mock_load_micro_data(self, data_dir):
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
            micro_pivot = micro_pivot.sort_index()
            micro_pivot = micro_pivot[~micro_pivot.index.duplicated(keep="last")]
            return micro_pivot - 100

        def mock_load_subcomp_data(self, data_dir):
            sub_df = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub_df["Date"] = pd.to_datetime(sub_df["Date"], format="%d.%m.%Y")
            sub_df = sub_df.set_index("Date").sort_index()
            sub_df.index = sub_df.index.to_period("M").to_timestamp()
            sub_df = sub_df[~sub_df.index.duplicated(keep="last")]
            return sub_df

        HierarchicalMicroForecaster._load_hierarchy = mock_load_hierarchy
        HierarchicalMicroForecaster._load_micro_data = mock_load_micro_data
        HierarchicalMicroForecaster._load_subcomp_data = mock_load_subcomp_data

        try:
            model = HierarchicalMicroForecaster()
            model.fit(sample_data)

            target_date = pd.Timestamp("2025-01-01")
            result = model.get_detailed_forecast(target_date)

            assert "total" in result
            assert "coverage" in result
            assert "components" in result
            assert "top_subcomponents" in result
            assert isinstance(result["top_subcomponents"], list)
        finally:
            HierarchicalMicroForecaster._load_hierarchy = original_load_hierarchy
            HierarchicalMicroForecaster._load_micro_data = original_load_micro_data
            HierarchicalMicroForecaster._load_subcomp_data = original_load_subcomp_data

    def test_get_coverage_report(self, sample_data, temp_data_dir):
        """Test get_coverage_report method."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        # Mock data directory
        original_load_hierarchy = HierarchicalMicroForecaster._load_hierarchy
        original_load_micro_data = HierarchicalMicroForecaster._load_micro_data
        original_load_subcomp_data = HierarchicalMicroForecaster._load_subcomp_data

        def mock_load_hierarchy(self, data_dir):
            micro_sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "micro_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )

            for _, row in micro_sprav.iterrows():
                item_code = row["Item_code"]
                self.micro_weights[item_code] = row["Weight"]
                self.micro_names[item_code] = row["Товар"]
                self.micro_to_subcomp[item_code] = str(row["Субкомпонент"])
                comp = row["Компонент"]
                if pd.notna(row["Субкомпонент"]):
                    self.subcomp_to_comp[str(row["Субкомпонент"])] = comp

            subcomp_sprav = pd.read_csv(
                Path(temp_data_dir) / "raw" / "subcomp_sprav.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )

            for _, row in subcomp_sprav.iterrows():
                code = str(row["Item_code"])
                self.subcomp_weights[code] = row["Weight"]
                self.subcomp_names[code] = row["Товар"]
                if code not in self.subcomp_to_comp:
                    self.subcomp_to_comp[code] = row["Компонент"]

        def mock_load_micro_data(self, data_dir):
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
            micro_pivot = micro_pivot.sort_index()
            micro_pivot = micro_pivot[~micro_pivot.index.duplicated(keep="last")]
            return micro_pivot - 100

        def mock_load_subcomp_data(self, data_dir):
            sub_df = pd.read_csv(
                Path(temp_data_dir) / "raw" / "sub_mom.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            sub_df["Date"] = pd.to_datetime(sub_df["Date"], format="%d.%m.%Y")
            sub_df = sub_df.set_index("Date").sort_index()
            sub_df.index = sub_df.index.to_period("M").to_timestamp()
            sub_df = sub_df[~sub_df.index.duplicated(keep="last")]
            return sub_df

        HierarchicalMicroForecaster._load_hierarchy = mock_load_hierarchy
        HierarchicalMicroForecaster._load_micro_data = mock_load_micro_data
        HierarchicalMicroForecaster._load_subcomp_data = mock_load_subcomp_data

        try:
            model = HierarchicalMicroForecaster()
            model.fit(sample_data)

            report = model.get_coverage_report()

            assert "micro_models" in report
            assert "subcomp_models" in report
            assert "micro_weight" in report
            assert "subcomp_weight" in report
            assert "coverage_by_component" in report
            assert isinstance(report["micro_models"], int)
            assert isinstance(report["subcomp_models"], int)
        finally:
            HierarchicalMicroForecaster._load_hierarchy = original_load_hierarchy
            HierarchicalMicroForecaster._load_micro_data = original_load_micro_data
            HierarchicalMicroForecaster._load_subcomp_data = original_load_subcomp_data

    def test_aggregate_comp_to_total(self, sample_data, temp_data_dir):
        """Test component to total aggregation logic."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        model = HierarchicalMicroForecaster()

        # Test with full component predictions
        comp_preds = {
            "Продовольственные товары": 0.5,
            "Непродовольственные товары": 0.3,
            "Услуги": 0.2,
        }

        total = model._aggregate_comp_to_total(comp_preds)

        # Weighted average: 0.395*0.5 + 0.365*0.3 + 0.24*0.2
        expected = (
            model.COMPONENTS["Продовольственные товары"]["weight"] * 0.5
            + model.COMPONENTS["Непродовольственные товары"]["weight"] * 0.3
            + model.COMPONENTS["Услуги"]["weight"] * 0.2
        )

        assert abs(total - expected) < 0.001

        # Test with empty components
        empty_total = model._aggregate_comp_to_total({})
        assert empty_total == 0.0

    def test_repr(self):
        """Test model string representation."""
        from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

        model = HierarchicalMicroForecaster()
        repr_str = str(model)

        assert "HierarchicalMicroForecaster" in repr_str
