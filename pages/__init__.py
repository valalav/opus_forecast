"""Pages module for dashboard refactoring."""

# Import functions from page modules
# Use exec to bypass Python's import limitations for files starting with numbers
import importlib.util
import os

# Load 1_Forecast.py
spec_forecast = importlib.util.spec_from_file_location(
    "forecast_functions", os.path.join(os.path.dirname(__file__), "1_Forecast.py")
)
module_forecast = importlib.util.module_from_spec(spec_forecast)
spec_forecast.loader.exec_module(module_forecast)

# Load 2_Backtest.py
spec_backtest = importlib.util.spec_from_file_location(
    "backtest_functions", os.path.join(os.path.dirname(__file__), "2_Backtest.py")
)
module_backtest = importlib.util.module_from_spec(spec_backtest)
spec_backtest.loader.exec_module(module_backtest)

# Load 3_Weekly.py
spec_weekly = importlib.util.spec_from_file_location(
    "weekly_functions", os.path.join(os.path.dirname(__file__), "3_Weekly.py")
)
module_weekly = importlib.util.module_from_spec(spec_weekly)
spec_weekly.loader.exec_module(module_weekly)

# Export forecast functions
get_best_model_for_horizon = module_forecast.get_best_model_for_horizon
forecast_with_model = module_forecast.forecast_with_model
calculate_kpi_corrections = module_forecast.calculate_kpi_corrections
render_forecast_tab = module_forecast.render_forecast_tab
render_forecast_h12_tab = module_forecast.render_forecast_h12_tab

# Export backtest functions
load_backtest_data = module_backtest.load_backtest_data
render_backtest_tab = module_backtest.render_backtest_tab

# Load 3_Research.py
spec_research = importlib.util.spec_from_file_location(
    "research_functions", os.path.join(os.path.dirname(__file__), "3_Research.py")
)
module_research = importlib.util.module_from_spec(spec_research)
spec_research.loader.exec_module(module_research)

# Export weekly/nowcast functions
render_alert_panel = module_weekly.render_alert_panel
render_weekly_tab = module_weekly.render_weekly_tab
render_nowcast_tab = module_weekly.render_nowcast_tab

# Export research functions
render_seasonality_tab = module_research.render_seasonality_tab
render_macro_tab = module_research.render_macro_tab
render_regime_indicator = module_research.render_regime_indicator

# Load 4_Compare.py
spec_compare = importlib.util.spec_from_file_location(
    "compare_functions", os.path.join(os.path.dirname(__file__), "4_Compare.py")
)
module_compare = importlib.util.module_from_spec(spec_compare)
spec_compare.loader.exec_module(module_compare)

# Export compare function
render_compare_tab = module_compare.render_compare_tab

# Load constants.py
spec_constants = importlib.util.spec_from_file_location(
    "constants", os.path.join(os.path.dirname(__file__), "constants.py")
)
module_constants = importlib.util.module_from_spec(spec_constants)
spec_constants.loader.exec_module(module_constants)

# Export constants
ALL_MODELS = module_constants.ALL_MODELS
MODEL_COLORS = module_constants.MODEL_COLORS
MONTH_NAMES_RU = module_constants.MONTH_NAMES_RU

__all__ = [
    "get_best_model_for_horizon",
    "forecast_with_model",
    "calculate_kpi_corrections",
    "render_forecast_tab",
    "render_h12_tab",
    "load_backtest_data",
    "render_backtest_tab",
    "render_alert_panel",
    "render_weekly_tab",
    "render_nowcast_tab",
    "render_seasonality_tab",
    "render_macro_tab",
    "render_regime_indicator",
    "render_compare_tab",
    "ALL_MODELS",
    "MODEL_COLORS",
    "MONTH_NAMES_RU",
]
