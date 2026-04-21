"""
Backtest Framework для СИРЕНА-КБР v4.7

Параметризованный класс для бэктестирования моделей с разными горизонтами:
- h=1: 1 месяц вперед (rolling, 12 окон)
- h=2: 2 месяца вперед (rolling, 12 окон)
- h=12: 12 месяцев вперед (1 окно, годовая траектория)

Автор: Claude Code
Дата: 2025-12-25
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from sirena.models.arima import SARIMAForecaster
from sirena.models.bvar import BayesianVAR
from sirena.models.ebm import EBMForecaster
from sirena.models.ets import ETSForecaster
from sirena.models.lightgbm import LightGBMForecaster
from sirena.models.prophet import ProphetForecaster
from sirena.models.ridge import RidgeForecaster
from sirena.models.ridge_extended import RidgeExtendedForecaster
from sirena.models.bayesian_ridge import BayesianRidgeForecaster
from sirena.models.elasticnet import ElasticNetForecaster
from sirena.models.huber import HuberForecaster
from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster
from sirena.models.ridge_macro import RidgeMacroForecaster  # Optimal macro features

# Rolling Seasonality (experimental)
try:
    from experiments.rolling_seasonality.models.rolling_seasonality_ridge import RollingSeasonalityRidge
    ROLLING_RIDGE_AVAILABLE = True
except ImportError:
    ROLLING_RIDGE_AVAILABLE = False
    print("WARNING: Rolling Seasonality Ridge not available")

try:
    from sirena.models.midas import MIDASForecaster

    MIDAS_AVAILABLE = True
except ImportError:
    MIDAS_AVAILABLE = False
    print("WARNING: MIDAS not available")
try:
    from sirena.models.lmmr_hybrid import LMMRHybridForecaster

    LMMR_HYBRID_AVAILABLE = True
except ImportError:
    LMMR_HYBRID_AVAILABLE = False
    print("WARNING: LMMR Hybrid not available")
try:
    from sirena.models.ngboost_shock import NGBoostShockForecaster

    NGBOOST_SHOCK_AVAILABLE = True
except ImportError:
    NGBOOST_SHOCK_AVAILABLE = False
    print("WARNING: NGBoost Shock not available")

# Optional models
try:
    from sirena.models.catboost_model import CatBoostForecaster

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("WARNING: CatBoost not available")

try:
    from sirena.models.ngboost_model import NGBoostForecaster

    NGBOOST_AVAILABLE = True
except ImportError:
    NGBOOST_AVAILABLE = False
    print("WARNING: NGBoost not available")

# v4.9: Subcomponent models
try:
    from sirena.models.subcomponent import SubcomponentForecaster
    from sirena.models.subcomponent_multi import SubcomponentMultiForecaster

    SUBCOMP_AVAILABLE = True
except ImportError:
    SUBCOMP_AVAILABLE = False
    print("WARNING: Subcomponent models not available")

# v4.9: Microcomponent model (537 items)
try:
    from sirena.models.microcomponent import MicrocomponentForecaster

    MICRO_AVAILABLE = True
except ImportError:
    MICRO_AVAILABLE = False
    print("WARNING: Microcomponent model not available")

# Micro ARIMA (user's external model from micro_test.csv)
try:
    from sirena.models.micro_arima import MicroARIMAForecaster

    MICRO_ARIMA_AVAILABLE = True
except ImportError:
    MICRO_ARIMA_AVAILABLE = False
    print("WARNING: Micro ARIMA loader not available")

warnings.filterwarnings("ignore")


class BacktestRunner:
    """
    Универсальный бэктестер для произвольного горизонта прогнозирования.

    Parameters
    ----------
    horizon : int
        Горизонт прогноза (1, 2, 12)
    test_months : int, default=12
        Количество месяцев для теста
        - h=1, h=2: 12 (rolling за последние 12 месяцев)
        - h=12: 1 (одна точка cutoff, 12 дат в траектории)
    output_dir : str, default='archive/results'
        Директория для сохранения результатов
    """

    def __init__(
        self, horizon: int, test_months: int = 12, output_dir: str = "archive/results"
    ):
        self.horizon = horizon
        self.test_months = test_months
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.bvar_data = None
        self.df_ridge = None
        self.results = []

        print(f"\n{'=' * 70}")
        print(f"BACKTEST h={horizon} ({test_months} test windows)")
        print(f"{'=' * 70}\n")

    def _prepare_data(self):
        """Загрузить данные из inflation_data.csv (Source of Truth)"""
        print("Загрузка данных...")

        # Load BVAR data (Source of Truth)
        bvar_df_full = pd.read_csv("data/inflation_data.csv", sep=";", decimal=",")

        # Fix numeric columns
        cols_to_fix = [
            "mom",
            "Prod",
            "Nonprod",
            "Serv",
            "usd_nom_i",
            "Ruonia",
            "Ki",
            "Ki_i",
        ]
        for col in cols_to_fix:
            if col in bvar_df_full.columns:
                if bvar_df_full[col].dtype == object:
                    bvar_df_full[col] = (
                        bvar_df_full[col].astype(str).str.replace(",", ".")
                    )
                bvar_df_full[col] = pd.to_numeric(bvar_df_full[col], errors="coerce")

        # Parse dates
        bvar_df_full["Date"] = pd.to_datetime(
            bvar_df_full["Date"], format="%d.%m.%Y", errors="coerce"
        )
        if bvar_df_full["Date"].isna().any():
            bvar_df_full["Date"] = pd.to_datetime(bvar_df_full["Date"])
        bvar_df_full["Date"] = bvar_df_full["Date"].dt.to_period("M").dt.to_timestamp()
        bvar_df_full = bvar_df_full.set_index("Date").sort_index()

        # Prepare BVAR vars (CPI is Actual Target)
        self.bvar_data = pd.DataFrame()
        self.bvar_data["CPI"] = bvar_df_full["mom"] - 100
        self.bvar_data["Food"] = bvar_df_full["Prod"] - 100
        self.bvar_data["NonFood"] = bvar_df_full["Nonprod"] - 100
        self.bvar_data["Services"] = bvar_df_full["Serv"] - 100
        self.bvar_data["USD"] = bvar_df_full["usd_nom_i"] - 100
        self.bvar_data["RUONIA"] = bvar_df_full["Ruonia"]
        self.bvar_data = self.bvar_data.dropna()

        # Load Ridge data (from infl_kbr.csv)
        try:
            df_ridge_raw = pd.read_csv("data/infl_kbr.csv", sep=";", decimal=",")

            # Fix dates
            if "Day" in df_ridge_raw.columns:
                df_ridge_raw["Date"] = pd.to_datetime(
                    df_ridge_raw["Day"], format="%d.%m.%Y", errors="coerce"
                )
            elif "Date" in df_ridge_raw.columns:
                df_ridge_raw["Date"] = pd.to_datetime(
                    df_ridge_raw["Date"], errors="coerce"
                )

            # Fix MoM numeric format
            if "MoM" in df_ridge_raw.columns:
                if df_ridge_raw["MoM"].dtype == object:
                    df_ridge_raw["MoM"] = (
                        df_ridge_raw["MoM"].astype(str).str.replace(",", ".")
                    )
                df_ridge_raw["MoM"] = pd.to_numeric(
                    df_ridge_raw["MoM"], errors="coerce"
                )

            # Pivot
            if "Товар" in df_ridge_raw.columns and "MoM" in df_ridge_raw.columns:
                self.df_ridge = df_ridge_raw.pivot_table(
                    index="Date", columns="Товар", values="MoM", aggfunc="first"
                )
            else:
                self.df_ridge = df_ridge_raw.set_index("Date")

            self.df_ridge = self.df_ridge.sort_index()
        except Exception as e:
            print(f"WARNING: Could not load infl_kbr.csv: {e}")
            print("Will use inflation_data.csv for Ridge models")
            # Create pivot from BVAR data
            self.df_ridge = pd.DataFrame(
                {
                    "Все товары и услуги": bvar_df_full["mom"],
                    "Продовольственные товары": bvar_df_full["Prod"],
                    "Непродовольственные товары": bvar_df_full["Nonprod"],
                    "Услуги": bvar_df_full["Serv"],
                }
            )

        # ADD MACRO DATA to df_ridge for models that support it
        macro_cols = ["usd_nom_i", "Ki", "Ruonia", "Ki_i"]
        for col in macro_cols:
            if col in bvar_df_full.columns:
                # Align by index
                self.df_ridge[col] = bvar_df_full[col]

        # Store full macro df for models that need more
        self.df_macro = bvar_df_full.copy()

        print(f"Данные загружены: {len(self.bvar_data)} месяцев")
        print(f"Последняя дата: {self.bvar_data.index.max().strftime('%Y-%m-%d')}\n")

    def _get_test_dates(self) -> pd.DatetimeIndex:
        """Определить test_dates на основе horizon"""
        last_fact = self.bvar_data.index.max()

        if self.horizon == 12:
            # h=12: фиксированный cutoff (last_fact - 12), 12 дат в траектории
            cutoff = last_fact - pd.DateOffset(months=12)
            test_dates = pd.date_range(
                start=cutoff + pd.DateOffset(months=1), end=last_fact, freq="MS"
            )
            print(
                f"h=12: cutoff={cutoff.strftime('%Y-%m')}, траектория {len(test_dates)} месяцев"
            )
        else:
            # h=1, h=2: последние test_months месяцев
            test_dates = pd.date_range(
                end=last_fact, periods=self.test_months, freq="MS"
            )
            print(f"h={self.horizon}: rolling window, {self.test_months} месяцев")

        return test_dates

    def _train_test_split(
        self, target_date: pd.Timestamp
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
        """
        Создать train до (target_date - horizon)

        Returns
        -------
        train_ridge : pd.DataFrame
            Train данные для Ridge/ML моделей
        train_bvar : pd.DataFrame
            Train данные для BVAR
        cutoff : pd.Timestamp
            Дата cutoff
        """
        cutoff = target_date - pd.DateOffset(months=self.horizon)

        train_ridge = self.df_ridge[self.df_ridge.index <= cutoff].copy()
        train_bvar = self.bvar_data[self.bvar_data.index <= cutoff].copy()

        return train_ridge, train_bvar, cutoff

    def _forecast_ridge(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз Ridge модели"""
        try:
            train_ext = train_ridge.copy()
            train_ext.loc[target_date] = np.nan
            model: RidgeForecaster = RidgeForecaster(use_macro=False)
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict(train_ext, target_date)
            return result["prediction"] - 100
        except Exception as e:
            return np.nan

    def _forecast_ridge_extended(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз Ridge Extended"""
        try:
            train_ext = train_ridge.copy()
            train_ext.loc[target_date] = np.nan
            model = RidgeExtendedForecaster()
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict(train_ext, target_date)
            return result["prediction"] - 100
        except Exception as e:
            return np.nan

    def _forecast_bayes_ridge(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз Bayesian Ridge"""
        try:
            train_ext = train_ridge.copy()
            train_ext.loc[target_date] = np.nan
            model = BayesianRidgeForecaster()
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict_with_ci(train_ext, target_date)
            return result["prediction"] - 100
        except Exception as e:
            return np.nan

    def _forecast_elasticnet(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз ElasticNet"""
        try:
            train_ext = train_ridge.copy()
            train_ext.loc[target_date] = np.nan
            model = ElasticNetForecaster()
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict(train_ext, target_date)
            return result["prediction"] - 100
        except Exception as e:
            return np.nan

    def _forecast_huber(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз Huber"""
        try:
            train_ext = train_ridge.copy()
            train_ext.loc[target_date] = np.nan
            model = HuberForecaster()
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict(train_ext, target_date)
            return result["prediction"] - 100
        except Exception as e:
            return np.nan

    def _forecast_ridge_shock(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз Ridge Shock Dummies"""
        try:
            train_ext = train_ridge.copy()
            train_ext.loc[target_date] = np.nan
            model = RidgeShockDummiesForecaster(use_2022_dummy=False)
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict(train_ext, target_date)
            return result["prediction"] - 100
        except Exception as e:
            return np.nan

    def _forecast_ridge_macro(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз Ridge Macro (optimal macro features: USD lag 2, Ki lag 6, Brent lag 5)"""
        try:
            train_ext = train_ridge.copy()
            train_ext.loc[target_date] = np.nan
            model = RidgeMacroForecaster()
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict(train_ext, target_date)
            return result["prediction"] - 100
        except Exception as e:
            return np.nan

    def _forecast_rolling_ridge(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз Rolling Seasonality Ridge (24m window - best experimental model)"""
        if not ROLLING_RIDGE_AVAILABLE:
            return np.nan
        try:
            train_ext = train_ridge.copy()
            train_ext.loc[target_date] = np.nan
            model = RollingSeasonalityRidge(seasonality_window=24)
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict(train_ext, target_date)
            return result["prediction"] - 100
        except Exception as e:
            return np.nan

    def _forecast_ngboost(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз NGBoost"""
        if not NGBOOST_AVAILABLE:
            return np.nan
        try:
            train_ext = train_ridge.copy()
            train_ext.loc[target_date] = np.nan
            model = NGBoostForecaster()
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict(train_ext, target_date)
            return result["prediction"] - 100
        except Exception as e:
            return np.nan

    def _forecast_ngboost_shock(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз NGBoost Shock (ЛУЧШАЯ МОДЕЛЬ!)"""
        if not NGBOOST_SHOCK_AVAILABLE:
            return np.nan
        try:
            train_ext = train_ridge.copy()
            train_ext.loc[target_date] = np.nan
            model = NGBoostShockForecaster()
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict(train_ext, target_date)
            return result["prediction"] - 100
        except Exception as e:
            return np.nan

    def _forecast_lmmr_hybrid(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз LMMR Hybrid"""
        if not LMMR_HYBRID_AVAILABLE:
            return np.nan
        try:
            train_ext = train_ridge.copy()
            train_ext.loc[target_date] = np.nan
            model = LMMRHybridForecaster(alpha=0.5)
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict(train_ext, target_date)
            return result["prediction"] - 100
        except Exception as e:
            return np.nan

    def _forecast_bvar(self, train_bvar: pd.DataFrame) -> float:
        """Прогноз BVAR"""
        try:
            model = BayesianVAR(
                lags=4, lambda1=1.0, var_names=["CPI", "Food", "USD", "RUONIA"]
            )
            model.fit(train_bvar, target_col="CPI")
            fc = model.forecast_full(horizon=self.horizon)
            return fc["median"][0, self.horizon - 1]
        except Exception as e:
            return np.nan

    def _forecast_sarima(self, train_ridge: pd.DataFrame) -> float:
        """Прогноз SARIMA"""
        try:
            sarima_df = pd.DataFrame(
                {"Все товары и услуги": train_ridge["Все товары и услуги"].dropna()}
            )
            model: SARIMAForecaster = SARIMAForecaster()
            model.fit(sarima_df, "Все товары и услуги")
            fc = model.forecast_with_intervals(horizon=self.horizon)
            return fc["mean"][self.horizon - 1]
        except Exception as e:
            return np.nan

    def _forecast_lightgbm(self, train_ridge: pd.DataFrame) -> float:
        """Прогноз LightGBM"""
        try:
            model = LightGBMForecaster()
            model.fit(train_ridge, "Все товары и услуги")
            fc = model.forecast(horizon=self.horizon)
            return fc[self.horizon - 1]
        except Exception as e:
            return np.nan

    def _forecast_prophet(self, train_ridge: pd.DataFrame) -> float:
        """Прогноз Prophet"""
        try:
            model = ProphetForecaster()
            model.fit(train_ridge, "Все товары и услуги")
            fc = model.forecast(horizon=self.horizon)
            return fc[self.horizon - 1]
        except Exception as e:
            return np.nan

    def _forecast_ets(self, train_ridge: pd.DataFrame) -> float:
        """Прогноз ETS"""
        try:
            model = ETSForecaster()
            model.fit(train_ridge, "Все товары и услуги")
            fc = model.forecast(horizon=self.horizon)
            return fc[self.horizon - 1]
        except Exception as e:
            return np.nan

    def _forecast_ebm(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз EBM"""
        try:
            model = EBMForecaster()
            model.fit(train_ridge, "Все товары и услуги")
            fc = model.forecast(horizon=self.horizon)
            return fc[self.horizon - 1] - 100
        except Exception as e:
            return np.nan

    def _forecast_catboost(self, train_ridge: pd.DataFrame) -> float:
        """Прогноз CatBoost"""
        if not CATBOOST_AVAILABLE:
            return np.nan
        try:
            model = CatBoostForecaster()
            model.fit(train_ridge, "Все товары и услуги")
            fc = model.forecast(horizon=self.horizon)
            return fc[self.horizon - 1] - 100
        except Exception as e:
            return np.nan

    def _forecast_midas(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз MIDAS (Mixed Data Sampling)"""
        if not MIDAS_AVAILABLE:
            return np.nan
        try:
            # Add target_date to df (MIDAS needs date to be present)
            train_ext = train_ridge.copy()
            train_ext.loc[target_date] = train_ext.iloc[-1].to_dict()
            train_ext.loc[target_date, "Все товары и услуги"] = np.nan

            model = MIDASForecaster(weight_type="almon", poly_order=2)
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict(train_ext, target_date)
            if result and "prediction" in result:
                return result["prediction"] - 100
            return np.nan
        except Exception as e:
            return np.nan

    def _forecast_subcomp(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз Subcomponent (optimal features)"""
        if not SUBCOMP_AVAILABLE:
            return np.nan
        try:
            model = SubcomponentForecaster(horizon=self.horizon)
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict(train_ridge, target_date)
            if result and "prediction" in result:
                return result["prediction"] - 100
            return np.nan
        except Exception as e:
            return np.nan

    def _forecast_subcomp_multi(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз SubcomponentMulti (optimal models: Ridge, Prophet, NGBoost)"""
        if not SUBCOMP_AVAILABLE:
            return np.nan
        try:
            model = SubcomponentMultiForecaster(horizon=self.horizon)
            model.fit(train_ridge, "Все товары и услуги")
            result = model.predict(train_ridge, target_date)
            if result and "prediction" in result:
                return result["prediction"] - 100
            return np.nan
        except Exception as e:
            return np.nan

    def _forecast_micro(
        self, train_ridge: pd.DataFrame, target_date: pd.Timestamp
    ) -> float:
        """Прогноз Micro ARIMA из micro_test.csv (внешняя модель пользователя)"""
        if not MICRO_ARIMA_AVAILABLE:
            return np.nan
        try:
            # Используем загрузчик прогнозов из файла
            model = MicroARIMAForecaster(
                horizon=self.horizon, file_path="micro_test.csv"
            )
            model.fit()
            result = model.predict(train_ridge, target_date)
            if result and "prediction" in result and not np.isnan(result["prediction"]):
                return result["prediction"] - 100
            return np.nan
        except Exception as e:
            return np.nan

    def _forecast_ensemble(self, predictions: Dict[str, float]) -> float:
        """Прогноз Ensemble (7 моделей с весами)"""
        try:
            preds = {
                "Ridge": (predictions.get("Ridge", np.nan), 0.40),
                "BVAR": (predictions.get("BVAR", np.nan), 0.20),
                "LightGBM": (predictions.get("LightGBM", np.nan), 0.15),
                "Prophet": (predictions.get("Prophet", np.nan), 0.10),
                "SARIMA": (predictions.get("SARIMA", np.nan), 0.05),
                "ETS": (predictions.get("ETS", np.nan), 0.05),
                "EBM": (predictions.get("EBM", np.nan), 0.05),
            }
            valid_preds = {k: v for k, v in preds.items() if not np.isnan(v[0])}
            if valid_preds:
                total_w = sum(w for _, w in valid_preds.values())
                pred_e = sum(p * w / total_w for p, w in valid_preds.values())
                return pred_e
            else:
                return np.nan
        except Exception as e:
            return np.nan

    def run(self):
        """Главный цикл бэктеста"""
        # 1. Prepare data
        self._prepare_data()

        # 2. Get test dates
        test_dates = self._get_test_dates()

        # 3. For h=12: single train/test split
        if self.horizon == 12:
            return self._run_h12(test_dates)

        # 4. For h=1, h=2: rolling window
        return self._run_rolling(test_dates)

    def _run_rolling(self, test_dates: pd.DatetimeIndex):
        """Rolling window backtest (h=1, h=2)"""
        print(
            f"\n{'Месяц':<10} | {'Факт':>6} | {'Ridge':>6} | {'RollRdg':>7} | {'BVAR':>6} | {'NGBoost':>7} | {'LMMR_Hyb':>8} | {'Ансамбль':>9}"
        )
        print("-" * 80)

        for target_date in test_dates:
            # Train/test split
            train_ridge, train_bvar, cutoff = self._train_test_split(target_date)

            # Actual
            if target_date not in self.bvar_data.index:
                continue
            actual = self.bvar_data.loc[target_date, "CPI"]

            # Forecast all models
            predictions = {}
            predictions["Ridge"] = self._forecast_ridge(train_ridge, target_date)
            predictions["Ridge_Ext"] = self._forecast_ridge_extended(
                train_ridge, target_date
            )
            predictions["Bayes_Ridge"] = self._forecast_bayes_ridge(
                train_ridge, target_date
            )
            predictions["ElasticNet"] = self._forecast_elasticnet(
                train_ridge, target_date
            )
            predictions["Huber"] = self._forecast_huber(train_ridge, target_date)
            predictions["Ridge_Shock"] = self._forecast_ridge_shock(
                train_ridge, target_date
            )
            predictions["Ridge_Macro"] = self._forecast_ridge_macro(
                train_ridge, target_date
            )
            predictions["Rolling_Ridge"] = self._forecast_rolling_ridge(
                train_ridge, target_date
            )
            predictions["NGBoost"] = self._forecast_ngboost(train_ridge, target_date)
            predictions["NGBoost_Shock"] = self._forecast_ngboost_shock(
                train_ridge, target_date
            )
            predictions["LMMR_Hybrid"] = self._forecast_lmmr_hybrid(
                train_ridge, target_date
            )
            predictions["BVAR"] = self._forecast_bvar(train_bvar)
            predictions["SARIMA"] = self._forecast_sarima(train_ridge)
            predictions["LightGBM"] = self._forecast_lightgbm(train_ridge)
            predictions["Prophet"] = self._forecast_prophet(train_ridge)
            predictions["ETS"] = self._forecast_ets(train_ridge)
            predictions["EBM"] = self._forecast_ebm(train_ridge, target_date)
            predictions["CatBoost"] = self._forecast_catboost(train_ridge)
            predictions["MIDAS"] = self._forecast_midas(train_ridge, target_date)
            predictions["Subcomp"] = self._forecast_subcomp(train_ridge, target_date)
            predictions["Subcomp_Multi"] = self._forecast_subcomp_multi(
                train_ridge, target_date
            )
            predictions["Micro"] = self._forecast_micro(train_ridge, target_date)
            predictions["Ensemble"] = self._forecast_ensemble(predictions)

            # Store result
            self.results.append({"Date": target_date, "Actual": actual, **predictions})

            # Print progress
            ridge_pred = predictions.get("Ridge", np.nan)
            rolling_pred = predictions.get("Rolling_Ridge", np.nan)
            bvar_pred = predictions.get("BVAR", np.nan)
            ngb_pred = predictions.get("NGBoost_Shock", np.nan)
            lmmr_pred = predictions.get("LMMR_Hybrid", np.nan)
            ens_pred = predictions.get("Ensemble", np.nan)
            print(
                f"{target_date.strftime('%Y-%m'):<10} | {actual:6.2f} | {ridge_pred:6.2f} | {rolling_pred:7.2f} | {bvar_pred:6.2f} | {ngb_pred:7.2f} | {lmmr_pred:6.2f} | {ens_pred:9.2f}"
            )

        return pd.DataFrame(self.results)

    def _run_h12(self, test_dates: pd.DatetimeIndex):
        """Backtest h=12 (фиксированный cutoff, траектория)"""
        print(f"Фиксированный cutoff: {test_dates[0] - pd.DateOffset(months=1)}")
        print(
            f"Траектория: {test_dates[0].strftime('%Y-%m')} → {test_dates[-1].strftime('%Y-%m')}\n"
        )

        # Single train/test split
        cutoff = test_dates[0] - pd.DateOffset(months=1)
        train_ridge = self.df_ridge[self.df_ridge.index <= cutoff].copy()
        train_bvar = self.bvar_data[self.bvar_data.index <= cutoff].copy()

        # Train all models ONCE
        print("Обучение моделей...")

        # Ridge models (use predict for each target_date)
        ridge_model: RidgeForecaster = RidgeForecaster(use_macro=False)
        ridge_model.fit(train_ridge, "Все товары и услуги")

        ridge_ext_model = RidgeExtendedForecaster()
        ridge_ext_model.fit(train_ridge, "Все товары и услуги")

        bayes_ridge_model = BayesianRidgeForecaster()
        bayes_ridge_model.fit(train_ridge, "Все товары и услуги")

        elasticnet_model = ElasticNetForecaster()
        elasticnet_model.fit(train_ridge, "Все товары и услуги")

        huber_model = HuberForecaster()
        huber_model.fit(train_ridge, "Все товары и услуги")

        ridge_shock_model = RidgeShockDummiesForecaster()
        ridge_shock_model.fit(train_ridge, "Все товары и услуги")

        ridge_macro_model = RidgeMacroForecaster()
        ridge_macro_model.fit(train_ridge, "Все товары и услуги")

        rolling_ridge_model = None
        if ROLLING_RIDGE_AVAILABLE:
            rolling_ridge_model = RollingSeasonalityRidge(seasonality_window=24)
            rolling_ridge_model.fit(train_ridge, "Все товары и услуги")

        if NGBOOST_AVAILABLE:
            ngboost_model = NGBoostForecaster()
            ngboost_model.fit(train_ridge, "Все товары и услуги")

        ngboost_shock_model = NGBoostShockForecaster()
        ngboost_shock_model.fit(train_ridge, "Все товары и услуги")

        lmmr_hybrid_model = None
        if LMMR_HYBRID_AVAILABLE:
            try:
                lmmr_hybrid_model = LMMRHybridForecaster()
                lmmr_hybrid_model.fit(train_ridge, "Все товары и услуги")
            except:
                lmmr_hybrid_model = None

        ebm_model = EBMForecaster()
        ebm_model.fit(train_ridge, "Все товары и услуги")

        if CATBOOST_AVAILABLE:
            catboost_model = CatBoostForecaster()
            catboost_model.fit(train_ridge, "Все товары и услуги")

        # MIDAS model
        midas_model = None
        if MIDAS_AVAILABLE:
            try:
                midas_model = MIDASForecaster(weight_type="almon", poly_order=2)
                midas_model.fit(train_ridge, "Все товары и услуги")
            except:
                midas_model = None

        # v4.9: Subcomponent models
        subcomp_model = None
        subcomp_multi_model = None
        if SUBCOMP_AVAILABLE:
            try:
                subcomp_model = SubcomponentForecaster(horizon=12)
                subcomp_model.fit(train_ridge, "Все товары и услуги")
            except:
                pass
            try:
                subcomp_multi_model = SubcomponentMultiForecaster(horizon=12)
                subcomp_multi_model.fit(train_ridge, "Все товары и услуги")
            except:
                pass

        # Models with forecast(12)
        sarima_model: SARIMAForecaster = SARIMAForecaster()
        sarima_train_df = pd.DataFrame(
            {"Все товары и услуги": train_ridge["Все товары и услуги"].dropna()}
        )
        sarima_model.fit(sarima_train_df, "Все товары и услуги")
        sarima_fc = sarima_model.forecast_with_intervals(horizon=12)["mean"]

        bvar_model = BayesianVAR(
            lags=1, lambda1=0.2, var_names=["CPI", "Food", "USD", "RUONIA"]
        )
        bvar_model.fit(train_bvar, "CPI")
        bvar_fc = bvar_model.forecast_full(12)["median"][
            :, 0
        ]  # All horizons, first var (CPI)

        lgb_model = LightGBMForecaster()
        lgb_model.fit(train_ridge, "Все товары и услуги")
        lgb_fc = lgb_model.forecast(12)

        prophet_model = ProphetForecaster()
        prophet_model.fit(train_ridge, "Все товары и услуги")
        prophet_fc = prophet_model.forecast(12)

        ets_model = ETSForecaster()
        ets_model.fit(train_ridge, "Все товары и услуги")
        ets_fc = ets_model.forecast(12)

        print("\nПрогноз траектории...")
        print(
            f"\n{'Месяц':<10} | {'Факт':>6} | {'Ridge':>6} | {'BVAR':>6} | {'SARIMA':>7} | {'ETS':>6} | {'Ансамбль':>9}"
        )
        print("-" * 80)

        # Forecast trajectory
        for i, target_date in enumerate(test_dates):
            # Actual
            if target_date not in self.bvar_data.index:
                continue
            actual = self.bvar_data.loc[target_date, "CPI"]

            # Predictions from forecasted trajectories
            predictions = {}

            # Ridge-based models (use predict for each date)
            train_ext = train_ridge.copy()
            train_ext.loc[target_date] = np.nan

            try:
                ridge_result = ridge_model.predict(train_ext, target_date)
                predictions["Ridge"] = ridge_result["prediction"] - 100
            except:
                predictions["Ridge"] = np.nan

            try:
                ridge_ext_result = ridge_ext_model.predict(train_ext, target_date)
                predictions["Ridge_Ext"] = ridge_ext_result["prediction"] - 100
            except:
                predictions["Ridge_Ext"] = np.nan

            try:
                bayes_result = bayes_ridge_model.predict(train_ext, target_date)
                predictions["Bayes_Ridge"] = bayes_result["prediction"] - 100
            except:
                predictions["Bayes_Ridge"] = np.nan

            try:
                elasticnet_result = elasticnet_model.predict(train_ext, target_date)
                predictions["ElasticNet"] = elasticnet_result["prediction"] - 100
            except:
                predictions["ElasticNet"] = np.nan

            try:
                huber_result = huber_model.predict(train_ext, target_date)
                predictions["Huber"] = huber_result["prediction"] - 100
            except:
                predictions["Huber"] = np.nan

            try:
                ridge_shock_result = ridge_shock_model.predict(train_ext, target_date)
                predictions["Ridge_Shock"] = ridge_shock_result["prediction"] - 100
            except:
                predictions["Ridge_Shock"] = np.nan

            try:
                ridge_macro_result = ridge_macro_model.predict(train_ext, target_date)
                predictions["Ridge_Macro"] = ridge_macro_result["prediction"] - 100
            except:
                predictions["Ridge_Macro"] = np.nan

            try:
                if rolling_ridge_model:
                    rolling_ridge_result = rolling_ridge_model.predict(train_ext, target_date)
                    predictions["Rolling_Ridge"] = rolling_ridge_result["prediction"] - 100
                else:
                    predictions["Rolling_Ridge"] = np.nan
            except:
                predictions["Rolling_Ridge"] = np.nan

            try:
                if NGBOOST_AVAILABLE:
                    ngboost_result = ngboost_model.predict(train_ext, target_date)
                    predictions["NGBoost"] = ngboost_result["prediction"] - 100
                else:
                    predictions["NGBoost"] = np.nan
            except:
                predictions["NGBoost"] = np.nan

            try:
                ngboost_shock_result = ngboost_shock_model.predict(
                    train_ext, target_date
                )
                predictions["NGBoost_Shock"] = ngboost_shock_result["prediction"] - 100
            except:
                predictions["NGBoost_Shock"] = np.nan

            try:
                lmmr_result = lmmr_model.predict(train_ext, target_date)
                predictions["LMMR"] = lmmr_result["prediction"] - 100
            except:
                predictions["LMMR"] = np.nan

            try:
                if lmmr_hybrid_model is not None:
                    lmmr_hybrid_result = lmmr_hybrid_model.predict(
                        train_ext, target_date
                    )
                    predictions["LMMR_Hybrid"] = lmmr_hybrid_result["prediction"] - 100
                else:
                    predictions["LMMR_Hybrid"] = np.nan
            except:
                predictions["LMMR_Hybrid"] = np.nan

            try:
                ebm_fc = ebm_model.forecast(12)
                predictions["EBM"] = ebm_fc[i] - 100 if i < len(ebm_fc) else np.nan
            except:
                predictions["EBM"] = np.nan

            try:
                if CATBOOST_AVAILABLE:
                    catboost_fc = catboost_model.forecast(12)
                    predictions["CatBoost"] = (
                        catboost_fc[i] - 100 if i < len(catboost_fc) else np.nan
                    )
                else:
                    predictions["CatBoost"] = np.nan
            except:
                predictions["CatBoost"] = np.nan

            # MIDAS model
            try:
                if midas_model is not None:
                    midas_result = midas_model.predict(train_ext, target_date)
                    if midas_result and "prediction" in midas_result:
                        predictions["MIDAS"] = midas_result["prediction"] - 100
                    else:
                        predictions["MIDAS"] = np.nan
                else:
                    predictions["MIDAS"] = np.nan
            except:
                predictions["MIDAS"] = np.nan

            # v4.9: Subcomponent models
            try:
                if subcomp_model is not None:
                    subcomp_result = subcomp_model.predict(train_ridge, target_date)
                    if subcomp_result and "prediction" in subcomp_result:
                        predictions["Subcomp"] = subcomp_result["prediction"] - 100
                    else:
                        predictions["Subcomp"] = np.nan
                else:
                    predictions["Subcomp"] = np.nan
            except:
                predictions["Subcomp"] = np.nan

            try:
                if subcomp_multi_model is not None:
                    subcomp_multi_result = subcomp_multi_model.predict(
                        train_ridge, target_date
                    )
                    if subcomp_multi_result and "prediction" in subcomp_multi_result:
                        predictions["Subcomp_Multi"] = (
                            subcomp_multi_result["prediction"] - 100
                        )
                    else:
                        predictions["Subcomp_Multi"] = np.nan
                else:
                    predictions["Subcomp_Multi"] = np.nan
            except:
                predictions["Subcomp_Multi"] = np.nan

            # v4.9: Microcomponent model
            try:
                predictions["Micro"] = self._forecast_micro(train_ridge, target_date)
            except:
                predictions["Micro"] = np.nan

            # Other models: use pre-computed forecasts (with bounds checking)
            predictions["BVAR"] = bvar_fc[i] if i < len(bvar_fc) else np.nan
            predictions["SARIMA"] = sarima_fc[i] if i < len(sarima_fc) else np.nan
            predictions["LightGBM"] = lgb_fc[i] if i < len(lgb_fc) else np.nan
            predictions["Prophet"] = prophet_fc[i] if i < len(prophet_fc) else np.nan
            predictions["ETS"] = ets_fc[i] if i < len(ets_fc) else np.nan

            # Ensemble
            predictions["Ensemble"] = self._forecast_ensemble(predictions)

            # Store result
            self.results.append({"Date": target_date, "Actual": actual, **predictions})

            # Print progress
            ridge_pred = predictions.get("Ridge", np.nan)
            bvar_pred = predictions.get("BVAR", np.nan)
            sarima_pred = predictions.get("SARIMA", np.nan)
            ets_pred = predictions.get("ETS", np.nan)
            ens_pred = predictions.get("Ensemble", np.nan)
            print(
                f"{target_date.strftime('%Y-%m'):<10} | {actual:6.2f} | {ridge_pred:6.2f} | {bvar_pred:6.2f} | {sarima_pred:7.2f} | {ets_pred:6.2f} | {ens_pred:9.2f}"
            )

        return pd.DataFrame(self.results)

    def calculate_metrics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Вычислить метрики для всех моделей"""
        print(f"\n{'=' * 70}")
        print("МЕТРИКИ")
        print(f"{'=' * 70}\n")

        metrics = []

        # Get model columns (exclude Date, Actual)
        model_cols = [
            col for col in results_df.columns if col not in ["Date", "Actual"]
        ]

        for model in model_cols:
            valid = results_df[[model, "Actual"]].dropna()

            if len(valid) == 0:
                continue

            errors = valid["Actual"] - valid[model]

            mae = errors.abs().mean()
            rmse = np.sqrt((errors**2).mean())
            mape = (errors.abs() / valid["Actual"].abs()).mean() * 100
            kpi_violations = (errors.abs() > 0.5).sum()
            coverage_50pct = (errors.abs() <= 0.5).mean() * 100
            max_error = errors.abs().max()
            std_error = errors.std()
            mean_error = errors.mean()

            metrics.append(
                {
                    "Model": model,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "KPI_Violations": kpi_violations,
                    "Coverage_50pct": coverage_50pct,
                    "Max_Error": max_error,
                    "Std_Error": std_error,
                    "Mean_Error": mean_error,
                }
            )

        metrics_df = pd.DataFrame(metrics).sort_values("MAE")

        # Print top 5
        print("Top 5 моделей по MAE:")
        for i, row in metrics_df.head(5).iterrows():
            print(
                f"  {row['Model']:<20s}: MAE {row['MAE']:.3f}, KPI violations {int(row['KPI_Violations'])}/{len(results_df)}"
            )

        return metrics_df

    def save_results(self, results_df: pd.DataFrame, metrics_df: pd.DataFrame):
        """Сохранить результаты в CSV и markdown"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save predictions
        pred_file = self.output_dir / f"backtest_h{self.horizon}_predictions.csv"
        results_df.to_csv(pred_file, index=False)
        print(f"\nПрогнозы сохранены: {pred_file}")

        # Save metrics
        metrics_file = self.output_dir / f"backtest_h{self.horizon}_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Метрики сохранены: {metrics_file}")

        # Create summary markdown
        summary_file = self.output_dir / f"backtest_h{self.horizon}_summary.md"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"# Backtest h={self.horizon} Summary\n\n")
            f.write(
                f"**Period:** {results_df['Date'].min().strftime('%Y-%m-%d')} to {results_df['Date'].max().strftime('%Y-%m-%d')} ({len(results_df)} months)\n"
            )
            f.write(f"**Horizon:** {self.horizon} month(s) ahead\n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("## Top 5 Models\n\n")
            for i, row in metrics_df.head(5).iterrows():
                f.write(
                    f"{i + 1}. **{row['Model']}** — MAE {row['MAE']:.3f} ({int(row['KPI_Violations'])} KPI violations)\n"
                )

            f.write("\n## KPI Violations (|error| > 0.5)\n\n")
            for i, row in metrics_df.head(10).iterrows():
                pct = row["KPI_Violations"] / len(results_df) * 100
                f.write(
                    f"- {row['Model']}: {int(row['KPI_Violations'])}/{len(results_df)} ({pct:.1f}%)\n"
                )

            f.write("\n## Metrics Table\n\n")
            f.write("| Model | MAE | RMSE | KPI Violations | Coverage 50% |\n")
            f.write("|-------|-----|------|----------------|-------------|\n")
            for i, row in metrics_df.head(10).iterrows():
                f.write(
                    f"| {row['Model']} | {row['MAE']:.3f} | {row['RMSE']:.3f} | {int(row['KPI_Violations'])} | {row['Coverage_50pct']:.1f}% |\n"
                )

        print(f"Summary сохранен: {summary_file}")
        print(f"\n{'=' * 70}\n")
