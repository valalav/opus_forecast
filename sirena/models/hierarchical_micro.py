#!/usr/bin/env python3
"""
ИЕРАРХИЧЕСКАЯ МИКРОКОМПОНЕНТНАЯ МОДЕЛЬ v2
==========================================
Полная иерархия: Микро (537) → Субкомп (45) → Компонент (3) → Total MoM

Архитектура:
1. Микрокомпоненты: Ridge/VotingRegressor для каждого из 537 товаров
2. Субкомпоненты: Fallback если микро недоступен + Prophet для услуг
3. Компоненты: Агрегация с весами (Прод 39.5%, Непрод 36.5%, Услуги 24%)
4. Total: Финальная агрегация в MoM

Особенности:
- 100% охват весов через fallback на субкомпоненты
- Prophet для ЖКХ, образования, туризма (сезонные услуги)
- Расширенные признаки для плодоовощей (высокая волатильность)
- Готовность к мультирегиональности (region_code параметр)

Использование:
    from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

    model = HierarchicalMicroForecaster(region_code=7)  # 7 = КБР
    model.fit(df)
    forecast = model.forecast(horizon=12)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler

# Optional: Prophet for seasonal services
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    pass


class HierarchicalMicroForecaster:
    """
    Hierarchical bottom-up forecaster: Micro → Subcomp → Component → Total

    Parameters
    ----------
    region_code : int
        Region code (7 = КБР, default)
    horizon : int
        Forecast horizon (1, 2, or 12)
    train_start : str
        Start date for training
    use_prophet_for_services : bool
        Use Prophet for seasonal services (ЖКХ, образование)
    """

    name = "hierarchical_micro"

    # Субкомпоненты для Prophet (сезонные услуги)
    PROPHET_SUBCOMPONENTS = {
        "14": "ЖКХ",  # Жилищные и коммунальные услуги
        "44": "Образование",  # Услуги образования
        "67": "Туризм",  # Услуги зарубежного туризма
        "46": "Культура",  # Услуги организаций культуры
    }

    # Волатильные микрокомпоненты (плодоовощи) - расширенные признаки
    VOLATILE_MICRO = {
        435,  # Огурцы свежие
        506,  # Помидоры свежие
        382,  # Морковь
        279,  # Капуста белокочанная свежая
        305,  # Картофель
        755,  # Яблоки
        342,  # Лук репчатый
        118,  # Бананы
        187,  # Виноград
        604,  # Свекла столовая
    }

    # Компоненты и их коды
    COMPONENTS = {
        "Продовольственные товары": {"weight": 0.395, "code": "food"},
        "Непродовольственные товары": {"weight": 0.365, "code": "nonfood"},
        "Услуги": {"weight": 0.240, "code": "services"},
    }

    def __init__(
        self,
        region_code=7,
        horizon=1,
        train_start="2016-01-01",
        use_prophet_for_services=True,
        random_state=42,
    ):
        self.region_code = region_code
        self.horizon = horizon
        self.train_start = train_start
        self.use_prophet = use_prophet_for_services and PROPHET_AVAILABLE
        self.random_state = random_state

        self._is_fitted = False

        # Models storage
        self.micro_models = {}  # {item_code: model_data}
        self.subcomp_models = {}  # {subcomp_code: model_data}
        self.component_models = {}  # {component_name: model_data}

        # Hierarchy data
        self.micro_weights = {}  # {item_code: weight}
        self.subcomp_weights = {}  # {subcomp_code: weight}
        self.micro_to_subcomp = {}  # {item_code: subcomp_code}
        self.subcomp_to_comp = {}  # {subcomp_code: component_name}
        self.micro_names = {}
        self.subcomp_names = {}

        # Coverage tracking
        self.micro_coverage = {}  # {subcomp_code: [fitted_micro_codes]}

    def _load_hierarchy(self, data_dir):
        """Load hierarchy справочник and weights."""
        # Микрокомпоненты
        micro_sprav = pd.read_csv(
            data_dir / "raw" / "micro_sprav.csv",
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )

        for _, row in micro_sprav.iterrows():
            item_code = row["Item_code"]
            self.micro_weights[item_code] = row["Weight"]
            self.micro_names[item_code] = row["Товар"]
            self.micro_to_subcomp[item_code] = (
                str(row["Субкомпонент"]) if pd.notna(row["Субкомпонент"]) else None
            )

            # Map to component
            comp = row["Компонент"]
            if pd.notna(row["Субкомпонент"]):
                self.subcomp_to_comp[str(row["Субкомпонент"])] = comp

        # Субкомпоненты
        subcomp_sprav = pd.read_csv(
            data_dir / "raw" / "subcomp_sprav.csv",
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

        # Services without subcomponent - map directly to component
        services_micro = micro_sprav[micro_sprav["Субкомпонент"].isna()]
        for _, row in services_micro.iterrows():
            self.micro_to_subcomp[row["Item_code"]] = f"service_{row['Item_code']}"
            self.subcomp_to_comp[f"service_{row['Item_code']}"] = "Услуги"
            self.subcomp_weights[f"service_{row['Item_code']}"] = row["Weight"]

    def _load_micro_data(self, data_dir):
        """Load microcomponent historical data."""
        micro_df = pd.read_csv(data_dir / "kbr_micro_full.csv", sep=",", decimal=".")

        # Filter by region if needed
        if "Region_code" in micro_df.columns:
            micro_df = micro_df[micro_df["Region_code"] == self.region_code]

        # Parse dates (format: MM/DD/YY HH:MM:SS)
        micro_df["DateParsed"] = pd.to_datetime(
            micro_df["Day"].str.split(" ").str[0], format="%m/%d/%y", errors="coerce"
        )
        micro_df["Period"] = micro_df["DateParsed"].dt.to_period("M").dt.to_timestamp()

        # Pivot
        micro_pivot = micro_df.pivot_table(
            index="Period", columns="Item_code", values="MoM", aggfunc="first"
        )
        micro_pivot = micro_pivot.sort_index()
        micro_pivot = micro_pivot[~micro_pivot.index.duplicated(keep="last")]

        # Convert to changes (MoM - 100)
        return micro_pivot - 100

    def _load_subcomp_data(self, data_dir):
        """Load subcomponent historical data."""
        sub_df = pd.read_csv(
            data_dir / "raw" / "sub_mom.csv", sep=";", decimal=",", encoding="utf-8-sig"
        )
        sub_df["Date"] = pd.to_datetime(sub_df["Date"], format="%d.%m.%Y")
        sub_df = sub_df.set_index("Date").sort_index()
        sub_df.index = sub_df.index.to_period("M").to_timestamp()
        sub_df = sub_df[~sub_df.index.duplicated(keep="last")]

        # NOTE: sub_mom.csv values are already in change format (e.g., 1.0 = +1.0%)
        # No conversion needed!
        return sub_df

    def _create_features(self, series, extended=False):
        """Create features for ML models."""
        df = pd.DataFrame({"y": series})

        # Basic lags
        for lag in [1, 2, 3, 6, 12]:
            df[f"L{lag}"] = df["y"].shift(lag)

        # Momentum
        df["D1"] = df["y"].diff(1)
        df["MA3"] = df["y"].rolling(3).mean()

        # Seasonality
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

        if extended:
            # Extended features for volatile items (vegetables)
            df["MA6"] = df["y"].rolling(6).mean()
            df["STD3"] = df["y"].rolling(3).std()
            df["STD6"] = df["y"].rolling(6).std()
            df["YoY"] = df["y"] - df["y"].shift(12)  # Year-over-year change
            df["MAX3"] = df["y"].rolling(3).max()
            df["MIN3"] = df["y"].rolling(3).min()
            df["RANGE3"] = df["MAX3"] - df["MIN3"]
            # Seasonal indicators for vegetables (summer = low prices)
            df["is_summer"] = df.index.month.isin([6, 7, 8]).astype(int)
            df["is_winter"] = df.index.month.isin([12, 1, 2]).astype(int)

        return df

    def _fit_ridge(self, series, alpha=100.0, extended=False):
        """Fit Ridge model."""
        df = self._create_features(series, extended=extended)
        df["target"] = df["y"].shift(-self.horizon)
        df = df.dropna()

        if self.train_start:
            df = df[df.index >= pd.to_datetime(self.train_start)]

        if len(df) < 24:
            return None

        feature_cols = [c for c in df.columns if c not in ["target", "y"]]
        X = df[feature_cols].values
        y = df["target"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_scaled, y)

        return {
            "type": "ridge",
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "extended": extended,
            "last_data": series.copy(),
        }

    def _fit_voting(self, series):
        """Fit VotingRegressor."""
        df = self._create_features(series, extended=False)
        df["target"] = df["y"].shift(-self.horizon)
        df = df.dropna()

        if self.train_start:
            df = df[df.index >= pd.to_datetime(self.train_start)]

        if len(df) < 24:
            return None

        feature_cols = [c for c in df.columns if c not in ["target", "y"]]
        X = df[feature_cols].values
        y = df["target"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = VotingRegressor(
            [
                ("ridge", Ridge(alpha=100.0, random_state=self.random_state)),
                (
                    "lasso",
                    Lasso(alpha=0.1, random_state=self.random_state, max_iter=5000),
                ),
            ]
        )
        model.fit(X_scaled, y)

        return {
            "type": "voting",
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "extended": False,
            "last_data": series.copy(),
        }

    def _fit_prophet(self, series, subcomp_code):
        """Fit Prophet for seasonal services."""
        if not PROPHET_AVAILABLE:
            return self._fit_ridge(series)

        df_prophet = pd.DataFrame({"ds": series.index, "y": series.values})

        if self.train_start:
            df_prophet = df_prophet[
                df_prophet["ds"] >= pd.to_datetime(self.train_start)
            ]

        if len(df_prophet) < 24:
            return None

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.05,
        )
        model.fit(df_prophet)

        return {"type": "prophet", "model": model, "last_data": series.copy()}

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "HierarchicalMicroForecaster":
        """
        Fit hierarchical model.

        Parameters
        ----------
        df : pd.DataFrame
            Main inflation DataFrame (for macro context)
        target_col : str
            Target column (ignored, using micro data)
        """
        data_dir = Path(__file__).parent.parent.parent / "data"

        # Load hierarchy
        self._load_hierarchy(data_dir)

        # Load data
        micro_data = self._load_micro_data(data_dir)
        subcomp_data = self._load_subcomp_data(data_dir)

        self.macro_df = df.copy()

        # Initialize coverage tracking
        for subcomp in self.subcomp_weights.keys():
            self.micro_coverage[subcomp] = []

        # 1. Fit micro models
        micro_fitted = 0
        for item_code in micro_data.columns:
            if item_code not in self.micro_weights:
                continue

            series = micro_data[item_code].dropna()
            if len(series) < 36:
                continue

            # Use extended features for volatile items
            is_volatile = item_code in self.VOLATILE_MICRO

            if is_volatile:
                result = self._fit_ridge(series, alpha=200.0, extended=True)
            else:
                result = self._fit_voting(series)

            if result:
                self.micro_models[item_code] = result
                micro_fitted += 1

                # Track coverage
                subcomp = self.micro_to_subcomp.get(item_code)
                if subcomp and subcomp in self.micro_coverage:
                    self.micro_coverage[subcomp].append(item_code)

        # 2. Fit subcomponent models (for fallback)
        subcomp_fitted = 0
        for subcomp_code in subcomp_data.columns:
            if subcomp_code not in self.subcomp_weights:
                continue

            series = subcomp_data[subcomp_code].dropna()
            if len(series) < 24:
                continue

            # Use Prophet for seasonal services
            if self.use_prophet and subcomp_code in self.PROPHET_SUBCOMPONENTS:
                result = self._fit_prophet(series, subcomp_code)
            else:
                result = self._fit_ridge(series)

            if result:
                self.subcomp_models[subcomp_code] = result
                subcomp_fitted += 1

        # Calculate coverage stats
        total_micro_weight = sum(
            self.micro_weights.get(k, 0) for k in self.micro_models.keys()
        )
        total_subcomp_weight = sum(
            self.subcomp_weights.get(k, 0) for k in self.subcomp_models.keys()
        )

        self._is_fitted = True

        print(f"HierarchicalMicroForecaster fitted:")
        print(
            f"  Micro models: {micro_fitted} ({total_micro_weight * 100:.1f}% weight)"
        )
        print(
            f"  Subcomp models (fallback): {subcomp_fitted} ({total_subcomp_weight * 100:.1f}% weight)"
        )
        if self.use_prophet:
            prophet_count = sum(
                1 for m in self.subcomp_models.values() if m["type"] == "prophet"
            )
            print(f"  Prophet models: {prophet_count}")

        return self

    def _predict_single(self, model_data, target_date):
        """Predict using a single model."""
        if model_data["type"] == "prophet":
            return self._predict_prophet(model_data, target_date)
        else:
            return self._predict_ml(model_data, target_date)

    def _predict_ml(self, model_data, target_date):
        """Predict using ML model (Ridge/Voting)."""
        series = model_data["last_data"]
        extended = model_data.get("extended", False)
        df = self._create_features(series, extended=extended)

        pred_date = target_date - pd.DateOffset(months=self.horizon)
        if pred_date not in df.index:
            pred_date = df.index[-1]

        feature_cols = model_data["feature_cols"]
        X = df.loc[[pred_date], feature_cols].values

        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=0)

        X_scaled = model_data["scaler"].transform(X)
        return model_data["model"].predict(X_scaled)[0]

    def _predict_prophet(self, model_data, target_date):
        """Predict using Prophet."""
        model = model_data["model"]
        future = model.make_future_dataframe(periods=self.horizon + 12, freq="MS")
        forecast = model.predict(future)

        pred_row = forecast[forecast["ds"] == target_date]
        if len(pred_row) > 0:
            return pred_row["yhat"].values[0]
        return forecast["yhat"].iloc[-1]

    def _aggregate_micro_to_subcomp(self, micro_preds, target_date):
        """
        Aggregate micro predictions to subcomponents.
        Uses fallback to subcomp model if micro coverage is incomplete.
        """
        subcomp_preds = {}

        for subcomp_code in self.subcomp_weights.keys():
            # Get micro predictions for this subcomp
            micro_in_subcomp = [
                (item, pred)
                for item, pred in micro_preds.items()
                if self.micro_to_subcomp.get(item) == subcomp_code
            ]

            if micro_in_subcomp:
                # Calculate covered weight
                covered_weight = sum(
                    self.micro_weights.get(item, 0) for item, _ in micro_in_subcomp
                )
                total_subcomp_weight = self.subcomp_weights.get(subcomp_code, 0)

                if covered_weight > 0:
                    # Weighted average of micro predictions
                    weighted_sum = sum(
                        self.micro_weights.get(item, 0) * pred
                        for item, pred in micro_in_subcomp
                    )
                    micro_avg = weighted_sum / covered_weight

                    # Check if we have full coverage
                    coverage_ratio = (
                        covered_weight / total_subcomp_weight
                        if total_subcomp_weight > 0
                        else 0
                    )

                    if coverage_ratio >= 0.8:
                        # Good coverage - use micro aggregation
                        subcomp_preds[subcomp_code] = micro_avg
                    else:
                        # Partial coverage - blend with subcomp model if available
                        if subcomp_code in self.subcomp_models:
                            try:
                                subcomp_pred = self._predict_single(
                                    self.subcomp_models[subcomp_code], target_date
                                )
                                # Blend based on coverage
                                subcomp_preds[subcomp_code] = (
                                    coverage_ratio * micro_avg
                                    + (1 - coverage_ratio) * subcomp_pred
                                )
                            except:
                                subcomp_preds[subcomp_code] = micro_avg
                        else:
                            subcomp_preds[subcomp_code] = micro_avg
            else:
                # No micro coverage - use subcomp model (fallback)
                if subcomp_code in self.subcomp_models:
                    try:
                        subcomp_preds[subcomp_code] = self._predict_single(
                            self.subcomp_models[subcomp_code], target_date
                        )
                    except:
                        pass

        return subcomp_preds

    def _aggregate_subcomp_to_comp(self, subcomp_preds):
        """Aggregate subcomponent predictions to components."""
        comp_preds = {}

        for comp_name, comp_info in self.COMPONENTS.items():
            # Get subcomps belonging to this component
            subcomps_in_comp = [
                (code, pred)
                for code, pred in subcomp_preds.items()
                if self.subcomp_to_comp.get(code) == comp_name
            ]

            if subcomps_in_comp:
                # Weighted average
                total_weight = sum(
                    self.subcomp_weights.get(code, 0) for code, _ in subcomps_in_comp
                )
                if total_weight > 0:
                    weighted_sum = sum(
                        self.subcomp_weights.get(code, 0) * pred
                        for code, pred in subcomps_in_comp
                    )
                    comp_preds[comp_name] = weighted_sum / total_weight

        return comp_preds

    def _aggregate_comp_to_total(self, comp_preds):
        """Aggregate component predictions to total MoM."""
        if not comp_preds:
            return 0.0

        # Use actual component weights
        total_weight = sum(
            self.COMPONENTS[comp]["weight"] for comp in comp_preds.keys()
        )

        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            self.COMPONENTS[comp]["weight"] * pred for comp, pred in comp_preds.items()
        )

        return weighted_sum / total_weight

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Predict MoM for target date using hierarchical aggregation.

        Returns
        -------
        dict
            {
                'prediction': total MoM index (e.g., 100.5),
                'components': {comp_name: pred},
                'coverage': weight coverage percentage
            }
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        # 1. Get micro predictions
        micro_preds = {}
        for item_code, model_data in self.micro_models.items():
            try:
                pred = self._predict_single(model_data, target_date)
                micro_preds[item_code] = pred
            except:
                continue

        # 2. Aggregate to subcomponents (with fallback)
        subcomp_preds = self._aggregate_micro_to_subcomp(micro_preds, target_date)

        # 3. Aggregate to components
        comp_preds = self._aggregate_subcomp_to_comp(subcomp_preds)

        # 4. Aggregate to total
        total_pred = self._aggregate_comp_to_total(comp_preds)

        # Calculate coverage
        covered_weight = sum(
            self.subcomp_weights.get(k, 0) for k in subcomp_preds.keys()
        )

        return {
            "prediction": 100 + total_pred,
            "components": comp_preds,
            "subcomponents": subcomp_preds,
            "coverage": covered_weight * 100,
        }

    def forecast(self, horizon: Optional[int] = None) -> np.ndarray:
        """Generate forecast trajectory."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        h = horizon or self.horizon
        forecasts = []
        last_date = self.macro_df.index[-1]

        for i in range(h):
            target_date = last_date + pd.DateOffset(months=i + 1)
            pred = self.predict(None, target_date)
            forecasts.append(pred["prediction"] - 100)

        return np.array(forecasts)

    def get_detailed_forecast(self, target_date):
        """Get detailed forecast with all hierarchy levels."""
        result = self.predict(None, target_date)

        output = {
            "total": result["prediction"] - 100,
            "coverage": result["coverage"],
            "components": result["components"],
            "top_subcomponents": sorted(
                [
                    (k, v, self.subcomp_names.get(k, k))
                    for k, v in result["subcomponents"].items()
                ],
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:10],
        }

        return output

    def get_coverage_report(self):
        """Get detailed coverage report."""
        if not self._is_fitted:
            return {}

        report = {
            "micro_models": len(self.micro_models),
            "subcomp_models": len(self.subcomp_models),
            "micro_weight": sum(
                self.micro_weights.get(k, 0) for k in self.micro_models.keys()
            )
            * 100,
            "subcomp_weight": sum(
                self.subcomp_weights.get(k, 0) for k in self.subcomp_models.keys()
            )
            * 100,
            "coverage_by_component": {},
            "missing_subcomponents": [],
        }

        # Coverage by component
        for comp_name in self.COMPONENTS.keys():
            subcomps = [
                k
                for k in self.subcomp_models.keys()
                if self.subcomp_to_comp.get(k) == comp_name
            ]
            weight = sum(self.subcomp_weights.get(k, 0) for k in subcomps) * 100
            report["coverage_by_component"][comp_name] = {
                "subcomponents": len(subcomps),
                "weight": weight,
            }

        # Missing subcomponents
        for code, weight in self.subcomp_weights.items():
            if code not in self.subcomp_models and not code.startswith("service_"):
                report["missing_subcomponents"].append(
                    {
                        "code": code,
                        "name": self.subcomp_names.get(code, code),
                        "weight": weight * 100,
                    }
                )

        return report
