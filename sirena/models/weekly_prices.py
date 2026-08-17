"""
Weekly Price Nowcaster for CPI Inflation
==========================================

Uses actual weekly prices from Rosstat to nowcast monthly CPI inflation.
Combines high-quality product price signals with macro variables.

Key features:
- Uses 22 high-quality products with <5% missing data
- Weighted aggregation matching CPI basket weights
- Combines with macro signals (Brent, USD, Ki)
- Target MAE: <0.10%

Training period: 2016-2026 (post-Crimea, stable methodology)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from typing import Any, Optional, Dict, List, Tuple, Union, cast
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def _as_timestamp(value: Any) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


def _index_month_numbers(index: Any) -> pd.Series:
    return pd.Series(pd.to_datetime(index), index=index).dt.month


def _index_year_numbers(index: Any) -> pd.Series:
    return pd.Series(pd.to_datetime(index), index=index).dt.year


try:
    from ..data.weekly_loader import (
        load_weekly_prices,
        get_high_quality_products,
        compute_basket_signal,
        HIGH_QUALITY_PRODUCTS,
        COMPONENT_WEIGHTS,
    )
    from ..models.regime_detector import detect_regime, MacroRegime
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from sirena.data.weekly_loader import (
        load_weekly_prices,
        get_high_quality_products,
        compute_basket_signal,
        HIGH_QUALITY_PRODUCTS,
        COMPONENT_WEIGHTS,
    )
    from sirena.models.regime_detector import detect_regime, MacroRegime


class WeeklyPriceNowcaster:
    """
    Nowcasting model using actual weekly prices.

    This model combines:
    1. Weekly price signals from high-quality products
    2. Lagged monthly CPI values
    3. Seasonal effects
    4. Optional: Macro signals (Brent, USD, Ki)
    5. NEW: Regime-specific weights (use_regime=True)

    Performance:
        Target MAE: <0.10%
        Backtest period: 2024-2025

    Regime Switching (NEW):
        - SHOCK: Equal weighting (all products ~11%)
        - HIGH_INFLATION: Weights focused on volatile products
        - NORMAL: Product-specific weights based on historical performance
        - Detection: Uses Ki/Ruonia changes and inflation YoY acceleration
    """

    MIN_TRAIN_SIZE = 24
    REGIME_WEIGHTS_FILE = "data/weekly_regime_weights.csv"

    def __init__(
        self,
        alpha: float = 1.0,
        use_macro: bool = True,
        use_components: bool = True,
        regressor: str = "ridge",
        regime_switching: bool = True,
        weighting: str = "basket",
        use_regime: Optional[bool] = None,
        regime_weights_path: Optional[str] = None,
    ):
        """
        Initialize nowcaster.

        Args:
            alpha: Regularization strength
            use_macro: Include macro signals (Brent, USD, Ki)
            use_components: Use food/nonfood component breakdown
            regressor: 'ridge' or 'huber'
            regime_switching: Enable regime-dependent weights (Task 567)
            weighting: Product weighting method for non-regime signals
            use_regime: Backward-compatible alias for regime_switching
            regime_weights_path: Optional path to regime-specific weights
        """
        self.alpha = alpha
        self.use_macro = use_macro
        self.use_components = use_components
        self.regressor = regressor
        self.weighting = weighting
        self.regime_switching = regime_switching if use_regime is None else use_regime
        self.use_regime = self.regime_switching

        self.model: Optional[Union[Ridge, HuberRegressor]] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self._is_fitted = False
        self.training_stats: Dict[str, Any] = {}

        # Regime-specific weights {regime: {product_code: weight}}
        self.regime_weights: Dict[str, Dict[int, float]] = {}
        self._current_regime: str = "normal"
        self._regime_diagnostics: Dict = {}
        self._df: Optional[pd.DataFrame] = None

        # Path to regime weights (from Task 414)
        self.regime_weights_path = regime_weights_path or str(
            Path(__file__).parent.parent.parent / "data" / "weekly_regime_weights.csv"
        )
        if self.regime_switching:
            self._load_regime_weights()

    def _load_regime_weights(self):
        """Load regime-specific product weights from Task 414 backtest results."""
        try:
            weights_df = pd.read_csv(self.regime_weights_path)

            # Group by regime
            for regime in weights_df["Regime"].unique():
                regime_data = weights_df[weights_df["Regime"] == regime]
                weights_dict = {}
                for _, row in regime_data.iterrows():
                    weights_dict[row["Product_code"]] = row["Weight"]
                self.regime_weights[regime] = weights_dict

        except Exception as e:
            warnings.warn(
                f"Failed to load regime weights: {e}. Using default equal weights."
            )
            # Fallback to equal weights
            self.regime_weights = {
                "normal": {},
                "shock": {},
                "high_inflation": {},
            }

    def detect_regime(self, df: Optional[pd.DataFrame] = None) -> Tuple[str, Dict]:
        """
        Detect current macroeconomic regime.

        Uses detect_regime() from regime_detector.py which analyzes:
        - Ki (key rate) changes
        - Ruonia changes
        - Inflation YoY acceleration

        Args:
            df: DataFrame with macro indicators (Ki, Ruonia, inflation)
                If None, uses data from fit()

        Returns:
            Tuple[str, Dict]: (regime_name, diagnostics_dict)

            regime_name: One of 'normal', 'shock', 'high_inflation'
            diagnostics: Dictionary with diagnostic values (ki_change, ruonia_change, etc.)
        """
        if df is None:
            if self._df is not None:
                df = self._df
            else:
                raise ValueError(
                    "No data available for regime detection. Provide df parameter or call fit() first."
                )

        # Use regime detector from regime_detector.py
        # Convert from MacroRegime enum to string
        macro_regime, diagnostics = detect_regime(
            df,
            ki_col="Ki_i",
            ruonia_col="Ruonia",
            mom_col="mom",
        )

        regime_name = macro_regime.value
        return regime_name, diagnostics

    def _compute_regime_weighted_signal(self, month_weeks: pd.DataFrame) -> float:
        """
        Compute regime-weighted signal for a month.

        Signal = Σ(wow_growth_i * weight_i)
        where weight_i depends on current regime.

        Args:
            month_weeks: Weekly data for one month

        Returns:
            Weighted average WoW growth using regime-specific weights
        """
        signals = []

        # Get weights for current regime
        current_weights = self.regime_weights.get(
            self._current_regime, self.regime_weights.get("normal", {})
        )

        if not current_weights:
            # Fallback: equal weights for all products
            products = month_weeks["product_code"].unique()
            n_products = len(products)
            current_weights = {code: 1.0 / n_products for code in products}

        for prod_code, weight in current_weights.items():
            prod_data = month_weeks[month_weeks["product_code"] == prod_code]

            if len(prod_data) == 0:
                continue

            # Average WoW growth for this product in month
            avg_growth = float(prod_data["wow_growth"].mean())

            if bool(pd.isna(avg_growth)):
                continue

            signals.append(avg_growth * weight)

        if not signals:
            return 0.0

        return sum(signals)

    def _compute_regime_weighted_signal_detailed(
        self, month_weeks: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute regime-weighted signal with component breakdown.

        Returns same format as compute_basket_signal() from weekly_loader.
        Compatible with nowcast() output requirements.

        Args:
            month_weeks: Weekly data for one month

        Returns:
            Dict with keys: signal, food_signal, nonfood_signal,
                         n_products, coverage
        """
        # Get weights for current regime
        current_weights = self.regime_weights.get(
            self._current_regime, self.regime_weights.get("normal", {})
        )

        if not current_weights:
            # Fallback: equal weights for all products
            products = month_weeks["product_code"].unique()
            n_products = len(products)
            current_weights = {code: 1.0 / n_products for code in products}

        # Compute signals per category
        food_sum = 0.0
        food_weight = 0.0
        nonfood_sum = 0.0
        nonfood_weight = 0.0
        overall_sum = 0.0
        total_weight = 0.0
        n_products = 0

        for prod_code, weight in current_weights.items():
            prod_data = month_weeks[month_weeks["product_code"] == prod_code]

            if len(prod_data) == 0:
                continue

            # Average WoW growth for this product in month
            avg_growth = float(prod_data["wow_growth"].mean())

            if bool(pd.isna(avg_growth)):
                continue

            n_products += 1

            # Get category from HIGH_QUALITY_PRODUCTS
            info = HIGH_QUALITY_PRODUCTS.get(prod_code, {})
            category = info.get("category", "nonfood")

            # Accumulate weighted signal
            weighted_signal = avg_growth * weight
            overall_sum += weighted_signal
            total_weight += weight

            if category == "food":
                food_sum += weighted_signal
                food_weight += weight
            else:
                nonfood_sum += weighted_signal
                nonfood_weight += weight

        # Compute final signals
        overall_signal = overall_sum / total_weight if total_weight > 0 else 0.0
        food_signal = food_sum / food_weight if food_weight > 0 else 0.0
        nonfood_signal = nonfood_sum / nonfood_weight if nonfood_weight > 0 else 0.0

        # Coverage: % of regime-weighted products with data
        expected_products = len(current_weights)
        coverage = n_products / expected_products if expected_products > 0 else 0.0

        return {
            "signal": overall_signal,
            "food_signal": food_signal,
            "nonfood_signal": nonfood_signal,
            "n_products": n_products,
            "coverage": coverage,
        }

    def _load_monthly_cpi(self) -> pd.DataFrame:
        """Load monthly CPI data."""
        base_paths = [
            Path.cwd() / "data" / "inflation_data.csv",
            Path.cwd().parent / "data" / "inflation_data.csv",
            Path(__file__).parent.parent.parent / "data" / "inflation_data.csv",
        ]

        for path in base_paths:
            if path.exists():
                df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
                df["Date"] = pd.to_datetime(
                    df["Date"], format="%d.%m.%Y", dayfirst=True
                )

                # Normalize to first day of month for joining with weekly data
                df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
                df = df.set_index("Date").sort_index()

                # Convert string columns to float
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].str.replace(",", ".").astype(float)

                # Add brent prices if available
                brent_path = path.parent / "brent_prices.csv"
                if brent_path.exists():
                    brent = pd.read_csv(brent_path)
                    brent["Date"] = (
                        pd.to_datetime(brent["Date"])
                        .dt.to_period("M")
                        .dt.to_timestamp()
                    )
                    brent = brent.set_index("Date")
                    df = df.join(brent[["brent"]], how="left")

                return df

        raise FileNotFoundError("Monthly CPI data not found")

    def _aggregate_weekly_to_monthly(
        self,
        weekly_df: pd.DataFrame,
        monthly_cpi: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Aggregate weekly prices to monthly signals.

        If use_regime=True, uses regime-specific weights.
        Otherwise, uses standard weighting (volatility/basket/equal).

        Returns DataFrame with monthly signals per component.
        """
        # Add year_month column
        weekly_df = weekly_df.copy()
        weekly_df["year_month"] = weekly_df["date"].dt.to_period("M")

        # Filter to high-quality products
        hq_codes = list(HIGH_QUALITY_PRODUCTS.keys())
        weekly_df = cast(
            pd.DataFrame,
            weekly_df[cast(pd.Series, weekly_df["product_code"]).isin(hq_codes)].copy(),
        )

        monthly_signals = []

        # Pre-calculate macro changes for regime detection if possible
        ki_diffs = {}
        if monthly_cpi is not None and 'Ki_i' in monthly_cpi.columns:
            ki_series = monthly_cpi['Ki_i'].diff().abs()
            ki_diffs = ki_series.to_dict()

        for period in weekly_df['year_month'].unique():
            month_data = cast(
                pd.DataFrame,
                weekly_df[cast(pd.Series, weekly_df['year_month']) == period].copy(),
            )
            target_date = period.to_timestamp()
            
            # Detect Regime
            regime = 'normal'
            if self.regime_switching and target_date in ki_diffs:
                if ki_diffs[target_date] > 0.5:
                    regime = 'shock'
            
            # Get weights
            weights = None
            if self.regime_switching and regime in self.regime_weights:
                weights = self.regime_weights[regime]
            
            # Compute signal
            if weights:
                signal = self._compute_custom_signal(month_data, weights)
            else:
                # Fallback to standard loader logic (basket or equal)
                signal = compute_basket_signal(month_data)

            monthly_signals.append({
                'date': target_date,
                'weekly_signal': signal['signal'],
                'food_signal': signal['food_signal'],
                'nonfood_signal': signal['nonfood_signal'],
                'coverage': signal['coverage'],
                'n_products': signal['n_products'],
                'regime': regime
            })

        return pd.DataFrame(monthly_signals).set_index('date').sort_index()

    def _compute_custom_signal(self, df: pd.DataFrame, weights: Dict[int, float]) -> Dict:
        """Compute signal using custom weight vector."""
        product_signals = cast(pd.Series, df.groupby('product_code')['wow_growth'].mean())
        
        weighted_sum = 0.0
        weight_sum = 0.0
        food_sum = 0.0
        food_w = 0.0
        nonfood_sum = 0.0
        nonfood_w = 0.0
        
        for raw_code, raw_sig in product_signals.items():
            code = cast(int, raw_code)
            sig = float(raw_sig)
            if bool(pd.isna(sig)): continue
            
            w = weights.get(code, 0.0)
            if w <= 0: continue
            
            info = HIGH_QUALITY_PRODUCTS.get(code, {})
            cat = info.get('category', 'nonfood')
            
            weighted_sum += sig * w
            weight_sum += w
            
            if cat == 'food':
                food_sum += sig * w
                food_w += w
            else:
                nonfood_sum += sig * w
                nonfood_w += w
                
        return {
            'signal': weighted_sum / weight_sum if weight_sum > 0 else np.nan,
            'food_signal': food_sum / food_w if food_w > 0 else np.nan,
            'nonfood_signal': nonfood_sum / nonfood_w if nonfood_w > 0 else np.nan,
            'n_products': len(product_signals),
            'coverage': len(product_signals) / len(HIGH_QUALITY_PRODUCTS)
        }

    def _create_features(
        self,
        monthly_cpi: pd.DataFrame,
        weekly_signals: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create feature matrix for training/prediction.
        
        Features:
        - Weekly price signals (overall, food, nonfood)
        - Lagged CPI values
        - Seasonal dummies
        - Optional: Macro variables
        """
        # Merge weekly signals with monthly CPI
        df = monthly_cpi.copy()
        df.index = pd.to_datetime(df.index)

        # Недельные данные опережают месячный ИПЦ: текущий месяц уже наблюдается
        # по неделям, но месячного значения ещё нет. Без этого расширения
        # nowcast откатывался на последний закрытый месяц и переставал быть
        # nowcast'ом. Продлеваем индекс только вперёд, историю не трогаем.
        weekly_index = pd.to_datetime(weekly_signals.index)
        future_months = weekly_index[weekly_index > df.index.max()]
        if len(future_months):
            df = df.reindex(df.index.union(future_months))

        df = df.join(weekly_signals, how='left')

        # Target: MoM inflation (subtract 100 to get percentage)
        df['y_target'] = df['mom'] - 100 if 'mom' in df.columns else np.nan

        # Weekly signal features
        df['weekly_L0'] = df['weekly_signal']
        df['weekly_L1'] = df['weekly_signal'].shift(1)
        df['food_L0'] = df['food_signal']
        df['food_L1'] = df['food_signal'].shift(1)
        df['nonfood_L0'] = df['nonfood_signal']
        df['nonfood_L1'] = df['nonfood_signal'].shift(1)

        # Momentum
        df['weekly_mom'] = df['weekly_signal'] - df['weekly_signal'].shift(1)

        # Lagged CPI
        if 'mom' in df.columns:
            y = df['mom'] - 100
            df['y_L1'] = y.shift(1)
            df['y_L2'] = y.shift(2)
            df['y_L3'] = y.shift(3)
            df['y_L12'] = y.shift(12)
            df['y_ma3'] = y.rolling(3).mean()

        # Seasonal features
        month_numbers = _index_month_numbers(df.index)
        df['month_sin'] = np.sin(2 * np.pi * month_numbers / 12)
        df['month_cos'] = np.cos(2 * np.pi * month_numbers / 12)
        df['is_jan'] = (month_numbers == 1).astype(int)
        df['is_jul'] = (month_numbers == 7).astype(int)  # Tariff adjustments
        df['is_dec'] = (month_numbers == 12).astype(int)

        # Macro features (if available and requested)
        if self.use_macro:
            if 'brent' in df.columns:
                df['brent_L1'] = df['brent'].shift(1)
                df['brent_pct'] = df['brent'].pct_change()
            if 'usd_nom_i' in df.columns:
                df['usd_L1'] = df['usd_nom_i'].shift(1)
                df['usd_pct'] = df['usd_nom_i'].pct_change()
            if 'Ki_i' in df.columns:
                df['ki_L1'] = df['Ki_i'].shift(1)
                df['ki_diff'] = df['Ki_i'].diff()

        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns to use."""
        feature_cols = []

        # Weekly signals (most important)
        weekly_cols = ['weekly_L0', 'weekly_L1', 'weekly_mom']
        if self.use_components:
            weekly_cols += ['food_L0', 'food_L1', 'nonfood_L0', 'nonfood_L1']
        feature_cols += [c for c in weekly_cols if c in df.columns]

        # Lagged CPI
        cpi_cols = ['y_L1', 'y_L2', 'y_L3', 'y_L12', 'y_ma3']
        feature_cols += [c for c in cpi_cols if c in df.columns]

        # Seasonal
        seasonal_cols = ['month_sin', 'month_cos', 'is_jan', 'is_jul', 'is_dec']
        feature_cols += [c for c in seasonal_cols if c in df.columns]

        # Macro (optional)
        if self.use_macro:
            macro_cols = ['brent_L1', 'brent_pct', 'usd_L1', 'usd_pct', 'ki_L1', 'ki_diff']
            feature_cols += [c for c in macro_cols if c in df.columns]

        return feature_cols

    def fit(
        self,
        start_date: str = '2016-01-01',
        end_date: Optional[str] = None,
        exclude_years: Optional[List[int]] = None,
    ) -> 'WeeklyPriceNowcaster':
        """
        Fit the nowcaster model.
        """
        exclude_years = exclude_years or [2022]

        # Load data
        weekly_df = load_weekly_prices(start_date=start_date, end_date=end_date)
        monthly_cpi = self._load_monthly_cpi()

        # Aggregate weekly to monthly signals
        weekly_signals = self._aggregate_weekly_to_monthly(weekly_df, monthly_cpi)

        # Create features
        df = self._create_features(monthly_cpi, weekly_signals)
        df = cast(pd.DataFrame, df)
        df.index = pd.to_datetime(df.index)

        # Filter to training period
        df = cast(pd.DataFrame, df[df.index >= pd.Timestamp(start_date)].copy())
        if end_date:
            df = cast(pd.DataFrame, df[df.index <= pd.Timestamp(end_date)].copy())

        # Exclude outlier years
        if exclude_years:
            year_mask = ~_index_year_numbers(df.index).isin(exclude_years)
            df = cast(pd.DataFrame, df[year_mask].copy())

        # Drop rows with missing target or features
        self.feature_names = self._get_feature_columns(df)
        train_df = cast(
            pd.DataFrame,
            df.dropna(subset=["y_target"] + self.feature_names),
        )

        if len(train_df) < self.MIN_TRAIN_SIZE:
            raise ValueError(
                f"Insufficient training data: {len(train_df)} < {self.MIN_TRAIN_SIZE}"
            )

        # Extract features and target
        X = train_df[self.feature_names].values
        y = train_df["y_target"].values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        if self.regressor == "huber":
            self.model = HuberRegressor(alpha=self.alpha, max_iter=1000)
        else:
            self.model = Ridge(alpha=self.alpha, random_state=42)

        self.model.fit(X_scaled, y)

        # Store training stats
        self.training_stats = {
            "n_samples": len(train_df),
            "date_range": (
                str(_as_timestamp(pd.to_datetime(train_df.index).min()).date()),
                str(_as_timestamp(pd.to_datetime(train_df.index).max()).date()),
            ),
            "n_features": len(self.feature_names),
            "features": self.feature_names,
        }

        self._is_fitted = True
        return self

    def nowcast(
        self,
        target_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, float]:
        """
        Generate nowcast for a specific month.

        Args:
            target_date: Month to nowcast (default: current month)

        Returns:
            Dict with:
                - prediction: Nowcast MoM inflation (%)
                - weekly_signal: Weekly price signal used
                - components: Food/nonfood breakdown
                - confidence: Confidence based on data coverage
                - regime: Current regime (if use_regime=True)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.scaler is None or self.model is None:
            raise ValueError("Model state is incomplete. Call fit() again.")

        # Load current data
        weekly_df = load_weekly_prices()
        monthly_cpi = self._load_monthly_cpi()

        # Determine target date
        if target_date is None:
            target_date = _as_timestamp(pd.to_datetime(cast(pd.Series, weekly_df["date"])).max())

        target_month = _as_timestamp(target_date).replace(day=1)

        # Detect current regime if using regime switching
        if self.use_regime:
            # Update regime based on latest data
            current_regime, diagnostics = self.detect_regime(monthly_cpi)
            self._current_regime = current_regime
            self._regime_diagnostics = diagnostics

        # Aggregate weekly signals
        weekly_signals = self._aggregate_weekly_to_monthly(weekly_df)

        # Create features
        df = self._create_features(monthly_cpi, weekly_signals)
        df = cast(pd.DataFrame, df)
        df.index = pd.to_datetime(df.index)

        # Get features for target month
        if target_month not in df.index:
            # Use last available data
            target_month = _as_timestamp(pd.to_datetime(df.index).max())

        row = df.loc[[target_month], self.feature_names].fillna(0)
        X = row.values

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]

        # Get weekly signal for context - use regime weighting if enabled
        week_months = cast(pd.Series, weekly_df["date"]).dt.to_period("M")
        month_data = cast(
            pd.DataFrame,
            weekly_df[week_months == _as_timestamp(target_month).to_period("M")].copy(),
        )

        if self.use_regime:
            # Use regime-specific weights
            signal_info = self._compute_regime_weighted_signal_detailed(month_data)
        else:
            # Use standard weighting
            signal_info = compute_basket_signal(month_data, weighting=self.weighting)

        # Модель обучена на ПОЛНЫХ месяцах. Если целевой месяц ещё набирается,
        # недельный сигнал несопоставим с обучающим, и прогноз недостоверен.
        weeks_observed = int(month_data["date"].nunique()) if len(month_data) else 0
        partial_month = 0 < weeks_observed < 4

        # Build result
        result = {
            "prediction": prediction,
            "target_date": str(_as_timestamp(target_month).date()),
            "weekly_signal": signal_info["signal"],
            "food_signal": signal_info["food_signal"],
            "nonfood_signal": signal_info["nonfood_signal"],
            "coverage": signal_info["coverage"],
            "n_products": signal_info["n_products"],
            "weeks_observed": weeks_observed,
            "partial_month": partial_month,
            "confidence": (
                "low (неполный месяц)"
                if partial_month
                else ("high" if signal_info["coverage"] > 0.8 else "medium")
            ),
        }
        if partial_month:
            result["warning"] = (
                f"Целевой месяц наблюдается лишь {weeks_observed} нед. из ~4; "
                "модель обучена на полных месяцах. Использовать "
                "scripts/weekly_calibrated_nowcast.py."
            )

        # Add regime info if using regime switching
        if self.use_regime:
            result["regime"] = self._current_regime

        return result

    def backtest(
        self,
        start_date: str = "2024-01-01",
        end_date: Optional[str] = None,
        exclude_years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Run backtest on historical data.

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            exclude_years: Years to exclude from training

        Returns:
            DataFrame with columns: date, actual, prediction, error, weekly_signal
        """
        exclude_years = exclude_years or [2022]

        # Load all data
        weekly_df = load_weekly_prices()
        monthly_cpi = self._load_monthly_cpi()
        weekly_signals = self._aggregate_weekly_to_monthly(weekly_df)
        df = self._create_features(monthly_cpi, weekly_signals)
        df = cast(pd.DataFrame, df)
        df.index = pd.to_datetime(df.index)

        # Get test dates
        test_start = pd.Timestamp(start_date)
        test_end = pd.Timestamp(end_date) if end_date else df.index.max()
        test_dates = df[(df.index >= test_start) & (df.index <= test_end)].index

        results = []

        for target_date in test_dates:
            # Train on data before target_date
            train_df = df[df.index < target_date].copy()
            train_df = cast(pd.DataFrame, train_df)

            if exclude_years:
                year_mask = ~_index_year_numbers(train_df.index).isin(exclude_years)
                train_df = cast(pd.DataFrame, train_df[year_mask].copy())

            # Get feature columns available in training data
            feature_cols = self._get_feature_columns(train_df)
            train_clean = cast(
                pd.DataFrame,
                train_df.dropna(subset=["y_target"] + feature_cols),
            )

            if len(train_clean) < self.MIN_TRAIN_SIZE:
                continue

            # Train model
            X_train = train_clean[feature_cols].values
            y_train = train_clean["y_target"].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            if self.regressor == "huber":
                model = HuberRegressor(alpha=self.alpha, max_iter=1000)
            else:
                model = Ridge(alpha=self.alpha, random_state=42)

            model.fit(X_train_scaled, y_train)

            # Predict for target date
            target_row = df.loc[[target_date], feature_cols].fillna(0)
            X_test = scaler.transform(target_row.values)
            prediction = model.predict(X_test)[0]

            # Get actual value
            actual = (
                df.loc[target_date, "y_target"] if "y_target" in df.columns else np.nan
            )

            # Get weekly signal
            weekly_signal = (
                df.loc[target_date, "weekly_signal"]
                if "weekly_signal" in df.columns
                else np.nan
            )

            results.append(
                {
                    "date": target_date,
                    "actual": actual,
                    "prediction": prediction,
                    "error": actual - prediction if not np.isnan(actual) else np.nan,
                    "weekly_signal": weekly_signal,
                }
            )

        return pd.DataFrame(results)

    def get_metrics(self, backtest_results: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results.

        Args:
            backtest_results: DataFrame from backtest() method

        Returns:
            Dict with MAE, RMSE, KPI violations, etc.
        """
        errors = backtest_results["error"].dropna()

        if len(errors) == 0:
            return {"mae": np.nan, "rmse": np.nan, "n_samples": 0}

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        kpi_violations = np.sum(np.abs(errors) > 0.5)
        kpi_rate = kpi_violations / len(errors)

        return {
            "mae": float(round(mae, 4)),
            "rmse": float(round(rmse, 4)),
            "kpi_violations": int(kpi_violations),
            "kpi_rate": float(round(kpi_rate, 3)),
            "n_samples": len(errors),
            "mean_error": float(round(np.mean(errors), 4)),
            "std_error": float(round(np.std(errors), 4)),
        }

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from fitted model.

        Returns:
            DataFrame with feature names and coefficients
        """
        if not self._is_fitted or self.model is None:
            return None

        importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient": self.model.coef_,
            }
        )
        importance["abs_coef"] = importance["coefficient"].abs()

        return importance.sort_values("abs_coef", ascending=False)

    @property
    def current_regime(self) -> str:
        """Get current detected regime (normal/shock/high_inflation)."""
        return self._current_regime

    def get_regime_diagnostics(self) -> Dict:
        """
        Get regime detection diagnostics.

        Returns:
            Dictionary with diagnostic values (ki_change, ruonia_change, yoy_change, etc.)
        """
        return self._regime_diagnostics.copy()


def run_backtest_comparison():
    """
    Compare WeeklyPriceNowcaster against baseline.
    """
    print("=" * 60)
    print("Weekly Price Nowcaster Backtest")
    print("=" * 60)

    # Test different configurations
    configs = [
        {
            "name": "Ridge + Macro + Components",
            "use_macro": True,
            "use_components": True,
            "regressor": "ridge",
        },
        {
            "name": "Ridge + Weekly Only",
            "use_macro": False,
            "use_components": False,
            "regressor": "ridge",
        },
        {
            "name": "Huber + Macro + Components",
            "use_macro": True,
            "use_components": True,
            "regressor": "huber",
        },
    ]

    results = []

    for config in configs:
        print(f"\n{config['name']}:")

        model = WeeklyPriceNowcaster(
            use_macro=config["use_macro"],
            use_components=config["use_components"],
            regressor=config["regressor"],
        )

        try:
            model.fit()
            backtest = model.backtest(start_date="2024-01-01")
            metrics = model.get_metrics(backtest)

            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(
                f"  KPI violations: {metrics['kpi_violations']}/{metrics['n_samples']}"
            )

            results.append(
                {
                    "config": config["name"],
                    **metrics,
                }
            )
        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("Summary:")
        best = min(results, key=lambda x: x["mae"])
        print(f"Best config: {best['config']} (MAE: {best['mae']:.4f})")

    return results


if __name__ == "__main__":
    run_backtest_comparison()
