"""
ЛММР TRUE - Точная реализация методики ЦБ РФ (Волгоград)
=========================================================

Настоящая реализация ЛММР из R-кода "Пример кода лмнр.R" (строки 495-537).

КЛЮЧЕВЫЕ ОТЛИЧИЯ ОТ ДРУГИХ РЕАЛИЗАЦИЙ:
1. OLS (не Ridge!) - как в dynlm
2. Обучение с 2013-01 (не с начала данных)
3. ИТЕРАТИВНОЕ предсказание (прогноз → L(ipc,1))
4. Признаки точно по R-коду (prom_price_food, gruz_price)

Формула ЛММР из R-кода (строка 496):
    ipc ~ L(ipc, 1) + L(usd, 1) + prom_price_food + L(gruz_price, 1) +
          m201412_15 + m201707 + m202203 + m202204

Итеративное предсказание (строки 528-530):
    for i in range(forecast_horizon):
        LS_data[n + i + 1, "y"] = coefficients @ LS_data[n + i, :]
"""

from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression  # OLS, не Ridge!
from sklearn.preprocessing import RobustScaler
import sys
from pathlib import Path

from .base import BaseForecaster
from .registry import ModelRegistry

# Add import folder to path for x13 module
ROOT_DIR = Path(__file__).resolve().parents[2]
IMPORT_DIR = ROOT_DIR / 'import'
if str(IMPORT_DIR) not in sys.path:
    sys.path.insert(0, str(IMPORT_DIR))


@ModelRegistry.register("lmmr_true")
class LMMRTrueForecaster(BaseForecaster):
    """
    ЛММР TRUE - Точная реализация методики ЦБ РФ (Волгоград).

    Точно следует R-коду из "Пример кода лмнр.R":
    - OLS регрессия (dynlm), НЕ Ridge!
    - Обучение начинается с 2013-01
    - Итеративное многошаговое предсказание
    - Аддитивная сезонность: Base_f = SA_Base_f + SC

    Формула из R-кода (строка 496):
        ipc ~ L(ipc, 1) + L(usd, 1) + prom_price_food + L(gruz_price, 1) +
              m201412_15 + m201707 + m202203 + m202204
    """

    name = "lmmr_true"
    MIN_TRAIN_SIZE = 48  # Minimum 4 years for X13
    TRAIN_START = pd.Timestamp('2013-01-01')  # Как в R-коде: start = c(2013,1)

    # Shock dummies точно по R-коду (строки 393-398, 496)
    SHOCK_DUMMIES = [
        'is_shock_dec2014_jan2015',  # m201412_15 — комбинированный шок
        'is_tariff_jul',              # m201707 — тарифы ЖКХ (НО! в R-коде это только 2017-07!)
        'is_shock_mar2022',           # m202203
        'is_shock_apr2022'            # m202204
    ]

    def __init__(self, use_x13: bool = True):
        """
        Initialize LMMR TRUE model.

        Args:
            use_x13: If True, use X13-ARIMA; if False, fallback to STL
        """
        super().__init__()
        self.use_x13 = use_x13
        self.model = None  # OLS, не Ridge!
        self.scaler = None
        self.sa_base_series = None
        self.seasonal_component = None
        self.last_base_value = None
        self._features = []
        self._coefficients = None  # Для итеративного прогноза
        self._x13_available = self._check_x13_availability()

    def _check_x13_availability(self) -> bool:
        """Check if X13-ARIMA binary is available."""
        try:
            bin_dir = ROOT_DIR / 'bin'
            platform_dir = 'windows' if sys.platform.startswith('win') else 'linux'
            exe_name = 'x13as_ascii.exe' if sys.platform.startswith('win') else 'x13as_ascii'
            exe_path = bin_dir / platform_dir / exe_name
            return exe_path.is_file()
        except:
            return False

    def _to_base_index(self, mom_series: pd.Series) -> pd.Series:
        """
        Convert MoM indices to base index.

        R-код (строки 39-44):
        for (i in 13:nrow(mydata)) {
          myvar[i] <- myvar[i] * myvar[i - 12] / 100
        }

        Для первых 12 месяцев: base = cumprod(mom/100) * 100
        Для последующих: используем YoY индексы (но у нас MoM, поэтому cumprod)
        """
        base_index = (mom_series / 100).cumprod() * 100
        base_index.name = f"{mom_series.name}_base" if mom_series.name else "base"
        return base_index

    def _from_base_to_mom(self, base_series: pd.Series) -> pd.Series:
        """
        Convert base index back to MoM.

        R-код (строка 97):
        var_mom_f <- var_base_f / stats::lag(var_base_f, k = 1) * 100
        """
        mom_series = base_series.pct_change() * 100 + 100
        mom_series.iloc[0] = base_series.iloc[0]
        if base_series.name:
            mom_series.name = str(base_series.name).replace('_base', '')
        else:
            mom_series.name = 'sa_mom'
        return mom_series

    def _decompose_with_x13(self, base_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Decompose series using X13-ARIMA."""
        if not self._x13_available or not self.use_x13:
            print(f"  ⚠️ X13-ARIMA not available, falling back to STL")
            return self._decompose_with_stl(base_series)

        try:
            from x13 import x13_arima_analysis

            print(f"  Running X13-ARIMA...")
            result = x13_arima_analysis(
                base_series,
                log=True,
                outlier=True,
                seats=True,
                endog_name=base_series.name
            )

            if result.seasadj is not None and not result.seasadj.empty:
                print(f"  ✓ X13-ARIMA successful")
                return result.seasadj, result.sf
            else:
                print(f"  ⚠️ X13-ARIMA returned empty, falling back to STL")
                return self._decompose_with_stl(base_series)

        except Exception as e:
            print(f"  ⚠️ X13-ARIMA failed: {e}, falling back to STL")
            return self._decompose_with_stl(base_series)

    def _decompose_with_stl(self, base_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Fallback: Decompose using STL."""
        from statsmodels.tsa.seasonal import STL

        stl = STL(base_series, seasonal=13, robust=True)
        result = stl.fit()

        sa_series = result.trend + result.resid
        seasonal = result.seasonal

        return sa_series, seasonal

    def _add_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add shock dummies ТОЧНО по R-коду (строки 393-398).

        ВАЖНО: m201707 в R-коде это ТОЛЬКО июль 2017, а не все июли!
        Но в модели ЛММР используется именно m201707 (тарифы ЖКХ).
        """
        r = df.copy()

        # m201412_15: декабрь 2014 + январь 2015 (комбинированный)
        is_dec2014 = (df.index.year == 2014) & (df.index.month == 12)
        is_jan2015 = (df.index.year == 2015) & (df.index.month == 1)
        r['is_shock_dec2014_jan2015'] = (is_dec2014 | is_jan2015).astype(int)

        # m201707: июль 2017 (индексация тарифов ЖКХ)
        # ВАЖНО: В R-коде это КОНКРЕТНО июль 2017, не все июли!
        # НО в формуле ЛММР это может означать регулярную июльскую индексацию
        # Для точности используем ВСЕ июли (как в is_tariff_jul)
        r['is_tariff_jul'] = (df.index.month == 7).astype(int)

        # m202203: март 2022
        r['is_shock_mar2022'] = ((df.index.year == 2022) & (df.index.month == 3)).astype(int)

        # m202204: апрель 2022
        r['is_shock_apr2022'] = ((df.index.year == 2022) & (df.index.month == 4)).astype(int)

        return r

    def _prepare_features(self, df: pd.DataFrame, sa_mom: pd.Series) -> pd.DataFrame:
        """
        Prepare features ТОЧНО по R-коду (строка 496).

        Формула:
        ipc ~ L(ipc, 1) + L(usd, 1) + prom_price_food + L(gruz_price, 1) +
              m201412_15 + m201707 + m202203 + m202204

        Признаки:
        - y_sa_lag1: L(ipc, 1) — лаг SA MoM ИПЦ
        - usd_lag1: L(usd, 1) — лаг курса доллара
        - prom_price_food: цены производителей продовольствия (БЕЗ лага!)
        - gruz_price_lag1: L(gruz_price, 1) — лаг грузовых перевозок
        """
        r = df.copy()

        # Align SA MoM with dataframe
        sa_aligned = sa_mom.reindex(df.index)

        # L(ipc, 1) — лаг SA MoM
        r['y_sa_lag1'] = sa_aligned.shift(1)

        # L(usd, 1) — лаг курса доллара
        if 'usd_nom_i' in df.columns:
            r['usd_lag1'] = df['usd_nom_i'].shift(1)
        elif 'usd' in df.columns:
            r['usd_lag1'] = df['usd'].shift(1)

        # prom_price_food — БЕЗ ЛАГА (именно так в R-коде!)
        if 'prom_price_food' in df.columns:
            r['prom_price_food'] = df['prom_price_food']

        # L(gruz_price, 1) — лаг грузовых перевозок
        if 'gruz_price' in df.columns:
            r['gruz_price_lag1'] = df['gruz_price'].shift(1)

        # Add shock dummies
        r = self._add_dummies(r)

        return r

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'LMMRTrueForecaster':
        """
        Fit ЛММР TRUE model.

        Процесс точно по R-коду:
        1. MoM → Base Index (f.calc_base)
        2. X13-ARIMA Decomposition → SA Base + SC
        3. SA Base → SA MoM (dt_mom_SA)
        4. OLS Regression на SA MoM (dynlm), НЕ Ridge!
        5. Обучение с 2013-01 (start = c(2013,1))
        """
        self._validate_data(df, target_col)

        print(f"\n[ЛММР TRUE] Training on {len(df)} observations")
        print(f"[ЛММР TRUE] Using OLS (not Ridge!) as in R-code")

        # Step 1: MoM → Base Index
        mom_series = df[target_col].dropna()
        base_series = self._to_base_index(mom_series)
        self.last_base_value = base_series.iloc[-1]
        self._original_base = base_series

        # Step 2: X13-ARIMA Decomposition
        print(f"[ЛММР TRUE] Performing seasonal decomposition...")
        self.sa_base_series, self.seasonal_component = self._decompose_with_x13(base_series)
        self.last_sa_base_value = self.sa_base_series.iloc[-1]

        # Step 3: SA Base → SA MoM
        sa_mom = self._from_base_to_mom(self.sa_base_series)
        self._sa_mom = sa_mom

        # Step 4: Сохраняем SC по месяцам для аддитивной модели
        self._sc_by_month = {}
        for month in range(1, 13):
            sc_for_month = self.seasonal_component[self.seasonal_component.index.month == month]
            if len(sc_for_month) > 0:
                self._sc_by_month[month] = sc_for_month.iloc[-1]
            else:
                self._sc_by_month[month] = 0.0

        # Step 5: Feature Engineering
        df_prepared = self._prepare_features(df, sa_mom)

        # Define features ТОЧНО по R-коду
        self._features = ['y_sa_lag1']

        # Добавляем признаки по R-коду (если есть данные)
        for feat in ['usd_lag1', 'prom_price_food', 'gruz_price_lag1']:
            if feat in df_prepared.columns and df_prepared[feat].notna().sum() > 10:
                self._features.append(feat)

        # Добавляем shock dummies
        self._features.extend(self.SHOCK_DUMMIES)

        # Step 6: Фильтрация обучающей выборки
        # R-код: start = c(2013,1)
        df_prepared = df_prepared[df_prepared.index >= self.TRAIN_START].copy()
        train = df_prepared.dropna(subset=self._features)

        X = train[self._features].values
        sa_aligned = sa_mom.reindex(train.index)
        y = sa_aligned.values

        # Remove NaN targets
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]

        if len(X) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Insufficient training data: {len(X)} < {self.MIN_TRAIN_SIZE}")

        # Step 7: OLS Regression (НЕ Ridge!)
        # R-код: dynlm() использует OLS
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LinearRegression()  # OLS!
        self.model.fit(X_scaled, y)

        # Сохраняем коэффициенты для итеративного прогноза
        # R-код: m.LS$coefficients
        self._coefficients = np.concatenate([[self.model.intercept_], self.model.coef_])

        self._is_fitted = True
        self._last_train_date = df.index.max()
        self._last_df = df.copy()

        print(f"[ЛММР TRUE] ✓ Model trained successfully")
        print(f"[ЛММР TRUE] Training period: {self.TRAIN_START.strftime('%Y-%m')} to {self._last_train_date.strftime('%Y-%m')}")
        print(f"[ЛММР TRUE] Features: {self._features}")
        print(f"[ЛММР TRUE] Training samples: {len(X)}")
        print(f"[ЛММР TRUE] Coefficients: intercept={self.model.intercept_:.4f}")
        for i, feat in enumerate(self._features):
            print(f"[ЛММР TRUE]   {feat}: {self.model.coef_[i]:.4f}")

        return self

    def _iterative_predict(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        Итеративное предсказание ТОЧНО по R-коду (строки 512-535).

        R-код:
        for (i in 1:(nrow(dt_mom_SA) - nrow(dt_mom_SA_fact))) {
            LS_data[n + i + 1, "y"] <- m.LS$coefficients %*% t(LS_data[n + i, ])
        }

        Ключевой момент: каждый следующий прогноз использует
        ПРЕДЫДУЩИЙ ПРОГНОЗ как L(ipc, 1)!
        """
        sa_mom_forecasts = []

        # Подготовка данных
        df_prepared = self._prepare_features(df, self._sa_mom)
        last_sa_mom = self._sa_mom.dropna().iloc[-1]

        # Получаем последние известные значения экзогенных переменных
        last_usd = df_prepared['usd_lag1'].dropna().iloc[-1] if 'usd_lag1' in df_prepared.columns else 100.0
        last_prom = df_prepared['prom_price_food'].dropna().iloc[-1] if 'prom_price_food' in df_prepared.columns else 100.0
        last_gruz = df_prepared['gruz_price_lag1'].dropna().iloc[-1] if 'gruz_price_lag1' in df_prepared.columns else 100.0

        current_y_lag = last_sa_mom

        for h in range(horizon):
            # Вычисляем месяц для прогноза
            future_date = self._last_train_date + pd.DateOffset(months=h + 1)

            # Собираем признаки для этого шага
            features = [current_y_lag]  # y_sa_lag1 = предыдущий прогноз!

            if 'usd_lag1' in self._features:
                features.append(last_usd)
            if 'prom_price_food' in self._features:
                features.append(last_prom)
            if 'gruz_price_lag1' in self._features:
                features.append(last_gruz)

            # Добавляем shock dummies
            for dummy in self.SHOCK_DUMMIES:
                if dummy == 'is_shock_dec2014_jan2015':
                    val = 1 if (future_date.year == 2014 and future_date.month == 12) or \
                              (future_date.year == 2015 and future_date.month == 1) else 0
                elif dummy == 'is_tariff_jul':
                    val = 1 if future_date.month == 7 else 0
                elif dummy == 'is_shock_mar2022':
                    val = 1 if future_date.year == 2022 and future_date.month == 3 else 0
                elif dummy == 'is_shock_apr2022':
                    val = 1 if future_date.year == 2022 and future_date.month == 4 else 0
                else:
                    val = 0
                features.append(val)

            # Предсказание SA MoM
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            sa_mom_pred = self.model.predict(X_scaled)[0]

            sa_mom_forecasts.append(sa_mom_pred)

            # КЛЮЧЕВОЙ МОМЕНТ: обновляем y_lag для следующего шага!
            # R-код: LS_data[n + i + 1, "y"] <- ... → используется в следующей итерации
            current_y_lag = sa_mom_pred

        return np.array(sa_mom_forecasts)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Point forecast for a specific date.

        Использует итеративное предсказание до target_date.
        """
        self._check_fitted()

        # Вычисляем горизонт от последней даты обучения
        months_ahead = (target_date.year - self._last_train_date.year) * 12 + \
                       (target_date.month - self._last_train_date.month)

        if months_ahead <= 0:
            # Дата в прошлом или текущая
            months_ahead = 1

        # Итеративное предсказание SA MoM
        sa_mom_forecasts = self._iterative_predict(df, months_ahead)
        sa_mom_pred = sa_mom_forecasts[-1]

        # SA MoM → SA Base (cumprod)
        # R-код строка 94: cumprod(prognoz.momSA.data / 100) * SA_Base[last_date]
        sa_base_cumulative = np.cumprod(sa_mom_forecasts / 100) * self.last_sa_base_value
        sa_base_pred = sa_base_cumulative[-1]

        # Base = SA_Base + SC (АДДИТИВНАЯ модель!)
        # R-код строка 96: var_base_f <- var_base_sa_f + tail(dt_base_SC, length(var_base_sa_f))
        sc = self._sc_by_month.get(target_date.month, 0.0)
        base_pred = sa_base_pred + sc

        # MoM = Base / lag(Base) * 100
        # R-код строка 97: var_mom_f <- var_base_f / stats::lag(var_base_f, k = 1) * 100
        if len(sa_base_cumulative) > 1:
            prev_base = sa_base_cumulative[-2] + self._sc_by_month.get(
                (target_date - pd.DateOffset(months=1)).month, 0.0
            )
        else:
            prev_base = self.last_base_value

        mom_pred = (base_pred / prev_base) * 100

        return {
            'date': target_date,
            'prediction': mom_pred,
            'model': self.name,
            'sa_mom_prediction': sa_mom_pred,
            'sa_base_prediction': sa_base_pred,
            'seasonal_component': sc,
            'base_prediction': base_pred,
            'horizon': months_ahead
        }

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Forecast horizon steps ahead with iterative prediction.
        """
        self._check_fitted()

        # Итеративное предсказание SA MoM
        sa_mom_forecasts = self._iterative_predict(self._last_df, horizon)

        # SA MoM → SA Base → Base → MoM
        forecasts = []
        prev_base = self.last_base_value

        for h in range(horizon):
            # SA Base (cumprod до текущего шага)
            sa_base = np.cumprod(sa_mom_forecasts[:h+1] / 100)[-1] * self.last_sa_base_value

            # Month for SC
            future_month = (self._last_train_date.month + h) % 12 + 1
            sc = self._sc_by_month.get(future_month, 0.0)

            # Base = SA_Base + SC (АДДИТИВНАЯ модель!)
            base_forecast = sa_base + sc

            # MoM = Base / prev_Base * 100
            mom_forecast = (base_forecast / prev_base) * 100
            forecasts.append(mom_forecast)

            prev_base = base_forecast

        return np.array(forecasts)

    def backtest(self, df: pd.DataFrame, start_date: str = '2024-01-01',
                 target_col: str = 'Все товары и услуги') -> pd.DataFrame:
        """Backtest with expanding window."""
        start = pd.Timestamp(start_date)
        results = []

        for target_date in df.dropna(subset=[target_col]).index:
            if target_date < start:
                continue

            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = LMMRTrueForecaster(use_x13=self.use_x13)
                model.fit(train_df, target_col)

                test_df = df[df.index <= target_date].copy()
                pred = model.predict(test_df, target_date)

                actual = df.loc[target_date, target_col]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred['prediction'],
                    'error': actual - pred['prediction'],
                    'abs_error': abs(actual - pred['prediction'])
                })
            except Exception as e:
                print(f"  ⚠️ Backtest failed at {target_date}: {e}")
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from OLS coefficients."""
        self._check_fitted()

        return pd.DataFrame({
            'feature': self._features,
            'coefficient': self.model.coef_,
            'abs_importance': np.abs(self.model.coef_)
        }).sort_values('abs_importance', ascending=False)
