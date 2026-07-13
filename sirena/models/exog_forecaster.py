#!/usr/bin/env python3
"""
ПРОГНОЗ ЭКЗОГЕННЫХ ПЕРЕМЕННЫХ v1.0
==================================

Модели прогноза для:
- Ki (Ключевая ставка): AR(1) с возвратом к нейтральной ставке
- Ruonia: Ki минус спред (исторический спред ~0.25-0.5%)
- USD: Random Walk или AR(1) с возвратом к среднему
- Brent: AR(1) с возвратом к долгосрочному среднему (~70$)

Методы:
- 'naive': Последнее значение (ffill) — baseline
- 'ar1': AR(1) с возвратом к среднему
- 'taylor': Для Ki — правило Тейлора
- 'futures': Для Brent — использование фьючерсов (если доступны)

Использование:
    from sirena.models.exog_forecaster import ExogForecaster

    ef = ExogForecaster()
    ef.fit(macro_df)

    # Прогноз на 12 месяцев
    forecast = ef.forecast(12)

    # Сохранить в файл
    ef.save_forecast('data/exog_forecast.csv')
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Literal
import json
import warnings
warnings.filterwarnings('ignore')


class ExogForecaster:
    """Прогноз экзогенных переменных для инфляционных моделей."""

    # Долгосрочные равновесные значения
    NEUTRAL_KI = 8.0       # Нейтральная ставка ЦБ (долгосрочная цель)
    NEUTRAL_RUONIA_SPREAD = 0.25  # Спред Ki - Ruonia
    NEUTRAL_USD = 75.0     # Долгосрочный "равновесный" курс
    NEUTRAL_BRENT = 70.0   # Долгосрочная цена нефти

    # Скорость возврата к среднему (в месяц)
    MEAN_REVERSION_KI = 0.05      # Медленный возврат (5% в месяц)
    MEAN_REVERSION_USD = 0.02     # Очень медленный (2% в месяц)
    MEAN_REVERSION_BRENT = 0.03   # Медленный (3% в месяц)

    def __init__(
        self,
        ki_method: Literal['naive', 'ar1', 'adaptive', 'taylor'] = 'adaptive',
        usd_method: Literal['naive', 'ar1', 'adaptive'] = 'adaptive',
        brent_method: Literal['naive', 'ar1', 'futures'] = 'ar1',
        current_usd_abs: Optional[float] = None,  # Текущий курс USD (RUB)
        current_ki: Optional[float] = None,       # Текущая ставка Ki (%)
        current_ruonia: Optional[float] = None,   # Текущая ставка Ruonia (%)
        current_brent: Optional[float] = None,    # Текущая цена Brent ($)
    ):
        """
        Args:
            ki_method: Метод прогноза Ki ('naive', 'ar1', 'adaptive', 'taylor')
                       'adaptive' — лучший по бэктесту (MAE -24% vs naive на h=1)
            usd_method: Метод прогноза USD ('naive', 'ar1')
            brent_method: Метод прогноза Brent ('naive', 'ar1', 'futures')
            current_usd_abs: Текущий курс USD в рублях (если None - из файла)
            current_ki: Текущая ключевая ставка (если None - из файла)
            current_ruonia: Текущая ставка Ruonia (если None - Ki - 0.25)
            current_brent: Текущая цена Brent (если None - из файла)
        """
        self.ki_method = ki_method
        self.usd_method = usd_method
        self.brent_method = brent_method
        self.current_usd_abs = current_usd_abs
        self.current_ki = current_ki
        self.current_ruonia = current_ruonia
        self.current_brent = current_brent

        self._is_fitted = False
        self.last_values = {}
        self.macro_df = None
        self.brent_df = None
        self.last_date = None

    def fit(self, macro_df: pd.DataFrame, brent_df: Optional[pd.DataFrame] = None):
        """
        Обучение на исторических данных.

        Args:
            macro_df: DataFrame с колонками Ki, Ruonia, usd_nom_i, Ki_i
            brent_df: DataFrame с колонками Date, brent (опционально)
        """
        self.macro_df = macro_df.copy()

        # Normalize index
        if not isinstance(self.macro_df.index, pd.DatetimeIndex):
            if 'Date' in self.macro_df.columns:
                self.macro_df['Date'] = pd.to_datetime(self.macro_df['Date'])
                self.macro_df = self.macro_df.set_index('Date')

        self.macro_df.index = self.macro_df.index.to_period('M').to_timestamp()
        self.last_date = self.macro_df.index.max()

        # Extract last known values from file
        last_row = self.macro_df.iloc[-1]

        # Use override values if provided, otherwise from file
        ki_val = self.current_ki if self.current_ki is not None else last_row.get('Ki', 16.5)
        ruonia_val = self.current_ruonia if self.current_ruonia is not None else last_row.get('Ruonia', ki_val - 0.25)

        self.last_values = {
            'Ki': ki_val,
            'Ruonia': ruonia_val,
            'Ki_i': last_row.get('Ki_i', 100.0),
            'usd_nom_i': last_row.get('usd_nom_i', 100.0),
        }

        # USD absolute value
        if self.current_usd_abs is not None:
            self.last_values['USD_ABS'] = self.current_usd_abs
            # Also add reconstructed history
            self._reconstruct_usd_abs_from_current()
        else:
            # Try to reconstruct from indices (less accurate)
            self._reconstruct_usd_abs()

        # Load Brent if provided
        if brent_df is not None:
            self.brent_df = brent_df.copy()
        else:
            self._load_brent()

        # Brent value
        if self.current_brent is not None:
            self.last_values['Brent'] = self.current_brent
        elif self.brent_df is not None and len(self.brent_df) > 0:
            self.last_values['Brent'] = self.brent_df['brent'].iloc[-1]
        else:
            self.last_values['Brent'] = self.NEUTRAL_BRENT

        self._is_fitted = True
        return self

    def _reconstruct_usd_abs(self):
        """Реконструкция абсолютного курса USD из индекса (fallback)."""
        # Use default 100 if no current value provided
        current = self.current_usd_abs if self.current_usd_abs is not None else 100.0
        usd_idx = self.macro_df['usd_nom_i'].values
        n = len(usd_idx)

        usd_abs = np.zeros(n)
        usd_abs[-1] = current

        for i in range(n-2, -1, -1):
            usd_abs[i] = usd_abs[i+1] / (usd_idx[i+1] / 100)

        self.macro_df['USD_ABS'] = usd_abs
        self.last_values['USD_ABS'] = current

    def _reconstruct_usd_abs_from_current(self):
        """Реконструкция истории USD от текущего известного курса."""
        current = self.current_usd_abs
        usd_idx = self.macro_df['usd_nom_i'].values
        n = len(usd_idx)

        usd_abs = np.zeros(n)
        usd_abs[-1] = current

        # Backwards reconstruction using indices
        for i in range(n-2, -1, -1):
            # usd_abs[i+1] = usd_abs[i] * (usd_idx[i+1] / 100)
            # => usd_abs[i] = usd_abs[i+1] / (usd_idx[i+1] / 100)
            usd_abs[i] = usd_abs[i+1] / (usd_idx[i+1] / 100)

        self.macro_df['USD_ABS'] = usd_abs

    def _load_brent(self):
        """Загрузка данных Brent из файла."""
        try:
            brent_path = Path(__file__).parent.parent.parent / 'data' / 'brent_prices.csv'
            self.brent_df = pd.read_csv(brent_path)
            self.brent_df['Date'] = pd.to_datetime(self.brent_df['Date'])
            self.brent_df = self.brent_df.set_index('Date').sort_index()
        except:
            self.brent_df = None

    def forecast(self, horizon: int = 12) -> pd.DataFrame:
        """
        Генерация прогноза экзогенных на horizon месяцев.

        Returns:
            DataFrame с колонками:
            - Date: дата прогноза
            - Ki: ключевая ставка (%)
            - Ki_i: индекс изменения Ki (MoM)
            - Ruonia: ставка RUONIA (%)
            - USD_ABS: курс USD (RUB)
            - usd_nom_i: индекс изменения USD (MoM)
            - Brent: цена нефти ($)
            - Brent_pct: изменение Brent (%)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        dates = pd.date_range(
            start=self.last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq='MS'
        )

        # Initialize forecast arrays
        ki_forecast = self._forecast_ki(horizon)
        usd_forecast = self._forecast_usd(horizon)
        brent_forecast = self._forecast_brent(horizon)

        # Calculate indices from absolute values
        ki_i = self._abs_to_index(ki_forecast, self.last_values['Ki'])
        usd_i = self._abs_to_index(usd_forecast, self.last_values['USD_ABS'])
        brent_pct = self._abs_to_pct(brent_forecast, self.last_values['Brent'])

        # Ruonia = Ki - spread
        ruonia_forecast = ki_forecast - self.NEUTRAL_RUONIA_SPREAD

        result = pd.DataFrame({
            'Date': dates,
            'Ki': ki_forecast,
            'Ki_i': ki_i,
            'Ruonia': ruonia_forecast,
            'USD_ABS': usd_forecast,
            'usd_nom_i': usd_i,
            'Brent': brent_forecast,
            'Brent_pct': brent_pct,
        })

        return result

    def _forecast_ki(self, horizon: int) -> np.ndarray:
        """Прогноз ключевой ставки."""
        ki_last = self.last_values['Ki']

        if self.ki_method == 'naive':
            return np.full(horizon, ki_last)

        elif self.ki_method == 'adaptive':
            # Adaptive: продолжаем тренд если Ki менялся в последние 6 месяцев
            # Лучший метод по бэктесту (MAE -24% vs naive на h=1)
            recent = self.macro_df['Ki'].tail(6)
            trend = (recent.iloc[-1] - recent.iloc[0]) / 5  # Средняя месячная дельта

            ki = np.zeros(horizon)
            ki[0] = ki_last

            if abs(trend) < 0.2:  # Нет значимого тренда (снизил порог)
                # Naive forecast
                return np.full(horizon, ki_last)
            else:
                # Продолжаем тренд с затуханием
                for t in range(1, horizon):
                    damping = 0.7 ** t  # Тренд ослабевает со временем
                    ki[t] = ki[t-1] + trend * damping
                    ki[t] = max(4.0, min(25.0, ki[t]))  # Разумные границы

                return ki

        elif self.ki_method == 'ar1':
            # AR(1) с возвратом к нейтральной ставке
            ki = np.zeros(horizon)
            ki[0] = ki_last

            for t in range(1, horizon):
                # Ki_t = Ki_{t-1} + lambda * (Ki_neutral - Ki_{t-1})
                ki[t] = ki[t-1] + self.MEAN_REVERSION_KI * (self.NEUTRAL_KI - ki[t-1])

            return ki

        elif self.ki_method == 'taylor':
            # Простое правило Тейлора (требует прогноз инфляции)
            # Пока используем AR(1) как fallback
            return self._forecast_ki_ar1(horizon)

        return np.full(horizon, ki_last)

    def _forecast_ki_ar1(self, horizon: int) -> np.ndarray:
        """AR(1) прогноз Ki с возвратом к среднему."""
        ki_last = self.last_values['Ki']
        ki = np.zeros(horizon)
        ki[0] = ki_last

        for t in range(1, horizon):
            ki[t] = ki[t-1] + self.MEAN_REVERSION_KI * (self.NEUTRAL_KI - ki[t-1])

        return ki

    def _forecast_usd(self, horizon: int) -> np.ndarray:
        """Прогноз курса USD."""
        usd_last = self.last_values['USD_ABS']

        if self.usd_method == 'naive':
            return np.full(horizon, usd_last)

        elif self.usd_method == 'adaptive':
            # Adaptive: продолжаем тренд если USD менялся в последние 6 месяцев
            # Используем восстановленные абсолютные значения
            if 'USD_ABS' in self.macro_df.columns:
                recent = self.macro_df['USD_ABS'].tail(6)
                trend = (recent.iloc[-1] - recent.iloc[0]) / 5  # RUB/месяц
            else:
                trend = 0

            usd = np.zeros(horizon)
            usd[0] = usd_last

            if abs(trend) < 0.5:  # Нет значимого тренда (менее 0.5 RUB/мес)
                return np.full(horizon, usd_last)
            else:
                # Продолжаем тренд с затуханием
                for t in range(1, horizon):
                    damping = 0.8 ** t  # Тренд ослабевает со временем
                    usd[t] = usd[t-1] + trend * damping
                    # Разумные границы для курса
                    usd[t] = max(50.0, min(150.0, usd[t]))

                return usd

        elif self.usd_method == 'ar1':
            # AR(1) с медленным возвратом к среднему
            usd = np.zeros(horizon)
            usd[0] = usd_last

            for t in range(1, horizon):
                usd[t] = usd[t-1] + self.MEAN_REVERSION_USD * (self.NEUTRAL_USD - usd[t-1])

            return usd

        return np.full(horizon, usd_last)

    def _forecast_brent(self, horizon: int) -> np.ndarray:
        """Прогноз цены Brent."""
        brent_last = self.last_values.get('Brent', self.NEUTRAL_BRENT)

        if self.brent_method == 'naive':
            return np.full(horizon, brent_last)

        elif self.brent_method == 'ar1':
            # AR(1) с возвратом к среднему
            brent = np.zeros(horizon)
            brent[0] = brent_last

            for t in range(1, horizon):
                brent[t] = brent[t-1] + self.MEAN_REVERSION_BRENT * (self.NEUTRAL_BRENT - brent[t-1])

            return brent

        elif self.brent_method == 'futures':
            # TODO: загрузка фьючерсной кривой
            return np.full(horizon, brent_last)

        return np.full(horizon, brent_last)

    def _abs_to_index(self, values: np.ndarray, base: float) -> np.ndarray:
        """Конвертация абсолютных значений в индексы MoM."""
        idx = np.zeros(len(values))
        prev = base
        for i, val in enumerate(values):
            idx[i] = (val / prev) * 100
            prev = val
        return idx

    def _abs_to_pct(self, values: np.ndarray, base: float) -> np.ndarray:
        """Конвертация абсолютных значений в проценты изменения."""
        pct = np.zeros(len(values))
        prev = base
        for i, val in enumerate(values):
            pct[i] = ((val / prev) - 1) * 100
            prev = val
        return pct

    def save_forecast(
        self,
        filepath: str = 'data/exog_forecast.csv',
        horizon: int = 12,
        include_history: int = 6
    ):
        """
        Сохранение прогноза в CSV файл.

        Args:
            filepath: путь к файлу
            horizon: горизонт прогноза
            include_history: сколько месяцев истории включить
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get forecast
        fc = self.forecast(horizon)
        fc['Type'] = 'Forecast'

        # Add historical data
        if include_history > 0:
            hist = self.macro_df.tail(include_history).copy()
            hist = hist.reset_index()
            hist.columns = ['Date'] + list(hist.columns[1:])

            # Add missing columns
            if 'USD_ABS' not in hist.columns:
                hist['USD_ABS'] = self.macro_df['USD_ABS'].tail(include_history).values

            if self.brent_df is not None:
                brent_hist = self.brent_df.tail(include_history)
                hist['Brent'] = brent_hist['brent'].values if len(brent_hist) == include_history else np.nan
                hist['Brent_pct'] = brent_hist['brent_pct'].values if len(brent_hist) == include_history else np.nan

            hist['Type'] = 'History'

            # Select matching columns
            common_cols = ['Date', 'Ki', 'Ruonia', 'USD_ABS', 'usd_nom_i', 'Brent', 'Type']
            hist = hist[[c for c in common_cols if c in hist.columns]]
            fc = fc[[c for c in common_cols if c in fc.columns]]

            result = pd.concat([hist, fc], ignore_index=True)
        else:
            result = fc

        # Save
        result.to_csv(filepath, index=False, float_format='%.2f')
        print(f"Прогноз сохранен: {filepath}")

        return result

    def save_forecast_json(self, filepath: str = 'data/exog_forecast.json', horizon: int = 12):
        """Сохранение прогноза в JSON формате."""
        fc = self.forecast(horizon)

        result = {
            'generated_at': datetime.now().isoformat(),
            'last_data_date': self.last_date.strftime('%Y-%m-%d'),
            'horizon': horizon,
            'methods': {
                'Ki': self.ki_method,
                'USD': self.usd_method,
                'Brent': self.brent_method,
            },
            'last_values': {k: float(v) for k, v in self.last_values.items()},
            'forecast': fc.to_dict(orient='records'),
        }

        # Convert dates to strings
        for rec in result['forecast']:
            rec['Date'] = rec['Date'].strftime('%Y-%m-%d')

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Прогноз сохранен: {filepath}")
        return result

    def print_forecast(self, horizon: int = 12):
        """Вывод прогноза в консоль."""
        fc = self.forecast(horizon)

        print("\n" + "="*80)
        print("ПРОГНОЗ ЭКЗОГЕННЫХ ПЕРЕМЕННЫХ")
        print(f"Последние данные: {self.last_date.strftime('%Y-%m')}")
        print(f"Методы: Ki={self.ki_method}, USD={self.usd_method}, Brent={self.brent_method}")
        print("="*80)

        print(f"\n{'Дата':<12} {'Ki,%':>8} {'Ruonia,%':>10} {'USD,RUB':>10} {'Brent,$':>10}")
        print("-"*52)

        for _, row in fc.iterrows():
            print(f"{row['Date'].strftime('%Y-%m'):<12} "
                  f"{row['Ki']:>8.2f} {row['Ruonia']:>10.2f} "
                  f"{row['USD_ABS']:>10.2f} {row['Brent']:>10.2f}")

        print("-"*52)
        print(f"{'Изменение':<12} "
              f"{fc['Ki'].iloc[-1] - self.last_values['Ki']:>+8.2f} "
              f"{fc['Ruonia'].iloc[-1] - self.last_values['Ruonia']:>+10.2f} "
              f"{fc['USD_ABS'].iloc[-1] - self.last_values['USD_ABS']:>+10.2f} "
              f"{fc['Brent'].iloc[-1] - self.last_values['Brent']:>+10.2f}")


def backtest_exog_forecaster(
    macro_df: pd.DataFrame,
    brent_df: Optional[pd.DataFrame] = None,
    test_start: str = '2023-01-01',
    horizons: list = [1, 3, 6, 12],
    methods: dict = None
) -> pd.DataFrame:
    """
    Бэктест моделей прогноза экзогенных.

    Args:
        macro_df: исторические данные
        brent_df: данные Brent
        test_start: начало тестового периода
        horizons: горизонты прогноза
        methods: dict с методами для каждой переменной

    Returns:
        DataFrame с метриками MAE для каждой переменной и горизонта
    """
    if methods is None:
        methods = {'ki': 'ar1', 'usd': 'naive', 'brent': 'ar1'}

    # Prepare data
    macro_df = macro_df.copy()
    if not isinstance(macro_df.index, pd.DatetimeIndex):
        if 'Date' in macro_df.columns:
            macro_df['Date'] = pd.to_datetime(macro_df['Date'])
            macro_df = macro_df.set_index('Date')
    macro_df.index = macro_df.index.to_period('M').to_timestamp()

    test_dates = macro_df[macro_df.index >= test_start].index

    results = []

    for h in horizons:
        ki_errors = []
        usd_errors = []
        brent_errors = []

        for test_date in test_dates:
            # Skip if not enough future data
            future_date = test_date + pd.DateOffset(months=h)
            if future_date > macro_df.index.max():
                continue

            # Train on data before test_date
            train_df = macro_df[macro_df.index < test_date].copy()

            if len(train_df) < 24:
                continue

            # Get current USD for calibration
            usd_current = 100.0  # Approximate

            # Fit and forecast
            ef = ExogForecaster(
                ki_method=methods.get('ki', 'ar1'),
                usd_method=methods.get('usd', 'naive'),
                brent_method=methods.get('brent', 'ar1'),
                current_usd_abs=usd_current
            )

            try:
                ef.fit(train_df, brent_df)
                fc = ef.forecast(h)

                # Get actual values
                actual_ki = macro_df.loc[future_date, 'Ki']
                pred_ki = fc['Ki'].iloc[-1]
                ki_errors.append(abs(actual_ki - pred_ki))

                # USD errors (using index)
                actual_usd_i = macro_df.loc[future_date, 'usd_nom_i']
                pred_usd_i = fc['usd_nom_i'].iloc[-1]
                usd_errors.append(abs(actual_usd_i - pred_usd_i))

            except Exception as e:
                continue

        if ki_errors:
            results.append({
                'Horizon': h,
                'Ki_MAE': np.mean(ki_errors),
                'Ki_N': len(ki_errors),
                'USD_i_MAE': np.mean(usd_errors) if usd_errors else np.nan,
            })

    return pd.DataFrame(results)


if __name__ == '__main__':
    # Demo
    import pandas as pd

    # Load data
    macro_df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
    for col in macro_df.columns:
        if macro_df[col].dtype == 'object' and col != 'Date':
            macro_df[col] = pd.to_numeric(str(macro_df[col]).replace(',', '.'), errors='coerce')

    # Create forecaster
    ef = ExogForecaster(ki_method='ar1', usd_method='naive', brent_method='ar1')
    ef.fit(macro_df)

    # Print forecast
    ef.print_forecast(12)

    # Save to file
    ef.save_forecast('data/exog_forecast.csv')
