"""
Ki Trajectory Forecaster — эндогенное моделирование ключевой ставки.

Использует правило Тейлора для прогнозирования траектории Ki:
    Ki_t+1 = inertia * Ki_t + (1-inertia) * Ki_target
    Ki_target = neutral_rate + phi_inf * (inf_forecast - inf_target)

Пример использования:
    >>> model = KiTrajectoryForecaster()
    >>> model.fit(df)
    >>> ki_path = model.forecast_trajectory(12, inf_forecast, current_ki=21.0)
    >>> scenarios = model.generate_scenarios(12, {'base': inf_base, 'high': inf_high})
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaylorRuleParams:
    """Параметры правила Тейлора для ЦБ РФ."""
    inertia: float = 0.85       # Инерция ставки (ЦБ меняет плавно)
    neutral_rate: float = 7.0   # Нейтральная ставка (r*)
    inf_target: float = 4.0     # Таргет инфляции ЦБ (%)
    phi_inf: float = 1.5        # Коэффициент реакции на инфляцию


class KiTrajectoryForecaster:
    """
    Эндогенное моделирование ключевой ставки по правилу Тейлора.

    Правило ЦБ РФ (упрощённое):
        Ki_target = neutral_rate + phi_inf * (inf_yoy - inf_target)
        Ki_t+1 = inertia * Ki_t + (1 - inertia) * Ki_target

    Параметры калибруются на исторических данных или задаются вручную.

    Attributes:
        params: TaylorRuleParams с параметрами модели
        _is_fitted: Флаг обученности модели
    """

    DEFAULT_PARAMS = TaylorRuleParams(
        inertia=0.85,
        neutral_rate=7.0,
        inf_target=4.0,
        phi_inf=1.5
    )

    def __init__(self, params: Optional[TaylorRuleParams] = None):
        """
        Инициализация модели.

        Args:
            params: Параметры правила Тейлора. Если None, используются значения по умолчанию.
        """
        self.params = params or TaylorRuleParams()
        self._is_fitted = False
        self._last_ki = None
        self._last_yoy = None
        self._calibration_results = None

    def fit(
        self,
        df: pd.DataFrame,
        ki_col: str = 'Ki_i',
        mom_col: str = 'mom',
        calibrate: bool = True
    ) -> 'KiTrajectoryForecaster':
        """
        Калибровка параметров на исторических данных.

        Args:
            df: DataFrame с историческими данными
            ki_col: Название колонки с ключевой ставкой (%)
            mom_col: Название колонки с MoM инфляцией (100 + %)
            calibrate: Калибровать параметры на данных

        Returns:
            self
        """
        # Подготовка данных
        if ki_col not in df.columns:
            logger.warning(f"Колонка {ki_col} не найдена, используем значения по умолчанию")
            self._is_fitted = True
            return self

        ki = df[ki_col].dropna()

        # Конвертация MoM в YoY
        if mom_col in df.columns:
            mom = df[mom_col].copy()
            if mom.mean() > 50:  # Формат 100+%
                mom = mom - 100
            yoy = mom.rolling(12).sum()
        else:
            yoy = pd.Series(dtype=float)

        # Сохраняем последние значения
        self._last_ki = ki.iloc[-1] if len(ki) > 0 else 21.0
        self._last_yoy = yoy.iloc[-1] if len(yoy.dropna()) > 0 else 8.0

        if calibrate and len(ki) > 24:
            self._calibrate(ki, yoy)

        self._is_fitted = True
        logger.info(f"KiTrajectoryForecaster fitted: last_ki={self._last_ki:.1f}%, last_yoy={self._last_yoy:.1f}%")
        return self

    def _calibrate(self, ki: pd.Series, yoy: pd.Series) -> None:
        """Калибровка параметров методом наименьших квадратов."""
        # Выравнивание данных
        data = pd.DataFrame({'ki': ki, 'yoy': yoy}).dropna()
        if len(data) < 24:
            logger.warning("Недостаточно данных для калибровки, используем значения по умолчанию")
            return

        ki_arr = data['ki'].values
        yoy_arr = data['yoy'].values

        def loss(x):
            inertia, neutral_rate, phi_inf = x
            ki_target = neutral_rate + phi_inf * (yoy_arr[:-1] - self.params.inf_target)
            ki_pred = inertia * ki_arr[:-1] + (1 - inertia) * ki_target
            return np.mean((ki_arr[1:] - ki_pred) ** 2)

        # Оптимизация
        result = minimize(
            loss,
            x0=[0.85, 7.0, 1.5],
            bounds=[(0.5, 0.99), (3.0, 12.0), (0.5, 3.0)],
            method='L-BFGS-B'
        )

        if result.success:
            self.params = TaylorRuleParams(
                inertia=result.x[0],
                neutral_rate=result.x[1],
                inf_target=self.params.inf_target,
                phi_inf=result.x[2]
            )
            self._calibration_results = {
                'inertia': result.x[0],
                'neutral_rate': result.x[1],
                'phi_inf': result.x[2],
                'loss': result.fun,
                'n_obs': len(data)
            }
            logger.info(f"Калибровка успешна: inertia={result.x[0]:.3f}, neutral={result.x[1]:.1f}, phi={result.x[2]:.2f}")

    def forecast_trajectory(
        self,
        horizon: int,
        inf_forecast: np.ndarray,
        current_ki: Optional[float] = None
    ) -> np.ndarray:
        """
        Прогноз траектории Ki на основе прогноза инфляции.

        Args:
            horizon: Горизонт прогноза (месяцев)
            inf_forecast: Прогноз MoM инфляции (%) — массив длины horizon
            current_ki: Текущая ставка (%). Если None, берётся последняя из fit()

        Returns:
            np.ndarray: Траектория Ki на horizon месяцев
        """
        if current_ki is None:
            current_ki = self._last_ki if self._last_ki is not None else 21.0

        # Конвертируем MoM в YoY (накопительно)
        if len(inf_forecast) < 12:
            # Дополняем историческим средним
            inf_extended = np.concatenate([
                np.full(12 - len(inf_forecast), self._last_yoy / 12 if self._last_yoy else 0.5),
                inf_forecast
            ])
        else:
            inf_extended = inf_forecast

        # Вычисляем YoY инфляцию (скользящая сумма за 12 месяцев)
        yoy_forecast = np.array([
            np.sum(inf_extended[max(0, i-11):i+1]) for i in range(len(inf_extended))
        ])
        yoy_forecast = yoy_forecast[-horizon:]  # Берём последние horizon значений

        # Генерация траектории Ki
        ki_path = np.zeros(horizon)
        ki_prev = current_ki

        for t in range(horizon):
            # Целевая ставка по правилу Тейлора
            ki_target = self.params.neutral_rate + self.params.phi_inf * (yoy_forecast[t] - self.params.inf_target)

            # Инерционная корректировка
            ki_next = self.params.inertia * ki_prev + (1 - self.params.inertia) * ki_target

            # Ограничения (ставка от 4% до 25%)
            ki_next = np.clip(ki_next, 4.0, 25.0)

            ki_path[t] = ki_next
            ki_prev = ki_next

        return ki_path

    def generate_scenarios(
        self,
        horizon: int,
        inf_scenarios: Dict[str, np.ndarray],
        current_ki: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Автоматическая генерация сценариев Ki на основе сценариев инфляции.

        Args:
            horizon: Горизонт прогноза
            inf_scenarios: Словарь {'scenario_name': inf_forecast_array}
            current_ki: Текущая ставка

        Returns:
            Dict: Словарь с траекториями Ki для каждого сценария
        """
        ki_scenarios = {}

        for name, inf_forecast in inf_scenarios.items():
            ki_path = self.forecast_trajectory(horizon, inf_forecast, current_ki)
            ki_scenarios[name] = ki_path

        # Добавляем производные сценарии (hike/cut) на основе base
        if 'base' in ki_scenarios and len(ki_scenarios) == 1:
            base_ki = ki_scenarios['base']

            # Сценарий повышения: +2 п.п. от базового
            ki_scenarios['hike'] = np.clip(base_ki + 2.0, 4.0, 25.0)

            # Сценарий снижения: -2 п.п. от базового
            ki_scenarios['cut'] = np.clip(base_ki - 2.0, 4.0, 25.0)

        return ki_scenarios

    def get_params(self) -> Dict:
        """Получить текущие параметры модели."""
        return {
            'inertia': self.params.inertia,
            'neutral_rate': self.params.neutral_rate,
            'inf_target': self.params.inf_target,
            'phi_inf': self.params.phi_inf,
            'is_fitted': self._is_fitted,
            'last_ki': self._last_ki,
            'last_yoy': self._last_yoy,
            'calibration': self._calibration_results
        }

    def simulate_policy_path(
        self,
        horizon: int,
        policy_change: float,
        delay: int = 0,
        current_ki: Optional[float] = None
    ) -> np.ndarray:
        """
        Симуляция траектории Ki при заданном изменении политики.

        Args:
            horizon: Горизонт
            policy_change: Изменение ставки (п.п.)
            delay: Через сколько месяцев начинается изменение
            current_ki: Текущая ставка

        Returns:
            Траектория Ki
        """
        if current_ki is None:
            current_ki = self._last_ki if self._last_ki is not None else 21.0

        ki_path = np.zeros(horizon)
        ki_target = current_ki + policy_change

        for t in range(horizon):
            if t < delay:
                ki_path[t] = current_ki
            else:
                # Плавный переход к целевому уровню
                progress = min(1.0, (t - delay + 1) / 6)  # За 6 месяцев
                ki_path[t] = current_ki + progress * policy_change

        return np.clip(ki_path, 4.0, 25.0)


if __name__ == '__main__':
    print("=" * 60)
    print("=== Тест KiTrajectoryForecaster ===")
    print("=" * 60)

    # Загрузка данных
    import pandas as pd
    from pathlib import Path

    data_dir = Path(__file__).parent.parent.parent / 'data'
    df = pd.read_csv(data_dir / 'inflation_data.csv', sep=';', decimal=',')

    # Конвертация
    for col in df.columns:
        if col != 'Date' and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df = df.set_index('Date').sort_index()

    # Тест модели
    model = KiTrajectoryForecaster()
    model.fit(df, ki_col='Ki', mom_col='mom')

    print(f"\nПараметры модели:")
    for k, v in model.get_params().items():
        print(f"  {k}: {v}")

    # Тест прогноза
    inf_forecast = np.array([0.6, 0.5, 0.4, 0.4, 0.4, 0.5, 0.6, 0.5, 0.4, 0.4, 0.4, 0.4])
    ki_path = model.forecast_trajectory(12, inf_forecast)

    print(f"\nПрогноз траектории Ki (при MoM инфляции ~0.5%):")
    for i, ki in enumerate(ki_path):
        print(f"  Месяц {i+1}: {ki:.1f}%")

    # Тест сценариев
    scenarios = model.generate_scenarios(12, {'base': inf_forecast})
    print(f"\nСценарии Ki:")
    for name, path in scenarios.items():
        print(f"  {name}: {path[0]:.1f}% → {path[-1]:.1f}%")
