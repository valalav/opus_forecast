"""
SIRENA Unified API v1.0
========================

Единый интерфейс для всех функций прогнозирования инфляции.

Использование:
    >>> from sirena import SIRENA
    >>>
    >>> # Инициализация
    >>> sirena = SIRENA()
    >>> sirena.load_data()
    >>>
    >>> # Текущий режим экономики
    >>> print(sirena.regime)  # 🔴 shock
    >>>
    >>> # Прогноз с авто-траекторией Ki
    >>> result = sirena.forecast(horizon=12, scenario='auto')
    >>> print(result.total)
    >>> print(result.ki_trajectory)
    >>>
    >>> # Сценарный анализ
    >>> scenarios = sirena.compare_scenarios(horizon=12)
    >>> print(scenarios['hike'].effect.sum())
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ForecastResult:
    """Результат прогноза."""
    dates: pd.DatetimeIndex
    baseline: np.ndarray
    effect: np.ndarray
    total: np.ndarray
    ki_trajectory: Optional[np.ndarray] = None
    ci_lower: Optional[np.ndarray] = None
    ci_upper: Optional[np.ndarray] = None
    regime: Optional[str] = None

    def summary(self) -> str:
        """Краткая сводка прогноза."""
        lines = [
            f"Прогноз на {len(self.total)} мес.",
            f"  Baseline: {self.baseline[0]:.2f}% → {self.baseline[-1]:.2f}%",
            f"  Effect:   {self.effect.sum():+.3f}% (cumulative)",
            f"  Total:    {self.total[0]:.2f}% → {self.total[-1]:.2f}%",
        ]
        if self.ki_trajectory is not None:
            lines.append(f"  Ki:       {self.ki_trajectory[0]:.1f}% → {self.ki_trajectory[-1]:.1f}%")
        if self.regime:
            lines.append(f"  Режим:    {self.regime}")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Конвертация в DataFrame."""
        df = pd.DataFrame({
            'Date': self.dates,
            'Baseline': self.baseline,
            'Effect': self.effect,
            'Total': self.total,
        })
        if self.ki_trajectory is not None:
            df['Ki'] = self.ki_trajectory
        if self.ci_lower is not None:
            df['CI_Lower'] = self.ci_lower
            df['CI_Upper'] = self.ci_upper
        return df.set_index('Date')


@dataclass
class RegimeInfo:
    """Информация о режиме экономики."""
    regime: str  # 'normal', 'shock', 'high_inflation'
    emoji: str
    label: str
    ki_change: float
    ruonia_change: float
    yoy_change: float
    lags: List[int]

    def __str__(self) -> str:
        return f"{self.emoji} {self.label} (ΔKi={self.ki_change:+.1f}, ΔRuonia={self.ruonia_change:+.1f})"


class SIRENA:
    """
    SIRENA — Система Интегрированного Регионального Экономического Анализа.

    Unified API для прогнозирования инфляции в КБР.

    Примеры:
        >>> sirena = SIRENA()
        >>> sirena.load_data()
        >>>
        >>> # Быстрый прогноз
        >>> fc = sirena.forecast(12)
        >>> print(fc.summary())
        >>>
        >>> # Прогноз со сценарием
        >>> fc = sirena.forecast(12, scenario='hike', ki_change=2.0)
        >>>
        >>> # Авто-сценарий по правилу Тейлора
        >>> fc = sirena.forecast(12, scenario='auto')
        >>>
        >>> # Текущий режим
        >>> print(sirena.regime)
    """

    VERSION = "1.0"

    def __init__(self, data_dir: str = None):
        """
        Инициализация SIRENA.

        Args:
            data_dir: Путь к директории с данными (по умолчанию: data/)
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / 'data'
        self.data_dir = Path(data_dir)

        self._df = None
        self._sa_df = None
        self._regime_info = None
        self._models = {}
        self._is_fitted = False

    def load_data(self, reload: bool = False) -> 'SIRENA':
        """
        Загрузить данные.

        Args:
            reload: Перезагрузить данные, даже если уже загружены

        Returns:
            self
        """
        if self._df is not None and not reload:
            return self

        # Загрузка macro данных
        infl_file = self.data_dir / 'inflation_data.csv'
        if not infl_file.exists():
            raise FileNotFoundError(f"Файл {infl_file} не найден")

        df = pd.read_csv(infl_file, sep=';', decimal=',', encoding='utf-8-sig')
        for col in df.columns:
            if col != 'Date' and df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        df = df.set_index('Date').sort_index()
        self._df = df

        # Загрузка SA данных (опционально)
        sa_file = self.data_dir / 'sa_fl.csv'
        if sa_file.exists():
            try:
                sa_df = pd.read_csv(sa_file, sep=';', decimal=',')
                sa_df['Дата'] = pd.to_datetime(sa_df['Дата'])
                sa_df = sa_df.pivot(index='Дата', columns='Код', values='Значение')
                sa_df.columns = [str(c) for c in sa_df.columns]
                for col in sa_df.columns:
                    if sa_df[col].dtype == object:
                        sa_df[col] = sa_df[col].astype(str).str.replace(',', '.').astype(float)
                self._sa_df = sa_df
            except Exception:
                pass

        # Определение режима
        self._detect_regime()

        return self

    def _detect_regime(self) -> None:
        """Определить текущий режим экономики."""
        if self._df is None:
            return

        try:
            from sirena.models.regime_detector import detect_regime, get_regime_lags

            regime, diag = detect_regime(self._df)

            emoji_map = {'normal': '🟢', 'shock': '🔴', 'high_inflation': '🟠'}
            label_map = {'normal': 'Нормальный', 'shock': 'Шок', 'high_inflation': 'Высокая инфляция'}

            self._regime_info = RegimeInfo(
                regime=regime.value,
                emoji=emoji_map[regime.value],
                label=label_map[regime.value],
                ki_change=diag['ki_change'],
                ruonia_change=diag['ruonia_change'],
                yoy_change=diag['yoy_change'],
                lags=get_regime_lags(regime)
            )
        except Exception:
            self._regime_info = None

    @property
    def regime(self) -> Optional[RegimeInfo]:
        """Текущий режим экономики."""
        return self._regime_info

    @property
    def data(self) -> pd.DataFrame:
        """Macro данные."""
        if self._df is None:
            raise RuntimeError("Данные не загружены. Вызовите load_data() сначала.")
        return self._df

    @property
    def last_date(self) -> pd.Timestamp:
        """Последняя дата в данных."""
        return self._df.index[-1]

    @property
    def current_ki(self) -> float:
        """Текущая ключевая ставка."""
        if 'Ki' in self._df.columns:
            return self._df['Ki'].iloc[-1]
        return 21.0  # default

    def fit(self, force: bool = False) -> 'SIRENA':
        """
        Обучить модели.

        Args:
            force: Переобучить, даже если уже обучены

        Returns:
            self
        """
        if self._is_fitted and not force:
            return self

        if self._df is None:
            self.load_data()

        # Инициализация и обучение моделей
        from sirena.models.subcomponent_multi import SubcomponentMultiForecaster
        from sirena.models.ki_trajectory import KiTrajectoryForecaster
        from sirena.models.subcomponent_scenario import SubcomponentScenarioForecaster

        # SubcomponentMulti (baseline)
        self._models['baseline'] = SubcomponentMultiForecaster(horizon=1)
        self._models['baseline'].fit(self._df, 'mom')

        # Ki Trajectory (Taylor rule)
        self._models['ki'] = KiTrajectoryForecaster()
        self._models['ki'].fit(self._df)

        # Scenario (IRF)
        self._models['scenario'] = SubcomponentScenarioForecaster()
        self._models['scenario'].fit(self._df, 'mom')

        self._is_fitted = True
        return self

    def forecast(
        self,
        horizon: int = 12,
        scenario: str = 'base',
        ki_change: Optional[float] = None,
        ki_trajectory: Optional[np.ndarray] = None
    ) -> ForecastResult:
        """
        Прогноз инфляции.

        Args:
            horizon: Горизонт прогноза (месяцев)
            scenario: Сценарий ('base', 'hike', 'cut', 'auto', 'custom')
            ki_change: Изменение Ki (для scenario='custom')
            ki_trajectory: Траектория Ki (для scenario='custom')

        Returns:
            ForecastResult
        """
        if not self._is_fitted:
            self.fit()

        # Даты прогноза
        dates = pd.date_range(
            start=self.last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq='MS'
        )

        # Baseline прогноз
        baseline = self._models['baseline'].forecast(horizon)

        # Определение ki_change по сценарию
        if scenario == 'auto':
            # Авто-траектория по правилу Тейлора
            ki_traj = self._models['ki'].forecast_trajectory(horizon, baseline)
            ki_change = ki_traj[-1] - self.current_ki
            ki_trajectory = ki_traj
        elif scenario == 'hike':
            ki_change = 2.0
        elif scenario == 'cut':
            ki_change = -2.0
        elif scenario == 'base':
            ki_change = 0.0
        elif scenario == 'custom':
            if ki_change is None and ki_trajectory is None:
                ki_change = 0.0
            elif ki_trajectory is not None:
                ki_change = ki_trajectory[-1] - self.current_ki

        # Эффект от ставки
        if ki_change != 0:
            result = self._models['scenario'].forecast_scenario(horizon, ki_change=ki_change)
            effect = result['effect']
        else:
            effect = np.zeros(horizon)

        # Ki траектория (если не задана)
        if ki_trajectory is None and ki_change != 0:
            ki_trajectory = np.linspace(self.current_ki, self.current_ki + ki_change, horizon)
        elif ki_trajectory is None:
            ki_trajectory = np.full(horizon, self.current_ki)

        total = baseline + effect

        return ForecastResult(
            dates=dates,
            baseline=baseline,
            effect=effect,
            total=total,
            ki_trajectory=ki_trajectory,
            regime=self._regime_info.regime if self._regime_info else None
        )

    def compare_scenarios(
        self,
        horizon: int = 12,
        scenarios: List[str] = None
    ) -> Dict[str, ForecastResult]:
        """
        Сравнение сценариев.

        Args:
            horizon: Горизонт прогноза
            scenarios: Список сценариев (по умолчанию: ['base', 'hike', 'cut', 'auto'])

        Returns:
            Dict[scenario_name, ForecastResult]
        """
        if scenarios is None:
            scenarios = ['base', 'hike', 'cut', 'auto']

        results = {}
        for sc in scenarios:
            results[sc] = self.forecast(horizon, scenario=sc)

        return results

    def quick_forecast(self, horizon: int = 12) -> pd.DataFrame:
        """
        Быстрый прогноз — возвращает DataFrame со всеми сценариями.

        Args:
            horizon: Горизонт прогноза

        Returns:
            DataFrame с колонками: Base, Hike, Cut, Auto
        """
        scenarios = self.compare_scenarios(horizon)

        df = pd.DataFrame(index=scenarios['base'].dates)
        df['Base'] = scenarios['base'].total
        df['Hike (+2 п.п.)'] = scenarios['hike'].total
        df['Cut (-2 п.п.)'] = scenarios['cut'].total
        df['Auto (Taylor)'] = scenarios['auto'].total
        df['Ki (Auto)'] = scenarios['auto'].ki_trajectory

        return df

    def regime_history(self, months: int = 24) -> pd.DataFrame:
        """
        История режимов за последние N месяцев.

        Args:
            months: Количество месяцев

        Returns:
            DataFrame с историей режимов
        """
        from sirena.models.regime_detector import get_regime_history

        history = get_regime_history(self._df)
        return history.tail(months)

    def info(self) -> str:
        """Информация о системе."""
        lines = [
            "=" * 50,
            "SIRENA — Unified API v" + self.VERSION,
            "=" * 50,
            "",
            f"Данные загружены: {'Да' if self._df is not None else 'Нет'}",
        ]

        if self._df is not None:
            lines.extend([
                f"Период: {self._df.index.min().strftime('%Y-%m')} — {self._df.index.max().strftime('%Y-%m')}",
                f"Текущая Ki: {self.current_ki:.1f}%",
            ])

        if self._regime_info:
            lines.append(f"Режим: {self._regime_info}")

        lines.extend([
            "",
            f"Модели обучены: {'Да' if self._is_fitted else 'Нет'}",
        ])

        if self._is_fitted:
            lines.append(f"Модели: {list(self._models.keys())}")

        lines.extend([
            "",
            "Использование:",
            "  sirena.forecast(12)              # Baseline прогноз",
            "  sirena.forecast(12, 'auto')      # Авто-сценарий Ki",
            "  sirena.forecast(12, 'hike')      # Повышение ставки",
            "  sirena.compare_scenarios(12)     # Все сценарии",
            "  sirena.quick_forecast(12)        # DataFrame со всеми сценариями",
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "ready" if self._is_fitted else "not fitted"
        regime = self._regime_info.regime if self._regime_info else "unknown"
        return f"<SIRENA v{self.VERSION} | {status} | regime={regime}>"


# Convenience function
def create_sirena(auto_fit: bool = True) -> SIRENA:
    """
    Создать и инициализировать SIRENA.

    Args:
        auto_fit: Автоматически загрузить данные и обучить модели

    Returns:
        SIRENA instance
    """
    sirena = SIRENA()
    if auto_fit:
        sirena.load_data().fit()
    return sirena


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("=" * 60)
    print("=== SIRENA Unified API Demo ===")
    print("=" * 60)

    # Создание и инициализация
    sirena = create_sirena()
    print(sirena.info())

    # Текущий режим
    print(f"\n📊 Текущий режим: {sirena.regime}")

    # Прогноз
    print("\n📈 Прогноз на 12 месяцев:")
    fc = sirena.forecast(12, scenario='auto')
    print(fc.summary())

    # Сравнение сценариев
    print("\n📊 Сравнение сценариев:")
    df = sirena.quick_forecast(12)
    print(df.head())

    print("\n✅ Demo completed!")
