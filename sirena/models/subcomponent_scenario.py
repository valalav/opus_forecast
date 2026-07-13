#!/usr/bin/env python3
"""
СУБКОМПОНЕНТНАЯ СЦЕНАРНАЯ МОДЕЛЬ v1.0
=====================================

Объединяет:
- SubcomponentMultiForecaster для базового прогноза (bottom-up)
- Субкомпонент-специфичные IRF для сценарного анализа ставки

Каждый субкомпонент имеет свою чувствительность к ставке:
- Кредитозависимые (авто, мебель, техника): высокая чувствительность
- Импортозависимые (одежда, обувь, рыба): средняя чувствительность
- Базовые продукты (мясо, молоко): низкая чувствительность
- Регулируемые (ЖКХ): минимальная чувствительность

Архитектура:
```
┌─────────────────────────────────────────────────────────────┐
│                    Сценарий Ki (+2 п.п.)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              SubcomponentScenarioForecaster                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │Кредитозав.  │  │Импортозав.  │  │ Базовые    │         │
│  │IRF: -0.15   │  │IRF: -0.10   │  │IRF: -0.02  │         │
│  │Lag: 4 мес   │  │Lag: 3 мес   │  │Lag: 6 мес  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │               │                  │
│         ▼                ▼               ▼                  │
│  Baseline + Effect для каждого субкомпонента                │
│         │                │               │                  │
│         └────────────────┼───────────────┘                  │
│                          ▼                                  │
│              Взвешенная агрегация (веса)                    │
│                          │                                  │
│                          ▼                                  │
│               CPI Total: -0.59% (vs base)                   │
└─────────────────────────────────────────────────────────────┘
```
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
from pathlib import Path
import warnings

# Импорт базовых моделей
try:
    from .subcomponent_multi import SubcomponentMultiForecaster
except ImportError:
    try:
        from sirena.models.subcomponent_multi import SubcomponentMultiForecaster
    except ImportError:
        SubcomponentMultiForecaster = None

try:
    from .scenario_rate import TransmissionParams, ScenarioRateModel
except ImportError:
    try:
        from sirena.models.scenario_rate import TransmissionParams, ScenarioRateModel
    except ImportError:
        TransmissionParams = None
        ScenarioRateModel = None


@dataclass
class SubcomponentSensitivity:
    """Параметры чувствительности субкомпонента к ставке."""
    code: str
    name: str
    group: str  # 'credit', 'import', 'basic', 'regulated', 'services', 'default'
    peak_effect: float
    peak_lag: int
    duration: int = 18


# ============================================================================
# СПРАВОЧНИК ЧУВСТВИТЕЛЬНОСТИ СУБКОМПОНЕНТОВ
# ============================================================================

SUBCOMPONENT_SENSITIVITY = {
    # -------------------------------------------------------------------------
    # КРЕДИТОЗАВИСИМЫЕ (high sensitivity)
    # Трансмиссия через кредитный канал: 3-6 месяцев
    # -------------------------------------------------------------------------
    '41': SubcomponentSensitivity('41', 'Телерадиотовары', 'credit', -0.15, 4, 18),
    '31': SubcomponentSensitivity('31', 'Персональные компьютеры', 'credit', -0.15, 4, 18),
    '20': SubcomponentSensitivity('20', 'Мебель', 'credit', -0.12, 4, 18),
    '43': SubcomponentSensitivity('43', 'Трикотажные изделия', 'credit', -0.10, 5, 18),

    # -------------------------------------------------------------------------
    # ИМПОРТОЗАВИСИМЫЕ (medium sensitivity)
    # Трансмиссия через валютный канал: 2-4 месяца
    # -------------------------------------------------------------------------
    '29': SubcomponentSensitivity('29', 'Одежда', 'import', -0.10, 3, 15),
    '30': SubcomponentSensitivity('30', 'Обувь', 'import', -0.10, 3, 15),
    '34': SubcomponentSensitivity('34', 'Рыба', 'import', -0.08, 4, 15),
    '50': SubcomponentSensitivity('50', 'Чай, кофе, какао', 'import', -0.08, 4, 15),

    # -------------------------------------------------------------------------
    # БАЗОВЫЕ ПРОДУКТЫ (low sensitivity)
    # Спрос неэластичен: 6-12 месяцев, слабый эффект
    # -------------------------------------------------------------------------
    '26': SubcomponentSensitivity('26', 'Мясо', 'basic', -0.03, 6, 18),
    '17': SubcomponentSensitivity('17', 'Молоко', 'basic', -0.02, 6, 18),
    '18': SubcomponentSensitivity('18', 'Масло, жиры', 'basic', -0.02, 6, 18),
    '33': SubcomponentSensitivity('33', 'Плодоовощи', 'basic', -0.02, 8, 24),
    '35': SubcomponentSensitivity('35', 'Крупы', 'basic', -0.02, 6, 18),
    '52': SubcomponentSensitivity('52', 'Яйца', 'basic', -0.02, 6, 18),

    # -------------------------------------------------------------------------
    # РЕГУЛИРУЕМЫЕ ТАРИФЫ (minimal sensitivity)
    # Государственное регулирование: 12+ месяцев, минимальный эффект
    # -------------------------------------------------------------------------
    '14': SubcomponentSensitivity('14', 'ЖКХ', 'regulated', -0.01, 12, 24),
    '12': SubcomponentSensitivity('12', 'Квартплата', 'regulated', -0.01, 12, 24),
    '15': SubcomponentSensitivity('15', 'Электроэнергия', 'regulated', -0.01, 12, 24),
    '16': SubcomponentSensitivity('16', 'Газ', 'regulated', -0.01, 12, 24),

    # -------------------------------------------------------------------------
    # УСЛУГИ (medium-low sensitivity)
    # -------------------------------------------------------------------------
    '44': SubcomponentSensitivity('44', 'Услуги образования', 'services', -0.04, 6, 18),
    '55': SubcomponentSensitivity('55', 'Медицинские услуги', 'services', -0.04, 6, 18),
    '67': SubcomponentSensitivity('67', 'Туризм', 'services', -0.06, 5, 18),
}

# Дефолтные параметры для субкомпонентов без явного указания
DEFAULT_SENSITIVITY = SubcomponentSensitivity('default', 'Default', 'default', -0.05, 6, 18)


class SubcomponentScenarioForecaster:
    """
    Сценарная модель с субкомпонент-специфичной чувствительностью к ставке.

    Объединяет:
    1. SubcomponentMultiForecaster для базового прогноза
    2. Субкомпонент-специфичные IRF для эффекта ставки

    Примеры:
        >>> model = SubcomponentScenarioForecaster()
        >>> model.fit(df)
        >>>
        >>> # Сценарий повышения ставки на 2 п.п.
        >>> result = model.forecast_scenario(horizon=12, ki_change=+2.0)
        >>> print(f"Эффект на CPI: {result['total_effect']:.2f}%")
        >>>
        >>> # Декомпозиция по группам
        >>> decomp = result['group_decomposition']
        >>> for group, effect in decomp.items():
        >>>     print(f"{group}: {effect:.3f}%")
    """

    name = "subcomponent_scenario"

    def __init__(
        self,
        horizon: int = 1,
        train_start: str = '2016-01-01',
        random_state: int = 42,
        use_asymmetric: bool = False,
        use_calibrated: bool = True
    ):
        """
        Инициализация модели.

        Args:
            horizon: Горизонт прогноза
            train_start: Начало обучающей выборки
            random_state: Seed для воспроизводимости
            use_asymmetric: Использовать асимметричный IRF
            use_calibrated: Использовать калиброванные параметры из JSON
        """
        self.horizon = horizon
        self.train_start = train_start
        self.random_state = random_state
        self.use_asymmetric = use_asymmetric
        self.use_calibrated = use_calibrated
        self._is_fitted = False

        self.base_model = None
        self.subcomponent_irfs = {}
        self.weights = {}
        self._calibrated_params = {}

    def _load_calibrated_sensitivity(self) -> Dict:
        """Загрузка калиброванных параметров из JSON."""
        import json
        json_path = Path(__file__).parent.parent.parent / 'data' / 'calibrated_sensitivity.json'
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _build_irf(self, sensitivity: SubcomponentSensitivity) -> np.ndarray:
        """
        Построение IRF для субкомпонента.

        Использует hump-shaped функцию как в ScenarioRateModel.
        """
        duration = sensitivity.duration
        peak_effect = sensitivity.peak_effect
        peak_lag = sensitivity.peak_lag
        start_lag = 2

        irf = np.zeros(duration)

        for s in range(start_lag, duration):
            x = (s - start_lag) / max(peak_lag - start_lag, 1)
            if x > 0:
                irf[s] = peak_effect * x * np.exp(1 - x)

        return irf

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги'):
        """
        Обучение модели.

        Args:
            df: DataFrame с макро-данными
            target_col: Целевая колонка
        """
        if SubcomponentMultiForecaster is None:
            raise ImportError("SubcomponentMultiForecaster not available")

        # 1. Обучаем базовую модель
        self.base_model = SubcomponentMultiForecaster(
            horizon=self.horizon,
            train_start=self.train_start,
            random_state=self.random_state
        )
        self.base_model.fit(df, target_col)

        # 2. Загружаем веса субкомпонентов
        self.weights = self.base_model.weights.copy()

        # 3. Загружаем калиброванные параметры (если включено)
        if self.use_calibrated:
            self._calibrated_params = self._load_calibrated_sensitivity()

        # 4. Строим IRF для каждого субкомпонента
        for code in self.base_model.subcomponent_models.keys():
            if self.use_calibrated and code in self._calibrated_params:
                # Используем калиброванные параметры из VAR
                params = self._calibrated_params[code]
                sensitivity = SubcomponentSensitivity(
                    code=code,
                    name=f"Субкомп {code}",
                    group='calibrated',
                    peak_effect=params['peak_effect'],
                    peak_lag=params['peak_lag'],
                    duration=params.get('duration', 18)
                )
            elif code in SUBCOMPONENT_SENSITIVITY:
                sensitivity = SUBCOMPONENT_SENSITIVITY[code]
            else:
                sensitivity = DEFAULT_SENSITIVITY
            self.subcomponent_irfs[code] = self._build_irf(sensitivity)

        self.macro_df = df.copy()
        self._is_fitted = True
        return self

    def _compute_effect(
        self,
        horizon: int,
        ki_change: Union[float, np.ndarray],
        irf: np.ndarray
    ) -> np.ndarray:
        """Вычисление эффекта от ставки через свёртку с IRF."""
        # Траектория изменения Ki
        if isinstance(ki_change, (int, float)):
            ki_path = np.zeros(horizon)
            ki_path[0] = ki_change
        else:
            ki_path = np.array(ki_change)
            if len(ki_path) < horizon:
                ki_path = np.concatenate([
                    ki_path,
                    np.full(horizon - len(ki_path), 0)
                ])

        # Свёртка
        effect = np.zeros(horizon)
        for t in range(horizon):
            for s in range(min(t + 1, len(irf))):
                if t - s >= 0 and s < len(irf):
                    effect[t] += ki_path[t - s] * irf[s]

        return effect

    def forecast_scenario(
        self,
        horizon: int = 12,
        ki_change: Union[float, List[float], np.ndarray] = 0.0,
        return_decomposition: bool = True
    ) -> Dict:
        """
        Прогноз при сценарии изменения ставки.

        Args:
            horizon: Горизонт прогноза
            ki_change: Изменение Ki (п.п.)
            return_decomposition: Вернуть декомпозицию по субкомпонентам

        Returns:
            Dict с:
            - 'baseline': базовый прогноз (массив)
            - 'effect': эффект от ставки (массив)
            - 'total': итоговый прогноз (массив)
            - 'cumulative_effect': кумулятивный эффект
            - 'subcomponent_effects': эффект по субкомпонентам
            - 'group_decomposition': эффект по группам
        """
        if not self._is_fitted:
            raise ValueError("Сначала вызовите fit()")

        # 1. Базовый прогноз
        baseline = self.base_model.forecast(horizon)

        # 2. Расчёт эффекта по субкомпонентам
        subcomp_effects = {}
        subcomp_totals = {}

        for code in self.base_model.subcomponent_models.keys():
            irf = self.subcomponent_irfs.get(code, self._build_irf(DEFAULT_SENSITIVITY))
            effect = self._compute_effect(horizon, ki_change, irf)
            subcomp_effects[code] = effect
            subcomp_totals[code] = np.sum(effect)

        # 3. Агрегация эффектов по весам
        total_weight = sum(self.weights[c] for c in subcomp_effects.keys())
        weighted_effect = np.zeros(horizon)

        for code, effect in subcomp_effects.items():
            weight = self.weights.get(code, 0) / total_weight
            weighted_effect += weight * effect

        # 4. Декомпозиция по группам
        group_effects = {
            'credit': 0.0,      # Кредитозависимые
            'import': 0.0,      # Импортозависимые
            'basic': 0.0,       # Базовые продукты
            'regulated': 0.0,   # Регулируемые
            'services': 0.0,    # Услуги
            'other': 0.0        # Остальные
        }

        for code, effect_sum in subcomp_totals.items():
            weight = self.weights.get(code, 0) / total_weight

            if code in SUBCOMPONENT_SENSITIVITY:
                group = SUBCOMPONENT_SENSITIVITY[code].group
            else:
                group = 'other'

            if group in group_effects:
                group_effects[group] += weight * effect_sum
            else:
                group_effects['other'] += weight * effect_sum

        # 5. Итоговый прогноз
        total = baseline + weighted_effect

        result = {
            'baseline': baseline,
            'effect': weighted_effect,
            'total': total,
            'cumulative_baseline': np.sum(baseline),
            'cumulative_effect': np.sum(weighted_effect),
            'cumulative_total': np.sum(total),
            'group_decomposition': group_effects
        }

        if return_decomposition:
            result['subcomponent_effects'] = subcomp_effects
            result['subcomponent_totals'] = subcomp_totals

        return result

    def compare_scenarios(
        self,
        horizon: int = 12,
        scenarios: Dict[str, Union[float, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Сравнение нескольких сценариев.

        Args:
            horizon: Горизонт
            scenarios: Dict {название: изменение Ki}

        Returns:
            DataFrame со сравнением
        """
        if scenarios is None:
            scenarios = {
                'Базовый': 0.0,
                'Повышение +2 п.п.': 2.0,
                'Повышение +4 п.п.': 4.0,
                'Снижение -2 п.п.': -2.0,
                'Снижение -4 п.п.': -4.0
            }

        results = []
        for name, ki_change in scenarios.items():
            fc = self.forecast_scenario(horizon, ki_change, return_decomposition=False)
            results.append({
                'Сценарий': name,
                'Ki change (п.п.)': ki_change if isinstance(ki_change, (int, float)) else ki_change[0],
                'Cum MoM (%)': fc['cumulative_total'],
                'Эффект (%)': fc['cumulative_effect'],
                'Baseline (%)': fc['cumulative_baseline']
            })

        return pd.DataFrame(results)

    def get_group_sensitivity_table(self) -> pd.DataFrame:
        """Таблица чувствительности по группам."""
        groups = {}
        for code, sensitivity in SUBCOMPONENT_SENSITIVITY.items():
            group = sensitivity.group
            if group not in groups:
                groups[group] = {
                    'count': 0,
                    'total_weight': 0,
                    'avg_peak_effect': 0,
                    'avg_peak_lag': 0
                }
            groups[group]['count'] += 1
            groups[group]['total_weight'] += self.weights.get(code, 0)
            groups[group]['avg_peak_effect'] += sensitivity.peak_effect
            groups[group]['avg_peak_lag'] += sensitivity.peak_lag

        rows = []
        for group, data in groups.items():
            rows.append({
                'Группа': group,
                'Субкомпонентов': data['count'],
                'Вес (%)': data['total_weight'] * 100,
                'Ср. peak effect': data['avg_peak_effect'] / data['count'],
                'Ср. peak lag': data['avg_peak_lag'] / data['count']
            })

        return pd.DataFrame(rows)

    def get_subcomponent_effects_table(
        self,
        horizon: int = 12,
        ki_change: float = 2.0
    ) -> pd.DataFrame:
        """Таблица эффектов по субкомпонентам."""
        fc = self.forecast_scenario(horizon, ki_change, return_decomposition=True)

        rows = []
        for code, effect_sum in fc['subcomponent_totals'].items():
            if code in SUBCOMPONENT_SENSITIVITY:
                sens = SUBCOMPONENT_SENSITIVITY[code]
                name = sens.name
                group = sens.group
            else:
                name = f"Субкомпонент {code}"
                group = 'other'

            rows.append({
                'Код': code,
                'Название': name,
                'Группа': group,
                'Вес (%)': self.weights.get(code, 0) * 100,
                f'Эффект при +{ki_change} п.п.': effect_sum
            })

        df = pd.DataFrame(rows)
        df = df.sort_values(f'Эффект при +{ki_change} п.п.')
        return df

    def calibrate_sensitivity(
        self,
        codes: Optional[List[str]] = None,
        max_lag: int = 12,
        save_to_json: bool = False
    ) -> Dict[str, Dict]:
        """
        Калибровка параметров чувствительности на данных через VAR IRF.

        Для каждого субкомпонента:
        1. Оцениваем VAR(y, Ki) на исторических данных
        2. Получаем IRF: response of y to Ki shock
        3. Извлекаем peak_effect и peak_lag

        Args:
            codes: Коды субкомпонентов для калибровки (None = все)
            max_lag: Максимальный лаг для IRF
            save_to_json: Сохранить результаты в JSON

        Returns:
            Dict с калиброванными параметрами для каждого субкомпонента
        """
        if not self._is_fitted:
            raise ValueError("Сначала вызовите fit()")

        try:
            from statsmodels.tsa.api import VAR
        except ImportError:
            raise ImportError("Для калибровки нужен statsmodels")

        # Загружаем субкомпонентные данные
        data_dir = Path(__file__).parent.parent.parent / 'data'
        sub_file = data_dir / 'raw' / 'subcomp.csv'
        if not sub_file.exists():
            sub_file = data_dir / 'raw' / 'sub_mom.csv'

        sub = pd.read_csv(sub_file, sep=';', decimal=',', encoding='utf-8-sig')
        date_col = 'Day' if 'Day' in sub.columns else 'Date'
        sub[date_col] = pd.to_datetime(sub[date_col], format='%d.%m.%Y')
        sub = sub.rename(columns={date_col: 'Date'}).set_index('Date').sort_index()
        sub.index = sub.index.to_period('M').to_timestamp()

        # Конвертируем колонки в строки (коды субкомпонентов)
        sub.columns = [str(c) for c in sub.columns]

        # Фиксируем десятичный разделитель (запятая → точка)
        for col in sub.columns:
            if sub[col].dtype == object:
                sub[col] = sub[col].astype(str).str.replace(',', '.').astype(float)

        # Ki из макро-данных
        if 'Ki_i' not in self.macro_df.columns:
            raise ValueError("Ki_i отсутствует в макро-данных")
        ki = self.macro_df['Ki_i'].copy()
        ki.index = ki.index.to_period('M').to_timestamp()

        calibrated = {}
        target_codes = codes or list(self.base_model.subcomponent_models.keys())

        print(f"\nКалибровка параметров чувствительности...")
        print(f"{'='*60}")

        for code in target_codes:
            if code not in sub.columns:
                continue

            try:
                # Подготовка данных
                y = sub[code]
                data = pd.DataFrame({'y': y, 'Ki': ki}).dropna()

                if len(data) < 36:  # Минимум 3 года
                    continue

                # Оценка VAR
                var_model = VAR(data)
                lag_order = var_model.select_order(maxlags=6)
                optimal_lag = lag_order.bic if hasattr(lag_order, 'bic') else 2
                optimal_lag = max(1, min(optimal_lag, 6))

                var_result = var_model.fit(optimal_lag)

                # IRF: response of y to Ki shock
                irf = var_result.irf(max_lag)
                response = irf.irfs[:, 0, 1]  # y response to Ki (0=y, 1=Ki)

                # Найти peak effect (минимум, т.к. отрицательный)
                peak_idx = np.argmin(response)
                peak_effect = response[peak_idx]

                # Если эффект положительный, товар не чувствителен к ставке
                if peak_effect > 0:
                    peak_effect = 0
                    peak_idx = 0

                calibrated[code] = {
                    'peak_effect': float(peak_effect),
                    'peak_lag': int(peak_idx),
                    'duration': max_lag,
                    'n_obs': len(data),
                    'var_lag': optimal_lag
                }

                if code in SUBCOMPONENT_SENSITIVITY:
                    name = SUBCOMPONENT_SENSITIVITY[code].name
                else:
                    name = f"Субкомпонент {code}"

                print(f"  {code} ({name}): peak_effect={peak_effect:.4f}, peak_lag={peak_idx}")

            except Exception as e:
                print(f"  {code}: ошибка калибровки - {e}")
                continue

        print(f"{'='*60}")
        print(f"Калибровано {len(calibrated)} субкомпонентов")

        # Сохранение в JSON
        if save_to_json:
            import json
            output_path = data_dir / 'calibrated_sensitivity.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(calibrated, f, indent=2, ensure_ascii=False)
            print(f"Сохранено: {output_path}")

        return calibrated

    def forecast_scenario_with_ci(
        self,
        horizon: int = 12,
        ki_change: float = 0.0,
        alpha: float = 0.1
    ) -> Dict:
        """
        Прогноз со сценарием и доверительными интервалами.

        CI рассчитываются на основе исторической волатильности ошибок.

        Args:
            horizon: Горизонт прогноза
            ki_change: Изменение Ki (п.п.)
            alpha: Уровень значимости (0.1 = 90% CI)

        Returns:
            Dict с mean, ci_lower, ci_upper для каждого горизонта
        """
        # Базовый прогноз
        result = self.forecast_scenario(horizon, ki_change, return_decomposition=False)

        # Оценка волатильности из бэктеста (упрощённо — используем фиксированную std)
        # Для h=1: std ≈ 0.3, для h=12: std ≈ 0.5
        std_by_horizon = 0.3 + 0.02 * np.arange(horizon)

        # z-score для alpha
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)

        # CI
        total = result['total']
        ci_lower = total - z * std_by_horizon
        ci_upper = total + z * std_by_horizon

        return {
            'mean': total,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'baseline': result['baseline'],
            'effect': result['effect'],
            'cumulative_effect': result['cumulative_effect'],
            'alpha': alpha,
            'std': std_by_horizon
        }


if __name__ == '__main__':
    print("=" * 60)
    print("=== Тест SubcomponentScenarioForecaster ===")
    print("=" * 60)

    # Загрузка данных
    data_dir = Path(__file__).parent.parent.parent / 'data'
    df = pd.read_csv(data_dir / 'inflation_data.csv', sep=';', decimal=',', encoding='utf-8-sig')
    for col in df.columns:
        if col != 'Date' and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df = df.set_index('Date').sort_index()

    # Создание и обучение модели
    print("\nОбучение модели...")
    model = SubcomponentScenarioForecaster()
    model.fit(df)
    print("Обучение завершено!")

    # Сравнение сценариев
    print("\n" + "=" * 60)
    print("Сравнение сценариев (h=12):")
    print("=" * 60)
    comparison = model.compare_scenarios(horizon=12)
    print(comparison.to_string(index=False))

    # Декомпозиция по группам
    print("\n" + "=" * 60)
    print("Декомпозиция эффекта +2 п.п. по группам:")
    print("=" * 60)
    fc = model.forecast_scenario(12, ki_change=2.0)
    for group, effect in fc['group_decomposition'].items():
        if effect != 0:
            print(f"  {group:15s}: {effect:+.4f}%")
    print(f"  {'ИТОГО':15s}: {fc['cumulative_effect']:+.4f}%")

    # Чувствительность по группам
    print("\n" + "=" * 60)
    print("Чувствительность по группам:")
    print("=" * 60)
    sens_table = model.get_group_sensitivity_table()
    print(sens_table.to_string(index=False))

    # Топ-10 субкомпонентов по эффекту
    print("\n" + "=" * 60)
    print("Топ-10 субкомпонентов по эффекту (при +2 п.п.):")
    print("=" * 60)
    effects_table = model.get_subcomponent_effects_table(12, 2.0)
    print(effects_table.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("Тест завершён!")
    print("=" * 60)
