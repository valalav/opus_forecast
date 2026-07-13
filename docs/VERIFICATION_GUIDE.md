# Верификация СИРЕНА-КБР

Скрипты и чеклисты для проверки корректности системы.

## Быстрая проверка

```bash
# Полная верификация всех вкладок и моделей
python3 scripts/verify_all_tabs.py
```

Результат должен быть: **✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ**

## Скрипты верификации

| Скрипт | Назначение |
|--------|------------|
| `scripts/verify_dashboard.py` | Проверка всех моделей |
| `scripts/verify_all_tabs.py` | Полная верификация 12 вкладок |
| `scripts/add_model_checklist.py` | Проверка добавления модели |
| `scripts/screenshot_dashboard.py` | Скриншоты всех вкладок |

## Что проверяется

### verify_dashboard.py

```bash
python3 scripts/verify_dashboard.py
```

Проверяет:
- `data/verify_forecast.csv` — прогнозы (все 10+ моделей)
- `data/verify_backtest.csv` — бэктест (Actual + все модели)
- `data/verify_summary.json` — status: "OK", errors: []

### verify_all_tabs.py

Проверяет:
1. `data/precomputed_forecasts.json` — все модели есть
2. `archive/results/backtest_h1_predictions.csv` — колонки моделей
3. `archive/results/backtest_h2_predictions.csv` — колонки моделей
4. `dashboard.py` — ALL_MODELS определён
5. `scripts/backtest_framework.py` — импорты моделей

## Чеклист после изменений

- [ ] Все модели присутствуют в verify_forecast.csv
- [ ] Все модели присутствуют в verify_backtest.csv
- [ ] Actual колонка не пустая в backtest
- [ ] MAE рассчитан для каждой модели
- [ ] status: "OK" в verify_summary.json
- [ ] errors: [] (пустой массив)

## Визуальная верификация

```bash
# Скриншоты всех 12 вкладок
python3 scripts/screenshot_dashboard.py
```

Результаты в `assets/screenshots/`:
- `tab1__Прогноз.png`
- `tab3__Бэктест.png`
- `tab9__Бэктест_h=1.png`
- и т.д.

## HTML графики

После изменений обязательно:

```bash
python3 scripts/precompute_forecasts.py
python3 scripts/generate_charts.py
```

Генерируются в `assets/charts/`:
- `forecasts.html`
- `backtest_h1_predictions.html`
- `backtest_h2_predictions.html`
- `metrics_comparison.html`
- `index.html`
