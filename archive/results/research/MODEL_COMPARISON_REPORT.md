# ОТЧЁТ: Сравнение моделей и признаков

Дата: 2025-12-28 23:17

## 1. СВОДКА

- Протестировано комбинаций: 273
- Моделей: 13
- Наборов признаков: 7

## 2. ЛУЧШИЕ МОДЕЛИ ПО ГОРИЗОНТАМ

### h=1

| # | Модель | Признаки | MAE | KPI |
|---|--------|----------|-----|-----|
| 1 | ElasticNet | Components | 0.468 | 35/46 |
| 2 | Lasso | Best_combined | 0.469 | 31/46 |
| 3 | Huber | Components | 0.470 | 34/46 |
| 4 | BayesianRidge | Components | 0.481 | 30/46 |
| 5 | Ridge_100 | Components | 0.482 | 32/46 |
| 6 | Huber | Best_combined | 0.484 | 32/46 |
| 7 | Ridge_100 | Best_combined | 0.485 | 31/46 |
| 8 | Ridge_10 | Components | 0.488 | 30/46 |
| 9 | ElasticNet | Best_combined | 0.491 | 31/46 |
| 10 | Ridge | Components | 0.495 | 30/46 |

### h=2

| # | Модель | Признаки | MAE | KPI |
|---|--------|----------|-----|-----|
| 1 | NGBoost | AR_extended | 0.534 | 30/45 |
| 2 | Lasso | Components | 0.535 | 30/45 |
| 3 | Lasso | IBVED_q | 0.537 | 30/45 |
| 4 | Lasso | AR_minimal | 0.537 | 30/45 |
| 5 | Lasso | AR_extended | 0.537 | 28/45 |
| 6 | Ridge_100 | IBVED_q | 0.540 | 31/45 |
| 7 | ElasticNet | AR_minimal | 0.543 | 30/45 |
| 8 | ElasticNet | IBVED_q | 0.543 | 30/45 |
| 9 | Ridge_100 | AR_minimal | 0.543 | 32/45 |
| 10 | NGBoost | Components | 0.545 | 31/45 |

### h=12

| # | Модель | Признаки | MAE | KPI |
|---|--------|----------|-----|-----|
| 1 | NGBoost | AR_extended | 0.334 | 26/35 |
| 2 | Huber | Components | 0.351 | 25/35 |
| 3 | Ridge_100 | AR_minimal | 0.356 | 27/35 |
| 4 | Ridge_100 | IBVED_q | 0.358 | 27/35 |
| 5 | Lasso | AR_minimal | 0.361 | 26/35 |
| 6 | Lasso | IBVED_q | 0.362 | 26/35 |
| 7 | Lasso | Components | 0.362 | 26/35 |
| 8 | Ridge_100 | Components | 0.364 | 25/35 |
| 9 | ElasticNet | AR_minimal | 0.365 | 27/35 |
| 10 | Lasso | AR_extended | 0.366 | 25/35 |

## 3. ЛУЧШИЕ НАБОРЫ ПРИЗНАКОВ (по среднему MAE)

| Набор | Средний MAE |
|-------|-------------|
| Components | 0.505 |
| AR_extended | 0.512 |
| IBVED_q | 0.527 |
| AR_minimal | 0.532 |
| Best_combined | 0.538 |
| Federal_macro | 0.555 |
| Regional | 0.571 |

## 4. ЛУЧШИЕ МОДЕЛИ (по среднему MAE)

| Модель | Средний MAE |
|--------|-------------|
| Lasso | 0.482 |
| Ridge_100 | 0.490 |
| ElasticNet | 0.492 |
| Huber | 0.498 |
| BayesianRidge | 0.514 |
| NGBoost | 0.523 |
| Ridge_10 | 0.530 |
| Ridge | 0.545 |
| LightGBM | 0.547 |
| RF | 0.557 |
| CatBoost | 0.563 |
| GradBoost | 0.592 |
| XGBoost | 0.614 |

## 5. РЕКОМЕНДАЦИИ

**Лучшая комбинация:** NGBoost + AR_extended на h=12
- MAE: 0.334
- KPI: 26/35