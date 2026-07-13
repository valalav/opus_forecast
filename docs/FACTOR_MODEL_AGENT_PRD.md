# Factor model agent project PRD

Дата: 2026-06-09

## 1. Контекст

В репозитории уже есть обязательная факторная линия:

- модель: `sirena/models/factor_policy.py`;
- registry key: `factor_policy`;
- исследовательский runner: `experiments/factor_model_research/factor_policy_backtest.py`;
- итоговый документ: `docs/FACTOR_MODEL_RESEARCH.md`;
- финальный rolling run: `experiments/factor_model_research/runs/factor_policy_rolling/`.

Текущий выбранный baseline: robust seasonal FAVAR на PCA-факторах из `Food`,
`NonFood`, `Services`, `USD`, `Ki_i`, `Ruonia`, 2 фактора, lag 1, Huber
equation estimation.

Rolling-метрики baseline:

| Horizon | MAE | RMSE | Coverage <= 0.5 | Non-shock MAE |
| ---: | ---: | ---: | ---: | ---: |
| h=1 | 0.371 | 0.619 | 74.0% | 0.334 |
| h=2 | 0.425 | 0.710 | 73.0% | 0.355 |
| h=12 | 0.439 | 0.718 | 70.0% | 0.374 |

Задача проекта - не обязательно обогнать production ensemble, а сделать
реальную, защищаемую факторную модель и показать, что были проверены
альтернативы.

## 2. Цели

1. Расширить исследование факторных моделей без утечки будущих фактов.
2. Проверить несколько профессиональных альтернатив текущему FAVAR baseline.
3. Выбрать оптимальную конфигурацию по h=1, h=2, h=12 и реалистичности h=12
   траекторий.
4. Подготовить управленческий отчет: что протестировано, что выбрано, почему
   модель годится как обязательная факторная линия, даже если она слабее
   основных моделей.

## 3. Не цели

- Не заменять production ensemble без доказанного улучшения.
- Не использовать будущие фактические макро-переменные.
- Не делать однооконный overfit.
- Не добавлять случайный шум ради "красивой" траектории.
- Не переписывать общий backtest framework без необходимости.

## 4. Кандидаты для тестирования

Минимальный список:

1. Current baseline: robust seasonal FAVAR lean f2 lag1.
2. Block FAVAR:
   - component factor from `Food`, `NonFood`, `Services`;
   - macro/financial factor from `USD`, `Ki_i`, `Ruonia`, optionally
     `Deposits`, `RetailReal`;
   - CPI plus block factors in VAR/HAR-style equation.
3. Supervised factor model:
   - PLS factors trained only on expanding/rolling training window;
   - Huber/Ridge forecast equation.
4. Sparse PCA or constrained PCA:
   - sparse/loadings-stable factors;
   - compare interpretability and trajectory stability.
5. Factor-Augmented Ridge/Huber:
   - PCA factors plus seasonal dummies and selected lags;
   - direct h=1/h=2/h=12 equations where appropriate.
6. DFM/Kalman control:
   - only if convergence is reliable;
   - report as control, not preferred model by default.

## 5. Acceptance gates

Model quality gates:

- rolling h=1, h=2, h=12 metrics saved;
- h=12 trajectory diagnostics saved;
- selected model has no explosive paths;
- selected model is leakage-safe by construction;
- results are reproducible from a documented command.

Repository gates:

- if code changes are made, run targeted py_compile;
- if forecast artifacts change, run `scripts/precompute_forecasts.py` and
  `scripts/generate_charts.py`;
- if dashboard-visible outputs change, run `scripts/verify_all_tabs.py`.

## 6. Agent roles

| Agent | Model/channel | Role | Output |
| --- | --- | --- | --- |
| Qwen local | `http://127.0.0.1:8082`, `qwen3.6-27b-mtp-q5` | Narrow analyst | Short critique and 3-5 practical ideas |
| MiniMax | `minimax/MiniMax-M3` via `pi` | Implementation planner | Experiment matrix and code-level change plan |
| Gemini | `omniroute/gemini-cli/gemini-3.1-pro-preview` via `pi` | Methodology reviewer | Leakage/backtest/selection risk review |
| Nemotron | `omniroute/nvidia/nemotron-3-super-120b-a12b` via `pi` | Econometric reviewer | Factor-model alternatives and final critique |

## 7. Agent operating rules

- Give each agent at most one narrow task.
- Give file snippets and metrics, not the whole repository.
- For direct `pi` calls use `--no-tools --no-session --no-context-files
  --no-skills`.
- Avoid Pi `/goals-set` until auto-continue behavior is controlled.
- Treat every agent output as untrusted until locally verified.
- Save useful reports under `experiments/factor_model_research/agent_reports/`.

## 8. Next work packages

1. Agent review package:
   - run Qwen, MiniMax, Gemini, Nemotron on narrow task cards;
   - save reports;
   - synthesize agreed and conflicting recommendations.
2. Experiment design package:
   - choose 2-3 variants with best expected value;
   - update runner only if needed;
   - define exact metric-selection rule before running.
3. Implementation package:
   - add selected variant(s) using existing `FactorPolicyForecaster` and
     `factor_policy_backtest.py` patterns;
   - avoid touching dashboard unless selected model changes forecast output.
4. Verification/report package:
   - run reproducible backtests;
   - refresh `docs/FACTOR_MODEL_RESEARCH.md`;
   - produce short management note.
