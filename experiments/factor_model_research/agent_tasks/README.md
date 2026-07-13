# Factor model agent task cards

Эти task cards предназначены для узких вызовов внешних моделей. Они не дают
агентам право менять файлы. Их задача - получить независимую критику и идеи,
после чего Codex проверяет и интегрирует только подтвержденные выводы.

Общий baseline:

- current model: robust seasonal FAVAR;
- information set: `Food`, `NonFood`, `Services`, `USD`, `Ki_i`, `Ruonia`;
- PCA factors: 2;
- lag: 1;
- estimator: Huber equations;
- h=1 MAE: 0.371;
- h=2 MAE: 0.425;
- h=12 MAE: 0.439;
- h=12 trajectory: no explosive paths, seasonal correlation 0.971.

Правила для всех агентов:

- Do not edit files.
- Do not request full repository context.
- Do not claim a variant is better without a leakage-safe rolling evaluation.
- Return concise markdown.
- Separate recommendations from assumptions.
