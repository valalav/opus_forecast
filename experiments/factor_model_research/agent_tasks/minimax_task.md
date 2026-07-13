# MiniMax task: experiment matrix and implementation plan

You are an implementation-planning subagent for the `sirena-kbr` repository.

Repository context:

- Current factor model document: `docs/FACTOR_MODEL_RESEARCH.md`.
- Current model: `sirena/models/factor_policy.py`.
- Research runner: `experiments/factor_model_research/factor_policy_backtest.py`.
- Current selected baseline: robust seasonal FAVAR with PCA factors from
  `Food`, `NonFood`, `Services`, `USD`, `Ki_i`, `Ruonia`, 2 factors, lag 1,
  Huber equation estimation.
- Baseline MAE: h=1 0.371, h=2 0.425, h=12 0.439.

Task:

Design a compact experiment matrix for improving the factor model without
rewriting the project. Focus on variants that can be implemented by extending
the existing runner and model patterns.

Constraints:

- Do not edit files.
- Do not use future actual macro values.
- Do not propose random trajectory noise.
- Prefer variants that fit expanding/rolling training windows.

Output format:

1. Proposed experiment matrix table.
2. Exact selection rule.
3. Minimal code-change plan.
4. Verification commands.
5. Stop/go recommendation.
