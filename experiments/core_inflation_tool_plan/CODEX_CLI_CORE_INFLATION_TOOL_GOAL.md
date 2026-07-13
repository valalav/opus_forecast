# СИРЕНА-КБР Codex CLI Core Inflation Tool Goal

Short launch prompt:

```text
/goal Follow experiments/core_inflation_tool_plan/CODEX_CLI_CORE_INFLATION_TOOL_GOAL.md exactly. Act as controller for the core inflation analysis tool; coordinate isolated subagents; verify patches; keep all implementation isolated under experiments/core_inflation_tool; do not change production data or existing model code; stop only when the final acceptance gate passes or a blocker is documented.
```

## Mission

Build an experimental, reproducible tool for analyzing stable/core inflation in KBR. The tool must compute multiple core inflation indicators, explain monthly jumps through component contributions, and detect input-data failures such as MoM/YoY contamination.

The first implementation must stay isolated under:

```text
experiments/core_inflation_tool/
```

Do not integrate into `sirena/`, `scripts/`, dashboard, `data/`, or `assets/charts/` during this goal unless explicitly instructed in a later goal.

## Baseline To Verify

Start by reading:

- `docs/index.md`
- `docs/reports/core_inflation_xlsm_audit_2026-05-27.md`
- `experiments/core_inflation_tool_plan/README.md`
- `experiments/core_inflation_tool_plan/PROJECT_PLAN.md`
- `experiments/core_inflation_tool_plan/METHODOLOGY.md`

Inspect source candidates without editing them:

- `data/inflation_data.csv`
- `data/mom_sa_kbr.csv`
- `data/kbr_indices.csv`
- `data/access_weights.csv`
- `data/items_names.csv`
- `data/micro_sprav.csv`
- `data/raw/sub_mom.csv`
- `data/raw/subcomp_sprav.csv`
- `data/trimmed_mean_cpi.csv`
- `data/sticky_price_index.csv`
- `data/inflation_persistence.csv`

Baseline commands:

```bash
git status --short --branch
python3 - <<'PY'
import pandas as pd
from pathlib import Path
for p in [
    "data/inflation_data.csv",
    "data/mom_sa_kbr.csv",
    "data/access_weights.csv",
    "data/micro_sprav.csv",
]:
    print(p, Path(p).exists())
PY
```

## Hard Rules

- Do not edit production data under `data/`.
- Do not edit existing model code under `sirena/`.
- Do not edit existing operational scripts under `scripts/`.
- Do not regenerate `assets/charts/`.
- Write implementation only under `experiments/core_inflation_tool/`.
- Write planning/status docs only under `experiments/core_inflation_tool_plan/`.
- All generated outputs must go under `experiments/core_inflation_tool/outputs/`.
- Subagents are untrusted until the controller verifies their diffs and tests.
- No destructive git commands.
- No claims of verification without command evidence.

## Model Roles

Use external/read-only model reviews only if available and economical:

- local-llama: review method edge cases and data diagnostics.
- GLM 5.1: false-pass/correctness review.
- Kimi K2.6: code reasoning and patch critique.
- MiniMax: report readability critique.

If external models are unavailable, record `expected_skip` in the review ledger and proceed with local verification.

## Baseline Gate

Run before implementation:

```bash
git status --short --branch
python3 -m py_compile scripts/calc_trimmed_mean.py scripts/calc_sticky_price.py scripts/analyze_persistence.py
```

If this fails due to unrelated baseline issues, document the failure in `experiments/core_inflation_tool_plan/BASELINE_NOTES.md` and continue only if the failure does not block the isolated experiment.

## Phase 1. Read-Only Design Panel

Prompt template for optional model review:

```text
Read only. We are building an isolated Python tool for KBR stable/core inflation. It must compute exclusion core, weighted trimmed mean, weighted median, and diagnostics catching MoM/YoY contamination. Review the methodology in experiments/core_inflation_tool_plan/METHODOLOGY.md for missing acceptance criteria, false-pass risks, and data-quality checks. Max 700 words.
```

Record outcomes in:

```text
experiments/core_inflation_tool_plan/REVIEW_LEDGER.md
```

The controller must verify every useful model finding locally.

## Worktree Layout

Suggested controller commands from the main checkout:

```bash
mkdir -p ../sirena-kbr-worktrees
git worktree add ../sirena-kbr-worktrees/core-inflation-data -b codex/core-inflation-data main
git worktree add ../sirena-kbr-worktrees/core-inflation-indicators -b codex/core-inflation-indicators main
git worktree add ../sirena-kbr-worktrees/core-inflation-report -b codex/core-inflation-report main
```

Before using any old worktree:

```bash
git -C <worktree> status --short --branch
git -C <worktree> log --oneline -5
```

## Worker A. Data Inventory And Loaders

Branch:

```text
codex/core-inflation-data
```

Write scope:

- `experiments/core_inflation_tool/core_inflation/loaders.py`
- `experiments/core_inflation_tool/core_inflation/weights.py`
- `experiments/core_inflation_tool/config/`
- `experiments/core_inflation_tool/tests/test_loaders.py`
- `experiments/core_inflation_tool/tests/test_weights.py`
- `experiments/core_inflation_tool_plan/DATA_INVENTORY.md`

Forbidden:

- `data/`
- `sirena/`
- `scripts/`
- `assets/`

Tasks:

1. Create source inventory for monthly CPI rows, component names, weights, and hierarchy fields.
2. Implement read-only loaders for selected CSV inputs.
3. Implement weight normalization helpers.
4. Create config files for exclusion groups and input paths.
5. Add tests with tiny synthetic fixtures.

Acceptance criteria:

- Loaders never write to `data/`.
- Tests cover comma decimal and semicolon CSV parsing.
- Weight normalization handles zero/NaN weights with explicit errors.
- `DATA_INVENTORY.md` explains chosen MVP sources and rejected alternatives.

Worker prompt:

```text
/goal You are Worker A for СИРЕНА-КБР core inflation tool. Follow Worker A in experiments/core_inflation_tool_plan/CODEX_CLI_CORE_INFLATION_TOOL_GOAL.md only. You may edit only your write-scope files. Do not touch forbidden files. Implement data inventory/loaders/weights, run targeted tests, and stop. Do not merge or push.
```

## Worker B. Indicators And Diagnostics

Branch:

```text
codex/core-inflation-indicators
```

Write scope:

- `experiments/core_inflation_tool/core_inflation/indicators.py`
- `experiments/core_inflation_tool/core_inflation/diagnostics.py`
- `experiments/core_inflation_tool/tests/test_indicators.py`
- `experiments/core_inflation_tool/tests/test_diagnostics.py`

Forbidden:

- `data/`
- `sirena/`
- `scripts/`
- `assets/`
- Worker A files unless coordinating through controller.

Tasks:

1. Implement index-to-growth conversion.
2. Implement weighted aggregation.
3. Implement exclusion core.
4. Implement weighted trimmed mean.
5. Implement weighted median.
6. Implement MoM/YoY contamination diagnostic.
7. Implement jump and range diagnostics.

Acceptance criteria:

- Synthetic weighted aggregation returns exact expected values.
- Synthetic weighted median handles uneven weights.
- Synthetic trimmed mean handles tail removal by weight.
- Diagnostic fails when two input matrices are numerically identical.
- No dependency on Excel.

Worker prompt:

```text
/goal You are Worker B for СИРЕНА-КБР core inflation tool. Follow Worker B in experiments/core_inflation_tool_plan/CODEX_CLI_CORE_INFLATION_TOOL_GOAL.md only. You may edit only your write-scope files. Do not touch forbidden files. Implement indicators/diagnostics, run targeted tests, and stop. Do not merge or push.
```

## Worker C. CLI, Contributions, Reports

Branch:

```text
codex/core-inflation-report
```

Write scope:

- `experiments/core_inflation_tool/core_inflation/contributions.py`
- `experiments/core_inflation_tool/core_inflation/report.py`
- `experiments/core_inflation_tool/core_inflation/cli.py`
- `experiments/core_inflation_tool/README.md`
- `experiments/core_inflation_tool/tests/test_contributions.py`
- `experiments/core_inflation_tool/tests/test_cli.py`

Forbidden:

- `data/`
- `sirena/`
- `scripts/`
- `assets/`
- Worker A/B files unless coordinating through controller.

Tasks:

1. Implement contribution tables.
2. Implement jump report Markdown.
3. Implement CLI that writes outputs under `experiments/core_inflation_tool/outputs/`.
4. Include config snapshot in each run output.
5. Document usage and limitations.

Acceptance criteria:

- CLI refuses output paths outside experiment folder unless an explicit test fixture overrides it.
- Output includes series, diagnostics, contributions, jump report, and config snapshot.
- Reports clearly mark failed diagnostics.

Worker prompt:

```text
/goal You are Worker C for СИРЕНА-КБР core inflation tool. Follow Worker C in experiments/core_inflation_tool_plan/CODEX_CLI_CORE_INFLATION_TOOL_GOAL.md only. You may edit only your write-scope files. Do not touch forbidden files. Implement CLI/contributions/reporting, run targeted tests, and stop. Do not merge or push.
```

## Review Panel

For each worker diff, optional read-only review prompt:

```bash
pi --provider omniroute --model "<MODEL>" --no-tools --no-skills --no-prompt-templates --no-session "@<PATCH_OR_FILE>" -p "Read only. Review this patch for correctness risks, false-pass tests, data-quality failures, regression risks, and missing acceptance criteria. Return P1/P2 findings with evidence and concrete fixes. Max 900 words."
```

Controller rules:

- Never accept a finding without local verification.
- Record real findings, hallucinations, and timeouts in `REVIEW_LEDGER.md`.
- Do not spend model budget on repetitive bulk review.

## Integration Order

1. Integrate Worker A first: loaders/config/weights define contracts.
2. Integrate Worker B second: indicators depend on data contracts.
3. Integrate Worker C third: CLI/report depends on A+B.
4. Run targeted tests after each integration.
5. Run final gate after all integration.

If worker patches conflict, manually port the accepted parts into the main checkout.

## Required Trial

Fixture trial:

- create tiny synthetic fixture inside tests;
- prove MoM/YoY identical diagnostic fails;
- prove weighted aggregation, weighted median, and trimmed mean expected values.

Real-data trial:

- run CLI against selected repo CSV inputs;
- output only under `experiments/core_inflation_tool/outputs/latest`;
- if source ambiguity blocks real trial, emit `expected_skip` with a clear reason and still pass fixture tests.

An all-skipped real-data trial is not a successful product run; it is a documented blocker or limitation.

## Final Acceptance Gate

Run from main checkout:

```bash
git status --short --branch
python3 -m pytest experiments/core_inflation_tool/tests
python3 -m py_compile $(find experiments/core_inflation_tool -name '*.py' -print)
python3 -m experiments.core_inflation_tool.core_inflation.cli \
  --config experiments/core_inflation_tool/config/core_inflation_config.yaml \
  --output experiments/core_inflation_tool/outputs/latest
test -f experiments/core_inflation_tool/outputs/latest/core_inflation_series.csv
test -f experiments/core_inflation_tool/outputs/latest/core_inflation_diagnostics.csv
test -f experiments/core_inflation_tool/outputs/latest/core_inflation_contributions.csv
test -f experiments/core_inflation_tool/outputs/latest/core_inflation_jump_report.md
git diff --check
```

If `pytest` is unavailable, use:

```bash
python3 -m unittest discover -s experiments/core_inflation_tool/tests
```

and record the substitution.

## Documentation Requirements

Update or create:

- `experiments/core_inflation_tool/README.md`
- `experiments/core_inflation_tool_plan/DATA_INVENTORY.md`
- `experiments/core_inflation_tool_plan/REVIEW_LEDGER.md`
- `experiments/core_inflation_tool_plan/IMPLEMENTATION_NOTES.md`

Do not update top-level docs until the experiment is accepted.

## Commit And Push

Suggested commits:

1. `Add isolated core inflation tool plan`
2. `Add experimental core inflation loaders and indicators`
3. `Add core inflation CLI diagnostics and reports`

Push only after final acceptance gate passes. If blocked, do not claim completion; report blocker, evidence, attempted checks, and safest next action.

## Final Report Contract

Final response must include:

- commit hash/range if committed;
- changed files by area;
- worker/subagent summary;
- model reviews used, useful findings, hallucinations, and timeouts;
- checks passed;
- fixture and real-data trial evidence;
- remaining risks;
- exact next recommended task.
