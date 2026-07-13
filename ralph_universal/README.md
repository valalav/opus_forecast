# Ralph Universal

**Ralph Universal** is a "Bulletproof" autonomous development loop for SIRENA-KBR v5.0. It uses a **Dual-Agent Architecture** to ensure high-quality code generation through rigorous verification.

## Quick Start

### 1. Configuration

Edit `ralph_universal/config.py` to configure:
- `PRIMARY_MODEL` / `CRITIC_MODEL` — LLM models (default: `opencode/glm-4.7-free`)
- `AGENT_CLI_CMD` — CLI command to invoke agent (default: `["opencode", "run"]`)
- `MAX_ITERATIONS` — maximum loop iterations (default: 50)

### 2. Define Your Tasks

Tasks are stored in `docs/prd.json`:

```json
{
  "project": "SIRENA-KBR v5.0 Test Coverage & Model Improvements",
  "user_stories": [
    {
      "id": 1,
      "title": "Test RidgeExtendedForecaster",
      "description": "Create unit tests for RidgeExtendedForecaster...",
      "acceptance_criteria": [
        "pytest tests/test_ridge_extended.py passes",
        "Coverage >= 80%"
      ],
      "priority": "high",
      "status": "TODO"
    }
  ]
}
```

**Task statuses:** `TODO` → `PENDING_REVIEW` → `DONE`

### 3. Run the System

```bash
python3 ralph_universal/orchestrator.py
```

This launches two parallel processes:
- **Worker** — picks TODO tasks, implements code, moves to PENDING_REVIEW
- **Critic** — verifies PENDING_REVIEW tasks, APPROVE → DONE or REJECT → TODO

## Architecture

```
ralph_universal/
├── config.py           # Configuration (models, paths, limits)
├── orchestrator.py     # Entry point - launches Worker & Critic
├── worker.py           # Executes tasks (Red-Green-Refactor)
├── critic.py           # Verifies tasks (Trust but Verify)
├── verify_installation.py
├── core/
│   ├── state.py        # StateManager (thread-safe JSON/log handling)
│   └── agent_wrapper.py # AgentWrapper (CLI interface to LLM)
└── docs/
    ├── prd.json        # Task definitions
    └── progress.txt    # Execution log
```

## Current Task List (30 tasks)

| Category | Count | Priority |
|----------|-------|----------|
| **Production Model Tests** | 9 | high |
| **Experimental Model Tests** | 9 | medium |
| **Integration Tests** | 2 | high |
| **New Models** | 3 | low-medium |
| **Improvements** | 3 | medium |
| **Infrastructure** | 4 | medium-high |

### Top Priority Tasks

| ID | Title | Status |
|----|-------|--------|
| 1 | Test RidgeExtendedForecaster | TODO |
| 6 | Test NGBoostForecaster | TODO |
| 10 | Test SubcomponentForecaster (h=1 leader) | TODO |
| 11 | Test SubcomponentMultiForecaster (h=12 leader) | TODO |
| 19 | Integration test: Full backtest pipeline | TODO |
| 28 | Data validation script | TODO |

## How It Works

### Worker Loop (Red-Green-Refactor)

1. Fetches `TODO` task from `prd.json`
2. Reads context: recent progress, methodology (GEMINI.md)
3. Formulates prompt with task + context
4. Executes via `opencode run`
5. Updates status to `PENDING_REVIEW`

### Critic Loop (Trust but Verify)

1. Fetches `PENDING_REVIEW` task
2. Runs verification (tests, file checks)
3. Parses output for `APPROVE` or `REJECT`
4. Updates status: APPROVE → `DONE`, REJECT → `TODO` + feedback

### State Management

- **Thread-safe**: File locks (`fcntl.flock`) prevent race conditions
- **Atomic writes**: PRD updates are all-or-nothing
- **Progress log**: Timestamped entries in `progress.txt`

## Commands

```bash
# Run the system
python3 ralph_universal/orchestrator.py

# Verify installation
python3 ralph_universal/verify_installation.py

# Check task status
cat ralph_universal/docs/prd.json | jq '.user_stories[] | {id, title, status}'

# View recent progress
tail -50 ralph_universal/docs/progress.txt
```

## Integration with SIRENA

This instance is configured to work with the SIRENA-KBR inflation forecasting system:

- **Target project**: `/home/valalav/_projects/sirena-kbr/`
- **Models to test**: 37 forecasters in `sirena/models/`
- **Test location**: `tests/`
- **Backtest framework**: `scripts/backtest_framework.py`

## See Also

- [GEMINI.md](GEMINI.md) — Methodology and protocols
- [SIRENA CLAUDE.md](../CLAUDE.md) — Main project documentation
