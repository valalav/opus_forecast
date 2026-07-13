# Ralph Universal: Architecture & Lessons Learned

> **Version**: 1.2 | **Updated**: 2026-01-22

## Overview

Ralph Universal is a **dual-loop autonomous agent system** for iterative development. It consists of:
- **Worker**: Executes tasks, writes code, runs tests
- **Critic**: Verifies work, approves or rejects with feedback
- **State Manager**: Thread-safe task state with race condition protection

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    prd.json (Task Queue)                │
│  ┌──────────┐    ┌───────────────┐    ┌──────────┐     │
│  │   TODO   │ ──▶│ PENDING_REVIEW│ ──▶│   DONE   │     │
│  └──────────┘    └───────────────┘    └──────────┘     │
│       ▲                │                   │           │
│       │                │                   │           │
│       └────────────────┘                   │           │
│         (on REJECT)                        │           │
└─────────────────────────────────────────────────────────┘
         ▲                    │
         │                    ▼
    ┌─────────┐         ┌──────────┐
    │ Worker  │ ◀──────▶│  Critic  │
    │(opencode)│         │(opencode)│
    └─────────┘         └──────────┘
```

---

## Critical Bugs Fixed (v1.1)

### 1. Race Condition Protection

**Problem**: Worker could override Critic's rejection by setting PENDING_REVIEW before feedback was processed.

**Solution** (`state.py`):
```python
# Don't allow PENDING_REVIEW if task has rejection feedback
if status == "PENDING_REVIEW" and old_feedback and "Reject" in old_feedback:
    return  # Block - let Worker handle rejection first
```

### 2. Orphaned Feedback Bug

**Problem**: Tasks marked DONE but with `feedback: "Rejected..."` - inconsistent state.

**Solution**: Critic now clears rejection feedback when approving:
```python
elif status == "DONE":
    story["feedback"] = "Approved by Critic"
```

### 3. Process "Death" Due to MAX_ITERATIONS Limit (v1.2 - 2026-01-22)

**Incident**: Orchestrator reported "Critic process died" every ~1.5 minutes.

**Root Cause**: `MAX_ITERATIONS = 50` in `config.py`. Worker/Critic ran 50 loops × 2 sec = 100 sec, then exited normally. Orchestrator interpreted normal exit as "death".

**Symptoms**:
```
💤 checking tasks... No pending tasks to review.
⚠️  Critic process died. Attempting to restart...
```

**Fix** (`config.py`):
```python
MAX_ITERATIONS = 1000  # Was 50, now allows ~33 min runs
```

**Prevention**:
1. Distinguish "normal exit" from "crash" in orchestrator
2. Log actual runtime before restart
3. Consider infinite loop with graceful shutdown signal

---

## Lessons Learned

### 1. Machine-Verifiable Acceptance Criteria (MVAC)

❌ **Bad**: "Parse file X"
✅ **Good**: `@file: data/result.csv exists (>1000 rows)`

Critic must be able to verify criteria with a terminal command.

### 2. Impossibility Detection

Some tasks are mathematically impossible (e.g., MAE < 0.30 for Prophet on volatile series). Criteria should include escape hatch:

```json
"@metric: MAE <= 0.50 on h=1 backtest OR documented architectural limitation"
```

### 3. Status Integrity

Valid transitions:
- `TODO → PENDING_REVIEW` (Worker completes)
- `PENDING_REVIEW → DONE` (Critic approves)
- `PENDING_REVIEW → TODO` (Critic rejects)
- `TODO → DONE` (Manual intervention only)

### 4. Data Task Safety

For huge files (>100MB):
- Use `openpyxl` read-only mode
- Extract headers first, don't load all data
- Always verify with `ls -la` and `head -n 5`

---

## File Structure

```
edge_lab/
├── system/
│   ├── worker.py          # Task executor
│   ├── critic.py          # Task verifier
│   ├── orchestrator.py    # Process manager
│   └── core/
│       ├── state.py       # Thread-safe state (PATCHED v1.1)
│       └── agent_wrapper.py
├── tasks/
│   ├── prd.json           # Task queue
│   └── progress.txt       # Execution log
├── agents/                # Generated agents
├── data/                  # Working data
└── AGENTS.md             # Agent directives
```

---

## Universal Tool Principles

For adapting Ralph to other projects:

1. **prd.json Schema**: Any project can use the same task format
2. **Sandboxing**: Worker is restricted to `PROJECT_ROOT`
3. **Lazy Context Loading**: Don't load all files - load on demand
4. **JSON Structured Output**: Critic outputs structured verification results
5. **Idempotent State**: Same input → same output (for debugging)

---

## Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Task Completion Rate | >80% | 78% |
| Rejection Rate | <20% | 15% |
| Avg Criteria per Task | ≥3 | 3.2 |
| Race Condition Errors | 0 | 0 (after patch) |

---

## Next Steps

1. [ ] Add circuit breaker for infinite rejection loops
2. [ ] Implement task dependency graph
3. [ ] Add priority queue (high→medium→low)
4. [ ] Metrics dashboard for monitoring
