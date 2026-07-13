# 🛡️ QUALITY MANIFESTO: Anti-Gaming Principles for Ralph

> **Core Truth**: An LLM saying "done" means nothing. Only verified output is real.

---

## The Problem

LLM agents optimize for:
- ❌ "Looks like I did it" → Reward signal
- ❌ "Output tokens generated" → Completion
- ❌ "No errors shown" → Success

This is **fake work**. It games metrics without delivering value.

---

## Anti-Gaming Principles

### 1. 🔬 PROOF > PROMISE

Every task must produce **verifiable artifacts**:

| Bad MVAC | Good MVAC |
|----------|-----------|
| "Implemented feature X" | "@file: feature_x.py exists (>100 lines)" |
| "Fixed the bug" | "@functional: pytest test_bug.py exits code 0" |
| "Analyzed data" | "@file: data/analysis.csv exists with >50 rows, columns: [A,B,C]" |

**Rule**: If Critic cannot verify it with `ls`, `head`, `pytest`, or `grep` — it's not done.

### 2. 🧪 SHOW, DON'T TELL

Worker must include **evidence** in completion:

```
COMPLETED_TASK

Files created:
- sirena/models/new_model.py (127 lines)

Verification commands run:
$ pytest tests/test_new_model.py -v
===== 5 passed in 2.3s =====

$ python -c "from sirena.models.new_model import *"
(no errors)
```

**Without evidence = Not completed.**

### 3. 📋 PRE-TASK RESEARCH (Mandatory)

Before starting work, Worker must:

1. **List relevant files**: `ls -la sirena/models/`
2. **Check existing code**: `head -50 file.py`
3. **Identify dependencies**: What imports are needed?
4. **State the approach**: "I will modify X by adding Y"

This prevents blind coding that breaks things.

### 4. 🔒 CRITIC AS ADVERSARY

Critic must:

- **Assume Worker is lying** until proven otherwise
- **Run every verification command** (not trust Worker's claims)
- **Reject ambiguous completions** — demand specifics
- **Check for regressions** — does existing code still work?

```python
# Critic verification checklist:
1. [ ] File exists at claimed path?
2. [ ] File has claimed line count?
3. [ ] pytest/command actually passes when I run it?
4. [ ] No new syntax errors introduced?
5. [ ] Imports resolve correctly?
```

### 5. ⚠️ IMPOSSIBILITY ESCAPE HATCH

Some tasks are genuinely impossible. Allow honest failure:

```
MVAC: "@metric: MAE < 0.30 OR documented why impossible with evidence"
```

**This prevents**: Worker claiming fake success to avoid admitting failure.

---

## Task Definition Template (Mandatory Fields)

```json
{
  "id": 500,
  "title": "Clear, Specific Title",
  "description": "What to do, WHY it matters, expected outcome",
  
  "prerequisites": [
    "File X exists",
    "Module Y is importable"
  ],
  
  "research_required": [
    "List files in sirena/models/",
    "Check current implementation of Z"
  ],
  
  "acceptance_criteria": [
    "@file: path/to/output.py exists (>N lines)",
    "@functional: pytest test_X.py -v passes",
    "@metric: MAE <= X.XX OR documented limitation"
  ],
  
  "known_pitfalls": [
    "Excel file has complex multi-sheet structure",
    "Data requires specific encoding (cp1251)"
  ],
  
  "verification_commands": [
    "python3 -m py_compile output.py",
    "pytest tests/test_output.py -v",
    "head -20 data/result.csv"
  ]
}
```

---

## Enforcement in Code

### Worker Changes (worker.py)

```python
# Before starting task:
state.append_progress(f"Research phase: ls, head, identify approach", "WORKER")

# Completion must include evidence:
if "COMPLETED_TASK" in output:
    if "Files created:" not in output:
        return "INCOMPLETE: Missing file evidence"
    if "Verification commands run:" not in output:
        return "INCOMPLETE: Missing verification evidence"
```

### Critic Changes (critic.py)

```python
# Actually run verification, don't trust claims:
for cmd in task.get("verification_commands", []):
    result = subprocess.run(cmd, shell=True, capture_output=True)
    if result.returncode != 0:
        return f"REJECT: Verification failed: {cmd}\n{result.stderr}"
```

---

## Summary: Quality Hierarchy

```
1. Real output artifacts (files, data)
2. Verified by independent execution
3. Evidence documented in completion
4. Pre-task research prevents blind work
5. Honest failure > Fake success
```

**The goal is not "tasks completed" — it's "value delivered".**
