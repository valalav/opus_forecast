---
name: verified-test
description: Use this skill when you need to run tests and provide UNDENIABLE PROOF of success to the Critic. It runs pytest and generates a signed receipt.
---

# Verified Test Skill

**Goal**: Run tests and generate a trusted execution report that prevents "Fake Work".

## Instructions
1.  **Select Tests**: Identify which tests to run (e.g., `tests/test_models.py`).
2.  **Execute**: Run `scripts/run_verified_test.py --target <test_file>`.
3.  **Output**: The script will output a JSON block. You MUST copy this JSON block into your final report.

## Usage
```bash
python edge_lab/skills/verified_test/scripts/run_verified_test.py --target tests/test_api.py
```

## Verification
The Critic looks for the `{"verified": true, ...}` JSON block in the Worker's output. If it's missing or shows failed tests, the task is Rejected.
