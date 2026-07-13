# 🎭 Ralph Universal

**Ralph Universal** is a "Bulletproof" autonomous development loop. It uses a **Dual-Agent Architecture** to ensuring high-quality code generation through rigorous verification.

## 🚀 Quick Start

### 1. Configuration
The system is pre-configured to use `opencode`.
If you need to change the model or the CLI command, edit:
`ralph_universal/config.py`

### 2. Define Your Tasks
The agent reads tasks from `docs/prd.json`.
On the first run, a sample file will be created automatically.
You can edit it to add your own tasks:
```json
{
  "project": "My Awesome Feature",
  "user_stories": [
    {
      "id": 1,
      "title": "Implement Login",
      "description": "Create login.py with a function authenticate(user, pass)",
      "acceptance_criteria": [
        "verify_login.py script passes",
        "Function returns True for valid credentials"
      ],
      "status": "TODO"
    }
  ]
}
```

### 3. Run the System
Run the orchestrator. It will launch two parallel processes: **Worker** (Doer) and **Critic** (Observer).

```bash
python3 ralph_universal/orchestrator.py
```

## 🧠 How it Works

1.  **Worker Loop**:
    *   Picks up `TODO` tasks.
    *   Writes a Test ("Red").
    *   Writes Implementation ("Green").
    *   Moves task to `PENDING_REVIEW`.

2.  **Critic Loop**:
    *   Picks up `PENDING_REVIEW` tasks.
    *   Runs the verification steps.
    *   **APPROVES**: Moves task to `DONE`.
    *   **REJECTS**: Moves task back to `TODO` with feedback.

## 📁 Project Structure

*   `core/`: Core logic for state management and agent communication.
*   `docs/`: Contains your `prd.json` (Tasks) and `progress.txt` (Log).
*   `worker.py` / `critic.py`: The agent scripts.
*   `orchestrator.py`: The entry point.
