# 🧪 Ralph Edge Lab (Sandbox)

This directory is a **Safe Experimentation Zone** for the Autonomous Agent "Ralph Universal".

## 📂 Structure

*   `system/`: The Core Agent Logic (Worker + Critic + Orchestrator).
    *   **Do NOT edit** unless upgrading the agent itself.
*   `tasks/`: Your Control Center.
    *   `prd.json`: Define what you want Ralph to do here.
    *   `progress.txt`: Watch Ralph work in real-time.
*   `research/` & `reports/`: Directories where Ralph will output its work (created by Agent).

## 🚀 How to Launch

Run the orchestrator from this directory:

```bash
cd edge_lab
python3 system/orchestrator.py
```

## 🛠️ CLI Tools

### add_task.py — Task Creator

Быстрое добавление задач в `prd.json` без ручного редактирования JSON:

```bash
# Тестовая задача (auto-MVAC)
python3 add_task.py -t "Test NewModel" --type test -p high

# Новая модель
python3 add_task.py -t "New: LSTM" --type model -p medium

# Data mining
python3 add_task.py -t "Mining: GDP Data" --type mining -p high

# Кастомные MVAC критерии
python3 add_task.py -t "Custom Task" -m "@file: foo.py exists" -m "@functional: runs"

# Интерактивный wizard
python3 add_task.py --interactive

# Справка
python3 add_task.py --list-types
python3 add_task.py --help
```

**Типы задач с auto-MVAC**: `test`, `model`, `script`, `mining`, `docs`, `integration`, `custom`

---

## 🎯 Current Mission (Grand Strategy)

Defined in `tasks/prd.json`:
1.  **Deep Seasonality Analysis**: Advanced correlation metrics for Jan/Feb.
2.  **Hyperparameter Evolution**: Finding better model settings using Optuna.
3.  **Anomaly Detection**: Automated structural break monitoring.

## 🛡️ Safety

This sandbox uses its own `config.py` pointing to `tasks/prd.json` relative to itself. It will NOT affect your main `ito` project or the `opus_forecast` production pipeline unless you explicitly copy files out.
