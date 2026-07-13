# Ralph Universal / Edge Lab Reference

Система автономной разработки с архитектурой Worker-Critic-Refiner.

## Ключевые файлы

| Файл | Назначение |
|------|------------|
| `edge_lab/docs/ARCHITECTURE.md` | Полная архитектура, диаграммы |
| `edge_lab/AGENTS.md` | Конституция агента (правила) |
| `edge_lab/tasks/prd.json` | Очередь задач |
| `edge_lab/system/orchestrator.py` | Главный оркестратор |

## Запуск

```bash
cd edge_lab
python3 system/orchestrator.py  # Worker + Critic параллельно
```

## Архитектура v1.2

```
┌─────────┐     ┌─────────┐     ┌──────────┐
│ Worker  │────▶│ Critic  │────▶│ Refiner  │
│ (does)  │     │(reviews)│     │(improves)│
└─────────┘     └─────────┘     └──────────┘
     │               │               │
     ▼               ▼               ▼
   TODO ──▶ PENDING_REVIEW ──▶ BLOCKED ──▶ Subtasks
                    │
                    ▼
                   DONE
```

## Статусы задач

- `TODO` — ожидает выполнения
- `PENDING_REVIEW` — Worker закончил, Critic проверяет
- `DONE` — Critic одобрил
- `BLOCKED` — 3 неудачные попытки
- `DECOMPOSED` — разбит на подзадачи

## Принципы v1.1

### 1. MVAC — Machine-Verifiable Acceptance Criteria

```
❌ Bad:  "Parse file X"
✅ Good: "@file: data/result.csv exists (>1000 rows)"
```

### 2. Impossibility Escape Hatch

```
"@metric: MAE <= 0.50 OR documented limitation"
```

### 3. Race Condition Protection

Worker не может override rejection.

### 4. Quality-First (v1.3)

**Worker (3 фазы):**
1. RESEARCH — `ls -la`, `head`
2. IMPLEMENT — реальный код
3. VERIFY — `pytest`, `ls -la`

**Critic:** "Assume Worker is lying until verified"

## CLI Tools

```bash
cd edge_lab

# Добавить тестовую задачу
python3 add_task.py -t "Test NewModel" --type test -p high

# Показать BLOCKED
python3 add_task.py --blocked

# Разблокировать
python3 add_task.py --unblock 124
```

## Safety Limits

```python
SAFETY_LIMITS = {
    'max_task_duration_seconds': 1800,  # 30 мин/задача
    'max_retries_per_task': 3,          # → BLOCKED
    'max_file_size_mb': 100,
}
```

## Metric Targets

| Метрика | Цель |
|---------|------|
| Completion Rate | >80% |
| Rejection Rate | <20% |
| Avg Criteria/Task | ≥3 |
