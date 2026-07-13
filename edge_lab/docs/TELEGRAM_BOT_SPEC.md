# 📱 Ralph Telegram Bot — Full Specification

> **Цель**: Интерактивный Telegram-бот для управления Ralph с меню, inline-кнопками и командами.

## 📦 Зависимости

```bash
pip install python-telegram-bot==20.7  # Async, современный API
```

## 🗂 Файлы

| Файл | Назначение |
|------|------------|
| `system/monitoring/telegram_controller.py` | Основной бот с командами |
| `system/monitoring/telegram_bot.py` | Существующий notifier (оставить) |
| `.env` | `RALPH_TG_BOT_TOKEN`, `RALPH_TG_CHAT_ID` |

---

## 🎛 UI: Меню команд (BotFather)

Настроить через @BotFather → /setcommands:

```
status - 📊 Статус Ralph (задачи, uptime)
tasks - 📋 Список активных задач
blocked - 🚫 Заблокированные задачи
done - ✅ Последние выполненные
restart - 🔄 Перезапустить Ralph
pause - ⏸ Приостановить Worker
resume - ▶️ Возобновить Worker
logs - 📄 Последние 20 строк progress.txt
help - ❓ Справка по командам
```

---

## 🎯 Команды и логика

### `/status`
```
📊 Ralph Status

🟢 Running (uptime: 4.2h)
├─ Worker: Active (Task #124)
├─ Critic: Idle
└─ Refiner: Watching

📋 Tasks: 47 done | 3 pending | 8 todo | 2 blocked
⏱ Last activity: 2 min ago
```

**Inline кнопки:**
```
[🔄 Refresh] [📋 Tasks] [🚫 Blocked]
```

### `/tasks`
```
📋 Active Tasks

🔵 TODO (8):
├─ #403: Weekly Nowcast Integration
├─ #404: Optimal Lag Discovery
└─ ... (показать первые 5)

🟡 PENDING_REVIEW (3):
├─ #125: Ideal Codes Registry
└─ #126: GDP Correlation

[Show All TODO] [Show Pending]
```

### `/blocked`
```
🚫 Blocked Tasks (2)

#124: Mining: HH.ru & DomClick
├─ Priority: high
├─ Attempts: 3/3
├─ Feedback: Only extracted HH Index...
└─ [Unblock] [Decompose] [Skip]

#130: Parse Complex Excel
├─ Priority: medium
├─ Attempts: 3/3
└─ [Unblock] [Decompose] [Skip]
```

**Callback handlers:**
- `unblock_124` → вызов `add_task.py --unblock 124`
- `decompose_124` → триггер Refiner для task 124
- `skip_124` → пометить DONE+SKIPPED

### `/done [N]`
```
✅ Last 5 Completed

#125: Ideal Codes Registry (2h ago)
#123: KBR Budget Extraction (5h ago)
#122: Brent Lag Analysis (8h ago)
...
```

### `/restart`
```
⚠️ Перезапуск Ralph

Это остановит все процессы и запустит заново.

[✅ Confirm Restart] [❌ Cancel]
```

**Логика:**
```python
subprocess.run(["pkill", "-f", "orchestrator.py"])
time.sleep(2)
subprocess.Popen(["python3", "system/orchestrator.py"], 
                 cwd="/home/valalav/_projects/sirena-kbr/edge_lab",
                 start_new_session=True)
```

### `/pause`
```
⏸ Worker приостановлен

Новые задачи не берутся. Critic и Refiner продолжают.

[▶️ Resume]
```

**Логика:** Создать файл `data/.worker_paused` — Worker проверяет его в цикле.

### `/resume`
Удалить `data/.worker_paused`.

### `/logs`
```
📄 Last 20 lines of progress.txt

[2026-01-23 07:15:01] [WORKER] Starting Task 124 (attempt 1/3)
[2026-01-23 07:15:32] [WORKER] Finished execution cycle
...
```

### `/help`
```
❓ Ralph Bot Commands

📊 /status — Статус системы
📋 /tasks — Список активных задач
🚫 /blocked — Заблокированные задачи
✅ /done — Последние выполненные
🔄 /restart — Перезапуск Ralph
⏸ /pause — Приостановить Worker
▶️ /resume — Возобновить Worker
📄 /logs — Последние логи

💡 Используй inline-кнопки для быстрых действий!
```

---

## 🔔 Автоматические уведомления

Интегрировать с существующим `TelegramNotifier`:

| Событие | Формат |
|---------|--------|
| Task DONE | ✅ Task #123 done: "Title" |
| Task BLOCKED | 🚫 Task #124 blocked after 3 attempts [Unblock] |
| Task DECOMPOSED | 🔀 Task #124 → Subtasks #411, #412, #413 |
| System down | 🔴 Ralph not responding (30 min) [Restart] |
| Daily summary | 📊 Daily: 5 done, 2 rejected, 12 remaining |

---

## 📐 Структура кода

```python
# telegram_controller.py

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler
import subprocess
import json
from pathlib import Path

# Paths
EDGE_LAB = Path("/home/valalav/_projects/sirena-kbr/edge_lab")
PRD_FILE = EDGE_LAB / "tasks" / "prd.json"
PROGRESS_FILE = EDGE_LAB / "tasks" / "progress.txt"
PAUSE_FLAG = EDGE_LAB / "data" / ".worker_paused"

class RalphController:
    def __init__(self, token: str, allowed_chat_ids: list[int]):
        self.token = token
        self.allowed_chat_ids = allowed_chat_ids
        
    def _check_auth(self, update: Update) -> bool:
        return update.effective_chat.id in self.allowed_chat_ids
    
    async def status(self, update: Update, context):
        if not self._check_auth(update): return
        # ... implementation
        
    async def blocked(self, update: Update, context):
        if not self._check_auth(update): return
        prd = json.load(open(PRD_FILE))
        blocked = [t for t in prd['user_stories'] if t['status'] == 'BLOCKED']
        # ... format and send with inline buttons
        
    async def handle_callback(self, update: Update, context):
        query = update.callback_query
        data = query.data
        
        if data.startswith("unblock_"):
            task_id = int(data.split("_")[1])
            # Call add_task.py --unblock
            subprocess.run(["python3", "add_task.py", "--unblock", str(task_id)],
                          cwd=EDGE_LAB)
            await query.answer(f"Task {task_id} unblocked!")
            
        elif data.startswith("restart"):
            # Restart Ralph
            ...
```

---

## 🚀 Запуск

```python
# В конце telegram_controller.py

def main():
    token = os.environ.get("RALPH_TG_BOT_TOKEN")
    chat_id = int(os.environ.get("RALPH_TG_CHAT_ID"))
    
    controller = RalphController(token, [chat_id])
    
    app = Application.builder().token(token).build()
    
    app.add_handler(CommandHandler("status", controller.status))
    app.add_handler(CommandHandler("tasks", controller.tasks))
    app.add_handler(CommandHandler("blocked", controller.blocked))
    app.add_handler(CommandHandler("done", controller.done))
    app.add_handler(CommandHandler("restart", controller.restart))
    app.add_handler(CommandHandler("pause", controller.pause))
    app.add_handler(CommandHandler("resume", controller.resume))
    app.add_handler(CommandHandler("logs", controller.logs))
    app.add_handler(CommandHandler("help", controller.help))
    
    app.add_handler(CallbackQueryHandler(controller.handle_callback))
    
    print("[TelegramController] Bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
```

---

## ✅ Checklist для Claude

- [ ] Создать `telegram_controller.py` с полной логикой
- [ ] Реализовать все 9 команд
- [ ] Добавить InlineKeyboardMarkup для кнопок
- [ ] Реализовать callback handlers (unblock, decompose, restart)
- [ ] Интегрировать с существующим notifier
- [ ] Добавить авторизацию (только разрешённые chat_id)
- [ ] Добавить graceful error handling
- [ ] Тест: `/status`, `/blocked`, `/restart`

---

## 📝 Env Variables

```bash
# .env
RALPH_TG_BOT_TOKEN=7123456789:AAHxxxxxxxxxxxxxxxxxxxxxxxx
RALPH_TG_CHAT_ID=123456789
```
