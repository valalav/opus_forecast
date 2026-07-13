#!/usr/bin/env python3
"""
Telegram Bot Server for Ralph Control.

Interactive bot with commands and inline keyboard for controlling Ralph.

Usage:
    python3 edge_lab/system/monitoring/telegram_bot_server.py
"""
import os
import sys
import json
import ssl
import time
import signal
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


class TelegramBotServer:
    """Interactive Telegram bot for Ralph control."""

    def __init__(self):
        self.token = os.environ.get('RALPH_TG_BOT_TOKEN')
        self.chat_id = os.environ.get('RALPH_TG_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)

        # Paths
        self.base_dir = Path(__file__).parent.parent.parent  # edge_lab/
        self.system_dir = self.base_dir / "system"
        self.heartbeat_file = self.base_dir / "data" / "heartbeat.json"
        self.prd_file = self.base_dir / "tasks" / "prd.json"
        self.progress_file = self.base_dir / "tasks" / "progress.txt"

        # State
        self.last_update_id = 0
        self.running = True

        # SSL context
        self.ssl_ctx = ssl.create_default_context()
        self.ssl_ctx.check_hostname = False
        self.ssl_ctx.verify_mode = ssl.CERT_NONE

        if not self.enabled:
            print("[Bot] ERROR: RALPH_TG_BOT_TOKEN or RALPH_TG_CHAT_ID not set")

    def _api_call(self, method: str, data: Dict = None) -> Optional[Dict]:
        """Make Telegram API call."""
        try:
            url = f"https://api.telegram.org/bot{self.token}/{method}"

            if data:
                data_encoded = json.dumps(data).encode('utf-8')
                req = urllib.request.Request(
                    url,
                    data=data_encoded,
                    headers={'Content-Type': 'application/json'}
                )
            else:
                req = urllib.request.Request(url)

            response = urllib.request.urlopen(req, context=self.ssl_ctx, timeout=30)
            return json.loads(response.read())

        except Exception as e:
            print(f"[Bot] API error: {e}")
            return None

    def send_message(self, text: str, reply_markup: Dict = None, parse_mode: str = "HTML") -> bool:
        """Send message with optional keyboard."""
        data = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_markup:
            data["reply_markup"] = reply_markup

        result = self._api_call("sendMessage", data)
        return result and result.get("ok", False)

    def answer_callback(self, callback_id: str, text: str = None):
        """Answer callback query."""
        data = {"callback_query_id": callback_id}
        if text:
            data["text"] = text
        self._api_call("answerCallbackQuery", data)

    def edit_message(self, message_id: int, text: str, reply_markup: Dict = None):
        """Edit existing message."""
        data = {
            "chat_id": self.chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "HTML",
        }
        if reply_markup:
            data["reply_markup"] = reply_markup
        self._api_call("editMessageText", data)

    def get_main_menu(self) -> Dict:
        """Get main menu keyboard."""
        return {
            "inline_keyboard": [
                [
                    {"text": "📊 Статус", "callback_data": "status"},
                    {"text": "📋 Задачи", "callback_data": "tasks"},
                ],
                [
                    {"text": "▶️ Запустить", "callback_data": "start"},
                    {"text": "⏹ Остановить", "callback_data": "stop"},
                ],
                [
                    {"text": "🔄 Перезапустить", "callback_data": "restart"},
                    {"text": "📜 Логи", "callback_data": "logs"},
                ],
                [
                    {"text": "💓 Heartbeat", "callback_data": "heartbeat"},
                    {"text": "❓ Помощь", "callback_data": "help"},
                ],
            ]
        }

    def get_tasks_menu(self) -> Dict:
        """Get tasks submenu."""
        return {
            "inline_keyboard": [
                [
                    {"text": "📋 TODO", "callback_data": "tasks_todo"},
                    {"text": "✅ Done", "callback_data": "tasks_done"},
                ],
                [
                    {"text": "🚫 Blocked", "callback_data": "tasks_blocked"},
                    {"text": "❌ Rejected", "callback_data": "tasks_rejected"},
                ],
                [
                    {"text": "📈 Статистика", "callback_data": "tasks_stats"},
                    {"text": "🔓 Unblock", "callback_data": "tasks_unblock_menu"},
                ],
                [
                    {"text": "◀️ Назад", "callback_data": "menu"},
                ],
            ]
        }

    # ==================== STATUS ====================

    def check_ralph_status(self) -> Dict:
        """Check Ralph orchestrator status."""
        status = {
            "running": False,
            "heartbeat_age": None,
            "tasks_done": 0,
            "tasks_todo": 0,
            "uptime": None,
        }

        # Check heartbeat file
        if self.heartbeat_file.exists():
            try:
                with open(self.heartbeat_file) as f:
                    hb = json.load(f)

                timestamp = datetime.fromisoformat(hb.get("timestamp", ""))
                age = (datetime.now() - timestamp).total_seconds()
                status["heartbeat_age"] = age
                status["running"] = age < 1800  # 30 min threshold
                status["tasks_done"] = hb.get("tasks_done", 0)
                status["tasks_todo"] = hb.get("tasks_todo", 0)
                status["uptime"] = hb.get("uptime_hours", 0)
                status["current_task"] = hb.get("current_task")

            except Exception as e:
                print(f"[Bot] Error reading heartbeat: {e}")

        # Also check process
        try:
            result = subprocess.run(
                ["pgrep", "-f", "orchestrator.py"],
                capture_output=True, timeout=5
            )
            status["process_running"] = result.returncode == 0
        except:
            status["process_running"] = False

        return status

    def cmd_status(self) -> str:
        """Get status message."""
        status = self.check_ralph_status()

        if status["running"]:
            icon = "🟢"
            state = "Работает"
        elif status["process_running"]:
            icon = "🟡"
            state = "Процесс есть, heartbeat устарел"
        else:
            icon = "🔴"
            state = "Остановлен"

        msg = f"<b>{icon} Ralph: {state}</b>\n\n"

        if status["heartbeat_age"] is not None:
            age_min = status["heartbeat_age"] / 60
            msg += f"Heartbeat: {age_min:.1f} мин назад\n"

        if status["uptime"]:
            msg += f"Uptime: {status['uptime']:.1f} часов\n"

        msg += f"\n<b>Задачи:</b>\n"
        msg += f"✅ Выполнено: {status['tasks_done']}\n"
        msg += f"📋 Осталось: {status['tasks_todo']}\n"

        if status.get("current_task"):
            msg += f"\n🔧 Текущая: #{status['current_task']}"

        return msg

    # ==================== TASKS ====================

    def read_prd(self) -> Dict:
        """Read PRD file."""
        if not self.prd_file.exists():
            return {"user_stories": []}
        try:
            with open(self.prd_file) as f:
                return json.load(f)
        except:
            return {"user_stories": []}

    def cmd_tasks_todo(self) -> str:
        """List TODO tasks."""
        prd = self.read_prd()
        tasks = [t for t in prd.get("user_stories", []) if t.get("status") == "TODO"]

        if not tasks:
            return "📋 Нет задач в TODO"

        msg = f"<b>📋 TODO задачи ({len(tasks)}):</b>\n\n"
        for t in tasks[:10]:  # Max 10
            priority = "🔴" if t.get("priority") == "high" else "🟡" if t.get("priority") == "medium" else "⚪"
            msg += f"{priority} #{t['id']}: {t['title'][:40]}\n"

        if len(tasks) > 10:
            msg += f"\n... и ещё {len(tasks) - 10}"

        return msg

    def cmd_tasks_done(self) -> str:
        """List DONE tasks (last 10)."""
        prd = self.read_prd()
        tasks = [t for t in prd.get("user_stories", []) if t.get("status") == "DONE"]

        msg = f"<b>✅ Выполнено ({len(tasks)}):</b>\n\n"
        for t in tasks[-10:]:  # Last 10
            msg += f"#{t['id']}: {t['title'][:40]}\n"

        return msg

    def cmd_tasks_rejected(self) -> str:
        """List rejected tasks."""
        prd = self.read_prd()
        tasks = [t for t in prd.get("user_stories", [])
                 if t.get("status") == "TODO" and t.get("feedback") and "Reject" in t.get("feedback", "")]

        if not tasks:
            return "❌ Нет отклонённых задач"

        msg = f"<b>❌ Отклонённые ({len(tasks)}):</b>\n\n"
        for t in tasks[:5]:
            feedback = t.get("feedback", "")[:100]
            msg += f"#{t['id']}: {t['title'][:30]}\n<i>{feedback}</i>\n\n"

        return msg

    def get_blocked_tasks(self) -> list:
        """Get blocked tasks (status=BLOCKED or rejected 2+ times)."""
        prd = self.read_prd()
        blocked = []
        for t in prd.get("user_stories", []):
            # Explicit BLOCKED status
            if t.get("status") == "BLOCKED":
                blocked.append(t)
            # Or rejected multiple times (check rejection_count or feedback pattern)
            elif t.get("status") == "TODO":
                rejection_count = t.get("rejection_count", 0)
                feedback = t.get("feedback", "")
                if rejection_count >= 2 or "BLOCKED" in feedback.upper():
                    blocked.append(t)
        return blocked

    def cmd_tasks_blocked(self) -> str:
        """List blocked tasks."""
        blocked = self.get_blocked_tasks()

        if not blocked:
            return "🚫 Нет заблокированных задач"

        msg = f"<b>🚫 Заблокированные ({len(blocked)}):</b>\n\n"
        for t in blocked[:10]:
            feedback = t.get("feedback", "")[:80]
            rej_count = t.get("rejection_count", 0)
            msg += f"#{t['id']}: {t['title'][:35]}\n"
            if rej_count:
                msg += f"   Отклонений: {rej_count}\n"
            if feedback:
                msg += f"   <i>{feedback}</i>\n"
            msg += "\n"

        return msg

    def cmd_tasks_unblock_menu(self) -> tuple:
        """Show blocked tasks with unblock buttons."""
        blocked = self.get_blocked_tasks()

        if not blocked:
            return "🚫 Нет заблокированных задач", self.get_tasks_menu()

        msg = "<b>🔓 Выберите задачу для разблокировки:</b>\n\n"
        for t in blocked[:5]:
            msg += f"#{t['id']}: {t['title'][:40]}\n"

        # Create keyboard with unblock buttons
        keyboard = []
        for t in blocked[:5]:
            keyboard.append([{"text": f"🔓 #{t['id']}", "callback_data": f"unblock_{t['id']}"}])
        keyboard.append([{"text": "◀️ Назад", "callback_data": "tasks"}])

        return msg, {"inline_keyboard": keyboard}

    def cmd_unblock_task(self, task_id: int) -> str:
        """Unblock a task."""
        prd = self.read_prd()
        updated = False

        for t in prd.get("user_stories", []):
            if t.get("id") == task_id:
                old_status = t.get("status")
                t["status"] = "TODO"
                t["feedback"] = f"Unblocked manually (was: {old_status})"
                t["rejection_count"] = 0  # Reset rejection counter
                updated = True
                break

        if updated:
            self.write_prd(prd)
            return f"✅ Задача #{task_id} разблокирована"
        else:
            return f"❌ Задача #{task_id} не найдена"

    def write_prd(self, prd: Dict):
        """Write PRD file."""
        try:
            with open(self.prd_file, "w") as f:
                json.dump(prd, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Bot] Error writing PRD: {e}")

    def cmd_tasks_stats(self) -> str:
        """Task statistics."""
        prd = self.read_prd()
        stories = prd.get("user_stories", [])

        total = len(stories)
        done = sum(1 for t in stories if t.get("status") == "DONE")
        todo = sum(1 for t in stories if t.get("status") == "TODO")
        pending = sum(1 for t in stories if t.get("status") == "PENDING_REVIEW")
        blocked = len(self.get_blocked_tasks())
        rejected = sum(1 for t in stories
                       if t.get("status") == "TODO" and t.get("feedback") and "Reject" in t.get("feedback", ""))

        pct = (done / total * 100) if total > 0 else 0

        msg = f"<b>📈 Статистика задач</b>\n\n"
        msg += f"Всего: {total}\n"
        msg += f"✅ Выполнено: {done} ({pct:.1f}%)\n"
        msg += f"📋 TODO: {todo}\n"
        msg += f"⏳ На проверке: {pending}\n"
        if blocked:
            msg += f"🚫 Заблокировано: {blocked}\n"
        if rejected:
            msg += f"❌ Отклонено: {rejected}\n"

        # Progress bar
        bar_len = 20
        filled = int(bar_len * done / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        msg += f"\n[{bar}] {pct:.0f}%"

        return msg

    # ==================== CONTROL ====================

    def cmd_start_ralph(self) -> str:
        """Start Ralph orchestrator."""
        status = self.check_ralph_status()

        if status["process_running"]:
            return "⚠️ Ralph уже запущен"

        try:
            # Start in background using supervisor (ensures auto-restart)
            cmd = f"cd {self.base_dir} && nohup ./scripts/start_ralph_supervisor.sh > /tmp/ralph_supervisor.log 2>&1 &"
            subprocess.Popen(cmd, shell=True)
            time.sleep(2)

            # Check if started
            status = self.check_ralph_status()
            if status["process_running"]:
                return "✅ Ralph запущен"
            else:
                return "❌ Не удалось запустить. Проверьте /tmp/ralph.log"

        except Exception as e:
            return f"❌ Ошибка: {e}"

    def cmd_stop_ralph(self) -> str:
        """Stop Ralph orchestrator."""
        try:
            result = subprocess.run(
                ["pkill", "-f", "orchestrator.py"],
                capture_output=True, timeout=10
            )

            time.sleep(1)
            status = self.check_ralph_status()

            if not status["process_running"]:
                return "⏹ Ralph остановлен"
            else:
                return "⚠️ Процесс ещё работает. Попробуйте pkill -9"

        except Exception as e:
            return f"❌ Ошибка: {e}"

    def cmd_restart_ralph(self) -> str:
        """Restart Ralph."""
        self.cmd_stop_ralph()
        time.sleep(2)
        return self.cmd_start_ralph()

    # ==================== LOGS ====================

    def cmd_logs(self) -> str:
        """Get last log entries."""
        if not self.progress_file.exists():
            return "📜 Лог пуст"

        try:
            with open(self.progress_file) as f:
                lines = f.readlines()

            last_lines = lines[-15:]  # Last 15
            msg = "<b>📜 Последние события:</b>\n\n"
            msg += "<code>"
            for line in last_lines:
                # Truncate long lines
                line = line.strip()[:80]
                msg += f"{line}\n"
            msg += "</code>"

            return msg

        except Exception as e:
            return f"❌ Ошибка чтения лога: {e}"

    def cmd_heartbeat(self) -> str:
        """Show heartbeat details."""
        if not self.heartbeat_file.exists():
            return "💓 Heartbeat файл не найден"

        try:
            with open(self.heartbeat_file) as f:
                hb = json.load(f)

            msg = "<b>💓 Heartbeat</b>\n\n"
            msg += f"<code>{json.dumps(hb, indent=2, ensure_ascii=False)}</code>"
            return msg

        except Exception as e:
            return f"❌ Ошибка: {e}"

    def cmd_help(self) -> str:
        """Help message."""
        return """<b>❓ Ralph Bot - Команды</b>

<b>Кнопки меню:</b>
📊 Статус — состояние Ralph
📋 Задачи — список задач
▶️ Запустить — запустить Ralph
⏹ Остановить — остановить Ralph
🔄 Перезапустить — рестарт
📜 Логи — последние события
💓 Heartbeat — детали heartbeat

<b>Текстовые команды:</b>
/menu — показать меню
/status — статус системы
/start_ralph — запустить
/stop_ralph — остановить
/tasks — список задач
/logs — последние логи
"""

    # ==================== MESSAGE HANDLING ====================

    def handle_message(self, message: Dict):
        """Handle incoming message."""
        text = message.get("text", "").strip()
        chat_id = message.get("chat", {}).get("id")

        # Security: only respond to authorized chat
        if str(chat_id) != str(self.chat_id):
            print(f"[Bot] Unauthorized chat: {chat_id}")
            return

        # Commands
        if text.startswith("/"):
            cmd = text.split()[0].lower()

            if cmd in ["/start", "/menu"]:
                self.send_message("🤖 <b>Ralph Control Panel</b>", self.get_main_menu())

            elif cmd == "/status":
                self.send_message(self.cmd_status(), self.get_main_menu())

            elif cmd == "/start_ralph":
                self.send_message(self.cmd_start_ralph(), self.get_main_menu())

            elif cmd == "/stop_ralph":
                self.send_message(self.cmd_stop_ralph(), self.get_main_menu())

            elif cmd == "/tasks":
                self.send_message(self.cmd_tasks_stats(), self.get_tasks_menu())

            elif cmd == "/logs":
                self.send_message(self.cmd_logs(), self.get_main_menu())

            elif cmd == "/help":
                self.send_message(self.cmd_help(), self.get_main_menu())

            else:
                self.send_message("❓ Неизвестная команда. /help для справки", self.get_main_menu())

        else:
            # Any other text - show menu
            self.send_message("🤖 <b>Ralph Control Panel</b>", self.get_main_menu())

    def handle_callback(self, callback: Dict):
        """Handle callback button press."""
        data = callback.get("data", "")
        callback_id = callback.get("id")
        message = callback.get("message", {})
        message_id = message.get("message_id")

        # Answer callback immediately
        self.answer_callback(callback_id)

        # Handle actions
        if data == "menu":
            self.edit_message(message_id, "🤖 <b>Ralph Control Panel</b>", self.get_main_menu())

        elif data == "status":
            self.edit_message(message_id, self.cmd_status(), self.get_main_menu())

        elif data == "tasks":
            self.edit_message(message_id, self.cmd_tasks_stats(), self.get_tasks_menu())

        elif data == "tasks_todo":
            self.edit_message(message_id, self.cmd_tasks_todo(), self.get_tasks_menu())

        elif data == "tasks_done":
            self.edit_message(message_id, self.cmd_tasks_done(), self.get_tasks_menu())

        elif data == "tasks_rejected":
            self.edit_message(message_id, self.cmd_tasks_rejected(), self.get_tasks_menu())

        elif data == "tasks_blocked":
            self.edit_message(message_id, self.cmd_tasks_blocked(), self.get_tasks_menu())

        elif data == "tasks_unblock_menu":
            msg, keyboard = self.cmd_tasks_unblock_menu()
            self.edit_message(message_id, msg, keyboard)

        elif data.startswith("unblock_"):
            task_id = int(data.split("_")[1])
            result = self.cmd_unblock_task(task_id)
            self.edit_message(message_id, result, self.get_tasks_menu())

        elif data == "tasks_stats":
            self.edit_message(message_id, self.cmd_tasks_stats(), self.get_tasks_menu())

        elif data == "start":
            self.edit_message(message_id, "⏳ Запускаю...", None)
            result = self.cmd_start_ralph()
            self.edit_message(message_id, result, self.get_main_menu())

        elif data == "stop":
            self.edit_message(message_id, "⏳ Останавливаю...", None)
            result = self.cmd_stop_ralph()
            self.edit_message(message_id, result, self.get_main_menu())

        elif data == "restart":
            self.edit_message(message_id, "⏳ Перезапускаю...", None)
            result = self.cmd_restart_ralph()
            self.edit_message(message_id, result, self.get_main_menu())

        elif data == "logs":
            self.edit_message(message_id, self.cmd_logs(), self.get_main_menu())

        elif data == "heartbeat":
            self.edit_message(message_id, self.cmd_heartbeat(), self.get_main_menu())

        elif data == "help":
            self.edit_message(message_id, self.cmd_help(), self.get_main_menu())

    def poll_updates(self):
        """Poll for updates using long polling."""
        result = self._api_call("getUpdates", {
            "offset": self.last_update_id + 1,
            "timeout": 30,
            "allowed_updates": ["message", "callback_query"]
        })

        if not result or not result.get("ok"):
            return []

        updates = result.get("result", [])

        for update in updates:
            self.last_update_id = max(self.last_update_id, update.get("update_id", 0))

        return updates

    def run(self):
        """Main bot loop."""
        print(f"[Bot] Starting Ralph Control Bot...")
        print(f"[Bot] Chat ID: {self.chat_id}")
        print(f"[Bot] Base dir: {self.base_dir}")

        # Send startup message
        self.send_message("🤖 <b>Ralph Bot запущен</b>\n\nИспользуйте кнопки для управления.", self.get_main_menu())

        # Set up signal handler
        def signal_handler(sig, frame):
            print("\n[Bot] Shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Main loop
        while self.running:
            try:
                updates = self.poll_updates()

                for update in updates:
                    if "message" in update:
                        self.handle_message(update["message"])
                    elif "callback_query" in update:
                        self.handle_callback(update["callback_query"])

            except Exception as e:
                print(f"[Bot] Error in main loop: {e}")
                time.sleep(5)

        print("[Bot] Stopped")


def main():
    """Entry point."""
    # Load .env
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

    bot = TelegramBotServer()

    if not bot.enabled:
        print("ERROR: Set RALPH_TG_BOT_TOKEN and RALPH_TG_CHAT_ID")
        sys.exit(1)

    bot.run()


if __name__ == "__main__":
    main()
