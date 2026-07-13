"""
Telegram Bot Notifier for Ralph Monitoring System.

Sends notifications about task status changes, system health, and alerts.
"""
import os
import ssl
import time
import json
import threading
import urllib.request
import urllib.parse
from typing import Optional, Dict, Any
from datetime import datetime


class TelegramNotifier:
    """Send notifications to Telegram about Ralph events."""

    # Rate limiting: max 1 message per event type per this many seconds
    RATE_LIMIT_SECONDS = {
        'HEARTBEAT': 300,        # 5 min between heartbeats
        'TASK_DONE': 5,          # Allow frequent done notifications
        'TASK_REJECTED': 5,
        'TASK_TIMEOUT': 60,
        'TASK_SKIPPED': 60,
        'TASK_BLOCKED': 60,
        'TASK_REFINED': 30,      # Refiner decomposition
        'SYSTEM_DOWN': 300,      # 5 min between down alerts
        'ERROR': 60,
    }

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize TelegramNotifier.

        Args:
            token: Bot token. If None, reads from RALPH_TG_BOT_TOKEN env var.
            chat_id: Chat ID to send messages to. If None, reads from RALPH_TG_CHAT_ID.
        """
        self.token = token or os.environ.get('RALPH_TG_BOT_TOKEN')
        self.chat_id = chat_id or os.environ.get('RALPH_TG_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)

        # Rate limiting state
        self._last_sent: Dict[str, float] = {}
        self._lock = threading.Lock()

        if not self.enabled:
            print("[TelegramNotifier] WARNING: Bot token or chat_id not configured. Notifications disabled.")
            print("  Set RALPH_TG_BOT_TOKEN and RALPH_TG_CHAT_ID environment variables.")

    def _can_send(self, event_type: str) -> bool:
        """Check if we can send this event type (rate limiting)."""
        with self._lock:
            now = time.time()
            last = self._last_sent.get(event_type, 0)
            limit = self.RATE_LIMIT_SECONDS.get(event_type, 10)

            if now - last < limit:
                return False

            self._last_sent[event_type] = now
            return True

    def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to Telegram.

        Args:
            message: The message text to send.
            parse_mode: "HTML" or "Markdown".

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.enabled:
            print(f"[TelegramNotifier] (disabled) Would send: {message[:100]}...")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
            }).encode()

            # Create SSL context (some environments have SSL issues)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(url, data=data)
            response = urllib.request.urlopen(req, context=ctx, timeout=10)
            result = json.loads(response.read())

            if result.get("ok"):
                return True
            else:
                print(f"[TelegramNotifier] API error: {result}")
                return False

        except Exception as e:
            print(f"[TelegramNotifier] Error sending message: {e}")
            return False

    def handle_event(self, event_type: str, data: Dict[str, Any]):
        """
        Handle an event from StateManager.

        This is the main entry point called by StateManager._emit_event().

        Args:
            event_type: Type of event (TASK_STATUS_CHANGE, etc.)
            data: Event data dict.
        """
        # Map event to notification type
        if event_type == 'TASK_STATUS_CHANGE':
            status = data.get('status', '')

            if status == 'DONE':
                self._notify_task_done(data)
            elif status == 'TODO' and data.get('feedback'):
                # Rejection
                self._notify_task_rejected(data)
            # PENDING_REVIEW - don't notify (intermediate state)

        elif event_type == 'TASK_TIMEOUT':
            self._notify_task_timeout(data)

        elif event_type == 'TASK_SKIPPED':
            self._notify_task_skipped(data)

        elif event_type == 'ERROR':
            self._notify_error(data)

        elif event_type == 'TASK_BLOCKED':
            self._notify_task_blocked(data)

        elif event_type == 'TASK_REFINED':
            self._notify_task_refined(data)

    def _notify_task_done(self, data: Dict[str, Any]):
        """Notify about completed task."""
        if not self._can_send('TASK_DONE'):
            return

        task_id = data.get('task_id', '?')
        title = data.get('title', 'Unknown task')

        message = f"<b>Task #{task_id} DONE</b>\n{title}"
        self.send(message)

    def _notify_task_rejected(self, data: Dict[str, Any]):
        """Notify about rejected task."""
        if not self._can_send('TASK_REJECTED'):
            return

        task_id = data.get('task_id', '?')
        title = data.get('title', 'Unknown task')
        feedback = data.get('feedback', '')[:200]  # Truncate

        message = f"<b>Task #{task_id} REJECTED</b>\n{title}\n\n<i>{feedback}</i>"
        self.send(message)

    def _notify_task_timeout(self, data: Dict[str, Any]):
        """Notify about task timeout."""
        if not self._can_send('TASK_TIMEOUT'):
            return

        task_id = data.get('task_id', '?')
        duration = data.get('duration_seconds', 0)

        message = f"<b>Task #{task_id} TIMEOUT</b>\nExceeded {duration}s limit"
        self.send(message)

    def _notify_task_skipped(self, data: Dict[str, Any]):
        """Notify about skipped task (max retries exceeded)."""
        if not self._can_send('TASK_SKIPPED'):
            return

        task_id = data.get('task_id', '?')
        retries = data.get('retries', 0)

        message = f"<b>Task #{task_id} SKIPPED</b>\nFailed {retries} attempts. Manual intervention needed."
        self.send(message)

    def _notify_error(self, data: Dict[str, Any]):
        """Notify about error."""
        if not self._can_send('ERROR'):
            return

        error = data.get('error', 'Unknown error')[:300]
        component = data.get('component', 'Unknown')

        message = f"<b>ERROR in {component}</b>\n<code>{error}</code>"
        self.send(message)

    def _notify_task_blocked(self, data: Dict[str, Any]):
        """Notify about blocked task."""
        if not self._can_send('TASK_BLOCKED'):
            return

        task_id = data.get('task_id', '?')
        title = data.get('title', 'Unknown task')
        reason = data.get('reason', 'Multiple rejections')

        message = f"<b>Task #{task_id} BLOCKED</b>\n{title}\n\n<i>Причина: {reason}</i>"
        self.send(message)

    def _notify_task_refined(self, data: Dict[str, Any]):
        """Notify about task refinement/decomposition."""
        if not self._can_send('TASK_REFINED'):
            return

        task_id = data.get('task_id', '?')
        title = data.get('title', 'Unknown task')
        subtasks = data.get('subtasks', [])
        count = len(subtasks) if subtasks else data.get('subtask_count', 0)

        message = f"<b>Task #{task_id} REFINED</b>\n{title}\n\nДекомпозирована на {count} подзадач"
        self.send(message)

    def send_heartbeat(self, stats: Dict[str, Any]):
        """
        Send periodic heartbeat message.

        Args:
            stats: Dict with done, pending, uptime_hours, current_task.
        """
        if not self._can_send('HEARTBEAT'):
            return

        done = stats.get('done', 0)
        pending = stats.get('pending', 0)
        uptime = stats.get('uptime_hours', 0)
        current = stats.get('current_task')

        current_str = f", working on #{current}" if current else ""

        message = f"Ralph alive: {done} done, {pending} pending, uptime {uptime:.1f}h{current_str}"
        self.send(message)

    def send_system_down_alert(self, last_heartbeat_age_seconds: float):
        """
        Send critical alert that system is down.

        Args:
            last_heartbeat_age_seconds: How old the last heartbeat is.
        """
        if not self._can_send('SYSTEM_DOWN'):
            return

        minutes = last_heartbeat_age_seconds / 60

        message = f"<b>ALERT: Ralph not responding!</b>\n\nNo heartbeat for {minutes:.0f} minutes.\nCheck the orchestrator process."
        self.send(message)

    def send_startup_message(self):
        """Send notification that Ralph has started."""
        message = "Ralph orchestrator started"
        self.send(message)

    def send_shutdown_message(self):
        """Send notification that Ralph is shutting down."""
        message = "Ralph orchestrator stopped"
        self.send(message)

    def send_daily_summary(self, stats: Dict[str, Any]):
        """
        Send daily summary report.

        Args:
            stats: Dict with done_today, rejected_today, total_done, total_pending, etc.
        """
        done_today = stats.get('done_today', 0)
        rejected_today = stats.get('rejected_today', 0)
        total_done = stats.get('total_done', 0)
        total_pending = stats.get('total_pending', 0)

        message = (
            f"<b>Daily Report</b>\n\n"
            f"Today: {done_today} done, {rejected_today} rejected\n"
            f"Total: {total_done}/{total_done + total_pending} tasks complete"
        )
        self.send(message)


# Singleton instance for easy import
_notifier: Optional[TelegramNotifier] = None

def get_notifier() -> TelegramNotifier:
    """Get the global TelegramNotifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
