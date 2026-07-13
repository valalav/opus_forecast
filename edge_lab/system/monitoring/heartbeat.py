"""
Heartbeat Service for Ralph Monitoring System.

Periodically writes heartbeat.json for watchdog to monitor.
Also sends heartbeat notifications to Telegram.
"""
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.state import StateManager
    from monitoring.telegram_bot import TelegramNotifier


class HeartbeatService:
    """
    Background service that:
    1. Writes heartbeat.json every interval for watchdog
    2. Sends heartbeat messages to Telegram
    """

    DEFAULT_INTERVAL_SECONDS = 900  # 15 minutes

    def __init__(
        self,
        state_manager: 'StateManager',
        notifier: Optional['TelegramNotifier'] = None,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        heartbeat_file: Optional[Path] = None
    ):
        """
        Initialize HeartbeatService.

        Args:
            state_manager: StateManager instance to read task stats.
            notifier: TelegramNotifier for sending heartbeat messages.
            interval_seconds: How often to write heartbeat (default 15 min).
            heartbeat_file: Path to heartbeat.json. Default: data/heartbeat.json
        """
        self.state_manager = state_manager
        self.notifier = notifier
        self.interval = interval_seconds

        # Determine heartbeat file path
        if heartbeat_file:
            self.heartbeat_file = heartbeat_file
        else:
            # Default to edge_lab/data/heartbeat.json
            base = Path(__file__).parent.parent.parent  # edge_lab/
            self.heartbeat_file = base / "data" / "heartbeat.json"

        # Ensure data directory exists
        self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)

        # Service state
        self._start_time = time.time()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_task: Optional[int] = None

    def set_current_task(self, task_id: Optional[int]):
        """Set the currently running task ID (called by worker)."""
        self._current_task = task_id

    def start(self):
        """Start the heartbeat service in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            print("[HeartbeatService] Already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="HeartbeatService")
        self._thread.start()
        print(f"[HeartbeatService] Started (interval: {self.interval}s)")

        # Write initial heartbeat immediately
        self._write_heartbeat()

    def stop(self):
        """Stop the heartbeat service."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[HeartbeatService] Stopped")

    def _run(self):
        """Main loop: write heartbeat at intervals."""
        while not self._stop_event.is_set():
            try:
                self._write_heartbeat()

                # Also send to Telegram
                if self.notifier:
                    stats = self._collect_stats()
                    self.notifier.send_heartbeat(stats)

            except Exception as e:
                print(f"[HeartbeatService] Error: {e}")

            # Wait for next interval (check stop every second)
            for _ in range(self.interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _collect_stats(self) -> dict:
        """Collect current statistics from StateManager."""
        try:
            prd = self.state_manager.read_prd()
            stories = prd.get("user_stories", [])

            done = sum(1 for s in stories if s.get("status") == "DONE")
            pending = sum(1 for s in stories if s.get("status") == "PENDING_REVIEW")
            todo = sum(1 for s in stories if s.get("status") == "TODO")

            uptime_seconds = time.time() - self._start_time
            uptime_hours = uptime_seconds / 3600

            return {
                "done": done,
                "pending": pending,
                "todo": todo,
                "total": len(stories),
                "uptime_hours": uptime_hours,
                "current_task": self._current_task,
            }

        except Exception as e:
            print(f"[HeartbeatService] Error collecting stats: {e}")
            return {
                "done": 0,
                "pending": 0,
                "todo": 0,
                "total": 0,
                "uptime_hours": (time.time() - self._start_time) / 3600,
                "current_task": self._current_task,
            }

    def _write_heartbeat(self):
        """Write heartbeat.json file."""
        stats = self._collect_stats()

        heartbeat_data = {
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": stats["uptime_hours"],
            "tasks_done": stats["done"],
            "tasks_pending": stats["pending"],
            "tasks_todo": stats["todo"],
            "tasks_total": stats["total"],
            "current_task": stats["current_task"],
            "pid": None,  # Could add os.getpid() if needed
        }

        try:
            # Atomic write: write to temp file, then rename
            tmp_file = self.heartbeat_file.with_suffix('.json.tmp')
            with open(tmp_file, "w") as f:
                json.dump(heartbeat_data, f, indent=2)
            tmp_file.replace(self.heartbeat_file)
        except Exception as e:
            print(f"[HeartbeatService] Error writing heartbeat file: {e}")

    def get_heartbeat_path(self) -> Path:
        """Return the path to heartbeat.json."""
        return self.heartbeat_file
