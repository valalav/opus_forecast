#!/usr/bin/env python3
"""
Watchdog - Standalone Supervisor for Ralph.

This script runs INDEPENDENTLY from the orchestrator.
It monitors heartbeat.json and sends alerts if Ralph stops responding.

Usage:
    python3 edge_lab/system/monitoring/watchdog.py

    # Or in background:
    screen -dmS ralph-watchdog python3 edge_lab/system/monitoring/watchdog.py
    nohup python3 edge_lab/system/monitoring/watchdog.py > watchdog.log 2>&1 &
"""
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.telegram_bot import TelegramNotifier


class Watchdog:
    """
    Standalone supervisor that monitors Ralph's heartbeat.

    - Checks heartbeat.json every CHECK_INTERVAL seconds
    - Sends alert if heartbeat is older than HEARTBEAT_TIMEOUT
    - Has cooldown between repeated alerts
    """

    CHECK_INTERVAL = 300       # Check every 5 minutes
    HEARTBEAT_TIMEOUT = 1800   # Alert if heartbeat older than 30 minutes
    ALERT_COOLDOWN = 3600      # 1 hour between repeated alerts

    def __init__(self, heartbeat_file: Optional[Path] = None):
        """
        Initialize Watchdog.

        Args:
            heartbeat_file: Path to heartbeat.json. Default: edge_lab/data/heartbeat.json
        """
        # Determine heartbeat file path
        if heartbeat_file:
            self.heartbeat_file = heartbeat_file
        else:
            base = Path(__file__).parent.parent.parent  # edge_lab/
            self.heartbeat_file = base / "data" / "heartbeat.json"

        self.notifier = TelegramNotifier()
        self._last_alert_time = 0
        self._was_alive = True  # Track state for recovery notification

    def check_heartbeat(self) -> dict:
        """
        Check the heartbeat file and return status.

        Returns:
            dict with keys:
                - alive: bool
                - age_seconds: float (how old the heartbeat is)
                - stats: dict (from heartbeat file, if available)
                - error: str (if any error occurred)
        """
        if not self.heartbeat_file.exists():
            return {
                "alive": False,
                "age_seconds": float("inf"),
                "stats": None,
                "error": "Heartbeat file not found",
            }

        try:
            with open(self.heartbeat_file, "r") as f:
                data = json.load(f)

            timestamp_str = data.get("timestamp")
            if not timestamp_str:
                return {
                    "alive": False,
                    "age_seconds": float("inf"),
                    "stats": data,
                    "error": "No timestamp in heartbeat file",
                }

            # Parse ISO timestamp
            heartbeat_time = datetime.fromisoformat(timestamp_str)
            now = datetime.now()
            age_seconds = (now - heartbeat_time).total_seconds()

            alive = age_seconds < self.HEARTBEAT_TIMEOUT

            return {
                "alive": alive,
                "age_seconds": age_seconds,
                "stats": data,
                "error": None,
            }

        except json.JSONDecodeError as e:
            return {
                "alive": False,
                "age_seconds": float("inf"),
                "stats": None,
                "error": f"Invalid JSON: {e}",
            }
        except Exception as e:
            return {
                "alive": False,
                "age_seconds": float("inf"),
                "stats": None,
                "error": str(e),
            }

    def check_process(self) -> bool:
        """
        Check if orchestrator process is running.

        Returns:
            True if process is found, False otherwise.
        """
        try:
            result = subprocess.run(
                ["pgrep", "-f", "orchestrator.py"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _can_send_alert(self) -> bool:
        """Check if we can send an alert (respecting cooldown)."""
        now = time.time()
        if now - self._last_alert_time >= self.ALERT_COOLDOWN:
            self._last_alert_time = now
            return True
        return False

    def run(self):
        """Main watchdog loop."""
        print(f"[Watchdog] Starting...")
        print(f"  Heartbeat file: {self.heartbeat_file}")
        print(f"  Check interval: {self.CHECK_INTERVAL}s")
        print(f"  Timeout threshold: {self.HEARTBEAT_TIMEOUT}s")
        print(f"  Alert cooldown: {self.ALERT_COOLDOWN}s")

        # Send startup notification
        self.notifier.send("Watchdog started - monitoring Ralph")

        while True:
            try:
                status = self.check_heartbeat()

                if status["alive"]:
                    if not self._was_alive:
                        # Ralph recovered!
                        self.notifier.send("Ralph is back online!")
                        self._was_alive = True

                    # Log status
                    stats = status.get("stats", {})
                    done = stats.get("tasks_done", "?")
                    pending = stats.get("tasks_pending", "?")
                    age = status["age_seconds"]
                    print(f"[Watchdog] Ralph alive (heartbeat {age:.0f}s ago, {done} done, {pending} pending)")

                else:
                    self._was_alive = False

                    # Check if process exists but heartbeat is stale
                    process_running = self.check_process()

                    if process_running:
                        msg = f"[Watchdog] WARNING: Orchestrator running but heartbeat stale ({status['age_seconds']:.0f}s)"
                    else:
                        msg = f"[Watchdog] ALERT: Ralph not responding! Heartbeat {status['age_seconds']:.0f}s old"

                    print(msg)

                    if status.get("error"):
                        print(f"  Error: {status['error']}")

                    # Send alert if cooldown allows
                    if self._can_send_alert():
                        self.notifier.send_system_down_alert(status["age_seconds"])

            except Exception as e:
                print(f"[Watchdog] Error in check loop: {e}")

            # Wait for next check
            time.sleep(self.CHECK_INTERVAL)


def main():
    """Entry point for watchdog."""
    watchdog = Watchdog()

    try:
        watchdog.run()
    except KeyboardInterrupt:
        print("\n[Watchdog] Stopped by user")
        watchdog.notifier.send("Watchdog stopped")


if __name__ == "__main__":
    main()
