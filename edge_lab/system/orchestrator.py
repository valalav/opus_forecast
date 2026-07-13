import multiprocessing
import time
import sys
import atexit
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (e.g., Z.AI API key) from .env
load_dotenv()

# Add the current directory to path so we can import worker and critic
sys.path.append(str(Path(__file__).parent))

import worker
import critic
import refiner
from config import BASE_DIR, PRD_FILE
from core.state import StateManager
from monitoring.telegram_bot import TelegramNotifier
from monitoring.heartbeat import HeartbeatService

# Global monitoring services
_notifier = None
_heartbeat = None

def run_worker():
    try:
        worker.main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"🔥 Worker Process Crashed: {e}")

def run_critic():
    try:
        critic.main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"🔥 Critic Process Crashed: {e}")

def run_refiner():
    try:
        refiner.main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"🔥 Refiner Process Crashed: {e}")

def main():
    global _notifier, _heartbeat

    print(f"Ralph Universal Orchestrator starting in {BASE_DIR}")

    # Initialize monitoring
    state = StateManager()
    _notifier = TelegramNotifier()
    _heartbeat = HeartbeatService(state, _notifier, interval_seconds=900)

    # Register Telegram notifier to receive events
    state.register_event_handler(_notifier.handle_event)

    # Start heartbeat service (background thread)
    _heartbeat.start()

    # Register cleanup on exit
    def cleanup():
        if _heartbeat:
            _heartbeat.stop()
        if _notifier:
            _notifier.send_shutdown_message()

    atexit.register(cleanup)

    # Send startup notification
    _notifier.send_startup_message()

    if not PRD_FILE.exists():
        print(f"  No PRD found at {PRD_FILE}. Creating sample...")
        state.write_prd({
            "project": "Ralph Universal Test",
            "user_stories": [
                {
                    "id": 1,
                    "title": "Create Verification File",
                    "description": "Create a file named 'verify_me.txt' with content 'Hello World'",
                    "acceptance_criteria": [
                        "File 'verify_me.txt' exists",
                        "File content contains 'Hello World'"
                    ],
                    "priority": "high",
                    "status": "TODO"
                }
            ]
        })

    # Create processes
    p_worker = multiprocessing.Process(target=run_worker, name="Worker")
    p_critic = multiprocessing.Process(target=run_critic, name="Critic")
    p_refiner = multiprocessing.Process(target=run_refiner, name="Refiner")

    # Start processes
    p_worker.start()
    p_critic.start()
    p_refiner.start()

    print("  Processes started. Press Ctrl+C to stop.")
    print(f"  Telegram notifications: {'enabled' if _notifier.enabled else 'disabled'}")
    print(f"  Heartbeat interval: {_heartbeat.interval}s")

    try:
        while True:
            time.sleep(1)
            # Check if processes are alive
            if not p_worker.is_alive():
                print("[Orchestrator] Worker process died. Restarting...")
                _notifier.send("Worker process crashed - restarting")
                p_worker = multiprocessing.Process(target=run_worker, name="Worker")
                p_worker.start()

            if not p_critic.is_alive():
                print("[Orchestrator] Critic process died. Restarting...")
                _notifier.send("Critic process crashed - restarting")
                p_critic = multiprocessing.Process(target=run_critic, name="Critic")
                p_critic.start()

            if not p_refiner.is_alive():
                print("[Orchestrator] Refiner process died. Restarting...")
                _notifier.send("Refiner process crashed - restarting")
                p_refiner = multiprocessing.Process(target=run_refiner, name="Refiner")
                p_refiner.start()

    except KeyboardInterrupt:
        print("\n🛑 Stopping Ralph Universal...")
        p_worker.terminate()
        p_critic.terminate()
        p_refiner.terminate()
        p_worker.join()
        p_critic.join()
        p_refiner.join()
        print("   -> Stopped.")

if __name__ == "__main__":
    main()
