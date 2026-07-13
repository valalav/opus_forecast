import multiprocessing
import time
import sys
from pathlib import Path

# Add the current directory to path so we can import worker and critic
sys.path.append(str(Path(__file__).parent))

import worker
import critic
from config import BASE_DIR, PRD_FILE

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

def main():
    print(f"🎭 Ralph Universal Orchestrator starting in {BASE_DIR}")
    
    if not PRD_FILE.exists():
        print(f"⚠️  No PRD found at {PRD_FILE}. Creating sample...")
        # Create a sample PRD if none exists
        from core.state import StateManager
        state = StateManager()
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

    # Start processes
    p_worker.start()
    p_critic.start()

    print("   -> Processes started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
            # Check if processes are alive
            if not p_worker.is_alive():
                print("⚠️  Worker process died. Attempting to restart...")
                p_worker = multiprocessing.Process(target=run_worker, name="Worker")
                p_worker.start()
            
            if not p_critic.is_alive():
                 print("⚠️  Critic process died. Attempting to restart...")
                 p_critic = multiprocessing.Process(target=run_critic, name="Critic")
                 p_critic.start()

    except KeyboardInterrupt:
        print("\n🛑 Stopping Ralph Universal...")
        p_worker.terminate()
        p_critic.terminate()
        p_worker.join()
        p_critic.join()
        print("   -> Stopped.")

if __name__ == "__main__":
    main()
