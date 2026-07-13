import time
import signal
from core.state import StateManager
from core.agent_wrapper import AgentWrapper
from config import MAX_ITERATIONS, SLEEP_INTERVAL, BASE_DIR, PROJECT_ROOT, SAFETY_LIMITS, TASK_RETRY_COUNTS


class TaskTimeout(Exception):
    """Raised when a task exceeds max duration."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for task timeout."""
    raise TaskTimeout("Task exceeded maximum duration")

def main():
    state = StateManager()
    agent = AgentWrapper("worker")
    
    print(f"🚀 Worker started in {BASE_DIR}")

    for i in range(MAX_ITERATIONS):
        time.sleep(SLEEP_INTERVAL)
        
        # 1. Fetch Plan
        tasks = state.read_prd().get("user_stories", [])
        
        # 2. Select Next Task (TODO)
        next_task = None
        for t in tasks:
            status = t.get("status", "TODO")
            if status == "TODO":
                next_task = t
                break
        
        if not next_task:
            # Check if any are pending
            pending = [t for t in tasks if t.get("status") == "PENDING_REVIEW"]
            if not pending:
                print("✅ checking tasks... No tasks left (All DONE). Worker exiting.")
                break
            else:
                print(f"⏳ checking tasks... Waiting for Critic ({len(pending)} pending)...")
                continue
                
        # 3. Formulate Prompt
        print(f"🛠️  Worker processing Task {next_task['id']}: {next_task['title']}")
        if next_task.get("feedback"):
             print(f"   ⚠️  Feedback from Critic: {next_task['feedback']}")
        
        # Read recent progress for context
        with open(state.progress_path, "r") as f:
            lines = f.readlines()
            progress_tail = "".join(lines[-20:])

        with open(BASE_DIR / "GEMINI.md", "r") as f:
            methodology = f.read()

        prompt = f"""
        You are the WORKER agent in a Quality-First system.
        
        TASK:
        {next_task}
        
        === QUALITY-FIRST METHODOLOGY (MANDATORY) ===
        
        PHASE 1: RESEARCH (Before any code)
        - List relevant files: ls -la <target_dir>
        - Read existing code: head -100 <file>
        - State your approach BEFORE implementing
        
        PHASE 2: IMPLEMENT
        - Work ONLY in: {PROJECT_ROOT}
        - DO NOT modify files outside this directory
        - Use 'Red-Green-Refactor' method
        - Create REAL output files (not just print statements)
        
        PHASE 3: SELF-VERIFICATION (CRITICAL)
        - You MUST run the verification command YOURSELF.
        - If the command fails, DO NOT CLAIM COMPLETION. Fix the code.
        - "Fake Work" (claiming success without running the check) is a specific violation of protocol.
        
        3.1. VERIFY CODING TASKS:
             - Run: python3 -m py_compile <your_file.py>
             - Run: pytest <test_file> -v
        
        3.2. VERIFY DATA TASKS:
             - Run: head -5 <output.csv> (Show me the data!)
             - Run: wc -l <output.csv> (Show me the rows!)
        
        === COMPLETION FORMAT (REQUIRED) ===
        
        Your output MUST end with:
        
        COMPLETED_TASK
        
        === Evidence of Work ===
        Files created/modified:
        - <path> (<line count> lines)
        
        Verification commands run (I have actually run these):
        $ <command>
        <real_terminal_output_of_command>
        
        === END ===
        
        [WARNING]: The Critic will re-run your commands. If they fail, you will be rejected.
        [WARNING]: "COMPLETED_TASK" without evidence = REJECTED.
        [WARNING]: Fake work = REJECTED.
        
        """
        
        full_context = agent.format_context(state.read_prd(), progress_tail, methodology)
        full_prompt = prompt + full_context
        
        # 4. Execute with TIMEOUT protection
        task_id = next_task['id']
        max_duration = SAFETY_LIMITS['max_task_duration_seconds']
        max_retries = SAFETY_LIMITS['max_retries_per_task']
        
        # Track retries
        if task_id not in TASK_RETRY_COUNTS:
            TASK_RETRY_COUNTS[task_id] = 0
        
        # Check if exceeded max retries
        if TASK_RETRY_COUNTS[task_id] >= max_retries:
            print(f"   🚫 Task {task_id} exceeded max retries ({max_retries}). Marking as BLOCKED...")
            state.append_progress(f"Task {task_id} BLOCKED after {max_retries} failed attempts - requires manual intervention", "WORKER")
            state.update_task_status(task_id, "BLOCKED", 
                feedback=f"BLOCKED: Exceeded {max_retries} retry attempts. Task needs to be refined or decomposed. Use 'python3 add_task.py --blocked' to view.")
            continue
        
        state.append_progress(f"Starting work on Task {task_id} (attempt {TASK_RETRY_COUNTS[task_id] + 1}/{max_retries})", "WORKER")
        
        # Set up timeout (Unix only)
        output = ""
        timed_out = False
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(max_duration)  # Set alarm
            
            output = agent.run(full_prompt)
            
            signal.alarm(0)  # Cancel alarm on success
        except TaskTimeout:
            timed_out = True
            signal.alarm(0)  # Cancel alarm
            print(f"   ⏰ TIMEOUT: Task {task_id} exceeded {max_duration}s limit!")
            state.append_progress(f"Task {task_id} TIMEOUT after {max_duration}s", "WORKER")
            TASK_RETRY_COUNTS[task_id] += 1
        except Exception as e:
            signal.alarm(0)  # Cancel alarm
            print(f"   🔥 ERROR: Task {task_id} crashed: {e}")
            state.append_progress(f"Task {task_id} ERROR: {str(e)[:100]}", "WORKER")
            TASK_RETRY_COUNTS[task_id] += 1
        
        # 5. Handle Result
        if timed_out:
            # Don't update status, let it retry
            continue
            
        print("   -> execution finished.")
        state.append_progress(f"Finished execution cycle for Task {task_id}", "WORKER")
        
        # Strict check for completion token
        if "COMPLETED_TASK" in output:
            TASK_RETRY_COUNTS[task_id] = 0  # Reset on success
            state.update_task_status(task_id, "PENDING_REVIEW")
        else:
            print("   ⚠️  Worker did not output COMPLETED_TASK. Incrementing retry count.")
            TASK_RETRY_COUNTS[task_id] += 1
            state.append_progress(f"Worker failed to complete Task {task_id} (no COMPLETED_TASK token)", "WORKER")


if __name__ == "__main__":
    main()
