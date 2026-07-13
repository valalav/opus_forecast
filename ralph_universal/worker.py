import time
from core.state import StateManager
from core.agent_wrapper import AgentWrapper
from config import MAX_ITERATIONS, SLEEP_INTERVAL, BASE_DIR

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
        You are the WORKER agent.
        
        TASK:
        {next_task}
        
        DIRECTIVE:
        1. Implement the requested feature/fix using 'Red-Green-Refactor'.
        2. Create a test verification script if one does not exist.
        3. Run the test to confirm it passes.
        4. If it fails, fix the code until it passes.
        5. DO NOT update 'prd.json' yourself. The system handles that.
        6. OUTPUT "COMPLETED_TASK" at the very end when you are done.
        
        """
        
        full_context = agent.format_context(state.read_prd(), progress_tail, methodology)
        full_prompt = prompt + full_context
        
        # 4. Execute
        state.append_progress(f"Starting work on Task {next_task['id']}", "WORKER")
        output = agent.run(full_prompt)
        
        # 5. Handle Result
        print("   -> execution finished.")
        # We assume the agent did the work (file edits). Now we update state.
        # Ideally, we parse the output to see if it actually finished or crashed.
        
        state.append_progress(f"Finished execution cycle for Task {next_task['id']}", "WORKER")
        
        # Move to PENDING_REVIEW regardless of output analysis for now (MVP)
        # In a real system, we'd check for "COMPLETED_TASK" string.
        if "COMPLETED_TASK" in output or True: # forcing true for MVP structure
             state.update_task_status(next_task['id'], "PENDING_REVIEW")

if __name__ == "__main__":
    main()
