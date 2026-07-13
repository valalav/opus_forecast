import time
from core.state import StateManager
from core.agent_wrapper import AgentWrapper
from config import MAX_ITERATIONS, SLEEP_INTERVAL, BASE_DIR

def main():
    state = StateManager()
    agent = AgentWrapper("critic")
    
    print(f"🧐 Critic started in {BASE_DIR}")

    for i in range(MAX_ITERATIONS):
        time.sleep(SLEEP_INTERVAL)
        
        # 1. Fetch Plan
        tasks = state.read_prd().get("user_stories", [])
        
        # 2. Select Next Task (PENDING_REVIEW)
        pending_task = None
        for t in tasks:
            if t.get("status") == "PENDING_REVIEW":
                pending_task = t
                break
        
        if not pending_task:
            print("💤 checking tasks... No pending tasks to review.")
            continue
            
        # 3. Formulate Prompt
        print(f"🔍 Critic reviewing Task {pending_task['id']}: {pending_task['title']}")
        
        with open(state.progress_path, "r") as f:
            lines = f.readlines()
            progress_tail = "".join(lines[-20:])

        with open(BASE_DIR / "GEMINI.md", "r") as f:
            methodology = f.read()

        prompt = f"""
        You are the CRITIC agent.
        
        TASK TO REVIEW:
        {pending_task}
        
        DIRECTIVE:
        1. Verify if the code for this task exists and works.
        2. Run the acceptance criteria tests (using 'run_command' or similar if available, or just check the file content).
        3. If satisfied:
           - Output "APPROVE"
        4. If rejected:
           - Output "REJECT: <reason>"
        
        """
        
        full_context = agent.format_context(state.read_prd(), progress_tail, methodology)
        full_prompt = prompt + full_context
        
        # 4. Execute
        state.append_progress(f"Starting review of Task {pending_task['id']}", "CRITIC")
        output = agent.run(full_prompt)
        
        # 5. Parse Decision
        decision = "UNKNOWN"
        if "APPROVE" in output:
            decision = "APPROVE"
            state.update_task_status(pending_task['id'], "DONE")
            print(f"   ✅ Approved Task {pending_task['id']}")
        elif "REJECT" in output:
            decision = "REJECT"
            # Extract feedback crudely
            feedback = output.split("REJECT")[-1].strip().split("\n")[0]
            state.update_task_status(pending_task['id'], "TODO", feedback=f"Rejected by Critic: {feedback}")
            print(f"   ❌ Rejected Task {pending_task['id']}: {feedback}")
        else:
            # Fallback if the model was chatty but didn't output keywords
            print(f"   ⚠️  Critic output unclear. Marking as TODO for retry. Output tail: {output[-100:]}")
            state.update_task_status(pending_task['id'], "TODO", feedback="Critic could not determine status (Output unclear).")

if __name__ == "__main__":
    main()
