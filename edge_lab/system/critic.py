import time
import re
import json
from core.state import StateManager
from core.agent_wrapper import AgentWrapper
from config import MAX_ITERATIONS, SLEEP_INTERVAL, BASE_DIR

# JSON schema for structured critic output
CRITIC_OUTPUT_SCHEMA = {
    "decision": "APPROVE|REJECT",
    "reason": "string",
    "metrics": {"MAE": 0.0, "passed_tests": 0},
    "evidence": "filepath or command output"
}

def parse_critic_output(output: str) -> dict:
    """Extract JSON from critic output.

    Tries to find a JSON object in the output that contains 'decision' key.
    Falls back to keyword parsing if no valid JSON found.
    """
    # Try to find JSON in the output
    json_patterns = [
        r'```json\s*(\{[\s\S]*?\})\s*```',  # Markdown code block
        r'```\s*(\{[\s\S]*?\})\s*```',       # Generic code block
        r'(\{[^{}]*"decision"[^{}]*\})',     # Simple JSON with decision
        r'(\{[\s\S]*?"decision"[\s\S]*?\})', # Complex JSON with decision
    ]

    for pattern in json_patterns:
        match = re.search(pattern, output)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if "decision" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

    # Fallback: keyword parsing
    if "APPROVE" in output.upper():
        return {"decision": "APPROVE", "reason": "Extracted from keyword", "fallback": True}
    elif "REJECT" in output.upper():
        reason = output.split("REJECT")[-1].strip().split("\n")[0] if "REJECT" in output else "Unknown"
        return {"decision": "REJECT", "reason": reason, "fallback": True}

    return {"decision": "UNKNOWN", "reason": "Could not parse output", "fallback": True}

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
        You are the CRITIC agent — the last line of defense against FAKE WORK.

        TASK TO REVIEW:
        {pending_task}

        === ADVERSARIAL VERIFICATION PROTOCOL ===
        
        [WARNING]: ASSUME WORKER IS LYING until you independently verify.
        
        MANDATORY VERIFICATION STEPS:
        
        1. CHECK FILES EXIST:
           - Run: ls -la <claimed_file_path>
           - Run: wc -l <file> (verify line count claims)
           
        2. CHECK CODE COMPILES:
           - Run: python3 -m py_compile <file.py>
           
        3. CHECK TESTS PASS (if applicable):
           - Run: pytest <test_file> -v
           - Exit code 0 = pass, anything else = fail
           
        4. CHECK DATA IS REAL (for data tasks):
           - Run: head -5 <data_file.csv>
           - Verify it's not empty/garbage
           
        5. CHECK EVIDENCE IN WORKER OUTPUT:
           - Worker MUST have "Verification commands run:" section
           - Are the outputs realistic? Did the Worker *actually* run them?
           - If Worker says "Done" but output shows error -> REJECT.
           - If Worker shows NO terminal output -> REJECT ("Ghost Work").
           - No evidence = AUTOMATIC REJECT
        

        === DECISION CRITERIA ===
        
        APPROVE only if:
        - ALL acceptance criteria are met
        - YOU ran verification commands yourself
        - Evidence is real (not just Worker's claims)
        
        REJECT if:
        - ANY verification fails
        - Worker provided no evidence
        - Files are empty or stub code
        - Tests fail or don't exist

        === OUTPUT FORMAT (REQUIRED JSON) ===
        
        ```json
        {{
            "decision": "APPROVE",  // or "REJECT"
            "reason": "All 3 acceptance criteria verified",
            "verification_commands_run": [
                {{"cmd": "ls -la file.py", "result": "exists, 150 lines"}},
                {{"cmd": "pytest test.py", "result": "5 passed"}}
            ],
            "criteria_results": [
                {{"criterion": "File exists", "passed": true, "evidence": "ls output"}}
            ],
            "confidence": 0.95
        }}
        ```
        
        IMPORTANT: You MUST run run_command to verify. Trust nothing.
        IMPORTANT: If Worker failed to include "Verification commands run" output, REJECT immediately.

        """
        
        full_context = agent.format_context(state.read_prd(), progress_tail, methodology)
        full_prompt = prompt + full_context
        
        # 4. Execute
        state.append_progress(f"Starting review of Task {pending_task['id']}", "CRITIC")
        output = agent.run(full_prompt)
        
        # 5. Parse Decision using structured JSON
        parsed = parse_critic_output(output)
        decision = parsed.get("decision", "UNKNOWN").upper()
        reason = parsed.get("reason", "No reason provided")
        is_fallback = parsed.get("fallback", False)

        if is_fallback:
            print(f"   ⚠️  Fallback parsing used (no valid JSON in output)")

        if decision == "APPROVE":
            state.update_task_status(pending_task['id'], "DONE")
            metrics = parsed.get("metrics", {})
            confidence = parsed.get("confidence", "N/A")
            print(f"   ✅ Approved Task {pending_task['id']}")
            print(f"      Reason: {reason}")
            if metrics:
                print(f"      Metrics: {metrics}")
            if confidence != "N/A":
                print(f"      Confidence: {confidence}")
        elif decision == "REJECT":
            criteria_results = parsed.get("criteria_results", [])
            failed_criteria = [c for c in criteria_results if not c.get("passed", True)]
            feedback = f"Rejected: {reason}"
            if failed_criteria:
                feedback += f" | Failed: {[c.get('criterion') for c in failed_criteria]}"
            state.update_task_status(pending_task['id'], "TODO", feedback=feedback)
            print(f"   ❌ Rejected Task {pending_task['id']}: {reason}")
            if failed_criteria:
                for fc in failed_criteria:
                    print(f"      - {fc.get('criterion')}: {fc.get('evidence', 'no evidence')}")
        else:
            # UNKNOWN decision
            print(f"   ⚠️  Critic output unclear. Marking as TODO for retry.")
            print(f"      Output tail: {output[-200:]}")
            state.update_task_status(pending_task['id'], "TODO", feedback=f"Critic could not determine status: {reason}")

if __name__ == "__main__":
    main()
