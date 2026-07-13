import subprocess
import json
import shlex
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import config
from config import AGENT_CLI_CMD, PRIMARY_MODEL, CRITIC_MODEL, PROJECT_ROOT

# API activity tracking for liveness detection
API_ACTIVITY_FILE = Path(config.DOCS_DIR) / ".." / "data" / "last_api_call.json"


class AgentWrapper:
    def __init__(self, role: str):
        self.role = role # "worker" or "critic"
        self.model = CRITIC_MODEL if role == "critic" else PRIMARY_MODEL

    def _log_api_call(self, status: str = "started"):
        """Log API call timestamp for liveness detection."""
        try:
            API_ACTIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "timestamp": datetime.now().isoformat(),
                "role": self.role,
                "model": self.model,
                "status": status
            }
            with open(API_ACTIVITY_FILE, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass  # Don't fail on tracking errors

    def run(self, prompt: str) -> str:
        """
        Executes the agent with the given prompt via CLI.
        """
        cmd = AGENT_CLI_CMD + ["--model", self.model]
        
        try:
            # Log API call start for liveness detection
            self._log_api_call("started")
            
            # Pass prompt via STDIN to avoid argument length limits and EBADF errors
            result = subprocess.run(
                cmd, 
                input=prompt,
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if result.returncode != 0:
                with open("agents/last_error.txt", "w") as f:
                    f.write(f"CMD: {cmd}\n")
                    f.write(f"STDERR: {result.stderr}\n")
                    f.write(f"STDOUT: {result.stdout}\n")
                return f"ERROR: Agent CLI failed with code {result.returncode}.\nStderr: {result.stderr}"
            
            with open("agents/last_success_log.txt", "w") as f:
                 f.write(f"Output: {result.stdout[:100]}...")

            return result.stdout.strip()

        except FileNotFoundError:
            return "ERROR: Agent CLI command not found. Check config.py."
        except Exception as e:
            return f"ERROR: Exception while running agent: {str(e)}"

    def run_command(self, cmd: str, timeout: int = 300, cwd: Optional[str] = None) -> dict:
        """Execute shell command and return structured result.

        Used by Critic for real verification of task completion.

        Args:
            cmd: Shell command to execute
            timeout: Max execution time in seconds (default 5 min)
            cwd: Working directory (defaults to PROJECT_ROOT)

        Returns:
            dict with keys: stdout, stderr, exit_code, success, command
        """
        work_dir = cwd or str(PROJECT_ROOT)

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )

            return {
                "command": cmd,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "exit_code": result.returncode,
                "success": result.returncode == 0,
                "cwd": work_dir
            }

        except subprocess.TimeoutExpired:
            return {
                "command": cmd,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "exit_code": -1,
                "success": False,
                "cwd": work_dir
            }
        except Exception as e:
            return {
                "command": cmd,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -2,
                "success": False,
                "cwd": work_dir
            }

    def format_context(self, prd: Dict, progress_tail: str, methodology: str) -> str:
        return f"""
CONTEXT:
- Role: {self.role.upper()}
- Task List (PRD):
{json.dumps(prd, indent=2)}

- Recent History:
{progress_tail}

- Methodology:
{methodology}
"""
