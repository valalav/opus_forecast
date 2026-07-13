import subprocess
import json
from typing import List, Dict
import config
from config import AGENT_CLI_CMD, PRIMARY_MODEL, CRITIC_MODEL

class AgentWrapper:
    def __init__(self, role: str):
        self.role = role # "worker" or "critic"
        self.model = CRITIC_MODEL if role == "critic" else PRIMARY_MODEL

    def run(self, prompt: str) -> str:
        """
        Executes the agent with the given prompt via CLI.
        """
        cmd = AGENT_CLI_CMD + ["--model", self.model, prompt]
        
        try:
            # Using subprocess to call the CLI tool
            # Assuming the CLI tool accepts the prompt as the last argument
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=False  # We handle errors manually
            )
            
            if result.returncode != 0:
                return f"ERROR: Agent CLI failed with code {result.returncode}.\nStderr: {result.stderr}"
            
            return result.stdout.strip()
            
        except FileNotFoundError:
            return "ERROR: Agent CLI command not found. Check config.py."
        except Exception as e:
            return f"ERROR: Exception while running agent: {str(e)}"

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
