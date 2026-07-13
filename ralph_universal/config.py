import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.absolute()
DOCS_DIR = BASE_DIR / "docs"
PRD_FILE = DOCS_DIR / "prd.json"
PROGRESS_FILE = DOCS_DIR / "progress.txt"

# Ensure docs exist
DOCS_DIR.mkdir(exist_ok=True)

# Loop Configuration
MAX_ITERATIONS = 50
SLEEP_INTERVAL = 2  # Seconds between loop cycles

# Model Configuration
PRIMARY_MODEL = "opencode/glm-4.7-free"  # Or your preferred model
CRITIC_MODEL = "opencode/glm-4.7-free"   # Critic should be smart

# Agent Interface
# This command is used to invoke the actual agent logic (CLI wrapper)
AGENT_CLI_CMD = ["opencode", "run"]

# Colors for CLI Output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
