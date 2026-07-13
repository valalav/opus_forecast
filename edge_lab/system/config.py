import os
from pathlib import Path

# Base Paths (Modified for Sandbox)
BASE_DIR = Path(__file__).parent.absolute()
# Tasks and progress should be in ../tasks/ relative to system/
PROJECT_ROOT = BASE_DIR.parent
DOCS_DIR = PROJECT_ROOT / "tasks"
PRD_FILE = DOCS_DIR / "prd.json"
PROGRESS_FILE = DOCS_DIR / "progress.txt"

# Ensure docs exist
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Loop Configuration
MAX_ITERATIONS = 1000  # Increased from 50 to allow longer runs
SLEEP_INTERVAL = 2

# Model Configuration
PRIMARY_MODEL = "zai-coding-plan/glm-4.7"
CRITIC_MODEL = "zai-coding-plan/glm-4.7"

# Agent Interface
AGENT_CLI_CMD = ["opencode", "run"]

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

# Safety Limits (prevent overnight hangs)
SAFETY_LIMITS = {
    'max_task_duration_seconds': 1800,  # 30 minutes per task
    'max_retries_per_task': 3,          # Max attempts before skipping
    'max_file_size_mb': 100,            # Skip files > 100MB in parsers
    'heartbeat_interval_seconds': 60,   # Watchdog check interval
    'stuck_threshold_seconds': 3600,    # 1 hour = consider stuck
}

# Track retry counts (in-memory, reset on restart)
TASK_RETRY_COUNTS = {}

