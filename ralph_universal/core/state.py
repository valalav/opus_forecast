import json
import fcntl
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import config
from config import PRD_FILE, PROGRESS_FILE

@dataclass
class Task:
    id: int
    title: str
    description: str
    acceptance_criteria: List[str]
    priority: str
    status: str  # "TODO", "PENDING_REVIEW", "DONE"
    feedback: Optional[str] = None

class StateManager:
    def __init__(self):
        self.prd_path = PRD_FILE
        self.progress_path = PROGRESS_FILE
        self.lock_file = self.prd_path.with_suffix(".lock")

    def _acquire_lock(self):
        """Acquire an exclusive lock on the lock file."""
        self.lock_fd = open(self.lock_file, "w")
        fcntl.flock(self.lock_fd, fcntl.LOCK_EX)

    def _release_lock(self):
        """Release the exclusive lock."""
        fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
        self.lock_fd.close()

    def read_prd(self) -> Dict[str, Any]:
        """Thread-safe read of PRD."""
        if not self.prd_path.exists():
            return {"project": "New Project", "user_stories": []}
            
        self._acquire_lock()
        try:
            with open(self.prd_path, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {"project": "New Project", "user_stories": []}
        finally:
            self._release_lock()

    def write_prd(self, data: Dict[str, Any]):
        """Thread-safe write of PRD."""
        self._acquire_lock()
        try:
            with open(self.prd_path, "w") as f:
                json.dump(data, f, indent=2)
        finally:
            self._release_lock()

    def append_progress(self, message: str, agent_type: str = "SYSTEM"):
        """Append a log entry to progress.txt."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{agent_type}] {message}\n"
        
        # Simple append doesn't strictly need a heavy lock if writes are small and atomic-ish on Linux,
        # but usage of the same lock ensures consistency if we wanted to read-modify-write.
        # For simple appending, we can just open in append mode.
        with open(self.progress_path, "a") as f:
            f.write(entry)

    def get_pending_tasks(self) -> List[Dict]:
        prd = self.read_prd()
        # Find tasks that are not DONE
        # Old format might use 'passes': false
        # New format uses 'status': 'TODO' | 'PENDING_REVIEW'
        tasks = []
        for story in prd.get("user_stories", []):
            status = story.get("status", "TODO")
            # Migration logic for old format
            if "passes" in story and "status" not in story:
                status = "DONE" if story["passes"] else "TODO"
            
            if status != "DONE":
                tasks.append(story)
        return tasks

    def update_task_status(self, task_id: int, status: str, feedback: str = None):
        prd = self.read_prd()
        updated = False
        for story in prd.get("user_stories", []):
            if story["id"] == task_id:
                story["status"] = status
                if feedback:
                    story["feedback"] = feedback
                elif feedback is None and status == "TODO":
                     # Clear feedback if moving back to TODO? 
                     # Maybe keep history? for now, let's keep it simple.
                     pass
                updated = True
                break
        
        if updated:
            self.write_prd(prd)
            self.append_progress(f"Task {task_id} status updated to {status}", "STATE")
