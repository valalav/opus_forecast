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

        # Event handlers for monitoring (Telegram notifications, etc.)
        self.event_handlers = []

    def register_event_handler(self, handler):
        """
        Register an event handler for monitoring.

        Args:
            handler: Callable(event_type: str, data: dict)
        """
        self.event_handlers.append(handler)

    def _emit_event(self, event_type: str, data: dict):
        """
        Emit an event to all registered handlers.

        Args:
            event_type: Type of event (TASK_STATUS_CHANGE, TASK_TIMEOUT, etc.)
            data: Event data dict.
        """
        for handler in self.event_handlers:
            try:
                handler(event_type, data)
            except Exception as e:
                # Don't crash on notification failure
                print(f"[StateManager] Event handler error: {e}")

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
                old_status = story.get("status", "TODO")
                old_feedback = story.get("feedback", "")
                
                # PROTECTION: Don't allow PENDING_REVIEW if task was just rejected
                # This prevents Worker from overriding Critic's rejection
                if status == "PENDING_REVIEW" and old_feedback and "Reject" in old_feedback:
                    # Increment rejection count - Worker is trying again without fixing the issue
                    reject_count = story.get("rejection_count", 0) + 1
                    story["rejection_count"] = reject_count
                    
                    self.append_progress(
                        f"Task {task_id}: Blocked PENDING_REVIEW (has rejection feedback, attempt {reject_count})", 
                        "STATE"
                    )
                    
                    # Auto-block after 3 failed attempts to fix rejection
                    if reject_count >= 3:
                        story["status"] = "BLOCKED"
                        story["feedback"] = f"BLOCKED: Worker failed to address rejection {reject_count} times. Original: {old_feedback[:100]}"
                        self.write_prd(prd)
                        self.append_progress(f"Task {task_id} auto-BLOCKED after {reject_count} rejection cycles", "STATE")
                        self._emit_event('TASK_BLOCKED', {
                            'task_id': task_id,
                            'title': story.get('title', 'Unknown'),
                            'reason': f'Failed to fix rejection {reject_count} times',
                        })
                        return
                    
                    # Clear the rejection feedback to give Worker one more chance
                    story["feedback"] = f"Retry {reject_count}/3: {old_feedback}"
                    self.write_prd(prd)
                    return  # Don't update status, but feedback is updated
                
                # PROTECTION: Don't allow setting DONE from Worker (only Critic can APPROVE)
                # Worker can only set PENDING_REVIEW, Critic sets DONE or TODO
                if status == "DONE" and old_status == "PENDING_REVIEW":
                    # This is Critic approving - allowed
                    pass
                elif status == "DONE" and old_status == "TODO":
                    # Direct TODO -> DONE should only happen via manual intervention
                    self.append_progress(
                        f"Task {task_id}: Direct TODO->DONE transition (manual?)", 
                        "STATE"
                    )
                
                story["status"] = status
                if feedback:
                    story["feedback"] = feedback
                elif status == "DONE":
                    # Clear rejection feedback when task is approved
                    story["feedback"] = "Approved by Critic"
                updated = True
                break
        
        if updated:
            self.write_prd(prd)
            self.append_progress(f"Task {task_id} status updated to {status}", "STATE")

            # Emit event for monitoring (Telegram notifications)
            self._emit_event('TASK_STATUS_CHANGE', {
                'task_id': task_id,
                'status': status,
                'old_status': old_status,
                'feedback': feedback,
                'title': story.get('title', 'Unknown'),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            })

    def block_task(self, task_id: int, reason: str = "Multiple rejections"):
        """Block a task and emit notification."""
        prd = self.read_prd()
        for story in prd.get("user_stories", []):
            if story["id"] == task_id:
                story["status"] = "BLOCKED"
                story["feedback"] = f"BLOCKED: {reason}"
                self.write_prd(prd)
                self.append_progress(f"Task {task_id} BLOCKED: {reason}", "STATE")

                self._emit_event('TASK_BLOCKED', {
                    'task_id': task_id,
                    'title': story.get('title', 'Unknown'),
                    'reason': reason,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                return True
        return False

    def increment_rejection_count(self, task_id: int) -> int:
        """Increment rejection count and auto-block if >= 3."""
        prd = self.read_prd()
        for story in prd.get("user_stories", []):
            if story["id"] == task_id:
                count = story.get("rejection_count", 0) + 1
                story["rejection_count"] = count
                self.write_prd(prd)

                # Auto-block after 3 rejections
                if count >= 3:
                    self.block_task(task_id, f"Auto-blocked after {count} rejections")

                return count
        return 0

    def add_subtasks(self, parent_id: int, subtasks: List[Dict]):
        """Add subtasks from Refiner and emit notification."""
        prd = self.read_prd()
        parent_title = None

        # Find parent task
        for story in prd.get("user_stories", []):
            if story["id"] == parent_id:
                parent_title = story.get("title", "Unknown")
                story["status"] = "REFINED"
                story["subtask_ids"] = []
                break

        if not parent_title:
            return False

        # Get max ID
        max_id = max((s.get("id", 0) for s in prd.get("user_stories", [])), default=0)

        # Add subtasks
        new_ids = []
        for i, subtask in enumerate(subtasks):
            new_id = max_id + 1 + i
            new_ids.append(new_id)
            prd["user_stories"].append({
                "id": new_id,
                "title": subtask.get("title", f"Subtask {i+1}"),
                "description": subtask.get("description", ""),
                "acceptance_criteria": subtask.get("acceptance_criteria", []),
                "priority": subtask.get("priority", "medium"),
                "status": "TODO",
                "parent_id": parent_id,
            })

        # Update parent with subtask IDs
        for story in prd.get("user_stories", []):
            if story["id"] == parent_id:
                story["subtask_ids"] = new_ids
                break

        self.write_prd(prd)
        self.append_progress(f"Task {parent_id} refined into {len(subtasks)} subtasks", "REFINER")

        self._emit_event('TASK_REFINED', {
            'task_id': parent_id,
            'title': parent_title,
            'subtasks': subtasks,
            'subtask_count': len(subtasks),
            'subtask_ids': new_ids,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        return True

