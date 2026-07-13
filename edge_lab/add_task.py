#!/usr/bin/env python3
"""
CLI Tool for adding tasks to Ralph's prd.json

Usage:
    python add_task.py --title "Test NewModel" --priority high --type test
    python add_task.py --title "Fix Bug X" --description "Detailed desc" --mvac "@functional: pytest passes"
    python add_task.py --interactive  # Interactive mode with prompts
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PRD_PATH = SCRIPT_DIR / "tasks" / "prd.json"

# MVAC Templates by task type
MVAC_TEMPLATES = {
    "test": [
        "@file: tests/test_{name}.py exists (>30 lines)",
        "@functional: pytest tests/test_{name}.py -v exits with code 0",
        "@metric: Test count >= 3 (grep -c 'def test_' tests/test_{name}.py)"
    ],
    "model": [
        "@file: sirena/models/{name}.py exists (>50 lines)",
        "@functional: python -c 'from sirena.models.{name} import *' passes",
        "@metric: MAE documented in backtest results"
    ],
    "script": [
        "@file: scripts/{name}.py exists (>30 lines)",
        "@functional: python3 scripts/{name}.py exits with code 0",
        "@metric: Output file generated as expected"
    ],
    "mining": [
        "@file: data/{name}.csv exists",
        "@functional: Script runs without error",
        "@metric: Output contains > 50 rows of data"
    ],
    "docs": [
        "@file: docs/{name}.md exists (>50 lines)",
        "@functional: Markdown renders correctly",
        "@metric: All sections from requirements present"
    ],
    "integration": [
        "@functional: Full pipeline runs without error",
        "@file: Expected output files exist",
        "@metric: All components integrated successfully"
    ],
    "custom": []  # User provides their own MVAC
}

VALID_PRIORITIES = ["high", "medium", "low"]
VALID_STATUSES = ["TODO", "DONE", "PENDING_REVIEW", "BLOCKED"]


def load_prd() -> dict:
    """Load the PRD file."""
    if not PRD_PATH.exists():
        print(f"❌ Error: PRD file not found at {PRD_PATH}")
        sys.exit(1)
    
    with open(PRD_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_prd(prd: dict) -> None:
    """Save the PRD file atomically."""
    tmp_path = PRD_PATH.with_suffix('.json.tmp')
    
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(prd, f, indent=2, ensure_ascii=False)
    
    # Atomic rename
    tmp_path.replace(PRD_PATH)
    print(f"✅ Saved to {PRD_PATH}")


def get_next_id(prd: dict) -> int:
    """Get the next available task ID."""
    existing_ids = [task.get('id', 0) for task in prd.get('user_stories', [])]
    return max(existing_ids, default=0) + 1


def check_duplicate_title(prd: dict, title: str) -> bool:
    """Check if a task with the same title already exists."""
    for task in prd.get('user_stories', []):
        if task.get('title', '').lower() == title.lower():
            return True
    return False


def show_blocked_tasks() -> None:
    """Show all BLOCKED tasks that need manual intervention."""
    prd = load_prd()
    blocked = [t for t in prd.get('user_stories', []) if t.get('status') == 'BLOCKED']
    
    if not blocked:
        print("\n✅ No blocked tasks! All clear.")
        return
    
    print(f"\n🚫 BLOCKED Tasks ({len(blocked)} total):\n")
    for t in blocked:
        print(f"  ID {t['id']}: {t['title']}")
        print(f"     Priority: {t.get('priority', 'N/A')}")
        if t.get('feedback'):
            print(f"     Feedback: {t['feedback'][:150]}..." if len(t.get('feedback', '')) > 150 else f"     Feedback: {t.get('feedback')}")
        print(f"     💡 To unblock: python3 add_task.py --unblock {t['id']}")
        print()


def unblock_task(task_id: int) -> None:
    """Reset a BLOCKED task to TODO status for retry."""
    prd = load_prd()
    
    for task in prd.get('user_stories', []):
        if task.get('id') == task_id:
            if task.get('status') != 'BLOCKED':
                print(f"⚠️  Task {task_id} is not BLOCKED (status: {task.get('status')})")
                return
            
            task['status'] = 'TODO'
            old_feedback = task.get('feedback', '')
            task['feedback'] = f"[UNBLOCKED] Previous: {old_feedback}"
            save_prd(prd)
            print(f"✅ Task {task_id} unblocked! Status: TODO")
            print(f"   Ralph will retry this task on next cycle.")
            return
    
    print(f"❌ Task {task_id} not found.")


def generate_mvac(task_type: str, name: str, custom_mvac: list = None) -> list:
    """Generate MVAC criteria based on task type."""
    if task_type == "custom" or custom_mvac:
        return custom_mvac or []
    
    template = MVAC_TEMPLATES.get(task_type, [])
    # Replace {name} placeholder with actual name (lowercase, underscored)
    safe_name = name.lower().replace(' ', '_').replace('-', '_')
    return [criterion.format(name=safe_name) for criterion in template]


def create_task(
    title: str,
    priority: str = "medium",
    task_type: str = "custom",
    description: str = None,
    mvac: list = None,
    depends_on: list = None
) -> dict:
    """Create a new task dictionary."""
    prd = load_prd()
    
    # Validate
    if check_duplicate_title(prd, title):
        print(f"⚠️  Warning: Task with title '{title}' already exists!")
        confirm = input("Continue anyway? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            sys.exit(0)
    
    if priority not in VALID_PRIORITIES:
        print(f"❌ Invalid priority '{priority}'. Must be one of: {VALID_PRIORITIES}")
        sys.exit(1)
    
    # Generate task
    task_id = get_next_id(prd)
    
    # Extract name from title for MVAC generation
    name = title.split(':')[-1].strip() if ':' in title else title
    name = name.replace('Test ', '').replace('New: ', '').strip()
    
    task = {
        "id": task_id,
        "title": title,
        "priority": priority,
        "status": "TODO",
        "description": description or f"Implement: {title}",
        "acceptance_criteria": generate_mvac(task_type, name, mvac),
        "created_at": datetime.now().isoformat()
    }
    
    if depends_on:
        task["depends_on"] = depends_on
    
    return task, prd


def add_task_to_prd(task: dict, prd: dict) -> None:
    """Add task to PRD and save."""
    prd['user_stories'].append(task)
    save_prd(prd)
    
    print(f"\n📋 Task Created:")
    print(f"   ID: {task['id']}")
    print(f"   Title: {task['title']}")
    print(f"   Priority: {task['priority']}")
    print(f"   Status: {task['status']}")
    print(f"   MVAC Criteria: {len(task['acceptance_criteria'])}")
    for i, crit in enumerate(task['acceptance_criteria'], 1):
        print(f"      {i}. {crit}")


def interactive_mode():
    """Interactive task creation wizard."""
    print("\n🧙 Ralph Task Creator - Interactive Mode\n")
    
    # Title
    title = input("📝 Task Title: ").strip()
    if not title:
        print("❌ Title is required.")
        sys.exit(1)
    
    # Type
    print("\n📦 Task Types: test, model, script, mining, docs, integration, custom")
    task_type = input("   Select type [custom]: ").strip().lower() or "custom"
    
    # Priority
    print("\n⚡ Priority: high, medium, low")
    priority = input("   Select priority [medium]: ").strip().lower() or "medium"
    
    # Description
    description = input("\n📖 Description (optional): ").strip() or None
    
    # Custom MVAC
    mvac = []
    if task_type == "custom":
        print("\n🎯 Enter MVAC criteria (empty line to finish):")
        print("   Prefix with @file:, @functional:, @metric:, or @integration:")
        while True:
            criterion = input("   > ").strip()
            if not criterion:
                break
            mvac.append(criterion)
    
    # Dependencies
    deps_input = input("\n🔗 Depends on (comma-separated IDs, or empty): ").strip()
    depends_on = [int(x.strip()) for x in deps_input.split(',') if x.strip().isdigit()] if deps_input else None
    
    # Confirm
    print("\n" + "="*50)
    task, prd = create_task(title, priority, task_type, description, mvac, depends_on)
    print(f"\n📋 Preview:")
    print(json.dumps(task, indent=2, ensure_ascii=False))
    
    confirm = input("\n✅ Add this task? [Y/n]: ").strip().lower()
    if confirm in ('', 'y', 'yes'):
        add_task_to_prd(task, prd)
    else:
        print("Aborted.")


def main():
    parser = argparse.ArgumentParser(
        description="Add tasks to Ralph's prd.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test task
  python add_task.py --title "Test NewForecaster" --type test --priority high
  
  # Model implementation
  python add_task.py --title "New: LSTM Model" --type model --priority medium
  
  # Data mining task  
  python add_task.py --title "Mining: Extract GDP Data" --type mining --priority high
  
  # Custom with manual MVAC
  python add_task.py --title "Custom Task" --mvac "@file: foo.py exists" --mvac "@functional: runs"
  
  # Interactive wizard
  python add_task.py --interactive
        """
    )
    
    parser.add_argument('--title', '-t', type=str, help='Task title (required unless --interactive)')
    parser.add_argument('--type', '-T', type=str, default='custom',
                        choices=list(MVAC_TEMPLATES.keys()),
                        help='Task type for MVAC template (default: custom)')
    parser.add_argument('--priority', '-p', type=str, default='medium',
                        choices=VALID_PRIORITIES,
                        help='Priority level (default: medium)')
    parser.add_argument('--description', '-d', type=str, help='Task description')
    parser.add_argument('--mvac', '-m', action='append', type=str,
                        help='Custom MVAC criterion (can be repeated)')
    parser.add_argument('--depends-on', type=int, nargs='+',
                        help='Task IDs this depends on')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode with prompts')
    parser.add_argument('--list-types', action='store_true',
                        help='List available task types and their MVAC templates')
    parser.add_argument('--blocked', '-b', action='store_true',
                        help='Show all BLOCKED tasks that need manual intervention')
    parser.add_argument('--unblock', '-u', type=int, metavar='ID',
                        help='Unblock a task (reset to TODO) by ID')
    
    args = parser.parse_args()
    
    # List types mode
    if args.list_types:
        print("\n📦 Available Task Types:\n")
        for task_type, template in MVAC_TEMPLATES.items():
            print(f"  {task_type}:")
            for crit in template:
                print(f"    - {crit}")
            print()
        return
    
    # Show blocked tasks
    if args.blocked:
        show_blocked_tasks()
        return
    
    # Unblock a task
    if args.unblock:
        unblock_task(args.unblock)
        return
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # CLI mode
    if not args.title:
        parser.error("--title is required (or use --interactive)")
    
    task, prd = create_task(
        title=args.title,
        priority=args.priority,
        task_type=args.type,
        description=args.description,
        mvac=args.mvac,
        depends_on=args.depends_on
    )
    
    add_task_to_prd(task, prd)


if __name__ == "__main__":
    main()
