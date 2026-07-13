#!/usr/bin/env python3
"""
Ralph Universal - Metrics Dashboard

Collects and displays project metrics from PRD and progress files.
Usage: python3 edge_lab/scripts/metrics_dashboard.py
"""

import json
import re
from datetime import datetime
from pathlib import Path
from collections import Counter

# Paths
BASE_DIR = Path(__file__).parent.parent
PRD_PATH = BASE_DIR / "tasks" / "prd.json"
PROGRESS_PATH = BASE_DIR / "tasks" / "progress.txt"
OUTPUT_DIR = BASE_DIR / "data"


def load_prd() -> dict:
    """Load PRD file."""
    if not PRD_PATH.exists():
        return {"user_stories": []}
    with open(PRD_PATH, "r") as f:
        return json.load(f)


def load_progress() -> list:
    """Load progress log and parse entries."""
    if not PROGRESS_PATH.exists():
        return []

    entries = []
    with open(PROGRESS_PATH, "r") as f:
        for line in f:
            # Parse format: [TIMESTAMP] [AGENT] Message
            match = re.match(r'\[([^\]]+)\]\s*\[([^\]]+)\]\s*(.*)', line.strip())
            if match:
                entries.append({
                    "timestamp": match.group(1),
                    "agent": match.group(2),
                    "message": match.group(3)
                })
    return entries


def collect_metrics(prd: dict, progress: list) -> dict:
    """Collect project metrics from PRD and progress."""
    tasks = prd.get("user_stories", [])
    total = len(tasks)

    if total == 0:
        return {"error": "No tasks in PRD"}

    # Task status counts
    status_counts = Counter(t.get("status", "TODO") for t in tasks)
    done = status_counts.get("DONE", 0)
    pending = status_counts.get("PENDING_REVIEW", 0)
    todo = status_counts.get("TODO", 0)

    # Rejection analysis
    rejected_tasks = [t for t in tasks if t.get("feedback", "").startswith("Reject")]
    rejection_reasons = [t.get("feedback", "") for t in rejected_tasks]

    # Acceptance criteria quality
    criteria_counts = []
    weak_criteria_tasks = []
    for t in tasks:
        criteria = t.get("acceptance_criteria", [])
        count = len(criteria)
        criteria_counts.append(count)
        if count < 3:
            weak_criteria_tasks.append(t.get("id", "unknown"))

    avg_criteria = sum(criteria_counts) / len(criteria_counts) if criteria_counts else 0

    # Progress analysis
    agent_activity = Counter(e["agent"] for e in progress)

    # Task completion timeline
    completion_events = [e for e in progress if "DONE" in e.get("message", "") or "Approved" in e.get("message", "")]

    return {
        "summary": {
            "total_tasks": total,
            "completed": done,
            "pending_review": pending,
            "todo": todo,
            "completion_rate": round(done / total * 100, 1),
            "pending_rate": round(pending / total * 100, 1),
        },
        "rejection_analysis": {
            "total_rejections": len(rejected_tasks),
            "rejection_rate": round(len(rejected_tasks) / total * 100, 1) if total > 0 else 0,
            "recent_reasons": rejection_reasons[-5:],
        },
        "criteria_quality": {
            "avg_criteria_per_task": round(avg_criteria, 2),
            "tasks_with_weak_criteria": weak_criteria_tasks,
            "weak_criteria_count": len(weak_criteria_tasks),
        },
        "agent_activity": dict(agent_activity),
        "recent_completions": len(completion_events),
        "last_updated": datetime.now().isoformat()
    }


def print_dashboard(metrics: dict):
    """Print metrics in a readable format."""
    print("\n" + "=" * 60)
    print("           RALPH UNIVERSAL - METRICS DASHBOARD")
    print("=" * 60)

    s = metrics.get("summary", {})
    print(f"\n### Task Summary")
    print(f"  Total Tasks:      {s.get('total_tasks', 0)}")
    print(f"  Completed:        {s.get('completed', 0)} ({s.get('completion_rate', 0)}%)")
    print(f"  Pending Review:   {s.get('pending_review', 0)} ({s.get('pending_rate', 0)}%)")
    print(f"  TODO:             {s.get('todo', 0)}")

    r = metrics.get("rejection_analysis", {})
    print(f"\n### Rejection Analysis")
    print(f"  Total Rejections: {r.get('total_rejections', 0)}")
    print(f"  Rejection Rate:   {r.get('rejection_rate', 0)}%")
    if r.get("recent_reasons"):
        print(f"  Recent Reasons:")
        for reason in r.get("recent_reasons", []):
            print(f"    - {reason[:60]}...")

    c = metrics.get("criteria_quality", {})
    print(f"\n### Acceptance Criteria Quality")
    print(f"  Avg Criteria/Task:    {c.get('avg_criteria_per_task', 0)}")
    print(f"  Tasks < 3 Criteria:   {c.get('weak_criteria_count', 0)}")
    if c.get("tasks_with_weak_criteria"):
        print(f"  Weak Tasks: {c.get('tasks_with_weak_criteria', [])[:5]}")

    a = metrics.get("agent_activity", {})
    print(f"\n### Agent Activity")
    for agent, count in a.items():
        print(f"  {agent}: {count} actions")

    print(f"\n### Status")
    print(f"  Last Updated: {metrics.get('last_updated', 'N/A')}")

    # Health indicators
    print(f"\n### Health Indicators")
    completion_rate = s.get('completion_rate', 0)
    rejection_rate = r.get('rejection_rate', 0)
    avg_criteria = c.get('avg_criteria_per_task', 0)

    print(f"  Completion Rate: {'OK' if completion_rate >= 80 else 'NEEDS ATTENTION'} ({completion_rate}% vs 80% target)")
    print(f"  Rejection Rate:  {'OK' if rejection_rate <= 20 else 'NEEDS ATTENTION'} ({rejection_rate}% vs 20% max)")
    print(f"  Criteria Quality: {'OK' if avg_criteria >= 3 else 'NEEDS ATTENTION'} ({avg_criteria:.1f} vs 3.0 min)")

    print("\n" + "=" * 60)


def save_metrics(metrics: dict):
    """Save metrics to JSON file."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "metrics_report.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {output_path}")


def main():
    """Main entry point."""
    prd = load_prd()
    progress = load_progress()
    metrics = collect_metrics(prd, progress)

    print_dashboard(metrics)
    save_metrics(metrics)

    # Return non-zero if health indicators are bad
    s = metrics.get("summary", {})
    if s.get("completion_rate", 0) < 50:
        print("\nWARNING: Completion rate below 50%!")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
