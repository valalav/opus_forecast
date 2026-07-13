#!/usr/bin/env python3
"""
Disk Usage Monitor for Opus Forecast.

Monitors growth of target folders (default: archive/) with configurable threshold.
Logs size trends to CSV and alerts if threshold exceeded.

Usage:
    python3 scripts/monitor_disk.py                    # Check archive/
    python3 scripts/monitor_disk.py --folder data/       # Check data/
    python3 scripts/monitor_disk.py --threshold 2GB        # 2GB threshold
    python3 scripts/monitor_disk.py --history              # Show trend history
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Configuration
PROJECT_ROOT = Path("/home/valalav/_projects/sirena-kbr")
DEFAULT_FOLDER = PROJECT_ROOT / "archive"
DEFAULT_THRESHOLD_GB = 1.0
LOG_FILE = PROJECT_ROOT / "data" / "disk_usage_log.csv"


def get_folder_size(folder: Path) -> int:
    """Calculate total size of folder in bytes recursively."""
    if not folder.exists():
        print(f"Error: Folder '{folder}' does not exist", file=sys.stderr)
        return 0

    total_bytes = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = Path(root) / file
            try:
                total_bytes += file_path.stat().st_size
            except (OSError, FileNotFoundError):
                continue

    return total_bytes


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string (KB, MB, GB, TB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def parse_threshold(threshold_str: str) -> int:
    """Parse threshold string (e.g., '1GB', '500MB') to bytes."""
    threshold_str = threshold_str.upper().strip()

    if threshold_str.isdigit():
        return int(threshold_str)

    # Parse unit suffix
    units = {"TB": 1024**4, "GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}

    # Sort by length (longest first) to avoid partial matches
    for unit, multiplier in sorted(units.items(), key=lambda x: -len(x[0])):
        if threshold_str.endswith(unit):
            value_str = threshold_str[: -len(unit)]
            try:
                value = float(value_str)
                bytes_val = value * multiplier
                return int(bytes_val)
            except ValueError:
                continue

    # Default to GB if no unit specified
    try:
        value = float(threshold_str)
        bytes_val = value * units["GB"]
        return int(bytes_val)
    except ValueError:
        raise ValueError(f"Invalid threshold format: {threshold_str}")


def load_history() -> list[dict]:
    """Load disk usage history from log file."""
    if not LOG_FILE.exists():
        return []

    history = []
    with open(LOG_FILE, "r") as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                history.append(
                    {
                        "timestamp": parts[0],
                        "folder": parts[1],
                        "size_bytes": int(parts[2]),
                    }
                )
    return history


def log_entry(folder: Path, size_bytes: int):
    """Append new measurement to log file."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    # Create file with header if it doesn't exist
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,folder,size_bytes\n")

    # Append new entry
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp},{folder},{size_bytes}\n")


def show_history(folder: Path, limit: int = 10):
    """Display recent disk usage history for a folder."""
    history = load_history()

    # Filter by folder
    folder_history = [h for h in history if h["folder"] == str(folder)]

    if not folder_history:
        print(f"No history found for folder: {folder}")
        return

    # Show last N entries
    recent = folder_history[-limit:]

    print(f"\nDisk Usage History for {folder} (last {len(recent)} entries):")
    print(f"{'Timestamp':<25} {'Size':<15} {'Change':<15}")
    print("-" * 55)

    prev_size = None
    for entry in recent:
        size_str = format_size(entry["size_bytes"])
        change_str = ""

        if prev_size is not None:
            diff_bytes = entry["size_bytes"] - prev_size
            diff_pct = (diff_bytes / prev_size) * 100
            change_str = f"{format_size(diff_bytes)} ({diff_pct:+.1f}%)"

        print(f"{entry['timestamp']:<25} {size_str:<15} {change_str:<15}")
        prev_size = entry["size_bytes"]

    # Show growth trend
    if len(recent) >= 2:
        first_size = recent[0]["size_bytes"]
        last_size = recent[-1]["size_bytes"]
        total_growth = last_size - first_size
        total_growth_pct = (total_growth / first_size) * 100
        print(f"\nTotal growth: {format_size(total_growth)} ({total_growth_pct:+.1f}%)")


def calculate_growth_trend(folder: Path) -> dict:
    """Calculate growth metrics from history."""
    history = [h for h in load_history() if h["folder"] == str(folder)]

    if len(history) < 2:
        return {"avg_daily_growth": 0, "days_tracked": len(history), "total_growth": 0}

    # Sort by timestamp
    history = sorted(history, key=lambda x: x["timestamp"])

    # Calculate growth between first and last
    first = history[0]
    last = history[-1]

    first_time = datetime.fromisoformat(first["timestamp"])
    last_time = datetime.fromisoformat(last["timestamp"])

    days = (last_time - first_time).days
    if days == 0:
        days = 1

    growth_bytes = last["size_bytes"] - first["size_bytes"]
    avg_daily_growth = growth_bytes / days

    return {
        "avg_daily_growth": avg_daily_growth,
        "days_tracked": days,
        "total_growth": growth_bytes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Disk Usage Monitor for Opus Forecast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/monitor_disk.py                    # Check archive/
  python3 scripts/monitor_disk.py --folder data/       # Check data/
  python3 scripts/monitor_disk.py --threshold 2GB        # 2GB threshold
  python3 scripts/monitor_disk.py --history              # Show trend history
        """,
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=str(DEFAULT_FOLDER),
        help="Folder to monitor (default: archive/)",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default=f"{DEFAULT_THRESHOLD_GB}GB",
        help="Size threshold for alert (e.g., 1GB, 500MB, default: 1GB)",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show historical trends instead of current size",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip logging to CSV (dry run)",
    )

    args = parser.parse_args()

    folder = Path(args.folder).absolute()
    threshold_bytes = parse_threshold(args.threshold)

    # History mode
    if args.history:
        show_history(folder)
        return 0

    # Check current size
    size_bytes = get_folder_size(folder)

    if size_bytes == 0 and not folder.exists():
        return 1

    # Get growth metrics
    trend = calculate_growth_trend(folder)

    # Display results
    print("=" * 60)
    print("Disk Usage Monitor")
    print("=" * 60)
    print(f"Folder:       {folder}")
    print(f"Current Size:  {format_size(size_bytes)}")
    print(f"Threshold:    {format_size(threshold_bytes)}")

    if trend["days_tracked"] > 0:
        print(f"Trend:")
        print(f"  Tracked for: {trend['days_tracked']} days")
        print(f"  Total growth: {format_size(trend['total_growth'])}")
        if trend["avg_daily_growth"] != 0:
            print(
                f"  Avg daily:    {format_size(abs(trend['avg_daily_growth']))}/day {'(growing)' if trend['avg_daily_growth'] > 0 else '(shrinking)'}"
            )

    # Check threshold
    usage_pct = (size_bytes / threshold_bytes) * 100
    print(f"\nUsage: {usage_pct:.1f}% of threshold")

    if size_bytes > threshold_bytes:
        print("\n" + "=" * 60)
        print("⚠️  ALERT: Folder exceeds threshold!")
        print("=" * 60)

        # Log this alert event
        if not args.no_log:
            log_entry(folder, size_bytes)

        return 1
    else:
        remaining_bytes = threshold_bytes - size_bytes
        print(f"✓ OK: {format_size(remaining_bytes)} remaining before threshold")

        # Log normal event
        if not args.no_log:
            log_entry(folder, size_bytes)

        return 0


if __name__ == "__main__":
    sys.exit(main())
