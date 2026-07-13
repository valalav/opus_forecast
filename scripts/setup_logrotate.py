#!/usr/bin/env python3
"""
Log Rotation Setup for Ralph/Opus System

Rotates log files when they exceed size limits.
Maintains a configurable number of backup files.

Usage:
    python3 scripts/setup_logrotate.py --rotate           # Perform rotation
    python3 scripts/setup_logrotate.py --verify          # Verify configuration
    python3 scripts/setup_logrotate.py --status          # Check current status
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional


# Configuration
LOG_ROTATION_CONFIG = {
    "edge_lab/tasks/progress.txt": {
        "max_size_mb": 10,
        "keep_count": 5,
        "backup_dir": "edge_lab/tasks/logs",
    },
    "edge_lab/tasks/prd.json": {
        "max_size_mb": 50,
        "keep_count": 3,
        "backup_dir": "edge_lab/tasks/logs",
    },
}

# Path to project root (assuming script is in scripts/)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


class LogRotator:
    """Handles log file rotation with configurable retention."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or LOG_ROTATION_CONFIG
        self.base_dir = Path.cwd()

    def rotate_file(self, log_path: str) -> Tuple[bool, str]:
        """
        Rotate a single log file if it exceeds size limit.

        Returns:
            (success: bool, message: str)
        """
        log_file = self.base_dir / log_path

        if not log_file.exists():
            return False, f"File not found: {log_path}"

        # Get file info
        file_size = log_file.stat().st_size
        file_config = self.config.get(log_path, {})

        max_size_bytes = file_config.get("max_size_mb", 10) * 1024 * 1024
        keep_count = file_config.get("keep_count", 5)
        backup_dir = self.base_dir / file_config.get("backup_dir", "logs")

        # Check if rotation is needed
        if file_size < max_size_bytes:
            return (
                False,
                f"File size {file_size / 1024 / 1024:.2f}MB < {max_size_bytes / 1024 / 1024:.2f}MB",
            )

        # Create backup directory if needed
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Rotate existing backups (progress.txt.1 -> progress.txt.2, etc.)
        base_name = log_file.stem
        extension = log_file.suffix

        # Shift existing backups
        for i in range(keep_count - 1, 0, -1):
            old_backup = backup_dir / f"{base_name}.{i}{extension}"
            new_backup = backup_dir / f"{base_name}.{i + 1}{extension}"

            if old_backup.exists():
                if i == keep_count - 1:
                    # Delete oldest backup if it would exceed keep_count
                    old_backup.unlink()
                else:
                    # Shift backup
                    shutil.move(str(old_backup), str(new_backup))

        # Move current log to .1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_backup = backup_dir / f"{base_name}.1{extension}"
        shutil.move(str(log_file), str(new_backup))

        # Create new empty log file
        log_file.touch()
        log_file.chmod(0o664)  # rw-rw-r--

        return (
            True,
            f"Rotated {log_path} ({file_size / 1024 / 1024:.2f}MB -> {new_backup})",
        )

    def rotate_all(self) -> Dict[str, Tuple[bool, str]]:
        """
        Rotate all configured log files.

        Returns:
            Dictionary mapping log paths to (success, message) tuples
        """
        results = {}
        for log_path in self.config.keys():
            results[log_path] = self.rotate_file(log_path)
        return results

    def verify_config(self) -> Tuple[bool, List[str]]:
        """
        Verify configuration is valid.

        Returns:
            (is_valid: bool, messages: List[str])
        """
        messages = []
        is_valid = True

        for log_path, config in self.config.items():
            log_file = self.base_dir / log_path

            # Check if parent directory exists
            if not log_file.parent.exists():
                messages.append(
                    f"ERROR: Parent directory does not exist: {log_file.parent}"
                )
                is_valid = False
                continue

            # Check backup directory path
            backup_dir = self.base_dir / config.get("backup_dir", "logs")
            backup_dir.mkdir(parents=True, exist_ok=True)
            messages.append(f"✓ Backup directory OK: {backup_dir}")

            # Check config values
            max_size = config.get("max_size_mb", 10)
            keep_count = config.get("keep_count", 5)

            if not isinstance(max_size, (int, float)) or max_size <= 0:
                messages.append(
                    f"ERROR: Invalid max_size_mb for {log_path}: {max_size}"
                )
                is_valid = False
            else:
                messages.append(f"✓ Max size OK: {log_path} ({max_size}MB)")

            if not isinstance(keep_count, int) or keep_count < 1:
                messages.append(
                    f"ERROR: Invalid keep_count for {log_path}: {keep_count}"
                )
                is_valid = False
            else:
                messages.append(f"✓ Keep count OK: {log_path} ({keep_count})")

        return is_valid, messages

    def get_status(self) -> Dict[str, Dict]:
        """
        Get current status of all monitored log files.

        Returns:
            Dictionary mapping log paths to status info
        """
        status = {}
        for log_path, config in self.config.items():
            log_file = self.base_dir / log_path
            backup_dir = self.base_dir / config.get("backup_dir", "logs")
            base_name = log_file.stem
            extension = log_file.suffix

            if log_file.exists():
                file_size = log_file.stat().st_size
                file_size_mb = file_size / 1024 / 1024
                max_size_mb = config.get("max_size_mb", 10)
                rotation_needed = file_size_mb > max_size_mb

                # Count existing backups
                backups = []
                for i in range(1, config.get("keep_count", 5) + 1):
                    backup_file = backup_dir / f"{base_name}.{i}{extension}"
                    if backup_file.exists():
                        stat = backup_file.stat()
                        backups.append(
                            {
                                "number": i,
                                "size_mb": stat.st_size / 1024 / 1024,
                                "modified": datetime.fromtimestamp(stat.st_mtime),
                            }
                        )

                status[log_path] = {
                    "exists": True,
                    "size_mb": round(file_size_mb, 2),
                    "max_size_mb": max_size_mb,
                    "rotation_needed": rotation_needed,
                    "backups": backups,
                    "backup_count": len(backups),
                }
            else:
                status[log_path] = {
                    "exists": False,
                    "size_mb": 0,
                    "rotation_needed": False,
                    "backups": [],
                }

        return status


def print_status(status: Dict):
    """Print formatted status report."""
    print("\n" + "=" * 80)
    print("LOG ROTATION STATUS")
    print("=" * 80)

    for log_path, info in status.items():
        print(f"\n📄 {log_path}")
        if not info["exists"]:
            print("  Status: ❌ File does not exist")
            continue

        print(f"  Current Size: {info['size_mb']:.2f} MB / {info['max_size_mb']} MB")

        if info["rotation_needed"]:
            print(f"  Rotation: ⚠️  NEEDED (exceeds limit)")
        else:
            print(f"  Rotation: ✅ Not needed")

        if info["backups"]:
            print(f"  Backups ({info['backup_count']}):")
            for backup in info["backups"]:
                print(
                    f"    .{backup['number']}: {backup['size_mb']:.2f} MB "
                    f"(modified: {backup['modified'].strftime('%Y-%m-%d %H:%M:%S')})"
                )
        else:
            print("  Backups: None")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Log rotation setup for Ralph/Opus system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/setup_logrotate.py --rotate
  python3 scripts/setup_logrotate.py --verify
  python3 scripts/setup_logrotate.py --status
        """,
    )

    parser.add_argument(
        "--rotate",
        "-r",
        action="store_true",
        help="Perform log rotation for all configured files",
    )

    parser.add_argument(
        "--verify", "-v", action="store_true", help="Verify log rotation configuration"
    )

    parser.add_argument(
        "--status", "-s", action="store_true", help="Show current status of log files"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually rotating",
    )

    args = parser.parse_args()

    if not any([args.rotate, args.verify, args.status]):
        parser.print_help()
        sys.exit(0)

    # Change to project root
    os.chdir(PROJECT_ROOT)

    # Initialize rotator
    rotator = LogRotator()

    if args.verify:
        print("\n🔍 Verifying log rotation configuration...")
        is_valid, messages = rotator.verify_config()

        for msg in messages:
            print(f"  {msg}")

        if is_valid:
            print("\n✅ Configuration is valid")
            sys.exit(0)
        else:
            print("\n❌ Configuration has errors")
            sys.exit(1)

    elif args.status:
        status = rotator.get_status()
        print_status(status)
        sys.exit(0)

    elif args.rotate:
        print("\n🔄 Checking for files that need rotation...")

        if args.dry_run:
            print("📋 DRY RUN MODE - No files will be modified\n")

        results = rotator.rotate_all()

        rotated_count = 0
        for log_path, (success, message) in results.items():
            if success:
                print(f"  ✅ {message}")
                rotated_count += 1
            else:
                print(f"  ℹ️  {message}")

        if rotated_count > 0:
            print(f"\n✅ Rotated {rotated_count} file(s)")
            if not args.dry_run:
                print("💡 Run with --status to see backup files")
        else:
            print("\n✅ No files needed rotation")

        if not args.dry_run:
            sys.exit(0)


if __name__ == "__main__":
    main()
