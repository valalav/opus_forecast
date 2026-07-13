#!/usr/bin/env python3
"""
Backup System for SIRENA-KBR Forecasting Project.

Creates compressed backups of critical directories (data/ and sirena/models/)
with automatic retention policy (keep last 5 backups).

Usage:
    python3 scripts/backup_system.py              # Create backup
    python3 scripts/backup_system.py --dry-run     # Preview without creating
    python3 scripts/backup_system.py --verbose     # Detailed output
"""

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

# Configuration
PROJECT_ROOT = Path("/home/valalav/_projects/sirena-kbr")
BACKUP_DIR = PROJECT_ROOT / "archive" / "backups"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "sirena" / "models"
MAX_BACKUPS = 5

# Patterns to exclude from backups (cache, temp files, etc.)
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "*.tmp",
    ".DS_Store",
]


def create_backup_directory() -> Path:
    """Create backup directory if it doesn't exist."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    return BACKUP_DIR


def get_backup_filename() -> str:
    """Generate timestamped backup filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"backup_{timestamp}.tar.gz"


def get_existing_backups() -> list[Path]:
    """Get list of existing backup files sorted by modification time (oldest first)."""
    if not BACKUP_DIR.exists():
        return []

    backups = sorted(
        [f for f in BACKUP_DIR.glob("backup_*.tar.gz")], key=lambda f: f.stat().st_mtime
    )
    return backups


def apply_retention_policy(verbose: bool = False) -> list[Path]:
    """Remove old backups, keeping only the most recent MAX_BACKUPS.

    Returns list of deleted files.
    """
    backups = get_existing_backups()
    deleted = []

    if len(backups) > MAX_BACKUPS:
        to_delete = backups[:-MAX_BACKUPS]
        for backup in to_delete:
            try:
                backup.unlink()
                deleted.append(backup)
                if verbose:
                    print(f"  Deleted old backup: {backup.name}")
            except Exception as e:
                print(f"  Error deleting {backup.name}: {e}", file=sys.stderr)

    return deleted


def create_backup(dry_run: bool = False, verbose: bool = False) -> dict:
    """Create compressed backup of data/ and sirena/models/.

    Returns dictionary with backup details.
    """
    # Ensure backup directory exists
    backup_dir = create_backup_directory()
    backup_path = backup_dir / get_backup_filename()

    # Verify source directories exist
    sources = []
    for source, label in [(DATA_DIR, "data"), (MODELS_DIR, "models")]:
        if source.exists():
            sources.append(source)
            if verbose:
                size_mb = sum(
                    f.stat().st_size for f in source.rglob("*") if f.is_file()
                ) / (1024 * 1024)
                print(f"  Found {label}/: {size_mb:.1f} MB")
        else:
            print(f"  Warning: {source} does not exist, skipping", file=sys.stderr)

    if not sources:
        print("Error: No source directories found to backup", file=sys.stderr)
        return {"success": False, "error": "No sources found"}

    # Create tar.gz archive
    if dry_run:
        print(f"\n[DRY RUN] Would create: {backup_path}")
        size_estimate = sum(
            sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
            for src in sources
        ) / (1024 * 1024 * 1024)
        print(f"[DRY RUN] Estimated size: {size_estimate:.2f} GB (compressed)")
        return {"success": True, "path": str(backup_path), "dry_run": True}

    try:
        print(f"\nCreating backup: {backup_path.name}")
        start_time = datetime.now()

        with tarfile.open(backup_path, "w:gz") as tar:
            for source in sources:
                if verbose:
                    print(f"  Adding {source.name}/...")

                for item in source.rglob("*"):
                    # Skip excluded patterns
                    if any(pattern in str(item) for pattern in EXCLUDE_PATTERNS):
                        continue

                    if item.is_file():
                        arcname = item.relative_to(PROJECT_ROOT)
                        tar.add(item, arcname=arcname)

        duration = (datetime.now() - start_time).total_seconds()
        backup_size_mb = backup_path.stat().st_size / (1024 * 1024)

        print(f"\n  Backup created: {backup_size_mb:.1f} MB in {duration:.1f}s")

        return {
            "success": True,
            "path": str(backup_path),
            "size_mb": backup_size_mb,
            "duration_seconds": duration,
            "sources_count": len(sources),
        }

    except Exception as e:
        print(f"Error creating backup: {e}", file=sys.stderr)
        if backup_path.exists():
            backup_path.unlink()
        return {"success": False, "error": str(e)}


def list_backups(verbose: bool = False) -> list[dict]:
    """List all existing backups with metadata."""
    backups = get_existing_backups()

    if not backups:
        print("No backups found")
        return []

    print(f"\nFound {len(backups)} backup(s):")
    result = []

    for backup in backups:
        stat = backup.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        backup_info = {
            "name": backup.name,
            "path": str(backup),
            "size_mb": size_mb,
            "modified": mtime,
        }
        result.append(backup_info)

        print(f"  {backup.name:30} {size_mb:8.1f} MB  {mtime}")

    if verbose:
        total_mb = sum(b["size_mb"] for b in result)
        print(f"\n  Total: {total_mb:.1f} MB")

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backup System for SIRENA-KBR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/backup_system.py              # Create backup
  python3 scripts/backup_system.py --list       # List existing backups
  python3 scripts/backup_system.py --dry-run     # Preview without creating
  python3 scripts/backup_system.py --verbose     # Detailed output
        """,
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview backup without creating archive"
    )
    parser.add_argument("--list", action="store_true", help="List existing backups")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=" * 60)
    print("SIRENA-KBR Backup System")
    print("=" * 60)

    # List mode
    if args.list:
        list_backups(verbose=args.verbose)
        return 0

    # Backup mode
    print(f"\nBackup configuration:")
    print(f"  Source directories: data/, sirena/models/")
    print(f"  Target directory: {BACKUP_DIR}")
    print(f"  Retention policy: keep last {MAX_BACKUPS} backups")

    # List current backups
    existing = get_existing_backups()
    if existing:
        print(f"\nExisting backups: {len(existing)}")

    # Create backup
    result = create_backup(dry_run=args.dry_run, verbose=args.verbose)

    if not result["success"]:
        print(
            f"\nBackup failed: {result.get('error', 'Unknown error')}", file=sys.stderr
        )
        return 1

    if not args.dry_run:
        # Apply retention policy
        print("\nApplying retention policy...")
        deleted = apply_retention_policy(verbose=args.verbose)

        if deleted:
            print(f"  Removed {len(deleted)} old backup(s)")
        else:
            print("  No old backups to remove")

    # Final status
    print("\n" + "=" * 60)
    print("Backup completed successfully!" if result["success"] else "Backup failed")
    print("=" * 60)

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
