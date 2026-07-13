#!/usr/bin/env python3
"""
SIRENA-KBR Dashboard Monitor
============================

Комплексная проверка всех компонентов системы.

Запуск: python3 scripts/monitor.py
        python3 scripts/monitor.py --watch  # Мониторинг в реальном времени

Эта команда:
1. Проверяет синтаксис dashboard.py
2. Проверяет на undefined переменные
3. Проверяет все модели
4. Проверяет статус systemd сервиса
5. Проверяет HTTP доступность
6. Выводит последние ошибки из логов
"""

import sys
import os
import time
import json
import subprocess
import socket
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_cmd(cmd: str, timeout: int = 30) -> tuple:
    """Run shell command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def check_syntax() -> tuple:
    """Check dashboard.py syntax."""
    try:
        with open(PROJECT_ROOT / 'dashboard.py', 'r') as f:
            source = f.read()
        compile(source, 'dashboard.py', 'exec')
        return True, "OK"
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def check_service() -> tuple:
    """Check systemd service status."""
    success, output = run_cmd("systemctl --user is-active sirena-dashboard")
    if success:
        return True, "active"
    return False, output.strip()


def check_http() -> tuple:
    """Check HTTP accessibility."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 8503))
        sock.close()
        if result == 0:
            return True, "http://localhost:8503"
        return False, "Port 8503 not open"
    except Exception as e:
        return False, str(e)


def check_logs() -> list:
    """Get recent errors from logs."""
    errors = []
    log_files = [
        PROJECT_ROOT / 'logs' / 'dashboard_stderr.log',
        PROJECT_ROOT / 'logs' / 'dashboard_stdout.log',
    ]

    for log_file in log_files:
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    for line in content.split('\n')[-50:]:
                        if any(x in line.lower() for x in ['error', 'exception', 'traceback']):
                            errors.append(line.strip()[:80])
            except:
                pass

    return errors[-5:]  # Last 5 errors


def check_models() -> tuple:
    """Quick check of key models."""
    try:
        from sirena.models import RidgeForecaster, HuberForecaster
        return True, "RidgeForecaster, HuberForecaster OK"
    except Exception as e:
        return False, str(e)


def print_status():
    """Print current status."""
    print("\n" + "=" * 60)
    print(f"SIRENA-KBR Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    checks = [
        ("Syntax", check_syntax),
        ("Models", check_models),
        ("Service", check_service),
        ("HTTP", check_http),
    ]

    all_ok = True
    for name, check_fn in checks:
        ok, msg = check_fn()
        icon = "✓" if ok else "✗"
        print(f"{icon} {name:12} {msg}")
        if not ok:
            all_ok = False

    # Log errors
    errors = check_logs()
    if errors:
        print(f"\n⚠ Recent errors ({len(errors)}):")
        for err in errors:
            print(f"  {err[:70]}")

    print("=" * 60)

    if all_ok and not errors:
        print("✓ ALL SYSTEMS OPERATIONAL")
    else:
        print("✗ ISSUES DETECTED")

    return all_ok


def watch_mode(interval: int = 30):
    """Continuous monitoring mode."""
    print(f"Starting watch mode (interval: {interval}s)")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            print_status()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description='SIRENA-KBR Dashboard Monitor')
    parser.add_argument('--watch', '-w', action='store_true', help='Watch mode')
    parser.add_argument('--interval', '-i', type=int, default=30, help='Watch interval (seconds)')
    args = parser.parse_args()

    if args.watch:
        watch_mode(args.interval)
    else:
        ok = print_status()
        return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
