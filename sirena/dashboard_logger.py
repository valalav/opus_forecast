"""
Dashboard Logging System for SIRENA-KBR
=======================================

Provides logging, debugging and status tracking for the Streamlit dashboard.

Usage in dashboard.py:
    from sirena.dashboard_logger import DashboardLogger, log_tab, log_model

    logger = DashboardLogger()

    @log_tab("Прогноз")
    def render_forecast_tab():
        ...

    @log_model("Ridge")
    def run_ridge_model():
        ...
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, Any, Callable
import threading

# Log directory
LOG_DIR = Path(__file__).parent.parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)


class DashboardLogger:
    """
    Centralized logger for SIRENA-KBR dashboard.

    Features:
    - Tab execution tracking
    - Model run logging
    - Error capture
    - Status file for external monitoring
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        # Status tracking
        self.status: Dict[str, Any] = {
            'start_time': datetime.now().isoformat(),
            'tabs': {},
            'models': {},
            'errors': [],
            'last_update': None
        }

        # File paths
        self.log_file = LOG_DIR / 'dashboard.log'
        self.status_file = LOG_DIR / 'dashboard_status.json'
        self.error_file = LOG_DIR / 'dashboard_errors.log'

        # Setup logging
        self._setup_logging()

        # Write initial status
        self._write_status()

    def _setup_logging(self):
        """Configure logging handlers."""
        # Main logger
        self.logger = logging.getLogger('sirena.dashboard')
        self.logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        # Console handler (for streamlit logs)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def _write_status(self):
        """Write current status to JSON file."""
        self.status['last_update'] = datetime.now().isoformat()
        try:
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(self.status, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to write status: {e}")

    def log_tab_start(self, tab_name: str):
        """Log tab rendering start."""
        self.status['tabs'][tab_name] = {
            'status': 'running',
            'started': datetime.now().isoformat(),
            'completed': None,
            'error': None
        }
        self._write_status()
        self.logger.info(f"TAB START: {tab_name}")

    def log_tab_end(self, tab_name: str, success: bool = True, error: str = None):
        """Log tab rendering end."""
        if tab_name in self.status['tabs']:
            self.status['tabs'][tab_name].update({
                'status': 'ok' if success else 'error',
                'completed': datetime.now().isoformat(),
                'error': error
            })
        self._write_status()

        if success:
            self.logger.info(f"TAB END: {tab_name} - OK")
        else:
            self.logger.error(f"TAB END: {tab_name} - ERROR: {error}")

    def log_model_run(self, model_name: str, success: bool,
                      prediction: float = None, duration: float = None,
                      error: str = None):
        """Log model execution."""
        self.status['models'][model_name] = {
            'status': 'ok' if success else 'error',
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'duration_sec': duration,
            'error': error
        }
        self._write_status()

        if success:
            self.logger.info(f"MODEL: {model_name} - {prediction:.4f} ({duration:.3f}s)")
        else:
            self.logger.error(f"MODEL: {model_name} - FAILED: {error}")

    def log_error(self, context: str, error: Exception):
        """Log an error with full traceback."""
        tb = traceback.format_exc()
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': tb
        }
        self.status['errors'].append(error_entry)

        # Keep only last 100 errors
        if len(self.status['errors']) > 100:
            self.status['errors'] = self.status['errors'][-100:]

        self._write_status()

        # Write to error log file
        with open(self.error_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Time: {error_entry['timestamp']}\n")
            f.write(f"Context: {context}\n")
            f.write(f"Error: {error_entry['error_type']}: {error_entry['error_message']}\n")
            f.write(f"Traceback:\n{tb}\n")

        self.logger.error(f"ERROR in {context}: {error}")

    def get_status_summary(self) -> str:
        """Get human-readable status summary."""
        lines = []
        lines.append("=" * 50)
        lines.append("SIRENA-KBR Dashboard Status")
        lines.append("=" * 50)
        lines.append(f"Started: {self.status['start_time']}")
        lines.append(f"Last Update: {self.status['last_update']}")
        lines.append("")

        # Tabs
        lines.append("TABS:")
        for tab_name, tab_info in self.status['tabs'].items():
            status_icon = "✓" if tab_info['status'] == 'ok' else "✗" if tab_info['status'] == 'error' else "..."
            lines.append(f"  {status_icon} {tab_name}: {tab_info['status']}")

        # Models
        lines.append("\nMODELS:")
        for model_name, model_info in self.status['models'].items():
            status_icon = "✓" if model_info['status'] == 'ok' else "✗"
            pred = f"{model_info['prediction']:.4f}" if model_info['prediction'] else "-"
            lines.append(f"  {status_icon} {model_name}: {pred}")

        # Errors
        error_count = len(self.status['errors'])
        lines.append(f"\nERRORS: {error_count}")
        if error_count > 0:
            for err in self.status['errors'][-3:]:
                lines.append(f"  - {err['context']}: {err['error_message'][:50]}")

        return "\n".join(lines)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)


def log_tab(tab_name: str):
    """
    Decorator to log tab execution.

    Usage:
        @log_tab("Прогноз")
        def render_forecast_tab():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = DashboardLogger()
            logger.log_tab_start(tab_name)
            try:
                result = func(*args, **kwargs)
                logger.log_tab_end(tab_name, success=True)
                return result
            except Exception as e:
                logger.log_tab_end(tab_name, success=False, error=str(e))
                logger.log_error(f"Tab: {tab_name}", e)
                raise
        return wrapper
    return decorator


def log_model(model_name: str):
    """
    Decorator to log model execution.

    Usage:
        @log_model("Ridge")
        def run_ridge():
            ...
            return prediction
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            logger = DashboardLogger()
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start

                # Try to extract prediction from result
                prediction = None
                if isinstance(result, (int, float)):
                    prediction = float(result)
                elif isinstance(result, dict) and 'prediction' in result:
                    prediction = float(result['prediction'])

                logger.log_model_run(model_name, True, prediction, duration)
                return result
            except Exception as e:
                duration = time.time() - start
                logger.log_model_run(model_name, False, error=str(e), duration=duration)
                logger.log_error(f"Model: {model_name}", e)
                raise
        return wrapper
    return decorator


# Convenience function
def get_logger() -> DashboardLogger:
    """Get the singleton dashboard logger."""
    return DashboardLogger()


# Status check script entry point
def check_status():
    """Check dashboard status from command line."""
    status_file = LOG_DIR / 'dashboard_status.json'

    if not status_file.exists():
        print("Dashboard status file not found. Dashboard may not be running.")
        return 1

    with open(status_file) as f:
        status = json.load(f)

    logger = DashboardLogger()
    logger.status = status
    print(logger.get_status_summary())
    return 0


if __name__ == '__main__':
    sys.exit(check_status())
