"""
Alert history management for volatility monitoring.

Stores and retrieves historical anomaly alerts for dashboard visualization.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd


ALERT_HISTORY_FILE = "data/alert_history.json"


def save_alerts_to_history(alerts: list) -> None:
    """Save current alerts to history file."""
    history_dir = Path("data")
    history_dir.mkdir(exist_ok=True)

    # Load existing history
    history = load_alert_history()

    # Add timestamp to alerts
    timestamp = datetime.now().isoformat()
    for alert in alerts:
        alert["detected_at"] = timestamp

    # Append new alerts
    history.extend(alerts)

    # Keep only last 100 alerts
    history = history[-100:]

    # Save
    with open(ALERT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_alert_history() -> list:
    """Load alert history from file."""
    if not os.path.exists(ALERT_HISTORY_FILE):
        return []

    with open(ALERT_HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_alert_history_dataframe() -> pd.DataFrame:
    """Convert alert history to DataFrame for visualization."""
    history = load_alert_history()
    if not history:
        return pd.DataFrame()

    df = pd.DataFrame(history)
    if not df.empty:
        df["detected_at"] = pd.to_datetime(df["detected_at"])
        df = df.sort_values("detected_at", ascending=False)

    return df


def get_alert_statistics(history: list) -> dict:
    """Calculate statistics for alert history."""
    if not history:
        return {"total": 0, "critical": 0, "warning": 0, "products": {}}

    stats = {
        "total": len(history),
        "critical": sum(1 for a in history if a["level"] == "critical"),
        "warning": sum(1 for a in history if a["level"] == "warning"),
        "products": {},
    }

    # Count alerts per product
    for alert in history:
        product = alert["product_name"]
        stats["products"][product] = stats["products"].get(product, 0) + 1

    return stats
