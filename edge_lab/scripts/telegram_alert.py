#!/usr/bin/env python3
"""
Telegram Alert Bot for Inflation Status.

Sends inflation forecast and status updates to Telegram.
"""

import os
import json
import ssl
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Optional, Dict, Any


class InflationAlertBot:
    """Send inflation status notifications to Telegram."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize InflationAlertBot.

        Args:
            token: Bot token. If None, reads from TELEGRAM_TOKEN env var.
            chat_id: Chat ID to send messages to. If None, reads from TELEGRAM_CHAT_ID.
        """
        self.token = token or os.environ.get("TELEGRAM_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

        if not self.token or not self.chat_id:
            raise ValueError(
                "TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables must be set"
            )

    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to Telegram.

        Args:
            message: The message text to send.
            parse_mode: "HTML" or "Markdown".

        Returns:
            True if sent successfully, False otherwise.
        """
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = urllib.parse.urlencode(
                {
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": parse_mode,
                }
            ).encode()

            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(url, data=data)
            response = urllib.request.urlopen(req, context=ctx, timeout=10)
            result = json.loads(response.read())

            if result.get("ok"):
                print(f"[Bot] Message sent successfully")
                return True
            else:
                print(f"[Bot] API error: {result}")
                return False

        except Exception as e:
            print(f"[Bot] Error sending message: {e}")
            return False

    def load_inflation_data(
        self, filepath: str = "data/last_inflation.json"
    ) -> Optional[Dict[str, Any]]:
        """
        Load inflation data from JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            Dict with inflation data or None if file not found.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"[Bot] Loaded inflation data from {filepath}")
                return data
        except FileNotFoundError:
            print(f"[Bot] Warning: {filepath} not found")
            return None
        except json.JSONDecodeError as e:
            print(f"[Bot] Error parsing JSON: {e}")
            return None

    def format_inflation_message(self, data: Optional[Dict[str, Any]]) -> str:
        """
        Format inflation data as Telegram message.

        Args:
            data: Inflation data dictionary.

        Returns:
            Formatted message string.
        """
        if data is None:
            return (
                "<b>⚠️ Inflation Data Not Available</b>\n\n"
                "File <code>data/last_inflation.json</code> not found or invalid.\n\n"
                "<i>Please ensure the data pipeline is running.</i>"
            )

        timestamp = data.get("timestamp", datetime.now().isoformat())
        last_value = data.get("last_value", "N/A")
        last_change = data.get("last_change", "N/A")
        forecast = data.get("forecast", "N/A")
        model = data.get("model", "Unknown")

        message = (
            f"<b>📊 Inflation Status Update</b>\n\n"
            f"<b>Last Value:</b> {last_value}%\n"
            f"<b>Change:</b> {last_change}\n"
            f"<b>Forecast:</b> {forecast}%\n"
            f"<b>Model:</b> {model}\n\n"
            f"<i>Updated: {timestamp}</i>"
        )

        return message

    def send_inflation_alert(self, filepath: str = "data/last_inflation.json") -> bool:
        """
        Send inflation status alert.

        Args:
            filepath: Path to inflation JSON file.

        Returns:
            True if sent successfully, False otherwise.
        """
        data = self.load_inflation_data(filepath)
        message = self.format_inflation_message(data)
        return self.send_message(message)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Send inflation alert to Telegram")
    parser.add_argument(
        "--data-file",
        default="data/last_inflation.json",
        help="Path to inflation data JSON file",
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="Custom message to send (overrides data file)",
    )

    args = parser.parse_args()

    try:
        bot = InflationAlertBot()

        if args.message:
            success = bot.send_message(args.message)
        else:
            success = bot.send_inflation_alert(args.data_file)

        if success:
            print("[Bot] Alert sent successfully")
            exit(0)
        else:
            print("[Bot] Failed to send alert")
            exit(1)

    except ValueError as e:
        print(f"[Bot] Configuration error: {e}")
        exit(1)
    except Exception as e:
        print(f"[Bot] Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
