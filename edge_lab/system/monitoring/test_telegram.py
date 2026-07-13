#!/usr/bin/env python3
"""
Test script for Telegram notifications.

Usage:
    export RALPH_TG_BOT_TOKEN="your-token"
    export RALPH_TG_CHAT_ID="your-chat-id"
    python3 edge_lab/system/monitoring/test_telegram.py
"""
import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.telegram_bot import TelegramNotifier


def main():
    print("Testing Telegram Notifier...")
    print()

    # Check environment variables
    token = os.environ.get('RALPH_TG_BOT_TOKEN')
    chat_id = os.environ.get('RALPH_TG_CHAT_ID')

    if not token:
        print("ERROR: RALPH_TG_BOT_TOKEN not set")
        print("  Get token from @BotFather on Telegram")
        print()
        print("  export RALPH_TG_BOT_TOKEN='1234567890:ABCdef...'")
        return

    if not chat_id:
        print("ERROR: RALPH_TG_CHAT_ID not set")
        print("  Add bot to a group/channel, then get chat_id from:")
        print(f"  https://api.telegram.org/bot{token}/getUpdates")
        print()
        print("  export RALPH_TG_CHAT_ID='-1001234567890'")
        return

    print(f"Token: {token[:10]}...{token[-5:]}")
    print(f"Chat ID: {chat_id}")
    print()

    # Create notifier
    notifier = TelegramNotifier()

    if not notifier.enabled:
        print("ERROR: Notifier not enabled (check token/chat_id)")
        return

    # Send test message
    print("Sending test message...")
    success = notifier.send("<b>Test from Ralph Monitoring</b>\n\nIf you see this, notifications work!")

    if success:
        print("SUCCESS! Check your Telegram.")
    else:
        print("FAILED to send message. Check token and chat_id.")


if __name__ == "__main__":
    main()
