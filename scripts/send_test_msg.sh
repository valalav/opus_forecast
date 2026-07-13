#!/bin/bash
# Test script for Telegram Alert Bot
# Usage: ./send_test_msg.sh [custom_message]

set -e

# Default test message
DEFAULT_MESSAGE="🔔 <b>Test Message from Opus Forecast</b>\n\nThis is a test alert from the Telegram bot.\n<i>Timestamp: $(date -Iseconds)</i>"

# Check if custom message provided
if [ -n "$1" ]; then
    MESSAGE="$1"
else
    # Use default message (will trigger inflation data loading)
    MESSAGE=""
fi

# Check environment variables
if [ -z "$TELEGRAM_TOKEN" ]; then
    echo "Error: TELEGRAM_TOKEN environment variable not set"
    echo "Usage: export TELEGRAM_TOKEN=your_bot_token"
    echo "       export TELEGRAM_CHAT_ID=your_chat_id"
    echo "       ./send_test_msg.sh"
    exit 1
fi

if [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo "Error: TELEGRAM_CHAT_ID environment variable not set"
    echo "Usage: export TELEGRAM_TOKEN=your_bot_token"
    echo "       export TELEGRAM_CHAT_ID=your_chat_id"
    echo "       ./send_test_msg.sh"
    exit 1
fi

# Change to script directory
cd "$(dirname "$0")/.."

# Run Python script
if [ -n "$MESSAGE" ]; then
    python3 scripts/telegram_alert.py --message "$MESSAGE"
else
    python3 scripts/telegram_alert.py
fi

echo "Test completed"
