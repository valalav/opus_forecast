#!/bin/bash
#
# SIRENA-KBR Dashboard Startup Script
# ====================================
#
# Usage:
#   ./bin/start_dashboard.sh          # Start in foreground
#   ./bin/start_dashboard.sh daemon   # Start in background
#   ./bin/start_dashboard.sh stop     # Stop dashboard
#   ./bin/start_dashboard.sh restart  # Restart dashboard
#   ./bin/start_dashboard.sh status   # Check status
#

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DASHBOARD_PORT=8503
PID_FILE="$PROJECT_DIR/logs/dashboard.pid"
LOG_FILE="$PROJECT_DIR/logs/dashboard_stdout.log"

# Ensure logs directory exists
mkdir -p "$PROJECT_DIR/logs"

cd "$PROJECT_DIR"

start_foreground() {
    echo "Starting SIRENA-KBR Dashboard on port $DASHBOARD_PORT..."
    exec streamlit run dashboard.py \
        --server.port $DASHBOARD_PORT \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false
}

start_daemon() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Dashboard is already running (PID: $PID)"
            exit 1
        fi
    fi

    echo "Starting SIRENA-KBR Dashboard in background..."
    nohup streamlit run dashboard.py \
        --server.port $DASHBOARD_PORT \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false \
        >> "$LOG_FILE" 2>&1 &

    PID=$!
    echo $PID > "$PID_FILE"
    sleep 2

    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Dashboard started successfully (PID: $PID)"
        echo "URL: http://localhost:$DASHBOARD_PORT"
        echo "Logs: $LOG_FILE"
    else
        echo "Failed to start dashboard. Check logs: $LOG_FILE"
        exit 1
    fi
}

stop_dashboard() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Stopping dashboard (PID: $PID)..."
            kill "$PID"
            sleep 2
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Force killing..."
                kill -9 "$PID"
            fi
            rm -f "$PID_FILE"
            echo "Dashboard stopped."
        else
            echo "Dashboard is not running (stale PID file)."
            rm -f "$PID_FILE"
        fi
    else
        # Try to find by port
        PIDS=$(lsof -t -i:$DASHBOARD_PORT 2>/dev/null)
        if [ -n "$PIDS" ]; then
            echo "Stopping processes on port $DASHBOARD_PORT..."
            echo "$PIDS" | xargs kill 2>/dev/null
            sleep 2
            echo "Stopped."
        else
            echo "Dashboard is not running."
        fi
    fi
}

check_status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Dashboard is RUNNING (PID: $PID)"
            echo "URL: http://localhost:$DASHBOARD_PORT"

            # Run health check
            echo ""
            python3 "$PROJECT_DIR/scripts/check_dashboard.py" 2>/dev/null
            return 0
        fi
    fi

    # Check port
    if lsof -i:$DASHBOARD_PORT > /dev/null 2>&1; then
        echo "Dashboard is RUNNING (port $DASHBOARD_PORT is active)"
        echo "URL: http://localhost:$DASHBOARD_PORT"
        return 0
    fi

    echo "Dashboard is NOT RUNNING"
    return 1
}

case "${1:-}" in
    daemon|bg|background)
        start_daemon
        ;;
    stop)
        stop_dashboard
        ;;
    restart)
        stop_dashboard
        sleep 2
        start_daemon
        ;;
    status)
        check_status
        ;;
    *)
        start_foreground
        ;;
esac
