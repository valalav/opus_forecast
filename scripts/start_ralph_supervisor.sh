#!/bin/bash

# Configuration
PROJECT_DIR="/home/valalav/opus_forecast"
LOG_FILE="$PROJECT_DIR/edge_lab/orchestrator.log"
CMD="python3 edge_lab/system/orchestrator.py"

cd "$PROJECT_DIR" || exit 1

echo "🚀 Ralph Supervisor Starting..."
echo "📂 Project Dir: $PROJECT_DIR"
echo "📝 Log File: $LOG_FILE"

while true; do
    echo "----------------------------------------"
    echo "⏰ Starting Ralph Orchestrator at $(date)"
    echo "----------------------------------------"
    
    # Run the orchestrator and append output to log
    # We use unbuffered python output (-u) to see logs in real-time
    $CMD >> "$LOG_FILE" 2>&1
    
    EXIT_CODE=$?
    
    echo "⚠️  Ralph exited with code: $EXIT_CODE"
    echo "⚠️  Restarting in 5 seconds..."
    sleep 5
done
