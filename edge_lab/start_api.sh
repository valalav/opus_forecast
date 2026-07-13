#!/bin/bash
# Startup script for Opus Edge Lab API
# This script starts the API server on port 8000

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: venv not found. Please create a virtual environment first."
    exit 1
fi

# Check if required packages are installed
if ! venv/bin/python -c "import fastapi" 2>/dev/null; then
    echo "Installing required packages..."
    venv/bin/pip install fastapi uvicorn shap scikit-learn numpy pandas -q
fi

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Port 8000 is already in use. Killing existing process..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Start the uvicorn server
echo "Starting Opus Edge Lab API on http://0.0.0.0:8000"
venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
