#!/bin/bash
# Quick start script for AutoTrade2 backend

echo "🚀 Starting AutoTrade2 Backend..."

# Activate virtual environment if exists
if [ -d "../myenv" ]; then
    source ../myenv/bin/activate
fi

# Start server
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
