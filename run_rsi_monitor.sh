#!/bin/bash
# Run the RSI Multi-Timeframe Monitor Dashboard
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
streamlit run rsi_monitor/dashboard.py
