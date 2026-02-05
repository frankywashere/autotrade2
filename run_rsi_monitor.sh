#!/bin/bash
# Run the RSI Multi-Timeframe Monitor Dashboard
cd "$(dirname "$0")"
streamlit run rsi_monitor/dashboard.py
