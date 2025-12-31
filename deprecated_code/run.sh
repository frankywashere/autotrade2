#!/bin/bash

# Linear Regression Trading System - Quick Start Script

echo "========================================"
echo "Linear Regression Trading System"
echo "========================================"
echo ""
echo "Select an option:"
echo "1) Dashboard - Launch GUI with integrated monitoring"
echo "2) Signal - Generate current signal (one-time)"
echo "3) Monitor - Console monitoring only (no GUI)"
echo "4) Test - Test all components"
echo "5) Exit"
echo ""
read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo "Launching dashboard with integrated monitoring..."
        echo "Dashboard will open at http://localhost:8501"
        echo ""
        echo "Features:"
        echo "  • Real-time charts and analysis"
        echo "  • Click 'Start Monitor' in sidebar for auto-alerts"
        echo "  • Telegram alerts when confidence >= 60"
        echo ""
        python3 main.py dashboard
        ;;
    2)
        read -p "Stock (TSLA/SPY) [TSLA]: " stock
        stock=${stock:-TSLA}
        read -p "Timeframe (1hour/4hour/daily) [4hour]: " tf
        tf=${tf:-4hour}
        python3 main.py signal --stock $stock --timeframe $tf
        ;;
    3)
        read -p "Stock (TSLA/SPY) [TSLA]: " stock
        stock=${stock:-TSLA}
        read -p "Check interval in minutes [60]: " interval
        interval=${interval:-60}
        echo "Starting monitor mode..."
        echo "Alerts will be sent to Telegram when confidence >= 60"
        python3 main.py monitor --stock $stock --interval $interval
        ;;
    4)
        python3 main.py test
        ;;
    5)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-5."
        ;;
esac