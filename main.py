#!/usr/bin/env python3
"""
Main entry point for the Linear Regression Channel Trading System.

This script can run in different modes:
- dashboard: Launch the Streamlit GUI dashboard
- signal: Generate a single signal and print to console
- monitor: Continuously monitor and send alerts via Telegram
- test: Test all components
"""
import sys
import argparse
import asyncio
from datetime import datetime
import time

sys.path.insert(0, 'src')

from src.data_handler import DataHandler
from src.signal_generator import SignalGenerator
from src.telegram_bot import TelegramAlertBot
import config


def run_dashboard(enhanced=True):
    """Launch the Streamlit dashboard."""
    import subprocess
    print("Launching Streamlit dashboard...")

    # Use enhanced dashboard with integrated monitoring
    dashboard_file = "src/gui_dashboard_enhanced.py" if enhanced else "src/gui_dashboard.py"

    subprocess.run([
        "streamlit", "run",
        dashboard_file,
        "--server.port", str(config.DASHBOARD_PORT)
    ])


def generate_signal(stock: str = config.DEFAULT_STOCK, timeframe: str = None):
    """Generate and display a single trading signal."""
    print(f"\n{'='*70}")
    if timeframe:
        print(f"Generating signal for {stock} (using {timeframe})...")
    else:
        print(f"Generating signal for {stock} (auto-selecting best channel)...")
    print(f"{'='*70}\n")

    try:
        generator = SignalGenerator(stock)
        signal = generator.generate_signal(timeframe)

        # Print summary
        summary = generator.get_signal_summary(signal)
        print(summary)

        return signal

    except Exception as e:
        print(f"Error generating signal: {e}")
        import traceback
        traceback.print_exc()
        return None


async def monitor_and_alert(stock: str = config.DEFAULT_STOCK,
                           timeframe: str = None,
                           interval_minutes: int = 60):
    """
    Continuously monitor stock and send alerts for high-confidence signals.

    Args:
        stock: Stock symbol to monitor
        timeframe: Primary timeframe for analysis
        interval_minutes: Minutes between checks
    """
    print(f"\n{'='*70}")
    print(f"Starting monitoring for {stock} ({timeframe})")
    print(f"Checking every {interval_minutes} minutes")
    print(f"Minimum confidence for alerts: {config.MIN_CONFLUENCE_SCORE}")
    print(f"{'='*70}\n")

    generator = SignalGenerator(stock)
    telegram_bot = TelegramAlertBot()

    # Test Telegram connection
    await telegram_bot.test_connection()

    last_signal_type = None
    last_alert_time = None

    try:
        while True:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking {stock}...")

            try:
                # Generate signal
                signal = generator.generate_signal(timeframe)

                print(f"Signal: {signal.signal_type.upper()} | "
                      f"Confidence: {signal.confidence_score:.1f}/100 | "
                      f"Price: ${signal.current_price:.2f}")

                # Send alert if high confidence and signal changed
                if (signal.confidence_score >= config.MIN_CONFLUENCE_SCORE and
                    signal.signal_type != "neutral" and
                    signal.signal_type != last_signal_type):

                    print(f"\n🚨 HIGH CONFIDENCE {signal.signal_type.upper()} SIGNAL DETECTED!")
                    print(f"Sending Telegram alert...")

                    await telegram_bot.send_signal_alert(signal)
                    last_signal_type = signal.signal_type
                    last_alert_time = datetime.now()

                    print(f"✓ Alert sent successfully")

                elif signal.signal_type == last_signal_type:
                    print(f"(Same signal as before, no alert sent)")

                else:
                    print(f"(Below confidence threshold or neutral)")

            except Exception as e:
                print(f"Error during check: {e}")
                import traceback
                traceback.print_exc()

            # Wait for next check
            print(f"\nNext check in {interval_minutes} minutes...")
            await asyncio.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")


def test_components():
    """Test all system components."""
    print("\n" + "="*70)
    print("TESTING ALL COMPONENTS")
    print("="*70)

    stock = config.DEFAULT_STOCK

    # Test 1: Data Handler
    print("\n1. Testing Data Handler...")
    try:
        handler = DataHandler(stock)
        handler.load_1min_data()
        data_4h = handler.get_data("4hour")
        print(f"   ✓ Data handler working: {len(data_4h)} bars of 4-hour data")
    except Exception as e:
        print(f"   ✗ Data handler failed: {e}")
        return

    # Test 2: Linear Regression
    print("\n2. Testing Linear Regression Channel...")
    try:
        from src.linear_regression import LinearRegressionChannel
        calc = LinearRegressionChannel()
        channel = calc.calculate_channel(data_4h)
        print(f"   ✓ Channel calculator working")
        print(f"     - Stability: {channel.stability_score:.1f}/100")
        print(f"     - Ping-pongs: {channel.ping_pongs}")
        print(f"     - Predicted high: ${channel.predicted_high:.2f}")
    except Exception as e:
        print(f"   ✗ Channel calculator failed: {e}")

    # Test 3: RSI
    print("\n3. Testing RSI Calculator...")
    try:
        from src.rsi_calculator import RSICalculator
        rsi_calc = RSICalculator()
        rsi_data = rsi_calc.get_rsi_data(data_4h)
        print(f"   ✓ RSI calculator working: RSI = {rsi_data.value:.1f}")
    except Exception as e:
        print(f"   ✗ RSI calculator failed: {e}")

    # Test 4: News Analyzer
    print("\n4. Testing News Analyzer...")
    try:
        from src.news_analyzer import NewsAnalyzer
        news = NewsAnalyzer(stock)
        articles = news.fetch_news(hours_back=24)
        print(f"   ✓ News analyzer working: {len(articles)} articles fetched")

        if articles:
            print("   Analyzing first article with Claude AI...")
            analyzed = news.analyze_article(articles[0])
            print(f"     - Sentiment: {analyzed.sentiment}")
            print(f"     - BS Score: {analyzed.bs_score:.1f}/100")
    except Exception as e:
        print(f"   ✗ News analyzer failed: {e}")

    # Test 5: Signal Generator
    print("\n5. Testing Signal Generator...")
    try:
        generator = SignalGenerator(stock)
        signal = generator.generate_signal("4hour")
        print(f"   ✓ Signal generator working")
        print(f"     - Signal: {signal.signal_type.upper()}")
        print(f"     - Confidence: {signal.confidence_score:.1f}/100")
    except Exception as e:
        print(f"   ✗ Signal generator failed: {e}")

    # Test 6: Telegram Bot
    print("\n6. Testing Telegram Bot...")
    try:
        async def test_telegram():
            bot = TelegramAlertBot()
            await bot.test_connection()

        asyncio.run(test_telegram())
    except Exception as e:
        print(f"   ⚠ Telegram bot test: {e}")

    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Linear Regression Channel Trading System"
    )

    parser.add_argument(
        "mode",
        choices=["dashboard", "signal", "monitor", "test"],
        help="Run mode: dashboard (GUI), signal (one-time), monitor (continuous), test (test components)"
    )

    parser.add_argument(
        "--stock",
        default=config.DEFAULT_STOCK,
        choices=config.STOCKS,
        help=f"Stock symbol (default: {config.DEFAULT_STOCK})"
    )

    parser.add_argument(
        "--timeframe",
        default=None,
        choices=["1hour", "2hour", "3hour", "4hour", "daily", "weekly", None],
        help="Primary timeframe for analysis (default: auto-select best channel)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Minutes between checks in monitor mode (default: 60)"
    )

    args = parser.parse_args()

    if args.mode == "dashboard":
        run_dashboard()

    elif args.mode == "signal":
        generate_signal(args.stock, args.timeframe)

    elif args.mode == "monitor":
        asyncio.run(monitor_and_alert(args.stock, args.timeframe, args.interval))

    elif args.mode == "test":
        test_components()


if __name__ == "__main__":
    main()
