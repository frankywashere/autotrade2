"""
RSI Multi-Timeframe Monitor Dashboard

A Streamlit dashboard for monitoring RSI signals across multiple timeframes and symbols.
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime

from rsi_monitor import RSIMonitor, DataFetcher, SignalGenerator


# Signal color mapping
SIGNAL_COLORS = {
    "STRONG_BUY": "#00C851",   # Bright green
    "BUY": "#007E33",          # Dark green
    "NEUTRAL": "#6c757d",      # Gray
    "SELL": "#CC0000",         # Dark red
    "STRONG_SELL": "#ff4444",  # Bright red
}

# RSI color thresholds
def get_rsi_color(rsi_value: float, oversold: float, overbought: float) -> str:
    """Return color based on RSI value."""
    if rsi_value <= oversold:
        return "#00C851"  # Green - oversold (buy opportunity)
    elif rsi_value >= overbought:
        return "#ff4444"  # Red - overbought (sell opportunity)
    else:
        return "#6c757d"  # Gray - neutral


def get_signal_emoji(signal: str) -> str:
    """Return emoji indicator for signal."""
    emoji_map = {
        "STRONG_BUY": "+++",
        "BUY": "++",
        "NEUTRAL": "~",
        "SELL": "--",
        "STRONG_SELL": "---",
    }
    return emoji_map.get(signal, "~")


def create_rsi_table(rsi_data: dict, oversold: float, overbought: float) -> pd.DataFrame:
    """Create a formatted DataFrame for RSI values across timeframes."""
    rows = []
    for timeframe, rsi_value in rsi_data.items():
        if rsi_value is not None:
            color = get_rsi_color(rsi_value, oversold, overbought)
            status = "Oversold" if rsi_value <= oversold else "Overbought" if rsi_value >= overbought else "Neutral"
            rows.append({
                "Timeframe": timeframe,
                "RSI": round(rsi_value, 2),
                "Status": status,
            })
    return pd.DataFrame(rows)


def calculate_confluence(rsi_data: dict, oversold: float, overbought: float) -> tuple:
    """Calculate confluence score for oversold/overbought conditions."""
    total = 0
    oversold_count = 0
    overbought_count = 0

    for timeframe, rsi_value in rsi_data.items():
        if rsi_value is not None:
            total += 1
            if rsi_value <= oversold:
                oversold_count += 1
            elif rsi_value >= overbought:
                overbought_count += 1

    return oversold_count, overbought_count, total


def get_vix_status(vix_rsi: float, oversold: float, overbought: float) -> tuple:
    """Determine VIX confirmation status."""
    if vix_rsi is None:
        return "Unknown", "#6c757d"

    if vix_rsi >= overbought:
        return "High Fear (Bullish for Stocks)", "#00C851"
    elif vix_rsi <= oversold:
        return "Low Fear (Bearish for Stocks)", "#ff4444"
    else:
        return "Neutral", "#6c757d"


def main():
    st.set_page_config(
        page_title="RSI Multi-Timeframe Monitor",
        page_icon="📊",
        layout="wide",
    )

    st.title("RSI Multi-Timeframe Monitor")

    # Initialize session state
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Symbol selection
        available_symbols = ["TSLA", "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN"]
        selected_symbols = st.multiselect(
            "Select Symbols",
            options=available_symbols,
            default=["TSLA", "SPY"],
            help="VIX is always included for confirmation"
        )

        # Ensure VIX is always included (but tracked separately)
        st.info("VIX is always monitored for confirmation")

        st.divider()

        # RSI Settings
        st.subheader("RSI Settings")
        rsi_period = st.slider(
            "RSI Period",
            min_value=5,
            max_value=21,
            value=14,
            help="Number of periods for RSI calculation"
        )

        oversold_threshold = st.slider(
            "Oversold Threshold",
            min_value=10,
            max_value=40,
            value=30,
            help="RSI below this = oversold (buy opportunity)"
        )

        overbought_threshold = st.slider(
            "Overbought Threshold",
            min_value=60,
            max_value=90,
            value=70,
            help="RSI above this = overbought (sell opportunity)"
        )

        st.divider()

        # Refresh controls
        st.subheader("Refresh")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh Now", width='stretch'):
                st.session_state.last_refresh = datetime.now()
                st.rerun()

        with col2:
            st.session_state.auto_refresh = st.checkbox(
                "Auto (60s)",
                value=st.session_state.auto_refresh
            )

        st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

    # Initialize components
    try:
        data_fetcher = DataFetcher()
        # RSIMonitor takes symbols, timeframes, and data_fetcher
        timeframes = ['5m', '15m', '1h', '4h', '1d', '1wk']
        rsi_monitor = RSIMonitor(
            symbols=list(set(selected_symbols + ["^VIX"])),
            timeframes=timeframes,
            data_fetcher=data_fetcher
        )
        signal_generator = SignalGenerator(
            oversold_threshold=oversold_threshold,
            overbought_threshold=overbought_threshold
        )
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        st.stop()

    # Fetch RSI for all symbols across all timeframes
    all_symbols = list(set(selected_symbols + ["^VIX"]))

    with st.spinner("Fetching market data and calculating RSI..."):
        try:
            # get_all_rsi() fetches data and calculates RSI for all symbols/timeframes
            rsi_results = rsi_monitor.get_all_rsi()
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    # Generate signals for all symbols
    signals = {}

    for symbol in all_symbols:
        try:
            # analyze() expects dict with 'symbol' and 'timeframes' keys
            signal_input = {'symbol': symbol, 'timeframes': rsi_results.get(symbol, {})}
            signals[symbol] = signal_generator.analyze(signal_input)
        except Exception as e:
            st.warning(f"Error generating signal for {symbol}: {e}")
            signals[symbol] = {"signal": "NEUTRAL", "strength": 0}

    # Get VIX status for confirmation
    vix_rsi = None
    if "^VIX" in rsi_results and rsi_results["^VIX"]:
        # Use daily timeframe for VIX
        vix_rsi = rsi_results["^VIX"].get("1d") or rsi_results["^VIX"].get("1wk") or list(rsi_results["^VIX"].values())[0] if rsi_results["^VIX"] else None
    vix_status, vix_color = get_vix_status(vix_rsi, oversold_threshold, overbought_threshold)

    # Summary Section
    st.header("Market Summary")

    summary_cols = st.columns([2, 1, 1])

    with summary_cols[0]:
        # Overall market status
        strong_buy_count = sum(1 for s in selected_symbols if signals.get(s, {}).get("signal") == "STRONG_BUY")
        buy_count = sum(1 for s in selected_symbols if signals.get(s, {}).get("signal") == "BUY")
        sell_count = sum(1 for s in selected_symbols if signals.get(s, {}).get("signal") == "SELL")
        strong_sell_count = sum(1 for s in selected_symbols if signals.get(s, {}).get("signal") == "STRONG_SELL")

        if strong_buy_count > 0 or buy_count >= len(selected_symbols) // 2:
            overall_status = "BULLISH"
            overall_color = "#00C851"
        elif strong_sell_count > 0 or sell_count >= len(selected_symbols) // 2:
            overall_status = "BEARISH"
            overall_color = "#ff4444"
        else:
            overall_status = "MIXED"
            overall_color = "#6c757d"

        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {overall_color}20; border-left: 5px solid {overall_color};">
            <h3 style="margin: 0; color: {overall_color};">Overall Market: {overall_status}</h3>
            <p style="margin: 5px 0 0 0;">
                Strong Buy: {strong_buy_count} | Buy: {buy_count} | Sell: {sell_count} | Strong Sell: {strong_sell_count}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with summary_cols[1]:
        # VIX Confirmation
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {vix_color}20; border-left: 5px solid {vix_color};">
            <h4 style="margin: 0;">VIX Confirmation</h4>
            <p style="margin: 5px 0 0 0; color: {vix_color};">{vix_status}</p>
            <p style="margin: 5px 0 0 0;">RSI: {round(vix_rsi, 2) if vix_rsi else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)

    with summary_cols[2]:
        # Active alerts
        alerts = []
        for symbol in selected_symbols:
            signal = signals.get(symbol, {}).get("signal", "NEUTRAL")
            if signal in ["STRONG_BUY", "STRONG_SELL"]:
                alerts.append(f"{symbol}: {signal}")

        alert_color = "#ff4444" if alerts else "#00C851"
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {alert_color}20; border-left: 5px solid {alert_color};">
            <h4 style="margin: 0;">Active Alerts</h4>
            <p style="margin: 5px 0 0 0;">{len(alerts)} alert(s)</p>
        </div>
        """, unsafe_allow_html=True)

        if alerts:
            for alert in alerts:
                st.warning(alert)

    st.divider()

    # Individual Symbol Cards
    st.header("Symbol Details")

    if not selected_symbols:
        st.info("Select at least one symbol from the sidebar to view RSI data.")
    else:
        # Create columns for symbols (2 per row)
        for i in range(0, len(selected_symbols), 2):
            cols = st.columns(2)

            for j, col in enumerate(cols):
                if i + j < len(selected_symbols):
                    symbol = selected_symbols[i + j]
                    signal_data = signals.get(symbol, {"signal": "NEUTRAL", "strength": 0})
                    signal = signal_data.get("signal", "NEUTRAL")
                    strength = signal_data.get("strength", 0)
                    signal_color = SIGNAL_COLORS.get(signal, "#6c757d")

                    with col:
                        with st.expander(f"{symbol} - {signal} {get_signal_emoji(signal)}", expanded=True):
                            # Signal header
                            st.markdown(f"""
                            <div style="padding: 10px; border-radius: 5px; background-color: {signal_color}; color: white; text-align: center; margin-bottom: 10px;">
                                <strong>{signal}</strong>
                            </div>
                            """, unsafe_allow_html=True)

                            # Confluence score
                            rsi_data = rsi_results.get(symbol, {})
                            oversold_count, overbought_count, total = calculate_confluence(
                                rsi_data, oversold_threshold, overbought_threshold
                            )

                            metric_cols = st.columns(3)
                            with metric_cols[0]:
                                st.metric("Oversold TFs", f"{oversold_count}/{total}")
                            with metric_cols[1]:
                                st.metric("Overbought TFs", f"{overbought_count}/{total}")
                            with metric_cols[2]:
                                st.metric("Strength", f"{strength:.0%}" if isinstance(strength, float) else str(strength))

                            # VIX confirmation for this symbol
                            if signal in ["BUY", "STRONG_BUY"] and vix_status == "High Fear (Bullish for Stocks)":
                                st.success("VIX Confirmed: High fear supports buy signal")
                            elif signal in ["SELL", "STRONG_SELL"] and vix_status == "Low Fear (Bearish for Stocks)":
                                st.success("VIX Confirmed: Low fear supports sell signal")
                            elif signal in ["BUY", "STRONG_BUY", "SELL", "STRONG_SELL"]:
                                st.warning("VIX Not Confirmed")

                            # RSI Table
                            if rsi_data:
                                df = create_rsi_table(rsi_data, oversold_threshold, overbought_threshold)

                                # Style the dataframe
                                def color_status(val):
                                    if val == "Oversold":
                                        return "background-color: #00C85133; color: #00C851"
                                    elif val == "Overbought":
                                        return "background-color: #ff444433; color: #ff4444"
                                    return ""

                                styled_df = df.style.map(
                                    color_status, subset=["Status"]
                                )

                                st.dataframe(
                                    styled_df,
                                    width='stretch',
                                    hide_index=True
                                )
                            else:
                                st.warning("No RSI data available")

    # Footer
    st.divider()
    st.caption(f"RSI Period: {rsi_period} | Oversold: <{oversold_threshold} | Overbought: >{overbought_threshold}")

    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
