"""
RSI Multi-Timeframe Monitor Dashboard

A Streamlit dashboard for monitoring RSI signals across multiple timeframes and symbols.
"""

import streamlit as st
import pandas as pd
import time
import sys
import os
from datetime import datetime

# Add parent directory to path for Streamlit Cloud deployment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsi_monitor import RSIMonitor, DataFetcher, SignalGenerator, VIXAnalyzer


# Signal color mapping
SIGNAL_COLORS = {
    "STRONG_BUY": "#00C851",   # Bright green
    "BUY": "#007E33",          # Dark green
    "NEUTRAL": "#6c757d",      # Gray
    "SELL": "#CC0000",         # Dark red
    "STRONG_SELL": "#ff4444",  # Bright red
}

# RSI 7-level color mapping
RSI_LEVEL_COLORS = {
    "Extremely Oversold": "#00FF00",      # Bright Green
    "Oversold": "#00C851",                # Green
    "Approaching Oversold": "#90EE90",    # Light Green
    "Neutral": "#6c757d",                 # Gray
    "Approaching Overbought": "#FF6B6B",  # Light Red
    "Overbought": "#ff4444",              # Red
    "Extremely Overbought": "#FF0000",    # Bright Red
}


def get_rsi_status(rsi_value: float, oversold: float = 30, overbought: float = 70) -> str:
    """Return 7-level status based on RSI value with dynamic boundaries."""
    extreme_oversold = oversold - 10
    approaching_oversold_upper = oversold + 10
    approaching_overbought_lower = overbought - 10
    extreme_overbought = overbought + 10

    if rsi_value < extreme_oversold:
        return "Extremely Oversold"
    elif rsi_value < oversold:
        return "Oversold"
    elif rsi_value < approaching_oversold_upper:
        return "Approaching Oversold"
    elif rsi_value <= approaching_overbought_lower:
        return "Neutral"
    elif rsi_value <= overbought:
        return "Approaching Overbought"
    elif rsi_value <= extreme_overbought:
        return "Overbought"
    else:
        return "Extremely Overbought"


def get_rsi_color(rsi_value: float, oversold: float = 30, overbought: float = 70) -> str:
    """Return color based on RSI value using 7-level gradations."""
    status = get_rsi_status(rsi_value, oversold, overbought)
    return RSI_LEVEL_COLORS.get(status, "#6c757d")


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
            status = get_rsi_status(rsi_value, oversold, overbought)
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


def get_vix_confirmation_color(confirmation) -> str:
    """Get color based on VIX confirmation strength."""
    if confirmation.strength >= 3:
        return "#00C851"  # Green - high fear (bullish for stocks)
    elif confirmation.strength <= 1:
        return "#ff4444"  # Red - low fear (bearish for stocks)
    else:
        return "#6c757d"  # Gray - neutral


def create_strength_bar(strength: int, total: int) -> str:
    """Create a visual strength bar."""
    filled = strength
    empty = total - strength
    return "[" + "|" * filled + "-" * empty + "]"


def render_vix_confirmation_card(confirmation, data_fetcher) -> None:
    """Render comprehensive VIX confirmation card with help tooltips."""
    vix_color = get_vix_confirmation_color(confirmation)
    strength_bar = create_strength_bar(confirmation.strength, confirmation.total_indicators)

    # Main card header
    st.markdown(f"""
    <div style="padding: 15px; border-radius: 10px; background-color: {vix_color}20; border-left: 5px solid {vix_color};">
        <h4 style="margin: 0 0 10px 0;">VIX Confirmation</h4>
        <p style="margin: 0; font-size: 1.1em;"><strong>Strength: {confirmation.strength}/{confirmation.total_indicators}</strong> <code>{strength_bar}</code></p>
    </div>
    """, unsafe_allow_html=True)

    # Detailed indicators with help tooltips
    st.markdown("---")

    # Row 1: VIX Price and Change
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="VIX",
            value=f"{confirmation.vix_price:.1f}",
            help="Fear gauge. <15 = complacency (bearish). 15-20 = normal. >30 = high fear (bullish for stocks)."
        )
    with col2:
        change_sign = "+" if confirmation.vix_change_pct >= 0 else ""
        st.metric(
            label="Change",
            value=f"{change_sign}{confirmation.vix_change_pct:.1f}%",
            help="Daily VIX change. Spike >10% = panic/capitulation (often bullish). Drop >10% = risk-on mood."
        )

    # Row 2: Percentile and Level
    col3, col4 = st.columns(2)
    with col3:
        st.metric(
            label="Percentile",
            value=f"{confirmation.percentile_rank:.0f}th",
            help="Where VIX sits vs past year. >75th = elevated fear (bullish). <25th = complacency (bearish)."
        )
    with col4:
        st.metric(
            label="Level",
            value=confirmation.level_status,
            help="VIX zone: Normal (15-20), Caution (20-30), Elevated (30-40), Panic (>40), Complacent (<15)."
        )

    # Row 3: Term Structure
    if confirmation.term_structure_status != "Unknown":
        ts_display = f"{confirmation.term_structure_status} ({confirmation.term_structure_pct:+.1f}%)"
        st.metric(
            label="Term Structure",
            value=ts_display,
            help="Backwardation (VIX > VIX3M) = near-term panic, often marks bottoms (bullish). Contango = normal/complacent."
        )

    # Row 4: VVIX
    if confirmation.vvix_level is not None:
        vvix_display = f"{confirmation.vvix_level:.0f} ({confirmation.vvix_status})"
        st.metric(
            label="VVIX",
            value=vvix_display,
            help="Volatility of VIX. >120 = extreme uncertainty (bullish). <80 = complacent. Normal = 80-120."
        )

    # Overall sentiment description
    sentiment_colors = {
        'extreme_fear': '#00C851',
        'fear': '#007E33',
        'neutral': '#6c757d',
        'greed': '#CC0000',
        'extreme_greed': '#ff4444',
        'unknown': '#6c757d',
    }
    sent_color = sentiment_colors.get(confirmation.overall_sentiment, '#6c757d')
    st.markdown(f"<small style='color:{sent_color}'>{confirmation.description}</small>", unsafe_allow_html=True)


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

    # Get comprehensive VIX confirmation using VIXAnalyzer
    vix_analyzer = VIXAnalyzer()

    # Fetch VIX data for comprehensive analysis
    vix_df = data_fetcher.fetch("^VIX", interval="1d", period="1y")
    vix3m_df = data_fetcher.fetch("^VIX3M", interval="1d", period="5d")
    vvix_df = data_fetcher.fetch("^VVIX", interval="1d", period="5d")

    # Get comprehensive VIX confirmation
    vix_confirmation = vix_analyzer.analyze_from_dataframe(vix_df, vix3m_df, vvix_df)
    vix_color = get_vix_confirmation_color(vix_confirmation)

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
        # VIX Confirmation - Comprehensive Display
        render_vix_confirmation_card(vix_confirmation, data_fetcher)

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

                            # VIX confirmation - consistent display for all tickers
                            weighted_score = vix_confirmation.weighted_score if hasattr(vix_confirmation, 'weighted_score') else vix_confirmation.strength
                            max_score = vix_confirmation.max_weighted_score if hasattr(vix_confirmation, 'max_weighted_score') else vix_confirmation.total_indicators
                            confirms_buy = "Yes" if vix_confirmation.confirms_buy else "No"
                            st.info(f"VIX: {weighted_score:.1f}/{max_score:.1f} weighted | Confirms Buy: {confirms_buy}")

                            # RSI Table
                            if rsi_data:
                                df = create_rsi_table(rsi_data, oversold_threshold, overbought_threshold)

                                # Style the dataframe with 7-level colors
                                def color_status(val):
                                    color = RSI_LEVEL_COLORS.get(val)
                                    if color:
                                        return f"background-color: {color}33; color: {color}"
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
