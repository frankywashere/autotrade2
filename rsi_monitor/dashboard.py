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


def get_vix_level_color(vix_price: float) -> str:
    """Get color for VIX level (high fear = green/bullish, low = red/bearish)."""
    if vix_price >= 40:
        return "#00FF00"  # Bright green - panic (very bullish)
    elif vix_price >= 30:
        return "#00C851"  # Green - elevated fear (bullish)
    elif vix_price >= 25:
        return "#90EE90"  # Light green - caution
    elif vix_price >= 15:
        return "#6c757d"  # Gray - normal
    elif vix_price >= 12:
        return "#FF6B6B"  # Light red - complacency
    else:
        return "#ff4444"  # Red - extreme complacency (bearish)


def get_vix_change_color(change_pct: float) -> str:
    """Get color for VIX change (spike = green/bullish, drop = red/bearish)."""
    if change_pct >= 15:
        return "#00FF00"  # Bright green - major spike
    elif change_pct >= 10:
        return "#00C851"  # Green - elevated spike
    elif change_pct >= 5:
        return "#90EE90"  # Light green - moderate up
    elif change_pct <= -10:
        return "#ff4444"  # Red - fear subsiding
    elif change_pct <= -5:
        return "#FF6B6B"  # Light red - moderate down
    else:
        return "#6c757d"  # Gray - normal


def get_vix_percentile_color(percentile: float) -> str:
    """Get color for VIX percentile (high = green/bullish, low = red/bearish)."""
    if percentile >= 90:
        return "#00FF00"  # Bright green - extreme fear
    elif percentile >= 75:
        return "#00C851"  # Green - elevated fear
    elif percentile >= 60:
        return "#90EE90"  # Light green - above average
    elif percentile <= 10:
        return "#ff4444"  # Red - extreme complacency
    elif percentile <= 25:
        return "#FF6B6B"  # Light red - low volatility
    else:
        return "#6c757d"  # Gray - normal


def get_term_structure_color(status: str, pct: float) -> str:
    """Get color for term structure (backwardation = green, contango = red)."""
    if status == "Backwardation":
        if pct >= 10:
            return "#00FF00"  # Bright green - severe backwardation
        elif pct >= 5:
            return "#00C851"  # Green - moderate backwardation
        else:
            return "#90EE90"  # Light green - mild backwardation
    elif status == "Contango":
        if pct <= -10:
            return "#ff4444"  # Red - deep contango
        elif pct <= -5:
            return "#FF6B6B"  # Light red - moderate contango
        else:
            return "#6c757d"  # Gray - shallow contango
    return "#6c757d"


def get_vvix_color(vvix: float) -> str:
    """Get color for VVIX (high = green/bullish, low = red/bearish)."""
    if vvix >= 140:
        return "#00FF00"  # Bright green - extreme
    elif vvix >= 120:
        return "#00C851"  # Green - elevated
    elif vvix >= 100:
        return "#90EE90"  # Light green - above average
    elif vvix <= 70:
        return "#ff4444"  # Red - very complacent
    elif vvix <= 80:
        return "#FF6B6B"  # Light red - complacent
    else:
        return "#6c757d"  # Gray - normal


def render_vix_indicator(label: str, value: str, color: str, help_text: str) -> None:
    """Render a single VIX indicator with color coding and working tooltip."""
    st.markdown(f"""
    <style>
    .vix-tooltip {{
        position: relative;
        display: inline-block;
        cursor: help;
    }}
    .vix-tooltip .vix-tooltiptext {{
        visibility: hidden;
        width: 180px;
        background-color: #333;
        color: #fff;
        text-align: left;
        padding: 6px 10px;
        border-radius: 4px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 0;
        opacity: 0;
        transition: opacity 0.2s;
        font-size: 0.75em;
        line-height: 1.3;
    }}
    .vix-tooltip:hover .vix-tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    <div style="margin-bottom: 8px;">
        <span class="vix-tooltip">
            <span style="color: #888; font-size: 0.85em;">{label}</span>
            <span style="color: #888; font-size: 0.7em;"> ⓘ</span>
            <span class="vix-tooltiptext">{help_text}</span>
        </span>
        <div style="color: {color}; font-size: 1.4em; font-weight: bold;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_vix_confirmation_card(confirmation, data_fetcher) -> None:
    """Render comprehensive VIX confirmation card with color-coded indicators."""
    # Get fear percentage (weighted average of all indicators)
    fear_pct = getattr(confirmation, 'fear_percentage', 0.0)

    # Color based on fear percentage
    if fear_pct >= 70:
        pct_color = "#00FF00"  # Bright green - extreme fear (bullish)
    elif fear_pct >= 50:
        pct_color = "#00C851"  # Green - elevated fear
    elif fear_pct >= 35:
        pct_color = "#90EE90"  # Light green - moderate
    elif fear_pct >= 20:
        pct_color = "#6c757d"  # Gray - low
    else:
        pct_color = "#FF6B6B"  # Light red - very low (complacent)

    # Compact summary header (always visible)
    vix_val_color = get_vix_level_color(confirmation.vix_price)
    change_color = get_vix_change_color(confirmation.vix_change_pct)
    change_sign = "+" if confirmation.vix_change_pct >= 0 else ""

    # Progress bar for fear percentage
    bar_width = int(fear_pct)

    st.markdown(f"""
    <div style="padding: 12px; border-radius: 8px; background-color: {pct_color}20; border-left: 4px solid {pct_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: bold;">VIX Fear Level</span>
            <span style="color: {pct_color}; font-weight: bold; font-size: 1.2em;">{fear_pct:.0f}%</span>
        </div>
        <div style="margin-top: 6px; background-color: #333; border-radius: 4px; height: 8px; width: 100%;">
            <div style="background-color: {pct_color}; border-radius: 4px; height: 8px; width: {bar_width}%;"></div>
        </div>
        <div style="margin-top: 8px; display: flex; gap: 20px;">
            <span>VIX: <strong style="color: {vix_val_color};">{confirmation.vix_price:.1f}</strong></span>
            <span>Change: <strong style="color: {change_color};">{change_sign}{confirmation.vix_change_pct:.1f}%</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Collapsible details section
    with st.expander("VIX Details", expanded=False):
        # Row 1: VIX and Change
        col1, col2 = st.columns(2)
        with col1:
            render_vix_indicator(
                "VIX Level",
                f"{confirmation.vix_price:.1f}",
                vix_val_color,
                "<15 complacent | 15-25 normal | >30 fear (bullish)"
            )
        with col2:
            render_vix_indicator(
                "Daily Change",
                f"{change_sign}{confirmation.vix_change_pct:.1f}%",
                change_color,
                ">10% spike (bullish) | <-10% drop (bearish)"
            )

        # Row 2: Percentile and Level Status
        col3, col4 = st.columns(2)
        with col3:
            pct_color = get_vix_percentile_color(confirmation.percentile_rank)
            render_vix_indicator(
                "Percentile (1Y)",
                f"{confirmation.percentile_rank:.0f}th",
                pct_color,
                ">75th elevated | <25th low"
            )
        with col4:
            level_color = get_vix_level_color(confirmation.vix_price)
            render_vix_indicator(
                "Zone",
                confirmation.level_status,
                level_color,
                "Normal | Caution | Elevated | Panic"
            )

        # Row 3: Term Structure and VVIX
        col5, col6 = st.columns(2)
        with col5:
            if confirmation.term_structure_status != "Unknown":
                ts_color = get_term_structure_color(confirmation.term_structure_status, confirmation.term_structure_pct)
                ts_display = f"{confirmation.term_structure_status} ({confirmation.term_structure_pct:+.1f}%)"
                render_vix_indicator(
                    "Term Structure",
                    ts_display,
                    ts_color,
                    "Backwardation = fear | Contango = calm"
                )
        with col6:
            if confirmation.vvix_level is not None:
                vvix_color = get_vvix_color(confirmation.vvix_level)
                vvix_display = f"{confirmation.vvix_level:.0f} ({confirmation.vvix_status})"
                render_vix_indicator(
                    "VVIX",
                    vvix_display,
                    vvix_color,
                    ">120 elevated | <80 complacent"
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

    # Get comprehensive VIX confirmation FIRST (needed for signal strength calculation)
    vix_analyzer = VIXAnalyzer()
    vix_df = data_fetcher.fetch("^VIX", interval="1d", period="1y")
    vix3m_df = data_fetcher.fetch("^VIX3M", interval="1d", period="5d")
    vvix_df = data_fetcher.fetch("^VVIX", interval="1d", period="5d")
    vix_confirmation = vix_analyzer.analyze_from_dataframe(vix_df, vix3m_df, vvix_df)
    vix_color = get_vix_confirmation_color(vix_confirmation)

    # Generate signals for all symbols (with VIX confirmation for strength bonus)
    signals = {}

    for symbol in all_symbols:
        try:
            # analyze() expects dict with 'symbol' and 'timeframes' keys
            signal_input = {'symbol': symbol, 'timeframes': rsi_results.get(symbol, {})}
            signals[symbol] = signal_generator.analyze(signal_input, vix_confirmation)
        except Exception as e:
            st.warning(f"Error generating signal for {symbol}: {e}")
            signals[symbol] = {"signal": "NEUTRAL", "strength": 0}

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
                            fear_pct = getattr(vix_confirmation, 'fear_percentage', 0.0)
                            confirms_buy = "Yes" if vix_confirmation.confirms_buy else "No"
                            st.info(f"VIX Fear: {fear_pct:.0f}% | Confirms Buy: {confirms_buy}")

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
