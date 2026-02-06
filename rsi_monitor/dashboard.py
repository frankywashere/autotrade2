"""
Market Pulse Dashboard

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


def ordinal(n: int) -> str:
    """Return ordinal string for number (1st, 2nd, 3rd, etc.)"""
    if 11 <= n % 100 <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def get_rsi_status(rsi_value: float, oversold: float = 30, overbought: float = 70, percentile: float = None) -> str:
    """Return 7-level status based on RSI value with dynamic boundaries."""
    extreme_oversold = oversold - 10
    approaching_oversold_upper = oversold + 10
    approaching_overbought_lower = overbought - 10
    extreme_overbought = overbought + 10

    if rsi_value < extreme_oversold:
        status = "Extremely Oversold"
    elif rsi_value < oversold:
        status = "Oversold"
    elif rsi_value < approaching_oversold_upper:
        status = "Approaching Oversold"
    elif rsi_value <= approaching_overbought_lower:
        status = "Neutral"
    elif rsi_value <= overbought:
        status = "Approaching Overbought"
    elif rsi_value <= extreme_overbought:
        status = "Overbought"
    else:
        status = "Extremely Overbought"

    # Add percentile suffix if notable (<=10 or >=90)
    if percentile is not None and (percentile <= 10 or percentile >= 90):
        display_pct = 100 - percentile if percentile <= 10 else percentile
        pct_int = int(round(display_pct))
        status = f"{status} ({ordinal(pct_int)} %ile)"

    return status


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
    """Create a formatted DataFrame for RSI values across timeframes.

    Supports both old format (just rsi value) and new format (dict with rsi and percentile).
    Old format: {timeframe: rsi_value}
    New format: {timeframe: {'rsi': value, 'percentile': value_or_none}}
    """
    rows = []
    for timeframe, data in rsi_data.items():
        # Handle both old format (just rsi value) and new format (dict with rsi and percentile)
        if isinstance(data, dict):
            rsi_value = data.get('rsi')
            percentile = data.get('percentile')
        else:
            rsi_value = data
            percentile = None

        if rsi_value is not None:
            status = get_rsi_status(rsi_value, oversold, overbought, percentile)
            rows.append({
                "Timeframe": timeframe,
                "RSI": round(rsi_value, 2),
                "Status": status,
            })
    return pd.DataFrame(rows)


def calculate_confluence(rsi_data: dict, oversold: float, overbought: float) -> tuple:
    """Calculate confluence score for oversold/overbought conditions.

    Supports both old format (just rsi value) and new format (dict with rsi and percentile).
    """
    total = 0
    oversold_count = 0
    overbought_count = 0

    for timeframe, data in rsi_data.items():
        # Handle both old format (just rsi value) and new format (dict with rsi and percentile)
        if isinstance(data, dict):
            rsi_value = data.get('rsi')
        else:
            rsi_value = data

        if rsi_value is not None:
            total += 1
            if rsi_value <= oversold:
                oversold_count += 1
            elif rsi_value >= overbought:
                overbought_count += 1

    return oversold_count, overbought_count, total


def get_vix_confirmation_color(confirmation) -> str:
    """Get color based on VIX confirmation strength (high fear = red, low fear = green)."""
    if confirmation.strength >= 3:
        return "#ff4444"  # Red - high fear
    elif confirmation.strength <= 1:
        return "#00C851"  # Green - low fear (calm/greed)
    else:
        return "#6c757d"  # Gray - neutral


def create_strength_bar(strength: int, total: int) -> str:
    """Create a visual strength bar."""
    filled = strength
    empty = total - strength
    return "[" + "|" * filled + "-" * empty + "]"


def get_vix_level_color(vix_price: float) -> str:
    """Get color for VIX level (high fear = red, low/calm = green)."""
    if vix_price >= 40:
        return "#FF0000"  # Bright red - panic (extreme fear)
    elif vix_price >= 30:
        return "#ff4444"  # Red - elevated fear
    elif vix_price >= 25:
        return "#FF6B6B"  # Light red - caution
    elif vix_price >= 18:
        return "#6c757d"  # Gray - normal
    elif vix_price >= 15:
        return "#B8D4B8"  # Dim green - calm
    elif vix_price >= 12:
        return "#90EE90"  # Light green - low
    else:
        return "#00C851"  # Green - very calm (greed)


def get_vix_change_color(change_pct: float) -> str:
    """Get color for VIX change (spike = red/fear, drop = green/calm)."""
    if change_pct >= 15:
        return "#FF0000"  # Bright red - major spike (fear)
    elif change_pct >= 10:
        return "#ff4444"  # Red - elevated spike
    elif change_pct >= 5:
        return "#FF6B6B"  # Light red - moderate up
    elif change_pct <= -15:
        return "#00C851"  # Green - major drop (calm/greed)
    elif change_pct <= -10:
        return "#90EE90"  # Light green - fear subsiding
    elif change_pct <= -5:
        return "#B8D4B8"  # Dim green - moderate down
    else:
        return "#6c757d"  # Gray - normal


def get_vix_percentile_color(percentile: float) -> str:
    """Get color for VIX percentile (high = red/fear, low = green/calm)."""
    if percentile >= 90:
        return "#FF0000"  # Bright red - extreme fear
    elif percentile >= 75:
        return "#ff4444"  # Red - elevated fear
    elif percentile >= 60:
        return "#FF6B6B"  # Light red - above average
    elif percentile <= 10:
        return "#00C851"  # Green - extreme calm (greed)
    elif percentile <= 25:
        return "#90EE90"  # Light green - low volatility
    elif percentile <= 40:
        return "#B8D4B8"  # Dim green - below average
    else:
        return "#6c757d"  # Gray - normal


def get_term_structure_color(status: str, pct: float) -> str:
    """Get color for term structure (backwardation = red/fear, contango = green/calm)."""
    if status == "Backwardation":
        if pct >= 10:
            return "#FF0000"  # Bright red - severe backwardation (fear)
        elif pct >= 5:
            return "#ff4444"  # Red - moderate backwardation
        else:
            return "#FF6B6B"  # Light red - mild backwardation
    elif status == "Contango":
        if pct <= -10:
            return "#00C851"  # Green - deep contango (calm/greed)
        elif pct <= -5:
            return "#90EE90"  # Light green - moderate contango
        elif pct <= -2:
            return "#B8D4B8"  # Dim green - light contango
        else:
            return "#6c757d"  # Gray - shallow contango
    return "#6c757d"


def get_vvix_color(vvix: float) -> str:
    """Get color for VVIX (high = red/fear, low = green/calm)."""
    if vvix >= 140:
        return "#FF0000"  # Bright red - extreme fear
    elif vvix >= 120:
        return "#ff4444"  # Red - elevated fear
    elif vvix >= 100:
        return "#FF6B6B"  # Light red - approaching elevated
    elif vvix <= 70:
        return "#00C851"  # Green - very low (greed)
    elif vvix <= 80:
        return "#90EE90"  # Light green - low
    elif vvix <= 90:
        return "#B8D4B8"  # Dim green - approaching low
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

    # Color based on fear percentage (high fear = red, low fear/greed = green)
    if fear_pct >= 70:
        pct_color = "#FF0000"  # Bright red - extreme fear
    elif fear_pct >= 50:
        pct_color = "#ff4444"  # Red - elevated fear
    elif fear_pct >= 35:
        pct_color = "#FF6B6B"  # Light red - moderate fear
    elif fear_pct >= 20:
        pct_color = "#6c757d"  # Gray - neutral
    else:
        pct_color = "#90EE90"  # Light green - calm/greed

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

    # Sentiment description removed - already shown in Overall Market card


def main():
    st.set_page_config(
        page_title="Market Pulse",
        page_icon="📊",
        layout="wide",
    )

    # Global CSS injection
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Apply Inter font globally */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit hamburger menu */
    #MainMenu {visibility: hidden;}

    /* Hide Streamlit footer */
    footer {visibility: hidden;}

    /* Hide deploy button */
    .stDeployButton {display: none;}

    /* Reduce top padding */
    div.block-container {padding-top: 1.5rem;}

    /* Style sidebar header area */
    [data-testid="stSidebar"] [data-testid="stHeader"] {
        background-color: transparent;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    /* Subtle border-bottom on section headers */
    .main .stMarkdown h2, .main .stMarkdown h3 {
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.4rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Styled header
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h1 style="margin: 0; padding: 0; font-family: 'Inter', sans-serif; font-weight: 700; color: #FFFFFF; font-size: 2.2rem; letter-spacing: -0.02em;">Market Pulse</h1>
        <p style="margin: 0.25rem 0 0.6rem 0; padding: 0; font-family: 'Inter', sans-serif; font-size: 0.95rem; color: #888888; font-weight: 400;">Real-time RSI signals across timeframes</p>
        <div style="width: 60px; height: 3px; background-color: #3A82FF; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)

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

        # Market Hours
        st.subheader("Market Hours")
        prepost = st.checkbox(
            "Include After-Hours Data",
            value=False,
            help="Include pre-market (4am-9:30am) and after-hours (4pm-8pm) data for intraday timeframes"
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
            # get_all_rsi_with_percentile() fetches data and calculates RSI with percentiles for all symbols/timeframes
            rsi_results = rsi_monitor.get_all_rsi_with_percentile(prepost=prepost)
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
            # Extract plain RSI values for signal analysis (signals.py expects {tf: float})
            symbol_data = rsi_results.get(symbol, {})
            plain_rsi = {}
            for tf, data in symbol_data.items():
                if isinstance(data, dict):
                    plain_rsi[tf] = data.get('rsi')
                else:
                    plain_rsi[tf] = data
            signal_input = {'symbol': symbol, 'timeframes': plain_rsi}
            signals[symbol] = signal_generator.analyze(signal_input, vix_confirmation)
        except Exception as e:
            st.warning(f"Error generating signal for {symbol}: {e}")
            signals[symbol] = {"signal": "NEUTRAL", "strength": 0}

    # Summary Section
    st.markdown('<h3 style="margin: 0 0 0.5rem 0; font-weight: 600; color: #e0e0e0; letter-spacing: 0.02em;">Market Summary</h3>', unsafe_allow_html=True)

    summary_cols = st.columns([2, 1, 1])

    with summary_cols[0]:
        # Overall market status based on VIX fear level
        fear_pct = getattr(vix_confirmation, 'fear_percentage', 0.0)
        greed_pct = getattr(vix_confirmation, 'greed_percentage', 0.0)
        confirms_sell = getattr(vix_confirmation, 'confirms_sell', False)

        if fear_pct >= 70:
            overall_status = "BUY THE DIP"
            overall_color = "#00C851"
        elif fear_pct >= 50:
            overall_status = "BUY THE DIP"
            overall_color = "#4CAF50"
        elif fear_pct >= 35:
            overall_status = "BUY THE DIP"
            overall_color = "#ffeb3b"
        elif confirms_sell:
            overall_status = "SELL THE RIP"
            overall_color = "#ff4444" if greed_pct >= 50 else "#ff9800"
        else:
            overall_status = "NEUTRAL"
            overall_color = "#6c757d"

        # Sub-label always shows fear level
        sub_label = f"Fear {fear_pct:.0f}%"
        if fear_pct >= 50:
            sub_color = "#ff4444"
        elif fear_pct >= 35:
            sub_color = "#ffeb3b"
        else:
            sub_color = "#888"

        # Signal count pills
        strong_buy_count = sum(1 for s in selected_symbols if signals.get(s, {}).get("signal") == "STRONG_BUY")
        buy_count = sum(1 for s in selected_symbols if signals.get(s, {}).get("signal") == "BUY")
        sell_count = sum(1 for s in selected_symbols if signals.get(s, {}).get("signal") == "SELL")
        strong_sell_count = sum(1 for s in selected_symbols if signals.get(s, {}).get("signal") == "STRONG_SELL")

        def _pill(label, count, color):
            return (
                f'<span style="display:inline-block;padding:2px 10px;margin:0 4px;'
                f'border-radius:20px;font-size:0.75rem;font-weight:600;'
                f'background:{color}22;color:{color};border:1px solid {color}44;">'
                f'{label} {count}</span>'
            )

        pills_html = (
            _pill("Strong Buy", strong_buy_count, "#00C851")
            + _pill("Buy", buy_count, "#4CAF50")
            + _pill("Sell", sell_count, "#ff9800")
            + _pill("Strong Sell", strong_sell_count, "#ff4444")
        )

        st.markdown(f"""
        <div style="padding: 20px 24px; border-radius: 12px; background-color: #1a1c23; border: 1px solid #2d3039;">
            <div style="margin: 0 0 10px 0; font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em;">Overall Market</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: {overall_color}; margin: 0 0 8px 0;">{overall_status}</div>
            <div style="font-size: 0.85rem; color: {sub_color}; font-weight: 600; margin: 0 0 12px 0;">{sub_label}</div>
            <div>{pills_html}</div>
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
        alert_count_text = f"{len(alerts)} alert{'s' if len(alerts) != 1 else ''}"
        alerts_list_html = ""
        if alerts:
            for alert in alerts:
                parts = alert.split(": ")
                sym = parts[0]
                sig = parts[1] if len(parts) > 1 else ""
                sig_color = "#ff4444" if "SELL" in sig else "#00C851"
                alerts_list_html += (
                    f'<div style="padding:4px 0;font-size:0.82rem;color:#ccc;">'
                    f'<span style="font-weight:600;">{sym}</span> '
                    f'<span style="color:{sig_color};font-weight:600;">{sig}</span></div>'
                )

        st.markdown(f"""
        <div style="padding: 20px 24px; border-radius: 12px; background-color: #1a1c23; border: 1px solid #2d3039;">
            <div style="margin: 0 0 10px 0; font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em;">Active Alerts</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: {alert_color}; margin: 0 0 8px 0;">{alert_count_text}</div>
            {alerts_list_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)

    # Individual Symbol Cards
    st.markdown('<h3 style="color: #a0a3b0; font-weight: 500; margin-bottom: 16px;">Symbol Details</h3>', unsafe_allow_html=True)

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
                        # Get current price (use intraday data for after-hours price)
                        if prepost:
                            price_df = data_fetcher.fetch(symbol, interval="5m", period="1d", prepost=True)
                        else:
                            price_df = data_fetcher.fetch(symbol, interval="1d", period="5d")
                        current_price = float(price_df["Close"].iloc[-1]) if price_df is not None and not price_df.empty else None
                        price_str = f"${current_price:.2f}" if current_price else "N/A"

                        # Confluence score
                        rsi_data = rsi_results.get(symbol, {})
                        oversold_count, overbought_count, total = calculate_confluence(
                            rsi_data, oversold_threshold, overbought_threshold
                        )

                        strength_display = f"{strength:.0%}" if isinstance(strength, float) else str(strength)

                        # Styled card with top accent bar
                        st.markdown(f"""
                        <div style="background-color: #1a1c23; border: 1px solid #2d3039; border-radius: 12px; padding: 0; margin-bottom: 16px; overflow: hidden;">
                            <div style="height: 3px; background-color: {signal_color};"></div>
                            <div style="padding: 20px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                                    <div style="display: flex; align-items: baseline; gap: 12px;">
                                        <span style="font-size: 1.3em; font-weight: 700; color: #e1e3ea;">{symbol}</span>
                                        <span style="font-size: 1.15em; font-weight: 600; color: #8b8fa3;">{price_str}</span>
                                    </div>
                                    <span style="display: inline-block; padding: 4px 12px; border-radius: 20px; background-color: {signal_color}33; color: {signal_color}; font-size: 0.8em; font-weight: 600; letter-spacing: 0.3px;">{signal}</span>
                                </div>
                                <div style="display: flex; gap: 10px; margin-bottom: 6px;">
                                    <div style="flex: 1; background-color: #14161b; border: 1px solid #2d3039; border-radius: 8px; padding: 10px 12px; text-align: center;">
                                        <div style="font-size: 0.7em; color: #6b7080; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Oversold TFs</div>
                                        <div style="font-size: 1.1em; font-weight: 600; color: #e1e3ea;">{oversold_count}/{total}</div>
                                    </div>
                                    <div style="flex: 1; background-color: #14161b; border: 1px solid #2d3039; border-radius: 8px; padding: 10px 12px; text-align: center;">
                                        <div style="font-size: 0.7em; color: #6b7080; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Overbought TFs</div>
                                        <div style="font-size: 1.1em; font-weight: 600; color: #e1e3ea;">{overbought_count}/{total}</div>
                                    </div>
                                    <div style="flex: 1; background-color: #14161b; border: 1px solid #2d3039; border-radius: 8px; padding: 10px 12px; text-align: center;">
                                        <div style="font-size: 0.7em; color: #6b7080; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Strength</div>
                                        <div style="font-size: 1.1em; font-weight: 600; color: #e1e3ea;">{strength_display}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # RSI Table
                        if rsi_data:
                            df = create_rsi_table(rsi_data, oversold_threshold, overbought_threshold)

                            # Style the dataframe with 7-level colors
                            def color_status(val):
                                # Extract base status (before any percentile info)
                                base_status = val.split(" (")[0] if " (" in val else val
                                color = RSI_LEVEL_COLORS.get(base_status)
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
