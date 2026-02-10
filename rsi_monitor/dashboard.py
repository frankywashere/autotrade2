"""
Market Pulse Dashboard

A Streamlit dashboard for monitoring RSI signals across multiple timeframes and symbols.
"""

import logging
import streamlit as st
import pandas as pd
import time
import sys
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Add parent directory to path for Streamlit Cloud deployment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsi_monitor import RSIMonitor, DataFetcher, SignalGenerator, VIXAnalyzer
from rsi_monitor.channel import get_channel_context_with_data, CHANNEL_WIN_RATES
from rsi_monitor.bounce_rules import BounceRuleEngine


# Signal color mapping
SIGNAL_COLORS = {
    "STRONG_BUY": "#00FF00",       # Bright green
    "LONG_TERM_BUY": "#00C851",    # Green
    "BUY": "#00C851",              # Green
    "SHORT_TERM_BUY": "#4CAF50",   # Muted green
    "NEUTRAL": "#6c757d",          # Gray
    "SHORT_TERM_SELL": "#ff9800",  # Orange
    "SELL": "#ff4444",             # Red
    "LONG_TERM_SELL": "#ff4444",   # Red
    "STRONG_SELL": "#FF0000",      # Bright red
}

# Human-readable signal display names
SIGNAL_DISPLAY = {
    "STRONG_BUY": "STRONG BUY",
    "LONG_TERM_BUY": "LONG-TERM BUY",
    "BUY": "BUY",
    "SHORT_TERM_BUY": "SHORT-TERM BUY",
    "NEUTRAL": "NEUTRAL",
    "SHORT_TERM_SELL": "SHORT-TERM SELL",
    "SELL": "SELL",
    "LONG_TERM_SELL": "LONG-TERM SELL",
    "STRONG_SELL": "STRONG SELL",
}

# RSI 7-level color mapping
RSI_LEVEL_COLORS = {
    "Extremely Oversold": "#00FF00",      # Bright Green
    "Oversold": "#00C851",                # Green
    "Approaching Oversold": "#90EE90",    # Light Green
    "Neutral": "#6c757d",                 # Gray
    "Approaching Overbought": "#ffeb3b",  # Yellow
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
        "LONG_TERM_BUY": "++",
        "BUY": "++",
        "SHORT_TERM_BUY": "+",
        "NEUTRAL": "~",
        "SHORT_TERM_SELL": "-",
        "SELL": "--",
        "LONG_TERM_SELL": "--",
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
    div.block-container {padding-top: 3.5rem;}

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
    if "recovery_mode_active" not in st.session_state:
        st.session_state.recovery_mode_active = False

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

        is_recovery = st.session_state.recovery_mode_active

        if is_recovery:
            st.info("Thresholds locked (recovery mode active)")

        oversold_threshold = st.slider(
            "Oversold Threshold",
            min_value=10,
            max_value=40,
            value=30,
            help="RSI below this = oversold (buy opportunity)",
            disabled=is_recovery
        )

        overbought_threshold = st.slider(
            "Overbought Threshold",
            min_value=60,
            max_value=90,
            value=70,
            help="RSI above this = overbought (sell opportunity)",
            disabled=is_recovery
        )

        st.divider()

        # Market Hours
        st.subheader("Market Hours")
        prepost = st.checkbox(
            "Include After-Hours Data",
            value=True,
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
                "Auto (5m)",
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

    # Extract recent VIX daily % changes for recovery lookback
    vix_daily_changes = []
    if vix_df is not None and len(vix_df) >= 3:
        closes = vix_df["Close"].dropna()
        if len(closes) >= 3:
            for i in range(-2, 0):  # Last 2 daily changes
                prev = float(closes.iloc[i - 1])
                curr = float(closes.iloc[i])
                if prev > 0:
                    vix_daily_changes.append(((curr - prev) / prev) * 100)

    vix_color = get_vix_confirmation_color(vix_confirmation)

    # Extract VIX RSI history for recovery window detection
    vix_rsi_history = {}
    vix_symbol_data = rsi_results.get("^VIX", {})
    for tf, data in vix_symbol_data.items():
        if isinstance(data, dict):
            hist = data.get('rsi_history')
            if hist:
                vix_rsi_history[tf] = hist

    # Determine which symbol to analyze in Channel Bounce tab
    equity_symbols = [s for s in all_symbols if not s.startswith("^")]
    if "bounce_symbol" not in st.session_state:
        st.session_state.bounce_symbol = equity_symbols[0] if equity_symbols else "TSLA"
    # If the stored symbol was deselected, fall back to first available
    if st.session_state.bounce_symbol not in equity_symbols and equity_symbols:
        st.session_state.bounce_symbol = equity_symbols[0]
    bounce_symbol = st.session_state.bounce_symbol

    # Compute channel context for the selected bounce symbol (5m, 15m, 1h, 4h)
    bounce_channel_context = {}
    bounce_ohlcv_data = {}
    if bounce_symbol in all_symbols:
        try:
            bounce_channel_context, bounce_ohlcv_data = get_channel_context_with_data(
                rsi_results=rsi_results,
                data_fetcher=data_fetcher,
                symbol=bounce_symbol,
                prepost=prepost
            )
        except Exception:
            bounce_channel_context = {}
            bounce_ohlcv_data = {}

    # Generate signals for all symbols (with VIX confirmation for strength bonus)
    signals = {}

    for symbol in all_symbols:
        try:
            # Extract plain RSI values and history for signal analysis
            symbol_data = rsi_results.get(symbol, {})
            plain_rsi = {}
            rsi_history = {}
            for tf, data in symbol_data.items():
                if isinstance(data, dict):
                    plain_rsi[tf] = data.get('rsi')
                    hist = data.get('rsi_history')
                    if hist:
                        rsi_history[tf] = hist
                else:
                    plain_rsi[tf] = data
            signal_input = {'symbol': symbol, 'timeframes': plain_rsi}
            ch_ctx = bounce_channel_context if symbol == bounce_symbol else None
            signals[symbol] = signal_generator.analyze(signal_input, vix_confirmation, rsi_history=rsi_history, vix_rsi_history=vix_rsi_history, vix_daily_changes=vix_daily_changes, channel_context=ch_ctx)
        except Exception as e:
            st.warning(f"Error generating signal for {symbol}: {e}")
            signals[symbol] = {"signal": "NEUTRAL", "strength": 0}

    # Persist recovery mode for next render cycle (controls slider locking)
    any_recovery = any(
        s.get("recovery_suppressed", False)
        for s in signals.values()
    )
    st.session_state.recovery_mode_active = any_recovery

    # Compute bounce assessment for selected symbol (used in Channel Bounce tab)
    bounce_assessment = None
    if bounce_symbol in all_symbols and bounce_channel_context:
        try:
            bounce_engine = BounceRuleEngine()
            symbol_rsi_data = rsi_results.get(bounce_symbol, {})
            spy_rsi_data = rsi_results.get("SPY", {})
            bounce_assessment = bounce_engine.evaluate(
                channel_context=bounce_channel_context,
                rsi_results=symbol_rsi_data,
                vix_confirmation=vix_confirmation,
                ohlcv_data=bounce_ohlcv_data,
                spy_rsi_results=spy_rsi_data if "SPY" in all_symbols else None,
                symbol=bounce_symbol,
            )
        except Exception as e:
            logger.warning("Bounce assessment failed for %s: %s", bounce_symbol, e)

    # --- Tabs ---
    tabs = st.tabs(["Market Pulse", "Channel Bounce"])

    # === Market Pulse Tab ===
    with tabs[0]:
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
                overall_status = "CAUTIOUS - BUY"
                overall_color = "#ffeb3b"
            elif fear_pct < 10 and confirms_sell:
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

            # Signal count pills (group BUY-family and SELL-family)
            strong_buy_count = sum(1 for s in selected_symbols if signals.get(s, {}).get("signal") == "STRONG_BUY")
            buy_count = sum(1 for s in selected_symbols if signals.get(s, {}).get("signal") in ("BUY", "SHORT_TERM_BUY", "LONG_TERM_BUY"))
            sell_count = sum(1 for s in selected_symbols if signals.get(s, {}).get("signal") in ("SELL", "SHORT_TERM_SELL", "LONG_TERM_SELL"))
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

            st.markdown(f"""<div style="padding: 20px 24px; border-radius: 12px; background-color: #1a1c23; border: 1px solid #2d3039;"><div style="margin: 0 0 10px 0; font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em;">Overall Market</div><div style="font-size: 1.6rem; font-weight: 700; color: {overall_color}; margin: 0 0 8px 0;">{overall_status}</div><div style="font-size: 0.85rem; color: {sub_color}; font-weight: 600; margin: 0 0 12px 0;">{sub_label}</div><div>{pills_html}</div></div>""", unsafe_allow_html=True)

        with summary_cols[1]:
            # VIX Confirmation - Comprehensive Display
            render_vix_confirmation_card(vix_confirmation, data_fetcher)

        with summary_cols[2]:
            # Active alerts
            alerts = []
            for symbol in selected_symbols:
                signal = signals.get(symbol, {}).get("signal", "NEUTRAL")
                if signal in ["STRONG_BUY", "LONG_TERM_BUY", "BUY", "SHORT_TERM_BUY",
                             "SHORT_TERM_SELL", "SELL", "LONG_TERM_SELL", "STRONG_SELL"]:
                    display_signal = SIGNAL_DISPLAY.get(signal, signal)
                    alerts.append(f"{symbol}: {display_signal}")

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

            st.markdown(f"""<div style="padding: 20px 24px; border-radius: 12px; background-color: #1a1c23; border: 1px solid #2d3039;"><div style="margin: 0 0 10px 0; font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em;">Active Alerts</div><div style="font-size: 1.6rem; font-weight: 700; color: {alert_color}; margin: 0 0 8px 0;">{alert_count_text}</div>{alerts_list_html}</div>""", unsafe_allow_html=True)

        st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)

        # Individual Symbol Cards
        if not selected_symbols:
            st.info("Select at least one symbol from the sidebar to view RSI data.")
        else:
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
                            if prepost:
                                price_df = data_fetcher.fetch(symbol, interval="5m", period="1d", prepost=True)
                            else:
                                price_df = data_fetcher.fetch(symbol, interval="1d", period="5d")
                            current_price = float(price_df["Close"].iloc[-1]) if price_df is not None and not price_df.empty else None
                            price_str = f"${current_price:.2f}" if current_price else "N/A"

                            rsi_data = rsi_results.get(symbol, {})
                            oversold_count, overbought_count, total = calculate_confluence(
                                rsi_data, oversold_threshold, overbought_threshold
                            )

                            strength_display = f"{strength:.0%}" if isinstance(strength, float) else str(strength)

                            vix_cooldown = signal_data.get("vix_cooldown_active", False)
                            recovery_suppressed = signal_data.get("recovery_suppressed", False)
                            recovery_type = signal_data.get("recovery_suppressed_signal")
                            cooldown_html = ""
                            if recovery_suppressed and recovery_type == 'sell':
                                cooldown_html = '<span style="display: inline-block; padding: 2px 8px; border-radius: 12px; background-color: #00C85122; color: #00C851; font-size: 0.7em; font-weight: 500; margin-left: 6px;">RECOVERY RALLY BUY</span>'
                            elif recovery_suppressed:
                                cooldown_html = '<span style="display: inline-block; padding: 2px 8px; border-radius: 12px; background-color: #ff980022; color: #ff9800; font-size: 0.7em; font-weight: 500; margin-left: 6px;">(recovery mode)</span>'
                            elif vix_cooldown:
                                cooldown_html = '<span style="display: inline-block; padding: 2px 8px; border-radius: 12px; background-color: #ffeb3b22; color: #ffeb3b; font-size: 0.7em; font-weight: 500; margin-left: 6px;">(VIX cooldown)</span>'

                            weekly_extreme = signal_data.get("weekly_extreme_buy", False)
                            weekly_extreme_html = ""
                            if weekly_extreme:
                                weekly_extreme_html = '<span style="display: inline-block; padding: 2px 8px; border-radius: 12px; background-color: #00C85122; color: #00C851; font-size: 0.7em; font-weight: 500; margin-left: 6px;">WEEKLY EXTREME BUY</span>'

                            channel_confluence = signal_data.get("channel_confluence", False)
                            channel_details = signal_data.get("channel_details", {})
                            channel_html = ""
                            if channel_details.get("in_valid_channel", False):
                                if channel_confluence:
                                    best_tf = None
                                    best_r2 = 0
                                    for tf, td in channel_details.get("timeframes", {}).items():
                                        if td.get("meets_criteria") and td.get("r_squared", 0) > best_r2:
                                            best_r2 = td["r_squared"]
                                            best_tf = tf
                                    win_key = (best_tf, 'rsi_channel_buy') if 'BUY' in signal else (best_tf, 'rsi_channel_buy')
                                    win_rate = CHANNEL_WIN_RATES.get(win_key, 0)
                                    pf_key = (best_tf, 'rsi_channel_pf')
                                    pf = CHANNEL_WIN_RATES.get(pf_key, 0)
                                    channel_html = f'<span style="display: inline-block; padding: 2px 8px; border-radius: 12px; background-color: #3A82FF22; color: #3A82FF; font-size: 0.7em; font-weight: 500; margin-left: 6px;">CHANNEL CONFLUENCE ({win_rate:.0f}% win)</span>'
                                else:
                                    channel_html = '<span style="display: inline-block; padding: 2px 8px; border-radius: 12px; background-color: #3A82FF11; color: #3A82FF88; font-size: 0.7em; font-weight: 500; margin-left: 6px;">in channel</span>'

                            st.markdown(f"""<div style="background-color: #1a1c23; border: 1px solid #2d3039; border-radius: 12px; padding: 0; margin-bottom: 16px; overflow: hidden;"><div style="height: 3px; background-color: {signal_color};"></div><div style="padding: 20px;"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;"><div style="display: flex; align-items: baseline; gap: 12px;"><span style="font-size: 1.3em; font-weight: 700; color: #e1e3ea;">{symbol}</span><span style="font-size: 1.15em; font-weight: 600; color: #8b8fa3;">{price_str}</span></div><div style="display: flex; align-items: center; flex-wrap: wrap;"><span style="display: inline-block; padding: 4px 12px; border-radius: 20px; background-color: {signal_color}33; color: {signal_color}; font-size: 0.8em; font-weight: 600; letter-spacing: 0.3px;">{SIGNAL_DISPLAY.get(signal, signal)}</span>{cooldown_html}{weekly_extreme_html}{channel_html}</div></div><div style="display: flex; gap: 10px; margin-bottom: 6px;"><div style="flex: 1; background-color: #14161b; border: 1px solid #2d3039; border-radius: 8px; padding: 10px 12px; text-align: center;"><div style="font-size: 0.7em; color: #6b7080; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Oversold TFs</div><div style="font-size: 1.1em; font-weight: 600; color: #e1e3ea;">{oversold_count}/{total}</div></div><div style="flex: 1; background-color: #14161b; border: 1px solid #2d3039; border-radius: 8px; padding: 10px 12px; text-align: center;"><div style="font-size: 0.7em; color: #6b7080; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Overbought TFs</div><div style="font-size: 1.1em; font-weight: 600; color: #e1e3ea;">{overbought_count}/{total}</div></div><div style="flex: 1; background-color: #14161b; border: 1px solid #2d3039; border-radius: 8px; padding: 10px 12px; text-align: center;"><div style="font-size: 0.7em; color: #6b7080; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Strength</div><div style="font-size: 1.1em; font-weight: 600; color: #e1e3ea;">{strength_display}</div></div></div></div></div>""", unsafe_allow_html=True)

                            # RSI Table
                            if rsi_data:
                                df = create_rsi_table(rsi_data, oversold_threshold, overbought_threshold)

                                def color_status(val):
                                    base_status = val.split(" (")[0] if " (" in val else val
                                    color = RSI_LEVEL_COLORS.get(base_status)
                                    if color:
                                        return f"background-color: {color}15; color: {color}; font-weight: 600"
                                    return ""

                                styled_df = df.style.map(
                                    color_status, subset=["Status"]
                                )

                                st.dataframe(
                                    styled_df,
                                    width='stretch',
                                    hide_index=True
                                )

                                if symbol == bounce_symbol and channel_details.get("timeframes"):
                                    with st.expander("Channel Analysis", expanded=False):
                                        ch_tfs = channel_details.get("timeframes", {})
                                        if ch_tfs:
                                            ch_rows = []
                                            for tf in ['5m', '15m', '1h', '4h']:
                                                td = ch_tfs.get(tf)
                                                if td:
                                                    pos = td.get('position', 0.5)
                                                    pos_label = "Near Lower" if pos < 0.25 else "Near Upper" if pos > 0.75 else "Middle"
                                                    ch_rows.append({
                                                        "Timeframe": tf,
                                                        "R²": f"{td.get('r_squared', 0):.3f}",
                                                        "Direction": td.get('direction', '?').title(),
                                                        "Position": f"{pos:.0%}",
                                                        "Zone": pos_label,
                                                        "Width": f"{td.get('width_pct', 0):.1f}%",
                                                        "Age": td.get('age', 0),
                                                        "Valid": "Yes" if td.get('meets_criteria') else "No",
                                                    })
                                            if ch_rows:
                                                st.dataframe(pd.DataFrame(ch_rows), hide_index=True, width='stretch')
                                            else:
                                                st.caption("No valid channels detected on any timeframe")
                                        else:
                                            st.caption("No channel data available")
                            else:
                                st.warning("No RSI data available")

    # === Channel Bounce Tab ===
    with tabs[1]:
        render_bounce_tab(bounce_assessment, bounce_channel_context, bounce_symbol, all_symbols, bounce_ohlcv_data)

    # Footer
    st.divider()
    st.caption(f"RSI Period: {rsi_period} | Oversold: <{oversold_threshold} | Overbought: >{overbought_threshold}")

    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(300)
        st.rerun()


def _render_channel_chart(df, channel, tf, symbol):
    """Render a plotly chart showing price with channel bands."""
    import plotly.graph_objects as go

    if df is None or df.empty:
        st.caption("No data available")
        return

    # Use last N bars for display
    display_bars = min(80, len(df))
    df_plot = df.tail(display_bars).copy()

    slope = channel.get('slope', 0)
    intercept = channel.get('intercept', 0)
    std_dev = channel.get('std_dev', 0)
    lookback = min(50, len(df))

    # Compute channel lines over the displayed range
    # The regression was fit on the last `lookback` bars of the full data
    # We need to map display indices to regression indices
    full_len = len(df)
    display_start = full_len - display_bars
    reg_start = full_len - lookback  # where regression x=0 starts

    x_vals = list(range(display_bars))
    mid_line = []
    upper_line = []
    lower_line = []
    for i in range(display_bars):
        reg_x = (display_start + i) - reg_start  # regression-relative x
        mid = slope * reg_x + intercept
        mid_line.append(mid)
        upper_line.append(mid + 2 * std_dev)
        lower_line.append(mid - 2 * std_dev)

    fig = go.Figure()

    # Channel band fill
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=upper_line, mode='lines',
        line=dict(color='rgba(255, 165, 0, 0.4)', width=1, dash='dash'),
        name='Upper', showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=lower_line, mode='lines',
        line=dict(color='rgba(255, 165, 0, 0.4)', width=1, dash='dash'),
        name='Lower', fill='tonexty', fillcolor='rgba(255, 165, 0, 0.05)',
        showlegend=False,
    ))

    # Midline
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=mid_line, mode='lines',
        line=dict(color='rgba(100, 149, 237, 0.5)', width=1, dash='dot'),
        name='Midline', showlegend=False,
    ))

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'],
        increasing_line_color='#00C851', decreasing_line_color='#ff4444',
        increasing_fillcolor='#00C851', decreasing_fillcolor='#ff4444',
        name=symbol, showlegend=False,
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#14161b',
        plot_bgcolor='#14161b',
        xaxis=dict(
            showgrid=False, zeroline=False,
            color='#6b7080', rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#1e2028', zeroline=False,
            color='#6b7080', side='right',
        ),
        font=dict(color='#8b8fa3', size=10),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_bounce_tab(assessment, channel_context, symbol, all_symbols, ohlcv_data=None):
    """Render the Channel Bounce tab for the selected symbol."""
    # Symbol selector — exclude VIX (^VIX) since it's not a tradeable equity channel
    equity_symbols = [s for s in all_symbols if not s.startswith("^")]
    if not equity_symbols:
        st.info("No equity symbols selected. Add at least one symbol to use Channel Bounce.")
        return

    current_idx = equity_symbols.index(symbol) if symbol in equity_symbols else 0
    selected = st.selectbox(
        "Symbol",
        options=equity_symbols,
        index=current_idx,
        key="bounce_symbol_select",
    )
    if selected != st.session_state.bounce_symbol:
        st.session_state.bounce_symbol = selected
        st.rerun()

    if assessment is None:
        st.info(f"No active channels detected for {symbol}. Channel bounce analysis requires at least one valid channel.")
        return

    # Score color
    score = assessment.score
    if score >= 65:
        score_color = "#00C851"
    elif score >= 40:
        score_color = "#ffeb3b"
    elif score >= 20:
        score_color = "#ff9800"
    else:
        score_color = "#ff4444"

    # Direction display
    dir_map = {
        'bounce_likely': ('Bounce Likely', '#00C851'),
        'break_likely': ('Break Likely', '#ff4444'),
        'neutral': ('Neutral', '#6c757d'),
    }
    dir_label, dir_color = dir_map.get(assessment.direction, ('Unknown', '#6c757d'))

    # Break direction display
    break_map = {
        'down': ('Down', '#ff4444'),
        'up': ('Up', '#00C851'),
        'unknown': ('Unknown', '#6c757d'),
    }
    break_label, break_color = break_map.get(assessment.predicted_break_dir, ('Unknown', '#6c757d'))

    # --- Score Card ---
    st.markdown(f"""<div style="background-color: #1a1c23; border: 1px solid #2d3039; border-radius: 12px; padding: 0; margin-bottom: 16px; overflow: hidden;"><div style="height: 3px; background-color: {score_color};"></div><div style="padding: 20px;"><div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;"><span style="font-size: 1.6em; font-weight: 700; color: #e0e0e0; margin-right: 4px;">{symbol}</span><div style="display: flex; align-items: baseline; gap: 8px;"><span style="font-size: 2em; font-weight: 700; color: {score_color};">{score:.0f}</span><span style="font-size: 1em; color: #6b7080;">/100</span></div><span style="display: inline-block; padding: 4px 12px; border-radius: 20px; background-color: {score_color}22; color: {score_color}; font-size: 0.85em; font-weight: 600;">{assessment.confidence_label.upper()}</span><span style="display: inline-block; padding: 4px 12px; border-radius: 20px; background-color: {dir_color}22; color: {dir_color}; font-size: 0.85em; font-weight: 600;">{dir_label}</span><span style="display: inline-block; padding: 4px 12px; border-radius: 20px; background-color: {break_color}22; color: {break_color}; font-size: 0.85em; font-weight: 600;">Break: {break_label}</span><span style="color: #6b7080; font-size: 0.85em;">{assessment.active_channels} active channel{'s' if assessment.active_channels != 1 else ''}</span></div></div></div>""", unsafe_allow_html=True)

    # --- Channel Cards with Position Gauge ---
    st.markdown('<h4 style="margin: 0.8rem 0 0.4rem 0; font-weight: 600; color: #e0e0e0;">Channels</h4>', unsafe_allow_html=True)

    for tf in ['5m', '15m', '1h', '4h']:
        ch_data = channel_context.get(tf, {})
        ch = ch_data.get('channel', {}) if isinstance(ch_data, dict) else {}
        valid = ch.get('valid', False)
        r2 = ch.get('r_squared', 0)
        direction = ch.get('direction', 'N/A').title()
        pos = ch.get('position', 0.5)
        upper_band = ch.get('upper_band', 0)
        lower_band = ch.get('lower_band', 0)
        midline = ch.get('midline', 0)
        age = ch.get('age', 0)

        if not valid:
            st.markdown(f"""<div style="background-color: #14161b; border: 1px solid #2d3039; border-radius: 8px; padding: 12px; margin-bottom: 8px; display: flex; align-items: center; gap: 16px;"><span style="font-size: 0.85em; font-weight: 600; color: #555; min-width: 36px;">{tf.upper()}</span><span style="font-size: 0.8em; color: #555;">No valid channel (R² {r2:.3f})</span></div>""", unsafe_allow_html=True)
            continue

        # Position-based signal
        if pos < 0.2:
            signal_text, signal_color = "BUY ZONE", "#00C851"
        elif pos < 0.35:
            signal_text, signal_color = "Near Support", "#4CAF50"
        elif pos > 0.8:
            signal_text, signal_color = "SELL ZONE", "#ff4444"
        elif pos > 0.65:
            signal_text, signal_color = "Near Resistance", "#ff9800"
        else:
            signal_text, signal_color = "Mid-Channel", "#6c757d"

        # Gauge: position bar with green (lower) → yellow (mid) → red (upper)
        pos_pct = max(0, min(100, pos * 100))
        price_display = f"${midline:,.2f}" if midline > 0 else ""

        st.markdown(f"""<div style="background-color: #14161b; border: 1px solid #2d3039; border-radius: 8px; padding: 14px 16px; margin-bottom: 8px;"><div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;"><div style="display: flex; align-items: center; gap: 12px;"><span style="font-size: 0.9em; font-weight: 700; color: #e1e3ea; min-width: 36px;">{tf.upper()}</span><span style="font-size: 0.75em; color: #8b8fa3;">{direction} · R² {r2:.3f} · Age {age}</span></div><span style="display: inline-block; padding: 3px 10px; border-radius: 12px; background-color: {signal_color}22; color: {signal_color}; font-size: 0.8em; font-weight: 600;">{signal_text}</span></div><div style="position: relative; height: 24px; background: linear-gradient(to right, #1a472a, #1a3a1a 20%, #2d2d1a 40%, #3a2a1a 70%, #3a1a1a); border-radius: 12px; overflow: visible; border: 1px solid #2d3039;"><div style="position: absolute; left: {pos_pct}%; top: -2px; transform: translateX(-50%); width: 14px; height: 28px; background-color: #e1e3ea; border-radius: 7px; border: 2px solid {signal_color}; box-shadow: 0 0 6px {signal_color}88;"></div></div><div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 0.7em; color: #6b7080;"><span>${lower_band:,.2f}</span><span>{price_display}</span><span>${upper_band:,.2f}</span></div></div>""", unsafe_allow_html=True)

        # Expandable plotly chart
        if ohlcv_data and tf in ohlcv_data:
            with st.expander(f"{tf.upper()} Chart", expanded=False):
                _render_channel_chart(ohlcv_data[tf], ch, tf, symbol)

    # --- Timing ---
    try:
        import pytz
        now_et = datetime.now(pytz.timezone('US/Eastern'))
        time_str = now_et.strftime('%H:%M ET')
        day_str = now_et.strftime('%A')
        minutes_since_open = now_et.hour * 60 + now_et.minute - (9 * 60 + 30)
        if 0 <= minutes_since_open <= 30:
            window = "Open 30min"
        elif 360 <= minutes_since_open <= 390:
            window = "Close 30min"
        elif 0 <= minutes_since_open <= 390:
            window = "Regular Hours"
        else:
            window = "After Hours"
    except Exception:
        time_str = "N/A"
        day_str = "N/A"
        window = "N/A"

    st.markdown(f"""<div style="background-color: #14161b; border: 1px solid #2d3039; border-radius: 8px; padding: 10px 16px; margin: 12px 0; display: flex; gap: 24px;"><span style="color: #6b7080; font-size: 0.85em;">Time: <strong style="color: #e1e3ea;">{time_str}</strong></span><span style="color: #6b7080; font-size: 0.85em;">Day: <strong style="color: #e1e3ea;">{day_str}</strong></span><span style="color: #6b7080; font-size: 0.85em;">Window: <strong style="color: #e1e3ea;">{window}</strong></span></div>""", unsafe_allow_html=True)

    # --- Rules Breakdown ---
    st.markdown('<h4 style="margin: 0.8rem 0 0.4rem 0; font-weight: 600; color: #e0e0e0;">Rules</h4>', unsafe_allow_html=True)

    for category, cat_label in [('core', 'Core Rules'), ('modifier', 'Modifier Rules'), ('avoid', 'Avoid Rules')]:
        cat_rules = [r for r in assessment.rules if r.category == category]
        if not cat_rules:
            continue

        st.markdown(f'<div style="font-size: 0.8em; color: #6b7080; text-transform: uppercase; letter-spacing: 0.5px; margin: 12px 0 6px 0;">{cat_label}</div>', unsafe_allow_html=True)

        for rule in cat_rules:
            if rule.fired:
                if rule.points > 0:
                    icon = "&#x2705;"  # green check
                    pts_color = "#00C851"
                else:
                    icon = "&#x274C;"  # red X
                    pts_color = "#ff4444"
            else:
                icon = "&#x2796;"  # gray dash
                pts_color = "#555"

            pts_str = f"{rule.points:+d}" if rule.fired else "—"
            wr_str = f"{rule.win_rate:.0f}%" if rule.win_rate > 0 else ""

            st.markdown(f"""<div style="display: flex; align-items: center; gap: 10px; padding: 6px 12px; margin: 2px 0; background-color: #14161b; border-radius: 6px;"><span style="font-size: 0.9em; width: 20px; text-align: center;">{icon}</span><span style="flex: 1; font-size: 0.85em; color: #e1e3ea;">{rule.name}</span><span style="font-size: 0.75em; color: #8b8fa3; min-width: 40px; text-align: right;">{wr_str}</span><span style="font-size: 0.85em; font-weight: 600; color: {pts_color}; min-width: 36px; text-align: right;">{pts_str}</span></div>""", unsafe_allow_html=True)

        # Show description for fired rules in an expander
        fired_in_cat = [r for r in cat_rules if r.fired]
        if fired_in_cat:
            with st.expander(f"{cat_label} Details", expanded=False):
                for rule in fired_in_cat:
                    st.caption(f"**{rule.name}**: {rule.description}")


if __name__ == "__main__":
    main()
