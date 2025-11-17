"""GUI dashboard using Streamlit."""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path for config
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

import config
from src.data_handler import DataHandler
from src.linear_regression import LinearRegressionChannel
from src.rsi_calculator import RSICalculator
from src.news_analyzer import NewsAnalyzer
from src.signal_generator import SignalGenerator


# Page config
st.set_page_config(
    page_title="Linear Regression Trading System",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Linear Regression Channel Trading System")


# Sidebar
st.sidebar.header("Settings")
stock = st.sidebar.selectbox("Stock", config.STOCKS, index=0)
timeframe = st.sidebar.selectbox(
    "Primary Timeframe",
    ["1hour", "4hour", "daily"],
    index=1
)

refresh_button = st.sidebar.button("🔄 Refresh Data")

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")


# Cache data for performance
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(stock_symbol):
    """Load and cache stock data."""
    handler = DataHandler(stock_symbol)
    return handler.get_all_timeframes()


@st.cache_resource
def get_components(stock_symbol):
    """Get analysis components."""
    return {
        'data_handler': DataHandler(stock_symbol),
        'channel_calc': LinearRegressionChannel(),
        'rsi_calc': RSICalculator(),
        'news_analyzer': NewsAnalyzer(stock_symbol),
        'signal_gen': SignalGenerator(stock_symbol)
    }


# Load data
try:
    with st.spinner(f"Loading data for {stock}..."):
        data_dict = load_data(stock)
        components = get_components(stock)

        st.sidebar.success(f"✓ Data loaded: {stock}")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Price Chart with Linear Regression Channels")

    # Get data for selected timeframe
    df = data_dict[timeframe]
    current_price = df['close'].iloc[-1]

    # Calculate channel
    channel = components['channel_calc'].calculate_channel(df)

    # Create candlestick chart with channels
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{stock} - {timeframe}", "RSI")
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ),
        row=1, col=1
    )

    # Channel lines
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=channel.upper_line,
            name="Upper Channel",
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=channel.center_line,
            name="Center Line",
            line=dict(color='blue', dash='dot')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=channel.lower_line,
            name="Lower Channel",
            line=dict(color='green', dash='dash')
        ),
        row=1, col=1
    )

    # RSI
    rsi_series = components['rsi_calc'].calculate_rsi(df)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=rsi_series,
            name="RSI",
            line=dict(color='purple')
        ),
        row=2, col=1
    )

    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)

    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


with col2:
    st.header("Current Signal")

    # Generate signal
    try:
        with st.spinner("Analyzing..."):
            signal = components['signal_gen'].generate_signal(timeframe)

        # Display signal
        if signal.signal_type == "buy":
            st.success(f"🟢 BUY SIGNAL")
        elif signal.signal_type == "sell":
            st.error(f"🔴 SELL SIGNAL")
        else:
            st.info(f"⚪ NEUTRAL")

        st.metric("Confidence Score", f"{signal.confidence_score:.1f}/100")
        st.metric("Current Price", f"${signal.current_price:.2f}")

        # Channel info
        st.subheader("Channel Analysis")
        st.metric("Position", signal.channel_position['zone'])
        st.metric("Stability", f"{signal.channel_stability:.0f}/100")
        st.metric("Ping-Pongs", f"{signal.rsi_confluence.get('confirming_timeframes', [])}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Predicted High", f"${signal.predicted_high:.2f}")
        with col_b:
            st.metric("Predicted Low", f"${signal.predicted_low:.2f}")

        # RSI info
        st.subheader("RSI Analysis")
        st.metric("Primary RSI", f"{signal.primary_rsi:.1f}")
        st.metric("RSI Score", f"{signal.rsi_confluence['score']:.0f}/100")
        st.metric("Confirmations", len(signal.rsi_confluence['confirming_timeframes']))

        # Trade levels
        if signal.signal_type != "neutral":
            st.subheader("Trade Levels")
            st.metric("Entry", f"${signal.entry_price:.2f}")
            st.metric("Target", f"${signal.target_price:.2f}",
                     delta=f"{((signal.target_price/signal.entry_price)-1)*100:+.1f}%")
            st.metric("Stop Loss", f"${signal.stop_loss:.2f}",
                     delta=f"{((signal.stop_loss/signal.entry_price)-1)*100:+.1f}%")

        # Reasoning
        st.subheader("Reasoning")
        st.info(signal.reasoning)

    except Exception as e:
        st.error(f"Error generating signal: {e}")


# Multi-timeframe RSI
st.header("Multi-Timeframe RSI")

rsi_dict = components['rsi_calc'].analyze_multiple_timeframes(data_dict)

rsi_cols = st.columns(len(rsi_dict))
for i, (tf, rsi_data) in enumerate(rsi_dict.items()):
    with rsi_cols[i]:
        if rsi_data.oversold:
            st.success(f"**{tf}**\n\n{rsi_data.value:.1f}\n\nOVERSOLD")
        elif rsi_data.overbought:
            st.error(f"**{tf}**\n\n{rsi_data.value:.1f}\n\nOVERBOUGHT")
        else:
            st.info(f"**{tf}**\n\n{rsi_data.value:.1f}\n\n{rsi_data.signal}")


# News panel
st.header("📰 News Analysis")

try:
    with st.spinner("Fetching and analyzing news..."):
        # Fetch and analyze news
        stock_context = f"{stock} trading at ${current_price:.2f}"
        components['news_analyzer'].fetch_news(hours_back=24)
        articles = components['news_analyzer'].analyze_all_news(stock_context)

        if articles:
            # Overall sentiment
            overall = components['news_analyzer'].get_overall_sentiment()

            col_n1, col_n2, col_n3, col_n4 = st.columns(4)
            with col_n1:
                st.metric("Avg Sentiment", f"{overall['avg_sentiment_score']:+.0f}")
            with col_n2:
                st.metric("Avg BS Score", f"{overall['avg_bs_score']:.0f}/100")
            with col_n3:
                st.metric("Signal", overall['signal'])
            with col_n4:
                st.metric("High BS Count", f"{overall['high_bs_count']}/{overall['total_articles']}")

            st.info(f"**Recommendation:** {overall['recommendation']}")

            # Show articles
            st.subheader("Recent Articles")

            for article in articles[:5]:  # Show top 5
                with st.expander(f"📄 {article.title} ({article.source})"):
                    st.write(f"**Published:** {article.published_at.strftime('%Y-%m-%d %H:%M')}")

                    col_art1, col_art2, col_art3 = st.columns(3)
                    with col_art1:
                        sentiment_color = "green" if article.sentiment == "positive" else "red" if article.sentiment == "negative" else "gray"
                        st.markdown(f"**Sentiment:** :{sentiment_color}[{article.sentiment}] ({article.sentiment_score:+.0f})")
                    with col_art2:
                        bs_color = "red" if article.bs_score > 70 else "orange" if article.bs_score > 40 else "green"
                        st.markdown(f"**BS Score:** :{bs_color}[{article.bs_score:.0f}/100]")
                    with col_art3:
                        if article.url:
                            st.markdown(f"[Read More]({article.url})")

                    st.write(f"**Description:** {article.description}")
                    st.write(f"**AI Analysis:** {article.analysis}")

        else:
            st.warning("No news articles found.")

except Exception as e:
    st.error(f"Error fetching news: {e}")


# Footer
st.markdown("---")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
st.markdown("*Data updates every 5 minutes. Use Refresh button for immediate update.*")
