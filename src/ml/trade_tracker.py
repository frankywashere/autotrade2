"""
Trade Tracker for High-Confidence Predictions

Logs high-confidence trading signals with full rationale including:
- Channel position (where in channel we're buying/selling)
- Multi-timeframe RSI levels
- SPY correlation and alignment
- Prediction details and validation tracking

Database: high_confidence_trades.db
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))


class TradeTracker:
    """
    Tracks high-confidence trading signals.

    Logs predictions that exceed confidence threshold along with:
    - Trade rationale (why this is a good setup)
    - Channel context (position, slope, ping-pongs)
    - RSI levels across timeframes
    - SPY correlation
    - Actual outcomes for validation
    """

    def __init__(
        self,
        db_path: str = 'data/high_confidence_trades.db',
        confidence_threshold: float = 0.75,
        auto_create_schema: bool = True
    ):
        """
        Initialize trade tracker.

        Args:
            db_path: Path to database file
            confidence_threshold: Minimum confidence to log (0-1)
            auto_create_schema: Create schema if doesn't exist
        """
        self.db_path = db_path
        self.confidence_threshold = confidence_threshold

        # Create database directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return dicts instead of tuples

        if auto_create_schema:
            self.create_schema()

    def create_schema(self):
        """Create database schema for trade tracking."""
        cursor = self.conn.cursor()

        # Main trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                model_type VARCHAR(20) NOT NULL,
                confidence FLOAT NOT NULL,
                confidence_threshold FLOAT NOT NULL,

                -- Predictions (percentage changes)
                predicted_high FLOAT NOT NULL,
                predicted_low FLOAT NOT NULL,
                current_price FLOAT NOT NULL,
                predicted_high_price FLOAT NOT NULL,
                predicted_low_price FLOAT NOT NULL,

                -- Trade Rationale (JSON string)
                rationale TEXT,

                -- Channel Context
                channel_timeframe VARCHAR(10),
                channel_position FLOAT,  -- -1 (bottom) to +1 (top)
                channel_slope FLOAT,
                ping_pong_count INT,
                time_in_channel INT,

                -- Multi-timeframe RSI
                rsi_15min FLOAT,
                rsi_1hour FLOAT,
                rsi_4hour FLOAT,
                rsi_daily FLOAT,
                rsi_confluence BOOLEAN,  -- All aligned (all low or all high)

                -- SPY Context
                spy_correlation FLOAT,
                spy_channel_position FLOAT,
                spy_rsi FLOAT,
                spy_tsla_aligned BOOLEAN,  -- Both at similar channel positions

                -- Layer weights (which layer is most confident)
                fast_layer_weight FLOAT,
                medium_layer_weight FLOAT,
                slow_layer_weight FLOAT,

                -- Trade Plan (NEW - Phase 1)
                stop_price FLOAT,                   -- Calculated stop loss price
                stop_pct FLOAT,                     -- Stop as % from entry
                max_hold_time_minutes INT,          -- Max duration for trade

                -- Actuals (filled later)
                actual_high FLOAT,
                actual_low FLOAT,
                trade_outcome VARCHAR(20),  -- 'hit_target', 'partial', 'stopped_out', 'pending'
                return_percentage FLOAT,

                -- Validation Metrics (NEW - Phase 1, filled during validation)
                hit_band BOOLEAN,                   -- Did price enter predicted band?
                hit_target_before_stop BOOLEAN,     -- Reached target before stop?
                overshoot_ratio FLOAT,              -- How far outside band (0-1+)
                time_to_target_minutes INT,         -- How long to reach target
                max_adverse_excursion_pct FLOAT,    -- Worst drawdown during trade

                -- Online Learning
                triggered_update BOOLEAN DEFAULT FALSE,
                layers_updated TEXT,  -- JSON: ["fast", "medium"]
                validation_time DATETIME,
                validated BOOLEAN DEFAULT FALSE,

                -- Metadata
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for quick lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp
            ON trades(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_validated
            ON trades(validated)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_confidence
            ON trades(confidence)
        """)

        # Online updates table (tracks when model learns from errors)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS online_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                trade_id INTEGER,
                layer VARCHAR(20) NOT NULL,
                error_high FLOAT NOT NULL,
                error_low FLOAT NOT NULL,
                learning_rate FLOAT NOT NULL,
                weight_delta_l2_norm FLOAT,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        """)

        self.conn.commit()

    def log_trade(
        self,
        timestamp: datetime,
        model_type: str,
        confidence: float,
        predicted_high: float,
        predicted_low: float,
        current_price: float,
        features_dict: Dict[str, float],
        layer_weights: Optional[List[float]] = None,
        validation_minutes: int = 30
    ) -> int:
        """
        Log a high-confidence trade.

        Args:
            timestamp: Current timestamp
            model_type: 'hierarchical' or 'ensemble'
            confidence: Prediction confidence (0-1)
            predicted_high: Predicted high (%)
            predicted_low: Predicted low (%)
            current_price: Current TSLA price
            features_dict: Dictionary of current features
            layer_weights: [fast, medium, slow] weights
            validation_minutes: Minutes until validation

        Returns:
            trade_id: Database ID of logged trade
        """
        # Only log if confidence exceeds threshold
        if confidence < self.confidence_threshold:
            return None

        # Calculate absolute prices
        predicted_high_price = current_price * (1 + predicted_high / 100.0)
        predicted_low_price = current_price * (1 + predicted_low / 100.0)

        # Extract channel context (prefer 1hour for swing trades)
        channel_timeframe = '1h'
        channel_position = features_dict.get(f'tsla_channel_position_norm_{channel_timeframe}', 0.0)
        channel_slope = features_dict.get(f'tsla_channel_{channel_timeframe}_slope', 0.0)
        ping_pongs = features_dict.get(f'tsla_channel_{channel_timeframe}_ping_pongs', 0)
        time_in_channel = features_dict.get(f'tsla_time_in_channel_{channel_timeframe}', 0)

        # Extract RSI levels
        rsi_15min = features_dict.get('tsla_rsi_15min', 50.0)
        rsi_1hour = features_dict.get('tsla_rsi_1h', 50.0)
        rsi_4hour = features_dict.get('tsla_rsi_4h', 50.0)
        rsi_daily = features_dict.get('tsla_rsi_daily', 50.0)

        # Check RSI confluence (all oversold or all overbought)
        rsi_confluence = (
            (rsi_15min < 35 and rsi_1hour < 35 and rsi_4hour < 35 and rsi_daily < 35) or
            (rsi_15min > 65 and rsi_1hour > 65 and rsi_4hour > 65 and rsi_daily > 65)
        )

        # Extract SPY context
        spy_correlation = features_dict.get('correlation_10', 0.0)
        spy_channel_pos = features_dict.get(f'spy_channel_position_norm_{channel_timeframe}', 0.0)
        spy_rsi = features_dict.get(f'spy_rsi_{channel_timeframe}', 50.0)

        # Check SPY-TSLA alignment
        spy_tsla_aligned = abs(channel_position - spy_channel_pos) < 0.3  # Within 30% range

        # Generate trade rationale
        rationale = self._generate_rationale(
            channel_position, predicted_high, predicted_low,
            rsi_15min, rsi_1hour, rsi_4hour, rsi_daily,
            spy_correlation, spy_channel_pos, rsi_confluence,
            spy_tsla_aligned, ping_pongs
        )

        # Layer weights
        if layer_weights is None:
            layer_weights = [1/3, 1/3, 1/3]

        # Calculate stop price (2% below predicted low or 2x volatility)
        volatility = features_dict.get('tsla_volatility_10', 0.02)  # Default 2%
        stop_loss_multiplier = 2.0  # 2x volatility
        atr_estimate = volatility * current_price

        # Stop price calculation
        if predicted_low < 0:
            # Expecting downside - stop below predicted low
            stop_price = predicted_low_price * (1 - stop_loss_multiplier * volatility)
        else:
            # Expecting upside - use default 2% stop
            stop_price = current_price * 0.98

        stop_pct = (stop_price - current_price) / current_price * 100.0

        # Max hold time based on timeframe (inferred from layer weights)
        # Dominant layer determines hold time
        dominant_layer = ['fast', 'medium', 'slow'][np.argmax(layer_weights)]
        max_hold_time_map = {
            'fast': 120,    # 2 hours
            'medium': 480,  # 8 hours
            'slow': 1440    # 1 day
        }
        max_hold_time = max_hold_time_map[dominant_layer]

        # Validation time
        validation_time = timestamp + timedelta(minutes=validation_minutes)

        # Insert into database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trades (
                timestamp, model_type, confidence, confidence_threshold,
                predicted_high, predicted_low, current_price,
                predicted_high_price, predicted_low_price,
                rationale,
                channel_timeframe, channel_position, channel_slope,
                ping_pong_count, time_in_channel,
                rsi_15min, rsi_1hour, rsi_4hour, rsi_daily, rsi_confluence,
                spy_correlation, spy_channel_position, spy_rsi, spy_tsla_aligned,
                fast_layer_weight, medium_layer_weight, slow_layer_weight,
                stop_price, stop_pct, max_hold_time_minutes,
                validation_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, model_type, confidence, self.confidence_threshold,
            predicted_high, predicted_low, current_price,
            predicted_high_price, predicted_low_price,
            json.dumps(rationale),
            channel_timeframe, channel_position, channel_slope,
            int(ping_pongs), int(time_in_channel),
            rsi_15min, rsi_1hour, rsi_4hour, rsi_daily, rsi_confluence,
            spy_correlation, spy_channel_pos, spy_rsi, spy_tsla_aligned,
            layer_weights[0], layer_weights[1], layer_weights[2],
            stop_price, stop_pct, max_hold_time,
            validation_time
        ))

        self.conn.commit()
        trade_id = cursor.lastrowid

        return trade_id

    def _generate_rationale(
        self,
        channel_pos: float,
        pred_high: float,
        pred_low: float,
        rsi_15min: float,
        rsi_1hour: float,
        rsi_4hour: float,
        rsi_daily: float,
        spy_corr: float,
        spy_pos: float,
        rsi_confluence: bool,
        spy_aligned: bool,
        ping_pongs: int
    ) -> Dict[str, Any]:
        """Generate human-readable trade rationale."""

        rationale = {
            'setup_type': None,
            'reasons': [],
            'risk_level': 'medium'
        }

        # Determine setup type based on channel position
        if channel_pos < -0.5:
            rationale['setup_type'] = 'Buy at channel bottom'
            rationale['reasons'].append(f'Price near channel bottom (position: {channel_pos:.2f})')
            rationale['risk_level'] = 'low' if rsi_confluence else 'medium'
        elif channel_pos > 0.5:
            rationale['setup_type'] = 'Sell at channel top'
            rationale['reasons'].append(f'Price near channel top (position: {channel_pos:.2f})')
            rationale['risk_level'] = 'low' if rsi_confluence else 'medium'
        else:
            rationale['setup_type'] = 'Mid-channel trade'
            rationale['reasons'].append(f'Price in channel middle (position: {channel_pos:.2f})')
            rationale['risk_level'] = 'medium'

        # RSI confluence
        if rsi_confluence:
            if rsi_daily < 35:
                rationale['reasons'].append(f'Strong RSI oversold across all timeframes (Daily: {rsi_daily:.1f})')
            else:
                rationale['reasons'].append(f'Strong RSI overbought across all timeframes (Daily: {rsi_daily:.1f})')

        # SPY alignment
        if spy_aligned:
            if abs(channel_pos - spy_pos) < 0.1:
                rationale['reasons'].append(f'Perfect SPY-TSLA alignment (both at {channel_pos:.2f})')
            else:
                rationale['reasons'].append(f'Good SPY-TSLA alignment (correlation: {spy_corr:.2f})')

        # Channel quality
        if ping_pongs >= 3:
            rationale['reasons'].append(f'Strong channel with {int(ping_pongs)} ping-pongs')

        # Expected return
        if pred_high > 0:
            rationale['expected_return_high'] = f'+{pred_high:.2f}%'
        else:
            rationale['expected_return_high'] = f'{pred_high:.2f}%'

        if pred_low > 0:
            rationale['expected_return_low'] = f'+{pred_low:.2f}%'
        else:
            rationale['expected_return_low'] = f'{pred_low:.2f}%'

        return rationale

    def update_actual(
        self,
        trade_id: int,
        actual_high: float,
        actual_low: float,
        price_sequence: Optional[np.ndarray] = None
    ):
        """
        Update trade with actual outcomes and validation metrics.

        Args:
            trade_id: Trade ID
            actual_high: Actual high achieved (%)
            actual_low: Actual low achieved (%)
            price_sequence: Optional tick-by-tick prices for sequential analysis
        """
        # Get trade
        trade = self.get_trade(trade_id)
        if trade is None:
            return

        # Calculate return
        pred_high = trade['predicted_high']
        pred_low = trade['predicted_low']
        current_price = trade['current_price']
        pred_high_price = trade['predicted_high_price']
        pred_low_price = trade['predicted_low_price']
        stop_price = trade['stop_price']

        # Determine outcome
        if actual_high >= pred_high * 0.8:  # Hit at least 80% of target
            outcome = 'hit_target'
            return_pct = actual_high
        elif actual_low <= pred_low * 1.2:  # Within 120% of predicted low
            outcome = 'partial'
            return_pct = actual_low
        else:
            outcome = 'stopped_out'
            return_pct = min(actual_high, actual_low, key=abs)

        # ===== Compute Validation Metrics =====

        # Metric 1: Hit Band (did price enter predicted range?)
        actual_high_price = current_price * (1 + actual_high / 100)
        actual_low_price = current_price * (1 + actual_low / 100)
        hit_band = (actual_low_price <= pred_high_price and actual_high_price >= pred_low_price)

        # Metric 2: Hit Target Before Stop (requires sequence)
        if price_sequence is not None:
            hit_target_before_stop = self._check_target_sequential(
                price_sequence, current_price, pred_high_price, stop_price
            )
        else:
            # Approximation: hit target and didn't hit stop
            hit_target_before_stop = (actual_high_price >= pred_high_price and
                                      actual_low_price >= stop_price)

        # Metric 3: Overshoot Ratio
        pred_range = abs(pred_high_price - pred_low_price)
        if pred_range > 0:
            overshoot_high_amt = max(0, actual_high_price - pred_high_price)
            overshoot_low_amt = max(0, pred_low_price - actual_low_price)
            overshoot_ratio = (overshoot_high_amt + overshoot_low_amt) / pred_range
        else:
            overshoot_ratio = 0.0

        # Update database with all metrics
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE trades
            SET actual_high = ?, actual_low = ?,
                trade_outcome = ?, return_percentage = ?,
                hit_band = ?, hit_target_before_stop = ?, overshoot_ratio = ?,
                validated = TRUE
            WHERE id = ?
        """, (actual_high, actual_low, outcome, return_pct,
              hit_band, hit_target_before_stop, overshoot_ratio,
              trade_id))

        self.conn.commit()

    def _check_target_sequential(
        self, prices: np.ndarray, entry: float, target: float, stop: float
    ) -> bool:
        """Check if target hit before stop."""
        for price in prices:
            if price >= target:
                return True
            if price <= stop:
                return False
        return False

    def get_trade(self, trade_id: int) -> Optional[Dict]:
        """Get trade by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def get_pending_validations(self, current_time: datetime) -> List[Dict]:
        """Get trades awaiting validation."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM trades
            WHERE validated = FALSE
              AND validation_time <= ?
            ORDER BY validation_time
        """, (current_time,))

        return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall trade statistics."""
        cursor = self.conn.cursor()

        # Total trades
        cursor.execute("SELECT COUNT(*) as count FROM trades")
        total_trades = cursor.fetchone()['count']

        # Win rate (validated trades only)
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN trade_outcome = 'hit_target' THEN 1 ELSE 0 END) as wins
            FROM trades
            WHERE validated = TRUE
        """)
        result = cursor.fetchone()
        validated_trades = result['total']
        wins = result['wins']
        win_rate = (wins / validated_trades * 100) if validated_trades > 0 else 0

        # Average return
        cursor.execute("""
            SELECT AVG(return_percentage) as avg_return
            FROM trades
            WHERE validated = TRUE AND return_percentage IS NOT NULL
        """)
        avg_return = cursor.fetchone()['avg_return'] or 0.0

        return {
            'total_trades': total_trades,
            'validated_trades': validated_trades,
            'pending_trades': total_trades - validated_trades,
            'wins': wins,
            'win_rate': win_rate,
            'average_return': avg_return
        }

    def close(self):
        """Close database connection."""
        self.conn.close()
