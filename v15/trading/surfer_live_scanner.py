"""
Surfer Live Scanner — Channel Surfer signal evaluation + position tracking.

No auto-trading. No broker integration. Dashboard notifications only.
Persistence via JSON at ~/.x14/surfer_scanner_state.json (local)
or GitHub Gist (when GIST_ID + GITHUB_TOKEN are provided, e.g. Streamlit Cloud).
"""

import json
import urllib.request
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

STATE_PATH = Path.home() / ".x14" / "surfer_scanner_state.json"
MAX_SIGNAL_HISTORY = 200
GIST_FILE_NAME = 'surfer_scanner_state.json'
MODEL_TAG = 'c11'  # Identifies this branch in the multi-model Gist

# Slippage + commission assumptions
SLIPPAGE_PCT = 0.0005     # 0.05% slippage per side
COMMISSION_PER_SHARE = 0.005  # $0.005/share (IBKR tiered)


@dataclass
class ScannerConfig:
    """User-configurable scanner parameters."""
    initial_capital: float = 100_000.0
    risk_per_trade_pct: float = 0.02     # 2% of capital risked per trade
    max_leverage: float = 4.0
    max_buying_power_pct: float = 0.25   # 25% of buying power per trade
    daily_loss_limit: float = -2000.0
    max_positions: int = 2
    kill_switch: bool = False
    min_confidence: float = 0.45


@dataclass
class HypotheticalPosition:
    """A hypothetical open position being tracked."""
    pos_id: str
    direction: str           # 'long' or 'short'
    entry_price: float
    entry_time: str          # ISO format
    shares: int
    notional: float
    stop_price: float
    tp_price: float
    signal_type: str         # 'bounce' or 'break'
    primary_tf: str
    confidence: float
    best_price: float        # For trailing stop
    reason: str = ''
    breakeven_applied: bool = False  # True once stop moved to entry after 30 min

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'HypotheticalPosition':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ClosedTrade:
    """Record of a closed hypothetical trade."""
    pos_id: str
    direction: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    entry_time: str
    exit_time: str
    exit_reason: str
    hold_minutes: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ClosedTrade':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ScannerAlert:
    """Structured output for dashboard rendering."""
    alert_type: str          # 'ENTRY', 'EXIT', 'RISK_WARNING'
    timestamp: str

    # Entry fields
    action: str = ''         # 'BUY' or 'SELL'
    price: float = 0.0
    shares: int = 0
    stop_price: float = 0.0
    tp_price: float = 0.0
    risk_reward: float = 0.0
    confidence: float = 0.0
    signal_type: str = ''
    primary_tf: str = ''
    reason: str = ''
    notional: float = 0.0

    # Exit fields
    pos_id: str = ''
    exit_reason: str = ''
    pnl: float = 0.0
    pnl_pct: float = 0.0

    # Warning fields
    warning_msg: str = ''


class SurferLiveScanner:
    """
    State machine for live Channel Surfer signal monitoring.

    Called on every dashboard refresh cycle. Evaluates the latest
    ChannelAnalysis, sizes positions, tracks hypothetical entries/exits.
    """

    TRAILING_STOP_PCT = 0.015   # 1.5% trailing from best price
    TIMEOUT_MINUTES = 300       # 5 hours
    EOD_CLOSE_HOUR_ET = 15      # Force-close at 3:45 PM ET
    EOD_CLOSE_MINUTE_ET = 45
    EQUITY_CEILING_PCT = 0.05   # Close if unrealized >= 5% of equity
    NEAR_TP_PCT = 0.01          # Close early if within 1% of TP price
    NEAR_TP_MIN_GAIN = 100.0    # Minimum dollar gain for near-TP early close
    NEAR_TP_WINDOW_MIN = 30     # Near-TP rule only active within first 30 min
    BREAKEVEN_TRIGGER_MIN = 30  # Move stop to entry after 30 min if in profit
    OUTLIER_MULT = 2.0          # Close if unrealized >= 2x expected TP gain

    def __init__(self, config: ScannerConfig, gist_id: str = '', github_token: str = ''):
        self.config = config
        self.gist_id = gist_id.strip()
        self.github_token = github_token.strip()
        self.equity = config.initial_capital
        self.positions: Dict[str, HypotheticalPosition] = {}
        self.closed_trades: List[ClosedTrade] = []
        self.signal_history: List[dict] = []
        self.daily_pnl: float = 0.0
        self.daily_trade_count: int = 0
        self._daily_date: str = ''
        self._load_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _gist_headers(self) -> dict:
        return {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json',
        }

    def _gist_load(self) -> Optional[dict]:
        """Load this model's state from GitHub Gist.

        The Gist holds a multi-model dict keyed by MODEL_TAG so multiple
        branches can share one Gist without overwriting each other.
        Returns our model's state dict, or None on failure / first run.
        """
        if not self.gist_id or not self.github_token:
            return None
        try:
            url = f'https://api.github.com/gists/{self.gist_id}'
            req = urllib.request.Request(url, headers=self._gist_headers())
            with urllib.request.urlopen(req, timeout=8) as resp:
                gist_data = json.loads(resp.read().decode())
            content = gist_data.get('files', {}).get(GIST_FILE_NAME, {}).get('content', '')
            if not content:
                return None
            full = json.loads(content)
            # Multi-model format: top-level keys are model tags
            return full.get(MODEL_TAG)
        except Exception as e:
            print(f"[SCANNER] Gist load failed: {e}")
            return None

    def _gist_load_full(self) -> dict:
        """Load the complete multi-model dict from Gist (for read-modify-write saves)."""
        if not self.gist_id or not self.github_token:
            return {}
        try:
            url = f'https://api.github.com/gists/{self.gist_id}'
            req = urllib.request.Request(url, headers=self._gist_headers())
            with urllib.request.urlopen(req, timeout=8) as resp:
                gist_data = json.loads(resp.read().decode())
            content = gist_data.get('files', {}).get(GIST_FILE_NAME, {}).get('content', '')
            return json.loads(content) if content else {}
        except Exception:
            return {}

    def _gist_save(self, data: dict):
        """Push this model's state to GitHub Gist (read-modify-write, non-blocking best-effort).

        Reads existing Gist, updates only our MODEL_TAG slot, writes back.
        Other models' data is preserved.
        """
        if not self.gist_id or not self.github_token:
            return
        try:
            full = self._gist_load_full()
            full[MODEL_TAG] = data
            full['_last_updated'] = datetime.now().isoformat()
            url = f'https://api.github.com/gists/{self.gist_id}'
            payload = json.dumps({
                'files': {GIST_FILE_NAME: {'content': json.dumps(full, indent=2)}}
            }).encode()
            req = urllib.request.Request(
                url, data=payload, headers=self._gist_headers(), method='PATCH'
            )
            with urllib.request.urlopen(req, timeout=8):
                pass
            print(f"[SCANNER] Gist saved ({MODEL_TAG})")
        except Exception as e:
            print(f"[SCANNER] Gist save failed: {e}")

    def _apply_state(self, data: dict):
        self.equity = data.get('equity', self.equity)
        self.daily_pnl = data.get('daily_pnl', 0.0)
        self.daily_trade_count = data.get('daily_trade_count', 0)
        self._daily_date = data.get('daily_date', '')
        self.positions = {
            k: HypotheticalPosition.from_dict(v)
            for k, v in data.get('positions', {}).items()
        }
        self.closed_trades = [
            ClosedTrade.from_dict(t)
            for t in data.get('closed_trades', [])
        ]
        self.signal_history = data.get('signal_history', [])

    def _load_state(self):
        # Try Gist first (authoritative on Streamlit Cloud), fall back to local
        data = self._gist_load()
        if data is None and STATE_PATH.exists():
            try:
                data = json.loads(STATE_PATH.read_text())
            except Exception as e:
                print(f"[SCANNER] Failed to load local state: {e}")
        if data:
            try:
                self._apply_state(data)
            except Exception as e:
                print(f"[SCANNER] Failed to apply state: {e}")

    def _save_state(self):
        data = {
            'equity': self.equity,
            'daily_pnl': self.daily_pnl,
            'daily_trade_count': self.daily_trade_count,
            'daily_date': self._daily_date,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'closed_trades': [t.to_dict() for t in self.closed_trades[-200:]],
            'signal_history': self.signal_history[-MAX_SIGNAL_HISTORY:],
        }
        # Always save locally
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATE_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[SCANNER] Local save failed: {e}")
        # Also push to Gist if configured
        self._gist_save(data)

    def _reset_daily_if_needed(self):
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self._daily_date:
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self._daily_date = today

    # ------------------------------------------------------------------
    # Signal evaluation
    # ------------------------------------------------------------------

    def evaluate_signal(self, analysis, current_price: float) -> Optional[ScannerAlert]:
        """Evaluate a ChannelAnalysis and return an ENTRY alert if warranted.

        Args:
            analysis: ChannelAnalysis from prepare_multi_tf_analysis()
            current_price: Latest TSLA price
        """
        self._reset_daily_if_needed()
        now = datetime.now().isoformat()

        sig = analysis.signal

        # Record in history
        self.signal_history.append({
            'time': now,
            'action': sig.action,
            'confidence': sig.confidence,
            'primary_tf': sig.primary_tf,
            'signal_type': getattr(sig, 'signal_type', 'bounce'),
            'reason': sig.reason,
        })

        # Kill switch
        if self.config.kill_switch:
            return ScannerAlert(
                alert_type='RISK_WARNING', timestamp=now,
                warning_msg='Kill switch is ON — all signals suppressed',
            )

        # Daily loss limit
        if self.daily_pnl <= self.config.daily_loss_limit:
            return ScannerAlert(
                alert_type='RISK_WARNING', timestamp=now,
                warning_msg=f'Daily loss limit reached (${self.daily_pnl:,.0f})',
            )

        # HOLD = no action
        if sig.action == 'HOLD':
            return None

        # Confidence gate
        if sig.confidence < self.config.min_confidence:
            return None

        # Max positions
        if len(self.positions) >= self.config.max_positions:
            return ScannerAlert(
                alert_type='RISK_WARNING', timestamp=now,
                warning_msg=f'Max positions ({self.config.max_positions}) reached — signal ignored',
            )

        # Duplicate direction check
        direction = 'long' if sig.action == 'BUY' else 'short'
        for pos in self.positions.values():
            if pos.direction == direction:
                return None  # Already have a position in this direction

        # Position sizing
        risk_dollars = self.equity * self.config.risk_per_trade_pct
        stop_pct = sig.suggested_stop_pct
        if stop_pct <= 0:
            stop_pct = 0.02

        # shares = risk / (stop_distance)
        stop_distance = stop_pct * current_price
        shares = int(risk_dollars / stop_distance) if stop_distance > 0 else 0

        # Cap at max buying power
        buying_power = self.equity * self.config.max_leverage
        max_notional = buying_power * self.config.max_buying_power_pct
        max_shares = int(max_notional / current_price) if current_price > 0 else 0
        shares = min(shares, max_shares)

        if shares <= 0:
            return None

        notional = shares * current_price

        # Compute stop/TP prices
        if direction == 'long':
            stop_price = current_price * (1 - stop_pct)
            tp_price = current_price * (1 + sig.suggested_tp_pct)
        else:
            stop_price = current_price * (1 + stop_pct)
            tp_price = current_price * (1 - sig.suggested_tp_pct)

        rr = sig.suggested_tp_pct / max(stop_pct, 0.001)

        alert = ScannerAlert(
            alert_type='ENTRY',
            timestamp=now,
            action=sig.action,
            price=current_price,
            shares=shares,
            stop_price=stop_price,
            tp_price=tp_price,
            risk_reward=rr,
            confidence=sig.confidence,
            signal_type=getattr(sig, 'signal_type', 'bounce'),
            primary_tf=sig.primary_tf,
            reason=sig.reason,
            notional=notional,
        )

        # Auto-enter hypothetical position
        self._enter_hypothetical(alert, direction)
        self._save_state()

        return alert

    def _enter_hypothetical(self, alert: ScannerAlert, direction: str):
        pos_id = str(uuid.uuid4())[:8]
        pos = HypotheticalPosition(
            pos_id=pos_id,
            direction=direction,
            entry_price=alert.price,
            entry_time=alert.timestamp,
            shares=alert.shares,
            notional=alert.notional,
            stop_price=alert.stop_price,
            tp_price=alert.tp_price,
            signal_type=alert.signal_type,
            primary_tf=alert.primary_tf,
            confidence=alert.confidence,
            best_price=alert.price,
            reason=alert.reason,
        )
        self.positions[pos_id] = pos
        alert.pos_id = pos_id

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    def check_exits(self, current_price: float, bar_high: float, bar_low: float) -> List[ScannerAlert]:
        """Check all open positions for exit conditions.

        Args:
            current_price: Current price
            bar_high: High of the current bar (for stop/TP hit detection)
            bar_low: Low of the current bar
        """
        alerts: List[ScannerAlert] = []
        to_close: List[str] = []
        now = datetime.now()

        # Compute ET time once for EOD check
        _is_eod = False
        try:
            import pytz
            _et = pytz.timezone('US/Eastern')
            _now_et = datetime.now(_et)
            _is_eod = (
                _now_et.hour > self.EOD_CLOSE_HOUR_ET or
                (_now_et.hour == self.EOD_CLOSE_HOUR_ET and
                 _now_et.minute >= self.EOD_CLOSE_MINUTE_ET)
            )
        except Exception:
            pass

        for pos_id, pos in self.positions.items():
            exit_reason = None
            exit_price = current_price

            # Compute hold time and unrealized PnL (used by multiple rules below)
            hold_minutes = 0.0
            try:
                entry_dt = datetime.fromisoformat(pos.entry_time)
                hold_minutes = (now - entry_dt).total_seconds() / 60
            except (ValueError, TypeError):
                pass

            if pos.direction == 'long':
                unrealized_pnl = (current_price - pos.entry_price) * pos.shares
            else:
                unrealized_pnl = (pos.entry_price - current_price) * pos.shares

            # --- Breakeven stop adjustment (not an exit — move stop to entry) ---
            # After 30 min in profit, stop can never be a loss again.
            if (not pos.breakeven_applied
                    and hold_minutes >= self.BREAKEVEN_TRIGGER_MIN
                    and unrealized_pnl > 0):
                if pos.direction == 'long':
                    pos.stop_price = max(pos.stop_price, pos.entry_price)
                else:
                    pos.stop_price = min(pos.stop_price, pos.entry_price)
                pos.breakeven_applied = True
                print(f"[SCANNER] Breakeven stop set for {pos_id} @ ${pos.entry_price:.2f}")

            # Update best price (for trailing stop)
            if pos.direction == 'long':
                if bar_high > pos.best_price:
                    pos.best_price = bar_high
            else:
                if bar_low < pos.best_price:
                    pos.best_price = bar_low

            # --- EOD force close (3:45 PM ET) ---
            if exit_reason is None and _is_eod:
                exit_reason = 'eod_close'

            # --- Near-TP early close (within first 30 min) ---
            # If within 1% of TP price and already up $100+, don't wait for the last tick
            if exit_reason is None and hold_minutes < self.NEAR_TP_WINDOW_MIN:
                if unrealized_pnl >= self.NEAR_TP_MIN_GAIN:
                    if (pos.direction == 'long' and bar_high >= pos.tp_price * (1 - self.NEAR_TP_PCT)):
                        exit_reason = 'near_tp'
                    elif (pos.direction == 'short' and bar_low <= pos.tp_price * (1 + self.NEAR_TP_PCT)):
                        exit_reason = 'near_tp'

            # --- Equity ceiling (unrealized >= 5% of account equity) ---
            if exit_reason is None:
                if unrealized_pnl >= self.equity * self.EQUITY_CEILING_PCT:
                    exit_reason = 'equity_ceiling'

            # --- Outlier winner (unrealized >= 2x expected TP gain) ---
            if exit_reason is None:
                if pos.direction == 'long':
                    expected_tp_gain = (pos.tp_price - pos.entry_price) * pos.shares
                else:
                    expected_tp_gain = (pos.entry_price - pos.tp_price) * pos.shares
                if expected_tp_gain > 0 and unrealized_pnl >= self.OUTLIER_MULT * expected_tp_gain:
                    exit_reason = 'outlier_winner'

            # --- Stop loss ---
            if exit_reason is None:
                if pos.direction == 'long' and bar_low <= pos.stop_price:
                    exit_reason = 'stop_loss'
                    exit_price = pos.stop_price
                elif pos.direction == 'short' and bar_high >= pos.stop_price:
                    exit_reason = 'stop_loss'
                    exit_price = pos.stop_price

            # --- Take profit ---
            if exit_reason is None:
                if pos.direction == 'long' and bar_high >= pos.tp_price:
                    exit_reason = 'take_profit'
                    exit_price = pos.tp_price
                elif pos.direction == 'short' and bar_low <= pos.tp_price:
                    exit_reason = 'take_profit'
                    exit_price = pos.tp_price

            # --- Trailing stop (only if in profit) ---
            if exit_reason is None:
                if pos.direction == 'long':
                    trail_price = pos.best_price * (1 - self.TRAILING_STOP_PCT)
                    in_profit = pos.best_price > pos.entry_price
                    if in_profit and bar_low <= trail_price:
                        exit_reason = 'trailing_stop'
                        exit_price = trail_price
                else:
                    trail_price = pos.best_price * (1 + self.TRAILING_STOP_PCT)
                    in_profit = pos.best_price < pos.entry_price
                    if in_profit and bar_high >= trail_price:
                        exit_reason = 'trailing_stop'
                        exit_price = trail_price

            # --- Timeout (5 hours) ---
            if exit_reason is None:
                if hold_minutes > self.TIMEOUT_MINUTES:
                    exit_reason = 'timeout'

            if exit_reason:
                alert = self._close_position(pos_id, pos, exit_price, exit_reason)
                alerts.append(alert)
                to_close.append(pos_id)

        for pos_id in to_close:
            del self.positions[pos_id]

        if alerts:
            self._save_state()

        return alerts

    def _close_position(
        self, pos_id: str, pos: HypotheticalPosition,
        exit_price: float, exit_reason: str,
    ) -> ScannerAlert:
        now = datetime.now()

        # P&L with slippage + commission
        if pos.direction == 'long':
            raw_pnl = (exit_price - pos.entry_price) * pos.shares
        else:
            raw_pnl = (pos.entry_price - exit_price) * pos.shares

        slippage_cost = pos.notional * SLIPPAGE_PCT * 2  # entry + exit
        commission = pos.shares * COMMISSION_PER_SHARE * 2
        pnl = raw_pnl - slippage_cost - commission
        pnl_pct = pnl / pos.notional if pos.notional > 0 else 0

        # Update equity and daily tracking
        self.equity += pnl
        self.daily_pnl += pnl
        self.daily_trade_count += 1

        try:
            entry_dt = datetime.fromisoformat(pos.entry_time)
            hold_minutes = (now - entry_dt).total_seconds() / 60
        except (ValueError, TypeError):
            hold_minutes = 0

        trade = ClosedTrade(
            pos_id=pos_id,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_time=pos.entry_time,
            exit_time=now.isoformat(),
            exit_reason=exit_reason,
            hold_minutes=hold_minutes,
        )
        self.closed_trades.append(trade)

        return ScannerAlert(
            alert_type='EXIT',
            timestamp=now.isoformat(),
            pos_id=pos_id,
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
            price=exit_price,
            action=pos.direction,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Total unrealized P&L across all open positions."""
        total = 0.0
        for pos in self.positions.values():
            if pos.direction == 'long':
                total += (current_price - pos.entry_price) * pos.shares
            else:
                total += (pos.entry_price - current_price) * pos.shares
        return total

    def reset(self):
        """Clear all state and start fresh."""
        self.positions.clear()
        self.closed_trades.clear()
        self.signal_history.clear()
        self.equity = self.config.initial_capital
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self._daily_date = ''
        self._save_state()
