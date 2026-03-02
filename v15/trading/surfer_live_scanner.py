"""
Surfer Live Scanner — Channel Surfer signal evaluation + position tracking.

No auto-trading. No broker integration. Dashboard notifications only.
Persistence via JSON at ~/.x14/surfer_scanner_state.json (local)
or GitHub Gist (when GIST_ID + GITHUB_TOKEN are provided, e.g. Streamlit Cloud).
"""

import json
import os
import urllib.request
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pytz

_ET = pytz.timezone('US/Eastern')


def _now_et() -> datetime:
    """Current time in US/Eastern (consistent with market hours and charts)."""
    return datetime.now(_ET)


def _is_market_open() -> bool:
    """True if current ET time is within regular trading hours (9:30-16:00, weekdays)."""
    now = _now_et()
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    t = now.time()
    from datetime import time as _time
    return _time(9, 30) <= t < _time(16, 0)


def _is_extended_hours() -> bool:
    """True if in premarket (4:00-9:30 ET) or after-hours (16:00-20:00 ET) on weekdays."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    t = now.time()
    from datetime import time as _time
    return _time(4, 0) <= t < _time(9, 30) or _time(16, 0) <= t < _time(20, 0)


STATE_PATH = Path.home() / ".x14" / "surfer_scanner_state.json"
MAX_SIGNAL_HISTORY = 200
GIST_FILE_NAME = 'surfer_scanner_state.json'
MODEL_TAG = 'c12a'  # Identifies this branch in the multi-model Gist

# Slippage + commission assumptions
SLIPPAGE_PCT = 0.0005     # 0.05% slippage per side
COMMISSION_PER_SHARE = 0.005  # $0.005/share (IBKR tiered)

# Telegram alerts
_TG_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
_TG_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
print(f"[SCANNER] Telegram config: token={'SET' if _TG_TOKEN else 'MISSING'} "
      f"({len(_TG_TOKEN)} chars), chat_id={'SET' if _TG_CHAT_ID else 'MISSING'}")

# Queue for alerts that fail to send directly (flushed to Gist on next save)
_PENDING_TG_ALERTS: list = []


def _send_telegram(msg: str):
    """Send a Telegram message directly, or queue for relay via GitHub Actions."""
    full_msg = f"[{MODEL_TAG}] {msg}"
    if not _TG_TOKEN or not _TG_CHAT_ID:
        # No direct creds — queue for Gist relay
        _PENDING_TG_ALERTS.append(full_msg)
        print(f"[SCANNER] Telegram queued for relay (no creds): {full_msg[:80]}")
        return
    try:
        url = f'https://api.telegram.org/bot{_TG_TOKEN}/sendMessage'
        payload = json.dumps({
            'chat_id': _TG_CHAT_ID,
            'text': full_msg,
            'parse_mode': 'HTML',
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as e:
        # Direct send failed (e.g. DNS blocked on HF Spaces) — queue for relay
        _PENDING_TG_ALERTS.append(full_msg)
        print(f"[SCANNER] Telegram direct failed, queued for relay: {e}")


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

    # Trailing stop uses profit-tier logic matching surfer_backtest.py (see _calc_trail_price)
    # TRAILING_STOP_PCT removed — replaced by profit-ratio tiers
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
        self._ext_opens_today: int = 0    # Extended-hours entries used today
        self._ext_closes_today: int = 0   # Extended-hours exits used today
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
        Other models' data is preserved. Also flushes any pending Telegram
        alerts into `_pending_telegram` for the GitHub Actions relay.
        """
        if not self.gist_id or not self.github_token:
            return
        try:
            full = self._gist_load_full()
            full[MODEL_TAG] = data
            full['_last_updated'] = _now_et().isoformat()
            # Flush pending Telegram alerts into Gist for relay
            if _PENDING_TG_ALERTS:
                existing = full.get('_pending_telegram', [])
                existing.extend(_PENDING_TG_ALERTS)
                full['_pending_telegram'] = existing[-50:]  # Cap at 50
                _PENDING_TG_ALERTS.clear()
                print(f"[SCANNER] Flushed {len(existing)} Telegram alerts to Gist")
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
        self._ext_opens_today = data.get('ext_opens_today', 0)
        self._ext_closes_today = data.get('ext_closes_today', 0)
        self.positions = {
            k: HypotheticalPosition.from_dict(v)
            for k, v in data.get('positions', {}).items()
        }
        self.closed_trades = [
            ClosedTrade.from_dict(t)
            for t in data.get('closed_trades', [])
        ]
        self.signal_history = data.get('signal_history', [])

    @staticmethod
    def _migrate_utc_to_et(data: dict) -> dict:
        """One-time migration: convert naive UTC timestamps to ET, purge after-hours trades.

        Before this fix, Streamlit Cloud logged datetime.now() which was UTC.
        This converts those naive timestamps to proper ET and removes any
        trades whose entry fell outside regular market hours (9:30-16:00 ET).
        """
        if data.get('_migrated_tz_v1'):
            return data  # Already migrated

        from datetime import time as _time
        _utc = pytz.utc
        _mkt_open = _time(9, 30)
        _mkt_close = _time(16, 0)

        def _convert_ts(iso_str: str) -> str:
            """Convert a naive-UTC ISO string to ET-aware ISO string."""
            try:
                dt = datetime.fromisoformat(iso_str)
                if dt.tzinfo is not None:
                    return iso_str  # Already tz-aware, skip
                # Naive → assume UTC (Streamlit Cloud) → convert to ET
                dt_utc = _utc.localize(dt)
                dt_et = dt_utc.astimezone(_ET)
                return dt_et.isoformat()
            except (ValueError, TypeError):
                return iso_str

        def _entry_during_market(iso_str: str) -> bool:
            """Return True if the timestamp falls within market hours."""
            try:
                dt = datetime.fromisoformat(iso_str)
                if dt.tzinfo is None:
                    dt = _utc.localize(dt).astimezone(_ET)
                else:
                    dt = dt.astimezone(_ET)
                if dt.weekday() >= 5:
                    return False
                return _mkt_open <= dt.time() < _mkt_close
            except (ValueError, TypeError):
                return True  # Keep if unparseable

        n_trades_before = len(data.get('closed_trades', []))
        n_pos_before = len(data.get('positions', {}))

        # Migrate closed trades: convert timestamps, drop after-hours entries
        migrated_trades = []
        removed_pnl = 0.0
        for t in data.get('closed_trades', []):
            et_entry = _convert_ts(t.get('entry_time', ''))
            if not _entry_during_market(t.get('entry_time', '')):
                removed_pnl += t.get('pnl', 0.0)
                continue  # Drop this trade
            t['entry_time'] = et_entry
            t['exit_time'] = _convert_ts(t.get('exit_time', ''))
            migrated_trades.append(t)
        data['closed_trades'] = migrated_trades

        # Migrate open positions: convert timestamps, drop after-hours entries
        migrated_pos = {}
        for k, p in data.get('positions', {}).items():
            et_entry = _convert_ts(p.get('entry_time', ''))
            if not _entry_during_market(p.get('entry_time', '')):
                continue  # Drop this position
            p['entry_time'] = et_entry
            migrated_pos[k] = p
        data['positions'] = migrated_pos

        # Migrate signal history timestamps
        migrated_signals = []
        for s in data.get('signal_history', []):
            et_time = _convert_ts(s.get('time', ''))
            if not _entry_during_market(s.get('time', '')):
                continue
            s['time'] = et_time
            migrated_signals.append(s)
        data['signal_history'] = migrated_signals

        # Adjust equity for removed trades' PnL
        if removed_pnl != 0:
            data['equity'] = data.get('equity', 100000.0) - removed_pnl

        n_trades_after = len(data['closed_trades'])
        n_pos_after = len(data['positions'])
        print(f"[SCANNER] TZ migration: trades {n_trades_before}→{n_trades_after} "
              f"(removed {n_trades_before - n_trades_after} after-hours), "
              f"positions {n_pos_before}→{n_pos_after}, "
              f"equity adjusted by ${-removed_pnl:+,.0f}")

        data['_migrated_tz_v1'] = True
        return data

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
                data = self._migrate_utc_to_et(data)
                self._apply_state(data)
                # Save migrated state back (persists the fix)
                if data.get('_migrated_tz_v1'):
                    self._save_state()
            except Exception as e:
                print(f"[SCANNER] Failed to apply state: {e}")

    def _save_state(self):
        data = {
            'equity': self.equity,
            'daily_pnl': self.daily_pnl,
            'daily_trade_count': self.daily_trade_count,
            'daily_date': self._daily_date,
            'ext_opens_today': self._ext_opens_today,
            'ext_closes_today': self._ext_closes_today,
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
        today = _now_et().strftime('%Y-%m-%d')
        if today != self._daily_date:
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self._ext_opens_today = 0
            self._ext_closes_today = 0
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
        now = _now_et().isoformat()

        sig = analysis.signal

        # Block entries outside regular + extended trading hours
        if not _is_market_open():
            if not _is_extended_hours() or self._ext_opens_today >= 1:
                return None

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
            stop_pct = 0.01

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
        if not _is_market_open() and _is_extended_hours():
            self._ext_opens_today += 1
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

        # Telegram entry alert
        action = 'BUY' if direction == 'long' else 'SELL'
        rr = abs(pos.tp_price - pos.entry_price) / max(abs(pos.entry_price - pos.stop_price), 0.01)
        _send_telegram(
            f"🟢 <b>TRADE OPENED</b>\n"
            f"<b>{action}</b> {pos.shares} shares @ ${pos.entry_price:.2f}\n"
            f"Stop: ${pos.stop_price:.2f} | TP: ${pos.tp_price:.2f} | R:R {rr:.1f}:1\n"
            f"Confidence: {pos.confidence:.0%} | {pos.primary_tf}\n"
            f"Notional: ${pos.notional:,.0f}"
        )

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_trail_price(pos: 'HypotheticalPosition') -> Optional[float]:
        """Compute trailing stop price matching surfer_backtest.py profit-tier logic.

        Mirrors evaluate_position() in surfer_backtest.py (simplified: no el_flagged,
        fast_reversion, or trail_width_mult). Returns the effective trailing stop price,
        or None if no trail is active yet (price hasn't moved enough to trigger a tier).

        Tiers for bounce trades (profit_ratio = progress toward TP):
          >= 80% → ultra-tight: initial_stop_dist × 0.005 from best price
          >= 55% → initial_stop_dist × 0.02 from best price
          >= 40% → initial_stop_dist × 0.06 from best price
          < 40%  → None (hard stop only; breakeven managed separately)

        Tiers for break trades (based on absolute % profit from best price):
          > 1.5% → initial_stop_dist × 0.01 from best price
          > 0.8% → initial_stop_dist × 0.02 from best price
          > 0.08% → initial_stop_dist × 0.01 from best price
          else   → None (hard stop only)
        """
        entry = pos.entry_price
        if entry <= 0:
            return None
        initial_stop_dist = abs(pos.stop_price - entry) / entry
        tp_dist = abs(pos.tp_price - entry) / entry
        is_breakout = pos.signal_type == 'break'

        if pos.direction == 'long':
            if pos.best_price <= entry:
                return None  # Not yet in profit
            if is_breakout:
                profit_from_best = (pos.best_price - entry) / entry
                if profit_from_best > 0.015:
                    trail_pct = initial_stop_dist * 0.01
                elif profit_from_best > 0.008:
                    trail_pct = initial_stop_dist * 0.02
                elif profit_from_best > 0.0008:
                    trail_pct = initial_stop_dist * 0.01
                else:
                    return None
            else:  # bounce
                profit_from_entry = (pos.best_price - entry) / entry
                profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
                if profit_ratio >= 0.80:
                    trail_pct = initial_stop_dist * 0.005
                elif profit_ratio >= 0.55:
                    trail_pct = initial_stop_dist * 0.02
                elif profit_ratio >= 0.40:
                    trail_pct = initial_stop_dist * 0.06
                else:
                    return None
            return max(pos.stop_price, pos.best_price * (1.0 - trail_pct))

        else:  # short
            if pos.best_price >= entry:
                return None  # Not yet in profit
            if is_breakout:
                profit_from_best = (entry - pos.best_price) / entry
                if profit_from_best > 0.015:
                    trail_pct = initial_stop_dist * 0.01
                elif profit_from_best > 0.008:
                    trail_pct = initial_stop_dist * 0.02
                elif profit_from_best > 0.0008:
                    trail_pct = initial_stop_dist * 0.01
                else:
                    return None
            else:  # bounce
                profit_from_entry = (entry - pos.best_price) / entry
                profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
                if profit_ratio >= 0.80:
                    trail_pct = initial_stop_dist * 0.005
                elif profit_ratio >= 0.55:
                    trail_pct = initial_stop_dist * 0.02
                elif profit_ratio >= 0.40:
                    trail_pct = initial_stop_dist * 0.06
                else:
                    return None
            return min(pos.stop_price, pos.best_price * (1.0 + trail_pct))

    def check_exits(self, current_price: float, bar_high: float, bar_low: float) -> List[ScannerAlert]:
        """Check all open positions for exit conditions.

        Args:
            current_price: Current price
            bar_high: High of the current bar (for stop/TP hit detection)
            bar_low: Low of the current bar
        """
        alerts: List[ScannerAlert] = []
        to_close: List[str] = []
        now = _now_et()
        ext_session = not _is_market_open() and _is_extended_hours()

        # EOD check using ET time
        _is_eod = (
            now.hour > self.EOD_CLOSE_HOUR_ET or
            (now.hour == self.EOD_CLOSE_HOUR_ET and
             now.minute >= self.EOD_CLOSE_MINUTE_ET)
        )

        for pos_id, pos in self.positions.items():
            exit_reason = None
            exit_price = current_price

            # Compute hold time and unrealized PnL (used by multiple rules below)
            hold_minutes = 0.0
            try:
                entry_dt = datetime.fromisoformat(pos.entry_time)
                # Normalize: if entry was stored tz-naive, assume ET
                if entry_dt.tzinfo is None:
                    entry_dt = _ET.localize(entry_dt)
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

            # --- Hard stop (2% max from entry) ---
            if exit_reason is None:
                max_stop_dist = 0.02
                if pos.direction == 'long':
                    hard_stop = min(pos.stop_price, pos.entry_price * (1 - max_stop_dist))
                    if bar_low <= hard_stop:
                        exit_reason = 'stop_loss'
                        exit_price = hard_stop
                elif pos.direction == 'short':
                    hard_stop = max(pos.stop_price, pos.entry_price * (1 + max_stop_dist))
                    if bar_high >= hard_stop:
                        exit_reason = 'stop_loss'
                        exit_price = hard_stop

            # --- Take profit ---
            if exit_reason is None:
                if pos.direction == 'long' and bar_high >= pos.tp_price:
                    exit_reason = 'take_profit'
                    exit_price = pos.tp_price
                elif pos.direction == 'short' and bar_low <= pos.tp_price:
                    exit_reason = 'take_profit'
                    exit_price = pos.tp_price

            # --- Trailing stop (profit-tier based, matching surfer_backtest.py) ---
            # Tiers based on how far price has moved toward TP (profit_ratio).
            # Near TP → ultra-tight trail (lock profits). Early profit → wider trail.
            # Mirrors the bounce trail logic in surfer_backtest.py evaluate_position().
            if exit_reason is None:
                trail_price = self._calc_trail_price(pos)
                if trail_price is not None:
                    if pos.direction == 'long' and bar_low <= trail_price:
                        exit_reason = 'trailing_stop'
                        exit_price = trail_price
                    elif pos.direction == 'short' and bar_high >= trail_price:
                        exit_reason = 'trailing_stop'
                        exit_price = trail_price

            # --- Timeout (5 hours) ---
            if exit_reason is None:
                if hold_minutes > self.TIMEOUT_MINUTES:
                    exit_reason = 'timeout'

            if exit_reason:
                if ext_session and self._ext_closes_today >= 1:
                    continue  # Already used extended-hours close allowance
                alert = self._close_position(pos_id, pos, exit_price, exit_reason)
                alerts.append(alert)
                to_close.append(pos_id)
                if ext_session:
                    self._ext_closes_today += 1

        for pos_id in to_close:
            del self.positions[pos_id]

        if alerts:
            self._save_state()

        return alerts

    def _close_position(
        self, pos_id: str, pos: HypotheticalPosition,
        exit_price: float, exit_reason: str,
    ) -> ScannerAlert:
        now = _now_et()

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
            if entry_dt.tzinfo is None:
                entry_dt = _ET.localize(entry_dt)
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

        # Telegram exit alert
        pnl_emoji = '🟢' if pnl >= 0 else '🔴'
        action = 'LONG' if pos.direction == 'long' else 'SHORT'
        hold_str = f'{hold_minutes:.0f}m' if hold_minutes < 120 else f'{hold_minutes/60:.1f}h'
        _send_telegram(
            f"{pnl_emoji} <b>TRADE CLOSED</b>\n"
            f"<b>{action}</b> {pos.shares} shares\n"
            f"Entry: ${pos.entry_price:.2f} → Exit: ${exit_price:.2f}\n"
            f"P&L: <b>${pnl:+,.0f}</b> ({pnl_pct:+.2%})\n"
            f"Reason: {exit_reason} | Held: {hold_str}"
        )

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
