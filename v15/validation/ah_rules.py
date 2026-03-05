"""Shared AH (after-hours / extended-hours) rules for all backtester engines.

Time boundaries use ET local time matching TSLAMin.txt timestamps:
  RTH:  09:30 - 16:00 ET
  Pre:  04:00 - 09:30 ET
  Post: 16:00 - 20:00 ET
"""
import datetime as dt


RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)
PRE_OPEN = dt.time(4, 0)
POST_CLOSE = dt.time(20, 0)


def is_rth(bar_time: dt.time) -> bool:
    """Return True if bar_time is within regular trading hours (9:30-16:00 ET)."""
    return RTH_OPEN <= bar_time < RTH_CLOSE


def is_extended_hours(bar_time: dt.time) -> bool:
    """Return True if bar_time is in pre-market (4:00-9:30) or post-market (16:00-20:00)."""
    return (PRE_OPEN <= bar_time < RTH_OPEN) or (RTH_CLOSE <= bar_time < POST_CLOSE)


class AHStateTracker:
    """Track AH open/close allowances per day.

    Rules:
      - 1 base AH open + 1 bonus per winning AH close (unlimited closes)
      - $250 default AH loss limit per individual trade
    """

    def __init__(self):
        self._current_date = None
        self.ext_opens_today = 0
        self.ext_closes_today = 0
        self.ext_wins_today = 0

    def reset_if_new_day(self, bar_date):
        if bar_date != self._current_date:
            self._current_date = bar_date
            self.ext_opens_today = 0
            self.ext_closes_today = 0
            self.ext_wins_today = 0

    def can_open_ah(self) -> bool:
        """1 base open + 1 per winning AH close."""
        allowed = 1 + self.ext_wins_today
        return self.ext_opens_today < allowed

    def record_ah_open(self):
        self.ext_opens_today += 1

    def record_ah_close(self, pnl: float):
        self.ext_closes_today += 1
        if pnl > 0:
            self.ext_wins_today += 1

    @staticmethod
    def check_ah_loss_limit(unrealized_pnl: float, limit: float = 250.0) -> bool:
        """Return True if trade should be force-closed (loss >= limit)."""
        return unrealized_pnl <= -abs(limit)
