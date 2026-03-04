"""
v15/validation/market_calendar.py — Major Market-Moving Calendar Events

Comprehensive calendar of macro events from Jan 2025 through Feb 2026.
Sources: Federal Reserve, BLS, BEA, Tesla IR, OPEC, Kansas City Fed.

NOTE: The Oct 1 - Nov 12, 2025 US government shutdown disrupted many
release schedules. October 2025 CPI and NFP were NOT released separately.
BLS revised dates are reflected below.

Usage:
    from v15.validation.market_calendar import CALENDAR_EVENTS, get_events_for_date, get_events_in_range
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# MASTER CALENDAR: date_str -> list of (event_type, description)
#
# Event types:
#   FOMC            - Federal Open Market Committee rate decision (announcement day)
#   TSLA_EARNINGS   - Tesla quarterly earnings release (after market close)
#   OPEX            - Monthly options expiration (3rd Friday)
#   QUAD_WITCH      - Quarterly options expiration / triple witching (Mar/Jun/Sep/Dec)
#   CPI             - Consumer Price Index release (8:30 AM ET)
#   NFP             - Non-Farm Payrolls / Employment Situation (8:30 AM ET)
#   GDP             - GDP estimate release (advance/second/third) (8:30 AM ET)
#   FED_SPEECH      - Major Fed Chair speech or Congressional testimony
#   JACKSON_HOLE    - Jackson Hole Economic Symposium (Powell speech day)
#   OPEC            - OPEC/OPEC+ ministerial or JMMC meeting
# ═══════════════════════════════════════════════════════════════════════════════

_RAW_EVENTS: List[Tuple[str, str, str]] = [
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  JANUARY 2025                                                        ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    ('2025-01-10', 'NFP',           'Dec 2024 Employment Situation'),
    ('2025-01-15', 'CPI',           'Dec 2024 CPI'),
    ('2025-01-17', 'OPEX',          'Jan 2025 monthly options expiration'),
    ('2025-01-29', 'FOMC',          'FOMC rate decision (Jan 28-29)'),
    ('2025-01-29', 'TSLA_EARNINGS', 'Tesla Q4 2024 earnings'),
    ('2025-01-30', 'GDP',           'Q4 2024 GDP advance estimate'),

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FEBRUARY 2025                                                       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    ('2025-02-03', 'OPEC',          '58th JMMC meeting'),
    ('2025-02-07', 'NFP',           'Jan 2025 Employment Situation'),
    ('2025-02-11', 'FED_SPEECH',    'Powell semiannual testimony (Senate)'),
    ('2025-02-12', 'FED_SPEECH',    'Powell semiannual testimony (House)'),
    ('2025-02-12', 'CPI',           'Jan 2025 CPI'),
    ('2025-02-21', 'OPEX',          'Feb 2025 monthly options expiration'),
    ('2025-02-27', 'GDP',           'Q4 2024 GDP second estimate'),

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  MARCH 2025                                                          ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    ('2025-03-07', 'NFP',           'Feb 2025 Employment Situation'),
    ('2025-03-12', 'CPI',           'Feb 2025 CPI'),
    ('2025-03-19', 'FOMC',          'FOMC rate decision (Mar 18-19)'),
    ('2025-03-21', 'QUAD_WITCH',    'Mar 2025 triple/quadruple witching'),
    ('2025-03-27', 'GDP',           'Q4 2024 GDP third estimate'),

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  APRIL 2025                                                          ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    ('2025-04-04', 'NFP',           'Mar 2025 Employment Situation'),
    ('2025-04-05', 'OPEC',          '59th JMMC meeting'),
    ('2025-04-10', 'CPI',           'Mar 2025 CPI'),
    ('2025-04-17', 'OPEX',          'Apr 2025 monthly options expiration'),
    ('2025-04-22', 'TSLA_EARNINGS', 'Tesla Q1 2025 earnings'),
    ('2025-04-30', 'GDP',           'Q1 2025 GDP advance estimate'),

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  MAY 2025                                                            ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    ('2025-05-02', 'NFP',           'Apr 2025 Employment Situation'),
    ('2025-05-07', 'FOMC',          'FOMC rate decision (May 6-7)'),
    ('2025-05-13', 'CPI',           'Apr 2025 CPI'),
    ('2025-05-16', 'OPEX',          'May 2025 monthly options expiration'),
    ('2025-05-28', 'OPEC',          '39th OPEC+ Ministerial / 60th JMMC'),
    ('2025-05-29', 'GDP',           'Q1 2025 GDP second estimate'),

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  JUNE 2025                                                           ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    ('2025-06-06', 'NFP',           'May 2025 Employment Situation'),
    ('2025-06-11', 'CPI',           'May 2025 CPI'),
    ('2025-06-18', 'FOMC',          'FOMC rate decision (Jun 17-18)'),
    ('2025-06-20', 'QUAD_WITCH',    'Jun 2025 triple/quadruple witching'),
    ('2025-06-24', 'FED_SPEECH',    'Powell semiannual testimony (House)'),
    ('2025-06-25', 'FED_SPEECH',    'Powell semiannual testimony (Senate)'),
    ('2025-06-26', 'GDP',           'Q1 2025 GDP third estimate'),

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  JULY 2025                                                           ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    ('2025-07-03', 'NFP',           'Jun 2025 Employment Situation'),
    ('2025-07-15', 'CPI',           'Jun 2025 CPI'),
    ('2025-07-18', 'OPEX',          'Jul 2025 monthly options expiration'),
    ('2025-07-23', 'TSLA_EARNINGS', 'Tesla Q2 2025 earnings'),
    ('2025-07-28', 'OPEC',          '61st JMMC meeting'),
    ('2025-07-30', 'FOMC',          'FOMC rate decision (Jul 29-30)'),
    ('2025-07-30', 'GDP',           'Q2 2025 GDP advance estimate'),

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  AUGUST 2025                                                         ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    ('2025-08-01', 'NFP',           'Jul 2025 Employment Situation'),
    ('2025-08-12', 'CPI',           'Jul 2025 CPI'),
    ('2025-08-15', 'OPEX',          'Aug 2025 monthly options expiration'),
    ('2025-08-22', 'JACKSON_HOLE',  'Powell Jackson Hole speech'),
    ('2025-08-28', 'GDP',           'Q2 2025 GDP second estimate'),

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  SEPTEMBER 2025                                                      ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    ('2025-09-05', 'NFP',           'Aug 2025 Employment Situation'),
    ('2025-09-11', 'CPI',           'Aug 2025 CPI'),
    ('2025-09-17', 'FOMC',          'FOMC rate decision (Sep 16-17)'),
    ('2025-09-19', 'QUAD_WITCH',    'Sep 2025 triple/quadruple witching'),
    ('2025-09-25', 'GDP',           'Q2 2025 GDP third estimate'),

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  OCTOBER 2025  ** Gov shutdown: Oct 1 - Nov 12 **                    ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    # NFP for Sep was originally early Oct but delayed to Nov 20 (see below)
    # CPI for Sep was delayed from Oct 15 to Oct 24
    ('2025-10-01', 'OPEC',          '62nd JMMC meeting'),
    ('2025-10-14', 'FED_SPEECH',    'Powell speech at NABE (Philadelphia)'),
    ('2025-10-17', 'OPEX',          'Oct 2025 monthly options expiration'),
    ('2025-10-22', 'TSLA_EARNINGS', 'Tesla Q3 2025 earnings'),
    ('2025-10-24', 'CPI',           'Sep 2025 CPI (delayed from Oct 15)'),
    ('2025-10-29', 'FOMC',          'FOMC rate decision (Oct 28-29)'),
    # NOTE: Oct 2025 CPI was NOT released (gov shutdown, data not collected)
    # NOTE: Oct 2025 NFP was NOT released separately (combined with Nov)

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  NOVEMBER 2025  ** Gov shutdown ends Nov 12 **                       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    ('2025-11-20', 'NFP',           'Sep 2025 Employment Situation (delayed 6+ weeks)'),
    ('2025-11-21', 'OPEX',          'Nov 2025 monthly options expiration'),
    ('2025-11-30', 'OPEC',          '40th OPEC+ Ministerial Meeting'),

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  DECEMBER 2025                                                       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    ('2025-12-10', 'FOMC',          'FOMC rate decision (Dec 9-10)'),
    ('2025-12-16', 'NFP',           'Oct+Nov 2025 Employment Situation (combined)'),
    ('2025-12-18', 'CPI',           'Nov 2025 CPI (Oct CPI not published)'),
    ('2025-12-19', 'QUAD_WITCH',    'Dec 2025 triple/quadruple witching'),
    ('2025-12-23', 'GDP',           'Q3 2025 GDP initial estimate (replaces adv+2nd)'),

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  JANUARY 2026                                                        ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    ('2026-01-09', 'NFP',           'Dec 2025 Employment Situation'),
    ('2026-01-13', 'CPI',           'Dec 2025 CPI'),
    ('2026-01-16', 'OPEX',          'Jan 2026 monthly options expiration'),
    ('2026-01-22', 'GDP',           'Q3 2025 GDP updated estimate (replaces 3rd)'),
    ('2026-01-28', 'FOMC',          'FOMC rate decision (Jan 27-28)'),
    ('2026-01-28', 'TSLA_EARNINGS', 'Tesla Q4 2025 earnings'),

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FEBRUARY 2026                                                       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    ('2026-02-11', 'NFP',           'Jan 2026 Employment Situation (delayed from Feb 6)'),
    ('2026-02-11', 'CPI',           'Jan 2026 CPI'),
    ('2026-02-20', 'OPEX',          'Feb 2026 monthly options expiration'),
    ('2026-02-20', 'GDP',           'Q4 2025 GDP advance estimate (delayed from Jan 29)'),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Build the main dictionary: date_str -> list of (event_type, description)
# ═══════════════════════════════════════════════════════════════════════════════
CALENDAR_EVENTS: Dict[str, List[Tuple[str, str]]] = {}
for date_str, etype, desc in _RAW_EVENTS:
    CALENDAR_EVENTS.setdefault(date_str, []).append((etype, desc))


# ═══════════════════════════════════════════════════════════════════════════════
# Flat dictionary version: date_str -> comma-joined event types
# Useful for quick tagging in DataFrames
# ═══════════════════════════════════════════════════════════════════════════════
CALENDAR_FLAT: Dict[str, str] = {
    date_str: ','.join(et for et, _ in events)
    for date_str, events in CALENDAR_EVENTS.items()
}


# ═══════════════════════════════════════════════════════════════════════════════
# High-impact subset (events that typically move TSLA > 2%)
# ═══════════════════════════════════════════════════════════════════════════════
HIGH_IMPACT_TYPES = {'FOMC', 'TSLA_EARNINGS', 'CPI', 'NFP', 'JACKSON_HOLE', 'QUAD_WITCH'}

CALENDAR_HIGH_IMPACT: Dict[str, List[Tuple[str, str]]] = {
    date_str: [(et, d) for et, d in events if et in HIGH_IMPACT_TYPES]
    for date_str, events in CALENDAR_EVENTS.items()
}
# Remove empty entries
CALENDAR_HIGH_IMPACT = {k: v for k, v in CALENDAR_HIGH_IMPACT.items() if v}


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def get_events_for_date(dt) -> List[Tuple[str, str]]:
    """Return list of (event_type, description) for a given date."""
    key = pd.Timestamp(dt).strftime('%Y-%m-%d')
    return CALENDAR_EVENTS.get(key, [])


def get_events_in_range(start, end, event_types: Optional[set] = None) -> pd.DataFrame:
    """
    Return DataFrame of events in [start, end] range.
    Optionally filter by event_types (set of type strings).
    """
    rows = []
    for date_str, events in CALENDAR_EVENTS.items():
        dt = pd.Timestamp(date_str)
        if dt < pd.Timestamp(start) or dt > pd.Timestamp(end):
            continue
        for etype, desc in events:
            if event_types and etype not in event_types:
                continue
            rows.append({'date': dt, 'event_type': etype, 'description': desc})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values('date').reset_index(drop=True)
    return df


def tag_trading_days(df: pd.DataFrame, date_col: str = 'date',
                     window: int = 0) -> pd.DataFrame:
    """
    Add 'event_type' column to a DataFrame of trading days.
    If window > 0, also tags rows within `window` trading days
    of an event (useful for pre/post event analysis).

    Args:
        df: DataFrame with a date column
        date_col: name of the date column
        window: number of trading days before/after to tag (0 = exact match only)

    Returns:
        DataFrame with 'calendar_event' column added
    """
    df = df.copy()
    df['calendar_event'] = ''

    event_dates = {pd.Timestamp(d) for d in CALENDAR_EVENTS}

    for idx, row in df.iterrows():
        dt = pd.Timestamp(row[date_col]).normalize()
        events = CALENDAR_EVENTS.get(dt.strftime('%Y-%m-%d'), [])
        if events:
            df.at[idx, 'calendar_event'] = ','.join(et for et, _ in events)

    if window > 0:
        # Also tag nearby trading days
        dates_sorted = df[date_col].sort_values().values
        for idx, row in df.iterrows():
            if df.at[idx, 'calendar_event']:
                continue
            dt = pd.Timestamp(row[date_col]).normalize()
            for ev_dt in event_dates:
                # Count trading days between
                mask = (dates_sorted >= min(dt, ev_dt)) & (dates_sorted <= max(dt, ev_dt))
                td = mask.sum() - 1  # trading days between
                if 0 < td <= window:
                    events = CALENDAR_EVENTS.get(ev_dt.strftime('%Y-%m-%d'), [])
                    prefix = 'pre_' if dt < ev_dt else 'post_'
                    tag = ','.join(f"{prefix}{et}" for et, _ in events)
                    existing = df.at[idx, 'calendar_event']
                    df.at[idx, 'calendar_event'] = f"{existing},{tag}" if existing else tag
                    break  # tag with nearest event only

    return df


def days_to_next_event(dt, event_types: Optional[set] = None) -> Optional[int]:
    """
    Return number of calendar days until the next event of the given type(s).
    If event_types is None, considers all events.
    """
    dt = pd.Timestamp(dt).normalize()
    best = None
    for date_str, events in CALENDAR_EVENTS.items():
        ev_dt = pd.Timestamp(date_str)
        if ev_dt < dt:
            continue
        if event_types:
            if not any(et in event_types for et, _ in events):
                continue
        delta = (ev_dt - dt).days
        if best is None or delta < best:
            best = delta
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# Quick summary when run standalone
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 80)
    print("MARKET CALENDAR: Jan 2025 - Feb 2026")
    print("=" * 80)

    # Count by type
    from collections import Counter
    type_counts = Counter()
    for events in CALENDAR_EVENTS.values():
        for etype, _ in events:
            type_counts[etype] += 1

    print(f"\nTotal event dates: {len(CALENDAR_EVENTS)}")
    print(f"Total events:      {sum(type_counts.values())}")
    print("\nBy type:")
    for etype, count in type_counts.most_common():
        print(f"  {etype:20s} {count:3d}")

    print("\n" + "-" * 80)
    print("ALL EVENTS (chronological):")
    print("-" * 80)
    for date_str in sorted(CALENDAR_EVENTS.keys()):
        for etype, desc in CALENDAR_EVENTS[date_str]:
            print(f"  {date_str}  {etype:20s}  {desc}")

    # Show high-impact dates
    print("\n" + "-" * 80)
    print("HIGH-IMPACT DATES (multi-event days):")
    print("-" * 80)
    for date_str in sorted(CALENDAR_EVENTS.keys()):
        events = CALENDAR_EVENTS[date_str]
        if len(events) > 1:
            tags = ' + '.join(et for et, _ in events)
            print(f"  {date_str}  [{tags}]")
