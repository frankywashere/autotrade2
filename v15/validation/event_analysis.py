"""
v15/validation/event_analysis.py — Cross-reference hard_stop trades with calendar events
and analyze hour-by-hour TSLA volatility around event days.
"""
import pandas as pd
import numpy as np
from datetime import timedelta, date
from collections import defaultdict
from typing import Dict, List, Set

from v15.validation.market_calendar import (
    CALENDAR_EVENTS, CALENDAR_FLAT, CALENDAR_HIGH_IMPACT,
    get_events_for_date,
)


def load_hard_stop_csv(path: str = 'v15/validation/hard_stop_trades.csv') -> pd.DataFrame:
    """Load the hard_stop trades CSV."""
    df = pd.read_csv(path)
    df['entry_date'] = pd.to_datetime(df['entry_date']).dt.date
    return df


def cross_reference_hard_stops_with_events(df: pd.DataFrame) -> None:
    """Cross-reference hard_stop trade dates with calendar events."""
    print("=" * 90)
    print("HARD STOP TRADES ON OR NEAR CALENDAR EVENT DAYS")
    print("=" * 90)

    # For each hard_stop trade, check if entry_date is day_before, day_of, or day_after an event
    event_dates: Set[date] = set()
    for d_str in CALENDAR_EVENTS:
        event_dates.add(pd.Timestamp(d_str).date())

    # Tag each trade
    on_event = []
    near_event = []  # day before or after
    no_event = []

    for _, row in df.iterrows():
        d = row['entry_date']
        if d in event_dates:
            events = get_events_for_date(str(d))
            on_event.append((row, events, 'day_of'))
        else:
            # Check day before and day after (trading days)
            d_prev = d - timedelta(days=1)
            d_next = d + timedelta(days=1)
            # Also check 2 days before/after (weekends)
            nearby = set()
            for offset in range(-3, 4):
                check = d + timedelta(days=offset)
                if check in event_dates and check != d:
                    nearby.add(check)
            if nearby:
                events = []
                for nd in nearby:
                    events.extend(get_events_for_date(str(nd)))
                near_event.append((row, events, f"near({','.join(str(nd) for nd in sorted(nearby))})"))
            else:
                no_event.append(row)

    print(f"\n  On event day:    {len(on_event):>3} trades")
    print(f"  Near event day:  {len(near_event):>3} trades (within 3 calendar days)")
    print(f"  No event:        {len(no_event):>3} trades")

    on_pnl = sum(r.pnl for r, _, _ in on_event)
    near_pnl = sum(r.pnl for r, _, _ in near_event)
    no_pnl = sum(r.pnl for r in no_event)
    print(f"\n  On event day P&L:   ${on_pnl:+,.0f}  (avg ${on_pnl/len(on_event):+,.0f}/trade)" if on_event else "")
    print(f"  Near event day P&L: ${near_pnl:+,.0f}  (avg ${near_pnl/len(near_event):+,.0f}/trade)" if near_event else "")
    print(f"  No event P&L:       ${no_pnl:+,.0f}  (avg ${no_pnl/len(no_event):+,.0f}/trade)" if no_event else "")

    # Detail the on-event trades
    if on_event:
        print(f"\n  {'Date':<12} {'Scanner':<8} {'Dir':<6} {'P&L':>10} {'VIX':>5} {'Events'}")
        print(f"  {'-'*80}")
        for row, events, tag in sorted(on_event, key=lambda x: x[0].entry_date):
            ev_str = ', '.join(f"{et}({desc[:30]})" for et, desc in events)
            print(f"  {row.entry_date!s:<12} {row.scanner:<8} {row.direction:<6} ${row.pnl:>+9,.0f} "
                  f"{row.prev_vix:>5.1f} {ev_str}")

    # By event type
    print(f"\n{'='*90}")
    print("HARD STOP TRADES BY NEARBY EVENT TYPE")
    print(f"{'='*90}")
    event_type_counts = defaultdict(lambda: {'count': 0, 'pnl': 0})
    for row, events, tag in on_event + near_event:
        for et, desc in events:
            event_type_counts[et]['count'] += 1
            event_type_counts[et]['pnl'] += row.pnl

    print(f"  {'Event Type':<16} {'HS Count':>8} {'Total P&L':>12} {'Avg P&L':>10}")
    print(f"  {'-'*50}")
    for et in sorted(event_type_counts.keys(), key=lambda k: event_type_counts[k]['pnl']):
        d = event_type_counts[et]
        print(f"  {et:<16} {d['count']:>8} ${d['pnl']:>+10,.0f} ${d['pnl']/d['count']:>+9,.0f}")


def analyze_hourly_volatility_around_events(
    tsla_5min_path: str = 'data/TSLAMin.txt',
    start: str = '2025-01-02',
    end: str = '2026-02-27',
) -> None:
    """Analyze TSLA hourly volatility on event days vs non-event days."""
    import pytz

    print(f"\n\n{'='*90}")
    print("TSLA HOURLY VOLATILITY: EVENT DAYS vs NON-EVENT DAYS")
    print(f"{'='*90}")

    # Load TSLA 1-min data
    et_tz = pytz.timezone('US/Eastern')
    df1m = pd.read_csv(tsla_5min_path, sep=';', header=None,
                        names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                        parse_dates=['datetime'], date_format='%Y%m%d %H%M%S',
                        index_col='datetime')
    df1m.index = pd.to_datetime(df1m.index).tz_localize('UTC').tz_convert(et_tz)
    df1m = df1m[start:end]

    # Filter RTH
    hours = df1m.index.hour
    minutes = df1m.index.minute
    rth_mask = ((hours > 9) | ((hours == 9) & (minutes >= 30))) & (hours < 16)
    df1m = df1m[rth_mask]

    # Resample to hourly bars
    df1h = df1m.resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df1h['range_pct'] = (df1h['high'] - df1h['low']) / df1h['open'] * 100
    df1h['date'] = df1h.index.date
    df1h['hour'] = df1h.index.hour

    # Build event date sets
    all_event_dates = set(pd.Timestamp(d).date() for d in CALENDAR_EVENTS)
    high_impact_dates = set(pd.Timestamp(d).date() for d in CALENDAR_HIGH_IMPACT)

    # Tag each hourly bar
    df1h['is_event'] = df1h['date'].apply(lambda d: d in all_event_dates)
    df1h['is_high_impact'] = df1h['date'].apply(lambda d: d in high_impact_dates)

    # Day before/after event
    day_before_event = set()
    day_after_event = set()
    for d in all_event_dates:
        for offset in [1, 2, 3]:  # account for weekends
            prev = d - timedelta(days=offset)
            nxt = d + timedelta(days=offset)
            if prev.weekday() < 5:
                day_before_event.add(prev)
                break
        for offset in [1, 2, 3]:
            nxt = d + timedelta(days=offset)
            if nxt.weekday() < 5:
                day_after_event.add(nxt)
                break

    df1h['is_day_before'] = df1h['date'].apply(lambda d: d in day_before_event)
    df1h['is_day_after'] = df1h['date'].apply(lambda d: d in day_after_event)

    # Hourly volatility comparison
    print(f"\n  Hour (ET)  | Non-Event Avg Range% | Event Day Avg Range% | High-Impact Avg Range%")
    print(f"  {'-'*85}")
    for h in range(9, 16):
        hbar = df1h[df1h['hour'] == h]
        non_ev = hbar[~hbar['is_event']]['range_pct']
        ev = hbar[hbar['is_event']]['range_pct']
        hi = hbar[hbar['is_high_impact']]['range_pct']
        print(f"  {h:>2}:00-{h:>2}:59 | {non_ev.mean():>18.3f}%  ({non_ev.count():>3}d) "
              f"| {ev.mean():>18.3f}%  ({ev.count():>3}d) "
              f"| {hi.mean():>20.3f}%  ({hi.count():>3}d)")

    # Day-before vs day-of vs day-after
    print(f"\n\n  {'Category':<20} {'Avg Range%':>12} {'Median':>10} {'Trading Days':>14}")
    print(f"  {'-'*60}")
    for label, mask in [
        ('Non-event', ~df1h['is_event'] & ~df1h['is_day_before'] & ~df1h['is_day_after']),
        ('Day before event', df1h['is_day_before']),
        ('Event day', df1h['is_event']),
        ('Day after event', df1h['is_day_after']),
        ('High-impact day', df1h['is_high_impact']),
    ]:
        subset = df1h[mask]['range_pct']
        n_days = df1h[mask]['date'].nunique()
        print(f"  {label:<20} {subset.mean():>10.3f}% {subset.median():>10.3f}% {n_days:>12}")

    # By event type: daily range on event day
    print(f"\n\n{'='*90}")
    print("TSLA DAILY RANGE BY EVENT TYPE")
    print(f"{'='*90}")
    # Get daily range
    daily = df1m.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    daily['range_pct'] = (daily['high'] - daily['low']) / daily['open'] * 100
    daily['date'] = daily.index.date

    event_type_ranges: Dict[str, List[float]] = defaultdict(list)
    for d_str, events in CALENDAR_EVENTS.items():
        d = pd.Timestamp(d_str).date()
        row = daily[daily['date'] == d]
        if len(row):
            rng = row['range_pct'].iloc[0]
            for et, desc in events:
                event_type_ranges[et].append(rng)

    non_event_daily = daily[~daily['date'].isin(all_event_dates)]['range_pct']
    print(f"  {'Event Type':<20} {'Avg Range%':>12} {'Median':>10} {'Count':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Non-event':<20} {non_event_daily.mean():>10.3f}% {non_event_daily.median():>10.3f}% {len(non_event_daily):>6}")
    for et in sorted(event_type_ranges.keys(),
                     key=lambda k: np.mean(event_type_ranges[k]), reverse=True):
        ranges = event_type_ranges[et]
        print(f"  {et:<20} {np.mean(ranges):>10.3f}% {np.median(ranges):>10.3f}% {len(ranges):>6}")


def main():
    hs_df = load_hard_stop_csv()
    print(f"Loaded {len(hs_df)} hard_stop trades\n")

    cross_reference_hard_stops_with_events(hs_df)
    analyze_hourly_volatility_around_events()


if __name__ == '__main__':
    main()
