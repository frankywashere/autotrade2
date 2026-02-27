#!/usr/bin/env python3
"""
Confluence Oversold Bounce Timing Study

When higher TFs all show oversold, how many hours/days until the bounce
actually materializes?  If the answer is "almost never same-day," the
override should delay (or use require_stabilizing) rather than flip
immediately.

Reuses load_all_tfs / compute_daily_states / signal helpers from
tf_state_backtest.py and adds a forward-return engine focused on
bounce timing (hours to +1%, +2%, +3%, +5%) with max-drawdown tracking.

Usage:
    python -m v15.validation.bounce_timing
    python -m v15.validation.bounce_timing --tsla data/TSLAMin.txt
    python -m v15.validation.bounce_timing --start 2020 --end 2025
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.tf_state_backtest import (
    load_all_tfs,
    compute_daily_states,
    _mt,
    _count_near_bottom,
    _norm_cols,
)


# ── SPY RSI helper ──────────────────────────────────────────────────────────

def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI from a close price series."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _load_spy_rsi(start: str, end: str) -> pd.Series:
    """Load SPY daily close and compute 14-period RSI, indexed by date."""
    from v15.data.native_tf import fetch_native_tf
    spy = _norm_cols(fetch_native_tf('SPY', 'daily', start, end))
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    return _compute_rsi(spy['close'], 14)

# ── Signal definitions (6 signals, increasing strictness) ───────────────────

BOUNCE_SIGNALS = [
    ('S1 daily_NB',
     'daily near_bottom alone',
     lambda s: s.get('daily') and s['daily']['near_bottom']),

    ('S2 daily_NB + weekly_NB',
     'daily + weekly both oversold',
     lambda s: (s.get('daily') and s['daily']['near_bottom'] and
                s.get('weekly') and s['weekly']['near_bottom'])),

    ('S3 daily_NB + weekly_NB + monthly_NB',
     'daily + weekly + monthly all oversold',
     lambda s: (s.get('daily') and s['daily']['near_bottom'] and
                s.get('weekly') and s['weekly']['near_bottom'] and
                s.get('monthly') and s['monthly']['near_bottom'])),

    ('S4 daily_NB + weekly_NB + daily_MT',
     'daily+weekly oversold + daily momentum turning',
     lambda s: (s.get('daily') and s['daily']['near_bottom'] and
                s.get('weekly') and s['weekly']['near_bottom'] and
                _mt(s, 'daily'))),

    ('S5 daily_NB + weekly_NB + monthly_NB + daily_MT',
     'all 3 higher TFs oversold + daily MT',
     lambda s: (s.get('daily') and s['daily']['near_bottom'] and
                s.get('weekly') and s['weekly']['near_bottom'] and
                s.get('monthly') and s['monthly']['near_bottom'] and
                _mt(s, 'daily'))),

    ('S6 4+ TFs near_bottom',
     'strong confluence (now incl monthly)',
     lambda s: _count_near_bottom(s) >= 4),

    ('S7 5+ TFs near_bottom',
     'very strong confluence (incl monthly)',
     lambda s: _count_near_bottom(s) >= 5),

    ('S8 daily_NB + weekly_NB + weekly_MT',
     'weekly momentum confirming',
     lambda s: (s.get('daily') and s['daily']['near_bottom'] and
                s.get('weekly') and s['weekly']['near_bottom'] and
                _mt(s, 'weekly'))),

    ('S9 daily_NB + weekly_NB + monthly_NB + weekly_MT',
     'strongest — all 3 higher TFs oversold + weekly MT',
     lambda s: (s.get('daily') and s['daily']['near_bottom'] and
                s.get('weekly') and s['weekly']['near_bottom'] and
                s.get('monthly') and s['monthly']['near_bottom'] and
                _mt(s, 'weekly'))),
]

# ── Forward measurement engine ──────────────────────────────────────────────

TARGETS = [0.01, 0.02, 0.03, 0.05]          # +1%, +2%, +3%, +5%
FORWARD_DAYS = 20                             # max look-forward window
HOURS_PER_DAY = 6.5                           # trading hours

def _build_hourly_series(tf_data: dict) -> pd.DataFrame | None:
    """
    Build an hourly-resolution forward price series from 1h bars.
    Returns DataFrame with columns [high, low, close] indexed by datetime,
    or None if 1h data is too sparse.
    """
    df_1h = tf_data.get('1h')
    if df_1h is None or len(df_1h) < 100:
        return None
    return df_1h[['high', 'low', 'close']].copy()


def measure_forward(ref_price: float, ref_date: pd.Timestamp,
                    daily_df: pd.DataFrame, hourly: pd.DataFrame | None) -> dict:
    """
    From ref_price on ref_date (daily close), walk forward starting NEXT DAY
    up to FORWARD_DAYS trading days.
    Returns dict with:
      hours_to_target: {0.01: hours or NaN, ...}
      max_drawdown:    worst drawdown before any bounce
    """
    # Start from next calendar day to avoid look-ahead (ref_price = daily close)
    next_day = ref_date + pd.Timedelta(days=1)
    end_date = ref_date + pd.Timedelta(days=FORWARD_DAYS * 2)  # calendar days
    result = {t: np.nan for t in TARGETS}
    max_dd = 0.0
    min_price_seen = ref_price

    # Try hourly resolution first
    if hourly is not None:
        fwd = hourly.loc[next_day: end_date]
        if len(fwd) > 0:
            cum_hours = 0.0
            for ts, row in fwd.iterrows():
                cum_hours += 1.0  # each 1h bar = 1 trading hour
                bar_low = row['low']
                bar_high = row['high']

                # Track drawdown
                if bar_low < min_price_seen:
                    min_price_seen = bar_low
                    dd = (min_price_seen / ref_price) - 1.0
                    if dd < max_dd:
                        max_dd = dd

                # Check targets (use high for upside targets)
                for t in TARGETS:
                    if np.isnan(result[t]) and bar_high >= ref_price * (1.0 + t):
                        result[t] = cum_hours

                # Stop once all targets hit or past window
                if cum_hours > FORWARD_DAYS * HOURS_PER_DAY:
                    break
                if all(not np.isnan(result[t]) for t in TARGETS):
                    break

            return {'hours_to_target': result, 'max_drawdown': max_dd}

    # Fallback: daily bars
    fwd_daily = daily_df.loc[next_day: end_date]
    cum_days = 0
    for ts, row in fwd_daily.iterrows():
        cum_days += 1
        if cum_days > FORWARD_DAYS:
            break
        bar_low = row['low']
        bar_high = row['high']

        if bar_low < min_price_seen:
            min_price_seen = bar_low
            dd = (min_price_seen / ref_price) - 1.0
            if dd < max_dd:
                max_dd = dd

        for t in TARGETS:
            if np.isnan(result[t]) and bar_high >= ref_price * (1.0 + t):
                # Approximate mid-day hit as half-day worth of hours
                result[t] = (cum_days - 0.5) * HOURS_PER_DAY

        if all(not np.isnan(result[t]) for t in TARGETS):
            break

    return {'hours_to_target': result, 'max_drawdown': max_dd}


# ── Formatting helpers ──────────────────────────────────────────────────────

def _fmt_hours(h: float) -> str:
    """Format hours as Xh or Xd for readability."""
    if np.isnan(h):
        return '  --  '
    if h < HOURS_PER_DAY * 1.5:
        return f'{h:>5.0f}h'
    days = h / HOURS_PER_DAY
    return f'{days:>4.1f}d'


def _classify_timing(h: float) -> str:
    """Classify as same-day / next-day / 2+ days."""
    if np.isnan(h):
        return 'miss'
    if h <= HOURS_PER_DAY:
        return 'same-day'
    elif h <= HOURS_PER_DAY * 2:
        return 'next-day'
    else:
        return '2+days'


# ── SPY RSI split analysis ───────────────────────────────────────────────────

def _print_rsi_split(edf: pd.DataFrame, rsi_vals: np.ndarray):
    """Print bounce timing split by SPY RSI regime."""
    # Three buckets: oversold (<35), neutral (35-65), overbought (>65)
    buckets = [
        ('SPY RSI < 35 (broad washout)', rsi_vals < 35),
        ('SPY RSI 35-65 (neutral)',       (rsi_vals >= 35) & (rsi_vals <= 65)),
        ('SPY RSI > 65 (strong market)',  rsi_vals > 65),
    ]

    print(f"\n  SPY RSI CORRELATION:")
    for label, mask in buckets:
        n = int(np.sum(mask & ~np.isnan(rsi_vals)))
        if n < 2:
            print(f"    {label}: n={n} (too few)")
            continue

        sub = edf[mask & edf['spy_rsi'].notna()]
        # Compute hit rates and median time for +2% target
        col = 'hrs_2pct'
        vals = sub[col].values
        n_hit = int(np.sum(~np.isnan(vals)))
        hit_pct = n_hit / len(vals) if len(vals) > 0 else 0
        valid = vals[~np.isnan(vals)]
        med = _fmt_hours(np.median(valid)) if len(valid) > 0 else '--'
        # Drawdown
        dds = sub['max_dd'].values
        med_dd = f"{np.median(dds):>+.1%}"
        # Same-day rate for +1%
        vals_1 = sub['hrs_1pct'].values
        same_day_1 = sum(1 for v in vals_1 if not np.isnan(v) and v <= HOURS_PER_DAY) / len(vals_1) if len(vals_1) > 0 else 0

        print(f"    {label}: n={n}")
        print(f"      +2% hit={hit_pct:.0%} median={med}  "
              f"+1% same-day={same_day_1:.0%}  "
              f"median DD={med_dd}")


# ── Main analysis ───────────────────────────────────────────────────────────

def run_bounce_analysis(tf_data: dict, state_rows: list, daily_df: pd.DataFrame,
                        spy_rsi: pd.Series | None = None):
    """Run the bounce timing study for all signals, with optional SPY RSI split."""

    hourly = _build_hourly_series(tf_data)
    hourly_note = 'hourly resolution' if hourly is not None else 'daily resolution only'
    print(f"\nForward measurement resolution: {hourly_note}")
    if spy_rsi is not None:
        print(f"SPY RSI loaded: {spy_rsi.dropna().index[0].strftime('%Y-%m-%d')} to "
              f"{spy_rsi.dropna().index[-1].strftime('%Y-%m-%d')}")

    # Date range of state data
    dates = [r['date'] for r in state_rows]
    print(f"State data: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
          f" ({len(dates)} trading days)")

    # Build date→state lookup for pos_pct reporting
    state_by_date = {}
    for row in state_rows:
        state_by_date[row['date']] = row

    for sig_name, sig_desc, sig_fn in BOUNCE_SIGNALS:
        # Find all dates this signal fires
        fire_dates = []
        for row in state_rows:
            states = {tf: row.get(tf) for tf in ['5min', '1h', '4h', 'daily', 'weekly', 'monthly']}
            try:
                if sig_fn(states):
                    fire_dates.append(row['date'])
            except Exception:
                continue

        # Deduplicate: require at least 5 trading days between signals
        deduped = []
        for d in fire_dates:
            if not deduped or (d - deduped[-1]).days >= 7:
                deduped.append(d)
        fire_dates = deduped

        print(f"\n{'='*75}")
        print(f"SIGNAL: {sig_name}  ({sig_desc})")
        print(f"  n={len(fire_dates)}, {dates[0].year}-{dates[-1].year}")
        print(f"{'='*75}")

        if len(fire_dates) < 3:
            print("  Too few events for meaningful statistics.\n")
            continue

        # Measure forward returns for each event
        events = []
        for d in fire_dates:
            if d not in daily_df.index:
                # Find nearest date
                idx = daily_df.index.searchsorted(d)
                if idx >= len(daily_df):
                    continue
                d = daily_df.index[idx]
            ref_price = daily_df.loc[d, 'close']
            fwd = measure_forward(ref_price, d, daily_df, hourly)
            # Grab pos_pct for higher TFs
            row_state = state_by_date.get(d, {})
            pos = {}
            for tf in ['daily', 'weekly', 'monthly']:
                s = row_state.get(tf)
                pos[f'{tf}_pos'] = s['pos_pct'] if s else np.nan
                pos[f'{tf}_mt'] = s['is_turning'] if s else False
            # SPY RSI on signal date
            spy_rsi_val = np.nan
            if spy_rsi is not None:
                idx = spy_rsi.index.searchsorted(d)
                if 0 < idx <= len(spy_rsi):
                    spy_rsi_val = spy_rsi.iloc[idx - 1]
            events.append({
                'date': d,
                'close': ref_price,
                **{f'hrs_{int(t*100)}pct': fwd['hours_to_target'][t] for t in TARGETS},
                'max_dd': fwd['max_drawdown'],
                'spy_rsi': spy_rsi_val,
                **pos,
            })

        if not events:
            print("  No measurable events.\n")
            continue

        edf = pd.DataFrame(events)

        # ── Time to bounce table ────────────────────────────────────────
        print(f"\nTIME TO BOUNCE:")
        print(f"  {'Target':<8} {'Hit%':>5} {'Median':>8} {'Mean':>8} "
              f"{'P25':>8} {'P75':>8} {'Same-day':>9} {'Next-day':>9}")
        print(f"  {'------':<8} {'-----':>5} {'-------':>8} {'-------':>8} "
              f"{'-------':>8} {'-------':>8} {'---------':>9} {'---------':>9}")

        for t in TARGETS:
            col = f'hrs_{int(t*100)}pct'
            vals = edf[col].values
            n_hit = int(np.sum(~np.isnan(vals)))
            hit_pct = n_hit / len(vals)
            valid = vals[~np.isnan(vals)]

            if len(valid) == 0:
                print(f"  +{int(t*100)}%{'':<4} {hit_pct:>4.0%}   {'--':>8} {'--':>8} "
                      f"{'--':>8} {'--':>8} {'--':>9} {'--':>9}")
                continue

            med = np.median(valid)
            mean = np.mean(valid)
            p25 = np.percentile(valid, 25)
            p75 = np.percentile(valid, 75)

            classes = [_classify_timing(v) for v in vals]
            same_day = sum(1 for c in classes if c == 'same-day') / len(vals)
            next_day = sum(1 for c in classes if c == 'next-day') / len(vals)

            print(f"  +{int(t*100)}%{'':<4} {hit_pct:>4.0%}  {_fmt_hours(med):>8} "
                  f"{_fmt_hours(mean):>8} {_fmt_hours(p25):>8} {_fmt_hours(p75):>8} "
                  f"{same_day:>8.0%}  {next_day:>8.0%}")

        # ── Drawdown before bounce ──────────────────────────────────────
        dds = edf['max_dd'].values
        pct_with_dd = np.mean(dds < -0.001) * 100
        print(f"\nDRAWDOWN BEFORE BOUNCE ({pct_with_dd:.0f}% of events saw drawdown):")
        print(f"  Median: {np.median(dds):>+.1%}   "
              f"P25 (worst 25%): {np.percentile(dds, 25):>+.1%}   "
              f"P5 (worst 5%): {np.percentile(dds, 5):>+.1%}")
        if pct_with_dd > 0:
            dd_only = dds[dds < -0.001]
            print(f"  Among events WITH drawdown (n={len(dd_only)}): "
                  f"median {np.median(dd_only):>+.1%}  "
                  f"worst {np.min(dd_only):>+.1%}")

        # ── Individual events (most recent 20) ──────────────────────────
        recent = edf.tail(20).iloc[::-1]
        has_rsi = spy_rsi is not None and edf['spy_rsi'].notna().any()
        print(f"\nINDIVIDUAL EVENTS (most recent {len(recent)}):")
        hdr = (f"  {'Date':<12} {'Close':>7} "
               f"{'D_pos':>5} {'W_pos':>5} {'M_pos':>5} ")
        if has_rsi:
            hdr += f"{'SPY_RSI':>7} "
        hdr += ''.join(f'+{int(t*100)}%hrs ' for t in TARGETS) + f'{"MaxDD":>7}'
        print(hdr)
        sep = (f"  {'----------':<12} {'------':>7} "
               f"{'-----':>5} {'-----':>5} {'-----':>5} ")
        if has_rsi:
            sep += f"{'-------':>7} "
        sep += ''.join(f'------- ' for _ in TARGETS) + f'-------'
        print(sep)

        for _, row in recent.iterrows():
            d_pos = f"{row['daily_pos']:.2f}" if not np.isnan(row['daily_pos']) else '  -- '
            w_pos = f"{row['weekly_pos']:.2f}" if not np.isnan(row['weekly_pos']) else '  -- '
            m_pos = f"{row['monthly_pos']:.2f}" if not np.isnan(row['monthly_pos']) else '  -- '
            # Mark MT with * suffix
            if row.get('daily_mt'):
                d_pos = d_pos[:4] + '*'
            if row.get('weekly_mt'):
                w_pos = w_pos[:4] + '*'
            if row.get('monthly_mt'):
                m_pos = m_pos[:4] + '*'
            parts = (f"  {row['date'].strftime('%Y-%m-%d'):<12} ${row['close']:>6,.0f} "
                     f"{d_pos:>5} {w_pos:>5} {m_pos:>5} ")
            if has_rsi:
                rsi_v = row['spy_rsi']
                parts += f"{rsi_v:>6.1f} " if not np.isnan(rsi_v) else f"{'--':>6} "
            for t in TARGETS:
                col = f'hrs_{int(t*100)}pct'
                parts += f"{_fmt_hours(row[col]):>7} "
            parts += f"{row['max_dd']:>+6.1%}"
            print(parts)

        # ── SPY RSI correlation split ─────────────────────────────────
        if has_rsi and len(edf) >= 10:
            rsi_vals = edf['spy_rsi'].values
            _print_rsi_split(edf, rsi_vals)

    print(f"\n{'='*75}")
    print("DONE")
    print(f"{'='*75}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Confluence oversold bounce timing study')
    parser.add_argument('--tsla', type=str, default='data/TSLAMin.txt',
                        help='Path to 1-min TSLA data (for hourly resolution)')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default='2026-12-31')
    args = parser.parse_args()

    print(f"\n{'='*75}")
    print("CONFLUENCE OVERSOLD BOUNCE TIMING STUDY")
    print(f"{'='*75}")
    print(f"Question: When higher TFs show oversold, how long until the bounce?")
    print(f"Forward window: {FORWARD_DAYS} trading days")
    print(f"Targets: {', '.join(f'+{int(t*100)}%' for t in TARGETS)}")

    # Load data
    tf_data = load_all_tfs(args.tsla, args.start, args.end)
    daily_df = tf_data['daily']

    # Load SPY RSI
    print("Loading SPY RSI...")
    spy_rsi = _load_spy_rsi(args.start, args.end)
    print(f"  SPY RSI: {len(spy_rsi.dropna())} bars")

    # Compute daily TF states
    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates)

    # Run the analysis
    run_bounce_analysis(tf_data, state_rows, daily_df, spy_rsi=spy_rsi)


if __name__ == '__main__':
    main()
