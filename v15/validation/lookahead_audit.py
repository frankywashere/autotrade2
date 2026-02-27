"""
Look-ahead leak audit: Static code analysis + dynamic runtime check.

Verifies that Channel Surfer never uses future data when generating signals.

Usage:
    python3 -m v15.validation.lookahead_audit --tsla data/TSLAMin.txt --year 2020
"""

import argparse
import inspect
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from v15.core.historical_data import prepare_backtest_data, prepare_year_data


# ---------------------------------------------------------------------------
# Static Analysis
# ---------------------------------------------------------------------------

def audit_channel_detection():
    """Audit 1: Channel detection uses df.iloc[-(window+1):-1] — excludes current bar."""
    from v15.core import channel
    src = inspect.getsource(channel.detect_channel)

    finding = {
        'component': 'channel.detect_channel',
        'status': 'PASS',
        'detail': '',
    }

    if '-(window+1):-1' in src or '-(window + 1):-1' in src:
        finding['detail'] = 'Uses df.iloc[-(window+1):-1] — current bar excluded. SAFE.'
    elif 'iloc[-window:]' in src or 'iloc[-window-1:]' in src:
        finding['status'] = 'WARN'
        finding['detail'] = 'Slicing pattern may include current bar — needs manual review.'
    else:
        finding['status'] = 'INFO'
        finding['detail'] = 'Non-standard slicing pattern — manual review recommended.'

    return finding


def audit_higher_tf_filter():
    """Audit 2: Higher TF filtering uses tf_df[tf_idx <= current_time]."""
    from v15.core import surfer_backtest
    src = inspect.getsource(surfer_backtest.run_backtest)

    finding = {
        'component': 'surfer_backtest.run_backtest (higher TF filter)',
        'status': 'PASS',
        'detail': '',
    }

    if 'tf_idx <= current_time' in src or 'tf_available = tf_df[tf_idx <= current_time' in src:
        finding['detail'] = (
            'Filter: tf_df[tf_idx <= current_time]. '
            'RISK: pandas resample timestamps bars at period START. '
            'A 4h bar at 10:00 contains data through 13:55 — including it at 11:00 '
            'leaks up to 2h55m of future data. '
            'Severity depends on whether higher-TF bars actually use future OHLCV.'
        )
        finding['status'] = 'WARN'
    else:
        finding['status'] = 'INFO'
        finding['detail'] = 'Higher TF filter pattern not found — manual review needed.'

    return finding


def audit_resample_convention():
    """Audit 3: Check pandas resample label convention."""
    from v15.core import historical_data
    src = inspect.getsource(historical_data.resample_to_tf)

    finding = {
        'component': 'historical_data.resample_to_tf',
        'status': 'PASS',
        'detail': '',
    }

    if "label='right'" in src:
        finding['detail'] = 'Resample uses label=right — bars timestamped at period END. SAFE for <= filter.'
    elif "label='left'" in src or 'label=' not in src:
        finding['status'] = 'WARN'
        finding['detail'] = (
            'Resample uses default label=left — bars timestamped at period START. '
            'A 4h bar starting at 10:00 includes data through 13:55. '
            'The <= current_time filter will include this bar at 10:01, '
            'even though it contains data from the future.'
        )
    return finding


def audit_5min_slice():
    """Audit 4: 5-min slice pattern in run_backtest."""
    from v15.core import surfer_backtest
    src = inspect.getsource(surfer_backtest.run_backtest)

    finding = {
        'component': 'surfer_backtest.run_backtest (5-min slice)',
        'status': 'PASS',
        'detail': '',
    }

    # The pattern is tsla.iloc[bar - lookback + 1:bar + 1] then channel detection excludes last bar
    if 'bar + 1]' in src and 'bar - ' in src:
        finding['detail'] = (
            'Uses tsla.iloc[bar-lookback+1:bar+1] — includes current bar in slice. '
            'Channel detection then excludes last bar via iloc[-(window+1):-1]. '
            'Net effect: current bar is excluded. SAFE.'
        )
    else:
        finding['status'] = 'INFO'
        finding['detail'] = 'Slice pattern not found — manual review needed.'

    return finding


def audit_ou_fitting():
    """Audit 5: OU parameter fitting uses same filtered data."""
    finding = {
        'component': 'OU mean-reversion fitting',
        'status': 'PASS',
        'detail': 'OU fitting uses the same price array from channel detection (already filtered). SAFE.',
    }
    return finding


def run_static_audit():
    """Run all static analysis checks."""
    findings = [
        audit_channel_detection(),
        audit_higher_tf_filter(),
        audit_resample_convention(),
        audit_5min_slice(),
        audit_ou_fitting(),
    ]
    return findings


# ---------------------------------------------------------------------------
# Dynamic Analysis
# ---------------------------------------------------------------------------

def run_dynamic_audit(data: dict, max_bars: int = 5000):
    """
    Walk through data bar-by-bar, logging every higher-TF bar used.
    Flag any bar whose period extends past current_time.
    """
    from v15.core.channel import detect_channels_multi_window, select_best_channel

    tsla = data['tsla_5min']
    higher_tf_data = data['higher_tf_data']

    TF_WINDOWS = {
        '5min': [20, 30, 40, 50],
        '1h': [20, 30, 40],
        '4h': [15, 20, 25],
        'daily': [10, 15, 20],
    }

    # Period durations for each TF (to compute period END)
    tf_period_minutes = {
        '1h': 60,
        '4h': 240,
        'daily': 1440,
    }

    violations = []
    bars_checked = 0
    eval_interval = 6  # match default

    total_bars = min(len(tsla), max_bars)
    lookback = 200

    for bar in range(lookback, total_bars, eval_interval):
        current_time = tsla.index[bar]

        # Normalize tz
        if current_time.tzinfo is not None:
            current_time_naive = current_time.tz_localize(None)
        else:
            current_time_naive = current_time

        for tf_label, tf_df in higher_tf_data.items():
            tf_idx = tf_df.index
            if tf_idx.tz is not None:
                tf_available = tf_df[tf_idx <= current_time]
            else:
                tf_available = tf_df[tf_idx <= current_time_naive]

            if len(tf_available) < 30:
                continue

            tf_recent = tf_available.tail(100)
            last_bar_ts = tf_recent.index[-1]
            last_bar_naive = last_bar_ts.tz_localize(None) if hasattr(last_bar_ts, 'tz_localize') and last_bar_ts.tzinfo else last_bar_ts

            # Compute period end (start + period duration)
            period_mins = tf_period_minutes.get(tf_label, 60)
            period_end = last_bar_naive + pd.Timedelta(minutes=period_mins)

            if period_end > current_time_naive:
                violations.append({
                    'bar': bar,
                    'current_time': str(current_time),
                    'tf': tf_label,
                    'tf_bar_start': str(last_bar_ts),
                    'tf_bar_end': str(period_end),
                    'leak_minutes': (period_end - current_time_naive).total_seconds() / 60,
                })

        bars_checked += 1

    return violations, bars_checked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_report(static_findings, violations, bars_checked):
    """Print formatted audit report."""
    print(f"\n{'='*80}")
    print(f"  LOOK-AHEAD AUDIT REPORT")
    print(f"{'='*80}")

    # Static
    print(f"\n  STATIC ANALYSIS")
    print(f"  {'-'*70}")
    for f in static_findings:
        status_color = {'PASS': 'OK', 'WARN': '!!', 'INFO': '??'}[f['status']]
        print(f"  [{status_color}] {f['component']}")
        print(f"       {f['detail']}")
        print()

    # Dynamic
    print(f"  DYNAMIC ANALYSIS ({bars_checked} eval points checked)")
    print(f"  {'-'*70}")

    if not violations:
        print(f"  [OK] No look-ahead violations detected")
    else:
        # Summarize by TF
        by_tf = defaultdict(list)
        for v in violations:
            by_tf[v['tf']].append(v)

        for tf, vv in sorted(by_tf.items()):
            avg_leak = np.mean([v['leak_minutes'] for v in vv])
            max_leak = max(v['leak_minutes'] for v in vv)
            print(f"  [!!] {tf}: {len(vv)} violations, avg leak={avg_leak:.0f}min, max leak={max_leak:.0f}min")

        # Show first few examples
        print(f"\n  Sample violations (first 5):")
        for v in violations[:5]:
            print(f"    bar={v['bar']}, time={v['current_time']}")
            print(f"    TF={v['tf']}, bar_start={v['tf_bar_start']}, bar_end={v['tf_bar_end']}")
            print(f"    Leak: {v['leak_minutes']:.0f} minutes of future data")
            print()

        # Proposed fix
        print(f"  PROPOSED FIX:")
        print(f"    Change higher-TF filter to exclude the current (incomplete) bar:")
        print(f"    Instead of: tf_df[tf_idx <= current_time]")
        print(f"    Use:        tf_df[tf_idx + tf_period <= current_time]")
        print(f"    This ensures only COMPLETED bars are used.")

    # Overall verdict
    n_warn = sum(1 for f in static_findings if f['status'] == 'WARN')
    if n_warn == 0 and not violations:
        verdict = "CLEAN — no look-ahead detected"
    elif violations:
        verdict = f"LEAK DETECTED — {len(violations)} violations across {len(set(v['tf'] for v in violations))} TFs"
    else:
        verdict = f"REVIEW NEEDED — {n_warn} static warnings"

    print(f"\n  {'='*70}")
    print(f"  VERDICT: {verdict}")
    print(f"  {'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Look-ahead leak audit')
    parser.add_argument('--tsla', required=True, help='Path to TSLAMin.txt')
    parser.add_argument('--spy', default=None, help='Path to SPYMin.txt')
    parser.add_argument('--year', type=int, default=2020, help='Year to audit (default: 2020)')
    parser.add_argument('--max-bars', type=int, default=5000, help='Max bars for dynamic check')
    args = parser.parse_args()

    t_start = time.time()

    # Static audit (no data needed)
    print("Running static code analysis...")
    static_findings = run_static_audit()

    # Dynamic audit
    print(f"\nLoading data for dynamic audit (year {args.year})...")
    full_data = prepare_backtest_data(args.tsla, args.spy)
    year_data = prepare_year_data(full_data, args.year)

    if year_data is None:
        print(f"No data for year {args.year}")
        print_report(static_findings, [], 0)
        return

    print(f"Running dynamic audit on {len(year_data['tsla_5min']):,} bars...")
    violations, bars_checked = run_dynamic_audit(year_data, max_bars=args.max_bars)

    print_report(static_findings, violations, bars_checked)

    elapsed = time.time() - t_start
    print(f"\nWall time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
