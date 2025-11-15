#!/usr/bin/env python3
"""
Event Data Validation Script

Validates completeness and quality of event data in tsla_events_REAL.csv.

Usage:
    python validate_event_data.py

Checks:
- Event coverage for training period (2015-2022)
- Future event coverage
- Data quality (missing values, gaps)
- Quarterly earnings/delivery completeness
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add parent to path
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

import config


def print_header(text):
    """Print section header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def validate_event_coverage(df, start_year, end_year):
    """Validate quarterly earnings and delivery coverage"""

    years = range(start_year, end_year + 1)
    expected_per_year = 4  # Quarterly events

    earnings_missing = []
    delivery_missing = []

    for year in years:
        # Check earnings
        year_earnings = df[(df['date'].dt.year == year) &
                           (df['event_type'] == 'earnings')]
        if len(year_earnings) < expected_per_year:
            earnings_missing.append(f"{year}: {len(year_earnings)}/4")

        # Check deliveries
        year_deliveries = df[(df['date'].dt.year == year) &
                            (df['event_type'] == 'delivery')]
        if len(year_deliveries) < expected_per_year:
            delivery_missing.append(f"{year}: {len(year_deliveries)}/4")

    return earnings_missing, delivery_missing


def main():
    """Run event data validation"""

    print_header("EVENT DATA VALIDATION")

    print(f"\n📅 Validation Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load event file
    events_file = config.TSLA_EVENTS_FILE

    if not Path(events_file).exists():
        print(f"\n❌ Event file not found: {events_file}")
        print(f"\n   Please ensure tsla_events_REAL.csv exists in data/ directory")
        return 1

    print(f"\n📂 Loading: {Path(events_file).name}")

    df = pd.read_csv(events_file)
    df['date'] = pd.to_datetime(df['date'])

    print(f"   Total events: {len(df)}")
    print(f"   Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    # Event type breakdown
    print_header("Event Type Breakdown")

    event_counts = df['event_type'].value_counts()
    for event_type, count in event_counts.items():
        print(f"  {event_type.ljust(15)}: {count:3d} events")

    # Training period coverage (2015-2022)
    print_header("Training Period Coverage (2015-2022)")

    training_df = df[(df['date'].dt.year >= 2015) & (df['date'].dt.year <= 2022)]

    print(f"\n  Training period events: {len(training_df)}")
    print(f"\n  Event breakdown:")
    for event_type, count in training_df['event_type'].value_counts().items():
        print(f"    {event_type.ljust(15)}: {count:3d}")

    # Check quarterly completeness
    earnings_missing, delivery_missing = validate_event_coverage(df, 2015, 2022)

    print(f"\n  Quarterly Coverage:")
    if not earnings_missing:
        print(f"    ✅ Earnings: Complete (all quarters covered)")
    else:
        print(f"    ⚠️  Earnings: Missing quarters - {', '.join(earnings_missing)}")

    if not delivery_missing:
        print(f"    ✅ Deliveries: Complete (all quarters covered)")
    else:
        print(f"    ⚠️  Deliveries: Missing quarters - {', '.join(delivery_missing)}")

    # Future coverage
    print_header("Future Event Coverage")

    max_date = df['date'].max()
    current_date = datetime.now()
    days_remaining = (max_date - current_date).days

    print(f"\n  CSV ends: {max_date.strftime('%Y-%m-%d')}")
    print(f"  Current date: {current_date.strftime('%Y-%m-%d')}")
    print(f"  Days remaining: {days_remaining}")

    if days_remaining < 0:
        print(f"\n  ❌ CRITICAL: CSV is OUTDATED ({abs(days_remaining)} days ago)")
        print(f"  ⚠️  Event features will be zeros for current predictions!")
        print(f"  ⚠️  Update CSV immediately for live trading")
    elif days_remaining < 90:
        print(f"\n  ⚠️  WARNING: Less than 3 months remaining")
        print(f"  ⚠️  Update CSV soon with {max_date.year + 1} events")
    elif days_remaining < 180:
        print(f"\n  ⚠️  NOTICE: Less than 6 months remaining")
        print(f"  ℹ️  Plan to update CSV by {(max_date - timedelta(days=90)).strftime('%Y-%m-%d')}")
    else:
        print(f"\n  ✅ Coverage is current ({days_remaining} days remaining)")

    # Data quality checks
    print_header("Data Quality")

    # Check for missing values
    missing_by_col = df.isnull().sum()
    if missing_by_col.any():
        print(f"\n  ⚠️  Missing values detected:")
        for col, count in missing_by_col[missing_by_col > 0].items():
            print(f"    {col}: {count} missing")
    else:
        print(f"\n  ✅ No missing values")

    # Check beat/miss data for earnings
    earnings_df = df[df['event_type'] == 'earnings']
    has_beat_miss = earnings_df['beat_miss'].notna().sum()
    print(f"\n  Beat/Miss Data:")
    print(f"    Earnings with beat/miss: {has_beat_miss}/{len(earnings_df)} ({has_beat_miss/len(earnings_df)*100:.0f}%)")

    # Final summary
    print_header("SUMMARY")

    all_good = (
        len(earnings_missing) == 0 and
        len(delivery_missing) == 0 and
        days_remaining >= 90 and
        not df.isnull().any().any()
    )

    if all_good:
        print(f"\n  ✅ EVENT DATA IS PRODUCTION READY")
        print(f"\n  Event data is complete and current.")
        print(f"  Safe to use for training and live trading.")
        print(f"\n  Next update needed: {(max_date - timedelta(days=90)).strftime('%Y-%m-%d')}")
        exit_code = 0
    else:
        print(f"\n  ⚠️  EVENT DATA NEEDS ATTENTION")
        print(f"\n  Review warnings above and update CSV as needed.")
        print(f"  See SPEC.md 'Event Data Maintenance' section for update instructions.")
        exit_code = 1

    print(f"\n{'='*70}")
    print(f"📅 Validation Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    return exit_code


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Validation failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
