#!/usr/bin/env python3
"""
Update Event Data from APIs

Fetches future TSLA earnings, FOMC meetings, and macro events from APIs
and merges them with existing CSV data.

APIs Used (all free):
- Alpha Vantage: TSLA earnings calendar (25 calls/day)
- FRED: FOMC dates, CPI, NFP (unlimited)

Usage:
    python update_events_from_api.py                    # Update with all sources
    python update_events_from_api.py --dry-run          # Preview without saving
    python update_events_from_api.py --year 2026        # Fetch specific year

Setup:
    1. Get free Alpha Vantage key: https://www.alphavantage.co/support/#api-key
    2. Get free FRED key: https://fred.stlouisfed.org/docs/api/api_key.html
    3. Set environment variables or enter when prompted
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
import argparse
import os

# Add parent to path
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

import config
from src.ml.api_fetchers import (
    AlphaVantageClient,
    FREDClient,
    EventScheduleGenerator,
    merge_events
)


def print_header(text):
    """Print section header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def check_api_keys():
    """Prompt for API keys if not configured"""
    keys_entered = False

    # Check Alpha Vantage
    if not os.getenv("ALPHA_VANTAGE_API_KEY") and not config.ALPHA_VANTAGE_API_KEY:
        print("\n📋 Alpha Vantage API Key Required (Free)")
        print("   Get key: https://www.alphavantage.co/support/#api-key")
        key = input("   Enter Alpha Vantage API key: ").strip()
        if key:
            os.environ['ALPHA_VANTAGE_API_KEY'] = key
            config.ALPHA_VANTAGE_API_KEY = key
            keys_entered = True
        else:
            print("   ⚠️  No key entered - will skip Alpha Vantage")

    # Check FRED
    if not os.getenv("FRED_API_KEY") and not config.FRED_API_KEY:
        print("\n📋 FRED API Key Required (Free)")
        print("   Get key: https://fred.stlouisfed.org/docs/api/api_key.html")
        key = input("   Enter FRED API key: ").strip()
        if key:
            os.environ['FRED_API_KEY'] = key
            config.FRED_API_KEY = key
            keys_entered = True
        else:
            print("   ⚠️  No key entered - will skip FRED")

    if keys_entered:
        print("\n✅ API keys configured for this session")
        print("   To persist: Set environment variables or add to config/api_keys.json")


def main():
    """Main update logic"""

    parser = argparse.ArgumentParser(description="Update event data from APIs")
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without saving')
    parser.add_argument('--year', type=int, default=None,
                        help='Specific year to fetch (default: current + next year)')
    parser.add_argument('--source', choices=['all', 'alpha', 'fred', 'schedule'],
                        default='all', help='Data source to use')
    args = parser.parse_args()

    print_header("Event Data Update from APIs")

    print(f"\n📅 Update Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.dry_run:
        print("   🔍 DRY RUN MODE - No files will be modified")

    # Load existing CSV
    csv_file = config.TSLA_EVENTS_FILE

    if not Path(csv_file).exists():
        print(f"\n❌ Event file not found: {csv_file}")
        print(f"   Cannot update non-existent file")
        return 1

    print(f"\n📂 Loading existing CSV: {Path(csv_file).name}")
    existing_df = pd.read_csv(csv_file)
    existing_df['date'] = pd.to_datetime(existing_df['date'])

    print(f"   Current events: {len(existing_df)}")
    print(f"   Date range: {existing_df['date'].min().strftime('%Y-%m-%d')} to {existing_df['date'].max().strftime('%Y-%m-%d')}")

    # Check API keys
    print_header("API Key Check")
    check_api_keys()

    # Determine year to fetch
    if args.year:
        fetch_year = args.year
    else:
        # Fetch current year + next year
        fetch_year = datetime.now().year

    # Fetch from APIs
    new_events = []

    # Alpha Vantage - TSLA Earnings
    if args.source in ['all', 'alpha']:
        print_header("Fetching from Alpha Vantage (TSLA Earnings)")
        av_client = AlphaVantageClient()
        earnings = av_client.fetch_earnings_calendar('TSLA', horizon='12month')
        if not earnings.empty:
            new_events.append(earnings)

    # FRED - FOMC, CPI, NFP
    if args.source in ['all', 'fred']:
        print_header("Fetching from FRED (FOMC, CPI, NFP)")
        fred_client = FREDClient()

        # FOMC dates
        fomc = fred_client.fetch_fomc_dates(start_year=fetch_year)
        if not fomc.empty:
            new_events.append(fomc)

        # Economic releases
        econ = fred_client.fetch_economic_releases()
        if not econ.empty:
            new_events.append(econ)

    # Generate predictable schedules
    if args.source in ['all', 'schedule']:
        print_header("Generating Predictable Event Schedules")
        generator = EventScheduleGenerator()

        print(f"   Generating CPI schedule for {fetch_year}...")
        cpi = generator.generate_cpi_schedule(fetch_year)
        new_events.append(cpi)

        print(f"   Generating NFP schedule for {fetch_year}...")
        nfp = generator.generate_nfp_schedule(fetch_year)
        new_events.append(nfp)

        print(f"   Generating Quad Witching for {fetch_year}...")
        quad = generator.generate_quad_witching_schedule(fetch_year)
        new_events.append(quad)

        # Also generate for next year
        next_year = fetch_year + 1
        print(f"   Generating schedules for {next_year}...")
        new_events.append(generator.generate_cpi_schedule(next_year))
        new_events.append(generator.generate_nfp_schedule(next_year))
        new_events.append(generator.generate_quad_witching_schedule(next_year))

    # Merge all new events
    if not new_events:
        print("\n❌ No new events fetched from any source")
        print("   Check API keys and try again")
        return 1

    all_new_events = pd.concat(new_events, ignore_index=True)

    print_header("Merging Events")

    print(f"\n   Existing CSV events: {len(existing_df)}")
    print(f"   New events fetched: {len(all_new_events)}")

    # Merge
    merged_df = merge_events(existing_df, all_new_events)

    new_count = len(merged_df) - len(existing_df)
    print(f"   Merged total: {len(merged_df)}")
    print(f"   New events added: {new_count}")

    # Show summary of new events
    if new_count > 0:
        print(f"\n   New events by type:")
        new_only = merged_df[merged_df['date'] > existing_df['date'].max()]
        for event_type, count in new_only['event_type'].value_counts().items():
            print(f"     {event_type.ljust(15)}: {count}")

    # Save or preview
    if args.dry_run:
        print_header("DRY RUN - Preview (not saved)")
        print(f"\n   Would add {new_count} new events")
        print(f"\n   Preview of new events:")
        new_only = merged_df[merged_df['date'] > existing_df['date'].max()].head(20)
        print(new_only[['date', 'event_type', 'category']].to_string(index=False))

        if len(merged_df[merged_df['date'] > existing_df['date'].max()]) > 20:
            print(f"\n   ... and {len(merged_df[merged_df['date'] > existing_df['date'].max()]) - 20} more")

        print(f"\n   To save: Run without --dry-run flag")
        return 0
    else:
        # Save backup
        backup_file = Path(csv_file).parent / f"{Path(csv_file).stem}_backup.csv"
        existing_df.to_csv(backup_file, index=False)
        print(f"\n   💾 Backup saved: {backup_file.name}")

        # Save updated CSV
        merged_df.to_csv(csv_file, index=False)
        print(f"   ✅ Updated CSV saved: {Path(csv_file).name}")

        print_header("SUCCESS")
        print(f"\n   Event data updated successfully!")
        print(f"   Added {new_count} new events")
        print(f"   New coverage: {existing_df['date'].min().strftime('%Y-%m-%d')} to {merged_df['date'].max().strftime('%Y-%m-%d')}")
        print(f"\n   Next steps:")
        print(f"   1. Run: python validate_event_data.py")
        print(f"   2. Verify coverage is extended")
        print(f"   3. Re-run training or dashboard with updated events")
        print(f"\n{'='*70}\n")

        return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Update interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Update failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
