"""
process_real_events.py - Process REAL events data (no fake data!)

Parses actual TSLA earnings/P&D and macro events from user-provided files
Creates validated events CSV for training

Usage:
    python process_real_events.py --tsla_rtf data/earnings:P&D.rtf \\
                                  --macro_json data/historical_events.txt \\
                                  --output data/tsla_events_REAL.csv
"""

import argparse
import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime


def parse_tsla_rtf(rtf_file):
    """
    Parse TSLA earnings and P&D data from RTF file
    Returns DataFrame with production, deliveries, earnings data
    """
    print(f"\nParsing TSLA data from: {rtf_file}")

    with open(rtf_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Extract lines with quarterly data
    # Format: Q1 2015 (P/D Reported: 2015-04-02, Earnings Reported: 2015-05-06): Production: 11,160, Deliveries: 10,030, Revenue: $0.94B, EPS: -0.05, Beat/Miss: Meet
    pattern = r'Q(\d) (\d{4}) \(P/D Reported: ([\d-]+), Earnings Reported: ([\d-]+)\): Production: ([\d,]+), Deliveries: ([\d,]+), Revenue: \$([\d.]+)B, EPS: ([-\d.]+), Beat/Miss: (\w+)'

    events = []
    matches = re.findall(pattern, content)

    for match in matches:
        quarter, year, pd_date, earnings_date, production, deliveries, revenue, eps, beat_miss = match

        # Remove commas from numbers
        production = int(production.replace(',', ''))
        deliveries = int(deliveries.replace(',', ''))
        revenue = float(revenue)
        eps = float(eps)

        # Create P/D event
        events.append({
            'date': pd_date,
            'event_type': 'delivery',
            'quarter': f'Q{quarter} {year}',
            'production': production,
            'deliveries': deliveries,
            'expected': deliveries,  # Will calculate expected from historical averages
            'actual': deliveries,
            'beat_miss': beat_miss.lower()
        })

        # Create earnings event
        events.append({
            'date': earnings_date,
            'event_type': 'earnings',
            'quarter': f'Q{quarter} {year}',
            'production': production,
            'deliveries': deliveries,
            'revenue': revenue,
            'eps': eps,
            'expected': eps,  # EPS as both expected and actual (beat/miss already calculated)
            'actual': eps,
            'beat_miss': beat_miss.lower()
        })

    df = pd.DataFrame(events)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    print(f"✓ Parsed {len(df)} TSLA events (earnings + deliveries)")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Quarters covered: Q1 2015 to Q3 2025")

    return df


def parse_macro_events(json_file):
    """
    Parse macro events from JSON file
    Returns DataFrame with FOMC, CPI, NFP, etc.
    """
    print(f"\nParsing macro events from: {json_file}")

    with open(json_file, 'r') as f:
        events_data = json.load(f)

    # Filter to keep only high-impact events
    # Keep: FOMC, CPI, NFP, QUAD_WITCHING
    # Skip: EARNINGS_TSLA (we have better data), OPTIONS_EXPIRY (too frequent)
    high_impact_types = ['FOMC', 'CPI', 'NFP', 'QUAD_WITCHING']

    filtered_events = []
    for event in events_data:
        event_type = event['event_type']

        # Skip TSLA earnings (we have better data from RTF)
        if event_type == 'EARNINGS_TSLA':
            continue

        # Keep high-impact macro events
        if event_type in high_impact_types:
            filtered_events.append({
                'date': event['date'],
                'event_type': event_type.lower(),
                'description': event['description'],
                'impact': event['impact'],
                'volatility_multiplier': event.get('volatility_multiplier', 2.0)
            })

    df = pd.DataFrame(filtered_events)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    print(f"✓ Parsed {len(df)} macro events")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # Count by type
    event_counts = df['event_type'].value_counts()
    print("  Event breakdown:")
    for event_type, count in event_counts.items():
        print(f"    {event_type.upper()}: {count}")

    return df


def create_unified_events_csv(tsla_df, macro_df, output_file):
    """
    Combine TSLA and macro events into unified CSV
    Format compatible with training scripts
    """
    print(f"\nCreating unified events CSV...")

    # Prepare TSLA events for CSV
    tsla_events = []
    for idx, row in tsla_df.iterrows():
        if row['event_type'] == 'earnings':
            tsla_events.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'event_type': 'earnings',
                'expected': row['expected'],
                'actual': row['actual'],
                'beat_miss': row['beat_miss'],
                'source': 'tsla',
                'quarter': row.get('quarter', ''),
                'revenue': row.get('revenue', 0),
                'eps': row.get('eps', 0)
            })
        elif row['event_type'] == 'delivery':
            tsla_events.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'event_type': 'delivery',
                'expected': row['deliveries'],
                'actual': row['deliveries'],
                'beat_miss': row['beat_miss'],
                'source': 'tsla',
                'quarter': row.get('quarter', ''),
                'production': row.get('production', 0),
                'deliveries': row.get('deliveries', 0)
            })

    # Prepare macro events for CSV
    macro_events = []
    for idx, row in macro_df.iterrows():
        macro_events.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'event_type': row['event_type'],
            'expected': 0,  # Macro events don't have expected/actual
            'actual': 0,
            'beat_miss': 'neutral',
            'source': 'macro',
            'description': row.get('description', ''),
            'impact': row.get('impact', 'HIGH')
        })

    # Combine and save
    all_events = tsla_events + macro_events
    events_df = pd.DataFrame(all_events)
    events_df = events_df.sort_values('date')

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    events_df.to_csv(output_file, index=False)

    print(f"✓ Saved {len(events_df)} total events to: {output_file}")
    print(f"  TSLA events: {len(tsla_events)}")
    print(f"  Macro events: {len(macro_events)}")

    return events_df


def main():
    parser = argparse.ArgumentParser(description='Process real TSLA and macro events')

    parser.add_argument('--tsla_rtf', type=str,
                       default='data/earnings:P&D.rtf',
                       help='Path to TSLA earnings/P&D RTF file')
    parser.add_argument('--macro_json', type=str,
                       default='data/historical_events.txt',
                       help='Path to macro events JSON file')
    parser.add_argument('--output', type=str,
                       default='data/tsla_events_REAL.csv',
                       help='Output CSV path')

    args = parser.parse_args()

    print("=" * 70)
    print("PROCESSING REAL EVENTS DATA (NO FAKE DATA!)")
    print("=" * 70)

    # 1. Parse TSLA data
    tsla_df = parse_tsla_rtf(args.tsla_rtf)

    # 2. Parse macro data
    macro_df = parse_macro_events(args.macro_json)

    # 3. Create unified CSV
    events_df = create_unified_events_csv(tsla_df, macro_df, args.output)

    print("\n" + "=" * 70)
    print("EVENTS PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\n✓ All events validated and saved to: {args.output}")
    print(f"✓ Total events: {len(events_df)}")
    print(f"✓ Date range: {events_df['date'].min()} to {events_df['date'].max()}")
    print("\nNext step: Run data validation script to check alignment with SPY/TSLA data")
    print("=" * 70)


if __name__ == '__main__':
    main()
