#!/usr/bin/env python3
"""
Update Live Predictions with Actuals

Cron job that runs hourly to update live predictions with actual outcomes
after 24 hours have elapsed.

Usage:
    python3 update_live_actuals.py

Cron setup:
    0 * * * * cd /path/to/autotrade2 && /path/to/myenv/bin/python3 update_live_actuals.py >> logs/actuals_update.log 2>&1
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.database import SQLitePredictionDB
from src.ml.live_data_loader import LiveDataLoader


def main():
    print("="*70)
    print(f"UPDATING LIVE PREDICTIONS WITH ACTUALS - {datetime.now()}")
    print("="*70)

    db = SQLitePredictionDB()

    # Find predictions >24 hours old without actuals
    # Only update live predictions (simulation_date IS NULL)
    query = """
        SELECT
            id,
            target_timestamp,
            symbol,
            current_price
        FROM predictions
        WHERE target_timestamp <= datetime('now')
          AND has_actuals = 0
          AND simulation_date IS NULL
        ORDER BY target_timestamp DESC
        LIMIT 100
    """

    predictions_to_update = pd.read_sql(query, db.session.bind, parse_dates=['target_timestamp'])

    if len(predictions_to_update) == 0:
        print("✓ No predictions need updating")
        print("="*70)
        return

    print(f"Found {len(predictions_to_update)} predictions to update")

    # Load live data (last 7 days should cover any pending predictions)
    loader = LiveDataLoader(timeframe='1min')

    try:
        df, status = loader.load_live_data(lookback_days=7)
        print(f"✓ Loaded {len(df)} bars of live data (status: {status})")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    # Update each prediction
    updated = 0
    failed = 0

    for idx, row in predictions_to_update.iterrows():
        pred_id = row['id']
        target_timestamp = row['target_timestamp']
        current_price = row['current_price']

        try:
            # Find actual high/low in the 24-hour window
            # Window: prediction_timestamp to target_timestamp
            window_end = target_timestamp

            # Get 24 hours before target
            window_start = target_timestamp - timedelta(hours=24)

            # Filter data to this window
            window_df = df[(df.index >= window_start) & (df.index <= window_end)]

            if len(window_df) == 0:
                print(f"  ⚠ Pred {pred_id}: No data in window")
                failed += 1
                continue

            # Get actuals
            actual_high = window_df['tsla_close'].max()
            actual_low = window_df['tsla_close'].min()

            # Update database
            db.update_actual(pred_id, float(actual_high), float(actual_low))

            updated += 1

            if updated % 10 == 0:
                print(f"  ✓ Updated {updated} predictions...")

        except Exception as e:
            print(f"  ✗ Pred {pred_id}: {e}")
            failed += 1

    print(f"\n✓ Updated {updated} predictions")
    if failed > 0:
        print(f"⚠ Failed to update {failed} predictions")

    print("="*70)


if __name__ == '__main__':
    main()
