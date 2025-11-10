#!/usr/bin/env python3
"""Convert TSLAMin.txt and SPYMin.txt to proper CSV format."""
import pandas as pd
from datetime import datetime

def convert_data_file(input_file, output_file):
    """Convert text data to CSV format."""
    print(f"Converting {input_file} to {output_file}...")

    data = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse: YYYYMMDD HHMMSS;open;high;low;close;volume
            parts = line.split(';')
            if len(parts) != 6:
                continue

            # Parse datetime
            date_time = parts[0].split(' ')
            if len(date_time) != 2:
                continue

            date_str = date_time[0]
            time_str = date_time[1]

            # Create timestamp
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6]) if len(time_str) > 4 else 0

            timestamp = datetime(year, month, day, hour, minute, second)

            # Parse OHLCV
            open_price = float(parts[1])
            high = float(parts[2])
            low = float(parts[3])
            close = float(parts[4])
            volume = int(parts[5])

            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.sort_values('timestamp', inplace=True)
    df.to_csv(output_file, index=False)

    print(f"  Converted {len(df)} rows")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Saved to: {output_file}")

    return df

if __name__ == "__main__":
    # Convert TSLA data
    tsla_df = convert_data_file('data/TSLAMin.txt', 'data/TSLA_1min.csv')
    print()

    # Convert SPY data
    spy_df = convert_data_file('data/SPYMin.txt', 'data/SPY_1min.csv')
    print()

    print("✅ Data conversion complete!")