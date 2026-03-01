#!/usr/bin/env python3
"""Download recent TSLA intraday data and append to TSLAMin.txt.
Uses yfinance for recent data (5-min, max 60 days).
Converts to TSLAMin.txt format: YYYYMMDD HHMMSS;open;high;low;close;volume
"""
import yfinance as yf
import pandas as pd
import os
from pathlib import Path

# Find TSLAMin.txt
for candidate in ['data/TSLAMin.txt', '../data/TSLAMin.txt', 'C:/AI/x14/data/TSLAMin.txt']:
    if os.path.isfile(candidate):
        tsla_path = candidate
        break
else:
    raise FileNotFoundError("TSLAMin.txt not found")

print(f"TSLAMin.txt: {tsla_path}")

# Read last date in TSLAMin.txt
with open(tsla_path, 'r') as f:
    lines = f.readlines()
last_line = lines[-1].strip()
last_dt_str = last_line.split(';')[0]
print(f"Last entry: {last_dt_str}")
last_date = pd.Timestamp(last_dt_str[:8])
print(f"Last date: {last_date}")

# Download recent 5-min data (max ~60 days)
print("\nDownloading recent TSLA 5-min data...")
ticker = yf.Ticker("TSLA")

# Try 1-minute first (max ~30 days), then fall back to 5-minute
for interval, max_days in [('1m', 7), ('5m', 60)]:
    try:
        df = ticker.history(period=f"{max_days}d", interval=interval, prepost=True)
        if len(df) > 0:
            print(f"  Downloaded {len(df)} bars at {interval} interval")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            break
    except Exception as e:
        print(f"  {interval} failed: {e}")
        continue
else:
    print("ERROR: Could not download any intraday data")
    exit(1)

# If 5-min, expand to 1-minute bars (duplicate each bar 5 times)
if interval == '5m':
    print("  Expanding 5-min to 1-min bars...")
    expanded = []
    for idx, row in df.iterrows():
        for i in range(5):
            new_idx = idx + pd.Timedelta(minutes=i)
            expanded.append({
                'datetime': new_idx,
                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Close': row['Close'],
                'Volume': int(row['Volume'] / 5)
            })
    df = pd.DataFrame(expanded).set_index('datetime')
    print(f"  Expanded to {len(df)} 1-min bars")

# Convert timezone to US/Eastern if needed
if df.index.tz is not None:
    df.index = df.index.tz_convert('US/Eastern')
else:
    df.index = df.index.tz_localize('US/Eastern')

# Filter: only keep data AFTER last entry in TSLAMin.txt
# Parse last datetime from TSLAMin.txt more precisely
last_full_dt = pd.Timestamp(f"{last_dt_str[:8]} {last_dt_str[9:]}")
last_full_dt = last_full_dt.tz_localize('US/Eastern')
print(f"Filtering after: {last_full_dt}")

new_data = df[df.index > last_full_dt]
print(f"New bars to append: {len(new_data)}")

if len(new_data) == 0:
    print("No new data to append!")
    exit(0)

# Get unique new dates
new_dates = sorted(set(new_data.index.date))
print(f"New trading dates: {len(new_dates)}")
for d in new_dates[:5]:
    print(f"  {d}")
if len(new_dates) > 5:
    print(f"  ... and {len(new_dates)-5} more")

# Convert to TSLAMin.txt format
print(f"\nWriting {len(new_data)} new bars to TSLAMin.txt...")
lines_to_append = []
for idx, row in new_data.iterrows():
    dt_str = idx.strftime('%Y%m%d %H%M%S')
    line = f"{dt_str};{row['Open']:.2f};{row['High']:.2f};{row['Low']:.2f};{row['Close']:.2f};{int(row['Volume'])}"
    lines_to_append.append(line)

# Append to file
with open(tsla_path, 'a') as f:
    for line in lines_to_append:
        f.write('\n' + line)

print(f"Appended {len(lines_to_append)} bars")

# Verify
with open(tsla_path, 'r') as f:
    total_lines = sum(1 for _ in f)
print(f"Total lines now: {total_lines}")

# Show last few lines
with open(tsla_path, 'r') as f:
    all_lines = f.readlines()
print("Last 3 lines:")
for l in all_lines[-3:]:
    print(f"  {l.strip()}")

print("\nDone! Now regenerate cache with: python -m v15.validation.combo_backtest --end 2026-12-31")
