#!/usr/bin/env python3
"""Fill gap in TSLAMin.txt with 5-min yfinance data expanded to 1-min.
Fills the gap between Sep 2025 (end of original data) and Feb 19, 2026 (start of 1-min download).
"""
import yfinance as yf
import pandas as pd
import os

tsla_path = None
for candidate in ['data/TSLAMin.txt', '../data/TSLAMin.txt', 'C:/AI/x14/data/TSLAMin.txt']:
    if os.path.isfile(candidate):
        tsla_path = candidate
        break
if not tsla_path:
    raise FileNotFoundError("TSLAMin.txt not found")

print(f"TSLAMin.txt: {tsla_path}")

# Read existing data
print("Reading existing TSLAMin.txt...")
with open(tsla_path, 'r') as f:
    lines = f.readlines()
print(f"Total lines: {len(lines)}")

# Find existing dates in 2026
existing_2026_dates = set()
for l in lines:
    s = l.strip()
    if s.startswith('2026'):
        existing_2026_dates.add(s[:8])
print(f"Existing 2026 dates: {len(existing_2026_dates)}")
for d in sorted(existing_2026_dates)[:5]:
    print(f"  {d}")

# Find last pre-2026 entry
last_pre_2026_dt = None
for l in reversed(lines):
    s = l.strip()
    if s and not s.startswith('2026'):
        last_pre_2026_dt = s.split(';')[0]
        break
print(f"Last pre-2026 entry: {last_pre_2026_dt}")

# Download 5-min data (60 days)
print("\nDownloading 5-min data (60 days)...")
ticker = yf.Ticker("TSLA")
df = ticker.history(period="60d", interval="5m", prepost=True)
print(f"Downloaded {len(df)} bars: {df.index[0]} to {df.index[-1]}")

# Convert timezone to Eastern
if df.index.tz is not None:
    df.index = df.index.tz_convert('US/Eastern')
else:
    df.index = df.index.tz_localize('US/Eastern')

# Expand 5-min to 1-min bars
print("Expanding 5-min to 1-min bars...")
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
df_1m = pd.DataFrame(expanded).set_index('datetime')
print(f"Expanded to {len(df_1m)} 1-min bars")

# Filter: only keep data AFTER the last pre-2026 entry
last_dt = pd.Timestamp(f"{last_pre_2026_dt[:8]} {last_pre_2026_dt[9:]}")
last_dt = last_dt.tz_localize('US/Eastern')
print(f"Filtering after: {last_dt}")

new_data = df_1m[df_1m.index > last_dt]

# Remove dates we already have (Feb 19-27 from previous 1-min download)
mask = new_data.index.map(lambda x: x.strftime('%Y%m%d') not in existing_2026_dates)
new_data = new_data[mask]

new_dates = sorted(set(new_data.index.date))
print(f"New bars to insert: {len(new_data)}")
print(f"New trading dates: {len(new_dates)}")
for d in new_dates[:5]:
    print(f"  {d}")
if len(new_dates) > 5:
    print(f"  ... and {len(new_dates) - 5} more")
for d in new_dates[-3:]:
    print(f"  {d}")

if len(new_data) == 0:
    print("No new data to add!")
    exit(0)

# Find insertion point (first 2026 line)
insert_idx = None
for i, l in enumerate(lines):
    if l.strip().startswith('2026'):
        insert_idx = i
        break

if insert_idx is None:
    # No existing 2026 data, append at end
    insert_idx = len(lines)

print(f"Insertion point: line {insert_idx}")

# Build new lines
new_lines = []
for idx, row in new_data.iterrows():
    dt_str = idx.strftime('%Y%m%d %H%M%S')
    line = f"{dt_str};{row['Open']:.2f};{row['High']:.2f};{row['Low']:.2f};{row['Close']:.2f};{int(row['Volume'])}"
    new_lines.append(line)

new_lines.sort()
print(f"Prepared {len(new_lines)} new lines")
print(f"First: {new_lines[0]}")
print(f"Last: {new_lines[-1]}")

# Rebuild file
print(f"\nRebuilding TSLAMin.txt...")
with open(tsla_path, 'w') as f:
    for l in lines[:insert_idx]:
        f.write(l)
    for nl in new_lines:
        f.write(nl + '\n')
    for l in lines[insert_idx:]:
        f.write(l)

# Verify
with open(tsla_path, 'r') as f:
    total = sum(1 for _ in f)
print(f"Total lines now: {total}")
print(f"Added {total - len(lines)} lines")

# Count 2026 dates
with open(tsla_path, 'r') as f:
    d26 = set()
    for l in f:
        s = l.strip()
        if s.startswith('2026'):
            d26.add(s[:8])
print(f"2026 trading dates now: {len(d26)}")

print("\nDone! Regenerate cache with: python -m v15.validation.combo_backtest --end 2026-12-31")
