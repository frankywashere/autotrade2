#!/usr/bin/env python3
"""Quick dataset summary to verify monthly ATR is feasible."""

import pandas as pd
import sys

print("="*80)
print("DATASET SUMMARY FOR MONTHLY ATR")
print("="*80)

print("\nLoading TSLA data...")
df = pd.read_csv("data/TSLA_1min.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

raw_bars = len(df)
start = df['timestamp'].min()
end = df['timestamp'].max()
days = (end - start).days
months = (days / 365.25) * 12

print(f"\nRaw Dataset:")
print(f"  Bars: {raw_bars:,}")
print(f"  Period: {start} to {end}")
print(f"  Days: {days:,}")
print(f"  Months: {months:.1f}")

print(f"\nMonthly ATR Requirements:")
print(f"  Minimum months needed: 14 (for 14-period ATR)")
print(f"  Available months: {months:.1f}")

if months >= 14:
    print(f"  Status: PASS - Sufficient data")
    result = "PASS"
else:
    print(f"  Status: FAIL - Insufficient data")
    result = "FAIL"

print("\n" + "="*80)
print(f"RESULT: {result}")
print("="*80)

if result == "PASS":
    print("\nConclusion: Monthly ATR will work with this dataset.")
    print(f"The {raw_bars:,} bars spanning {months:.1f} months provides")
    print("sufficient data for monthly ATR calculation.")
    sys.exit(0)
else:
    sys.exit(1)
