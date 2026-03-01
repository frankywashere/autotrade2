#!/usr/bin/env python3
"""Check what WR/PnL look like after removing wins < $50."""
import pandas as pd
import numpy as np
from pathlib import Path

d = Path(__file__).parent / 'combo_results'

combos = [
    'B:_CS-ALL', 'C:_CS_V5', 'D:_CS_Filters', 'E:_CS_V5_Filters',
    'H:_CS_VIX', 'O:_CS_TF4', 'U:_TF4_Filt', 'W:_TF5',
    'X:_TF4_VIX', 'Y:_TF4_VIX_V5', 'Z:_ShortsTF4',
]

print(f"{'Combo':<18} | {'ALL':>4} {'WR':>6} {'PnL':>10} | {'>=50':>4} {'WR':>6} {'PnL':>10} | {'Tiny':>4} {'%':>5}")
print('-' * 82)

for c in combos:
    f = d / f'combo_trades_{c}.csv'
    if not f.exists():
        continue
    df = pd.read_csv(f)
    n = len(df)
    wins = (df.pnl > 0).sum()
    wr = wins / n * 100
    pnl = df.pnl.sum()

    # Remove wins under $50
    tiny_wins = (df.pnl > 0) & (df.pnl < 50)
    big = df[~tiny_wins]
    nb = len(big)
    bw = (big.pnl > 0).sum()
    bwr = bw / nb * 100 if nb > 0 else 0
    bpnl = big.pnl.sum()
    removed = tiny_wins.sum()

    name = c.replace('_', ' ').replace(':', ':')
    print(f"{name:<18} | {n:>4} {wr:>5.1f}% ${pnl:>+9,.0f} | "
          f"{nb:>4} {bwr:>5.1f}% ${bpnl:>+9,.0f} | "
          f"{removed:>4} {removed/n*100:>4.0f}%")

# Also show distribution of win sizes for top combos
print("\n\nWin size distribution (X: TF4+VIX):")
f = d / 'combo_trades_X:_TF4_VIX.csv'
if f.exists():
    df = pd.read_csv(f)
    wins = df[df.pnl > 0].pnl
    for bucket in [10, 25, 50, 100, 250, 500, 1000, 5000]:
        below = (wins < bucket).sum()
        print(f"  Wins < ${bucket:>5}: {below:>4} ({below/len(wins)*100:>5.1f}%)")
    print(f"  Total wins: {len(wins)}")
    print(f"  Median win: ${wins.median():,.0f}")
    print(f"  Mean win:   ${wins.mean():,.0f}")

print("\nWin size distribution (Y: TF4+VIX+V5):")
f = d / 'combo_trades_Y:_TF4_VIX_V5.csv'
if f.exists():
    df = pd.read_csv(f)
    wins = df[df.pnl > 0].pnl
    for bucket in [10, 25, 50, 100, 250, 500, 1000, 5000]:
        below = (wins < bucket).sum()
        print(f"  Wins < ${bucket:>5}: {below:>4} ({below/len(wins)*100:>5.1f}%)")
    print(f"  Total wins: {len(wins)}")
    print(f"  Median win: ${wins.median():,.0f}")
    print(f"  Mean win:   ${wins.mean():,.0f}")
