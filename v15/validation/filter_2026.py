#!/usr/bin/env python3
"""Filter combo trades to 2026 only."""
import pandas as pd
from pathlib import Path

d = Path(__file__).parent / 'combo_results'

combos = [
    'B:_CS-ALL', 'D:_CS_Filters',
    'O:_CS_TF4', 'U:_TF4_Filt',
    'W:_TF5', 'X:_TF4_VIX', 'Y:_TF4_VIX_V5', 'Z:_ShortsTF4',
    'AA:_Persist24', 'AB:_TF4VIXTight', 'AC:_Persist_VIX',
    'AD:_s1_tf3', 'AE:_s1tf3_VIX', 'AF:_TF4VIXHealth',
    'AJ:_TF4VIX_SPY', 'AK:_Y_SPY',
    'AN:_s1tf3VIX_SPY', 'AV:_TF4VIX_SPY1%',
    'AG:_X_cd=0', 'AH:_Y_cd=0', 'AI:_AE_cd=0_sex',
    'AL:_AJ_cd=0', 'AM:_AK_cd=0',
    'AO:_AN_cd=0', 'AP:_AN_cd=0_sex', 'AW:_AV_cd=0',
]

print("=" * 80)
print("  2026 ONLY (Jan 1 - Feb 27)")
print("=" * 80)
print(f"{'Combo':<18} {'Trades':>6} {'Wins':>5} {'Loss':>5} {'WR':>6} {'PnL':>10} {'AvgWin':>8} {'BigLoss':>8}")
print("-" * 80)

for c in combos:
    f = d / f'combo_trades_{c}.csv'
    if not f.exists():
        continue
    df = pd.read_csv(f)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df26 = df[df['entry_date'] >= '2026-01-01']

    n = len(df26)
    name = c.replace('_', ' ').replace(':', ':')
    if n == 0:
        print(f"{name:<18} {'0':>6}   ---   ---    ---        ---      ---      ---")
        continue

    wins = (df26.pnl > 0).sum()
    losses = n - wins
    wr = wins / n * 100
    pnl = df26.pnl.sum()
    avg_w = df26[df26.pnl > 0].pnl.mean() if wins > 0 else 0
    big_l = df26.pnl.min()

    print(f"{name:<18} {n:>6} {wins:>5} {losses:>5} {wr:>5.1f}% ${pnl:>+9,.0f} ${avg_w:>+7,.0f} ${big_l:>+7,.0f}")

    # Show individual trades
    if n <= 20:
        for _, t in df26.iterrows():
            w = 'W' if t.pnl > 0 else 'L'
            print(f"  {w} {t.entry_date.date()} {t.direction:>5} "
                  f"${t.entry_price:>7.1f} -> ${t.exit_price:>7.1f} "
                  f"PnL=${t.pnl:>+8,.0f} conf={t.confidence:.2f} [{t.source}]")
