#!/usr/bin/env python3
"""Analyze remaining losses in top combos."""
import pandas as pd
from pathlib import Path

d = Path(__file__).parent / 'combo_results'

combos = [
    'X:_TF4_VIX', 'AJ:_TF4VIX_SPY', 'AN:_s1tf3VIX_SPY',
    'AI:_AE_cd=0_sex', 'AH:_Y_cd=0', 'AL:_AJ_cd=0',
    'AO:_AN_cd=0', 'AP:_AN_cd=0_sex', 'AW:_AV_cd=0',
    'AV:_TF4VIX_SPY1%',
]

for c in combos:
    f = d / f'combo_trades_{c}.csv'
    try:
        df = pd.read_csv(f)
        losses = df[df['pnl'] < 0]
        if len(losses) > 0:
            print(f'=== {c} === {len(losses)} losses of {len(df)} trades')
            print(f'  Columns: {list(df.columns)}')
            for _, t in losses.iterrows():
                print(f'  date={t.entry_date} dir={t.direction} '
                      f'entry={t.entry_price:.2f} exit={t.exit_price:.2f} '
                      f'PnL=${t.pnl:+.0f} conf={t.confidence:.3f} '
                      f'hold={t.hold_days}d '
                      f'exit={t.exit_reason} src={t.source}')
        else:
            print(f'=== {c} === 0 losses of {len(df)} trades (100% WR)')
    except Exception as e:
        print(f'=== {c} === ERROR: {e}')
