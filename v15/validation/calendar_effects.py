"""
v15/validation/calendar_effects.py — Unorthodox Calendar & Astronomical Effects

Tests whether any unusual temporal patterns predict TSLA break success or
daily return direction. These are drawn from academic literature and market lore:

  1. OPEX week effect (documented for high-options-volume stocks)
  2. Moon phase effect (Dichev & Janes 2001, ~1.4% annual effect in stocks)
  3. Presidential cycle (year 1-4 of term)
  4. Seasonal / month-of-year
  5. Day of week (already known: Thursday best for c9)
  6. VIX "eye of the storm" — pre-break compression → violent release
  7. Post-holiday drift (day after major US holidays)

Usage: python3 -m v15.validation.calendar_effects
"""
from __future__ import annotations
import calendar as cal_mod
import warnings
from typing import List
import ephem
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ── OPEX helpers ─────────────────────────────────────────────────────────────
def get_opex_dates(start=2015, end=2026):
    dates = []
    for year in range(start, end+1):
        for month in range(1,13):
            c = cal_mod.monthcalendar(year, month)
            fri = [w[4] for w in c if w[4] != 0]
            dates.append(pd.Timestamp(year, month, fri[2]))
    return dates

def opex_tag(dt, opex_dates):
    d = pd.Timestamp(dt).normalize()
    to_next  = min(abs((d - o).days) for o in opex_dates if o >= d) if any(o>=d for o in opex_dates) else 30
    from_last= min(abs((d - o).days) for o in opex_dates if o <= d) if any(o<=d for o in opex_dates) else 30
    return to_next, from_last

# ── Moon phase ────────────────────────────────────────────────────────────────
def moon_phase_pct(dt) -> float:
    """0 = new moon, 100 = full moon (ephem.Moon.phase)."""
    try:
        m = ephem.Moon(str(pd.Timestamp(dt).date()))
        return float(m.phase)
    except Exception:
        return 50.0

def lunar_cycle_day(dt) -> int:
    """Approximate day in lunar cycle (0-29), 0=new moon, 14-15=full moon."""
    phase = moon_phase_pct(dt) / 100.0
    # ephem.phase is illumination fraction; convert to cycle day
    # 0=new(0%), 7=first quarter(50%), 14-15=full(100%), 22=last quarter(50%)
    import math
    return int(round(math.acos(max(-1, min(1, 1 - 2*phase))) / math.pi * 14.75))

# ── US holidays ───────────────────────────────────────────────────────────────
US_HOLIDAYS_APPROX = set()
for yr in range(2015, 2026):
    US_HOLIDAYS_APPROX.update([
        pd.Timestamp(yr, 1, 1),   # New Year
        pd.Timestamp(yr, 7, 4),   # Independence Day
        pd.Timestamp(yr, 12, 25), # Christmas
        pd.Timestamp(yr, 11, 11), # Veterans Day
    ])

def post_holiday_flag(dt) -> int:
    """1 if yesterday or 2 days ago was a major US holiday."""
    d = pd.Timestamp(dt).normalize()
    for lag in [1, 2]:
        prev = d - pd.Timedelta(days=lag)
        if prev in US_HOLIDAYS_APPROX:
            return 1
    return 0

# ── Presidential cycle ────────────────────────────────────────────────────────
# US election years: 2016, 2020, 2024; inauguration Jan 20
# Year 1 of term = 2017, 2021, 2025 (post-inauguration to next election -3yr)
def presidential_year(dt) -> int:
    """1=post-election year, 2=mid-1, 3=mid-2, 4=election year."""
    yr = pd.Timestamp(dt).year
    # Election years: 2016, 2020, 2024, ...
    election_yrs = [2016, 2020, 2024, 2028]
    for ey in sorted(election_yrs, reverse=True):
        if yr >= ey:
            return ((yr - ey) % 4) + 1 if yr >= ey else 4
    return 4

# ── Load daily data ───────────────────────────────────────────────────────────
def load_daily(tsla_path='data/TSLAMin.txt', spy_path='data/SPYMin.txt'):
    print("Loading data...")
    def read_min(path):
        df = pd.read_csv(path, header=None, sep=';',
                         names=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S', errors='coerce')
        df = df.dropna(subset=['ts']).set_index('ts')
        return df
    tsla_m = read_min(tsla_path)
    spy_m  = read_min(spy_path)
    tsla_d = tsla_m.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    spy_d  = spy_m.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    trading = tsla_d.index[tsla_d['volume'] > 0]
    tsla_d  = tsla_d.loc[trading]
    spy_d   = spy_d.reindex(trading).ffill()
    print(f"  {len(tsla_d)} trading days: {tsla_d.index[0].date()} → {tsla_d.index[-1].date()}")
    return tsla_d, spy_d

# ── Build feature/return table ─────────────────────────────────────────────────
def build_table(tsla: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    print("Building calendar feature table...")
    opex_dates = get_opex_dates()
    rows = []
    closes = tsla['close'].values

    for i in range(5, len(tsla) - 1):
        dt = tsla.index[i]
        yr = dt.year
        if yr < 2015 or yr > 2024:
            continue

        # Next-day return
        ret_1d = float(closes[i+1] / closes[i] - 1)

        # 5-day return
        if i + 5 < len(tsla):
            ret_5d = float(closes[i+5] / closes[i] - 1)
        else:
            ret_5d = np.nan

        # OPEX
        to_opex, from_opex = opex_tag(dt, opex_dates)

        # Moon phase
        phase_pct = moon_phase_pct(dt)   # 0=new, 100=full
        moon_day  = lunar_cycle_day(dt)

        rows.append({
            'date':        dt,
            'year':        yr,
            'ret_1d':      ret_1d,
            'ret_5d':      ret_5d,
            'dow':         dt.weekday(),       # 0=Mon
            'month':       dt.month,
            'week_of_yr':  dt.isocalendar()[1],
            'opex_to':     to_opex,
            'opex_from':   from_opex,
            'is_opex_day': 1 if to_opex == 0 else 0,
            'is_opex_week': 1 if to_opex <= 5 or from_opex <= 5 else 0,
            'is_post_opex': 1 if 1 <= from_opex <= 3 else 0,
            'moon_phase':  phase_pct,
            'is_full_moon': 1 if phase_pct >= 85 else 0,
            'is_new_moon':  1 if phase_pct <= 15 else 0,
            'lunar_quarter': int(phase_pct // 25),  # 0-3
            'pres_year':   presidential_year(dt),
            'post_holiday': post_holiday_flag(dt),
            'spy_5d_ret':  float(spy['close'].iloc[i] / spy['close'].iloc[i-5] - 1),
        })

    return pd.DataFrame(rows).set_index('date')

# ── Analysis ───────────────────────────────────────────────────────────────────
def analyze(df: pd.DataFrame):
    print("\n" + "="*65)
    print("CALENDAR & ASTRONOMICAL EFFECTS ON TSLA DAILY RETURNS")
    print(f"  {len(df)} trading days, 2015-2024")
    print("="*65)

    base_ret = df['ret_1d'].mean() * 100
    base_5d  = df['ret_5d'].mean() * 100
    print(f"\nBaseline: avg 1d={base_ret:.3f}%  5d={base_5d:.3f}%")

    def bucket_analysis(col, df, label, ret_col='ret_1d', n_min=20):
        print(f"\n--- {label} ---")
        groups = df.groupby(col)[ret_col]
        results = []
        for name, grp in groups:
            if len(grp) < n_min:
                continue
            m = grp.mean() * 100
            s = grp.std() * 100
            t_stat, p_val = stats.ttest_1samp(grp.dropna(), 0)
            results.append((name, len(grp), m, s, p_val))
        results.sort(key=lambda x: x[2], reverse=True)
        for name, n, m, s, p in results:
            sig = '**' if p < 0.05 else ' *' if p < 0.10 else '  '
            bar = '█' * int(abs(m) * 20) if abs(m) < 5 else '██████████'
            sign = '+' if m >= 0 else ''
            print(f"  {str(name):<12s} n={n:4d}  avg={sign}{m:+.3f}%  p={p:.3f}{sig}  {bar}")
        return results

    # 1. Day of week
    dow_names = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri'}
    df2 = df.copy(); df2['dow_name'] = df2['dow'].map(dow_names)
    bucket_analysis('dow_name', df2, 'Day of Week Effect (1d return)')

    # 2. Month of year
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    df2['month_name'] = df2['month'].map(month_names)
    bucket_analysis('month_name', df2, 'Month of Year Effect (5d return)', ret_col='ret_5d')

    # 3. OPEX week
    bucket_analysis('is_opex_week', df2, 'OPEX Week Effect (5d return)', ret_col='ret_5d', n_min=50)
    bucket_analysis('is_post_opex', df2, 'Post-OPEX Effect (5d return)', ret_col='ret_5d', n_min=30)

    # 4. Moon phase (by quarter)
    phase_names = {0:'New Moon', 1:'Waxing', 2:'Full Moon', 3:'Waning'}
    df2['lunar_name'] = df2['lunar_quarter'].map(phase_names)
    bucket_analysis('lunar_name', df2, 'Lunar Phase Effect (5d return)', ret_col='ret_5d', n_min=50)
    bucket_analysis('is_full_moon', df2, 'Full Moon Day (1d return)', n_min=20)
    bucket_analysis('is_new_moon',  df2, 'New Moon Day (1d return)', n_min=20)

    # 5. Presidential cycle
    pres_names = {1:'Year1(post-elec)',2:'Year2(mid-1)',3:'Year3(mid-2)',4:'Year4(elec)'}
    df2['pres_name'] = df2['pres_year'].map(pres_names)
    bucket_analysis('pres_name', df2, 'Presidential Cycle (5d return)', ret_col='ret_5d', n_min=50)

    # 6. Post-holiday
    bucket_analysis('post_holiday', df2, 'Post-Holiday Effect (1d return)', n_min=20)

    # 7. VIX-style OPEX distance continuous analysis
    print("\n--- OPEX Distance (days to OPEX) bucketed ---")
    df2['opex_bucket'] = pd.cut(df2['opex_to'], bins=[0,3,7,14,21,35],
                                 labels=['0-3d','4-7d','8-14d','15-21d','22-35d'])
    bucket_analysis('opex_bucket', df2, 'Days to OPEX (5d return)',
                    ret_col='ret_5d', n_min=30)

    # 8. Correlation matrix of calendar features with returns
    print("\n--- Pearson correlation with 5d return ---")
    num_cols = ['opex_to','opex_from','moon_phase','dow','month','pres_year','spy_5d_ret']
    corr = df[['ret_5d'] + num_cols].corr()['ret_5d'].drop('ret_5d')
    corr_sorted = corr.abs().sort_values(ascending=False)
    for col in corr_sorted.index:
        r = corr[col]
        n = len(df)
        t = r * np.sqrt(n - 2) / np.sqrt(max(1 - r**2, 1e-10))
        p = float(2 * stats.t.sf(abs(t), df=n-2))
        sig = '**' if p < 0.05 else ' *' if p < 0.10 else '  '
        print(f"  {col:<20s} r={r:+.3f}  p={p:.3f}{sig}")

    print("\n" + "="*65)
    print("KEY FINDINGS:")
    print("  (Look for p<0.05 effects that could improve S32 entry timing)")
    print("="*65)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsla', default='data/TSLAMin.txt')
    parser.add_argument('--spy',  default='data/SPYMin.txt')
    args = parser.parse_args()
    tsla, spy = load_daily(args.tsla, args.spy)
    df = build_table(tsla, spy)
    analyze(df)
