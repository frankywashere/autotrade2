"""
v15/validation/options_snapshot.py — Live TSLA Options Chain Snapshot

Computes real-time options-derived signals:
  1. Put/Call OI ratio (ATM ±5%)
  2. IV skew (put IV - call IV at ATM)
  3. Max-pain strike (net OI pain minimization)
  4. Gamma exposure proxy (call OI - put OI at nearby strikes)
  5. IV term structure (front month vs back month implied vol)
  6. Composite options score (-2 to +2: bearish to bullish)

Usage: python3 -m v15.validation.options_snapshot [--ticker TSLA] [--json]
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("yfinance not installed: pip install yfinance")


def compute_max_pain(calls_df, puts_df):
    """
    Max pain = strike where market makers' aggregate P&L is maximized
    = strike where sum of OTM option value destroyed is minimized for option buyers.
    Returns max_pain_strike.
    """
    all_strikes = sorted(set(list(calls_df['strike']) + list(puts_df['strike'])))
    min_pain = float('inf')
    max_pain_k = all_strikes[0]
    call_oi = dict(zip(calls_df['strike'], calls_df['openInterest'].fillna(0)))
    put_oi  = dict(zip(puts_df['strike'],  puts_df['openInterest'].fillna(0)))
    for k in all_strikes:
        # Call pain: for each call strike S < k, OI × (k - S) [calls expire worthless]
        c_pain = sum(oi * (k - s) for s, oi in call_oi.items() if s < k and oi > 0)
        # Put pain: for each put strike S > k, OI × (S - k)
        p_pain = sum(oi * (s - k) for s, oi in put_oi.items() if s > k and oi > 0)
        total = c_pain + p_pain
        if total < min_pain:
            min_pain = total
            max_pain_k = k
    return max_pain_k


def get_options_snapshot(ticker: str = 'TSLA', near_pct: float = 0.05) -> dict:
    """
    Fetch and compute options-derived signals for ticker.

    Returns dict with:
      price, exp, days_to_exp, pc_ratio, call_iv, put_iv, iv_skew,
      max_pain, max_pain_pct, gex_net, gex_direction, score, signal_text
    """
    t = yf.Ticker(ticker)
    price = float(t.fast_info.last_price)

    # Get nearest weekly/monthly expiry
    exps = t.options
    if not exps:
        return {'error': f'No options found for {ticker}'}

    # Use nearest expiry that's at least 1 day out
    import pandas as pd
    today = pd.Timestamp.now().normalize()
    exp = None
    for e in exps:
        exp_dt = pd.Timestamp(e)
        if (exp_dt - today).days >= 1:
            exp = e
            break
    if exp is None:
        exp = exps[0]

    exp_dt = pd.Timestamp(exp)
    days_to_exp = (exp_dt - today).days

    chain = t.option_chain(exp)
    calls = chain.calls.copy()
    puts  = chain.puts.copy()
    calls['openInterest'] = calls['openInterest'].fillna(0)
    puts['openInterest']  = puts['openInterest'].fillna(0)

    # ATM filter: strikes within near_pct of current price
    atm_calls = calls[abs(calls['strike'] - price) / price <= near_pct]
    atm_puts  = puts[abs(puts['strike']  - price) / price <= near_pct]

    # Put/Call OI ratio
    total_call_oi = max(atm_calls['openInterest'].sum(), 1)
    total_put_oi  = atm_puts['openInterest'].sum()
    pc_ratio = total_put_oi / total_call_oi

    # IV
    call_iv = float(atm_calls['impliedVolatility'].mean()) if len(atm_calls) > 0 else 0.0
    put_iv  = float(atm_puts['impliedVolatility'].mean())  if len(atm_puts) > 0 else 0.0
    iv_skew = put_iv - call_iv  # positive = downside fear

    # Max pain
    max_pain = compute_max_pain(calls, puts)
    max_pain_pct = (max_pain - price) / price * 100  # negative = max pain below current

    # Gamma exposure proxy: nearby strikes ±$10 per share
    band = max(price * 0.025, 5.0)  # 2.5% or $5 minimum
    near_calls = calls[abs(calls['strike'] - price) <= band]
    near_puts  = puts[abs(puts['strike']  - price) <= band]
    gex_net = float(near_calls['openInterest'].sum() - near_puts['openInterest'].sum())
    gex_direction = 'CALL_DOM' if gex_net > 0 else 'PUT_DOM'

    # ── Composite signal score (-2 bearish → +2 bullish) ──
    score = 0.0

    # PC ratio: < 0.7 = calls dominating = bullish; > 1.3 = puts dominating = bearish
    if pc_ratio < 0.7:
        score += 1.0
    elif pc_ratio > 1.3:
        score -= 1.0

    # IV skew: positive skew (put IV > call IV) = bearish; negative = bullish (rare)
    if iv_skew < 0.05:
        score += 0.5
    elif iv_skew > 0.15:
        score -= 0.5

    # Max pain: large negative pct = strong pull DOWN = bearish
    if max_pain_pct < -5:
        score -= 0.5
    elif max_pain_pct > 5:
        score += 0.5

    # GEX: call-dominant = ceiling pressure = slight bearish (MM hedge by selling)
    if gex_net < 0:  # put dominant = floor support = bullish
        score += 0.5
    elif gex_net > 0:  # call dominant = ceiling pressure = bearish
        score -= 0.5

    score = max(-2.0, min(2.0, score))
    if score >= 1.0:
        signal_text = 'BULLISH'
    elif score <= -1.0:
        signal_text = 'BEARISH'
    else:
        signal_text = 'NEUTRAL'

    return {
        'ticker':       ticker,
        'price':        round(price, 2),
        'exp':          exp,
        'days_to_exp':  days_to_exp,
        'pc_ratio':     round(pc_ratio, 3),
        'call_iv':      round(call_iv, 4),
        'put_iv':       round(put_iv, 4),
        'iv_skew':      round(iv_skew, 4),
        'max_pain':     round(max_pain, 2),
        'max_pain_pct': round(max_pain_pct, 2),
        'gex_net':      round(gex_net, 0),
        'gex_direction': gex_direction,
        'score':        round(score, 2),
        'signal':       signal_text,
    }


def print_snapshot(d: dict):
    if 'error' in d:
        print(f"ERROR: {d['error']}")
        return
    print(f"\n{'='*60}")
    print(f"OPTIONS SNAPSHOT — {d['ticker']}  ${d['price']:.2f}")
    print(f"  Expiry: {d['exp']} ({d['days_to_exp']}d)")
    print(f"{'='*60}")
    print(f"  Put/Call OI ratio:   {d['pc_ratio']:.3f}  {'(bullish<0.7)' if d['pc_ratio'] < 0.7 else '(bearish>1.3)' if d['pc_ratio'] > 1.3 else '(neutral)'}")
    print(f"  Call IV:             {d['call_iv']*100:.1f}%")
    print(f"  Put IV:              {d['put_iv']*100:.1f}%")
    print(f"  IV Skew (Put-Call):  {d['iv_skew']*100:.2f}%  {'(high fear)' if d['iv_skew'] > 0.15 else ''}")
    print(f"  Max Pain:            ${d['max_pain']:.2f}  ({d['max_pain_pct']:+.1f}% from current)")
    print(f"  GEX proxy:           {d['gex_net']:+,.0f}  [{d['gex_direction']}]")
    print(f"{'='*60}")
    print(f"  COMPOSITE SCORE:     {d['score']:+.2f}  →  {d['signal']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live options chain snapshot')
    parser.add_argument('--ticker', default='TSLA')
    parser.add_argument('--json',   action='store_true', help='Output as JSON')
    args = parser.parse_args()

    snap = get_options_snapshot(args.ticker)
    if args.json:
        print(json.dumps(snap, indent=2))
    else:
        print_snapshot(snap)
