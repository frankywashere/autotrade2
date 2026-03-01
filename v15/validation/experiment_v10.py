#!/usr/bin/env python3
"""
V10 Experiments: Codex-inspired advanced filters.

From Codex's recommendations:
  1. Lempel-Ziv Complexity — measure compressibility of return signs
  2. Cross-Asset Lead-Lag Residual — TSLA vs SPY regression residual
  3. Volatility Relaxation Half-Life — smooth vol decay after shocks
  4. Loss-Only Anomaly Detection — isolation forest on rare losses
  5. Channel Curvature / Geometric Tension — channel midline bending
  6. Combined SPY+LZ+VolRelax stack
"""

import pickle, sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming,
    _SigProxy, _AnalysisProxy, _build_filter_cascade, _floor_stop_tp,
    MIN_SIGNAL_CONFIDENCE, CAPITAL, DEFAULT_STOP_PCT, DEFAULT_TP_PCT,
    TRAILING_STOP_BASE, MAX_HOLD_DAYS,
    simulate_trades,
)


def _summary_line(trades, name=''):
    n = len(trades)
    if n == 0:
        return f"  {name:<55} {'0':>5}  {'---':>5}  {'---':>10}  {'---':>5}  {'---':>5}  {'---':>8}"
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    total = sum(t.pnl for t in trades)
    pnls = np.array([t.pnl for t in trades])
    sharpe = (pnls.mean() / pnls.std() * np.sqrt(252 / max(np.mean([t.hold_days for t in trades]), 1))
              ) if pnls.std() > 0 else 0
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    mdd = float(dd.max()) / CAPITAL * 100 if len(dd) > 0 else 0
    big_l = min(t.pnl for t in trades)
    train = [t for t in trades if t.entry_date.year <= 2021]
    test = [t for t in trades if t.entry_date.year > 2021]
    tr_wr = sum(1 for t in train if t.pnl > 0) / len(train) * 100 if train else 0
    ts_wr = sum(1 for t in test if t.pnl > 0) / len(test) * 100 if test else 0
    return (f"  {name:<55} {n:>5} {wr:>5.1f}% ${total:>+9,.0f}  Sh={sharpe:>5.1f}  "
            f"DD={mdd:>4.1f}%  Tr={tr_wr:.0f}% Ts={ts_wr:.0f}%  BL=${big_l:>+8,.0f}")


# ---------------------------------------------------------------------------
# Combo factories
# ---------------------------------------------------------------------------

def make_tf4_vix_combo(cascade_vix):
    def fn(day):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) < 4:
                return None
            sig = _SigProxy(day)
            ana = _AnalysisProxy(day.cs_tf_states)
            ok, adj, _ = cascade_vix.evaluate(
                sig, ana, feature_vec=None, bar_datetime=day.date,
                higher_tf_data=None, spy_df=None, vix_df=None,
            )
            if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                return None
            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return (day.cs_action, adj, s, t, 'CS')
        return None
    return fn


# ---------------------------------------------------------------------------
# EXPERIMENT 1: LEMPEL-ZIV COMPLEXITY
# ---------------------------------------------------------------------------

def lempel_ziv_complexity(s):
    """Compute normalized Lempel-Ziv complexity of a binary/symbol sequence.
    Returns value in [0, 1] where 0=fully compressible, 1=random."""
    n = len(s)
    if n <= 1:
        return 0.0
    c = 1  # complexity counter
    l = 1  # current prefix length
    k = 1  # current matching position
    i = 0  # start of current window
    while l + k <= n:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
        else:
            if k > l - i:
                i = l
            c += 1
            l += k
            k = 1
            i = 0
    # Normalize by theoretical max for sequence of length n
    b = len(set(s))
    if b <= 1:
        return 0.0
    max_c = n / np.log2(n) if n > 1 else 1
    return c / max_c


def rolling_lz_complexity(returns, window=30):
    """Rolling LZ complexity of return signs (+1/-1/0)."""
    signs = np.sign(returns)
    # Convert to symbols: 0=down, 1=up, 2=flat
    symbols = (signs + 1).astype(int)  # 0,1,2

    lz_vals = np.full(len(returns), np.nan)
    for i in range(window, len(returns)):
        seq = tuple(symbols[i-window:i])
        lz_vals[i] = lempel_ziv_complexity(seq)
    return lz_vals


def run_lz_experiment(signals, cascade_vix):
    """Test Lempel-Ziv complexity as a filter."""
    print("=" * 100)
    print("  EXPERIMENT 1: LEMPEL-ZIV COMPLEXITY FILTER")
    print("=" * 100)

    closes = np.array([s.day_close for s in signals], dtype=float)
    returns = np.diff(closes) / closes[:-1]
    returns = np.insert(returns, 0, 0.0)

    lz30 = rolling_lz_complexity(returns, window=30)
    lz60 = rolling_lz_complexity(returns, window=60)

    lz30_by_date = {signals[i].date: lz30[i] for i in range(len(signals)) if not np.isnan(lz30[i])}
    lz60_by_date = {signals[i].date: lz60[i] for i in range(len(signals)) if not np.isnan(lz60[i])}

    valid_lz30 = [v for v in lz30 if not np.isnan(v)]
    print(f"  LZ30 range: {min(valid_lz30):.4f} - {max(valid_lz30):.4f}, mean={np.mean(valid_lz30):.4f}")
    valid_lz60 = [v for v in lz60 if not np.isnan(v)]
    print(f"  LZ60 range: {min(valid_lz60):.4f} - {max(valid_lz60):.4f}, mean={np.mean(valid_lz60):.4f}")

    # Check LZ at loss dates
    trades_x = simulate_trades(signals, make_tf4_vix_combo(cascade_vix), "X")
    print(f"\n  LZ complexity at X: TF4+VIX entry dates:")
    for t in trades_x:
        if t.pnl <= 0:
            lz = lz30_by_date.get(t.entry_date)
            lz6 = lz60_by_date.get(t.entry_date)
            print(f"    LOSS {t.entry_date.date()} PnL=${t.pnl:>+7,.0f} "
                  f"LZ30={lz:.4f} LZ60={lz6:.4f}" if lz and lz6 else f"LZ30={'N/A' if not lz else f'{lz:.4f}'} LZ60={'N/A' if not lz6 else f'{lz6:.4f}'}")

    win_lz = [lz30_by_date.get(t.entry_date) for t in trades_x if t.pnl > 0]
    win_lz = [v for v in win_lz if v is not None]
    if win_lz:
        print(f"  Win LZ30 mean: {np.mean(win_lz):.4f} (std {np.std(win_lz):.4f})")

    # LZ filter sweep
    print(f"\n  --- LZ30 filter sweep (block if LZ < threshold = too ordered) ---")
    for pctile in [10, 20, 30, 40, 50]:
        threshold = np.percentile(valid_lz30, pctile)
        def make_lz_filter(base_fn, lz_dict, thresh):
            def fn(day):
                lz = lz_dict.get(day.date)
                if lz is not None and lz < thresh:
                    return None
                return base_fn(day)
            return fn
        fn = make_lz_filter(make_tf4_vix_combo(cascade_vix), lz30_by_date, threshold)
        trades = simulate_trades(signals, fn, f"LZ30>p{pctile}")
        print(_summary_line(trades, f"LZ30 > p{pctile} ({threshold:.4f})"))

    # Also try blocking high LZ (too random)
    print(f"\n  --- LZ30 filter (block if LZ > threshold = too random) ---")
    for pctile in [60, 70, 80, 90]:
        threshold = np.percentile(valid_lz30, pctile)
        def make_lz_filter_high(base_fn, lz_dict, thresh):
            def fn(day):
                lz = lz_dict.get(day.date)
                if lz is not None and lz > thresh:
                    return None
                return base_fn(day)
            return fn
        fn = make_lz_filter_high(make_tf4_vix_combo(cascade_vix), lz30_by_date, threshold)
        trades = simulate_trades(signals, fn, f"LZ30<p{pctile}")
        print(_summary_line(trades, f"LZ30 < p{pctile} ({threshold:.4f})"))


# ---------------------------------------------------------------------------
# EXPERIMENT 2: CROSS-ASSET LEAD-LAG RESIDUAL
# ---------------------------------------------------------------------------

def run_leadlag_experiment(signals, cascade_vix, spy_daily):
    """TSLA vs SPY regression residual as a filter."""
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 2: CROSS-ASSET LEAD-LAG RESIDUAL (TSLA vs SPY)")
    print("=" * 100)

    if spy_daily is None:
        print("  SPY data not available, skipping")
        return

    # Build aligned return series
    tsla_dates = [s.date for s in signals]
    tsla_close = np.array([s.day_close for s in signals], dtype=float)
    tsla_ret = np.diff(np.log(tsla_close))
    tsla_ret = np.insert(tsla_ret, 0, 0.0)

    spy_close_dict = {}
    spy_ret_dict = {}
    spy_c = spy_daily['close'].values.astype(float)
    spy_d = spy_daily.index
    spy_rets = np.diff(np.log(spy_c))
    spy_rets = np.insert(spy_rets, 0, 0.0)
    for i, d in enumerate(spy_d):
        spy_close_dict[d] = spy_c[i]
        spy_ret_dict[d] = spy_rets[i]

    # Rolling regression: TSLA_ret = alpha + beta * SPY_ret + residual
    window = 60
    residual_by_date = {}
    beta_by_date = {}

    for i in range(window, len(signals)):
        tsla_r = []
        spy_r = []
        for j in range(i - window, i):
            sr = spy_ret_dict.get(tsla_dates[j])
            if sr is not None:
                tsla_r.append(tsla_ret[j])
                spy_r.append(sr)

        if len(tsla_r) < window // 2:
            continue

        tsla_r = np.array(tsla_r)
        spy_r = np.array(spy_r)

        # Simple OLS
        spy_mean = spy_r.mean()
        tsla_mean = tsla_r.mean()
        cov = ((spy_r - spy_mean) * (tsla_r - tsla_mean)).sum()
        var = ((spy_r - spy_mean) ** 2).sum()
        if var > 0:
            beta = cov / var
            alpha = tsla_mean - beta * spy_mean
        else:
            beta = 0
            alpha = tsla_mean

        # Current day residual
        spy_today = spy_ret_dict.get(tsla_dates[i])
        if spy_today is not None:
            predicted = alpha + beta * spy_today
            residual = tsla_ret[i] - predicted
            residual_by_date[tsla_dates[i]] = residual
            beta_by_date[tsla_dates[i]] = beta

    valid_resid = list(residual_by_date.values())
    print(f"  Residual stats: mean={np.mean(valid_resid):.5f}, std={np.std(valid_resid):.5f}")
    print(f"  Beta stats: mean={np.mean(list(beta_by_date.values())):.3f}")

    # Check residuals at loss dates
    trades_x = simulate_trades(signals, make_tf4_vix_combo(cascade_vix), "X")
    print(f"\n  Residual at X: TF4+VIX entry dates:")
    for t in trades_x:
        if t.pnl <= 0:
            r = residual_by_date.get(t.entry_date)
            b = beta_by_date.get(t.entry_date)
            print(f"    LOSS {t.entry_date.date()} PnL=${t.pnl:>+7,.0f} "
                  f"resid={r:.5f} beta={b:.3f}" if r is not None and b is not None else
                  f"resid=N/A beta=N/A")

    # Residual z-score filter
    resid_std = np.std(valid_resid)
    print(f"\n  --- Residual z-score filter (block extreme residuals) ---")
    for max_z in [3.0, 2.0, 1.5, 1.0, 0.5]:
        def make_resid_filter(base_fn, r_dict, r_std, max_zscore):
            def fn(day):
                r = r_dict.get(day.date)
                if r is not None and abs(r / r_std) > max_zscore:
                    return None
                return base_fn(day)
            return fn
        fn = make_resid_filter(make_tf4_vix_combo(cascade_vix), residual_by_date, resid_std, max_z)
        trades = simulate_trades(signals, fn, f"|z|<{max_z}")
        print(_summary_line(trades, f"residual |z| < {max_z}"))


# ---------------------------------------------------------------------------
# EXPERIMENT 3: VOLATILITY RELAXATION HALF-LIFE
# ---------------------------------------------------------------------------

def run_vol_relaxation_experiment(signals, cascade_vix):
    """Measure how smoothly volatility decays after a shock.
    Smooth decay = safe entry; jagged aftershocks = avoid."""
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 3: VOLATILITY RELAXATION HALF-LIFE")
    print("=" * 100)

    closes = np.array([s.day_close for s in signals], dtype=float)
    returns = np.diff(closes) / closes[:-1]
    returns = np.insert(returns, 0, 0.0)

    # Compute rolling vol (5-day and 20-day)
    vol5 = pd.Series(np.abs(returns)).rolling(5).mean().values
    vol20 = pd.Series(np.abs(returns)).rolling(20).mean().values

    # Vol ratio: short-term / long-term. >1 means recent shock, <1 means calm
    vol_ratio = np.where(vol20 > 0, vol5 / vol20, 1.0)

    # Relaxation smoothness: std of vol_ratio over last 10 days
    # Low std = smooth decay, high std = jagged
    relax_smoothness = pd.Series(vol_ratio).rolling(10).std().values

    dates = [s.date for s in signals]
    vol_ratio_by_date = {dates[i]: vol_ratio[i] for i in range(len(signals)) if not np.isnan(vol_ratio[i])}
    smoothness_by_date = {dates[i]: relax_smoothness[i] for i in range(len(signals)) if not np.isnan(relax_smoothness[i])}

    valid_smooth = [v for v in relax_smoothness if not np.isnan(v)]
    print(f"  Relaxation smoothness: mean={np.mean(valid_smooth):.4f}, "
          f"std={np.std(valid_smooth):.4f}")

    # Check at loss dates
    trades_x = simulate_trades(signals, make_tf4_vix_combo(cascade_vix), "X")
    print(f"\n  Vol relaxation at X: TF4+VIX entry dates:")
    for t in trades_x:
        if t.pnl <= 0:
            vr = vol_ratio_by_date.get(t.entry_date)
            sm = smoothness_by_date.get(t.entry_date)
            print(f"    LOSS {t.entry_date.date()} PnL=${t.pnl:>+7,.0f} "
                  f"vol_ratio={vr:.3f} smoothness={sm:.4f}" if vr is not None and sm is not None else
                  f"vol_ratio=N/A smoothness=N/A")

    # Smoothness filter sweep
    print(f"\n  --- Relaxation smoothness filter (block if smoothness > threshold = jagged) ---")
    for pctile in [50, 60, 70, 80, 90]:
        threshold = np.percentile(valid_smooth, pctile)
        def make_smooth_filter(base_fn, s_dict, thresh):
            def fn(day):
                sm = s_dict.get(day.date)
                if sm is not None and sm > thresh:
                    return None
                return base_fn(day)
            return fn
        fn = make_smooth_filter(make_tf4_vix_combo(cascade_vix), smoothness_by_date, threshold)
        trades = simulate_trades(signals, fn, f"smooth<p{pctile}")
        print(_summary_line(trades, f"smoothness < p{pctile} ({threshold:.4f})"))

    # Vol ratio filter (block during shock)
    print(f"\n  --- Vol ratio filter (block if vol5/vol20 > threshold) ---")
    for max_vr in [0.8, 1.0, 1.2, 1.5, 2.0]:
        def make_vr_filter(base_fn, vr_dict, thresh):
            def fn(day):
                vr = vr_dict.get(day.date)
                if vr is not None and vr > thresh:
                    return None
                return base_fn(day)
            return fn
        fn = make_vr_filter(make_tf4_vix_combo(cascade_vix), vol_ratio_by_date, max_vr)
        trades = simulate_trades(signals, fn, f"volratio<{max_vr}")
        print(_summary_line(trades, f"vol_ratio < {max_vr}"))


# ---------------------------------------------------------------------------
# EXPERIMENT 4: LOSS-ONLY ANOMALY DETECTION
# ---------------------------------------------------------------------------

def run_anomaly_experiment(signals, cascade_vix, vix_daily):
    """Use isolation forest trained only on loss features to veto trades."""
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 4: LOSS-ONLY ANOMALY DETECTION")
    print("=" * 100)

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  scikit-learn not available, skipping")
        return

    # Build feature vectors for all trades
    trades_x = simulate_trades(signals, make_tf4_vix_combo(cascade_vix), "X")

    # VIX lookup
    vix_by_date = {}
    if vix_daily is not None:
        for d, row in vix_daily.iterrows():
            vix_by_date[d] = float(row['close'])

    # Match trades to signal features
    features = []
    labels = []  # 1=loss, 0=win
    trade_dates = []

    for t in trades_x:
        # Find signal day (day before entry)
        for i, s in enumerate(signals):
            if i + 1 < len(signals) and signals[i+1].date == t.entry_date:
                feat = [
                    t.confidence,
                    _count_tf_confirming(s, 'BUY' if t.direction == 'LONG' else 'SELL'),
                    1 if t.direction == 'LONG' else -1,
                    t.entry_date.dayofweek,
                    t.entry_date.month,
                    vix_by_date.get(s.date, 20.0),
                    s.cs_position_score,
                    s.cs_energy_score,
                    s.cs_entropy_score,
                    s.cs_confluence_score,
                    s.cs_timing_score,
                    s.cs_channel_health,
                    # Price-derived features
                    (s.day_high - s.day_low) / s.day_close if s.day_close > 0 else 0,  # daily range
                ]
                features.append(feat)
                labels.append(1 if t.pnl <= 0 else 0)
                trade_dates.append(t.entry_date)
                break

    X = np.array(features)
    y = np.array(labels)
    print(f"  Total trades: {len(y)} ({sum(y)} losses, {len(y)-sum(y)} wins)")

    if sum(y) == 0:
        print("  No losses to detect!")
        return

    # Isolation Forest: train on all data, see if losses are flagged as anomalies
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sweep contamination parameter
    print(f"\n  --- Isolation Forest anomaly detection ---")
    for contam in [0.01, 0.02, 0.05, 0.10, 0.15]:
        clf = IsolationForest(contamination=contam, random_state=42, n_estimators=100)
        preds = clf.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal

        anomaly_mask = preds == -1
        normal_mask = preds == 1

        # How many losses caught?
        losses_caught = sum(anomaly_mask & (y == 1))
        wins_caught = sum(anomaly_mask & (y == 0))  # false positives
        remaining_trades = sum(normal_mask)
        remaining_wins = sum(normal_mask & (y == 0))
        remaining_losses = sum(normal_mask & (y == 1))
        remaining_wr = remaining_wins / remaining_trades * 100 if remaining_trades > 0 else 0

        print(f"  contam={contam:.2f}: flagged {sum(anomaly_mask)} anomalies "
              f"(caught {losses_caught}/{sum(y)} losses, {wins_caught} false pos) "
              f"-> {remaining_trades} trades, {remaining_wr:.1f}% WR")

    # LOF approach
    print(f"\n  --- Local Outlier Factor ---")
    for n_neighbors in [5, 10, 20]:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
        preds = lof.fit_predict(X_scaled)
        anomaly_mask = preds == -1
        normal_mask = preds == 1
        losses_caught = sum(anomaly_mask & (y == 1))
        wins_caught = sum(anomaly_mask & (y == 0))
        remaining = sum(normal_mask)
        rem_wr = sum(normal_mask & (y == 0)) / remaining * 100 if remaining > 0 else 0
        print(f"  k={n_neighbors}: flagged {sum(anomaly_mask)} "
              f"(caught {losses_caught}/{sum(y)} losses, {wins_caught} FP) "
              f"-> {remaining} trades, {rem_wr:.1f}% WR")

    # Feature importance: which features differ most between wins and losses?
    print(f"\n  Feature analysis (wins vs losses):")
    feat_names = ['confidence', 'tf_count', 'direction', 'dow', 'month', 'vix',
                  'position', 'energy', 'entropy', 'confluence', 'timing',
                  'health', 'daily_range']
    for i, name in enumerate(feat_names):
        w_vals = X[y == 0, i]
        l_vals = X[y == 1, i]
        if len(l_vals) == 0:
            continue
        w_mean = w_vals.mean()
        l_mean = l_vals.mean()
        effect = (l_mean - w_mean) / w_vals.std() if w_vals.std() > 0 else 0
        print(f"    {name:<15} Win={w_mean:>7.3f}  Loss={l_mean:>7.3f}  effect={effect:>+.3f}")


# ---------------------------------------------------------------------------
# EXPERIMENT 5: COMBINED BEST (SPY+LZ+VolRelax)
# ---------------------------------------------------------------------------

def run_combined_best(signals, cascade_vix, spy_daily):
    """Combine SPY regime + best additional filter."""
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 5: COMBINED BEST (SPY + additional filters)")
    print("=" * 100)

    if spy_daily is None:
        print("  SPY data not available, skipping")
        return

    # Pre-compute SPY SMA20
    spy_close = spy_daily['close'].values.astype(float)
    spy_sma20 = pd.Series(spy_close).rolling(20).mean().values
    spy_above = set()
    for i in range(20, len(spy_close)):
        if spy_close[i] > spy_sma20[i]:
            spy_above.add(spy_daily.index[i])

    # Pre-compute vol relaxation
    closes = np.array([s.day_close for s in signals], dtype=float)
    returns = np.diff(closes) / closes[:-1]
    returns = np.insert(returns, 0, 0.0)
    vol5 = pd.Series(np.abs(returns)).rolling(5).mean().values
    vol20 = pd.Series(np.abs(returns)).rolling(20).mean().values
    vol_ratio = np.where(vol20 > 0, vol5 / vol20, 1.0)
    vr_by_date = {signals[i].date: vol_ratio[i] for i in range(len(signals)) if not np.isnan(vol_ratio[i])}

    configs = [
        ('X: TF4+VIX baseline', lambda: make_tf4_vix_combo(cascade_vix), 2, 4),
    ]

    # SPY filter
    def make_spy_combo(cascade, spy_set):
        def fn(day):
            if day.date not in spy_set:
                return None
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                if _count_tf_confirming(day, day.cs_action) < 4:
                    return None
                sig = _SigProxy(day)
                ana = _AnalysisProxy(day.cs_tf_states)
                ok, adj, _ = cascade.evaluate(
                    sig, ana, feature_vec=None, bar_datetime=day.date,
                    higher_tf_data=None, spy_df=None, vix_df=None,
                )
                if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                    return None
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, adj, s, t, 'CS')
            return None
        return fn

    configs.append(('AJ: TF4+VIX+SPY', lambda: make_spy_combo(cascade_vix, spy_above), 2, 4))
    configs.append(('AJ cd=0', lambda: make_spy_combo(cascade_vix, spy_above), 0, 4))

    # SPY + health>=0.35
    def make_spy_health_combo(cascade, spy_set, min_h):
        def fn(day):
            if day.date not in spy_set:
                return None
            if day.cs_channel_health < min_h:
                return None
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                if _count_tf_confirming(day, day.cs_action) < 4:
                    return None
                sig = _SigProxy(day)
                ana = _AnalysisProxy(day.cs_tf_states)
                ok, adj, _ = cascade.evaluate(
                    sig, ana, feature_vec=None, bar_datetime=day.date,
                    higher_tf_data=None, spy_df=None, vix_df=None,
                )
                if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                    return None
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, adj, s, t, 'CS')
            return None
        return fn

    configs.append(('SPY+health>=0.35', lambda: make_spy_health_combo(cascade_vix, spy_above, 0.35), 2, 4))

    # SPY + vol relaxation
    def make_spy_vr_combo(cascade, spy_set, vr_dict, max_vr):
        def fn(day):
            if day.date not in spy_set:
                return None
            vr = vr_dict.get(day.date)
            if vr is not None and vr > max_vr:
                return None
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                if _count_tf_confirming(day, day.cs_action) < 4:
                    return None
                sig = _SigProxy(day)
                ana = _AnalysisProxy(day.cs_tf_states)
                ok, adj, _ = cascade.evaluate(
                    sig, ana, feature_vec=None, bar_datetime=day.date,
                    higher_tf_data=None, spy_df=None, vix_df=None,
                )
                if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                    return None
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, adj, s, t, 'CS')
            return None
        return fn

    configs.append(('SPY+volratio<1.5', lambda: make_spy_vr_combo(cascade_vix, spy_above, vr_by_date, 1.5), 2, 4))
    configs.append(('SPY+volratio<1.2', lambda: make_spy_vr_combo(cascade_vix, spy_above, vr_by_date, 1.2), 2, 4))

    # Best combos with cd=0
    configs.append(('SPY+health>=0.35 cd=0', lambda: make_spy_health_combo(cascade_vix, spy_above, 0.35), 0, 4))
    configs.append(('SPY+vr<1.5 cd=0', lambda: make_spy_vr_combo(cascade_vix, spy_above, vr_by_date, 1.5), 0, 4))

    for label, fn_maker, cd, tp in configs:
        trades = simulate_trades(signals, fn_maker(), label, cooldown=cd, trail_power=tp)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cache_path = Path(__file__).parent / 'combo_cache' / 'combo_signals.pkl'
    print(f"Loading from {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    signals = data['signals']
    vix_daily = data.get('vix_daily')
    spy_daily = data.get('spy_daily')
    print(f"  {len(signals)} days, {signals[0].date.date()} to {signals[-1].date.date()}\n")

    cascade_vix = _build_filter_cascade(vix=True)
    if vix_daily is not None:
        cascade_vix.precompute_vix_cooldown(vix_daily)
        print(f"[FILTER] VIX cooldown precomputed\n")

    run_lz_experiment(signals, cascade_vix)
    run_leadlag_experiment(signals, cascade_vix, spy_daily)
    run_vol_relaxation_experiment(signals, cascade_vix)
    run_anomaly_experiment(signals, cascade_vix, vix_daily)
    run_combined_best(signals, cascade_vix, spy_daily)

    print("\n\n" + "=" * 100)
    print("  ALL v10 EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
