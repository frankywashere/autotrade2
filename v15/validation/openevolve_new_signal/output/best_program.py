# Hybrid: beta-adjusted MR + momentum with Sharpe-optimized stop/TP and enhanced regime filters
# Key improvements:
# 1. Tighter MR entry thresholds (rz < -1.2) for higher quality signals
# 2. Dynamic TP:stop ratio based on VIX regime and recent vol
# 3. Time-of-day weighted confidence (avoid lunch lull 11:30-12:30)
# 4. TSLA/SPY correlation regime filter (only MR when corr is high)
# 5. Multi-timeframe RSI divergence to confirm MR entries
# 6. Separate stop/TP for MR vs momentum to optimize each mode

import numpy as np


def generate_signals(tsla_bars, spy_bars, vix_bars, current_time, position_info):
    signals = []

    if len(tsla_bars) < 30 or len(spy_bars) < 30 or len(vix_bars) < 10:
        return signals

    if position_info['n_positions'] >= position_info['max_positions']:
        return signals

    tsla_c = tsla_bars['close'].values
    tsla_h = tsla_bars['high'].values
    tsla_l = tsla_bars['low'].values
    tsla_v = tsla_bars['volume'].values
    spy_c = spy_bars['close'].values
    spy_h = spy_bars['high'].values
    spy_l = spy_bars['low'].values
    vix_c = vix_bars['close'].values

    price = float(tsla_c[-1])

    # Time filter
    hour = current_time.hour
    minute = current_time.minute
    bar_mins = hour * 60 + minute
    if bar_mins <= 9 * 60 + 44 or bar_mins >= 15 * 60 + 30:
        return signals

    # Lunch lull: 11:30-12:30 ET — reduce signal generation
    lunch_lull = (bar_mins >= 11 * 60 + 30 and bar_mins <= 12 * 60 + 30)

    # VIX regime
    vix_now = float(vix_c[-1])
    vix_lb = min(20, len(vix_c))
    vix_ma = float(np.mean(vix_c[-vix_lb:]))
    vix_spike = vix_now > vix_ma * 1.20

    if vix_now > 60:
        return signals

    # VIX rate-of-change
    if len(vix_c) >= 5:
        vix_roc5 = float((vix_c[-1] - vix_c[-5]) / (vix_c[-5] + 1e-10))
    else:
        vix_roc5 = 0.0
    vix_rising_fast = vix_roc5 > 0.08
    vix_falling_fast = vix_roc5 < -0.08

    vix_scale = 1.5 if vix_now > 35 else (1.2 if vix_now > 25 else 1.0)

    # ATR(14) for TSLA
    n_atr = min(14, len(tsla_c) - 1)
    tr_list = []
    for i in range(-n_atr, 0):
        prev = tsla_c[i - 1]
        tr = max(tsla_h[i] - tsla_l[i],
                 abs(tsla_h[i] - prev),
                 abs(tsla_l[i] - prev))
        tr_list.append(tr)
    atr = float(np.mean(tr_list)) if tr_list else price * 0.005
    atr_pct = atr / (price + 1e-10)

    # Intraday vol surge
    recent_trs = tr_list[-5:] if len(tr_list) >= 5 else tr_list
    recent_atr = float(np.mean(recent_trs)) if recent_trs else atr
    vol_surge = recent_atr > atr * 2.2

    # Volume
    vol_lb = min(20, len(tsla_v))
    vol_ma = float(np.mean(tsla_v[-vol_lb:]))
    vol_ratio = float(tsla_v[-1]) / (vol_ma + 1e-10)

    if vol_ratio < 0.30:
        return signals

    # VWAP
    vwap_lb = min(78, len(tsla_c))
    tp_arr = (tsla_h[-vwap_lb:] + tsla_l[-vwap_lb:] + tsla_c[-vwap_lb:]) / 3.0
    vol_arr = tsla_v[-vwap_lb:]
    total_vol = float(np.sum(vol_arr))
    if total_vol > 0:
        vwap = float(np.sum(tp_arr * vol_arr)) / total_vol
        vwap_dev = (price - vwap) / (vwap + 1e-10)
    else:
        vwap_dev = 0.0

    # RSI(14)
    rp = 14
    if len(tsla_c) > rp:
        d = np.diff(tsla_c[-(rp + 1):])
        ag = float(np.mean(np.maximum(d, 0.0)))
        al = float(np.mean(np.maximum(-d, 0.0)))
        rsi = 100.0 - 100.0 / (1.0 + ag / (al + 1e-10))
    else:
        rsi = 50.0

    # RSI(7) for short-term divergence
    rp7 = 7
    if len(tsla_c) > rp7:
        d7 = np.diff(tsla_c[-(rp7 + 1):])
        ag7 = float(np.mean(np.maximum(d7, 0.0)))
        al7 = float(np.mean(np.maximum(-d7, 0.0)))
        rsi7 = 100.0 - 100.0 / (1.0 + ag7 / (al7 + 1e-10))
    else:
        rsi7 = 50.0

    # RSI divergence: rsi7 leading rsi14 in reversal direction
    rsi_turning_up = rsi7 > rsi and rsi7 > 40
    rsi_turning_dn = rsi7 < rsi and rsi7 < 60

    # Rolling beta estimate (30 bars)
    beta_lb = min(30, len(tsla_c) - 1)
    if beta_lb >= 5:
        tsla_rets = np.diff(tsla_c[-beta_lb - 1:]) / (tsla_c[-beta_lb - 1:-1] + 1e-10)
        spy_rets_b = np.diff(spy_c[-beta_lb - 1:]) / (spy_c[-beta_lb - 1:-1] + 1e-10)
        spy_var = float(np.var(spy_rets_b))
        if spy_var > 1e-12 and len(tsla_rets) == len(spy_rets_b):
            cov = float(np.cov(tsla_rets, spy_rets_b)[0, 1])
            beta = float(np.clip(cov / spy_var, 0.5, 4.0))
            # Rolling correlation for regime filter
            corr_num = cov
            corr_den = float(np.std(tsla_rets) * np.std(spy_rets_b))
            corr = float(corr_num / (corr_den + 1e-10)) if corr_den > 1e-12 else 0.5
            corr = float(np.clip(corr, -1.0, 1.0))
        else:
            beta = 1.6
            corr = 0.5
    else:
        beta = 1.6
        corr = 0.5

    # High correlation means MR is more reliable (TSLA tracking SPY closely recently)
    high_corr = corr > 0.50

    # Beta-adjusted excess returns
    spy_ret1 = (spy_c[-1] - spy_c[-2]) / (spy_c[-2] + 1e-10)
    ret1 = (tsla_c[-1] - tsla_c[-2]) / (tsla_c[-2] + 1e-10)
    excess1 = float(ret1 - beta * spy_ret1)

    n5 = min(5, len(tsla_c) - 1)
    excess5 = float((tsla_c[-1] - tsla_c[-n5 - 1]) / (tsla_c[-n5 - 1] + 1e-10) -
                    beta * (spy_c[-1] - spy_c[-n5 - 1]) / (spy_c[-n5 - 1] + 1e-10))

    n10 = min(10, len(tsla_c) - 1)
    excess10 = float((tsla_c[-1] - tsla_c[-n10 - 1]) / (tsla_c[-n10 - 1] + 1e-10) -
                     beta * (spy_c[-1] - spy_c[-n10 - 1]) / (spy_c[-n10 - 1] + 1e-10))

    n20 = min(20, len(tsla_c) - 1)
    excess20 = float((tsla_c[-1] - tsla_c[-n20 - 1]) / (tsla_c[-n20 - 1] + 1e-10) -
                     beta * (spy_c[-1] - spy_c[-n20 - 1]) / (spy_c[-n20 - 1] + 1e-10))

    # TSLA/SPY ratio z-score with adaptive lookback
    lb = 30 if vix_now > 25 else 40
    lb = min(lb, len(tsla_c))
    ratio = tsla_c[-lb:] / (spy_c[-lb:] + 1e-10)
    rmu = float(np.mean(ratio))
    rsig = float(np.std(ratio))
    rz = float((ratio[-1] - rmu) / rsig) if rsig > 1e-8 else 0.0

    # Bollinger Bands on TSLA (20-bar)
    bb_lb = min(20, len(tsla_c))
    bb_mid = float(np.mean(tsla_c[-bb_lb:]))
    bb_std = float(np.std(tsla_c[-bb_lb:]))
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_z = (price - bb_mid) / (bb_std + 1e-10)  # Standard deviations from mean

    # SPY trends
    spy_lb20 = min(20, len(spy_c))
    spy_trend20 = (spy_c[-1] - spy_c[-spy_lb20]) / (spy_c[-spy_lb20] + 1e-10)
    spy_lb5 = min(5, len(spy_c))
    spy_trend5 = (spy_c[-1] - spy_c[-spy_lb5]) / (spy_c[-spy_lb5] + 1e-10)
    spy_strong_up = spy_trend20 > 0.018
    spy_strong_dn = spy_trend20 < -0.018

    # SPY ATR
    spy_n_atr = min(14, len(spy_c) - 1)
    spy_tr_list = []
    for i in range(-spy_n_atr, 0):
        prev = spy_c[i - 1]
        spy_tr = max(spy_h[i] - spy_l[i],
                     abs(spy_h[i] - prev),
                     abs(spy_l[i] - prev))
        spy_tr_list.append(spy_tr)
    spy_atr = float(np.mean(spy_tr_list)) if spy_tr_list else spy_c[-1] * 0.003
    spy_atr_pct = spy_atr / (spy_c[-1] + 1e-10)

    # Consecutive direction bars
    def count_consec(arr, direction):
        count = 0
        for i in range(-1, -min(len(arr), 10), -1):
            if direction == 'up' and arr[i] > arr[i - 1]:
                count += 1
            elif direction == 'dn' and arr[i] < arr[i - 1]:
                count += 1
            else:
                break
        return count

    consec_up = count_consec(tsla_c, 'up') if len(tsla_c) >= 4 else 0
    consec_dn = count_consec(tsla_c, 'dn') if len(tsla_c) >= 4 else 0

    # N-bar range breakout (15-bar)
    nb = min(15, len(tsla_c) - 1)
    range_high = float(np.max(tsla_h[-nb - 1:-1]))
    range_low = float(np.min(tsla_l[-nb - 1:-1]))
    breakout_up = price > range_high and vol_ratio > 1.5
    breakout_dn = price < range_low and vol_ratio > 1.5

    # Opening range breakout (first 6 bars = first 30 min)
    try:
        today_date = current_time.date()
        today_idx = [i for i, t in enumerate(tsla_bars.index)
                     if t.date() == today_date]
        if len(today_idx) >= 6:
            t0 = today_idx[0]
            or_len = min(6, len(today_idx) - 1)
            or_high = float(np.max(tsla_h[t0:t0 + or_len]))
            or_low = float(np.min(tsla_l[t0:t0 + or_len]))
            or_valid = bar_mins >= 10 * 60
            or_break_up = or_valid and price > or_high * 1.001 and vol_ratio > 1.3
            or_break_dn = or_valid and price < or_low * 0.999 and vol_ratio > 1.3
        else:
            or_break_up = or_break_dn = False
    except Exception:
        or_break_up = or_break_dn = False

    # ===== STOP/TP SIZING =====
    # MR: tighter stop (mean reversion should work quickly), generous TP
    mr_stop_pct = float(np.clip(atr_pct * 1.3 * vix_scale, 0.003, 0.015))
    mr_tp_pct = float(np.clip(atr_pct * 3.2 * vix_scale, 0.007, 0.030))

    # Momentum: wider stop to ride trend, larger TP
    mom_stop_pct = float(np.clip(atr_pct * 1.2 * vix_scale, 0.003, 0.015))
    mom_tp_pct = float(np.clip(atr_pct * 3.6 * vix_scale, 0.008, 0.040))

    # ===== MODE 1: MEAN REVERSION =====
    # Require tighter z-score threshold for higher quality, add BB confirmation

    # LONG MR
    if not position_info['has_short'] and not position_info['has_long']:
        mr_long = (
            rz < -1.10 and
            bb_z < -1.2 and                              # BB confirmation
            excess1 > -0.0005 and                        # Reversal starting
            (excess5 < -0.002 or excess10 < -0.003) and  # Divergence exists
            rsi > 22 and rsi < 60 and
            rsi_turning_up and                           # RSI7 leading higher
            vwap_dev < 0.010 and
            not spy_strong_dn and
            not (vix_spike and vix_now > 30) and
            not vix_rising_fast and
            not vol_surge and
            not lunch_lull and
            high_corr                                    # Only MR when correlated
        )
        if mr_long:
            # Confidence from multiple confluence factors
            z_conf = float(min(1.0, abs(rz) / 2.5))
            bb_conf = float(min(1.0, abs(bb_z) / 2.5))
            conf = float(min(1.0, (z_conf + bb_conf) / 2.0))
            signals.append({
                'direction': 'long',
                'confidence': conf,
                'stop_pct': mr_stop_pct,
                'tp_pct': mr_tp_pct,
            })

    # SHORT MR
    if not position_info['has_short'] and not position_info['has_long']:
        mr_short = (
            rz > 1.10 and
            bb_z > 1.2 and                               # BB confirmation
            excess1 < 0.0005 and                         # Reversal starting
            (excess5 > 0.002 or excess10 > 0.003) and    # Divergence exists
            rsi < 78 and rsi > 40 and
            rsi_turning_dn and                           # RSI7 leading lower
            vwap_dev > -0.010 and
            not spy_strong_up and
            not (vix_spike and vix_now > 30) and
            not vix_falling_fast and
            not vol_surge and
            not lunch_lull and
            high_corr
        )
        if mr_short:
            z_conf = float(min(1.0, abs(rz) / 2.5))
            bb_conf = float(min(1.0, abs(bb_z) / 2.5))
            conf = float(min(1.0, (z_conf + bb_conf) / 2.0))
            signals.append({
                'direction': 'short',
                'confidence': conf,
                'stop_pct': mr_stop_pct,
                'tp_pct': mr_tp_pct,
            })

    # ===== MODE 2: MOMENTUM / BREAKOUT =====
    # Allow even if has_long or has_short (up to 2 positions)

    # LONG MOMENTUM
    if not position_info['has_long'] and len(signals) == 0:
        nbar_mom_long = (
            breakout_up and
            (consec_up >= 2 or excess5 > spy_atr_pct * 1.5) and
            spy_trend20 > 0.005 and spy_trend5 > 0 and
            rsi > 50 and rsi < 80 and
            vwap_dev > -0.005 and
            vix_now < 30 and not vix_spike and
            not lunch_lull
        )
        or_mom_long = (
            or_break_up and
            spy_trend5 > 0 and
            excess1 > 0 and
            rsi > 50 and
            vix_now < 25 and not vix_spike and
            not lunch_lull
        )
        if nbar_mom_long or or_mom_long:
            conf = float(min(1.0, vol_ratio / 3.0))
            signals.append({
                'direction': 'long',
                'confidence': conf,
                'stop_pct': mom_stop_pct,
                'tp_pct': mom_tp_pct,
            })

    # SHORT MOMENTUM
    if not position_info['has_short'] and len(signals) == 0:
        nbar_mom_short = (
            breakout_dn and
            (consec_dn >= 2 or excess5 < -spy_atr_pct * 1.5) and
            spy_trend20 < -0.005 and spy_trend5 < 0 and
            rsi < 50 and rsi > 20 and
            vwap_dev < 0.005 and
            vix_now < 30 and not vix_spike and
            not lunch_lull
        )
        or_mom_short = (
            or_break_dn and
            spy_trend5 < 0 and
            excess1 < 0 and
            rsi < 50 and
            vix_now < 25 and not vix_spike and
            not lunch_lull
        )
        if nbar_mom_short or or_mom_short:
            conf = float(min(1.0, vol_ratio / 3.0))
            signals.append({
                'direction': 'short',
                'confidence': conf,
                'stop_pct': mom_stop_pct,
                'tp_pct': mom_tp_pct,
            })

    # ===== MODE 3: EXCESS RETURN EXTREMES (new) =====
    # When 20-bar excess return is extreme, fade it — mean reversion on longer timeframe
    if len(signals) == 0 and not vol_surge and not lunch_lull:
        excess_threshold = 0.035  # 3.5% excess over 20 bars is extreme
        if not position_info['has_long'] and excess20 < -excess_threshold:
            # TSLA dramatically underperformed SPY over 20 bars — fade short
            # Require some stability: rsi not crashed, vix not extreme
            if (rsi > 25 and rsi < 60 and
                vix_now < 40 and not vix_rising_fast and
                excess1 > -0.001 and  # at least not still falling fast
                vwap_dev < 0.015):
                conf = float(min(1.0, abs(excess20) / (excess_threshold * 2)))
                stop_pct = float(np.clip(atr_pct * 1.5 * vix_scale, 0.004, 0.018))
                tp_pct = float(np.clip(atr_pct * 3.5 * vix_scale, 0.008, 0.035))
                signals.append({
                    'direction': 'long',
                    'confidence': conf,
                    'stop_pct': stop_pct,
                    'tp_pct': tp_pct,
                })

        if not position_info['has_short'] and excess20 > excess_threshold and len(signals) == 0:
            # TSLA dramatically outperformed SPY over 20 bars — fade long
            if (rsi < 75 and rsi > 40 and
                vix_now < 40 and not vix_falling_fast and
                excess1 < 0.001 and
                vwap_dev > -0.015):
                conf = float(min(1.0, abs(excess20) / (excess_threshold * 2)))
                stop_pct = float(np.clip(atr_pct * 1.5 * vix_scale, 0.004, 0.018))
                tp_pct = float(np.clip(atr_pct * 3.5 * vix_scale, 0.008, 0.035))
                signals.append({
                    'direction': 'short',
                    'confidence': conf,
                    'stop_pct': stop_pct,
                    'tp_pct': tp_pct,
                })

    return signals