#!/usr/bin/env python3
"""V14b: Jan-Feb breakdown per year + $100K compound year-by-year.
Quick analysis requested by user to contextualize 2026 OOS (Jan-Feb only).
"""
import argparse
import os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime as dt

from v15.validation.ah_rules import is_rth, is_extended_hours, AHStateTracker

def _flush(): sys.stdout.flush()

SLIPPAGE_PCT = 0.0002
COMM_PER_SHARE = 0.005
TRAIN_END = pd.Timestamp('2021-12-31')
TEST_END  = pd.Timestamp('2026-03-04')
MKT_OPEN  = dt.time(9, 30)
MKT_CLOSE = dt.time(16, 0)

def load_1min(path=None, rth_only=True):
    if path is None:
        for p in ['data/TSLAMin.txt', r'C:\AI\x14\data\TSLAMin.txt',
                   os.path.expanduser('~/Desktop/Coding/x14/data/TSLAMin.txt')]:
            if os.path.exists(p): path = p; break
    if path is None: raise FileNotFoundError("TSLAMin.txt not found")
    print(f"Loading 1-min data from {path}..."); _flush()
    t0 = time.time()
    df = pd.read_csv(path, sep=';', names=['datetime','open','high','low','close','volume'],
                     parse_dates=['datetime'], date_format='%Y%m%d %H%M%S')
    df = df.set_index('datetime').sort_index()
    if rth_only:
        times = df.index.time
        df = df[(times >= MKT_OPEN) & (times < MKT_CLOSE)].copy()
    else:
        # Keep extended hours (4:00-20:00) for AH rules
        times = df.index.time
        df = df[(times >= dt.time(4, 0)) & (times < dt.time(20, 0))].copy()
    print(f"  Loaded {len(df):,} bars ({'RTH' if rth_only else 'RTH+AH'}) in {time.time()-t0:.1f}s"); _flush()
    return df

def resample_ohlcv(df, rule):
    return df.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

def channel_position(close_arr, window=60):
    n = len(close_arr); close = close_arr.astype(np.float64); w = window
    sx = w*(w-1)/2.0; sx2 = (w-1)*w*(2*w-1)/6.0; denom = w*sx2-sx**2
    cy = np.cumsum(close); sy = np.full(n,np.nan); sy[w-1]=cy[w-1]
    if n>w: sy[w:]=cy[w:]-cy[:n-w]
    idx=np.arange(n,dtype=np.float64); cwy=np.cumsum(idx*close)
    sxy=np.full(n,np.nan); sxy[w-1]=cwy[w-1]
    if n>w:
        si=np.arange(w,n,dtype=np.float64)-w+1
        sxy[w:]=(cwy[w:]-cwy[:n-w])-si*sy[w:]
    slope=(w*sxy-sx*sy)/denom; intercept=(sy-slope*sx)/w
    fl=slope*(w-1)+intercept
    cy2=np.cumsum(close**2); sy2=np.full(n,np.nan); sy2[w-1]=cy2[w-1]
    if n>w: sy2[w:]=cy2[w:]-cy2[:n-w]
    vy=(sy2-sy**2/w)/w; vx=denom/(w**2)
    vr=np.maximum(vy-slope**2*vx,0); sr=np.sqrt(vr)
    u=fl+2*sr; l_=fl-2*sr; wi=u-l_
    pos=np.full(n,np.nan); v=(wi>1e-10)&~np.isnan(wi)
    pos[v]=(close[v]-l_[v])/wi[v]
    return np.clip(pos,0.0,1.0)

def channel_slope(close_arr, window=60):
    n = len(close_arr); close = close_arr.astype(np.float64); w = window
    sx = w*(w-1)/2.0; sx2 = (w-1)*w*(2*w-1)/6.0; denom = w*sx2-sx**2
    cy = np.cumsum(close); sy = np.full(n,np.nan); sy[w-1]=cy[w-1]
    if n>w: sy[w:]=cy[w:]-cy[:n-w]
    idx=np.arange(n,dtype=np.float64); cwy=np.cumsum(idx*close)
    sxy=np.full(n,np.nan); sxy[w-1]=cwy[w-1]
    if n>w:
        si=np.arange(w,n,dtype=np.float64)-w+1
        sxy[w:]=(cwy[w:]-cwy[:n-w])-si*sy[w:]
    slope_arr=(w*sxy-sx*sy)/denom
    ns = np.full(n, np.nan); valid = close > 0
    ns[valid] = slope_arr[valid] / close[valid]
    return ns

def compute_rsi(c,p=14):
    n=len(c); r=np.full(n,np.nan); d=np.diff(c)
    if len(d)<p: return r
    g=np.maximum(d,0); lo=np.maximum(-d,0)
    ag=g[:p].mean(); al=lo[:p].mean()
    for i in range(p,len(d)):
        ag=(ag*(p-1)+g[i])/p; al=(al*(p-1)+lo[i])/p
        r[i+1]=100.0 if al<1e-10 else 100.0-100.0/(1.0+ag/al)
    return r

def compute_bvc(c,v,lb=20):
    n=len(c); dp=np.diff(c); dp=np.concatenate([[0],dp])
    s=pd.Series(dp).rolling(lb,min_periods=lb).std().values
    z=np.zeros(n); vl=s>1e-10; z[vl]=dp[vl]/s[vl]
    bp=norm.cdf(z); nf=v*(2*bp-1)
    nc=pd.Series(nf).rolling(lb,min_periods=lb).sum().values
    tv=pd.Series(v).rolling(lb,min_periods=lb).sum().values
    b=np.full(n,np.nan); vv=tv>0; b[vv]=nc[vv]/tv[vv]
    return b

def compute_vwap(o,h,l,c,v,dates):
    n=len(c); vwap=np.full(n,np.nan); vd=np.full(n,np.nan)
    tp=(h+l+c)/3.0; ctv=cv=0.0; pd_=None
    for i in range(n):
        d=dates[i]
        if d!=pd_: ctv=cv=0.0; pd_=d
        ctv+=tp[i]*v[i]; cv+=v[i]
        if cv>0: vwap[i]=ctv/cv; vd[i]=(c[i]-vwap[i])/vwap[i]*100.0
    return vwap,vd

def compute_volume_ratio(v, lb=20):
    n=len(v); vr=np.full(n,np.nan)
    rv=pd.Series(v).rolling(lb,min_periods=lb).mean().values
    valid=(rv>0)&~np.isnan(rv)
    vr[valid]=v[valid]/rv[valid]
    return vr

def compute_vwap_slope(vd, lb=5):
    n = len(vd); vs = np.full(n, np.nan)
    for i in range(lb, n):
        seg = vd[i-lb+1:i+1]
        if np.any(np.isnan(seg)): continue
        x = np.arange(lb, dtype=np.float64); mx = x.mean(); my = seg.mean()
        vs[i] = np.sum((x-mx)*(seg-my)) / np.sum((x-mx)**2)
    return vs

def compute_rsi_slope(rsi, lb=5):
    n = len(rsi); rs = np.full(n, np.nan)
    for i in range(lb, n):
        seg = rsi[i-lb+1:i+1]
        if np.any(np.isnan(seg)): continue
        x = np.arange(lb, dtype=np.float64); mx = x.mean(); my = seg.mean()
        rs[i] = np.sum((x-mx)*(seg-my)) / np.sum((x-mx)**2)
    return rs

def compute_spread_pct(h, l, c):
    sp = np.full(len(c), np.nan); valid = c > 0
    sp[valid] = (h[valid] - l[valid]) / c[valid] * 100.0
    return sp

def compute_gap_pct(c, dates):
    n = len(c); gap = np.full(n, np.nan)
    prev_close = None; prev_date = None
    for i in range(n):
        d = dates[i]
        if d != prev_date:
            if prev_close is not None and prev_close > 0:
                gap[i] = (c[i] - prev_close) / prev_close * 100.0
            prev_date = d
        prev_close = c[i]
    cur_gap = np.nan
    for i in range(n):
        if not np.isnan(gap[i]): cur_gap = gap[i]
        gap[i] = cur_gap
    return gap

def compute_bullish_1m_count(f1m_close, f1m_open, f5m_index, lookback=5):
    n5 = len(f5m_index); result = np.full(n5, np.nan)
    ti = f1m_close.index.values; c1 = f1m_close.values; o1 = f1m_open.values; j = 0
    for i in range(n5):
        t = f5m_index[i]
        while j < len(ti)-1 and ti[j+1] <= t: j += 1
        if j >= lookback:
            bullish = 0
            for k in range(j-lookback+1, j+1):
                if c1[k] > o1[k]: bullish += 1
            result[i] = bullish
    return result

def build_features(df1m):
    t0=time.time(); print("Resampling..."); _flush()
    tfs={}
    for rule,name in [('5min','5m'),('15min','15m'),('30min','30m'),('1h','1h'),('4h','4h')]:
        tfs[name]=resample_ohlcv(df1m,rule)
    tfs['daily']=resample_ohlcv(df1m,'1D'); tfs['1m']=df1m.copy()
    for name,df in tfs.items(): print(f"  {name}: {len(df):,}"); _flush()
    features={}
    wins={'1m':60,'5m':60,'15m':40,'30m':30,'1h':24,'4h':20,'daily':40}
    for name,df in tfs.items():
        print(f"Features {name}..."); _flush()
        c=df['close'].values.astype(np.float64); v=df['volume'].values.astype(np.float64)
        h=df['high'].values.astype(np.float64); l=df['low'].values.astype(np.float64)
        o=df['open'].values.astype(np.float64)
        feat=pd.DataFrame(index=df.index)
        feat['open']=o; feat['high']=h; feat['low']=l; feat['close']=c; feat['volume']=v
        feat['chan_pos']=channel_position(c,wins[name])
        if name in ('1h','4h','daily'):
            feat['chan_slope'] = channel_slope(c, wins[name])
        if name!='1m': feat['rsi']=compute_rsi(c); feat['bvc']=compute_bvc(c,v)
        if name=='5m':
            dates=np.array([t.date() for t in df.index])
            vw,vd=compute_vwap(o,h,l,c,v,dates)
            feat['vwap']=vw; feat['vwap_dist']=vd
            feat['vol_ratio']=compute_volume_ratio(v)
            feat['vwap_slope']=compute_vwap_slope(vd, lb=5)
            feat['spread_pct']=compute_spread_pct(h, l, c)
            feat['rsi_slope']=compute_rsi_slope(feat['rsi'].values, lb=5)
            feat['gap_pct']=compute_gap_pct(c, dates)
        features[name]=feat
    f5m = features['5m']; f1m = features['1m']
    f5m['bullish_1m'] = compute_bullish_1m_count(f1m['close'], f1m['open'], f5m.index, lookback=5)
    print(f"Done in {time.time()-t0:.1f}s"); _flush()
    return features

def precompute_all(features, f5m):
    print("Pre-computing..."); _flush()
    t0=time.time(); n=len(f5m)
    bar_dates=np.array([t.date() for t in f5m.index])
    daily=features['daily']
    dcp=daily['chan_pos'].values; dslope=daily['chan_slope'].values
    ddates=np.array([idx.date() if hasattr(idx,'date') else idx for idx in daily.index])
    dcp_arr=np.full(n,np.nan); dslope_arr=np.full(n,np.nan)
    ud=sorted(set(bar_dates)); d2d={}
    for d in ud:
        best=-1
        for k in range(len(ddates)):
            if ddates[k]<d: best=k
            elif ddates[k]>=d: break
        if best>=0: d2d[d]=best
    for i in range(n):
        d=bar_dates[i]
        if d in d2d:
            k=d2d[d]; dcp_arr[i]=dcp[k]; dslope_arr[i]=dslope[k]
    arrays={
        '1h_cp':np.full(n,np.nan),'1h_slope':np.full(n,np.nan),
        '4h_cp':np.full(n,np.nan),'4h_slope':np.full(n,np.nan),
        '15m_cp':np.full(n,np.nan),'30m_cp':np.full(n,np.nan),
    }
    cfgs=[
        (features['1h'],{'1h_cp':'chan_pos','1h_slope':'chan_slope'}),
        (features['4h'],{'4h_cp':'chan_pos','4h_slope':'chan_slope'}),
        (features['15m'],{'15m_cp':'chan_pos'}),
        (features['30m'],{'30m_cp':'chan_pos'}),
    ]
    for tf,cm in cfgs:
        ti=tf.index.values; ca={k:tf[v].values for k,v in cm.items()}; j=0
        for i in range(n):
            t=f5m.index[i]
            while j<len(ti)-1 and ti[j+1]<=t: j+=1
            if j<len(ti) and ti[j]<=t:
                for k in cm: arrays[k][i]=ca[k][j]
    print(f"  Done in {time.time()-t0:.1f}s"); _flush()
    return (dcp_arr,dslope_arr), arrays

# ---- SIGNALS ----

def sig_vwap(i, f5m, ctx, params):
    cp5=f5m['chan_pos'].iloc[i]; vd=f5m['vwap_dist'].iloc[i]
    if np.isnan(cp5) or np.isnan(vd): return None
    dcp=ctx['daily_cp']; hcp=ctx['1h_cp'][i]
    if np.isnan(dcp) or np.isnan(hcp): return None
    if vd>params.get('vwap_thresh',-0.30): return None
    if dcp<params.get('d_min',0.20): return None
    if hcp<params.get('h1_min',0.15): return None
    if cp5>params.get('f5_thresh',0.25): return None
    min_vr = params.get('min_vol_ratio', 0.0)
    if min_vr > 0:
        vr = f5m['vol_ratio'].iloc[i] if 'vol_ratio' in f5m.columns else np.nan
        if not np.isnan(vr) and vr < min_vr: return None
    s=params.get('stop',0.008); t=params.get('tp',0.020)
    conf=0.55+min(abs(vd)*0.05,0.15)+0.10*(1.0-cp5)
    if min_vr > 0:
        vr = f5m['vol_ratio'].iloc[i] if 'vol_ratio' in f5m.columns else 1.0
        if not np.isnan(vr) and vr > 1.5: conf += 0.05
    return ('LONG',min(conf,0.95),s,t)

def sig_div(i, f5m, ctx, params):
    cp5=f5m['chan_pos'].iloc[i]
    if np.isnan(cp5): return None
    dcp=ctx['daily_cp']; hcp=ctx['1h_cp'][i]; h4cp=ctx['4h_cp'][i]
    if np.isnan(dcp) or np.isnan(hcp) or np.isnan(h4cp): return None
    ha=dcp*0.35+h4cp*0.35+hcp*0.30; div=ha-cp5
    if div<params.get('div_thresh',0.35): return None
    if cp5>params.get('f5_thresh',0.30): return None
    min_vr = params.get('min_vol_ratio', 0.0)
    if min_vr > 0:
        vr = f5m['vol_ratio'].iloc[i] if 'vol_ratio' in f5m.columns else np.nan
        if not np.isnan(vr) and vr < min_vr: return None
    vd=f5m['vwap_dist'].iloc[i]; vb=0.0
    if not np.isnan(vd) and vd<0: vb=min(abs(vd)*0.02,0.10)
    s=params.get('stop',0.008); t=params.get('tp',0.020)
    conf=0.55+0.25*min(div,0.7)+0.10*(1.0-cp5)+vb
    return ('LONG',min(conf,0.95),s,t)

def sig_union(i, f5m, ctx, params):
    best=None; bc=0
    for fn in [sig_vwap, sig_div]:
        r=fn(i,f5m,ctx,params)
        if r and r[1]>bc: best=r; bc=r[1]
    return best

def _enhance_conf(conf, i, f5m, ctx):
    vs = f5m['vwap_slope'].iloc[i] if 'vwap_slope' in f5m.columns else np.nan
    if not np.isnan(vs) and vs < -0.05: conf += 0.02
    bc = f5m['bullish_1m'].iloc[i] if 'bullish_1m' in f5m.columns else np.nan
    if not np.isnan(bc) and bc >= 4: conf += 0.02
    gap = f5m['gap_pct'].iloc[i] if 'gap_pct' in f5m.columns else np.nan
    if not np.isnan(gap) and gap < -0.5: conf += 0.02
    rsi_sl = f5m['rsi_slope'].iloc[i] if 'rsi_slope' in f5m.columns else np.nan
    if not np.isnan(rsi_sl) and rsi_sl > 0.5: conf += 0.02
    ds = ctx.get('daily_slope', np.nan)
    hs = ctx['1h_slope'][i] if '1h_slope' in ctx else np.nan
    h4s = ctx['4h_slope'][i] if '4h_slope' in ctx else np.nan
    up = 0
    if not np.isnan(ds) and ds > 0: up += 1
    if not np.isnan(hs) and hs > 0: up += 1
    if not np.isnan(h4s) and h4s > 0: up += 1
    if up == 3: conf += 0.02
    sp = f5m['spread_pct'].iloc[i] if 'spread_pct' in f5m.columns else np.nan
    if not np.isnan(sp) and sp < 0.3: conf += 0.01
    return min(conf, 0.95)

def sig_stacked(i, f5m, ctx, params):
    rv = sig_vwap(i, f5m, ctx, params)
    rd = sig_div(i, f5m, ctx, params)
    if rv is None or rd is None: return None
    conf = min(0.95, max(rv[1], rd[1]) + 0.05)
    return ('LONG', conf, rv[2], rv[3])

def sig_union_enh(i, f5m, ctx, params):
    base = sig_union(i, f5m, ctx, params)
    if base is None: return None
    conf = _enhance_conf(base[1], i, f5m, ctx)
    return ('LONG', conf, base[2], base[3])

def sig_stacked_enh(i, f5m, ctx, params):
    base = sig_stacked(i, f5m, ctx, params)
    if base is None: return None
    conf = _enhance_conf(base[1], i, f5m, ctx)
    return ('LONG', conf, base[2], base[3])

# ---- SIMULATORS ----

def simulate_fixed(f5m, signal_fn, name, params, precomp,
                   tb=0.006, tp=6, cd=0, mtd=20,
                   base_capital=20_000.0, max_capital=100_000.0,
                   conf_size=True,
                   tod_start=dt.time(13,0), tod_end=dt.time(15,25),
                   f1m=None, eval_interval_1m=2,
                   ah_rules=False, ah_loss_limit=250.0):
    """Confidence-scaled sizing capped at max_capital.

    New params:
      f1m:              1-min DataFrame for high-res exit checking (signals still on 5-min)
      eval_interval_1m: bars between exit checks on 1-min data (default 2)
      ah_rules:         enable AH gated opens + unlimited closes + loss limit
      ah_loss_limit:    max loss per AH trade before force close ($250 default)
    """
    (dcp_arr,dslope_arr), htf = precomp
    o_arr=f5m['open'].values; h_arr=f5m['high'].values
    l_arr=f5m['low'].values; c_arr=f5m['close'].values
    times=f5m.index; tod_arr=np.array([t.time() for t in times])
    n=len(f5m); trades=[]
    ctx={'daily_cp':0.0,'daily_slope':0.0,**htf}
    in_trade=False; ep=et=None; conf=sp=tpp=bp=0.0; hb=cr=tt=0; cd_=None; ps=None
    tes=dt.time(9,35); tee=dt.time(15,30); tfe=dt.time(15,50)
    timeout_bars = 78  # 5-min bars

    # AH state
    ah_tracker = AHStateTracker() if ah_rules else None
    ah_entry = False  # was this trade opened during AH?

    # 1-min exit support: build mapping from 5-min bar index -> 1-min bar slice
    f1m_h = f1m_l = f1m_c = f1m_times = None
    bar_to_1m = None
    if f1m is not None:
        f1m_h = f1m['high'].values; f1m_l = f1m['low'].values
        f1m_c = f1m['close'].values; f1m_times = f1m.index
        # Map each 5-min bar to range of 1-min bar indices
        bar_to_1m = {}
        j = 0
        for i in range(n):
            t5 = times[i]
            # Find all 1-min bars from this 5-min bar start to next 5-min bar start
            t5_end = times[i+1] if i+1 < n else t5 + pd.Timedelta(minutes=5)
            start_j = j
            while j < len(f1m_times) and f1m_times[j] < t5_end:
                j += 1
            bar_to_1m[i] = (start_j, j)  # half-open range [start_j, j)
        # Adjust timeout for 1-min resolution: 78 5-min bars = 390 1-min bars
        timeout_bars = 78 * 5

    if ah_rules:
        # Expand entry/exit windows for AH
        tes = dt.time(4, 0)    # pre-market open
        tee = dt.time(19, 55)  # post-market near close
        tfe = dt.time(19, 55)  # force EOD at AH end

    for i in range(n):
        bt=times[i]; bd=bt.date(); btod=tod_arr[i]; o,h,l,c=o_arr[i],h_arr[i],l_arr[i],c_arr[i]
        if bd!=cd_:
            cd_=bd; tt=0
            if ah_tracker: ah_tracker.reset_if_new_day(bd)
        if ps is not None and not in_trade:
            sc,ss,st=ps; ps=None
            if btod>=tes and btod<=tee:
                # AH gating: check if we can open in extended hours
                if ah_rules and is_extended_hours(btod):
                    if not ah_tracker.can_open_ah():
                        continue
                    ah_tracker.record_ah_open()
                    ah_entry = True
                else:
                    ah_entry = False
                ep=o*(1+SLIPPAGE_PCT); et=bt; conf=sc; in_trade=True; hb=0; tt+=1
                sp=ep*(1-ss); tpp=ep*(1+st); bp=ep
        if in_trade:
            xp=xr=None

            if f1m is not None and i in bar_to_1m:
                # --- 1-min exit checking ---
                s1, e1 = bar_to_1m[i]
                for k in range(s1, e1, eval_interval_1m):
                    hb += 1
                    # Window high/low over eval interval
                    wend = min(k + eval_interval_1m, e1)
                    wh = f1m_h[k:wend].max(); wl = f1m_l[k:wend].min()
                    wc = f1m_c[min(wend-1, len(f1m_c)-1)]
                    bp = max(bp, wh)
                    trail = tb*(1.0-conf)**tp; ts_ = bp*(1-trail)
                    if ts_ > sp: sp = ts_

                    # AH loss limit check
                    if ah_rules and ah_entry and is_extended_hours(f1m_times[k].time()):
                        unrealized = (wl - ep) * max(1, int(max_capital / ep))
                        if AHStateTracker.check_ah_loss_limit(unrealized, ah_loss_limit):
                            xp = wl; xr = 'ah_loss_limit'; break

                    if wl <= sp: xp = max(sp, wl); xr = 'stop' if sp < ep else 'trail'; break
                    elif wh >= tpp: xp = tpp; xr = 'tp'; break
                    elif hb >= timeout_bars: xp = wc; xr = 'timeout'; break

                    # EOD check on 1-min bar time
                    bar_1m_tod = f1m_times[k].time()
                    if bar_1m_tod >= tfe: xp = wc; xr = 'eod'; break
            else:
                # --- Original 5-min exit checking ---
                hb += 1; bp = max(bp, h)
                trail = tb*(1.0-conf)**tp; ts_ = bp*(1-trail)
                if ts_ > sp: sp = ts_

                # AH loss limit check
                if ah_rules and ah_entry and is_extended_hours(btod):
                    unrealized = (l - ep) * max(1, int(max_capital / ep))
                    if AHStateTracker.check_ah_loss_limit(unrealized, ah_loss_limit):
                        xp = l; xr = 'ah_loss_limit'

                if xp is None:
                    if l <= sp: xp = max(sp, l); xr = 'stop' if sp < ep else 'trail'
                    elif h >= tpp: xp = tpp; xr = 'tp'
                    elif hb >= 78: xp = c; xr = 'timeout'
                    elif btod >= tfe: xp = c; xr = 'eod'

            if xp is not None:
                xa=xp*(1-SLIPPAGE_PCT)
                if conf_size:
                    cap = base_capital + (max_capital - base_capital) * conf
                    sh = max(1, int(cap * conf / ep))
                else:
                    sh = max(1, int(max_capital / ep))
                pnl=(xa-ep)*sh-COMM_PER_SHARE*sh*2
                trades.append((et,bt,ep,xa,conf,sh,pnl,hb,xr,name))
                if ah_rules and ah_entry:
                    ah_tracker.record_ah_close(pnl)
                in_trade=False; cr=cd; ah_entry=False
        if cr>0: cr-=1; continue
        if in_trade or tt>=mtd: continue
        if not ah_rules:
            # Original RTH-only entry window
            if btod<tod_start or btod>tod_end: continue
        else:
            # With AH rules: allow entries in extended hours too
            if not (is_rth(btod) or is_extended_hours(btod)): continue
            if is_rth(btod) and (btod<tod_start or btod>tod_end): continue
        ctx['daily_cp']=dcp_arr[i]; ctx['daily_slope']=dslope_arr[i]
        result=signal_fn(i,f5m,ctx,params)
        if result is not None:
            _,co,s,t=result; ps=(co,s,t)
    return trades

def simulate_compound(f5m, signal_fn, name, params, precomp,
                      tb=0.006, tp=6, cd=0, mtd=20,
                      initial_capital=100_000.0,
                      tod_start=dt.time(13,0), tod_end=dt.time(15,25)):
    """Full compound: deploy ALL equity on every trade. Start $100K, reinvest everything."""
    (dcp_arr,dslope_arr), htf = precomp
    o_arr=f5m['open'].values; h_arr=f5m['high'].values
    l_arr=f5m['low'].values; c_arr=f5m['close'].values
    times=f5m.index; tod_arr=np.array([t.time() for t in times])
    n=len(f5m); trades=[]
    ctx={'daily_cp':0.0,'daily_slope':0.0,**htf}
    in_trade=False; ep=et=None; conf=sp=tpp=bp=0.0; hb=cr=tt=0; cd_=None; ps=None
    equity = initial_capital
    tes=dt.time(9,35); tee=dt.time(15,30); tfe=dt.time(15,50)
    for i in range(n):
        bt=times[i]; bd=bt.date(); btod=tod_arr[i]; o,h,l,c=o_arr[i],h_arr[i],l_arr[i],c_arr[i]
        if bd!=cd_: cd_=bd; tt=0
        if ps is not None and not in_trade:
            sc,ss,st=ps; ps=None
            if btod>=tes and btod<=tee:
                ep=o*(1+SLIPPAGE_PCT); et=bt; conf=sc; in_trade=True; hb=0; tt+=1
                sp=ep*(1-ss); tpp=ep*(1+st); bp=ep
        if in_trade:
            hb+=1; xp=xr=None; bp=max(bp,h)
            trail=tb*(1.0-conf)**tp; ts_=bp*(1-trail)
            if ts_>sp: sp=ts_
            if l<=sp: xp=max(sp,l); xr='stop' if sp<ep else 'trail'
            elif h>=tpp: xp=tpp; xr='tp'
            elif hb>=78: xp=c; xr='timeout'
            elif btod>=tfe: xp=c; xr='eod'
            if xp is not None:
                xa=xp*(1-SLIPPAGE_PCT)
                # Full equity deployment - use ALL of current equity
                sh=max(1,int(equity/ep))
                pnl=(xa-ep)*sh-COMM_PER_SHARE*sh*2
                equity += pnl
                trades.append((et,bt,ep,xa,conf,sh,pnl,hb,xr,name,equity))
                in_trade=False; cr=cd
        if cr>0: cr-=1; continue
        if in_trade or tt>=mtd: continue
        if btod<tod_start or btod>tod_end: continue
        if equity < 10_000: continue
        ctx['daily_cp']=dcp_arr[i]; ctx['daily_slope']=dslope_arr[i]
        result=signal_fn(i,f5m,ctx,params)
        if result is not None:
            _,co,s,t=result; ps=(co,s,t)
    return trades, equity

def main():
    parser = argparse.ArgumentParser(description='V14b Intraday Backtest')
    parser.add_argument('--tsla', type=str, default=None, help='Path to TSLAMin.txt')
    parser.add_argument('--1min', dest='use_1min', action='store_true',
                        help='Use 1-min bars for exit checking (signals still on 5-min)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='Bars between exit checks (default 2 with --1min)')
    parser.add_argument('--flat-sizing', action='store_true',
                        help='Flat $100K per trade (conf_size=False)')
    parser.add_argument('--ah-rules', action='store_true',
                        help='Enable AH gated opens + unlimited closes')
    parser.add_argument('--ah-loss-limit', type=float, default=250.0,
                        help='Max loss per AH trade (default $250)')
    args = parser.parse_args()

    # Set eval_interval default based on --1min
    eval_interval_1m = args.eval_interval if args.eval_interval is not None else (2 if args.use_1min else 1)

    print("="*70)
    print("V14b: ALL CONFIGS HEAD-TO-HEAD @ FLAT $100K PER TRADE")
    if args.use_1min: print("  >> 1-MIN EXIT CHECKING (eval_interval=%d)" % eval_interval_1m)
    if args.flat_sizing: print("  >> FLAT $100K SIZING")
    if args.ah_rules: print("  >> AH RULES (loss limit=$%.0f)" % args.ah_loss_limit)
    print("="*70); _flush()

    # Load data: keep AH bars if ah_rules enabled
    rth_only = not args.ah_rules
    df1m = load_1min(path=args.tsla, rth_only=rth_only)

    # Features are always built on RTH data for signal generation
    df1m_rth = df1m
    if not rth_only:
        times = df1m.index.time
        df1m_rth = df1m[(times >= MKT_OPEN) & (times < MKT_CLOSE)].copy()

    features = build_features(df1m_rth)
    f5m = features['5m']
    precomp = precompute_all(features, f5m)

    # Prepare 1-min data for exit checking (may include AH bars)
    f1m_for_exits = df1m if args.use_1min else None

    # Wider params (used by most configs)
    pw = {'stop':0.008, 'tp':0.020, 'd_min':0.20, 'h1_min':0.15, 'f5_thresh':0.35,
          'div_thresh':0.20, 'vwap_thresh':-0.10, 'min_vol_ratio':0.8}
    # Default (tighter) params
    pd_ = {'stop':0.008, 'tp':0.020, 'd_min':0.20, 'h1_min':0.15, 'f5_thresh':0.25,
           'div_thresh':0.35, 'vwap_thresh':-0.30, 'min_vol_ratio':0.0}

    # All configs: confidence-scaled with $100K max
    # conf_size=True: cap = base + (max-base)*conf, shares = cap*conf/price
    # conf_size=False: cap = max_capital flat, shares = cap/price
    configs = [
        # (name, signal_fn, params, mtd, tod_start, tod_end, base_cap, max_cap, conf_size)
        ("A: PM eUnion m20 conf $20K-$100K",  sig_union_enh,   pw,  20, dt.time(13,0), dt.time(15,25), 20_000, 100_000, True),
        ("B: PM eUnion m30 conf $20K-$100K",  sig_union_enh,   pw,  30, dt.time(13,0), dt.time(15,25), 20_000, 100_000, True),
        ("C: PM eUnion m20 conf $50K-$100K",  sig_union_enh,   pw,  20, dt.time(13,0), dt.time(15,25), 50_000, 100_000, True),
        ("D: PM eStack wider conf $20K-$100K",sig_stacked_enh, pw,  20, dt.time(13,0), dt.time(15,25), 20_000, 100_000, True),
        ("E: PM eStack def conf $20K-$100K",  sig_stacked_enh, pd_, 20, dt.time(13,0), dt.time(15,25), 20_000, 100_000, True),
        ("F: FD eUnion m20 conf $20K-$100K",  sig_union_enh,   pw,  20, dt.time(9,30), dt.time(15,25), 20_000, 100_000, True),
        ("G: FD eUnion m30 conf $20K-$100K",  sig_union_enh,   pw,  30, dt.time(9,30), dt.time(15,25), 20_000, 100_000, True),
        ("H: PM eUnion m20 FLAT $100K",       sig_union_enh,   pw,  20, dt.time(13,0), dt.time(15,25), 100_000, 100_000, False),
        ("I: FD eUnion m30 FLAT $100K",       sig_union_enh,   pw,  30, dt.time(9,30), dt.time(15,25), 100_000, 100_000, False),
        ("J: PM eUnion m20 conf $0-$100K",    sig_union_enh,   pw,  20, dt.time(13,0), dt.time(15,25), 0, 100_000, True),
    ]

    print("\n" + "="*70)
    print("ALL CONFIGS - CONFIDENCE SCALED, $100K MAX")
    print("Conf sizing: cap = base + (max-base)*conf, shares = cap*conf/price")
    print("="*70); _flush()

    # Summary table
    print(f"\n{'#':>2} {'Config':<40} {'Trades':>7} {'WR':>7} {'Total PnL':>12} {'BL':>8} {'AvgSh':>7} {'2026 OOS':>20}")
    print("-" * 110)

    all_results = []
    for name, sig_fn, params, mtd, ts, te, bc, mc, cs in configs:
        # Override conf_size if --flat-sizing
        effective_cs = False if args.flat_sizing else cs
        trades = simulate_fixed(f5m, sig_fn, name, params, precomp, mtd=mtd,
                                base_capital=float(bc), max_capital=float(mc),
                                conf_size=effective_cs, tod_start=ts, tod_end=te,
                                f1m=f1m_for_exits, eval_interval_1m=eval_interval_1m,
                                ah_rules=args.ah_rules, ah_loss_limit=args.ah_loss_limit)
        if not trades:
            print(f"   {name:<40} {'0':>7}"); _flush()
            continue

        n = len(trades); pnls = [t[6] for t in trades]
        w = sum(1 for p in pnls if p > 0); wr = w/n*100
        total = sum(pnls); bl = min(pnls)
        avg_sh = np.mean([t[5] for t in trades])

        # 2026 OOS
        oos = [t for t in trades if t[0].year >= 2026]
        if oos:
            on = len(oos); ow = sum(1 for t in oos if t[6] > 0)
            owr = ow/on*100; op = sum(t[6] for t in oos)
            oos_s = f"{on}t {owr:.0f}% ${op:+,.0f}"
        else:
            oos_s = "n/a"

        ltr = name[0:2].strip().rstrip(':')
        print(f"{ltr:>2} {name:<40} {n:>7} {wr:>6.1f}% ${total:>+10,.0f} ${bl:>+6,.0f} {avg_sh:>7.0f} {oos_s:>20}")
        _flush()
        all_results.append((name, trades, total))

    # Year-by-year for top 3
    all_results.sort(key=lambda x: -x[2])
    print("\n" + "="*70)
    print("YEAR-BY-YEAR FOR TOP 3 CONFIGS")
    print("="*70); _flush()

    for rank, (name, trades, total) in enumerate(all_results[:3], 1):
        print(f"\n--- #{rank}: {name} (${total:+,.0f} total) ---")
        years = sorted(set(t[0].year for t in trades))
        print(f"{'Year':>6} {'Trades':>7} {'Wins':>6} {'WR':>7} {'PnL':>12} {'BL':>8} {'Cum':>12}")
        print("-" * 65)
        cum = 0
        for yr in years:
            yt = [t for t in trades if t[0].year == yr]
            if not yt: continue
            yw = sum(1 for t in yt if t[6] > 0)
            yp = sum(t[6] for t in yt); ybl = min(t[6] for t in yt)
            ywr = yw/len(yt)*100; cum += yp
            print(f"{yr:>6} {len(yt):>7} {yw:>6} {ywr:>6.1f}% ${yp:>+10,.0f} ${ybl:>+6,.0f} ${cum:>+10,.0f}")
        _flush()

    print("\n" + "="*70)
    print("DONE")
    print("="*70); _flush()

if __name__ == '__main__':
    main()
