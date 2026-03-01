#!/usr/bin/env python3
"""V11: Novel / unorthodox signal enhancements.
V10 champion: Stack N vr>0.8 m=10 $75K-$300K = $3.17M at 86.9% WR.
Testing ideas NOT yet explored:
1. VWAP slope (distance acceleration) - trade only when VWAP dist is STILL FALLING (not recovering)
2. Spread compression - narrow 5m range = coiled spring, tighter trail works better
3. 1m bullish pressure - count bullish 1m bars in last 5 bars before signal
4. Gap-relative signal - signal stronger when price gapped down from prior close
5. Intraday RSI divergence - 5m RSI rising while price still falling (hidden bullish div)
6. Cross-TF channel slope agreement - 1h/4h/daily channels all sloping UP
7. Volume profile (cumulative intraday volume vs historical) - trade during volume surges
8. Bar-to-bar momentum (close-to-close returns) autocorrelation - trade when mean-reverting regime
9. Optimal trailing stop exponent sweep - is power=4 really best? test 2-8
10. Multi-signal confidence weighting - combine signals with weighted average
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime as dt

def _flush(): sys.stdout.flush()

CAPITAL = 100_000.0
SLIPPAGE_PCT = 0.0002
COMM_PER_SHARE = 0.005
TRAIN_END = pd.Timestamp('2021-12-31')
TEST_END  = pd.Timestamp('2025-12-31')
MKT_OPEN  = dt.time(9, 30)
MKT_CLOSE = dt.time(16, 0)

def load_1min(path=None):
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
    times = df.index.time
    df = df[(times >= MKT_OPEN) & (times < MKT_CLOSE)].copy()
    print(f"  Loaded {len(df):,} bars in {time.time()-t0:.1f}s"); _flush()
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
    """Return normalized channel slope (regression slope / price)."""
    n = len(close_arr); close = close_arr.astype(np.float64); w = window
    sx = w*(w-1)/2.0; sx2 = (w-1)*w*(2*w-1)/6.0; denom = w*sx2-sx**2
    cy = np.cumsum(close); sy = np.full(n,np.nan); sy[w-1]=cy[w-1]
    if n>w: sy[w:]=cy[w:]-cy[:n-w]
    idx=np.arange(n,dtype=np.float64); cwy=np.cumsum(idx*close)
    sxy=np.full(n,np.nan); sxy[w-1]=cwy[w-1]
    if n>w:
        si=np.arange(w,n,dtype=np.float64)-w+1
        sxy[w:]=(cwy[w:]-cwy[:n-w])-si*sy[w:]
    slope=(w*sxy-sx*sy)/denom
    # normalize by price level
    norm_slope = np.full(n, np.nan)
    valid = close > 0
    norm_slope[valid] = slope[valid] / close[valid]
    return norm_slope

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
    """Slope of VWAP distance over last lb bars. Negative = getting more extreme, positive = recovering."""
    n = len(vd); vs = np.full(n, np.nan)
    for i in range(lb, n):
        seg = vd[i-lb+1:i+1]
        if np.any(np.isnan(seg)): continue
        # simple linear regression slope
        x = np.arange(lb, dtype=np.float64)
        mx = x.mean(); my = seg.mean()
        vs[i] = np.sum((x-mx)*(seg-my)) / np.sum((x-mx)**2)
    return vs

def compute_spread_pct(h, l, c):
    """High-Low range as % of close."""
    sp = np.full(len(c), np.nan)
    valid = c > 0
    sp[valid] = (h[valid] - l[valid]) / c[valid] * 100.0
    return sp

def compute_gap_pct(c, dates):
    """Overnight gap: today open vs yesterday close, as % of close."""
    n = len(c)
    gap = np.full(n, np.nan)
    prev_close = None; prev_date = None
    for i in range(n):
        d = dates[i]
        if d != prev_date:
            if prev_close is not None and prev_close > 0:
                # gap computed at first bar of new day
                gap[i] = (c[i] - prev_close) / prev_close * 100.0  # uses open-ish first bar close
            prev_date = d
        prev_close = c[i]
    # Forward fill gap for the day
    cur_gap = np.nan
    for i in range(n):
        d = dates[i]
        if not np.isnan(gap[i]): cur_gap = gap[i]
        gap[i] = cur_gap
    return gap

def compute_rsi_slope(rsi, lb=5):
    """Slope of RSI over last lb bars."""
    n = len(rsi); rs = np.full(n, np.nan)
    for i in range(lb, n):
        seg = rsi[i-lb+1:i+1]
        if np.any(np.isnan(seg)): continue
        x = np.arange(lb, dtype=np.float64)
        mx = x.mean(); my = seg.mean()
        rs[i] = np.sum((x-mx)*(seg-my)) / np.sum((x-mx)**2)
    return rs

def compute_bullish_1m_count(f1m_close, f1m_open, f5m_index, lookback=5):
    """Count bullish 1m bars (close>open) in the last `lookback` 1m bars before each 5m bar."""
    n5 = len(f5m_index)
    result = np.full(n5, np.nan)
    ti = f1m_close.index.values
    c1 = f1m_close.values
    o1 = f1m_open.values
    j = 0
    for i in range(n5):
        t = f5m_index[i]
        while j < len(ti)-1 and ti[j+1] <= t: j += 1
        if j >= lookback:
            bullish = 0
            for k in range(j-lookback+1, j+1):
                if c1[k] > o1[k]: bullish += 1
            result[i] = bullish
    return result

def compute_returns_autocorr(c, lb=20):
    """Rolling autocorrelation of bar-to-bar returns. Negative = mean-reverting."""
    n = len(c)
    ac = np.full(n, np.nan)
    rets = np.diff(c) / c[:-1] * 100.0
    rets = np.concatenate([[0], rets])
    for i in range(lb+1, n):
        r = rets[i-lb:i]
        r1 = r[:-1]; r2 = r[1:]
        if np.std(r1) < 1e-10 or np.std(r2) < 1e-10: continue
        ac[i] = np.corrcoef(r1, r2)[0,1]
    return ac

def build_features(df1m):
    t0=time.time(); print("Resampling..."); _flush()
    tfs={}
    for rule,name in [('3min','3m'),('5min','5m'),('15min','15m'),('30min','30m'),('1h','1h'),('4h','4h')]:
        tfs[name]=resample_ohlcv(df1m,rule)
    tfs['daily']=resample_ohlcv(df1m,'1D'); tfs['1m']=df1m.copy()
    for name,df in tfs.items(): print(f"  {name}: {len(df):,}"); _flush()
    features={}
    wins={'1m':60,'3m':60,'5m':60,'15m':40,'30m':30,'1h':24,'4h':20,'daily':40}
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
            feat['ret_autocorr']=compute_returns_autocorr(c, lb=20)
        features[name]=feat
    # 1m bullish count aligned to 5m
    f5m = features['5m']
    f1m = features['1m']
    f5m['bullish_1m'] = compute_bullish_1m_count(f1m['close'], f1m['open'], f5m.index, lookback=5)
    print(f"Done in {time.time()-t0:.1f}s"); _flush()
    return features

def precompute_all(features, f5m):
    print("Pre-computing..."); _flush()
    t0=time.time(); n=len(f5m)
    bar_dates=np.array([t.date() for t in f5m.index])
    daily=features['daily']
    dcp=daily['chan_pos'].values; drsi=daily['rsi'].values; dbvc=daily['bvc'].values
    dslope=daily['chan_slope'].values
    ddates=np.array([idx.date() if hasattr(idx,'date') else idx for idx in daily.index])
    dcp_arr=np.full(n,np.nan); drsi_arr=np.full(n,np.nan); dbvc_arr=np.full(n,np.nan)
    dslope_arr=np.full(n,np.nan)
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
            k=d2d[d]; dcp_arr[i]=dcp[k]; drsi_arr[i]=drsi[k]; dbvc_arr[i]=dbvc[k]
            dslope_arr[i]=dslope[k]
    arrays={
        '1h_cp':np.full(n,np.nan),'1h_rsi':np.full(n,np.nan),'1h_bvc':np.full(n,np.nan),
        '1h_slope':np.full(n,np.nan),
        '4h_cp':np.full(n,np.nan),'4h_slope':np.full(n,np.nan),
        '15m_cp':np.full(n,np.nan),
        '30m_cp':np.full(n,np.nan),
        '3m_cp':np.full(n,np.nan),'3m_rsi':np.full(n,np.nan),
    }
    cfgs=[
        (features['1h'],{'1h_cp':'chan_pos','1h_rsi':'rsi','1h_bvc':'bvc','1h_slope':'chan_slope'}),
        (features['4h'],{'4h_cp':'chan_pos','4h_slope':'chan_slope'}),
        (features['15m'],{'15m_cp':'chan_pos'}),
        (features['30m'],{'30m_cp':'chan_pos'}),
        (features['3m'],{'3m_cp':'chan_pos','3m_rsi':'rsi'}),
    ]
    for tf,cm in cfgs:
        ti=tf.index.values; ca={k:tf[v].values for k,v in cm.items()}; j=0
        for i in range(n):
            t=f5m.index[i]
            while j<len(ti)-1 and ti[j+1]<=t: j+=1
            if j<len(ti) and ti[j]<=t:
                for k in cm: arrays[k][i]=ca[k][j]
    print(f"  Done in {time.time()-t0:.1f}s"); _flush()
    return (dcp_arr,drsi_arr,dbvc_arr,dslope_arr), arrays

def simulate(f5m, signal_fn, name, params, precomp,
             tb=0.010, tp=4, cd=0, mtd=10, conf_size=False,
             base_capital=75_000.0, max_capital=300_000.0,
             tod_start=dt.time(9,40), tod_end=dt.time(15,25)):
    (dcp_arr,_,_,dslope_arr), htf = precomp
    o_arr=f5m['open'].values; h_arr=f5m['high'].values
    l_arr=f5m['low'].values; c_arr=f5m['close'].values
    times=f5m.index; tod_arr=np.array([t.time() for t in times])
    n=len(f5m); trades=[]
    ctx={'daily_cp':0.0,'daily_slope':0.0,**htf}
    in_trade=False; ep=et=None; conf=sp=tpp=bp=0.0; hb=cr=tt=0; cd_=None; ps=None
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
                cap = (base_capital + (max_capital - base_capital) * conf) if conf_size else CAPITAL
                sh=max(1,int(cap*conf/ep))
                pnl=(xa-ep)*sh-COMM_PER_SHARE*sh*2
                trades.append((et,bt,ep,xa,conf,sh,pnl,hb,xr,name))
                in_trade=False; cr=cd
        if cr>0: cr-=1; continue
        if in_trade or tt>=mtd: continue
        if btod<tod_start or btod>tod_end: continue
        ctx['daily_cp']=dcp_arr[i]; ctx['daily_slope']=dslope_arr[i]
        result=signal_fn(i,f5m,ctx,params)
        if result is not None:
            _,co,s,t=result; ps=(co,s,t)
    return trades

def report(trades, label, show_years=True):
    if not trades: print(f"  {label}: 0 trades"); _flush(); return
    n=len(trades); pnls=[t[6] for t in trades]
    w=sum(1 for p in pnls if p>0); wr=w/n*100; total=sum(pnls)
    bl=min(pnls); bw=max(pnls)
    aw=np.mean([p for p in pnls if p>0]) if w else 0
    al=np.mean([p for p in pnls if p<=0]) if n-w else 0
    train=[t for t in trades if t[0]<=TRAIN_END]
    test=[t for t in trades if TRAIN_END<t[0]<=TEST_END]
    oos=[t for t in trades if t[0]>TEST_END]
    tw=sum(1 for t in train if t[6]>0); tp_=sum(t[6] for t in train)
    tsw=sum(1 for t in test if t[6]>0); tsp=sum(t[6] for t in test)
    ow=sum(1 for t in oos if t[6]>0); op=sum(t[6] for t in oos)
    bal = tsp/tp_*100 if tp_>0 else 0
    print(f"  {label}: {n}t {w}W/{n-w}L ({wr:.1f}%WR) ${total:+,.0f} BW=${bw:+,.0f} BL=${bl:+,.0f} aW=${aw:+,.0f} aL=${al:+,.0f}")
    print(f"    TRAIN:{len(train)}t {tw}W ${tp_:+,.0f} | TEST:{len(test)}t {tsw}W ${tsp:+,.0f} | BAL={bal:.0f}% | OOS:{len(oos)}t {ow}W ${op:+,.0f}")
    if show_years:
        years=sorted(set(t[0].year for t in trades)); cum=0
        for yr in years:
            yt=[t for t in trades if t[0].year==yr]
            yw=sum(1 for t in yt if t[6]>0); yp=sum(t[6] for t in yt); ybl=min(t[6] for t in yt)
            cum+=yp; ywr=yw/len(yt)*100
            print(f"    {yr}: {len(yt):3d}t {ywr:5.1f}% ${yp:+8,.0f} BL=${ybl:+,.0f} cum=${cum:+,.0f}")
    _flush()

def rs(trades, label):
    """Report short."""
    if not trades: print(f"  {label}: 0 trades"); _flush(); return
    n=len(trades); pnls=[t[6] for t in trades]
    w=sum(1 for p in pnls if p>0); wr=w/n*100; total=sum(pnls); bl=min(pnls)
    train=[t for t in trades if t[0]<=TRAIN_END]
    test=[t for t in trades if TRAIN_END<t[0]<=TEST_END]
    oos=[t for t in trades if t[0]>TEST_END]
    tp_=sum(t[6] for t in train); tsp=sum(t[6] for t in test)
    bal = tsp/tp_*100 if tp_>0 else 0
    on=len(oos); ow=sum(1 for t in oos if t[6]>0); op=sum(t[6] for t in oos)
    print(f"  {label}: {n:5d}t {wr:5.1f}% ${total:+10,.0f} BL=${bl:+,.0f} BAL={bal:.0f}% OOS:{on}t {ow}W ${op:+,.0f}")
    _flush()

# ─── SIGNALS ───────────────────────────────────────────────

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

def sig_stacked(i, f5m, ctx, params):
    rv = sig_vwap(i, f5m, ctx, params)
    rd = sig_div(i, f5m, ctx, params)
    if rv is None or rd is None: return None
    conf = min(0.95, max(rv[1], rd[1]) + 0.05)
    return ('LONG', conf, rv[2], rv[3])

# ─── NOVEL ENHANCED SIGNALS ─────────────────────────────

def sig_stacked_vwap_slope(i, f5m, ctx, params):
    """Only take stacked signal when VWAP distance is still FALLING (not already recovering).
    Hypothesis: if VWAP dist slope < 0, price is still diverging from VWAP = max mean-reversion potential."""
    base = sig_stacked(i, f5m, ctx, params)
    if base is None: return None
    vs = f5m['vwap_slope'].iloc[i] if 'vwap_slope' in f5m.columns else np.nan
    max_slope = params.get('max_vwap_slope', 0.0)  # negative = still falling
    if not np.isnan(vs) and vs > max_slope: return None
    return base

def sig_stacked_spread(i, f5m, ctx, params):
    """Only take stacked signal when 5m spread is compressed (low vol bar = coiled spring)."""
    base = sig_stacked(i, f5m, ctx, params)
    if base is None: return None
    sp = f5m['spread_pct'].iloc[i] if 'spread_pct' in f5m.columns else np.nan
    max_spread = params.get('max_spread', 0.5)  # max spread % of close
    if not np.isnan(sp) and sp > max_spread: return None
    return base

def sig_stacked_bullish1m(i, f5m, ctx, params):
    """Only take stacked signal when 1m bars are already showing bullish pressure (>=N bullish in last 5)."""
    base = sig_stacked(i, f5m, ctx, params)
    if base is None: return None
    bc = f5m['bullish_1m'].iloc[i] if 'bullish_1m' in f5m.columns else np.nan
    min_bullish = params.get('min_bullish_1m', 3)
    if not np.isnan(bc) and bc < min_bullish: return None
    # boost confidence if strong micro-bullish pressure
    if not np.isnan(bc) and bc >= 4:
        c = min(0.95, base[1] + 0.03)
        return (base[0], c, base[2], base[3])
    return base

def sig_stacked_gap(i, f5m, ctx, params):
    """Stacked + gap filter: stronger signal on gap-down days (gap fill momentum)."""
    base = sig_stacked(i, f5m, ctx, params)
    if base is None: return None
    gap = f5m['gap_pct'].iloc[i] if 'gap_pct' in f5m.columns else np.nan
    max_gap = params.get('max_gap', 0.0)  # only on gap-down days (gap < 0)
    if not np.isnan(gap) and gap > max_gap: return None
    # confidence boost on big gap-down days
    if not np.isnan(gap) and gap < -1.0:
        c = min(0.95, base[1] + 0.05)
        return (base[0], c, base[2], base[3])
    return base

def sig_stacked_rsi_div(i, f5m, ctx, params):
    """Stacked + intraday RSI divergence: 5m RSI slope positive while price still falling.
    This is hidden bullish divergence - momentum recovering before price does."""
    base = sig_stacked(i, f5m, ctx, params)
    if base is None: return None
    rsi_sl = f5m['rsi_slope'].iloc[i] if 'rsi_slope' in f5m.columns else np.nan
    vwap_sl = f5m['vwap_slope'].iloc[i] if 'vwap_slope' in f5m.columns else np.nan
    min_rsi_slope = params.get('min_rsi_slope', 0.5)  # RSI must be rising
    if not np.isnan(rsi_sl) and rsi_sl < min_rsi_slope: return None
    # extra boost: RSI rising + VWAP dist still falling = maximum divergence
    if not np.isnan(rsi_sl) and not np.isnan(vwap_sl) and rsi_sl > 1.0 and vwap_sl < -0.05:
        c = min(0.95, base[1] + 0.05)
        return (base[0], c, base[2], base[3])
    return base

def sig_stacked_slope_agree(i, f5m, ctx, params):
    """Stacked + channel slopes agree: 1h/4h/daily channels all sloping UP.
    Higher TFs all in uptrend = maximum alignment for mean-reversion from dip."""
    base = sig_stacked(i, f5m, ctx, params)
    if base is None: return None
    ds = ctx['daily_slope']
    hs = ctx['1h_slope'][i] if '1h_slope' in ctx else np.nan
    h4s = ctx['4h_slope'][i] if '4h_slope' in ctx else np.nan
    min_slopes_up = params.get('min_slopes_up', 2)  # at least N of 3 slopes > 0
    count = 0
    if not np.isnan(ds) and ds > 0: count += 1
    if not np.isnan(hs) and hs > 0: count += 1
    if not np.isnan(h4s) and h4s > 0: count += 1
    if count < min_slopes_up: return None
    # boost confidence for 3/3 alignment
    if count == 3:
        c = min(0.95, base[1] + 0.03)
        return (base[0], c, base[2], base[3])
    return base

def sig_stacked_autocorr(i, f5m, ctx, params):
    """Stacked + returns autocorrelation filter: only trade when market is mean-reverting (neg autocorr).
    Hypothesis: our signal IS mean-reversion, so it should work best in mean-reverting regimes."""
    base = sig_stacked(i, f5m, ctx, params)
    if base is None: return None
    ac = f5m['ret_autocorr'].iloc[i] if 'ret_autocorr' in f5m.columns else np.nan
    max_autocorr = params.get('max_autocorr', 0.0)  # negative = mean-reverting
    if not np.isnan(ac) and ac > max_autocorr: return None
    return base

def sig_stacked_enhanced_conf(i, f5m, ctx, params):
    """Stacked with enhanced confidence formula incorporating all novel features.
    Instead of filtering, use all features to adjust confidence up/down."""
    base = sig_stacked(i, f5m, ctx, params)
    if base is None: return None
    conf = base[1]
    # VWAP slope: still falling = boost
    vs = f5m['vwap_slope'].iloc[i] if 'vwap_slope' in f5m.columns else np.nan
    if not np.isnan(vs) and vs < -0.05: conf += 0.02
    # Bullish 1m bars: high count = boost
    bc = f5m['bullish_1m'].iloc[i] if 'bullish_1m' in f5m.columns else np.nan
    if not np.isnan(bc) and bc >= 4: conf += 0.02
    # Gap down: boost
    gap = f5m['gap_pct'].iloc[i] if 'gap_pct' in f5m.columns else np.nan
    if not np.isnan(gap) and gap < -0.5: conf += 0.02
    # RSI divergence: rising RSI = boost
    rsi_sl = f5m['rsi_slope'].iloc[i] if 'rsi_slope' in f5m.columns else np.nan
    if not np.isnan(rsi_sl) and rsi_sl > 0.5: conf += 0.02
    # Slope agreement: all up = boost
    ds = ctx['daily_slope']
    hs = ctx['1h_slope'][i] if '1h_slope' in ctx else np.nan
    h4s = ctx['4h_slope'][i] if '4h_slope' in ctx else np.nan
    up = 0
    if not np.isnan(ds) and ds > 0: up += 1
    if not np.isnan(hs) and hs > 0: up += 1
    if not np.isnan(h4s) and h4s > 0: up += 1
    if up == 3: conf += 0.02
    # Spread compressed: boost
    sp = f5m['spread_pct'].iloc[i] if 'spread_pct' in f5m.columns else np.nan
    if not np.isnan(sp) and sp < 0.3: conf += 0.01
    return ('LONG', min(conf, 0.95), base[2], base[3])

def main():
    print("="*70); print("V11: NOVEL / UNORTHODOX SIGNAL ENHANCEMENTS"); print("="*70); _flush()
    df1m = load_1min()
    features = build_features(df1m)
    f5m = features['5m']
    precomp = precompute_all(features, f5m)

    dp = {'stop':0.008, 'tp':0.020, 'd_min':0.20, 'h1_min':0.15, 'f5_thresh':0.25,
          'div_thresh':0.35, 'vwap_thresh':-0.30, 'need_turn':False, 'min_vol_ratio':0.8}

    # ── BASELINE ──
    print("\n" + "="*70)
    print("BASELINE: Stack N vr>0.8 m=10 75K-300K (full day)")
    print("="*70); _flush()
    trades = simulate(f5m, sig_stacked, "baseline", dp, precomp, conf_size=True)
    report(trades, "Baseline Stack", show_years=False)

    # ── EXP 1: VWAP SLOPE FILTER ──
    print("\n" + "="*70)
    print("EXP 1: VWAP slope filter (only trade when VWAP dist still falling)")
    print("="*70); _flush()
    for max_sl in [0.10, 0.05, 0.0, -0.02, -0.05, -0.10]:
        p = dict(dp, max_vwap_slope=max_sl)
        trades = simulate(f5m, sig_stacked_vwap_slope, f"vws<={max_sl}", p, precomp, conf_size=True)
        rs(trades, f"Stack+VWslope<={max_sl:.2f}")

    # ── EXP 2: SPREAD COMPRESSION ──
    print("\n" + "="*70)
    print("EXP 2: Spread compression filter (tight 5m bars only)")
    print("="*70); _flush()
    for max_sp in [0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.50]:
        p = dict(dp, max_spread=max_sp)
        trades = simulate(f5m, sig_stacked_spread, f"sp<{max_sp}", p, precomp, conf_size=True)
        rs(trades, f"Stack+spread<{max_sp:.2f}")

    # ── EXP 3: 1M BULLISH PRESSURE ──
    print("\n" + "="*70)
    print("EXP 3: 1m bullish pressure filter (bullish bars in last 5 1m bars)")
    print("="*70); _flush()
    for min_bc in [1, 2, 3, 4]:
        p = dict(dp, min_bullish_1m=min_bc)
        trades = simulate(f5m, sig_stacked_bullish1m, f"b1m>={min_bc}", p, precomp, conf_size=True)
        rs(trades, f"Stack+bullish1m>={min_bc}")

    # ── EXP 4: GAP FILTER ──
    print("\n" + "="*70)
    print("EXP 4: Gap filter (gap-down days = gap fill momentum)")
    print("="*70); _flush()
    for mg in [0.5, 0.0, -0.5, -1.0, -2.0, 99.0]:
        label = "any" if mg > 50 else f"<{mg}"
        p = dict(dp, max_gap=mg)
        trades = simulate(f5m, sig_stacked_gap, f"gap{label}", p, precomp, conf_size=True)
        rs(trades, f"Stack+gap{label}")

    # ── EXP 5: RSI DIVERGENCE ──
    print("\n" + "="*70)
    print("EXP 5: Intraday RSI divergence (RSI rising while price falling)")
    print("="*70); _flush()
    for min_rs in [0.0, 0.3, 0.5, 1.0, 1.5, 2.0]:
        p = dict(dp, min_rsi_slope=min_rs)
        trades = simulate(f5m, sig_stacked_rsi_div, f"rsi_s>{min_rs}", p, precomp, conf_size=True)
        rs(trades, f"Stack+RSIslope>{min_rs:.1f}")

    # ── EXP 6: CHANNEL SLOPE AGREEMENT ──
    print("\n" + "="*70)
    print("EXP 6: Channel slope agreement (higher TFs all sloping up)")
    print("="*70); _flush()
    for min_up in [1, 2, 3]:
        p = dict(dp, min_slopes_up=min_up)
        trades = simulate(f5m, sig_stacked_slope_agree, f"slopes>={min_up}", p, precomp, conf_size=True)
        rs(trades, f"Stack+slopes_up>={min_up}")

    # ── EXP 7: RETURNS AUTOCORRELATION ──
    print("\n" + "="*70)
    print("EXP 7: Returns autocorrelation (mean-reverting regime filter)")
    print("="*70); _flush()
    for mac in [0.2, 0.1, 0.0, -0.1, -0.2]:
        p = dict(dp, max_autocorr=mac)
        trades = simulate(f5m, sig_stacked_autocorr, f"ac<{mac}", p, precomp, conf_size=True)
        rs(trades, f"Stack+autocorr<{mac:.1f}")

    # ── EXP 8: TRAILING STOP EXPONENT SWEEP ──
    print("\n" + "="*70)
    print("EXP 8: Trail exponent sweep (is power=4 optimal?)")
    print("="*70); _flush()
    for base in [0.008, 0.010, 0.015, 0.020, 0.025]:
        for power in [2, 3, 4, 5, 6, 8]:
            trades = simulate(f5m, sig_stacked, f"tb{base}_p{power}", dp, precomp,
                            tb=base, tp=power, conf_size=True)
            rs(trades, f"Stack tb={base:.3f} p={power}")

    # ── EXP 9: ENHANCED CONFIDENCE (ALL FEATURES) ──
    print("\n" + "="*70)
    print("EXP 9: Enhanced confidence formula (all novel features -> conf adjustment)")
    print("="*70); _flush()
    trades = simulate(f5m, sig_stacked_enhanced_conf, "enh_conf", dp, precomp, conf_size=True)
    report(trades, "Stack+enhanced_conf")

    # ── EXP 10: BEST FILTERS COMBINED ──
    print("\n" + "="*70)
    print("EXP 10: Combine best filters from above")
    print("="*70); _flush()

    # Stacked + slope agree (2+) + VWAP slope <= 0.05
    def sig_combo_a(i, f5m, ctx, params):
        base = sig_stacked_slope_agree(i, f5m, ctx, params)
        if base is None: return None
        vs = f5m['vwap_slope'].iloc[i] if 'vwap_slope' in f5m.columns else np.nan
        if not np.isnan(vs) and vs > 0.05: return None
        return base
    p = dict(dp, min_slopes_up=2)
    trades = simulate(f5m, sig_combo_a, "combo_a", p, precomp, conf_size=True)
    rs(trades, "Combo A: slopes>=2 + VWslope<=0.05")

    # Stacked + RSI divergence + spread < 0.5
    def sig_combo_b(i, f5m, ctx, params):
        base = sig_stacked_rsi_div(i, f5m, ctx, params)
        if base is None: return None
        sp = f5m['spread_pct'].iloc[i] if 'spread_pct' in f5m.columns else np.nan
        if not np.isnan(sp) and sp > 0.5: return None
        return base
    p = dict(dp, min_rsi_slope=0.5)
    trades = simulate(f5m, sig_combo_b, "combo_b", p, precomp, conf_size=True)
    rs(trades, "Combo B: RSI_div>0.5 + spread<0.5")

    # Stacked + gap<=0 + bullish1m>=3
    def sig_combo_c(i, f5m, ctx, params):
        base = sig_stacked_gap(i, f5m, ctx, params)
        if base is None: return None
        bc = f5m['bullish_1m'].iloc[i] if 'bullish_1m' in f5m.columns else np.nan
        if not np.isnan(bc) and bc < 3: return None
        return base
    p = dict(dp, max_gap=0.0)
    trades = simulate(f5m, sig_combo_c, "combo_c", p, precomp, conf_size=True)
    rs(trades, "Combo C: gap<=0 + bullish1m>=3")

    # Enhanced conf + afternoon only
    trades = simulate(f5m, sig_stacked_enhanced_conf, "enh_pm", dp, precomp,
                     conf_size=True, tod_start=dt.time(13,0), tod_end=dt.time(15,25))
    rs(trades, "Enhanced conf + afternoon")

    # ── EXP 11: WIDER PARAMS + BEST NOVEL FILTERS ──
    print("\n" + "="*70)
    print("EXP 11: Wider params + best novel filters")
    print("="*70); _flush()
    # Wider stacked (vw<-0.10, dv>0.20, f5<0.35) + enhanced conf
    pw = dict(dp, vwap_thresh=-0.10, div_thresh=0.20, f5_thresh=0.35)
    trades = simulate(f5m, sig_stacked_enhanced_conf, "wide_enh", pw, precomp, conf_size=True)
    report(trades, "Wider + enhanced conf")

    # Wider + slope agree 2+
    p = dict(pw, min_slopes_up=2)
    trades = simulate(f5m, sig_stacked_slope_agree, "wide_slope2", p, precomp, conf_size=True)
    rs(trades, "Wider + slopes>=2")

    # Wider + afternoon + enhanced
    trades = simulate(f5m, sig_stacked_enhanced_conf, "wide_pm_enh", pw, precomp,
                     conf_size=True, tod_start=dt.time(13,0), tod_end=dt.time(15,25))
    rs(trades, "Wider + PM + enhanced conf")

    # ── EXP 12: Full reports on best configs ──
    print("\n" + "="*70)
    print("EXP 12: Full year-by-year on BEST NOVEL configs")
    print("="*70); _flush()

    # A: Enhanced conf (full day, default params)
    print("\n--- Enhanced conf (full day) ---"); _flush()
    trades = simulate(f5m, sig_stacked_enhanced_conf, "enh_full", dp, precomp, conf_size=True)
    report(trades, "Enhanced conf full day")

    # B: Wider + enhanced conf
    print("\n--- Wider + enhanced conf ---"); _flush()
    trades = simulate(f5m, sig_stacked_enhanced_conf, "wide_enh2", pw, precomp, conf_size=True)
    report(trades, "Wider + enhanced conf")

    # C: Enhanced conf + PM
    print("\n--- Enhanced conf + PM ---"); _flush()
    trades = simulate(f5m, sig_stacked_enhanced_conf, "enh_pm2", dp, precomp,
                     conf_size=True, tod_start=dt.time(13,0), tod_end=dt.time(15,25))
    report(trades, "Enhanced conf + PM")

    print("\n" + "="*70)
    print("V11 DONE.")
    print("="*70); _flush()

if __name__=='__main__':
    main()
