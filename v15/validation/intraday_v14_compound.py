#!/usr/bin/env python3
"""V14: Capital compounding + final optimizations.
V13 champion: PM Enh-Union mtd=20 $200K-$800K = $24.7M at 84.6% WR.
Tests:
1. Capital compounding (reinvest profits, sizing scales with equity)
2. Drawdown-aware sizing (reduce after consecutive losses)
3. Time-block analysis (different params for 13:00-14:00 vs 14:00-15:25)
4. Cooldown impact (cd=1 or cd=2 after losses only)
5. Monthly seasonality
6. Final champion configurations with compounding
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime as dt

def _flush(): sys.stdout.flush()

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

def simulate_compound(f5m, signal_fn, name, params, precomp,
                      tb=0.006, tp=6, cd=0, mtd=20,
                      initial_capital=500_000.0, risk_pct=0.02,
                      max_risk_pct=0.04, min_size=50_000.0,
                      tod_start=dt.time(13,0), tod_end=dt.time(15,25),
                      dd_reduce=False, dd_threshold=3):
    """Compounding simulator: position size = equity * risk_pct * confidence."""
    (dcp_arr,dslope_arr), htf = precomp
    o_arr=f5m['open'].values; h_arr=f5m['high'].values
    l_arr=f5m['low'].values; c_arr=f5m['close'].values
    times=f5m.index; tod_arr=np.array([t.time() for t in times])
    n=len(f5m); trades=[]
    ctx={'daily_cp':0.0,'daily_slope':0.0,**htf}
    in_trade=False; ep=et=None; conf=sp=tpp=bp=0.0; hb=cr=tt=0; cd_=None; ps=None
    equity = initial_capital
    consec_loss = 0
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
                # Compounding: size based on current equity
                rp = risk_pct
                if dd_reduce and consec_loss >= dd_threshold:
                    rp = risk_pct * 0.5  # halve risk after consecutive losses
                cap = max(min_size, equity * rp * conf)
                cap = min(cap, equity * max_risk_pct)  # cap max risk
                sh=max(1,int(cap/ep))
                pnl=(xa-ep)*sh-COMM_PER_SHARE*sh*2
                equity += pnl
                if pnl <= 0:
                    consec_loss += 1
                else:
                    consec_loss = 0
                trades.append((et,bt,ep,xa,conf,sh,pnl,hb,xr,name,equity))
                in_trade=False; cr=cd
        if cr>0: cr-=1; continue
        if in_trade or tt>=mtd: continue
        if btod<tod_start or btod>tod_end: continue
        if equity < 10_000: continue  # stop trading if almost wiped out
        ctx['daily_cp']=dcp_arr[i]; ctx['daily_slope']=dslope_arr[i]
        result=signal_fn(i,f5m,ctx,params)
        if result is not None:
            _,co,s,t=result; ps=(co,s,t)
    return trades, equity

def simulate_fixed(f5m, signal_fn, name, params, precomp,
                   tb=0.006, tp=6, cd=0, mtd=20, conf_size=True,
                   base_capital=200_000.0, max_capital=800_000.0,
                   tod_start=dt.time(13,0), tod_end=dt.time(15,25)):
    """Fixed sizing simulator (same as V13 for comparison)."""
    (dcp_arr,dslope_arr), htf = precomp
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
                cap = (base_capital + (max_capital - base_capital) * conf) if conf_size else 100_000.0
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

def report_compound(trades, label, initial_cap):
    if not trades: print(f"  {label}: 0 trades"); _flush(); return
    n=len(trades); pnls=[t[6] for t in trades]
    w=sum(1 for p in pnls if p>0); wr=w/n*100; total=sum(pnls)
    bl=min(pnls); bw=max(pnls)
    final_eq = trades[-1][10] if trades else initial_cap
    ret = (final_eq / initial_cap - 1) * 100
    # Max drawdown
    peak = initial_cap; max_dd = 0
    eq = initial_cap
    for t in trades:
        eq += t[6]
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd
    print(f"  {label}: {n}t {wr:.1f}% WR | Init=${initial_cap:,.0f} -> Final=${final_eq:,.0f} ({ret:+.1f}%)")
    print(f"    Total PnL: ${total:+,.0f} | BW=${bw:+,.0f} BL=${bl:+,.0f} | MaxDD={max_dd:.1f}%")
    years=sorted(set(t[0].year for t in trades))
    for yr in years:
        yt=[t for t in trades if t[0].year==yr]
        yw=sum(1 for t in yt if t[6]>0); yp=sum(t[6] for t in yt); ybl=min(t[6] for t in yt)
        ye = yt[-1][10] if yt else 0
        ywr=yw/len(yt)*100
        print(f"    {yr}: {len(yt):3d}t {ywr:5.1f}% ${yp:+10,.0f} BL=${ybl:+,.0f} eq=${ye:,.0f}")
    _flush()

def rs(trades, label):
    if not trades: print(f"  {label}: 0 trades"); _flush(); return
    n=len(trades); pnls=[t[6] for t in trades]
    w=sum(1 for p in pnls if p>0); wr=w/n*100; total=sum(pnls); bl=min(pnls)
    train=[t for t in trades if t[0]<=TRAIN_END]
    test=[t for t in trades if TRAIN_END<t[0]<=TEST_END]
    tp_=sum(t[6] for t in train); tsp=sum(t[6] for t in test)
    bal = tsp/tp_*100 if tp_>0 else 0
    oos=[t for t in trades if t[0]>TEST_END]
    on=len(oos); ow=sum(1 for t in oos if t[6]>0); op=sum(t[6] for t in oos)
    print(f"  {label}: {n:5d}t {wr:5.1f}% ${total:+10,.0f} BL=${bl:+,.0f} BAL={bal:.0f}% OOS:{on}t {ow}W ${op:+,.0f}")
    _flush()

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

def sig_stacked(i, f5m, ctx, params):
    rv = sig_vwap(i, f5m, ctx, params)
    rd = sig_div(i, f5m, ctx, params)
    if rv is None or rd is None: return None
    conf = min(0.95, max(rv[1], rd[1]) + 0.05)
    return ('LONG', conf, rv[2], rv[3])

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

def main():
    print("="*70); print("V14: CAPITAL COMPOUNDING + FINAL OPTIMIZATIONS"); print("="*70); _flush()
    df1m = load_1min()
    features = build_features(df1m)
    f5m = features['5m']
    precomp = precompute_all(features, f5m)

    pw = {'stop':0.008, 'tp':0.020, 'd_min':0.20, 'h1_min':0.15, 'f5_thresh':0.35,
          'div_thresh':0.20, 'vwap_thresh':-0.10, 'min_vol_ratio':0.8}

    # ── EXP 1: Compounding with various starting capitals ──
    print("\n" + "="*70)
    print("EXP 1: Compounding - Enh-Union PM mtd=20")
    print("="*70); _flush()
    for init_cap in [100_000, 250_000, 500_000, 1_000_000]:
        for rp in [0.01, 0.02, 0.03, 0.05]:
            trades, final = simulate_compound(f5m, sig_union_enh, "cmp", pw, precomp,
                                            mtd=20, initial_capital=float(init_cap),
                                            risk_pct=rp, max_risk_pct=rp*2, min_size=50_000.0)
            if trades:
                n=len(trades); pnls=[t[6] for t in trades]
                w=sum(1 for p in pnls if p>0); wr=w/n*100; total=sum(pnls)
                ret = (final / init_cap - 1) * 100
                print(f"  Init=${init_cap//1000}K rp={rp:.0%}: {n}t {wr:.1f}% ${total:+,.0f} -> ${final:,.0f} ({ret:+.0f}%)")
                _flush()

    # ── EXP 2: Compounding with drawdown reduction ──
    print("\n" + "="*70)
    print("EXP 2: Compounding with DD-aware sizing")
    print("="*70); _flush()
    for init_cap in [500_000]:
        for rp in [0.02, 0.03]:
            for dd_thresh in [2, 3, 5]:
                trades, final = simulate_compound(f5m, sig_union_enh, "cmp_dd", pw, precomp,
                                                mtd=20, initial_capital=float(init_cap),
                                                risk_pct=rp, max_risk_pct=rp*2,
                                                dd_reduce=True, dd_threshold=dd_thresh)
                if trades:
                    n=len(trades); total=sum(t[6] for t in trades)
                    ret = (final / init_cap - 1) * 100
                    print(f"  Init=$500K rp={rp:.0%} dd_thresh={dd_thresh}: ${total:+,.0f} -> ${final:,.0f} ({ret:+.0f}%)")
                    _flush()

    # ── EXP 3: Monthly seasonality ──
    print("\n" + "="*70)
    print("EXP 3: Monthly seasonality (which months are best?)")
    print("="*70); _flush()
    trades = simulate_fixed(f5m, sig_union_enh, "season", pw, precomp, mtd=20)
    for m in range(1, 13):
        mt = [t for t in trades if t[0].month == m]
        if mt:
            mn = len(mt); mp = [t[6] for t in mt]
            mw = sum(1 for p in mp if p > 0); mwr = mw/mn*100
            mtotal = sum(mp)
            print(f"  Month {m:2d}: {mn:4d}t {mwr:5.1f}% ${mtotal:+10,.0f}")
    _flush()

    # ── EXP 4: Day-of-week analysis ──
    print("\n" + "="*70)
    print("EXP 4: Day-of-week analysis")
    print("="*70); _flush()
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    for d in range(5):
        dt_ = [t for t in trades if t[0].weekday() == d]
        if dt_:
            dn = len(dt_); dp = [t[6] for t in dt_]
            dw = sum(1 for p in dp if p > 0); dwr = dw/dn*100
            dtotal = sum(dp)
            print(f"  {dow_names[d]}: {dn:4d}t {dwr:5.1f}% ${dtotal:+10,.0f}")
    _flush()

    # ── EXP 5: Full compound reports on best configs ──
    print("\n" + "="*70)
    print("EXP 5: Full compound year-by-year reports")
    print("="*70); _flush()

    # A: $500K start, 2% risk, Enh-Union mtd=20
    print("\n--- A: Compound $500K start, 2% risk, Enh-Union mtd=20 ---"); _flush()
    trades, final = simulate_compound(f5m, sig_union_enh, "best_a", pw, precomp,
                                    mtd=20, initial_capital=500_000.0,
                                    risk_pct=0.02, max_risk_pct=0.04)
    report_compound(trades, "Compound Enh-Union $500K 2%", 500_000.0)

    # B: $500K start, 3% risk, Enh-Union mtd=20
    print("\n--- B: Compound $500K start, 3% risk, Enh-Union mtd=20 ---"); _flush()
    trades, final = simulate_compound(f5m, sig_union_enh, "best_b", pw, precomp,
                                    mtd=20, initial_capital=500_000.0,
                                    risk_pct=0.03, max_risk_pct=0.06)
    report_compound(trades, "Compound Enh-Union $500K 3%", 500_000.0)

    # C: $1M start, 2% risk, Enh-Union mtd=20
    print("\n--- C: Compound $1M start, 2% risk, Enh-Union mtd=20 ---"); _flush()
    trades, final = simulate_compound(f5m, sig_union_enh, "best_c", pw, precomp,
                                    mtd=20, initial_capital=1_000_000.0,
                                    risk_pct=0.02, max_risk_pct=0.04)
    report_compound(trades, "Compound Enh-Union $1M 2%", 1_000_000.0)

    # D: Fixed sizing comparison (same signal, $200K-$800K)
    print("\n--- D: Fixed sizing $200K-$800K (comparison) ---"); _flush()
    trades = simulate_fixed(f5m, sig_union_enh, "fixed_d", pw, precomp, mtd=20)
    report(trades, "Fixed Enh-Union $200K-$800K mtd=20")

    # E: Enh-Stack compound (high WR)
    dp = {'stop':0.008, 'tp':0.020, 'd_min':0.20, 'h1_min':0.15, 'f5_thresh':0.25,
          'div_thresh':0.35, 'vwap_thresh':-0.30, 'min_vol_ratio':0.8}
    print("\n--- E: Compound $500K 2%, Enh-Stack default (high WR) ---"); _flush()
    trades, final = simulate_compound(f5m, sig_stacked_enh, "best_e", dp, precomp,
                                    mtd=10, initial_capital=500_000.0,
                                    risk_pct=0.02, max_risk_pct=0.04)
    report_compound(trades, "Compound Enh-Stack $500K 2%", 500_000.0)

    print("\n" + "="*70)
    print("V14 DONE.")
    print("="*70); _flush()

if __name__=='__main__':
    main()
