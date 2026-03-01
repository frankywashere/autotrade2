#!/usr/bin/env python3
"""V12: Aggressive sizing + hybrid time-of-day strategies.
V10 found: PM Union vw<-0.1 dv>0.20 = $6.65M at 82.9% WR (13.5K trades!)
V10 found: PM Stack vw<-0.1 dv>0.20 f5<0.35 = $4.45M at 86.4% WR
Test:
1. Aggressive conf sizing on best PM Union configs ($100K-400K, 100K-500K)
2. Hybrid: AM stacked (high WR) + PM union (high volume)
3. Extended PM window (12:00 or 11:00 start)
4. Trail base/power sweep on PM Union
5. Ultra-wide params on PM Union (vw<-0.05, dv>0.15)
6. Afternoon + wider + aggressive sizing = max PnL target
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

def build_features(df1m):
    t0=time.time(); print("Resampling..."); _flush()
    tfs={}
    for rule,name in [('5min','5m'),('15min','15m'),('30min','30m'),('1h','1h'),('4h','4h')]:
        tfs[name]=resample_ohlcv(df1m,rule)
    tfs['daily']=resample_ohlcv(df1m,'1D')
    for name,df in tfs.items(): print(f"  {name}: {len(df):,}"); _flush()
    features={}
    wins={'5m':60,'15m':40,'30m':30,'1h':24,'4h':20,'daily':40}
    for name,df in tfs.items():
        print(f"Features {name}..."); _flush()
        c=df['close'].values.astype(np.float64); v=df['volume'].values.astype(np.float64)
        h=df['high'].values.astype(np.float64); l=df['low'].values.astype(np.float64)
        o=df['open'].values.astype(np.float64)
        feat=pd.DataFrame(index=df.index)
        feat['open']=o; feat['high']=h; feat['low']=l; feat['close']=c; feat['volume']=v
        feat['chan_pos']=channel_position(c,wins[name])
        feat['rsi']=compute_rsi(c); feat['bvc']=compute_bvc(c,v)
        if name=='5m':
            dates=np.array([t.date() for t in df.index])
            vw,vd=compute_vwap(o,h,l,c,v,dates)
            feat['vwap']=vw; feat['vwap_dist']=vd
            feat['vol_ratio']=compute_volume_ratio(v)
        features[name]=feat
    print(f"Done in {time.time()-t0:.1f}s"); _flush()
    return features

def precompute_all(features, f5m):
    print("Pre-computing..."); _flush()
    t0=time.time(); n=len(f5m)
    bar_dates=np.array([t.date() for t in f5m.index])
    daily=features['daily']
    dcp=daily['chan_pos'].values; drsi=daily['rsi'].values; dbvc=daily['bvc'].values
    ddates=np.array([idx.date() if hasattr(idx,'date') else idx for idx in daily.index])
    dcp_arr=np.full(n,np.nan); drsi_arr=np.full(n,np.nan); dbvc_arr=np.full(n,np.nan)
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
    arrays={
        '1h_cp':np.full(n,np.nan),'1h_rsi':np.full(n,np.nan),'1h_bvc':np.full(n,np.nan),
        '4h_cp':np.full(n,np.nan),
        '15m_cp':np.full(n,np.nan),
        '30m_cp':np.full(n,np.nan),
    }
    cfgs=[
        (features['1h'],{'1h_cp':'chan_pos','1h_rsi':'rsi','1h_bvc':'bvc'}),
        (features['4h'],{'4h_cp':'chan_pos'}),
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
    return (dcp_arr,drsi_arr,dbvc_arr), arrays

def simulate(f5m, signal_fn, name, params, precomp,
             tb=0.010, tp=4, cd=0, mtd=10, conf_size=False,
             base_capital=75_000.0, max_capital=300_000.0,
             tod_start=dt.time(9,40), tod_end=dt.time(15,25)):
    (dcp_arr,_,_), htf = precomp
    o_arr=f5m['open'].values; h_arr=f5m['high'].values
    l_arr=f5m['low'].values; c_arr=f5m['close'].values
    times=f5m.index; tod_arr=np.array([t.time() for t in times])
    n=len(f5m); trades=[]
    ctx={'daily_cp':0.0,**htf}
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
        ctx['daily_cp']=dcp_arr[i]
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

def sig_union(i, f5m, ctx, params):
    best=None; bc=0
    for fn in [sig_vwap, sig_div]:
        r=fn(i,f5m,ctx,params)
        if r and r[1]>bc: best=r; bc=r[1]
    return best

def main():
    print("="*70); print("V12: AGGRESSIVE SIZING + HYBRID TOD STRATEGIES"); print("="*70); _flush()
    df1m = load_1min()
    features = build_features(df1m)
    f5m = features['5m']
    precomp = precompute_all(features, f5m)

    dp = {'stop':0.008, 'tp':0.020, 'd_min':0.20, 'h1_min':0.15, 'f5_thresh':0.30,
          'div_thresh':0.35, 'vwap_thresh':-0.30, 'need_turn':False, 'min_vol_ratio':0.8}

    # ── EXP 1: Aggressive sizing on PM Union ──
    print("\n" + "="*70)
    print("EXP 1: Aggressive conf sizing on PM Union (wider params)")
    print("="*70); _flush()
    pw_union = dict(dp, vwap_thresh=-0.10, div_thresh=0.20, f5_thresh=0.30)
    for bc, mc in [(50_000, 200_000), (75_000, 300_000), (100_000, 400_000),
                   (100_000, 500_000), (150_000, 600_000), (200_000, 800_000)]:
        trades = simulate(f5m, sig_union, f"Un_{bc//1000}K-{mc//1000}K", pw_union, precomp,
                         conf_size=True, base_capital=float(bc), max_capital=float(mc),
                         tod_start=dt.time(13,0), tod_end=dt.time(15,25))
        rs(trades, f"PM Union vw<-0.1 dv>0.20 ${bc//1000}K-${mc//1000}K")

    # Higher WR variant
    pw_union2 = dict(dp, vwap_thresh=-0.20, div_thresh=0.25, f5_thresh=0.30)
    for bc, mc in [(75_000, 300_000), (100_000, 400_000), (100_000, 500_000), (150_000, 600_000)]:
        trades = simulate(f5m, sig_union, f"Un2_{bc//1000}K-{mc//1000}K", pw_union2, precomp,
                         conf_size=True, base_capital=float(bc), max_capital=float(mc),
                         tod_start=dt.time(13,0), tod_end=dt.time(15,25))
        rs(trades, f"PM Union vw<-0.2 dv>0.25 ${bc//1000}K-${mc//1000}K")

    # ── EXP 2: Aggressive sizing on PM Stacked ──
    print("\n" + "="*70)
    print("EXP 2: Aggressive conf sizing on PM Stacked (wider params)")
    print("="*70); _flush()
    pw_stack = dict(dp, vwap_thresh=-0.10, div_thresh=0.20, f5_thresh=0.35)
    for bc, mc in [(75_000, 300_000), (100_000, 400_000), (100_000, 500_000),
                   (150_000, 600_000), (200_000, 800_000)]:
        trades = simulate(f5m, sig_stacked, f"St_{bc//1000}K-{mc//1000}K", pw_stack, precomp,
                         conf_size=True, base_capital=float(bc), max_capital=float(mc),
                         tod_start=dt.time(13,0), tod_end=dt.time(15,25))
        rs(trades, f"PM Stack vw<-0.1 dv>0.20 f5<0.35 ${bc//1000}K-${mc//1000}K")

    # ── EXP 3: Hybrid AM+PM strategy ──
    print("\n" + "="*70)
    print("EXP 3: Hybrid AM stacked + PM union (best of both)")
    print("="*70); _flush()
    # AM (9:40-12:00) use stacked (high WR), PM (13:00-15:25) use union (high volume)
    am_params = dict(dp)  # default params for AM stacked
    pm_params_union = dict(dp, vwap_thresh=-0.10, div_thresh=0.20, f5_thresh=0.30)
    pm_params_stack = dict(dp, vwap_thresh=-0.10, div_thresh=0.20, f5_thresh=0.35)

    for bc, mc in [(75_000, 300_000), (100_000, 400_000), (100_000, 500_000)]:
        # AM stacked
        am_trades = simulate(f5m, sig_stacked, "am_st", am_params, precomp,
                           conf_size=True, base_capital=float(bc), max_capital=float(mc),
                           tod_start=dt.time(9,40), tod_end=dt.time(12,0))
        # PM union
        pm_trades = simulate(f5m, sig_union, "pm_un", pm_params_union, precomp,
                           conf_size=True, base_capital=float(bc), max_capital=float(mc),
                           tod_start=dt.time(13,0), tod_end=dt.time(15,25))
        hybrid = am_trades + pm_trades
        hybrid.sort(key=lambda t: t[0])
        rs(hybrid, f"Hybrid AM-Stack+PM-Union ${bc//1000}K-${mc//1000}K")

        # AM stacked + PM stacked wider
        pm_trades2 = simulate(f5m, sig_stacked, "pm_st", pm_params_stack, precomp,
                            conf_size=True, base_capital=float(bc), max_capital=float(mc),
                            tod_start=dt.time(13,0), tod_end=dt.time(15,25))
        hybrid2 = am_trades + pm_trades2
        hybrid2.sort(key=lambda t: t[0])
        rs(hybrid2, f"Hybrid AM-Stack+PM-Stack ${bc//1000}K-${mc//1000}K")

    # ── EXP 4: Extended PM window ──
    print("\n" + "="*70)
    print("EXP 4: Extended PM window (earlier start)")
    print("="*70); _flush()
    pw = dict(dp, vwap_thresh=-0.10, div_thresh=0.20, f5_thresh=0.30)
    for start_h, start_m in [(11,0), (11,30), (12,0), (12,30), (13,0)]:
        for fn, nm in [(sig_stacked, "Stack"), (sig_union, "Union")]:
            trades = simulate(f5m, fn, f"ext_{nm}", pw, precomp, conf_size=True,
                            tod_start=dt.time(start_h, start_m), tod_end=dt.time(15,25))
            rs(trades, f"PM-ext {nm} {start_h}:{start_m:02d}-15:25 vw<-0.1 dv>0.20 75K-300K")

    # ── EXP 5: Ultra-wide params on PM Union ──
    print("\n" + "="*70)
    print("EXP 5: Ultra-wide params on PM Union")
    print("="*70); _flush()
    for vt in [-0.05, -0.08, -0.10, -0.15]:
        for dvt in [0.10, 0.15, 0.20]:
            for f5t in [0.30, 0.35, 0.40]:
                p = dict(dp, vwap_thresh=vt, div_thresh=dvt, f5_thresh=f5t, min_vol_ratio=0.0)
                trades = simulate(f5m, sig_union, f"uw", p, precomp, conf_size=True,
                                tod_start=dt.time(13,0), tod_end=dt.time(15,25))
                if trades and len(trades) >= 50:
                    n=len(trades); pnls=[t[6] for t in trades]
                    w=sum(1 for pp in pnls if pp>0); wr=w/n*100; total=sum(pnls)
                    if total > 7000000 or (wr >= 85 and total > 5000000):
                        rs(trades, f"UW Union vw<{vt} dv>{dvt} f5<{f5t} noVR")

    # With vol ratio
    for vt in [-0.05, -0.08, -0.10]:
        for dvt in [0.10, 0.15, 0.20]:
            p = dict(dp, vwap_thresh=vt, div_thresh=dvt, f5_thresh=0.35, min_vol_ratio=0.8)
            trades = simulate(f5m, sig_union, f"uwvr", p, precomp, conf_size=True,
                            tod_start=dt.time(13,0), tod_end=dt.time(15,25))
            if trades and len(trades) >= 50:
                n=len(trades); pnls=[t[6] for t in trades]
                w=sum(1 for pp in pnls if pp>0); wr=w/n*100; total=sum(pnls)
                if total > 6000000:
                    rs(trades, f"UW Union vw<{vt} dv>{dvt} f5<0.35 vr>0.8")

    # ── EXP 6: Trail base/power sweep on PM Union ──
    print("\n" + "="*70)
    print("EXP 6: Trail base/power sweep on PM Union vw<-0.1 dv>0.20")
    print("="*70); _flush()
    for tb in [0.006, 0.008, 0.010, 0.012, 0.015, 0.020]:
        for tp_val in [3, 4, 5, 6]:
            trades = simulate(f5m, sig_union, f"trail", pw_union, precomp,
                            tb=tb, tp=tp_val, conf_size=True,
                            tod_start=dt.time(13,0), tod_end=dt.time(15,25))
            rs(trades, f"PM Union tb={tb:.3f} p={tp_val}")

    # ── EXP 7: Best configs with mtd sweep ──
    print("\n" + "="*70)
    print("EXP 7: Max trades per day sweep on PM Union")
    print("="*70); _flush()
    for m in [5, 8, 10, 15, 20, 30]:
        trades = simulate(f5m, sig_union, f"mtd{m}", pw_union, precomp,
                        mtd=m, conf_size=True,
                        tod_start=dt.time(13,0), tod_end=dt.time(15,25))
        rs(trades, f"PM Union vw<-0.1 dv>0.20 mtd={m}")

    # ── EXP 8: Full year-by-year on NEW CHAMPIONS ──
    print("\n" + "="*70)
    print("EXP 8: Full year-by-year on NEW CHAMPIONS")
    print("="*70); _flush()

    # A: PM Union vw<-0.1 dv>0.20 $100K-$500K
    print("\n--- PM Union vw<-0.1 dv>0.20 $100K-$500K ---"); _flush()
    trades = simulate(f5m, sig_union, "champ_a", pw_union, precomp,
                     conf_size=True, base_capital=100_000.0, max_capital=500_000.0,
                     tod_start=dt.time(13,0), tod_end=dt.time(15,25))
    report(trades, "PM Union $100K-$500K")

    # B: PM Stack vw<-0.1 dv>0.20 f5<0.35 $100K-$500K
    print("\n--- PM Stack vw<-0.1 dv>0.20 f5<0.35 $100K-$500K ---"); _flush()
    trades = simulate(f5m, sig_stacked, "champ_b", pw_stack, precomp,
                     conf_size=True, base_capital=100_000.0, max_capital=500_000.0,
                     tod_start=dt.time(13,0), tod_end=dt.time(15,25))
    report(trades, "PM Stack $100K-$500K")

    # C: Hybrid AM-Stack + PM-Union $100K-$500K
    print("\n--- Hybrid AM-Stack + PM-Union $100K-$500K ---"); _flush()
    am_trades = simulate(f5m, sig_stacked, "hyb_am", am_params, precomp,
                       conf_size=True, base_capital=100_000.0, max_capital=500_000.0,
                       tod_start=dt.time(9,40), tod_end=dt.time(12,0))
    pm_trades = simulate(f5m, sig_union, "hyb_pm", pw_union, precomp,
                       conf_size=True, base_capital=100_000.0, max_capital=500_000.0,
                       tod_start=dt.time(13,0), tod_end=dt.time(15,25))
    hybrid = am_trades + pm_trades
    hybrid.sort(key=lambda t: t[0])
    report(hybrid, "Hybrid AM-Stack+PM-Union $100K-$500K")

    # D: PM Union vw<-0.1 dv>0.20 $150K-$600K
    print("\n--- PM Union vw<-0.1 dv>0.20 $150K-$600K ---"); _flush()
    trades = simulate(f5m, sig_union, "champ_d", pw_union, precomp,
                     conf_size=True, base_capital=150_000.0, max_capital=600_000.0,
                     tod_start=dt.time(13,0), tod_end=dt.time(15,25))
    report(trades, "PM Union $150K-$600K")

    print("\n" + "="*70)
    print("V12 DONE.")
    print("="*70); _flush()

if __name__=='__main__':
    main()
