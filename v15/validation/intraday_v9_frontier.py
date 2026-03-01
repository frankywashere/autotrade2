#!/usr/bin/env python3
"""V9: Push the frontier further.
V8 champion: Stack N vr>0.8 m=10 SCALED = $2.2M at 86.9% WR.
Can we do better? Test:
1. Wider VWAP/Div params on stacked (more trades at high WR)
2. BVC (institutional flow) filter
3. Daily RSI regime filter
4. Time-of-day analysis
5. 3m/15m channel agreement
6. Confidence formula tuning
7. Stack + different vol thresholds + conf sizing grid
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

def compute_mom_turn(c,lb=10):
    n=len(c); t=np.zeros(n,dtype=bool)
    for i in range(lb*2,n):
        m=c[i]-c[i-lb]; pm=c[i-lb]-c[i-2*lb]
        if m<0 and (m-pm)>0: t[i]=True
    return t

def compute_vwap(o,h,l,c,v,dates):
    n=len(c); vwap=np.full(n,np.nan); vd=np.full(n,np.nan)
    tp=(h+l+c)/3.0; ctv=cv=0.0; pd_=None
    for i in range(n):
        d=dates[i]
        if d!=pd_: ctv=cv=0.0; pd_=d
        ctv+=tp[i]*v[i]; cv+=v[i]
        if cv>0: vwap[i]=ctv/cv; vd[i]=(c[i]-vwap[i])/vwap[i]*100.0
    return vwap,vd

def compute_reversal(o,h,l,c):
    n=len(c); r=np.zeros(n,dtype=bool)
    for i in range(1,n):
        body=c[i]-o[i]; br=h[i]-l[i]
        if br<1e-10: continue
        ls=min(o[i],c[i])-l[i]; us=h[i]-max(o[i],c[i])
        if ls>2*abs(body) and ls>2*us and body>=0: r[i]=True; continue
        pb=c[i-1]-o[i-1]
        if pb<0 and body>0 and body>abs(pb):
            if o[i]<=c[i-1] and c[i]>=o[i-1]: r[i]=True
    return r

def compute_micro_mom(c,w=5):
    n=len(c); m=np.full(n,np.nan)
    for i in range(w,n):
        if c[i-w]>0: m[i]=(c[i]-c[i-w])/c[i-w]*100.0
    return m

def compute_volume_ratio(v, lb=20):
    n=len(v); vr=np.full(n,np.nan)
    rv=pd.Series(v).rolling(lb,min_periods=lb).mean().values
    valid=(rv>0)&~np.isnan(rv)
    vr[valid]=v[valid]/rv[valid]
    return vr

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
        if name!='1m': feat['rsi']=compute_rsi(c); feat['bvc']=compute_bvc(c,v)
        if name in ('5m','3m'): feat['mom_turn']=compute_mom_turn(c)
        if name=='5m':
            dates=np.array([t.date() for t in df.index])
            vw,vd=compute_vwap(o,h,l,c,v,dates)
            feat['vwap']=vw; feat['vwap_dist']=vd
            feat['vol_ratio']=compute_volume_ratio(v)
        if name in ('1m','3m'): feat['reversal']=compute_reversal(o,h,l,c)
        if name=='1m': feat['micro_mom']=compute_micro_mom(c)
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
        '4h_cp':np.full(n,np.nan),'15m_cp':np.full(n,np.nan),'15m_rsi':np.full(n,np.nan),
        '30m_cp':np.full(n,np.nan),
        '3m_cp':np.full(n,np.nan),'3m_rsi':np.full(n,np.nan),
        '3m_turn':np.full(n,0.0),'3m_reversal':np.full(n,0.0),
    }
    cfgs=[
        (features['1h'],{'1h_cp':'chan_pos','1h_rsi':'rsi','1h_bvc':'bvc'}),
        (features['4h'],{'4h_cp':'chan_pos'}),
        (features['15m'],{'15m_cp':'chan_pos','15m_rsi':'rsi'}),
        (features['30m'],{'30m_cp':'chan_pos'}),
        (features['3m'],{'3m_cp':'chan_pos','3m_rsi':'rsi','3m_turn':'mom_turn','3m_reversal':'reversal'}),
    ]
    for tf,cm in cfgs:
        ti=tf.index.values; ca={k:tf[v].values for k,v in cm.items()}; j=0
        for i in range(n):
            t=f5m.index[i]
            while j<len(ti)-1 and ti[j+1]<=t: j+=1
            if j<len(ti) and ti[j]<=t:
                for k in cm: arrays[k][i]=ca[k][j]
    f1m=features['1m']
    m1={'1m_mm':np.full(n,np.nan),'1m_rev':np.full(n,0.0),'1m_cp':np.full(n,np.nan)}
    ti=f1m.index.values; mm=f1m['micro_mom'].values; rv=f1m['reversal'].values.astype(float); cp1=f1m['chan_pos'].values
    j=0
    for i in range(n):
        t=f5m.index[i]
        while j<len(ti)-1 and ti[j+1]<=t: j+=1
        if j<len(ti) and ti[j]<=t:
            m1['1m_mm'][i]=mm[j]; m1['1m_cp'][i]=cp1[j]
            s=max(0,j-4)
            if np.any(rv[s:j+1]>0): m1['1m_rev'][i]=1.0
    print(f"  Done in {time.time()-t0:.1f}s"); _flush()
    return (dcp_arr,drsi_arr,dbvc_arr), arrays, m1

def simulate(f5m, signal_fn, name, params, precomp,
             tb=0.010, tp=4, cd=0, mtd=10, conf_size=False,
             base_capital=100_000.0, max_capital=200_000.0,
             tod_start=dt.time(9,40), tod_end=dt.time(15,25)):
    (dcp_arr,drsi_arr,dbvc_arr), htf, m1 = precomp
    o_arr=f5m['open'].values; h_arr=f5m['high'].values
    l_arr=f5m['low'].values; c_arr=f5m['close'].values
    times=f5m.index; tod_arr=np.array([t.time() for t in times])
    n=len(f5m); trades=[]
    ctx={'daily_cp':0.0,'daily_rsi':50.0,'daily_bvc':0.0,**htf,**m1}
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
        ctx['daily_cp']=dcp_arr[i]; ctx['daily_rsi']=drsi_arr[i]; ctx['daily_bvc']=dbvc_arr[i]
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
    print(f"  {label}: {n}t {w}W/{n-w}L ({wr:.1f}%WR) ${total:+,.0f} BW=${bw:+,.0f} BL=${bl:+,.0f} aW=${aw:+,.0f} aL=${al:+,.0f}")
    print(f"    TRAIN:{len(train)}t {tw}W ${tp_:+,.0f} | TEST:{len(test)}t {tsw}W ${tsp:+,.0f} | OOS:{len(oos)}t {ow}W ${op:+,.0f}")
    if show_years:
        years=sorted(set(t[0].year for t in trades)); cum=0
        for yr in years:
            yt=[t for t in trades if t[0].year==yr]
            yw=sum(1 for t in yt if t[6]>0); yp=sum(t[6] for t in yt); ybl=min(t[6] for t in yt)
            cum+=yp; ywr=yw/len(yt)*100
            print(f"    {yr}: {len(yt):3d}t {ywr:5.1f}% ${yp:+8,.0f} BL=${ybl:+,.0f} cum=${cum:+,.0f}")
    _flush()

def report_short(trades, label):
    if not trades: print(f"  {label}: 0 trades"); _flush(); return
    n=len(trades); pnls=[t[6] for t in trades]
    w=sum(1 for p in pnls if p>0); wr=w/n*100; total=sum(pnls); bl=min(pnls)
    oos=[t for t in trades if t[0]>TEST_END]; on=len(oos)
    ow=sum(1 for t in oos if t[6]>0); op=sum(t[6] for t in oos)
    train=[t for t in trades if t[0]<=TRAIN_END]
    test=[t for t in trades if TRAIN_END<t[0]<=TEST_END]
    tp_=sum(t[6] for t in train); tsp=sum(t[6] for t in test)
    bal = tsp/tp_*100 if tp_>0 else 0
    print(f"  {label}: {n:5d}t {wr:5.1f}% ${total:+10,.0f} BL=${bl:+,.0f} "
          f"TR/TS={bal:.0f}% OOS:{on}t {ow}W ${op:+,.0f}")
    _flush()

# ================================================================
# Signal functions
# ================================================================
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
    # BVC filter
    min_bvc = params.get('min_bvc', -999)
    if min_bvc > -999:
        bvc1h = ctx['1h_bvc'][i]
        if not np.isnan(bvc1h) and bvc1h < min_bvc: return None
    # Daily RSI filter
    max_drsi = params.get('max_daily_rsi', 100)
    if max_drsi < 100:
        drsi = ctx['daily_rsi']
        if not np.isnan(drsi) and drsi > max_drsi: return None
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
    # BVC filter
    min_bvc = params.get('min_bvc', -999)
    if min_bvc > -999:
        bvc1h = ctx['1h_bvc'][i]
        if not np.isnan(bvc1h) and bvc1h < min_bvc: return None
    # Daily RSI filter
    max_drsi = params.get('max_daily_rsi', 100)
    if max_drsi < 100:
        drsi = ctx['daily_rsi']
        if not np.isnan(drsi) and drsi > max_drsi: return None
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

# ================================================================
def main():
    print("="*70); print("V9: FRONTIER EXPLORATION"); print("="*70); _flush()
    df1m = load_1min()
    features = build_features(df1m)
    f5m = features['5m']
    precomp = precompute_all(features, f5m)

    dp = {'stop':0.008, 'tp':0.020, 'd_min':0.20, 'h1_min':0.15, 'f5_thresh':0.25,
          'div_thresh':0.35, 'vwap_thresh':-0.30, 'need_turn':False, 'min_vol_ratio':0.8}

    # ── EXP 1: Wider VWAP/Div params on stacked ──
    print("\n"+"="*70)
    print("EXP 1: Stacked with wider params (more trades)")
    print("="*70); _flush()
    for vt in [-0.10, -0.20, -0.30, -0.50]:
        for dt_ in [0.20, 0.25, 0.30, 0.35]:
            for f5t in [0.25, 0.30, 0.35]:
                p = dict(dp, vwap_thresh=vt, div_thresh=dt_, f5_thresh=f5t)
                label = f"St vw<{vt:.1f} dv>{dt_:.2f} f5<{f5t:.2f}"
                trades = simulate(f5m, sig_stacked, label, p, precomp, conf_size=True)
                if trades and len(trades)>=50:
                    n=len(trades); pnls=[t[6] for t in trades]
                    w=sum(1 for pp in pnls if pp>0); wr=w/n*100; total=sum(pnls)
                    if total > 2000000 or (wr>=88 and total>1500000):
                        report_short(trades, label)

    # ── EXP 2: BVC (institutional flow) filter ──
    print("\n"+"="*70)
    print("EXP 2: BVC (Buy Volume Concentration) filter")
    print("="*70); _flush()
    for min_bvc in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3]:
        p = dict(dp, min_bvc=min_bvc)
        label = f"Stack bvc>{min_bvc:.1f}"
        trades = simulate(f5m, sig_stacked, label, p, precomp, conf_size=True)
        report_short(trades, label)

    # ── EXP 3: Daily RSI regime filter ──
    print("\n"+"="*70)
    print("EXP 3: Daily RSI regime filter")
    print("="*70); _flush()
    for max_rsi in [30, 40, 50, 60, 70, 80, 100]:
        p = dict(dp, max_daily_rsi=max_rsi)
        label = f"Stack dRSI<{max_rsi}"
        trades = simulate(f5m, sig_stacked, label, p, precomp, conf_size=True)
        report_short(trades, label)

    # ── EXP 4: Time-of-day analysis ──
    print("\n"+"="*70)
    print("EXP 4: Time-of-day windows")
    print("="*70); _flush()
    windows = [
        ("Morning 9:40-11:00", dt.time(9,40), dt.time(11,0)),
        ("MidDay 11:00-13:00", dt.time(11,0), dt.time(13,0)),
        ("Afternoon 13:00-15:25", dt.time(13,0), dt.time(15,25)),
        ("PowerHr 14:30-15:25", dt.time(14,30), dt.time(15,25)),
        ("Opening 9:40-10:30", dt.time(9,40), dt.time(10,30)),
        ("Full 9:40-15:25", dt.time(9,40), dt.time(15,25)),
        ("ExMorning 10:30-15:25", dt.time(10,30), dt.time(15,25)),
        ("MornPower 9:40-11:00+14:30-15:25", dt.time(9,40), dt.time(15,25)),  # handled separately
    ]
    p = dict(dp)
    for name, ts, te in windows:
        if name.startswith("MornPower"): continue  # skip compound
        trades = simulate(f5m, sig_stacked, name, p, precomp, conf_size=True,
                         tod_start=ts, tod_end=te)
        report_short(trades, name)

    # ── EXP 5: Conf sizing grid (base/max) ──
    print("\n"+"="*70)
    print("EXP 5: Confidence sizing grid")
    print("="*70); _flush()
    p = dict(dp)
    for base in [50_000, 75_000, 100_000, 150_000]:
        for mx in [150_000, 200_000, 300_000, 400_000]:
            if mx <= base: continue
            trades = simulate(f5m, sig_stacked, "grid", p, precomp,
                            conf_size=True, base_capital=float(base), max_capital=float(mx))
            if trades:
                n=len(trades); pnls=[t[6] for t in trades]
                w=sum(1 for pp in pnls if pp>0); wr=w/n*100; total=sum(pnls); bl=min(pnls)
                print(f"  base=${base//1000}K max=${mx//1000}K: {n}t {wr:.1f}% ${total:+,.0f} BL=${bl:+,.0f}")
                _flush()

    # ── EXP 6: Combined best filters ──
    print("\n"+"="*70)
    print("EXP 6: Combined best filters - full year-by-year")
    print("="*70); _flush()

    # Champion baseline
    print("\n--- Baseline: Stack N vr>0.8 m=10 SCALED ---"); _flush()
    p = dict(dp)
    trades = simulate(f5m, sig_stacked, "base", p, precomp, conf_size=True)
    report(trades, "BASELINE")

    # With BVC filter (if helpful from EXP 2)
    print("\n--- Stack + BVC>-0.1 ---"); _flush()
    p = dict(dp, min_bvc=-0.1)
    trades = simulate(f5m, sig_stacked, "bvc", p, precomp, conf_size=True)
    report(trades, "Stack + BVC>-0.1")

    # With daily RSI<60
    print("\n--- Stack + dRSI<60 ---"); _flush()
    p = dict(dp, max_daily_rsi=60)
    trades = simulate(f5m, sig_stacked, "rsi", p, precomp, conf_size=True)
    report(trades, "Stack + dRSI<60")

    # Wider params for more trades
    print("\n--- Stack wider: vw<-0.20 dv>0.25 f5<0.30 ---"); _flush()
    p = dict(dp, vwap_thresh=-0.20, div_thresh=0.25, f5_thresh=0.30)
    trades = simulate(f5m, sig_stacked, "wide", p, precomp, conf_size=True)
    report(trades, "Stack WIDER")

    # Higher scaling
    print("\n--- Stack vr>0.8 m=10 base=$75K max=$300K ---"); _flush()
    p = dict(dp)
    trades = simulate(f5m, sig_stacked, "big", p, precomp,
                     conf_size=True, base_capital=75_000.0, max_capital=300_000.0)
    report(trades, "Stack 75K-300K SCALED")

    print("\nDone."); _flush()

if __name__=='__main__':
    main()
