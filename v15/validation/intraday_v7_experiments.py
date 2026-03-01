#!/usr/bin/env python3
"""V7: Profit maximization experiments.
Key insight: profit scales linearly with trade count when WR is stable.
Tests: VWAP mtd=5, stacked signals, confidence sizing, exhaustion union,
       wider VWAP + tight channel, volume confirmation, TOD filtering.
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
    u=fl+2*sr; l=fl-2*sr; wi=u-l
    pos=np.full(n,np.nan); v=(wi>1e-10)&~np.isnan(wi)
    pos[v]=(close[v]-l[v])/wi[v]
    return np.clip(pos,0.0,1.0)

def compute_rsi(c,p=14):
    n=len(c); r=np.full(n,np.nan); d=np.diff(c)
    if len(d)<p: return r
    g=np.maximum(d,0); l=np.maximum(-d,0)
    ag=g[:p].mean(); al=l[:p].mean()
    for i in range(p,len(d)):
        ag=(ag*(p-1)+g[i])/p; al=(al*(p-1)+l[i])/p
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
    """Ratio of current volume to rolling mean."""
    n=len(v); vr=np.full(n,np.nan)
    rv=pd.Series(v).rolling(lb,min_periods=lb).mean().values
    valid=(rv>0)&~np.isnan(rv)
    vr[valid]=v[valid]/rv[valid]
    return vr

def compute_atr_pct(h,l,c,p=14):
    """ATR as percentage of close."""
    n=len(c); atr=np.full(n,np.nan)
    tr=np.zeros(n)
    tr[0]=h[0]-l[0]
    for i in range(1,n):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    s=tr[:p].mean()
    if p<n: atr[p-1]=s/c[p-1]*100.0 if c[p-1]>0 else 0
    for i in range(p,n):
        s=(s*(p-1)+tr[i])/p
        atr[i]=s/c[i]*100.0 if c[i]>0 else 0
    return atr

def compute_rsi_fast(c, p=5):
    """Fast RSI for 5-min bars."""
    return compute_rsi(c, p)

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
        if name!='1m':
            feat['rsi']=compute_rsi(c); feat['bvc']=compute_bvc(c,v)
        if name in ('5m','3m'): feat['mom_turn']=compute_mom_turn(c)
        if name=='5m':
            dates=np.array([t.date() for t in df.index])
            vw,vd=compute_vwap(o,h,l,c,v,dates)
            feat['vwap']=vw; feat['vwap_dist']=vd
            feat['vol_ratio']=compute_volume_ratio(v)
            feat['atr_pct']=compute_atr_pct(h,l,c)
            feat['rsi5']=compute_rsi_fast(c,5)
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

# ================================================================
# Simulator with confidence sizing option
# ================================================================
def simulate(f5m, signal_fn, name, params, precomp,
             tb=0.010, tp=4, cd=6, mtd=2, conf_size=False,
             base_capital=100_000.0, max_capital=200_000.0):
    (dcp_arr,_,_), htf, m1 = precomp
    o_arr=f5m['open'].values; h_arr=f5m['high'].values
    l_arr=f5m['low'].values; c_arr=f5m['close'].values
    times=f5m.index; tod_arr=np.array([t.time() for t in times])
    n=len(f5m); trades=[]
    ctx={'daily_cp':0.0,**htf,**m1}
    in_trade=False; ep=et=None; conf=sp=tpp=bp=0.0; hb=cr=tt=0; cd_=None; ps=None
    tes=dt.time(9,35); tee=dt.time(15,30); tss=dt.time(9,40); tse=dt.time(15,25); tfe=dt.time(15,50)
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
                if conf_size:
                    cap = base_capital + (max_capital - base_capital) * conf
                else:
                    cap = CAPITAL
                sh=max(1,int(cap*conf/ep))
                pnl=(xa-ep)*sh-COMM_PER_SHARE*sh*2
                trades.append((et,bt,ep,xa,conf,sh,pnl,hb,xr,name))
                in_trade=False; cr=cd
        if cr>0: cr-=1; continue
        if in_trade or tt>=mtd: continue
        if btod<tss or btod>tse: continue
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
    f=" ***" if wr>=90 else ""
    print(f"  {label}: {n}t {w}W/{n-w}L ({wr:.1f}%WR) ${total:+,.0f} BW=${bw:+,.0f} BL=${bl:+,.0f} aW=${aw:+,.0f} aL=${al:+,.0f}{f}")
    print(f"    TRAIN:{len(train)}t {tw}W ${tp_:+,.0f} | TEST:{len(test)}t {tsw}W ${tsp:+,.0f} | OOS:{len(oos)}t {ow}W ${op:+,.0f}")
    if show_years:
        years=sorted(set(t[0].year for t in trades)); cum=0
        for yr in years:
            yt=[t for t in trades if t[0].year==yr]
            yw=sum(1 for t in yt if t[6]>0); yp=sum(t[6] for t in yt); ybl=min(t[6] for t in yt)
            cum+=yp; ywr=yw/len(yt)*100
            f=" ***" if ywr>=90 else ""
            print(f"    {yr}: {len(yt):3d}t {ywr:5.1f}% ${yp:+8,.0f} BL=${ybl:+,.0f} cum=${cum:+,.0f}{f}")
    _flush()

def report_short(trades, label):
    if not trades: print(f"  {label}: 0 trades"); _flush(); return
    n=len(trades); pnls=[t[6] for t in trades]
    w=sum(1 for p in pnls if p>0); wr=w/n*100; total=sum(pnls); bl=min(pnls)
    oos=[t for t in trades if t[0]>TEST_END]; on=len(oos)
    ow=sum(1 for t in oos if t[6]>0); op=sum(t[6] for t in oos)
    print(f"  {label}: {n:5d}t {wr:5.1f}% ${total:+10,.0f} BL=${bl:+,.0f} OOS:{on}t {ow}W ${op:+,.0f}")
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
    if params.get('need_turn',True):
        t5=f5m.get('mom_turn'); ht5=t5 is not None and t5.iloc[i]
        ht3=ctx['3m_turn'][i]>0.5; hr1=ctx['1m_rev'][i]>0.5; hr3=ctx['3m_reversal'][i]>0.5
        if not (ht5 or ht3 or hr1 or hr3): return None
    s=params.get('stop',0.008); t=params.get('tp',0.020)
    conf=0.55+min(abs(vd)*0.05,0.15)+0.10*(1.0-cp5)
    return ('LONG',min(conf,0.95),s,t)

def sig_div(i, f5m, ctx, params):
    cp5=f5m['chan_pos'].iloc[i]
    if np.isnan(cp5): return None
    dcp=ctx['daily_cp']; hcp=ctx['1h_cp'][i]; h4cp=ctx['4h_cp'][i]
    if np.isnan(dcp) or np.isnan(hcp) or np.isnan(h4cp): return None
    ha=dcp*0.35+h4cp*0.35+hcp*0.30; div=ha-cp5
    if div<params.get('div_thresh',0.35): return None
    if cp5>params.get('f5_thresh',0.30): return None
    if params.get('need_turn',True):
        t5=f5m.get('mom_turn'); ht5=t5 is not None and t5.iloc[i]
        ht3=ctx['3m_turn'][i]>0.5; hr1=ctx['1m_rev'][i]>0.5; hr3=ctx['3m_reversal'][i]>0.5
        if not (ht5 or ht3 or hr1 or hr3): return None
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

def sig_stacked(i, f5m, ctx, params):
    """Only fire when BOTH VWAP and Div agree."""
    rv = sig_vwap(i, f5m, ctx, params)
    rd = sig_div(i, f5m, ctx, params)
    if rv is None or rd is None: return None
    # Both agree - boost confidence
    conf = min(0.95, max(rv[1], rd[1]) + 0.05)
    return ('LONG', conf, rv[2], rv[3])

def sig_vwap_vol(i, f5m, ctx, params):
    """VWAP signal + volume confirmation."""
    r = sig_vwap(i, f5m, ctx, params)
    if r is None: return None
    vr = f5m['vol_ratio'].iloc[i] if 'vol_ratio' in f5m.columns else np.nan
    if np.isnan(vr): return r  # no vol data, pass through
    min_vr = params.get('min_vol_ratio', 0.8)
    if vr < min_vr: return None  # too quiet, skip
    # Boost confidence on high volume
    if vr > 1.5:
        conf = min(0.95, r[1] + 0.05)
        return ('LONG', conf, r[2], r[3])
    return r

def sig_vwap_rsi(i, f5m, ctx, params):
    """VWAP + RSI5 oversold confirmation."""
    r = sig_vwap(i, f5m, ctx, params)
    if r is None: return None
    rsi5 = f5m['rsi5'].iloc[i] if 'rsi5' in f5m.columns else np.nan
    if np.isnan(rsi5): return r
    max_rsi = params.get('max_rsi5', 35)
    if rsi5 > max_rsi: return None  # not oversold enough
    # Lower RSI = higher confidence
    boost = min(0.10, (max_rsi - rsi5) / 100.0)
    conf = min(0.95, r[1] + boost)
    return ('LONG', conf, r[2], r[3])

def sig_vwap_atr(i, f5m, ctx, params):
    """VWAP signal filtered by ATR regime (low volatility = safer bounce)."""
    r = sig_vwap(i, f5m, ctx, params)
    if r is None: return None
    atr = f5m['atr_pct'].iloc[i] if 'atr_pct' in f5m.columns else np.nan
    if np.isnan(atr): return r
    max_atr = params.get('max_atr_pct', 3.0)
    if atr > max_atr: return None  # too volatile
    return r

def sig_multi_tf_confirm(i, f5m, ctx, params):
    """VWAP + multi-TF channel position agreement.
    Require 15m, 30m, 1h channels all above threshold."""
    r = sig_vwap(i, f5m, ctx, params)
    if r is None: return None
    min_htf = params.get('min_htf_cp', 0.25)
    cp15 = ctx['15m_cp'][i]; cp30 = ctx['30m_cp'][i]; cp1h = ctx['1h_cp'][i]
    above = 0
    if not np.isnan(cp15) and cp15 >= min_htf: above += 1
    if not np.isnan(cp30) and cp30 >= min_htf: above += 1
    if not np.isnan(cp1h) and cp1h >= min_htf: above += 1
    min_count = params.get('min_htf_count', 2)
    if above < min_count: return None
    # More TF agreement = higher conf
    conf = min(0.95, r[1] + 0.03 * above)
    return ('LONG', conf, r[2], r[3])

def sig_exhaustion(i, f5m, ctx, params):
    """Volume exhaustion bounce: declining volume on consecutive down bars."""
    cp5 = f5m['chan_pos'].iloc[i]
    if np.isnan(cp5) or cp5 > 0.30: return None
    dcp = ctx['daily_cp']
    if np.isnan(dcp) or dcp < 0.15: return None
    # Check last 3-5 bars for declining volume on down moves
    if i < 5: return None
    c = f5m['close'].values; v = f5m['volume'].values
    down_count = 0; vol_declining = True
    for j in range(1, min(6, i+1)):
        if c[i-j+1] < c[i-j]:
            down_count += 1
            if j >= 2 and v[i-j+1] >= v[i-j]: vol_declining = False
        else:
            break
    if down_count < params.get('min_down_bars', 3): return None
    if not vol_declining: return None
    # Micro turn confirmation
    if params.get('need_turn', True):
        ht3 = ctx['3m_turn'][i] > 0.5; hr1 = ctx['1m_rev'][i] > 0.5
        if not (ht3 or hr1): return None
    s = params.get('stop', 0.008); t = params.get('tp', 0.020)
    conf = 0.60 + 0.05 * down_count + 0.10 * (1.0 - cp5)
    return ('LONG', min(conf, 0.95), s, t)

def sig_mega_union(i, f5m, ctx, params):
    """Union of VWAP + Div + Exhaustion."""
    best = None; bc = 0
    for fn in [sig_vwap, sig_div, sig_exhaustion]:
        r = fn(i, f5m, ctx, params)
        if r and r[1] > bc: best = r; bc = r[1]
    return best

# ================================================================
def main():
    print("="*70); print("V7: PROFIT MAXIMIZATION EXPERIMENTS"); print("="*70); _flush()
    df1m = load_1min()
    features = build_features(df1m)
    f5m = features['5m']
    precomp = precompute_all(features, f5m)

    dp = {'stop':0.008, 'tp':0.020, 'd_min':0.20, 'h1_min':0.15, 'f5_thresh':0.25,
          'div_thresh':0.35, 'vwap_thresh':-0.30, 'need_turn':True}

    # ── EXP 1: VWAP with mtd sweep (key profit driver) ──
    print("\n"+"="*70)
    print("EXP 1: VWAP-only with turn, varying trades/day (cd=0)")
    print("="*70); _flush()
    for mtd in [1,2,3,5,8,10]:
        for nt in [True, False]:
            p = dict(dp, need_turn=nt)
            nm = f"VWAP {'T' if nt else 'N'} mtd={mtd}"
            trades = simulate(f5m, sig_vwap, nm, p, precomp, cd=0, mtd=mtd)
            report_short(trades, nm)

    # ── EXP 2: Stacked signals (VWAP + Div both fire) ──
    print("\n"+"="*70)
    print("EXP 2: Stacked (VWAP AND Div must both fire)")
    print("="*70); _flush()
    for nt in [True, False]:
        for cd in [0, 3, 6]:
            for mtd in [2, 3, 5]:
                p = dict(dp, need_turn=nt)
                nm = f"Stacked {'T' if nt else 'N'} cd={cd} mtd={mtd}"
                trades = simulate(f5m, sig_stacked, nm, p, precomp, cd=cd, mtd=mtd)
                report_short(trades, nm)

    # ── EXP 3: Confidence-scaled sizing ──
    print("\n"+"="*70)
    print("EXP 3: Confidence-scaled sizing (base $100K, max $200K)")
    print("="*70); _flush()
    # Compare: fixed $100K vs confidence-scaled
    for mtd in [2, 5]:
        for nt in [True, False]:
            p = dict(dp, need_turn=nt)
            nm_fix = f"Union {'T' if nt else 'N'} mtd={mtd} fixed"
            nm_scl = f"Union {'T' if nt else 'N'} mtd={mtd} scaled"
            t_fix = simulate(f5m, sig_union, nm_fix, p, precomp, cd=0, mtd=mtd, conf_size=False)
            t_scl = simulate(f5m, sig_union, nm_scl, p, precomp, cd=0, mtd=mtd, conf_size=True)
            report_short(t_fix, nm_fix)
            report_short(t_scl, nm_scl)

    # ── EXP 4: VWAP + Volume filter ──
    print("\n"+"="*70)
    print("EXP 4: VWAP + Volume ratio filter")
    print("="*70); _flush()
    for min_vr in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        p = dict(dp, need_turn=True, min_vol_ratio=min_vr)
        nm = f"VWAP_vol vr>{min_vr:.1f}"
        trades = simulate(f5m, sig_vwap_vol, nm, p, precomp, cd=0, mtd=5)
        report_short(trades, nm)
    p = dict(dp, need_turn=False, min_vol_ratio=1.0)
    trades = simulate(f5m, sig_vwap_vol, "VWAP_vol N vr>1.0", p, precomp, cd=0, mtd=5)
    report_short(trades, "VWAP_vol N vr>1.0")

    # ── EXP 5: VWAP + RSI5 oversold ──
    print("\n"+"="*70)
    print("EXP 5: VWAP + RSI5 oversold filter")
    print("="*70); _flush()
    for max_rsi in [20, 25, 30, 35, 40, 50, 60, 100]:
        p = dict(dp, need_turn=True, max_rsi5=max_rsi)
        nm = f"VWAP_rsi rsi5<{max_rsi}"
        trades = simulate(f5m, sig_vwap_rsi, nm, p, precomp, cd=0, mtd=5)
        report_short(trades, nm)

    # ── EXP 6: VWAP + ATR regime filter ──
    print("\n"+"="*70)
    print("EXP 6: VWAP + ATR regime filter (low vol = safer)")
    print("="*70); _flush()
    for max_atr in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]:
        p = dict(dp, need_turn=True, max_atr_pct=max_atr)
        nm = f"VWAP_atr atr<{max_atr:.1f}"
        trades = simulate(f5m, sig_vwap_atr, nm, p, precomp, cd=0, mtd=5)
        report_short(trades, nm)

    # ── EXP 7: Multi-TF channel confirmation ──
    print("\n"+"="*70)
    print("EXP 7: VWAP + multi-TF channel confirmation")
    print("="*70); _flush()
    for min_htf in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        for min_cnt in [1, 2, 3]:
            p = dict(dp, need_turn=True, min_htf_cp=min_htf, min_htf_count=min_cnt)
            nm = f"VWAP_mtf htf>{min_htf:.2f} cnt>={min_cnt}"
            trades = simulate(f5m, sig_multi_tf_confirm, nm, p, precomp, cd=0, mtd=5)
            report_short(trades, nm)

    # ── EXP 8: Exhaustion bounce + Mega union ──
    print("\n"+"="*70)
    print("EXP 8: Exhaustion bounce standalone + mega union")
    print("="*70); _flush()
    for min_db in [2, 3, 4]:
        for nt in [True, False]:
            p = dict(dp, need_turn=nt, min_down_bars=min_db)
            nm = f"Exhaust db>={min_db} {'T' if nt else 'N'}"
            trades = simulate(f5m, sig_exhaustion, nm, p, precomp, cd=0, mtd=5)
            report_short(trades, nm)

    # Mega union
    for mtd in [3, 5, 8]:
        for nt in [True, False]:
            p = dict(dp, need_turn=nt)
            nm = f"MegaUnion {'T' if nt else 'N'} mtd={mtd}"
            trades = simulate(f5m, sig_mega_union, nm, p, precomp, cd=0, mtd=mtd)
            report_short(trades, nm)

    # ── EXP 9: BEST CONFIG CANDIDATES (full report) ──
    print("\n"+"="*70)
    print("EXP 9: Best candidates - full year-by-year")
    print("="*70); _flush()

    # A: VWAP noTurn mtd=5
    p = dict(dp, need_turn=False)
    trades = simulate(f5m, sig_vwap, "VWAP_noTurn_mtd5", p, precomp, cd=0, mtd=5)
    report(trades, "VWAP noTurn cd=0 mtd=5")

    # B: Union noTurn mtd=5
    trades = simulate(f5m, sig_union, "Union_noTurn_mtd5", p, precomp, cd=0, mtd=5)
    report(trades, "Union noTurn cd=0 mtd=5")

    # C: Union turn mtd=5
    p = dict(dp, need_turn=True)
    trades = simulate(f5m, sig_union, "Union_turn_mtd5", p, precomp, cd=0, mtd=5)
    report(trades, "Union turn cd=0 mtd=5")

    # D: Mega union noTurn mtd=5
    p = dict(dp, need_turn=False)
    trades = simulate(f5m, sig_mega_union, "Mega_noTurn_mtd5", p, precomp, cd=0, mtd=5)
    report(trades, "MegaUnion noTurn cd=0 mtd=5")

    # E: VWAP noTurn mtd=5 with confidence sizing
    trades = simulate(f5m, sig_vwap, "VWAP_noTurn_mtd5_scaled", p, precomp,
                      cd=0, mtd=5, conf_size=True)
    report(trades, "VWAP noTurn cd=0 mtd=5 CONF-SIZED")

    # F: Union turn mtd=5 with confidence sizing
    p = dict(dp, need_turn=True)
    trades = simulate(f5m, sig_union, "Union_turn_mtd5_scaled", p, precomp,
                      cd=0, mtd=5, conf_size=True)
    report(trades, "Union turn cd=0 mtd=5 CONF-SIZED")

    print("\nDone."); _flush()

if __name__=='__main__':
    main()
