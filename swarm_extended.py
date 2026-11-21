import sys, json, time, copy
sys.path.insert(0,'repo')
import numpy as np
import curvtrust_pai_dgx_full as C
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from concurrent.futures import ProcessPoolExecutor

WSETS = {
 'full':        dict(id=.25,k=.35,r=.25,q=.15),
 'no_residual': dict(id=.25/.65,k=0,r=.25/.65,q=.15/.65),
 'no_innov':    dict(id=.25/.75,k=.35/.75,r=0,q=.15/.75),
 'no_isol':     dict(id=.25/.85,k=.35/.85,r=.25/.85,q=0),
 'no_identity': dict(id=0,k=.35/.75,r=.25/.75,q=.15/.75),
 'residual_only': dict(id=0,k=1.0,r=0,q=0),
}
def make_est(cfg, seed, W):
    class E(C.TrustEstimator): pass
    E.W=W
    sc = C.make_scenario(cfg, seed+1000, attack=False, mode='overt')
    rep = C._clean_reports(cfg, sc)
    return E(cfg).fit(*rep, seed=seed)

def run(args):
    seed, mode = args
    cfg = C.Cfg()
    out = dict(seed=seed, mode=mode, ctrl={}, abl={}, det={})
    est = make_est(cfg, seed, WSETS['full'])
    sc = C.make_scenario(cfg, seed, attack=True, mode=mode)
    # matched fixed margin 2.1 m  (lambda=0.8 -> large_d = 1.5*(1+0.8*0.5)... set so large_d=2.1)
    cfg21 = copy.deepcopy(cfg); cfg21.lambda_tau = 0.8
    cfg25 = copy.deepcopy(cfg); cfg25.lambda_tau = 1.333333333  # 2.5 m
    runs = [('nominal',cfg,est,'nominal',True),('fixed_1.5',cfg,None,'fixed',False),
            ('fixed_2.1',cfg21,None,'fixed_large',False),('fixed_2.5',cfg25,None,'fixed_large',False),
            ('fixed_3.0',cfg,None,'fixed_large',False),('trust',cfg,est,'trust',False)]
    log=None
    for name,c,e,ctl,lg in runs:
        r = C.simulate(c, sc, ctl, e, log_evidence=lg)
        if lg: log=r['log']
        r['log']=None; r.pop('qp',None)
        out['ctrl'][name]=r
    # trust-term ablation with closed-loop control
    for nm,W in WSETS.items():
        if nm in ('full','residual_only'): continue
        e2 = make_est(cfg, seed, W)
        r = C.simulate(cfg, sc, 'trust', e2, log_evidence=False); r['log']=None; r.pop('qp',None)
        out['abl'][nm]=r
    # open-loop detection AUCs on nominal log
    lab=log['lab'].ravel()
    ek,er,eq=est._n(log['rk'],'k'),est._n(log['rp'],'r'),est._n(log['rq'],'q'); sid=log['sid']
    d={}
    for nm,W in WSETS.items():
        e2 = make_est(cfg, seed, W)
        tau = e2.tau_from(sid,e2._n(log['rk'],'k'),e2._n(log['rp'],'r'),e2._n(log['rq'],'q'))
        d['auc_trust_'+nm]=C.auc_safe(lab,(1-tau).ravel())
        H=~log['lab'].astype(bool)
        d['tau_comp_'+nm]=float(tau[log['lab']==1].mean()); d['tau_hon_'+nm]=float(tau[H].mean())
    d['auc_resid']=C.auc_safe(lab,ek.ravel()); d['auc_innov']=C.auc_safe(lab,er.ravel()); d['auc_isol']=C.auc_safe(lab,eq.ravel())
    # reference-free / classic baselines on same reports (fit on clean calibration reports)
    sc0 = C.make_scenario(cfg, seed+1000, attack=False, mode='overt')
    rp,rv,ry = C._clean_reports(cfg, sc0)
    F0 = np.concatenate([rp,rv,ry],2).reshape(-1,14)
    F0s = np.clip(est.scaler.transform(F0),-C.CLIP,C.CLIP)
    Fl = log['F'].reshape(-1,14)
    rng=np.random.RandomState(seed); sub=rng.choice(len(F0s),min(5000,len(F0s)),replace=False)
    lof=LocalOutlierFactor(n_neighbors=20,novelty=True).fit(F0s[sub])
    d['auc_LOF']=C.auc_safe(lab,-lof.score_samples(Fl))
    nn=NearestNeighbors(n_neighbors=5).fit(F0s[sub]); dist=nn.kneighbors(Fl)[0].mean(1)
    d['auc_kNNdist']=C.auc_safe(lab,dist)
    kap=est.curv.kappa(Fl)
    d['auc_kappa_raw_low']=C.auc_safe(lab,-kap); d['auc_kappa_raw_high']=C.auc_safe(lab,kap)
    d['auc_idonly']=C.auc_safe(lab,(1-sid).ravel())
    out['det']=d
    # stealthier attacks (open loop)
    st={}
    for f in (1.0,0.5,0.25,0.125):
        c2=copy.deepcopy(cfg); c2.emb_poison=0.8*f; c2.spoof_bias=1.0*f; c2.spoof_sigma=0.10*f
        s2=C.make_scenario(c2,seed,attack=True,mode='authenticated')
        r=C.simulate(c2,s2,'nominal',est,log_evidence=True); lg=r['log']
        tau=lg['tau']; L=lg['lab']
        st[str(f)]=dict(auc=C.auc_safe(L.ravel(),(1-tau).ravel()),
            auc_resid=C.auc_safe(L.ravel(),est._n(lg['rk'],'k').ravel()),
            tau_comp=float(tau[L==1].mean()),tau_hon=float(tau[L==0].mean()))
    out['stealth']=st
    return out

if __name__=='__main__':
    seeds=list(range(42,42+int(sys.argv[1]))); res=[]
    jobs=[(s,m) for s in seeds for m in ('overt','authenticated')]
    t=time.time()
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(run,jobs):
            res.append(r); print('done',r['seed'],r['mode'],round(time.time()-t),flush=True)
            json.dump(res,open(sys.argv[2],'w'))
