import os
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""Recompute every number the revised paper quotes. Prints and writes numbers.json."""
import pandas as pd, numpy as np, json
from scipy import stats
S=R
# R is the project root. Handover layout: code in R/code; repo layout (tqe_local/): code in R itself.
OLD=os.path.join(R,'code') if os.path.isdir(os.path.join(R,'code')) else R
NL=pd.read_csv(f'{S}/data/cells_noiseless.csv'); NZ=pd.read_csv(f'{S}/data/cells_noisy.csv')
PK=pd.read_csv(f'{OLD}/outputs_f4/pathk_cells.csv'); G23=pd.concat([pd.read_csv(f'{OLD}/outputs_gap23/cells.csv'),pd.read_csv(f'{OLD}/outputs_hh_true/rot_cells.csv')])
MD=pd.read_csv(f'{OLD}/master_dataset.csv'); HHo=pd.read_csv(f'{OLD}/outputs_tqe/heavy_hex_cells.csv')
VQE=pd.concat([pd.read_csv(f'{OLD}/outputs_gap4/vqe_results.csv'),pd.read_csv(f'{OLD}/outputs_hh_true/vqe_results.csv')])
T=['brick','linear','heavy_hex','heavy_hex_true','star','all_to_all','ring']
def tab(df,metric):
    d=df[df.metric==metric]; return {(a,n,L):m for a,n,L,m in zip(d.ansatz,d.n_qubits,d.n_layers,d['mean'])}
def slope(x,y):
    x=np.log(np.array(x,float)); y=np.log(np.array(y,float)); r=stats.linregress(x,y); return r.slope, r.rvalue**2
def exps(g,topo,nrange=range(3,11),nb=8):
    al=[slope(list(nrange),[g[(topo,n,L)] for n in nrange])[0] for L in range(1,9)]
    a8,ra=slope(list(nrange),[g[(topo,n,8)] for n in nrange])
    b8,rb=slope(list(range(2,9)),[g[(topo,nb,L)] for L in range(2,9)])
    r=stats.linregress(range(2,9),al[1:])
    return dict(alpha=a8,alpha_r2=ra,beta=b8,beta_r2=rb,ratio=abs(a8/b8),dadl=r.slope,dadl_r2=r.rvalue**2,alpha_L=al)
out={}
gN=tab(NL,'grad_var_theta0')
out['main']={t:exps(gN,t) for t in T}
out['main_n12']={t:exps(gN,t,range(3,13)) for t in T}
out['G68']={t:gN[(t,6,8)] for t in T}
out['G88']={t:gN[(t,8,8)] for t in T}
# robustness: other diagnostics
for m in ['grad_var_z0_all','grad_var_local_all']:
    g=tab(NL,m); out['robust_'+m]={t:exps(g,t) for t in T}
gz=tab(NL,'grad_var_z0_all'); out['theta0_over_all_88']={t:gN[(t,8,8)]/gz[(t,8,8)] for t in T}
out['ring_L1_theta0_max']=max(gN[('ring',n,1)] for n in range(2,13))
out['ring_L1_z0all']={n:gz[('ring',n,1)] for n in range(2,13)}
# topology spread
sp={}
for nmax in (10,12):
    best=(0,None)
    for n in range(2,nmax+1):
        for L in range(2,9):
            v=[gN[(t,n,L)] for t in T if t!='heavy_hex']; r=max(v)/min(v)
            if r>best[0]: best=(r,(n,L))
    sp[nmax]=best
out['spread']={k:[v[0],v[1]] for k,v in sp.items()}
out['spread_68']=max(out['G68'][t] for t in T if t!='heavy_hex')/min(out['G68'][t] for t in T if t!='heavy_hex')
# expressivity
kr=tab(NL,'kernel_rank'); kl=tab(NL,'haar_kl_real'); fp=tab(NL,'frame_potential_ratio')
out['kernel_rank_88']={t:kr[(t,8,8)] for t in T}
d=NL[(NL.metric=='kernel_rank')&(NL.ansatz=='star')&(NL.n_qubits==8)&(NL.n_layers==8)]; out['star_kr_ci']=[float(d.lo.iloc[0]),float(d.hi.iloc[0])]
out['haar_kl_108']={t:kl[(t,10,8)] for t in T}; out['fp_108']={t:fp[(t,10,8)] for t in T}
# noise at n=6 L=8
gz6={nz:tab(NZ[NZ.noise==nz],'grad_var_theta0') for nz in NZ.noise.unique()}
P=['depolarizing_realistic','depolarizing_strong','amplitude_realistic','amplitude_strong']
out['noise_drop_68']={t:{p:100*(1-gz6[p][(t,6,8)]/gz6['noiseless'][(t,6,8)]) for p in P} for t in T}
allr=[]
for t in T:
  for n in range(2,7):
    for L in range(1,9):
      b=gz6['noiseless'][(t,n,L)]
      if b>1e-20:
        for p in P: allr.append((p,t,n,L,100*(1-gz6[p][(t,n,L)]/b)))
A=pd.DataFrame(allr,columns=['p','t','n','L','drop'])
out['noise_drop_grid']={p:[float(A[A.p==p]['drop'].min()),float(A[A.p==p]['drop'].max())] for p in P}
out['noise_increase_cells']=int((A['drop']< -0.5).sum())
out['dep_over_amp_strong_68']={t:out['noise_drop_68'][t]['depolarizing_strong']/out['noise_drop_68'][t]['amplitude_strong'] for t in T}
# path-k and classifier
gP=tab(PK,'grad_var'); pk=sorted(PK.ansatz.unique())
out['pathk']={t:exps(gP,t) for t in pk}
xs=[];ys=[];names=[]
for t in T:
    if t=='ring': continue
    e=out['main'][t]; xs.append(e['dadl']); ys.append(e['ratio']); names.append(t)
for t in pk: e=out['pathk'][t]; xs.append(e['dadl']); ys.append(e['ratio']); names.append(t)
r=stats.linregress(xs,ys); out['classifier']=dict(slope=r.slope,intercept=r.intercept,r2=r.rvalue**2,p=r.pvalue,n=len(xs),names=names,x=xs,y=ys,
   ring_pred=r.intercept+r.slope*out['main']['ring']['dadl'])
# without chord-chain (heavy_hex legacy) — paper should use one heavy-hex
xs2=[x for x,nm in zip(xs,names) if nm!='heavy_hex']; ys2=[y for y,nm in zip(ys,names) if nm!='heavy_hex']
r2=stats.linregress(xs2,ys2); out['classifier_noChord']=dict(slope=r2.slope,intercept=r2.intercept,r2=r2.rvalue**2,p=r2.pvalue,n=len(xs2))
# rotation richness (RX-RY-RZ, noiseless, legacy heavy-hex only)
gR=tab(G23,'grad_var'); RT=['brick','linear','heavy_hex_true','heavy_hex','star','all_to_all','ring']
out['rxryrz']={t:exps(gR,t) for t in RT}
out['amplification']={t:out['rxryrz'][t]['dadl']/out['main'][t]['dadl'] for t in RT}
# large n (RX-RY-RZ, L=8)
ln={}
for t in RT:
    a10,_=slope(range(3,11),[gR[(t,n,8)] for n in range(3,11)]); a16,_=slope(range(3,17),[gR[(t,n,8)] for n in range(3,17)])
    ln[t]=dict(a_le10=a10,a_le16=a16,G16=gR[(t,16,8)],G12=gR[(t,12,8)])
out['large_n']=ln
# effective dimension (old noiseless, valid)
ed=pd.concat([MD[(MD.metric=='eff_dim')&(MD.noise=='noiseless')],HHo[(HHo.metric=='eff_dim')&(HHo.noise=='noiseless')]])
ed=ed[ed.n_layers>=1]; rr=ed['mean']/ed['n_params']; out['eff_dim_over_p']=[float(rr.min()),float(rr.max())]
# VQE
v=VQE.groupby(['ansatz','config']).error_final.agg(['mean','sem']).reset_index()
gaps={};pv={}
for t in VQE.ansatz.unique():
    w=VQE[(VQE.ansatz==t)&(VQE.config=='wide')].error_final; dd=VQE[(VQE.ansatz==t)&(VQE.config=='deep')].error_final
    gaps[t]=float(w.mean()-dd.mean()); pv[t]=float(stats.ttest_ind(w,dd,equal_var=False).pvalue)
out['vqe_gap']=gaps; out['vqe_p']=pv; out['vqe_deep']={t:float(VQE[(VQE.ansatz==t)&(VQE.config=='deep')].error_final.mean()) for t in VQE.ansatz.unique()}
out['vqe_nseeds']=int(VQE.seed.nunique())
out['ring_G_83']=gN[('ring',8,3)]
# reviewer checks (Nov 2026): VQE on heavy-hex-edge Ising; heavy-hex measured on qubit 1 (degree 3)
RC=os.path.join(OLD,'outputs_reviewer_checks')
VH=pd.read_csv(f'{RC}/vqe_hh_hamiltonian.csv'); VH=VH[(VH.n_steps==300)&(VH.hamiltonian=='hh')]
out['vqehh_n']=int(len(VH)); out['vqehh_nseeds']=int(VH.seed.nunique()); out['vqehh_edges']={c:int(VH[VH.config==c].n_edges.iloc[0]) for c in ('wide','square','deep')}
out['vqehh_mean']={t:{c:float(VH[(VH.ansatz==t)&(VH.config==c)].error_final.mean()) for c in ('wide','square','deep')} for t in VH.ansatz.unique()}
gh={};ph={}
for t in VH.ansatz.unique():
    w=VH[(VH.ansatz==t)&(VH.config=='wide')].error_final; dd=VH[(VH.ansatz==t)&(VH.config=='deep')].error_final
    gh[t]=float(w.mean()-dd.mean()); ph[t]=float(stats.ttest_ind(w,dd,equal_var=False).pvalue)
out['vqehh_gap']=gh; out['vqehh_p']=ph
sh=[t for t in gh if t in out['vqe_gap']]
rs=stats.spearmanr([out['vqe_gap'][t] for t in sh],[gh[t] for t in sh])
out['vqe_gap_rank_corr']=dict(rho=float(rs.correlation),p=float(rs.pvalue),n=len(sh),topos=sh)
GQ=pd.read_csv(f'{RC}/grad_measured_qubit.csv'); GQ=GQ[GQ.n_runs==20]
gq={m:{('heavy_hex_true',n,L):x for n,L,x in zip(GQ[GQ.metric==m].n_qubits,GQ[GQ.metric==m].n_layers,GQ[GQ.metric==m]['mean'])} for m in GQ.metric.unique()}
out['q1check']={m:exps(gq[m],'heavy_hex_true') for m in gq}
out['q1check_n12']={m:exps(gq[m],'heavy_hex_true',range(3,13)) for m in gq}
out['q1check_degree']={int(n):int(d) for n,d in GQ[GQ.metric=='G_q1'].groupby('n_qubits').degree.first().items()}
c=out['classifier']; q1=out['q1check']['G_q1']
out['q1_classifier']=dict(pred=c['intercept']+c['slope']*q1['dadl'],obs=q1['ratio'],train_x_min=min(c['x']),train_x_max=max(c['x']),
   max_abs_resid_train=max(abs(y-(c['intercept']+c['slope']*x)) for x,y in zip(c['x'],c['y'])))
# share of Hamiltonian couplings (wide config, n=8, L=3) that coincide with a two-qubit gate pair of the ansatz
from collections import deque
_F=[(0,1),(1,2),(1,4),(2,3),(3,5),(4,7),(5,8),(6,7),(7,10),(8,9),(8,11),(10,12),(11,14),(12,13),(12,15),(13,14),(14,16),(15,18),(16,19),(17,18),(18,21),(19,20),(19,22),(21,23),(22,25),(23,24),(24,25),(25,26)]
def _hh(n):  # same BFS subgraph as code/ansatze.py::_falcon_subgraph
    adj={}
    for a,b in _F: adj.setdefault(a,[]).append(b); adj.setdefault(b,[]).append(a)
    order,seen,q=[],{0},deque([0])
    while q and len(order)<n:
        x=q.popleft(); order.append(x)
        for w in sorted(adj[x]):
            if w not in seen: seen.add(w); q.append(w)
    lab={x:i for i,x in enumerate(order)}; return [(lab[a],lab[b]) for a,b in _F if a in lab and b in lab]
def _gates(t,n,L):
    if t=='linear': return [(q,q+1) for q in range(n-1)]
    if t=='ring': return [(q,q+1) for q in range(n-1)]+[(n-1,0)]
    if t=='brick': return [(q,q+1) for l in range(L) for q in range(0 if l%2==0 else 1,n-1,2)]
    if t=='star': return [(0,j) for j in range(1,n)]
    if t=='heavy_hex': return [(i,i+1) for i in range(n-1)]+[(i,i+2) for i in range(0,n-2,3) if i+2<n]
    if t=='heavy_hex_true': return _hh(n)
_S=lambda E:{tuple(sorted(e)) for e in E}
_Hc=_S([(i,i+1) for i in range(7)]); _Hh=_S(_hh(8))
out['coupling_coverage']={H:{t:len(He&_S(_gates(t,8,3)))/len(He) for t in sh} for H,He in (('chain',_Hc),('hh',_Hh))}
_x=[out['coupling_coverage'][H][t] for H in ('chain','hh') for t in sh]; _y=[out['vqe_gap'][t] for t in sh]+[gh[t] for t in sh]
_r=stats.spearmanr(_x,_y); out['coverage_gap_corr']=dict(rho=float(_r.correlation),p=float(_r.pvalue),n=len(_x))
# relative wide-deep gap (error / |E_ground|): wide (n=8) and deep (n=4) are different problem sizes
def _relgap(D):
    D=D.assign(rel=D.error_final/D.E_ground.abs()); g={};p={}
    for t in D.ansatz.unique():
        w=D[(D.ansatz==t)&(D.config=='wide')].rel; dd=D[(D.ansatz==t)&(D.config=='deep')].rel
        g[t]=float(w.mean()-dd.mean()); p[t]=float(stats.ttest_ind(w,dd,equal_var=False).pvalue)
    return g,p
out['vqe_relgap'],out['vqe_relgap_p']=_relgap(VQE); out['vqehh_relgap'],out['vqehh_relgap_p']=_relgap(VH)
out['vqe_E0']={c:float(VQE[VQE.config==c].E_ground.iloc[0]) for c in ('wide','square','deep')}
out['vqehh_E0']={c:float(VH[VH.config==c].E_ground.iloc[0]) for c in ('wide','square','deep')}
out['q0_reproduces']=dict(dadl_diff=out['q1check']['G_q0']['dadl']-out['main']['heavy_hex_true']['dadl'],ratio_diff=out['q1check']['G_q0']['ratio']-out['main']['heavy_hex_true']['ratio'])
json.dump(out,open(f'{S}/numbers/numbers.json','w'),indent=1,default=float)
def f(x,k=3): return f'{x:.{k}f}'
print('MAIN (theta0, n=3..10)'); 
for t in T: e=out['main'][t]; print(f"{t:15s} a={e['alpha']:.2f} b={e['beta']:.2f} ratio={e['ratio']:.2f} dadl={e['dadl']:.3f} r2={e['dadl_r2']:.2f} a1={e['alpha_L'][0]:.2f} G68={out['G68'][t]:.4f}")
print('robust z0_all ratios',{t:round(out['robust_grad_var_z0_all'][t]['ratio'],2) for t in T})
print('robust local ratios',{t:round(out['robust_grad_var_local_all'][t]['ratio'],2) for t in T})
print('robust z0_all dadl',{t:round(out['robust_grad_var_z0_all'][t]['dadl'],3) for t in T})
print('theta0/all',{t:round(v,2) for t,v in out['theta0_over_all_88'].items()})
print('spread',out['spread'],'spread68',round(out['spread_68'],1))
print('pathk',{t:(round(e['dadl'],3),round(e['ratio'],2)) for t,e in out['pathk'].items()})
print('classifier',{k:v for k,v in out['classifier'].items() if k not in('names','x','y')}); print('noChord',out['classifier_noChord'])
print('rxryrz',{t:(round(e['dadl'],3),round(e['ratio'],2)) for t,e in out['rxryrz'].items()}); print('ampl',{t:round(v,2) for t,v in out['amplification'].items()})
print('large_n',{t:{k:round(v,4) for k,v in d.items()} for t,d in out['large_n'].items()})
print('noise68',{t:{p[:5]+p.split('_')[1][0]:round(v,1) for p,v in d.items()} for t,d in out['noise_drop_68'].items()})
print('noise grid',out['noise_drop_grid'],'increase cells',out['noise_increase_cells']); print('dep/amp',{t:round(v,2) for t,v in out['dep_over_amp_strong_68'].items()})
print('kr',{t:round(v,1) for t,v in out['kernel_rank_88'].items()},out['star_kr_ci']); print('kl',{t:round(v,4) for t,v in out['haar_kl_108'].items()}); print('fp',{t:round(v,3) for t,v in out['fp_108'].items()})
print('effdim/p',out['eff_dim_over_p']); print('vqe gap',out['vqe_gap'],'p',out['vqe_p'],'deep',out['vqe_deep'],out['vqe_nseeds'],'ringG83',out['ring_G_83'])
print('vqe hh gap',{t:round(v,3) for t,v in out['vqehh_gap'].items()},'p',{t:float(f'{v:.2g}') for t,v in out['vqehh_p'].items()})
print('vqe hh means',{t:{c:round(x,4) for c,x in d.items()} for t,d in out['vqehh_mean'].items()},out['vqehh_edges'],out['vqehh_n'],out['vqehh_nseeds'])
print('gap rank corr',out['vqe_gap_rank_corr'])
print('q1check',{m:(round(e['dadl'],3),round(e['ratio'],2),round(e['alpha'],2),round(e['beta'],2),round(e['dadl_r2'],2)) for m,e in out['q1check'].items()})
print('q1check n12 alpha',{m:round(e['alpha'],2) for m,e in out['q1check_n12'].items()},'degree',out['q1check_degree'])
print('q1 classifier',out['q1_classifier'],'q0 repro',out['q0_reproduces'])
print('coverage',{H:{t:round(v,2) for t,v in d.items()} for H,d in out['coupling_coverage'].items()},out['coverage_gap_corr'])
print('relgap chain',{t:(round(v,4),float(f'{out["vqe_relgap_p"][t]:.2g}')) for t,v in out['vqe_relgap'].items()}); print('relgap hh',{t:(round(v,4),float(f'{out["vqehh_relgap_p"][t]:.2g}')) for t,v in out['vqehh_relgap'].items()}); print('E0',out['vqe_E0'],out['vqehh_E0'])
