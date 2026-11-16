import os
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
S=R; OLD=os.path.join(R,'code') if os.path.isdir(os.path.join(R,'code')) else R  # handover vs repo layout
FD=os.environ.get('FIG_DIR') or (f'{S}/paper_ieee_access/figures' if os.path.isdir(f'{S}/paper_ieee_access') else f'{S}/figures')
os.makedirs(FD,exist_ok=True)
J=json.load(open(f'{S}/numbers/numbers.json'))
NL=pd.read_csv(f'{S}/data/cells_noiseless.csv'); NZ=pd.read_csv(f'{S}/data/cells_noisy.csv'); G23=pd.concat([pd.read_csv(f'{OLD}/outputs_gap23/cells.csv'),pd.read_csv(f'{OLD}/outputs_hh_true/rot_cells.csv')])
MD=pd.read_csv(f'{OLD}/master_dataset.csv'); HHo=pd.read_csv(f'{OLD}/outputs_tqe/heavy_hex_cells.csv'); VQE=pd.concat([pd.read_csv(f'{OLD}/outputs_gap4/vqe_results.csv'),pd.read_csv(f'{OLD}/outputs_hh_true/vqe_results.csv')])
T=['brick','linear','heavy_hex','heavy_hex_true','star','all_to_all','ring']
LAB={'brick':'Brick','linear':'Linear','heavy_hex':'Chord-chain','heavy_hex_true':'Heavy-hex (IBM)','star':'Star','all_to_all':'All-to-all','ring':'Ring',
     'path_2':'Path-2','path_3':'Path-3','path_4':'Path-4','path_5':'Path-5','path_half':'Path-n/2'}
COL={'brick':'#1b9e77','linear':'#d95f02','heavy_hex':'#7570b3','heavy_hex_true':'#000000','star':'#e7298a','all_to_all':'#66a61e','ring':'#e6ab02'}
plt.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'legend.fontsize':6.5,'figure.dpi':150})
def tab(df,metric):
    d=df[df.metric==metric]; return {(a,n,L):m for a,n,L,m in zip(d.ansatz,d.n_qubits,d.n_layers,d['mean'])}
gN=tab(NL,'grad_var_theta0')
# Fig 2 heatmaps
fig,ax=plt.subplots(2,4,figsize=(7.2,3.8)); ax=ax.ravel()
for i,t in enumerate(T):
    M=np.array([[gN[(t,n,L)] for L in range(1,9)] for n in range(2,13)])
    im=ax[i].imshow(np.log10(np.clip(M,1e-5,None)),origin='lower',aspect='auto',cmap='magma',vmin=-5,vmax=-0.5,extent=[0.5,8.5,1.5,12.5])
    ax[i].set_title(LAB[t]); ax[i].set_xlabel('L'); ax[i].set_ylabel('n') if i%4==0 else None
ax[7].axis('off'); fig.colorbar(im,ax=ax[7],fraction=0.6,label=r'$\log_{10}\mathcal{G}$ (clipped at $-5$)')
fig.tight_layout(); fig.savefig(f'{FD}/fig2_heatmaps.pdf'); plt.close()
# alpha vs depth
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6))
for t in T: ax[0].plot(range(1,9),J['main'][t]['alpha_L'],'o-',ms=3,color=COL[t],label=LAB[t])
ax[0].set_xlabel('Depth L'); ax[0].set_ylabel(r'$\hat\alpha(L)$'); ax[0].legend(ncol=2); ax[0].set_title('(a) Width exponent vs depth')
x=np.arange(len(T)); ax[1].bar(x,[J['main'][t]['dadl'] for t in T],color=[COL[t] for t in T])
ax[1].set_xticks(x); ax[1].set_xticklabels([LAB[t] for t in T],rotation=40,ha='right'); ax[1].set_ylabel(r'$d\hat\alpha/dL$ ($L=2$–8)'); ax[1].set_title('(b) Alpha growth rate')
fig.tight_layout(); fig.savefig(f'{FD}/fig_alpha_vs_depth.pdf'); plt.close()
# Fig 1 asymmetry
fig,ax=plt.subplots(1,3,figsize=(7.2,2.5))
for k,(key,tt) in enumerate([('alpha',r'(A) $\hat\alpha$ ($L=8$)'),('beta',r'(B) $\hat\beta$ ($n=8$)'),('ratio',r'(C) $|\hat\alpha/\hat\beta|$')]):
    v=[J['main'][t][key] for t in T]; ax[k].bar(x,v,color=[COL[t] for t in T]); ax[k].set_title(tt)
    ax[k].set_xticks(x); ax[k].set_xticklabels([LAB[t] for t in T],rotation=50,ha='right')
ax[2].axhline(1,ls='--',c='gray',lw=1); ax[2].set_yscale('log')
fig.tight_layout(); fig.savefig(f'{FD}/fig1_asymmetry_direction.pdf'); plt.close()
# Fig 4 noise
P=['depolarizing_realistic','amplitude_realistic','depolarizing_strong','amplitude_strong']; PL=['Dep. $10^{-3}$','Amp. $10^{-3}$','Dep. $10^{-2}$','Amp. $10^{-2}$']
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7)); w=0.2
for j,p in enumerate(P): ax[0].bar(x+(j-1.5)*w,[J['noise_drop_68'][t][p] for t in T],w,label=PL[j])
ax[0].set_xticks(x); ax[0].set_xticklabels([LAB[t] for t in T],rotation=40,ha='right'); ax[0].set_ylabel(r'Drop in $\mathcal{G}$ (%)'); ax[0].legend(); ax[0].set_title('(A) Per-layer noise, $n=6$, $L=8$')
for t in T: ax[1].scatter(J['noise_drop_68'][t]['amplitude_strong'],J['noise_drop_68'][t]['depolarizing_strong'],color=COL[t],label=LAB[t],s=18)
lim=[0,65]; ax[1].plot(lim,lim,'--',c='gray',lw=1,label='equal'); ax[1].plot(lim,[1.6*l for l in lim],':',c='k',lw=1,label=r'$1.6\times$')
ax[1].set_xlim(0,40); ax[1].set_ylim(0,65); ax[1].set_xlabel('Amplitude damping drop (%)'); ax[1].set_ylabel('Depolarizing drop (%)'); ax[1].set_title(r'(B) Strong rate ($10^{-2}$)'); ax[1].legend(ncol=2)
fig.tight_layout(); fig.savefig(f'{FD}/fig4_noise_impact.pdf'); plt.close()
# Fig 3: expressivity vs trainability (noiseless)
ed=pd.concat([MD[(MD.metric=='eff_dim')&(MD.noise=='noiseless')],HHo[(HHo.metric=='eff_dim')&(HHo.noise=='noiseless')]])
fp=tab(NL,'frame_potential_ratio')
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7))
for t in ['brick','linear','heavy_hex','star','all_to_all','ring']:
    d=ed[ed.ansatz==t]; xs=[np.log10(max(gN[(t,n,L)],1e-12)) for n,L in zip(d.n_qubits,d.n_layers)]
    ax[0].scatter(xs,d['mean']/d['n_params'],s=5,color=COL[t],label=LAB[t],alpha=.7)
ax[0].set_xlabel(r'$\log_{10}\mathcal{G}$'); ax[0].set_ylabel(r'$d_{\rm eff}/p$'); ax[0].set_title(r'(A) Effective dimension ($n\leq 10$)'); ax[0].legend(ncol=2); ax[0].set_xlim(-6,0)
for t in T:
    pts=[(np.log10(max(gN[(t,n,L)],1e-12)),fp[(t,n,L)]) for n in range(2,13) for L in range(1,9)]
    ax[1].scatter(*zip(*pts),s=5,color=COL[t],label=LAB[t],alpha=.7)
ax[1].set_yscale('log'); ax[1].set_xlim(-6,0); ax[1].axhline(1,ls='--',c='gray',lw=1); ax[1].set_xlabel(r'$\log_{10}\mathcal{G}$'); ax[1].set_ylabel('Frame-potential ratio'); ax[1].set_title(r'(B) Frame potential ($n\leq 12$; 1 = real 2-design)')
fig.tight_layout(); fig.savefig(f'{FD}/fig3_pareto_frontier.pdf'); plt.close()
# Fig 5: parameter efficiency
kr=tab(NL,'kernel_rank')
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6))
for t in T:
    pts=sorted([(n*L,kr[(t,n,L)]/64/(n*L)) for n in range(2,13) for L in range(1,9)])
    df=pd.DataFrame(pts,columns=['p','e']).groupby('p').e.max(); ax[0].plot(df.index,df.values,'-',lw=1,color=COL[t],label=LAB[t])
    ax[1].plot(range(2,13),[kr[(t,n,8)]/64 for n in range(2,13)],'o-',ms=3,color=COL[t],label=LAB[t])
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('Parameters p'); ax[0].set_ylabel('(rank/64) per parameter (best config.)'); ax[0].set_title('(A) Kernel-rank efficiency'); ax[0].legend(ncol=2)
ax[1].set_xlabel('n (L=8)'); ax[1].set_ylabel('Effective rank / 64'); ax[1].set_title('(B) Normalised kernel rank')
fig.tight_layout(); fig.savefig(f'{FD}/fig5_param_efficiency.pdf'); plt.close()
# classifier
C=J['classifier']; fig,ax=plt.subplots(1,2,figsize=(7.2,2.7))
for nm,xx,yy in zip(C['names'],C['x'],C['y']):
    ax[0].scatter(xx,yy,s=22,color=COL.get(nm,'#999999'),marker='o' if nm in COL else 's'); ax[0].annotate(LAB[nm],(xx,yy),fontsize=5.5,xytext=(3,2),textcoords='offset points')
xx=np.linspace(-0.5,0.0,10); ax[0].plot(xx,C['intercept']+C['slope']*xx,'--',c='gray',lw=1)
ax[0].axhline(1,ls=':',c='k',lw=.8); ax[0].set_xlabel(r'$d\hat\alpha/dL$'); ax[0].set_ylabel(r'$|\hat\alpha/\hat\beta|$')
ax[0].set_title(f"(a) Regime classifier: $R^2={C['r2']:.3f}$, $p={C['p']:.1e}$ (11 non-ring)")
for t in ['brick','linear','heavy_hex_true','star','ring']: ax[1].plot(range(1,9),J['main'][t]['alpha_L'],'o-',ms=3,color=COL[t],label=LAB[t])
for t in ['path_3','path_half']: ax[1].plot(range(1,9),J['pathk'][t]['alpha_L'],'s--',ms=3,color='#999999' if t=='path_3' else '#555555',label=LAB[t])
ax[1].set_xlabel('L'); ax[1].set_ylabel(r'$\hat\alpha(L)$'); ax[1].legend(ncol=2); ax[1].set_title('(b) Path-k vs main topologies')
fig.tight_layout(); fig.savefig(f'{FD}/fig_f4_regression.pdf'); plt.close()
# robustness of grouping across diagnostics
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6)); w=0.27
for j,(k,lab) in enumerate([('main',r'$\theta_0$, $\langle Z_0\rangle$'),('robust_grad_var_z0_all',r'all params, $\langle Z_0\rangle$'),('robust_grad_var_local_all',r'all params, $\frac{1}{n}\sum\langle Z_i\rangle$')]):
    ax[0].bar(x+(j-1)*w,[J[k][t]['ratio'] for t in T],w,label=lab); ax[1].bar(x+(j-1)*w,[J[k][t]['dadl'] for t in T],w,label=lab)
for a,tt in zip(ax,[r'(a) $|\hat\alpha/\hat\beta|$',r'(b) $d\hat\alpha/dL$']):
    a.set_xticks(x); a.set_xticklabels([LAB[t] for t in T],rotation=40,ha='right'); a.set_title(tt)
ax[0].set_yscale('log'); ax[0].axhline(1,ls='--',c='gray',lw=1); ax[0].legend()
fig.tight_layout(); fig.savefig(f'{FD}/fig_diagnostic_robustness.pdf'); plt.close()
# rotation comparison (RX-RY-RZ data has the legacy graph only)
RT=['brick','linear','heavy_hex_true','heavy_hex','star','all_to_all','ring']; gR=tab(G23,'grad_var')
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6),sharey=True)
for t in RT:
    ax[0].plot(range(1,9),J['main'][t]['alpha_L'],'o-',ms=3,color=COL[t],label=LAB[t]); ax[1].plot(range(1,9),J['rxryrz'][t]['alpha_L'],'o-',ms=3,color=COL[t],label=LAB[t])
ax[0].set_title('RY'); ax[1].set_title('RX-RY-RZ'); [a.set_xlabel('L') for a in ax]; ax[0].set_ylabel(r'$\hat\alpha(L)$'); ax[0].legend(ncol=2)
fig.tight_layout(); fig.savefig(f'{FD}/fig_alpha_rotation_comparison.pdf'); plt.close()
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6)); xr=np.arange(len(RT)); w=0.38
ax[0].bar(xr-w/2,[J['main'][t]['dadl'] for t in RT],w,label='RY',color='#bbbbbb'); ax[0].bar(xr+w/2,[J['rxryrz'][t]['dadl'] for t in RT],w,label='RX-RY-RZ',color='#444444')
ax[1].bar(xr-w/2,[J['main'][t]['ratio'] for t in RT],w,label='RY',color='#bbbbbb'); ax[1].bar(xr+w/2,[J['rxryrz'][t]['ratio'] for t in RT],w,label='RX-RY-RZ',color='#444444')
ax[1].axhline(1,ls='--',c='gray',lw=1); ax[1].set_yscale('log')
for a,tt in zip(ax,[r'(a) $d\hat\alpha/dL$',r'(b) $|\hat\alpha/\hat\beta|$']): a.set_xticks(xr); a.set_xticklabels([LAB[t] for t in RT],rotation=40,ha='right'); a.set_title(tt); a.legend()
fig.tight_layout(); fig.savefig(f'{FD}/fig_rotation_richness.pdf'); plt.close()
fig,ax=plt.subplots(figsize=(3.5,2.6))
for t in ['heavy_hex','heavy_hex_true','linear','brick']:
    ns=np.arange(3,17); g=[gR[(t,n,8)] for n in ns]; ax.loglog(ns,g,'o',ms=3,color=COL[t],label=LAB[t])
    s,i=np.polyfit(np.log(ns[ns<=10]),np.log(np.array(g)[ns<=10]),1); ax.loglog(ns,np.exp(i)*ns**s,'--',lw=1,color=COL[t])
ax.axvline(10,ls=':',c='gray'); ax.set_xlabel('n'); ax.set_ylabel(r'$\mathcal{G}$ (RX-RY-RZ, L=8)'); ax.legend()
fig.tight_layout(); fig.savefig(f'{FD}/fig_gap2_large_n.pdf'); plt.close()
# VQE
VT=['brick','linear','heavy_hex_true','heavy_hex','star','ring']; v=VQE.groupby(['ansatz','config']).error_final.agg(['mean','sem'])
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6)); xv=np.arange(len(VT)); w=0.27
for j,c in enumerate(['wide','square','deep']):
    ax[0].bar(xv+(j-1)*w,[v.loc[(t,c),'mean'] for t in VT],w,yerr=[v.loc[(t,c),'sem'] for t in VT],label={'wide':'Wide (8,3)','square':'Square (6,4)','deep':'Deep (4,6)'}[c],capsize=1.5)
ax[0].set_xticks(xv); ax[0].set_xticklabels([LAB[t] for t in VT],rotation=30,ha='right'); ax[0].set_ylabel(r'Energy error $\Delta E$'); ax[0].legend(); ax[0].set_title('(a) VQE, $p=24$, RY')
for t in VT: ax[1].scatter(J['main'][t]['ratio'],J['vqe_gap'][t],color=COL[t],s=25,label=LAB[t]); ax[1].annotate(LAB[t],(J['main'][t]['ratio'],J['vqe_gap'][t]),fontsize=6,xytext=(3,2),textcoords='offset points')
ax[1].set_xscale('log'); ax[1].axvline(1,ls='--',c='gray',lw=1); ax[1].set_xlabel(r'$|\hat\alpha/\hat\beta|$ (RY)'); ax[1].set_ylabel(r'$\Delta E_{\rm wide}-\Delta E_{\rm deep}$'); ax[1].set_title('(b) Training gap vs asymmetry ratio')
fig.tight_layout(); fig.savefig(f'{FD}/fig_gap4_vqe.pdf'); plt.close()
print('ok')
