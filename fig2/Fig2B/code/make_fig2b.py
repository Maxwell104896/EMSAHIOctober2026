"""Rebuild Fig. 2B from the archived Step55 out-of-fold prediction CSV."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PRED='ridge_delta_base_pred_next_HI_score'
STATES=['HI_low_state','HI_intermediate_state','HI_high_state']
COLORS={'HI_low_state':'#447F9D','HI_intermediate_state':'#DEA353','HI_high_state':'#B76179'}
NAMES={'HI_low_state':'Low HI','HI_intermediate_state':'Intermediate HI','HI_high_state':'High HI'}
BIN_EDGES=[0,7.5,10.5,14.5,20.5,30.5]
BIN_LABELS=['2–7','8–10','11–14','15–20','21–30']

def main(root):
    data=root/'data';out=root/'figure';out.mkdir(exist_ok=True)
    full=pd.read_csv(data/'source_cv_predictions.csv')
    assert len(full)==152 and full.participant_id.nunique()==66
    assert full[['participant_id','sample_t3','time_t2','time_t3','dt23','score_t3',PRED,'state_t2']].notna().all().all()
    assert np.allclose(full.dt23,full.time_t3-full.time_t2)
    d=full[['participant_id','sample_t2','sample_t3','time_t2','time_t3','dt23','score_t2',
            'score_t3',PRED,'state_t2','observed_three_state_t3']].copy()
    d['absolute_error']=(d.score_t3-d[PRED]).abs()
    d['interval_group']=pd.cut(d.dt23,BIN_EDGES,labels=BIN_LABELS,include_lowest=True)
    assert d.interval_group.notna().all()
    d.to_csv(data/'all_windows_error_by_interval.csv',index=False)
    summary=d.groupby('interval_group',observed=True).agg(
        n_windows=('absolute_error','size'),n_participants=('participant_id','nunique'),
        median_absolute_error=('absolute_error','median'),mean_absolute_error=('absolute_error','mean'))
    summary.to_csv(data/'interval_group_summary.csv')
    rho=spearmanr(d.dt23,d.absolute_error).statistic
    # Resample participants rather than windows to respect repeated observations.
    rng=np.random.default_rng(20260928)
    people=d.participant_id.unique()
    patient_rows={p:d[d.participant_id==p] for p in people}
    boot=[];bin_boot={b:[] for b in BIN_LABELS}
    for _ in range(2000):
        sample=pd.concat([patient_rows[p] for p in rng.choice(people,size=len(people),replace=True)],ignore_index=True)
        boot.append(spearmanr(sample.dt23,sample.absolute_error).statistic)
        med=sample.groupby('interval_group',observed=True).absolute_error.median()
        for b in BIN_LABELS:bin_boot[b].append(med.get(b,np.nan))
    rho_ci=np.nanquantile(boot,[.025,.975])
    summary['cluster_bootstrap_median_low']=[np.nanquantile(bin_boot[b],.025) for b in BIN_LABELS]
    summary['cluster_bootstrap_median_high']=[np.nanquantile(bin_boot[b],.975) for b in BIN_LABELS]
    summary.to_csv(data/'interval_group_summary.csv')
    pd.DataFrame({'spearman_rho':[rho], 'participant_cluster_95ci_low':[rho_ci[0]],
                  'participant_cluster_95ci_high':[rho_ci[1]],'n_windows':[len(d)],
                  'n_participants':[len(people)]}).to_csv(data/'association_summary.csv',index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(11.1,5.5),facecolor='white')
    gs=fig.add_gridspec(1,2,width_ratios=[2.25,1],left=.09,right=.98,top=.82,bottom=.27,wspace=.29)
    ax=fig.add_subplot(gs[0]);ax2=fig.add_subplot(gs[1])
    for i,st in enumerate(STATES):
        g=d[d.state_t2==st]
        jitter=np.random.default_rng(300+i).uniform(-.22,.22,len(g))
        ax.scatter(g.dt23+jitter,g.absolute_error,s=32,c=COLORS[st],alpha=.74,
                   edgecolors='white',linewidth=.45,label=f'{NAMES[st]} (n={len(g)})',zorder=3)
    mids=[(BIN_EDGES[i]+BIN_EDGES[i+1])/2 for i in range(5)]
    y=summary.median_absolute_error.to_numpy()
    low=summary.cluster_bootstrap_median_low.to_numpy()
    high=summary.cluster_bootstrap_median_high.to_numpy()
    ax.errorbar(mids,y,yerr=[y-low,high-y],fmt='o-',color='#243F4B',lw=1.8,
                ms=5,capsize=3,label='Interval median (cluster bootstrap 95% CI)',zorder=5)
    ax.set(xlabel='Interval from t2 to t3 (weeks)',ylabel='Absolute next-HI prediction error',
           xlim=(1,31),ylim=(0,2.6))
    ax.set_xticks([2,5,8,10,12,15,20,25,30]);ax.grid(axis='y',color='#EDF1F2',lw=.75,zorder=0)
    ax.text(.97,.96,f'Spearman ρ = {rho:.2f}\nParticipant bootstrap 95% CI\n[{rho_ci[0]:.2f}, {rho_ci[1]:.2f}]',
            transform=ax.transAxes,ha='right',va='top',fontsize=9,color='#244E5D',
            bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':5})
    # Counts expose sparse long intervals and prevent over-reading the trend.
    counts=summary.n_windows.to_numpy()
    ax2.barh(np.arange(5),counts,color='#AED1D0',height=.64)
    ax2.set_yticks(np.arange(5),BIN_LABELS)
    ax2.invert_yaxis();ax2.set(xlabel='Prediction windows (n)',xlim=(0,max(counts)*1.22))
    ax2.set_title('Sample size by interval',loc='left',fontsize=10.5,pad=12,color='#244957')
    ax2.spines['left'].set_visible(False);ax2.tick_params(axis='y',length=0)
    for j,n in enumerate(counts):ax2.text(n+.9,j,str(n),va='center',fontsize=9,color='#244957')
    fig.suptitle('Prediction error across time to the next visit',x=.09,y=.96,ha='left',
                 fontsize=15,fontweight='bold',color='#244957')
    handles=[Line2D([0],[0],ls='none',marker='o',color=COLORS[st],label=NAMES[st]) for st in STATES]
    handles.append(Line2D([0],[0],marker='o',color='#243F4B',label='Binned median ± 95% CI'))
    fig.legend(handles=handles,ncol=4,frameon=False,loc='lower left',bbox_to_anchor=(.09,.085),fontsize=9)
    fig.text(.09,.045,'Each point is one out-of-fold window. Colors refer to the observed HI state at t2; '
             'intervals beyond 20 weeks have few observations.',fontsize=8.4,color='#546975')
    for ext in ('svg','pdf','png'):
        fig.savefig(out/f'Fig2B_interval_vs_absolute_error.{ext}',
                    dpi=600 if ext=='png' else None,facecolor='white')
    plt.close(fig)
    print(f'n={len(d)} windows / {len(people)} participants; Spearman rho={rho:.4f}; 95% CI {rho_ci}')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    main(ap.parse_args().root)
