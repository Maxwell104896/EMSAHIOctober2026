"""Reproduce Fig. 2A from the two archived Step55 CSV files.

Run from any directory: python code/make_fig2a.py --root /path/to/Fig2A_reproducible
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score, mean_absolute_error

PRED='ridge_delta_base_pred_next_HI_score'
IDS=['H4001','H4004','M2014','H4009','P6009','H4010']
COLORS={'HI_low_state':'#457D9A','HI_intermediate_state':'#E4AA57','HI_high_state':'#B96378'}
LABELS={'HI_low_state':'Low','HI_intermediate_state':'Intermediate','HI_high_state':'High'}

def main(root):
    data=root/'data';out=root/'figure';out.mkdir(exist_ok=True)
    d=pd.read_csv(data/'source_cv_predictions.csv')
    summary=pd.read_csv(data/'source_trajectory_summary.csv')
    assert len(d)==152 and d.participant_id.nunique()==66
    assert d[['participant_id','time_t1','time_t2','time_t3','score_t1','score_t2','score_t3',PRED]].notna().all().all()
    r2=r2_score(d.score_t3,d[PRED]);mae=mean_absolute_error(d.score_t3,d[PRED])
    assert abs(r2-.705572)<.00001 and abs(mae-.420718)<.00001
    assert {'R2','MAE'}.issubset(set(summary.metric))

    selected=d[d.participant_id.isin(IDS)].copy()
    assert set(selected.participant_id)==set(IDS)
    assert selected.groupby('participant_id').size().min()>=3
    # One row per participant, prediction window. Retain original sample IDs.
    cols=['participant_id','sample_t1','sample_t2','sample_t3','time_t1','time_t2','time_t3',
          'score_t1','score_t2','score_t3','state_t1','state_t2','observed_three_state_t3',PRED]
    selected[cols].to_csv(data/'selected_prediction_windows.csv',index=False)
    # Reconstruct observed visits from all windows, verify repeated samples agree.
    visits=[]
    for _,r in selected.iterrows():
        for n in (1,2,3):
            visits.append(dict(participant_id=r.participant_id,sample_id=r[f'sample_t{n}'],
                 week=r[f'time_t{n}'],observed_HI=r[f'score_t{n}'],
                 state=r[f'state_t{n}'] if n<3 else r.observed_three_state_t3,
                 label_source_rank=n))
    visits=pd.DataFrame(visits)
    g=visits.groupby(['participant_id','sample_id'])
    assert g.week.nunique().max()==g.observed_HI.nunique().max()==1
    # A borderline H4001 sample has historical state=intermediate but t3=low.
    # The endpoint's observed t3 label takes precedence for the same sample.
    visits=visits.sort_values('label_source_rank').drop_duplicates(
        ['participant_id','sample_id'],keep='last').sort_values(['participant_id','week'])
    assert visits.groupby('participant_id').week.apply(lambda x:x.is_monotonic_increasing).all()
    visits.to_csv(data/'selected_observed_visits.csv',index=False)
    # Cohort-wide scatter receives only observed and out-of-fold predicted values.
    cohort=d[['participant_id','sample_t3','time_t3','score_t3',PRED,'observed_three_state_t3']].copy()
    cohort.to_csv(data/'all_152_out_of_fold_predictions.csv',index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(13.4,9.6),facecolor='white')
    grid=fig.add_gridspec(3,3,width_ratios=[1,1,.92],hspace=.40,wspace=.22,
                          left=.085,right=.98,top=.89,bottom=.125)
    # Deliberate strata and sampling-density selection; errors were not screened.
    for idx,pid in enumerate(IDS):
        ax=fig.add_subplot(grid[idx//2,idx%2]); v=visits[visits.participant_id==pid]
        w=selected[selected.participant_id==pid].sort_values('time_t3')
        ax.plot(v.week,v.observed_HI,color='#293E4A',lw=1.65,zorder=2)
        for st,label in LABELS.items():
            part=v[v.state==st]
            ax.scatter(part.week,part.observed_HI,s=32,color=COLORS[st],edgecolor='white',
                       linewidth=.7,zorder=4)
        # Each predicted point is the next visit of its own (t1,t2)->t3 window.
        ax.scatter(w.time_t3,w[PRED],marker='D',facecolors='none',edgecolors='#2A8A89',
                   s=46,linewidths=1.5,zorder=5)
        for _,r in w.iterrows():
            ax.plot([r.time_t3,r.time_t3],[r.score_t3,r[PRED]],color='#8CADAE',
                    lw=.9,ls=':',zorder=1)
        ax.axhline(0,color='#DDE5E8',lw=.8,zorder=0)
        ax.set_ylim(-2.65,3.15);ax.set_xlim(-3,56);ax.set_xticks([0,10,20,30,40,50])
        ax.grid(axis='y',color='#EDF1F2',lw=.7)
        ax.set_title(f'{pid}  ·  {len(w)} prediction windows',loc='left',fontweight='bold',
                     fontsize=10,color='#2A4755',pad=8)
        if idx//2==2:ax.set_xlabel('Sampling time (weeks)')
        if idx%2==0:ax.set_ylabel('HI score')
    ax=fig.add_subplot(grid[:,2]); lo=min(d.score_t3.min(),d[PRED].min())-.12
    hi=max(d.score_t3.max(),d[PRED].max())+.12
    ax.scatter(d.score_t3,d[PRED],s=20,c='#4D9291',alpha=.58,
               edgecolors='white',linewidth=.28)
    ax.plot([lo,hi],[lo,hi],color='#536774',lw=1.1,ls='--')
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Observed next HI score',
           ylabel='Out-of-fold predicted next HI score')
    ax.set_aspect('equal',adjustable='box');ax.grid(color='#EEF2F3',lw=.75)
    ax.set_title('All prediction windows',loc='left',fontweight='bold',fontsize=11,pad=13)
    ax.text(.04,.97,f'R² = {r2:.3f}\nMAE = {mae:.3f}\n152 windows / 66 participants',
            va='top',transform=ax.transAxes,color='#2B5260',fontsize=10,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=4))
    fig.suptitle('Observed HI trajectories and out-of-fold next-visit predictions',
                 fontsize=15.5,fontweight='bold',x=.085,y=.97,ha='left',color='#244957')
    handles=[Line2D([0],[0],color='#293E4A',marker='o',mfc='white',label='Observed HI'),
             Line2D([0],[0],ls='none',marker='D',mfc='none',mec='#2A8A89',
                    label='Predicted HI at next visit')]
    handles.extend(Line2D([0],[0],ls='none',marker='o',color=c,label=LABELS[s])
                   for s,c in COLORS.items())
    fig.legend(handles=handles,ncol=5,loc='lower left',bbox_to_anchor=(.085,.045),
               frameon=False,columnspacing=1.25,handletextpad=.4,fontsize=9)
    fig.text(.085,.037,'States label observed visits. Predictions use participant-grouped cross-validation; '
             'vertical dotted connectors show prediction error at each t3.',fontsize=8,color='#536875')
    for ext in ['svg','pdf','png']:
        fig.savefig(out/f'Fig2A_observed_predicted_trajectories.{ext}',
                    dpi=600 if ext=='png' else None,facecolor='white')
    plt.close(fig)
    print(f'R2={r2:.6f}; MAE={mae:.6f}; six selected participants; {len(selected)} selected windows')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,
                                default=Path(__file__).resolve().parents[1]);a=parser.parse_args()
    main(a.root)
