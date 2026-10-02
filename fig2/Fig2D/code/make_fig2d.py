"""Audit and plot next-high-HI AUROC and next-score MAE by diagnosis, HI state and interval."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

KEY=['participant_id','sample_t1','sample_t2','sample_t3']
RISK='ridge_delta_base_uncalibrated_next_high_state_risk_score'
LABEL='ridge_delta_base_uncalibrated_next_high_state_observed'
SCORE='ridge_delta_base_pred_next_HI_score'
GROUPS=[('Diagnosis','CD','CD'),('Diagnosis','UC','UC'),('Diagnosis','nonIBD','Non-IBD'),
        ('Current HI','HI_low_state','Low'),('Current HI','HI_intermediate_state','Intermediate'),
        ('Current HI','HI_high_state','High'),('Visit interval','Short (≤10 weeks)','≤10 weeks'),
        ('Visit interval','Long (>10 weeks)','>10 weeks')]
PAL={'Diagnosis':'#4B8299','Current HI':'#68A195','Visit interval':'#BD8D6B'}

def main(root):
    data=root/'data';out=root/'figure';out.mkdir(exist_ok=True)
    step54=pd.read_csv(data/'source_step54_windows.csv')
    step55=pd.read_csv(data/'source_step55_oof_predictions.csv')
    step57=pd.read_csv(data/'source_step57_oof_predictions.csv')
    locked=pd.read_csv(data/'source_locked_sample_HI.csv')
    assert len(step54)==len(step55)==len(step57)==152
    assert all(not x.duplicated(KEY).any() for x in [step54,step55,step57])
    assert set(map(tuple,step54[KEY].to_numpy()))==set(map(tuple,step55[KEY].to_numpy()))==set(map(tuple,step57[KEY].to_numpy()))
    assert locked.sample_id.is_unique
    assert locked.groupby('participant_id').diagnosis_label.nunique().max()==1
    d=step55[KEY+['dt23','score_t2','score_t3','state_t2',SCORE]].merge(
        step57[KEY+[LABEL,RISK]],on=KEY,validate='one_to_one')
    # Match t3 sample AND participant. Every sample must join uniquely.
    diag=locked[['sample_id','participant_id','diagnosis_label']]
    d=d.merge(diag,left_on=['sample_t3','participant_id'],right_on=['sample_id','participant_id'],
              how='left',validate='many_to_one')
    assert d.diagnosis_label.notna().all()
    d=d.drop(columns='sample_id')
    d['visit_interval']=np.where(d.dt23<=10,'Short (≤10 weeks)','Long (>10 weeks)')
    d['absolute_error']=(d.score_t3-d[SCORE]).abs()
    assert np.isclose(d.absolute_error.mean(),.420718,atol=1e-5)
    assert np.isclose(roc_auc_score(d[LABEL],d[RISK]),.893807,atol=1e-5)
    d.to_csv(data/'audited_joined_window_predictions.csv',index=False)
    people=d.participant_id.unique();grouped={p:d[d.participant_id==p] for p in people}
    rng=np.random.default_rng(20260928)
    # Same participant resamples across all subgroups and both metrics.
    draws=[pd.concat([grouped[p] for p in rng.choice(people,size=len(people),replace=True)],
                     ignore_index=True) for _ in range(1500)]
    stats=[];bootstrap=[]
    for family,value,display in GROUPS:
        col={'Diagnosis':'diagnosis_label','Current HI':'state_t2','Visit interval':'visit_interval'}[family]
        g=d[d[col]==value];y=g[LABEL].astype(int)
        npos=int(y.sum());nneg=int(len(g)-npos)
        auc=roc_auc_score(y,g[RISK]) if npos and nneg else np.nan
        mae=g.absolute_error.mean()
        boots_auc=[];boots_mae=[]
        for rep,draw in enumerate(draws):
            z=draw[draw[col]==value]
            if len(z):
                boots_mae.append(z.absolute_error.mean())
                if z[LABEL].nunique()==2:
                    boots_auc.append(roc_auc_score(z[LABEL],z[RISK]))
        # AUROC requires enough distinct people in both event classes, plus
        # adequate bootstrap validity; the audit table records the decision.
        positive_people=g.loc[y.eq(1),'participant_id'].nunique()
        negative_people=g.loc[y.eq(0),'participant_id'].nunique()
        eligible=(g.participant_id.nunique()>=10 and npos>=10 and nneg>=10 and
                  positive_people>=5 and negative_people>=5 and len(boots_auc)>=1200)
        ci_auc=np.quantile(boots_auc,[.025,.975]) if eligible else [np.nan,np.nan]
        ci_mae=np.quantile(boots_mae,[.025,.975])
        stats.append(dict(family=family,stratum=display,n_windows=len(g),
                          n_participants=g.participant_id.nunique(),n_positive=npos,n_negative=nneg,
                          n_positive_participants=positive_people,n_negative_participants=negative_people,
                          high_HI_AUROC=auc if eligible else np.nan,
                          AUROC_CI_low=ci_auc[0],AUROC_CI_high=ci_auc[1],
                          score_MAE=mae,MAE_CI_low=ci_mae[0],MAE_CI_high=ci_mae[1],
                          valid_auc_bootstraps=len(boots_auc),AUROC_estimable=eligible,
                          reason='' if eligible else ('no positive events' if npos==0 else 'insufficient events or participants')))
        for rep,draw in enumerate(draws):
            z=draw[draw[col]==value]
            if len(z):
                bootstrap.append(dict(replicate=rep,family=family,stratum=display,
                    AUROC=roc_auc_score(z[LABEL],z[RISK]) if z[LABEL].nunique()==2 else np.nan,
                    MAE=z.absolute_error.mean()))
    table=pd.DataFrame(stats);table.to_csv(data/'stratum_count_and_performance_audit.csv',index=False)
    pd.DataFrame(bootstrap).to_csv(data/'participant_bootstrap_subgroup_metrics.csv',index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,(a,b)=plt.subplots(1,2,figsize=(11.3,6.3),sharey=True,
                           gridspec_kw={'width_ratios':[1.25,1]},facecolor='white')
    fig.subplots_adjust(left=.22,right=.74,top=.81,bottom=.16,wspace=.29)
    ys=np.array([8,7,6,4.65,3.65,2.65,1.3,.3])
    for k,r in table.iterrows():
        y=ys[k];color=PAL[r.family]
        if r.AUROC_estimable:
            a.plot([r.AUROC_CI_low,r.AUROC_CI_high],[y,y],color=color,lw=2)
            a.scatter(r.high_HI_AUROC,y,s=62,c=color,edgecolors='white',linewidth=.8,zorder=3)
        else:a.text(.56,y,'not estimable',va='center',color='#8A6161',fontsize=9)
        b.plot([r.MAE_CI_low,r.MAE_CI_high],[y,y],color=color,lw=2)
        b.scatter(r.score_MAE,y,s=62,c=color,edgecolors='white',linewidth=.8,zorder=3)
        b.text(1.04,y,f"{int(r.n_participants)} people / {int(r.n_windows)} windows / {int(r.n_positive)} events",
               transform=b.get_yaxis_transform(),va='center',fontsize=8,color='#4E6670',clip_on=False)
    a.set_yticks(ys,table.stratum);a.set_ylim(-.25,8.75)
    a.set(xlim=(.45,1.01),xlabel='Next high-HI AUROC')
    b.set(xlim=(0,.85),xlabel='Next HI score MAE')
    for ax in [a,b]:
        ax.grid(axis='x',color='#E9EFF0',lw=.75)
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
        for boundary in (5.4,2.0):ax.axhline(boundary,color='#E0E8EA',lw=.9)
    a.axvline(.5,color='#AFBFC4',ls='--',lw=.85)
    a.set_title('High-HI event discrimination',loc='left',color='#254954',fontsize=11,pad=12)
    b.set_title('Continuous HI score error',loc='left',color='#254954',fontsize=11,pad=12)
    fig.suptitle('Trajectory performance by observed subgroup',x=.09,y=.96,ha='left',
                 color='#254954',fontsize=16,fontweight='bold')
    fig.text(.09,.895,'Points: subgroup estimate  •  lines: participant-bootstrap 95% interval',
             fontsize=9.5,color='#536A75')
    fig.text(.09,.065,'Non-IBD and current Low HI have zero next-high-HI events; their AUROC cannot be computed. '
             'Groups are descriptive and overlap across panels.',fontsize=8.7,color='#546A73')
    for ext in ['svg','pdf','png']:
        fig.savefig(out/f'Fig2D_subgroup_performance.{ext}',dpi=600 if ext=='png' else None,facecolor='white')
    plt.close(fig)
    print(table[['family','stratum','n_participants','n_windows','n_positive','high_HI_AUROC','score_MAE']].to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    main(ap.parse_args().root)
