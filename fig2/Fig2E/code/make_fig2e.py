"""Fig. 2E: interval-stratified next-high-HI AUROC and next-HI-score MAE."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

KEY=['participant_id','sample_t1','sample_t2','sample_t3']
EVENT='ridge_delta_base_uncalibrated_next_high_state_observed'
RISK='ridge_delta_base_uncalibrated_next_high_state_risk_score'
PRED='ridge_delta_base_pred_next_HI_score'
LABELS=['2–8','9–10','11–14','15–30']
EDGES=[0,8,10,14,30]

def main(root):
    data=root/'data';out=root/'figure';out.mkdir(exist_ok=True)
    a=pd.read_csv(data/'source_step55_oof_predictions.csv')
    b=pd.read_csv(data/'source_step57_oof_predictions.csv')
    ref=pd.read_csv(data/'source_final_endpoint_performance.csv')
    assert len(a)==len(b)==152
    assert not a.duplicated(KEY).any() and not b.duplicated(KEY).any()
    d=a[KEY+['time_t2','time_t3','dt23','score_t3',PRED]].merge(
         b[KEY+[EVENT,RISK]],on=KEY,validate='one_to_one')
    assert len(d)==152 and d.participant_id.nunique()==66
    assert np.allclose(d.dt23,d.time_t3-d.time_t2)
    assert np.isclose(roc_auc_score(d[EVENT],d[RISK]),
                      ref.loc[ref.endpoint=='next_high_state','AUROC'].iloc[0])
    d['absolute_error']=(d.score_t3-d[PRED]).abs()
    d['interval_group']=pd.cut(d.dt23,EDGES,labels=LABELS,include_lowest=True)
    assert d.interval_group.notna().all()
    d.to_csv(data/'all_152_interval_predictions.csv',index=False)

    # Cut points are the empirical 25th/50th/75th percentiles (8,10,14.25).
    # Weeks are integers, so >14.25 is equivalent to ≥15.
    assert np.allclose(d.dt23.quantile([.25,.5,.75]).values,[8,10,14.25])
    participants=d.participant_id.unique()
    groups={p:d[d.participant_id==p] for p in participants}
    rng=np.random.default_rng(20260928)
    bootstrap=[]
    for rep in range(2000):
        draw=pd.concat([groups[p] for p in rng.choice(participants,len(participants),replace=True)],
                       ignore_index=True)
        for lab in LABELS:
            z=draw[draw.interval_group==lab]
            bootstrap.append(dict(replicate=rep,interval_group=lab,
                 n_windows=len(z),n_events=int(z[EVENT].sum()),
                 AUROC=roc_auc_score(z[EVENT],z[RISK]) if z[EVENT].nunique()==2 else np.nan,
                 MAE=z.absolute_error.mean() if len(z) else np.nan))
    boots=pd.DataFrame(bootstrap)
    boots.to_csv(data/'participant_bootstrap_interval_metrics.csv',index=False)
    output=[]
    for lab in LABELS:
        z=d[d.interval_group==lab];q=boots[boots.interval_group==lab]
        assert z[EVENT].nunique()==2
        valid=q.AUROC.dropna()
        output.append(dict(interval_group=lab,min_weeks=z.dt23.min(),max_weeks=z.dt23.max(),
           n_windows=len(z),n_participants=z.participant_id.nunique(),n_high_events=int(z[EVENT].sum()),
           n_other_events=int(len(z)-z[EVENT].sum()),
           AUROC=roc_auc_score(z[EVENT],z[RISK]),AUROC_95ci_low=valid.quantile(.025),
           AUROC_95ci_high=valid.quantile(.975),MAE=z.absolute_error.mean(),
           MAE_95ci_low=q.MAE.quantile(.025),MAE_95ci_high=q.MAE.quantile(.975),
           valid_auc_bootstrap_replicates=len(valid)))
    result=pd.DataFrame(output);result.to_csv(data/'interval_quartile_performance_audit.csv',index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.7,5.4),facecolor='white')
    fig.subplots_adjust(left=.10,right=.97,top=.79,bottom=.29,wspace=.28)
    x=np.arange(4)
    for axis,metric,low,high,color,ylim,ylabel in [
        (ax,'AUROC','AUROC_95ci_low','AUROC_95ci_high','#3C8490',(.4,1.01),'Next high-HI AUROC'),
        (bx,'MAE','MAE_95ci_low','MAE_95ci_high','#AD7890',(0,.83),'Next HI score MAE')]:
        vals=result[metric].to_numpy()
        err=[vals-result[low].to_numpy(),result[high].to_numpy()-vals]
        axis.errorbar(x,vals,yerr=err,fmt='o-',markersize=7,capsize=4,
                      capthick=1.2,elinewidth=1.3,lw=1.5,color=color,zorder=4)
        axis.set(ylim=ylim,ylabel=ylabel,xticks=x,xticklabels=[
            f'{r.interval_group} weeks\nn{int(r.n_windows)} · p{int(r.n_participants)} · e{int(r.n_high_events)}'
            for _,r in result.iterrows()])
        axis.tick_params(axis='x',labelsize=8.2,pad=6)
        axis.grid(axis='y',color='#EBF0F1',lw=.75)
        axis.set_xlabel('Time to next sample (t2→t3)')
    ax.axhline(.5,color='#B3C3C6',lw=.8,ls='--')
    fig.suptitle('Performance across time to the next sample',x=.10,y=.96,
                 ha='left',fontsize=16,fontweight='bold',color='#254954')
    fig.text(.10,.88,'Empirical interval quartiles  •  152 out-of-fold windows / 66 participants',
             fontsize=10,color='#526A73')
    fig.text(.10,.07,'n = windows; p = participants; e = next high-HI events. Lines show 95% percentile CIs '
             'from 2,000 participant-cluster bootstrap resamples.',
             fontsize=8.2,color='#546C75')
    for ext in ('svg','pdf','png'):
        fig.savefig(out/f'Fig2E_followup_interval_stability.{ext}',
                    dpi=600 if ext=='png' else None,facecolor='white')
    plt.close(fig)
    print(result[['interval_group','n_participants','n_windows','n_high_events',
                  'AUROC','AUROC_95ci_low','AUROC_95ci_high','MAE']].to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    main(ap.parse_args().root)
