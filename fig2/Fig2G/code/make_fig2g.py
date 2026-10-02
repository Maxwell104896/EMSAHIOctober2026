"""Figure 2G: honest, same-cohort cross-sectional external HI projection ROC."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve

ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'data'; F=ROOT/'figure'
TASKS=[('IBD vs non-IBD',['CD','UC','nonIBD'],'IBD_vs_nonIBD'),
       ('CD vs non-IBD',['CD','nonIBD'],'CD_vs_nonIBD'),
       ('UC vs non-IBD',['UC','nonIBD'],'UC_vs_nonIBD')]
SCORES={'Primary module projection':'EMSA_HI_external_primary_weighted_modules',
        'No-bile sensitivity':'EMSA_HI_external_no_bile_sensitivity'}
def main():
 F.mkdir(exist_ok=True)
 raw=pd.read_csv(D/'source_step61_external_MBX_projected_HI_scores.csv')
 ref=pd.read_csv(D/'source_step61_external_HI_locked_projection_performance.csv')
 assert len(raw)==220 and raw.sample_id.is_unique
 assert raw.participant_id.notna().all()
 metrics=[];curves=[];records=[]
 fig,axes=plt.subplots(1,3,figsize=(13.5,5.8),facecolor='white')
 fig.subplots_adjust(left=.063,right=.98,top=.73,bottom=.30,wspace=.23)
 for ax,(title,allowed,code) in zip(axes,TASKS):
  sub=raw[raw.diagnosis_group.isin(allowed)].copy()
  y=(sub.diagnosis_group!='nonIBD').astype(int)
  assert set(sub.diagnosis_group)==set(allowed)
  assert sub.participant_id.nunique()==len(sub), 'Audit repeated participants before CI'
  for label,col in SCORES.items():
   risk=sub[col].to_numpy();fpr,tpr,_=roc_curve(y,risk);auc=roc_auc_score(y,risk)
   published=ref[(ref.task==code)&(ref.score_name==col)].iloc[0]
   assert len(sub)==published.n and np.isclose(auc,published.AUROC,atol=1e-10)
   rng=np.random.default_rng(20260928+TASKS.index((title,allowed,code)))
   pos=risk[y.to_numpy()==1]; neg=risk[y.to_numpy()==0]
   bs=[roc_auc_score(np.r_[np.ones(len(pos)),np.zeros(len(neg))],
      np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]) for _ in range(2000)]
   lo,hi=np.percentile(bs,[2.5,97.5])
   metrics.append(dict(endpoint=code,score=label,n=len(sub),positive_n=int(y.sum()),negative_n=int((1-y).sum()),
      n_participants=sub.participant_id.nunique(),AUROC=auc,CI_low=lo,CI_high=hi))
   curves.extend(dict(endpoint=code,score=label,FPR=a,TPR=b) for a,b in zip(fpr,tpr))
   colr='#177C88' if label.startswith('Primary') else '#B28B45'
   ax.plot(fpr,tpr,color=colr,linestyle='-',linewidth=2.5,
      label=f'{label}: {auc:.3f} ({lo:.3f}–{hi:.3f})')
  records.append(sub[['sample_id','participant_id','diagnosis_group',*SCORES.values()]])
  ax.plot([0,1],[0,1],ls='--',color='#C4D0D1',lw=1)
  ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='1 − specificity',ylabel='Sensitivity' if ax is axes[0] else '')
  ax.set_title(title,fontsize=12.5,fontweight='bold',loc='left',pad=12)
  ax.text(.05,.04,f'n={len(sub)} · {int(y.sum())} cases / {int((1-y).sum())} non-IBD',transform=ax.transAxes,fontsize=8.5,color='#516970')
  ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.14)
  ax.legend(loc='upper left',bbox_to_anchor=(0,-.22),fontsize=7.4,frameon=False,title='AUROC (95% bootstrap CI)',title_fontsize=8)
 fig.text(.063,.93,'G   External cross-sectional HI state-axis projection',fontsize=17,fontweight='bold',color='#244C55')
 fig.text(.063,.855,'ST001000 / PR000677 · 220 samples · recoverable MBX modules · no external fitting',fontsize=10,color='#526C74')
 fig.text(.063,.04,'External metabolic projection only; not validation of the full longitudinal EMSA-HI model. The no-bile series is a sensitivity analysis.',fontsize=8.8,color='#56686F')
 pd.DataFrame(metrics).to_csv(D/'external_endpoint_auc_and_bootstrap_ci.csv',index=False)
 pd.DataFrame(curves).to_csv(D/'external_roc_coordinates.csv',index=False)
 raw[['sample_id','participant_id','diagnosis_group',*SCORES.values()]].to_csv(D/'external_220_samples_selected_scores.csv',index=False)
 for ext in ('svg','pdf','png'):
  fig.savefig(F/f'Fig2G_external_cross_sectional_projection.{ext}',dpi=600 if ext=='png' else None,facecolor='white')
 plt.close(fig)
 print(pd.DataFrame(metrics).to_string(index=False))
if __name__=='__main__':main()
