"""Reproduce Fig.2F from original Step55/57 out-of-fold prediction tables."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; FIG=ROOT/'figure'
KEY=['participant_id','sample_t1','sample_t2','sample_t3']
MODEL={
 'Next high HI':('ridge_delta_base_uncalibrated_next_high_state_observed','ridge_delta_base_uncalibrated_next_high_state_risk_score',.8938070277615997),
 'Next extreme HI':('ridge_plus_modules_score_calibrated_next_high_extreme_observed','ridge_plus_modules_score_calibrated_next_high_extreme_risk_score',.9104166666666667),
}
def main():
 FIG.mkdir(exist_ok=True)
 s55=pd.read_csv(DATA/'source_step55_oof_predictions.csv')
 s57=pd.read_csv(DATA/'source_step57_oof_predictions.csv')
 assert len(s55)==len(s57)==152 and s55.participant_id.nunique()==66
 assert s55[KEY].duplicated().sum()==s57[KEY].duplicated().sum()==0
 joined=s55[KEY+['score_t1','score_t2','score_t3','linear_pred_score_t3','dt12','dt23']].merge(
    s57[KEY+[z for pair in MODEL.values() for z in pair[:2]]+
    ['persistence_next_high_state_risk_score','persistence_next_high_extreme_risk_score']],
    on=KEY,validate='one_to_one')
 assert len(joined)==152
 DATA.joinpath('matched_152_out_of_fold_scores.csv').write_text(joined.to_csv(index=False))
 fig,axes=plt.subplots(1,2,figsize=(12.4,6.6),facecolor='white')
 fig.subplots_adjust(left=.075,right=.98,bottom=.38,top=.78,wspace=.19)
 colors={'EMSA-HI':'#087E8B','LOCF / current HI':'#DB8C29','Two-point linear extrapolation':'#936B98'}
 metric=[];curves=[];seed=20260928
 participants=joined.participant_id.unique();groups={p:joined[joined.participant_id==p] for p in participants}
 for ax,(title,(outcome,model,reference)) in zip(axes,MODEL.items()):
  y=joined[outcome].astype(int)
  assert len(np.unique(y))==2
  assert np.isclose(roc_auc_score(y,joined[model]),reference,atol=1e-6)
  persistent='persistence_next_high_state_risk_score' if title=='Next high HI' else 'persistence_next_high_extreme_risk_score'
  assert np.array_equal(joined[persistent].to_numpy(),joined.score_t2.to_numpy())
  scores={'EMSA-HI':model,'LOCF / current HI':'score_t2','Two-point linear extrapolation':'linear_pred_score_t3'}
  rng=np.random.default_rng(seed+(0 if title=='Next high HI' else 1))
  draws=rng.integers(len(participants),size=(2000,len(participants)))
  for label,col in scores.items():
   x=joined[col];fpr,tpr,_=roc_curve(y,x);auc=roc_auc_score(y,x)
   bs=[]
   for draw in draws:
    z=pd.concat([groups[participants[i]] for i in draw],ignore_index=True)
    if z[outcome].nunique()==2:bs.append(roc_auc_score(z[outcome],z[col]))
   lo,hi=np.percentile(bs,[2.5,97.5]);metric.append({'endpoint':title,'model':label,'n_windows':152,'n_participants':66,
   'n_positive_windows':int(y.sum()),'AUROC':auc,'CI_low':lo,'CI_high':hi,'valid_bootstrap_draws':len(bs)})
   curves.extend({'endpoint':title,'model':label,'FPR':a,'TPR':b} for a,b in zip(fpr,tpr))
   ax.plot(fpr,tpr,linestyle='-',color=colors[label],linewidth=2.65 if label=='EMSA-HI' else 2.2,
           label=f'{label}  {auc:.3f} ({lo:.3f}–{hi:.3f})')
  ax.plot([0,1],[0,1],color='#B9C7CA',linewidth=1.2,linestyle='--',zorder=0)
  ax.set(xlim=(0,1),ylim=(0,1.01),xlabel='1 − specificity',ylabel='Sensitivity' if ax is axes[0] else '')
  ax.set_title(f'{"A" if ax is axes[0] else "B"}  {title}',loc='left',fontweight='bold',fontsize=13,pad=14)
  ax.text(.03,.045,f'{int(y.sum())} events / 152 windows / 66 participants',transform=ax.transAxes,fontsize=9,color='#5C6E73')
  ax.legend(loc='upper left',bbox_to_anchor=(0,-.18),frameon=False,fontsize=8.7,title='AUROC (participant-cluster 95% CI)',title_fontsize=9)
  ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
 fig.text(.075,.94,'F   ROC comparison on matched prediction windows',fontweight='bold',fontsize=17,color='#234953')
 fig.text(.075,.875,'Out-of-fold EMSA-HI and prespecified score-based baselines',fontsize=10,color='#567078')
 fig.text(.075,.035,'LOCF and current HI have identical ranking (both use HI at t2). Clinical-variable out-of-fold scores were unavailable.',fontsize=8.6,color='#5B6A6F')
 pd.DataFrame(metric).to_csv(DATA/'model_auc_cluster_bootstrap.csv',index=False)
 pd.DataFrame(curves).to_csv(DATA/'roc_curve_coordinates.csv',index=False)
 for ext in ['svg','pdf','png']:
  fig.savefig(FIG/f'Fig2F_model_ROC_comparison.{ext}',dpi=600 if ext=='png' else None,facecolor='white')
 plt.close(fig)
 print(pd.DataFrame(metric).to_string(index=False))
 for p in sorted(DATA.glob('source_*')): print(hashlib.sha256(p.read_bytes()).hexdigest(),p.name)
if __name__=='__main__':main()
