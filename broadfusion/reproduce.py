import itertools
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
long=pd.read_csv(ROOT/'data'/'branch_scores.csv')
bf=pd.read_csv(ROOT/'data'/'outer_folds.csv')
inner=pd.read_csv(ROOT/'data'/'inner_folds.csv')
branches=['MBX_metabolomics_branch','MPX_protein_branch']
foldmap=bf.groupby('participant_id').outer_fold.agg(['first','nunique'])
assert foldmap['nunique'].eq(1).all()
CGRID=[.01,.03,.1,.3,1.,3.,10.]
configs=[]
for hist,dyn,horizon,alpha in itertools.product([1,2],[0,1,3],[0],np.logspace(-3,3,13)):
 fs=['score_t2'] if hist==1 else ['score_t1','score_t2']
 fs+={0:[],1:['delta12'],3:['delta12','abs_delta12','velocity12']}[dyn]
 fs+=['dt12']+(['dt23'] if horizon else [])
 configs.append(dict(config_id=f'C{len(configs)+1:03d}',features=fs,alpha=float(alpha)))
def Xfor(meta,f,role,bs):
 d=long[(long.outer_fold==f)&(long.prediction_role==role)&long.branch_model_name.isin(bs)]
 assert not d.duplicated(['sample_id','branch_model_name']).any()
 x=d.pivot(index='sample_id',columns='branch_model_name',values='branch_state_score').reindex(index=meta.sample_id,columns=bs)
 if len(bs)>2:
  return np.column_stack([x.fillna(0).to_numpy(),x.notna().astype(int).to_numpy()])
 assert x.notna().all().all()
 return x.to_numpy()
def fusion(meta,bs,name):
 meta=meta[['sample_id','participant_id','diagnosis_label','disease_group_binary']].copy().reset_index(drop=True)
 meta['global_outer_fold']=meta.participant_id.map(foldmap['first'])
 scores=[];tables={};foldrows=[];tune=[];overlap=[]
 (OUT/'models').mkdir(exist_ok=True)
 for f in range(5):
  tr=meta[meta.global_outer_fold!=f].copy();te=meta[meta.global_outer_fold==f].copy()
  assert not set(tr.participant_id)&set(te.participant_id)
  # All upstream training participants exclude this SAME outer fold.
  held=set(bf.loc[bf.outer_fold==f,'participant_id'])
  upstream=set(bf.loc[bf.outer_fold!=f,'participant_id'])
  assert not held&upstream
  overlap.append(dict(model=name,outer_fold=f,n_test_participants=te.participant_id.nunique(),upstream_test_overlap=len(held&upstream),fusion_test_overlap=len(set(tr.participant_id)&set(te.participant_id))))
  xt=Xfor(tr,f,'inner_oof_train',bs);xe=Xfor(te,f,'outer_test',bs)
  y=tr.disease_group_binary.to_numpy();im=inner[inner.outer_fold==f].drop_duplicates('sample_id').set_index('sample_id').inner_fold
  split=tr.sample_id.map(im).to_numpy();assert np.isfinite(split).all()
  ranks=[]
  for c in CGRID:
   pr=np.full(len(tr),np.nan)
   for j in np.unique(split):
    a=split!=j;b=split==j
    assert not set(tr.loc[a,'participant_id'])&set(tr.loc[b,'participant_id'])
    model=make_pipeline(StandardScaler(),LogisticRegression(C=c,penalty='l2',solver='lbfgs',max_iter=5000))
    model.fit(xt[a],y[a]);pr[b]=model.predict_proba(xt[b])[:,1]
   auc=roc_auc_score(y,pr);ranks.append((-auc,c));tune.append(dict(model=name,outer_fold=f,C=c,training_only_inner_auc=auc))
  c=min(ranks)[1];model=make_pipeline(StandardScaler(),LogisticRegression(C=c,penalty='l2',solver='lbfgs',max_iter=5000)).fit(xt,y)
  lp=model.decision_function(xt);mu=float(lp.mean());sd=float(lp.std());assert sd>0
  joblib.dump(dict(model=model,branch_order=bs,lp_mean=mu,lp_sd=sd,train_samples=tr.sample_id.tolist(),test_samples=te.sample_id.tolist()),OUT/'models'/f'{name}_fusion_fold_{f}.joblib',compress=3)
  tr['HI_score']=(lp-mu)/sd;te['HI_score']=(model.decision_function(xe)-mu)/sd
  tr['HI_probability']=model.predict_proba(xt)[:,1];te['HI_probability']=model.predict_proba(xe)[:,1]
  tr['role']='train_inner_branch_input';te['role']='outer_test_branch_input'
  both=pd.concat([tr,te]);both['axis_outer_fold']=f;tables[f]=both.set_index('sample_id')
  both.to_csv(OUT/f'{name}_axis_fold_{f}_all_visits.csv',index=False);scores.append(te)
  foldrows.append(dict(model=name,outer_fold=f,C=c,n_train=len(tr),n_test=len(te),train_lp_mean=mu,train_lp_sd=sd,test_auc=roc_auc_score(te.disease_group_binary,te.HI_probability)))
 sc=pd.concat(scores).sort_index();sc.to_csv(OUT/f'{name}_outer_test_scores.csv',index=False)
 return tables,dict(model=name,n_samples=len(sc),n_participants=sc.participant_id.nunique(),pooled_auc=roc_auc_score(sc.disease_group_binary,sc.HI_probability),pooled_auprc=average_precision_score(sc.disease_group_binary,sc.HI_probability)),foldrows,tune,overlap

meta=long[['sample_id','participant_id','diagnosis_label','disease_group_binary']].drop_duplicates('sample_id')
_,metrics,folds,tuning,checks=fusion(meta,sorted(long.branch_model_name.unique()),'Broad')
import json
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2));pd.DataFrame(folds).to_csv(OUT/'fold_fits.csv',index=False);pd.DataFrame(tuning).to_csv(OUT/'tuning.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'isolation_checks.csv',index=False);print(metrics)
