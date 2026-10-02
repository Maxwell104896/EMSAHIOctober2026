"""Repair outer participant isolation using archived fold-specific branch predictions.
This is a conditional reanalysis, not a rerun of upstream matrices or original Step55.
Run from the audit workspace root. Outputs do not overwrite source data.
"""
from pathlib import Path
import glob,json,itertools
import numpy as np,pandas as pd
import joblib,sklearn,platform
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score,mean_absolute_error,r2_score,balanced_accuracy_score,confusion_matrix
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
def read(p):return pd.read_csv(glob.glob(p)[0])
long=pd.read_csv(OUT/'all_branch_scores.csv')
bf=pd.read_csv(DATA/'EMSA_global_primary_outer_grouped_folds.csv')
inner=pd.read_csv(DATA/'EMSA_global_primary_inner_grouped_folds_long.csv')
old=pd.read_csv(DATA/'core_308_metadata.csv')
mbx=pd.read_csv(DATA/'MBX_546_metadata.csv')
w=pd.read_csv(DATA/'ordered_152_windows.csv')
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
    model=make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='lbfgs',max_iter=5000))
    model.fit(xt[a],y[a]);pr[b]=model.predict_proba(xt[b])[:,1]
   auc=roc_auc_score(y,pr);ranks.append((-auc,c));tune.append(dict(model=name,outer_fold=f,C=c,training_only_inner_auc=auc))
  c=min(ranks)[1];model=make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='lbfgs',max_iter=5000)).fit(xt,y)
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
axes,s308,fr1,tu1,ov1=fusion(old,branches,'HI308')
_,s546,fr2,tu2,ov2=fusion(mbx,['MBX_metabolomics_branch'],'MBX546')
all_meta=long[['sample_id','participant_id','diagnosis_label','disease_group_binary']].drop_duplicates('sample_id')
_,sall,fra,tua,ova=fusion(all_meta,sorted(long.branch_model_name.unique()),'Broad')
pd.DataFrame(fr1+fr2).to_csv(OUT/'fusion_fold_fit_audit.csv',index=False)
pd.DataFrame(tu1+tu2).to_csv(OUT/'fusion_training_only_tuning.csv',index=False)
pd.DataFrame(ov1+ov2).to_csv(OUT/'outer_isolation_checks.csv',index=False)
window_fold=w.participant_id.map(foldmap['first']).to_numpy();assert np.isfinite(window_fold).all()
predrows=[];tunerows=[];selections=[];foldmetrics=[]
def stat(y,p):return dict(mae=float(mean_absolute_error(y,p)),r2=float(r2_score(y,p)),spearman=float(spearmanr(y,p).statistic))
for f in range(5):
 d=w[['participant_id','sample_t1','sample_t2','sample_t3','time_t1','time_t2','time_t3','dt12','dt23']].copy()
 for j in [1,2,3]:d[f'score_t{j}']=d[f'sample_t{j}'].map(axes[f].HI_score)
 assert d[['score_t1','score_t2','score_t3']].notna().all().all()
 d['delta12']=d.score_t2-d.score_t1;d['abs_delta12']=abs(d.delta12);d['velocity12']=d.delta12/d.dt12;d['target_delta23']=d.score_t3-d.score_t2
 tr=np.flatnonzero(window_fold!=f);te=np.flatnonzero(window_fold==f);g=d.participant_id.to_numpy();y=d.target_delta23.to_numpy();obs=d.score_t3.to_numpy();current=d.score_t2.to_numpy()
 assert not set(g[tr])&set(g[te]);inner_splits=list(GroupKFold(4).split(d.iloc[tr],y[tr],g[tr]));ranking=[]
 for c in configs:
  xp=d[c['features']].to_numpy();ip=np.full(len(tr),np.nan)
  for itr,ite in inner_splits:
   m=make_pipeline(StandardScaler(),Ridge(alpha=c['alpha'])).fit(xp[tr[itr]],y[tr[itr]]);ip[ite]=m.predict(xp[tr[ite]])+current[tr[ite]]
  err=mean_absolute_error(obs[tr],ip);ranking.append((err,len(c['features']),c['alpha'],c['config_id'],c));tunerows.append(dict(outer_fold=f,config_id=c['config_id'],inner_training_mae=err))
 chosen=min(ranking,key=lambda z:z[:4])[-1];xp=d[chosen['features']].to_numpy();m=make_pipeline(StandardScaler(),Ridge(alpha=chosen['alpha'])).fit(xp[tr],y[tr]);p=m.predict(xp[te])+current[te]
 linear=current[te]+d.delta12.to_numpy()[te]
 ql,qh=np.quantile(obs[tr],[1/3,2/3]);classes=lambda v:np.where(v<=ql,0,np.where(v>=qh,2,1))
 # Binary event model tuned entirely within outer-training windows.
 xb=d[['score_t1','score_t2','delta12','abs_delta12','velocity12','dt12']].to_numpy();event=(obs>=qh).astype(int);br=[]
 for c in CGRID:
  ip=np.full(len(tr),np.nan)
  for itr,ite in inner_splits:
   # Recompute threshold on this inner-training set; no validation outcomes determine it.
   ih=np.quantile(obs[tr[itr]],2/3);iy=(obs[tr[itr]]>=ih).astype(int)
   bm=make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='lbfgs',max_iter=5000)).fit(xb[tr[itr]],iy);ip[ite]=bm.predict_proba(xb[tr[ite]])[:,1]
  br.append((-roc_auc_score(event[tr],ip),c))
 bc=min(br)[1];bm=make_pipeline(StandardScaler(),LogisticRegression(C=bc,solver='lbfgs',max_iter=5000)).fit(xb[tr],event[tr]);risk=bm.predict_proba(xb[te])[:,1]
 joblib.dump(dict(ridge=m,event_model=bm,ridge_features=chosen['features'],event_features=['score_t1','score_t2','delta12','abs_delta12','velocity12','dt12'],low_threshold=float(ql),high_threshold=float(qh),train_window_indices=tr,test_window_indices=te),OUT/'models'/f'trajectory_fold_{f}.joblib',compress=3)
 out=d.iloc[te].copy();out['outer_fold']=f;out['window_index']=te;out['pred_HI']=p;out['persistence']=current[te];out['linear']=linear;out['pred_delta']=p-current[te];out['low_threshold']=ql;out['high_threshold']=qh;out['observed_class']=classes(obs[te]);out['predicted_class']=classes(p);out['persistence_class']=classes(current[te]);out['event_probability']=risk;out['event_label']=event[te]
 predrows.append(out);selections.append(dict(outer_fold=f,n_train_windows=len(tr),n_test_windows=len(te),n_train_participants=len(set(g[tr])),n_test_participants=len(set(g[te])),config_id=chosen['config_id'],features='|'.join(chosen['features']),alpha=chosen['alpha'],event_C=bc,low_threshold=ql,high_threshold=qh))
 for name,pr in [('ridge',p),('persistence',current[te]),('linear',linear)]:foldmetrics.append(dict(outer_fold=f,model=name,n_windows=len(te),**stat(obs[te],pr)))
out=pd.concat(predrows).sort_values('window_index');assert len(out)==152 and out.window_index.nunique()==152
out.to_csv(OUT/'corrected_trajectory_predictions.csv',index=False);pd.DataFrame(tunerows).to_csv(OUT/'trajectory_inner_search.csv',index=False);pd.DataFrame(selections).to_csv(OUT/'trajectory_selected_configurations.csv',index=False);pd.DataFrame(foldmetrics).to_csv(OUT/'trajectory_fold_metrics.csv',index=False)
metrics=[]
for name,col in [('ridge','pred_HI'),('persistence','persistence'),('linear','linear')]:metrics.append(dict(model=name,**stat(out.score_t3,out[col])))
delta_r2=float(r2_score(out.target_delta23,out.pred_delta));ba=float(balanced_accuracy_score(out.observed_class,out.predicted_class));baseba=float(balanced_accuracy_score(out.observed_class,out.persistence_class));auc=float(roc_auc_score(out.event_label,out.event_probability))
rng=np.random.default_rng(20260930);ids=out.participant_id.unique();ix={i:np.flatnonzero(out.participant_id.to_numpy()==i) for i in ids};boots=[]
for _ in range(2000):
 ii=np.concatenate([ix[i] for i in rng.choice(ids,len(ids),replace=True)]);s=out.iloc[ii];boots.append(float(np.mean(abs(s.pred_HI-s.score_t3))-np.mean(abs(s.persistence-s.score_t3))))
summary=dict(scope='Outer-aligned conditional reanalysis from recomputed fold-specific branch predictions; all 11 upstream matrices refitted; fusion/trajectory inner representations are conditional on outer-isolated branch fits.',fusion=[s308,s546,sall],trajectory=metrics,delta_r2=delta_r2,three_state_balanced_accuracy=ba,persistence_three_state_balanced_accuracy=baseba,confusion_matrix=confusion_matrix(out.observed_class,out.predicted_class,labels=[0,1,2]).tolist(),next_high_event_auc=auc,event_model_features=['score_t1','score_t2','delta12','abs_delta12','velocity12','dt12'],event_count=int(out.event_label.sum()),n_windows=len(out),n_participants=out.participant_id.nunique(),ridge_minus_persistence_mae_ci95=np.quantile(boots,[.025,.975]).tolist(),outer_test_participant_overlap=0,normalization='Each outer axis uses only training lp mean/std; each state threshold uses only outer-training t3 values.',note='New fold-specific target axis differs from archived globally standardized HI; old and new absolute metrics are not paired comparisons.')
pd.DataFrame(metrics).to_csv(OUT/'corrected_performance.csv',index=False);(OUT/'recomputed_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
