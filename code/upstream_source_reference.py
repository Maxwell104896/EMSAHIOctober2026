"""Fig5 reanalysis, aligned to finalized Fig1/Fig2, with outer-isolated module additions.
Run from this file location. Input snapshots in ../data. No future-visit interval is a predictor.
Conditional exploratory analysis: locked HI axes and upstream outer-training C choices.
"""
from pathlib import Path
import json, numpy as np,pandas as pd,joblib, warnings,platform
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
warnings.filterwarnings('error',category=__import__('sklearn').exceptions.ConvergenceWarning)
threadpool_limits(limits=1)
R=Path(__file__).resolve().parents[1];D=R/'data';M=R/'models';M.mkdir(exist_ok=True)
MODS=[('Bile acid','bile_acid'),('Oxidative lipid','oxidative_stress_lipid'),('Aromatic AA / indole','aromatic_amino_acid_indole'),('SCFA / butyrate','SCFA_butyrate')]
BR=['MBX_metabolomics_branch','MGX_EC_function_branch'];BASE=['score_t1','score_t2','delta12','abs_delta12','velocity12','dt12']
w=pd.read_csv(D/'Fig2_all_152_windows.csv').sort_values('window_index').reset_index(drop=True);assert len(w)==152 and w.participant_id.nunique()==66 and w.event_label.sum()==47
outer=pd.read_csv(D/'outer_folds.csv');fm=outer.groupby('participant_id').outer_fold.agg(['first','nunique']);assert fm['nunique'].eq(1).all();fmap=fm['first']
assert w.outer_fold.eq(w.participant_id.map(fmap)).all()
inner=pd.read_csv(D/'inner_folds.csv');params=pd.read_csv(D/'module_outer_selected_parameters.csv')
module=pd.read_csv(D/'module_descriptive_OOF_scores.csv').set_index('sample_id');assert module.index.is_unique
rng=np.random.default_rng(20261001)
def ci_cluster(d,fun,b=3000):
 ids=d.participant_id.unique();by={k:np.flatnonzero(d.participant_id.values==k) for k in ids}; vals=[]
 for _ in range(b):
  ix=np.concatenate([by[k] for k in rng.choice(ids,len(ids),replace=True)])
  v=fun(d.iloc[ix]);
  if np.isfinite(v):vals.append(v)
 return np.quantile(vals,[.025,.975]).tolist() if vals else [None,None]
# A, assay coverage (old C): mutually exclusive samples, overlapping participants.
c=pd.read_csv(D/'Fig5C_intersection_counts.csv');membership=pd.read_csv(D/'Fig5C_eligible_sample_membership.csv')
# B, internal locked MBX axis vs descriptive external module projection.
internal=pd.read_csv(D/'MBX546_locked_scores.csv');external=pd.read_csv(D/'fig5a_external_primary_analysis_rows.csv');external=external.rename(columns={'group':'diagnosis_label'});external['disease_group_binary']=(external.diagnosis_label!='nonIBD').astype(int)
rows=[]
for cohort,d,col in [('Internal MBX axis',internal,'HI_probability'),('External module projection',external,'score')]:
 for endpoint in ['IBD','CD','UC']:
  q=d if endpoint=='IBD' else d[d.diagnosis_label.isin([endpoint,'nonIBD'])]
  y=q.disease_group_binary;auc=roc_auc_score(y,q[col]);ci=ci_cluster(q,lambda a:roc_auc_score(a.disease_group_binary,a[col]) if a.disease_group_binary.nunique()==2 else np.nan)
  rows.append(dict(cohort=cohort,endpoint=endpoint,n=len(q),people=q.participant_id.nunique(),events=int(y.sum()),auc=auc,ci=ci))
pd.DataFrame(rows).to_csv(D/'panel_B_diagnostic_performance.csv',index=False)
# C, interval coverage and state-conditioned observed high-HI proportions.
introws=[];inttotal=[]
for bn,q in w.groupby('interval_group',sort=False):
 inttotal.append(dict(name=bn,n=len(q),people=q.participant_id.nunique()))
 for state in range(3):
  v=q[q.current_class==state];e=int(v.event_label.sum());n=len(v);ci=ci_cluster(v,lambda a:a.event_label.mean()) if e>0 and e<n else [None,None]
  introws.append(dict(interval=bn,state=state,n=n,people=v.participant_id.nunique(),events=e,rate=e/n if n else None,ci=ci,degenerate_bootstrap=e in [0,n]))
pd.DataFrame(introws).to_csv(D/'panel_C_interval_transition.csv',index=False)
old=pd.read_csv(D/'Fig5B_analysis_windows.csv');j=old.merge(w,on=['participant_id','sample_t1','sample_t2','sample_t3'],suffixes=('_old','_new'));assert len(j)==152
j['event_changed']=j.target_high_state!=j.event_label;j['current_state_changed']=j.state_t2.map({'HI_low_state':0,'HI_intermediate_state':1,'HI_high_state':2})!=j.current_class
j.to_csv(D/'old_new_window_label_audit.csv',index=False)
# D, descriptive updated OOF module scores, display z standardization only.
dist=[];long=[]
for br in BR:
 for label,s in MODS:
  raw=w.sample_t2.map(module[f'{br}__{s}']);valid=raw.notna();z=(raw-raw[valid].mean())/raw[valid].std(ddof=0)
  for state in range(3):
   take=valid&(w.current_class==state);a=z[take].values;qr=np.quantile(a,[.25,.5,.75]);iq=qr[2]-qr[0];inside=a[(a>=qr[0]-1.5*iq)&(a<=qr[2]+1.5*iq)]
   dist.append(dict(branch=br,module=label,state=state,n=len(a),people=w.loc[take,'participant_id'].nunique(),values=a.tolist(),q1=qr[0],median=qr[1],q3=qr[2],whisker_low=float(inside.min()),whisker_high=float(inside.max())))
   for ix in np.flatnonzero(take):long.append(dict(participant_id=w.loc[ix,'participant_id'],sample_id=w.loc[ix,'sample_t2'],state=state,branch=br,module=label,raw_score=raw[ix],z_score=z[ix]))
pd.DataFrame(long).to_csv(D/'panel_D_module_scores_by_current_state.csv',index=False)
# E, participant-median cross-omic correlations over the paired cohort.
paired=pd.read_csv(D/'paired_388_metadata.csv');pm=module.reindex(paired.sample_id);pm['participant_id']=paired.participant_id_x.values
mbxs=['SCFA_butyrate','aromatic_amino_acid_indole','bile_acid','mucin_carbohydrate','oxidative_stress_lipid'];mgxs=mbxs+['pathobiont_inflammatory_potential'];names={'mucin_carbohydrate':'Mucin / carbohydrate','pathobiont_inflammatory_potential':'Pathobiont potential',**{s:l for l,s in MODS}}
cols=[f'{BR[0]}__{s}' for s in mbxs]+[f'{BR[1]}__{s}' for s in mgxs];assert pm[cols].notna().all().all();u=pm.groupby('participant_id')[cols].median();assert len(u)==105
u.to_csv(D/'panel_E_participant_medians.csv');hr=[]
for i,a in enumerate(mbxs):
 for jj,b in enumerate(mgxs):
  rho,p=spearmanr(u[f'{BR[0]}__{a}'],u[f'{BR[1]}__{b}']);hr.append(dict(row=i,col=jj,mbx=names[a],mgx=names[b],rho=rho,p=p))
p=np.array([v['p'] for v in hr]);order=np.argsort(p);adj=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];qq=np.empty(len(p));qq[order]=np.minimum(adj,1)
for v,q in zip(hr,qq):v['q']=q
pd.DataFrame(hr).to_csv(D/'panel_E_correlation_BH.csv',index=False)
# F/G upstream fit and score inside each SAME outer participant split.
# Reuse outer-training-only C choices from finalized module layer. Inner train inputs
# are crossfit at participant level. The C selection and locked HI representations
# are conditional within this outer fold, matching Fig1's stated conditional scope.
audit=[];pre=[];foldmodule={};replay=[]
for br in BR:
 meta=pd.read_csv(D/f'{br}_metadata.csv');meta['fold']=meta.participant_id.map(fmap);assert meta.fold.notna().all();ids=meta.sample_id.astype(str);disease=meta.disease_group_binary.to_numpy();mg=meta.participant_id.values
 for label,s in MODS:
  X=sparse.load_npz(D/f'{br}__{s}_raw.npz').tocsr();assert X.shape[0]==len(meta)
  for f in range(5):
   tr=np.flatnonzero(meta.fold!=f);te=np.flatnonzero(meta.fold==f);pids=set(w.loc[w.outer_fold==f,'participant_id']);assert not pids&set(mg[tr])
   cc=float(params[(params.branch_model_name==br)&(params.module_name==s)&(params.outer_fold==f)].selected_C.iloc[0])
   def fit_score(train,test,name):
    cache=M/name
    if cache.exists():
     saved=joblib.load(cache);assert saved['train_participants']==sorted(set(mg[train])) and saved['test_participants']==sorted(set(mg[test]))
     xe=X[test].astype(float).copy()
     if saved['log1p']:xe.data=np.log1p(xe.data)
     xe=saved['scaler'].transform(xe[:,saved['keep']]);score=(np.asarray(xe@saved['model'].coef_[0]).ravel()/saved['coefficient_abs_sum']-saved['weighted_mean'])/saved['weighted_sd']
     audit.append(dict(branch=br,module=s,outer_fold=f,fit=name,n_train=len(train),n_test=len(test),train_test_participant_overlap=len(set(mg[train])&set(mg[test])),outer_heldout_training_overlap=len(pids&set(mg[train]))))
     pre.append(dict(branch=br,module=s,outer_fold=f,fit=name,C=saved['model'].C,features_retained=int(saved['keep'].sum()),log1p=saved['log1p'],mean=saved['weighted_mean'],sd=saved['weighted_sd']))
     return score
    xt=X[train].astype(float).copy();log=(xt.nnz==0 or xt.data.min()>=0)
    if log:xt.data=np.log1p(xt.data)
    nnz=np.asarray(xt.getnnz(axis=0)).ravel();mu=np.asarray(xt.mean(axis=0)).ravel();var=np.asarray(xt.power(2).mean(axis=0)).ravel()-mu**2;keep=(nnz>=2)&(var>1e-12);assert keep.any()
    scaler=StandardScaler(with_mean=False).fit(xt[:,keep]);xt=scaler.transform(xt[:,keep]);model=LogisticRegression(C=cc,solver='lbfgs',max_iter=5000).fit(xt,disease[train]);beta=model.coef_[0];den=abs(beta).sum();raw=np.asarray(xt@beta).ravel()/den;mu=raw.mean();sd=raw.std();assert sd>0
    xe=X[test].astype(float).copy()
    if log:xe.data=np.log1p(xe.data)
    xe=scaler.transform(xe[:,keep]);score=(np.asarray(xe@beta).ravel()/den-mu)/sd
    joblib.dump(dict(model=model,scaler=scaler,keep=keep,log1p=log,weighted_mean=mu,weighted_sd=sd,coefficient_abs_sum=den,train_participants=sorted(set(mg[train])),test_participants=sorted(set(mg[test])),source_module=s),M/name,compress=3)
    assert not set(mg[train])&set(mg[test]);audit.append(dict(branch=br,module=s,outer_fold=f,fit=name,n_train=len(train),n_test=len(test),train_test_participant_overlap=0,outer_heldout_training_overlap=len(pids&set(mg[train]))))
    pre.append(dict(branch=br,module=s,outer_fold=f,fit=name,C=cc,features_retained=int(keep.sum()),log1p=log,mean=mu,sd=sd))
    return score
   sc=np.full(len(meta),np.nan);sc[te]=fit_score(tr,te,f'upstream_{br}_{s}_outer_{f}.joblib')
   im=inner[inner.outer_fold==f].groupby('participant_id').inner_fold.agg(['first','nunique']);assert im['nunique'].eq(1).all();iv=meta.participant_id.map(im['first']).to_numpy()
   for jj in sorted(np.unique(iv[tr])):
    itr=np.flatnonzero((meta.fold!=f)&(iv!=jj));ival=np.flatnonzero((meta.fold!=f)&(iv==jj));sc[ival]=fit_score(itr,ival,f'upstream_{br}_{s}_outer_{f}_inner_{int(jj)}.joblib')
   assert np.isfinite(sc).all();foldmodule[(f,br,s)]=pd.Series(sc,index=ids)
   original=module.reindex(ids)[f'{br}__{s}'].to_numpy();diff=np.max(abs(sc[te]-original[te]));replay.append(dict(branch=br,module=s,outer_fold=f,outer_module_replay_max_abs_diff=diff));assert diff<1e-5
   pd.DataFrame({'sample_id':ids,'participant_id':mg,'score':sc,'role':np.where(meta.fold==f,'outer_test','inner_crossfit_training')}).to_csv(D/f'module_{br}_{s}_outer_{f}.csv',index=False)
   print('MODULE',br,s,'fold',f,'replay',diff,flush=True)
pd.DataFrame(audit).to_csv(D/'upstream_isolation_audit.csv',index=False);pd.DataFrame(pre).to_csv(D/'upstream_preprocessing.csv',index=False);pd.DataFrame(replay).to_csv(D/'upstream_replay.csv',index=False)
# Match cohort selection based on assay availability, never on future outcomes.
paired_mask=w.sample_t2.map(module[f'{BR[1]}__{MODS[0][1]}']).notna().to_numpy();assert paired_mask.sum()==112 and w.loc[paired_mask,'participant_id'].nunique()==63
# Cross-omic x-coordinate uses participant medians in this paired prediction subset.
px=w.loc[paired_mask,['participant_id','sample_t2']].copy();xc=[]
for label,s in MODS:
 for br in BR:
  px[f'{br}__{s}']=[foldmodule[(int(w.loc[i,'outer_fold']),br,s)].loc[w.loc[i,'sample_t2']] for i in np.flatnonzero(paired_mask)]
up=px.groupby('participant_id').median(numeric_only=True)
for label,s in MODS:
 rho=spearmanr(up[f'{BR[0]}__{s}'],up[f'{BR[1]}__{s}']).statistic;xc.append(dict(module=label,rho=rho,dissimilarity=1-abs(rho),n_windows=112,n_participants=63))
pd.DataFrame(xc).to_csv(D/'FG_cross_omic_participant_correlations.csv',index=False)
results=[];prtables=[]
for cohort,mask,branches in [('All windows',np.ones(len(w),bool),[BR[0]]),('Paired windows',paired_mask,BR)]:
 idx=np.flatnonzero(mask);pred={};basepr=np.full(len(w),np.nan);labely=w.event_label.values
 for f in range(5):
  axis=pd.read_csv(D/f'HI308_axis_fold_{f}_all_visits.csv').set_index('sample_id');d=w[['participant_id','sample_t1','sample_t2','sample_t3','dt12']].copy()
  for jjj in [1,2,3]:d[f'score_t{jjj}']=d[f'sample_t{jjj}'].map(axis.HI_score)
  d['delta12']=d.score_t2-d.score_t1;d['abs_delta12']=abs(d.delta12);d['velocity12']=d.delta12/d.dt12
  cfg=joblib.load(M/f'locked_trajectory_fold_{f}.joblib');threshold=cfg['high_threshold'];y=((d.score_t3>=threshold)|np.isclose(d.score_t3,threshold,rtol=0,atol=1e-12)).astype(int).values;tr=np.flatnonzero(mask&(w.outer_fold!=f));te=np.flatnonzero(mask&(w.outer_fold==f));assert not set(d.participant_id[tr])&set(d.participant_id[te]);assert np.array_equal(y[te],labely[te])
  cc=cfg['event_model'].named_steps['logisticregression'].C
  for br in branches:
   for label,s in MODS:
    d[f'{br}__{s}']=d.sample_t2.map(foldmodule[(f,br,s)])
  base=make_pipeline(StandardScaler(),LogisticRegression(C=cc,solver='lbfgs',max_iter=5000)).fit(d.loc[tr,BASE],y[tr]);basepr[te]=base.predict_proba(d.loc[te,BASE])[:,1];joblib.dump(base,M/f'{cohort.replace(" ","_")}_baseline_fold_{f}.joblib',compress=3)
  for br in branches:
   for label,s in MODS:
    key=br+'__'+s;cols=BASE+[key];assert d.loc[np.r_[tr,te],cols].notna().all().all();model=make_pipeline(StandardScaler(),LogisticRegression(C=cc,solver='lbfgs',max_iter=5000)).fit(d.loc[tr,cols],y[tr]);pred.setdefault(key,np.full(len(w),np.nan))[te]=model.predict_proba(d.loc[te,cols])[:,1];joblib.dump(dict(model=model,features=cols,outer_fold=f,train_window_indices=tr,test_window_indices=te),M/f'{cohort.replace(" ","_")}_{key}_fold_{f}.joblib',compress=3)
  if cohort=='All windows':assert np.max(abs(basepr[te]-w.event_probability[te]))<1e-7
 pp=w.loc[idx,['participant_id','sample_t1','sample_t2','sample_t3','outer_fold','event_label']].copy();pp['baseline_probability']=basepr[idx]
 for key,p in pred.items():pp[key]=p[idx]
 pp.to_csv(D/f'{cohort.replace(" ","_")}_OOF_predictions.csv',index=False)
 ids=pp.participant_id.unique();by={k:np.flatnonzero(pp.participant_id.values==k) for k in ids};draws=[]
 for _ in range(3000):
  ii=np.concatenate([by[k] for k in rng.choice(ids,len(ids),replace=True)])
  if len(np.unique(labely[idx][ii]))==2:draws.append(ii)
 yy=labely[idx];bp=basepr[idx];ba=roc_auc_score(yy,bp)
 for key,p in pred.items():
  br,s=key.split('__');label=dict((s,l) for l,s in MODS)[s];p=p[idx];delta=roc_auc_score(yy,p)-ba;bs=[roc_auc_score(yy[ii],p[ii])-roc_auc_score(yy[ii],bp[ii]) for ii in draws];x=next(v for v in xc if v['module']==label)
  rr=dict(cohort=cohort,branch=br,module=label,n=len(idx),people=len(ids),events=int(yy.sum()),baseline_auc=ba,added_auc=ba+delta,delta=delta,ci=np.quantile(bs,[.025,.975]).tolist(),dissimilarity=x['dissimilarity'],rho=x['rho']);results.append(rr)
  print('INCREMENT',rr,flush=True)
pd.DataFrame(results).to_csv(D/'FG_increment_performance.csv',index=False)
# A samples: participant overlaps and scored subset membership audit.
print('membership columns',membership.columns.tolist(),flush=True)
Dplot=dict(coverage=c.to_dict('records'),diagnostic=rows,interval_totals=inttotal,transitions=introws,distributions=dist,heatmap=hr,heatmap_rows=[names[s] for s in mbxs],heatmap_cols=[names[s] for s in mgxs],increments=results,current_counts=w.current_class.value_counts().sort_index().to_dict(),summary=dict(n_windows=152,n_participants=66,high_events=47,n_paired_windows=112,n_paired_participants=63,event_changed=int(j.event_changed.sum()),current_changed=int(j.current_state_changed.sum()),locked_auc=float(roc_auc_score(w.event_label,w.event_probability)),upstream_overlap_max=max(v['outer_heldout_training_overlap'] for v in audit),replay_diff_max=max(v['outer_module_replay_max_abs_diff'] for v in replay)))
(D/'plot_data.json').write_text(json.dumps(Dplot,ensure_ascii=False,indent=2))
(D/'summary.json').write_text(json.dumps(Dplot['summary'],indent=2));print('DONE',Dplot['summary'],flush=True)
