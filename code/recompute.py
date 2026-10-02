from pathlib import Path
import pandas as pd,numpy as np,json,joblib
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,r2_score
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[1];D=R/'data';M=R/'models';M.mkdir(exist_ok=True)
w=pd.read_csv(D/'locked_predictions.csv').sort_values('window_index').reset_index(drop=True);sel=pd.read_csv(D/'trajectory_selected_configurations.csv');g=w.participant_id.to_numpy();n=len(w)
mods=['bile_acid','oxidative_stress_lipid','aromatic_amino_acid_indole','SCFA_butyrate'];alpha=np.logspace(-3,3,13);families=[]
for h in [1,2]:
 for k in [0,1,3]:
  fs=(['score_t2'] if h==1 else ['score_t1','score_t2'])+{0:[],1:['delta12'],3:['delta12','abs_delta12','velocity12']}[k]+['dt12'];families.append((h,k,fs))
pr={k:np.full(n,np.nan) for k in ['ridge','modules','shrink_ridge','shrink_modules']};sens=[];sels=[];audit=[];rawinner=[];replay=[];familypr={i:np.full(n,np.nan) for i in range(6)}
for f in range(5):
 d=w.copy();ax=pd.read_csv(D/f'HI308_axis_fold_{f}_all_visits.csv').set_index('sample_id')
 for j in [1,2,3]:d[f'score_t{j}']=d[f'sample_t{j}'].map(ax.HI_score)
 d['delta12']=d.score_t2-d.score_t1;d['abs_delta12']=abs(d.delta12);d['velocity12']=d.delta12/d.dt12;d['target_delta23']=d.score_t3-d.score_t2
 for mod in mods:
  mm=pd.read_csv(D/f'module_MBX_metabolomics_branch_{mod}_outer_{f}.csv').set_index('sample_id');d[mod]=d.sample_t2.map(mm.score)
 tr=np.flatnonzero(w.outer_fold!=f);te=np.flatnonzero(w.outer_fold==f);spl=list(GroupKFold(4).split(d.iloc[tr],groups=g[tr]));y=d.target_delta23.to_numpy();obs=d.score_t3.to_numpy();cur=d.score_t2.to_numpy()
 assert not set(g[tr])&set(g[te]);assert d[mods].notna().all().all()
 def fit(fs,a,tt,vv):return make_pipeline(StandardScaler(),Ridge(alpha=float(a))).fit(d.iloc[tt][fs],y[tt])
 def innerpred(fs,a):
  ip=np.full(len(tr),np.nan)
  for it,iv in spl:
   model=fit(fs,a,tr[it],tr[iv]);ip[iv]=model.predict(d.iloc[tr[iv]][fs])+cur[tr[iv]]
  return ip
 for fi,(h,k,fs) in enumerate(families):
  candidates=[]
  for a in alpha:
   ip=innerpred(fs,a);err=float(mean_absolute_error(obs[tr],ip));sens.append(dict(outer_fold=f,family=fi,history=h,dynamics=k,alpha=float(a),mae=err));candidates.append((err,float(a)))
  best=min(candidates)[1];model=fit(fs,best,tr,te);familypr[fi][te]=model.predict(d.iloc[te][fs])+cur[te]
 c=sel[sel.outer_fold==f].iloc[0];basefs=c.features.split('|');aa=c.alpha
 for name,fs in [('ridge',basefs),('modules',basefs+mods)]:
  if name=='modules':
   rank=[]
   for a in alpha:
    ip=innerpred(fs,a);rank.append((mean_absolute_error(obs[tr],ip),float(a),ip))
   _,a,ip=min(rank,key=lambda q:q[:2])
  else:a=aa;ip=innerpred(fs,a)
  model=fit(fs,a,tr,te);pred=model.predict(d.iloc[te][fs])+cur[te];pr[name][te]=pred
  dd=ip-cur[tr];lam=float(np.clip(np.dot(dd,y[tr])/np.dot(dd,dd),0,1)) if np.dot(dd,dd)>0 else 0
  pr['shrink_'+name][te]=cur[te]+lam*(pred-cur[te]);joblib.dump(dict(model=model,features=fs,lambda_shrinkage=lam,train_window_indices=tr,test_window_indices=te),M/f'{name}_fold_{f}.joblib',compress=3)
  sels.append(dict(outer_fold=f,model=name,features='|'.join(fs),alpha=float(a),shrinkage_lambda=lam));audit.append(dict(outer_fold=f,model=name,participant_overlap=len(set(g[tr])&set(g[te])),future_interval_used=False))
  for jj,ix in enumerate(tr):rawinner.append(dict(outer_fold=f,model=name,window_index=int(ix),observed_HI=float(obs[ix]),current_HI=float(cur[ix]),inner_pred_HI=float(ip[jj])))
 replay.append(float(np.max(abs(pr['ridge'][te]-w.pred_HI.to_numpy()[te]))))
assert max(replay)<1e-10
pr['persistence']=w.persistence.to_numpy();pr['linear']=w.linear.to_numpy();obs=w.score_t3.to_numpy();out=w[['participant_id','sample_t1','sample_t2','sample_t3','outer_fold','window_index','score_t1','score_t2','score_t3','low_threshold','high_threshold','observed_class']].copy()
for k,p in pr.items():assert np.isfinite(p).all();out[k]=p
out.to_csv(D/'all_model_predictions.csv',index=False);pd.DataFrame(sens).to_csv(D/'inner_parameter_sensitivity.csv',index=False);pd.DataFrame(sels).to_csv(D/'selected_configurations.csv',index=False);pd.DataFrame(rawinner).to_csv(D/'inner_calibration_inputs.csv',index=False);pd.DataFrame(audit).to_csv(D/'isolation_audit.csv',index=False)
rng=np.random.default_rng(20261002);ids=np.unique(g);groups=[np.flatnonzero(g==p) for p in ids];draws=[np.concatenate([groups[i] for i in rng.integers(0,len(ids),len(ids))]) for _ in range(2000)]
def metrics(y,p):return {'mae':float(mean_absolute_error(y,p)),'r2':float(r2_score(y,p)),'spearman':float(spearmanr(y,p).statistic)}
perf=[]
for name,p in pr.items():
 q={'model':name,**metrics(obs,p)};bs=np.array([[mean_absolute_error(obs[ii],p[ii]),r2_score(obs[ii],p[ii])] for ii in draws]);q['mae_ci']=np.quantile(bs[:,0],[.025,.975]).tolist();q['r2_ci']=np.quantile(bs[:,1],[.025,.975]).tolist();ds=[mean_absolute_error(obs[ii],p[ii])-mean_absolute_error(obs[ii],pr['ridge'][ii]) for ii in draws];q['delta_mae_vs_locked']=q['mae']-mean_absolute_error(obs,pr['ridge']);q['delta_mae_ci']=np.quantile(ds,[.025,.975]).tolist();perf.append(q)
pd.DataFrame(perf).to_csv(D/'performance_and_paired_differences.csv',index=False)
# Frozen full OOF prediction quantiles define bins. Resample participants, retain bin labels.
bins=pd.qcut(pr['ridge'],5,labels=False);cal=[]
for b in range(5):
 ix=np.flatnonzero(bins==b);means=[]
 for ii in draws:
  jj=ii[bins[ii]==b]
  if len(jj):means.append(float(np.mean(obs[jj])))
 cal.append(dict(bin=b,n=len(ix),people=len(np.unique(g[ix])),predicted=float(np.mean(pr['ridge'][ix])),observed=float(np.mean(obs[ix])),ci=np.quantile(means,[.025,.975]).tolist()))
slope,intercept=np.polyfit(pr['ridge'],obs,1);coef=np.array([np.polyfit(pr['ridge'][ii],obs[ii],1) for ii in draws]);calstats=dict(slope=float(slope),intercept=float(intercept),slope_ci=np.quantile(coef[:,0],[.025,.975]).tolist(),intercept_ci=np.quantile(coef[:,1],[.025,.975]).tolist())
pd.DataFrame(cal).to_csv(D/'locked_calibration_bins.csv',index=False);pd.DataFrame({'participant_id':g,'window_index':w.window_index,'bin':bins,'predicted':pr['ridge'],'observed':obs,'residual':obs-pr['ridge']}).to_csv(D/'locked_calibration_membership.csv',index=False)
pd.DataFrame({**{'participant_id':g,'window_index':w.window_index},**{f'family_{i}':p for i,p in familypr.items()}}).to_csv(D/'family_nested_OOF_predictions.csv',index=False)
plot=dict(sensitivity=sens,calibration=cal,calstats=calstats,performance=perf,points={k:[dict(x=float(p[i]),y=float(obs[i]),state=int(w.observed_class.iloc[i])) for i in range(n)] for k,p in pr.items()},n=n,people=len(ids),replay_max=max(replay))
(D/'plot_data.json').write_text(json.dumps(plot));(R/'verification.json').write_text(json.dumps(dict(n=n,participants=len(ids),locked_ridge_replay_max=max(replay),future_interval_predictors=False,outer_participant_overlap_max=max(v['participant_overlap'] for v in audit),bootstrap_replicates=2000,calibration=calstats),indent=2));print(json.dumps({'performance':perf,'calstats':calstats,'replay':max(replay)},indent=2))
