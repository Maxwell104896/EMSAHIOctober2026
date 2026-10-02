from pathlib import Path
import pandas as pd,numpy as np,joblib,json
from scipy import sparse
from sklearn.metrics import mean_absolute_error,r2_score
R=Path(__file__).resolve().parents[1];D=R/'data';w=pd.read_csv(D/'locked_predictions.csv').sort_values('window_index').reset_index(drop=True);p=pd.read_csv(D/'all_model_predictions.csv');diff=[];updiff=[]
for f in range(5):
 d=w.copy();ax=pd.read_csv(D/f'HI308_axis_fold_{f}_all_visits.csv').set_index('sample_id')
 for j in [1,2,3]:d[f'score_t{j}']=d[f'sample_t{j}'].map(ax.HI_score)
 d['delta12']=d.score_t2-d.score_t1;d['abs_delta12']=abs(d.delta12);d['velocity12']=d.delta12/d.dt12
 mods=['bile_acid','oxidative_stress_lipid','aromatic_amino_acid_indole','SCFA_butyrate'];meta=pd.read_csv(D/'MBX_metabolomics_branch_metadata.csv');ids=meta.sample_id.to_numpy()
 for mod in mods:
  mm=pd.read_csv(D/f'module_MBX_metabolomics_branch_{mod}_outer_{f}.csv').set_index('sample_id');d[mod]=d.sample_t2.map(mm.score)
  raw=sparse.load_npz(D/f'MBX_metabolomics_branch__{mod}_raw.npz').astype(float)
  for file in (R/'models').glob(f'upstream_MBX_metabolomics_branch_{mod}_outer_{f}_*.joblib'):
   m=joblib.load(file);rows=np.flatnonzero(meta.participant_id.isin(m['test_participants']));x=raw[rows].copy()
   if m['log1p']:x.data=np.log1p(x.data)
   score=(m['scaler'].transform(x[:,m['keep']])@m['model'].coef_.ravel()/m['coefficient_abs_sum']-m['weighted_mean'])/m['weighted_sd'];updiff.append(np.max(abs(score-mm.loc[ids[rows],'score'].to_numpy())))
 for k in ['ridge','modules']:
  m=joblib.load(R/'models'/f'{k}_fold_{f}.joblib');te=m['test_window_indices'];pr=m['model'].predict(d.iloc[te][m['features']])+d.iloc[te].score_t2.to_numpy();diff.append(np.max(abs(pr-p.iloc[te][k])));assert 'dt23' not in m['features'];assert not set(w.iloc[m['train_window_indices']].participant_id)&set(w.iloc[te].participant_id)
assert max(diff)<1e-10 and max(updiff)<1e-10
v={'downstream_replay_max':float(max(diff)),'upstream_module_replay_max':float(max(updiff)),'n_windows':len(w),'n_participants':w.participant_id.nunique(),'future_interval_used':False,'state_counts':p.observed_class.value_counts().sort_index().to_dict()};(R/'replay_verification.json').write_text(json.dumps(v,indent=2));print(v)
