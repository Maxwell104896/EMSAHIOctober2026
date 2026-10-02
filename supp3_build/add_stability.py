from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE,trustworthiness
R=Path(__file__).resolve().parents[1];O=R/'supp3_final';d=pd.read_csv(O/'current_module_inputs.csv');X=StandardScaler().fit_transform(d.iloc[:,2:]);rows=[];neigh={}
for p in [5,15,30]:
 for seed in range(20260930,20260940):
  model=TSNE(perplexity=p,random_state=seed,init='random',learning_rate='auto',method='exact',max_iter=1000,n_jobs=1);Z=model.fit_transform(X);dist=((Z[:,None,:]-Z[None,:,:])**2).sum(2);np.fill_diagonal(dist,np.inf);neigh[(p,seed)]=np.argsort(dist,axis=1)[:,:5];rows.append(dict(perplexity=p,seed=seed,initialization='random',trustworthiness_k5=trustworthiness(X,Z,n_neighbors=5),kl_divergence=model.kl_divergence_))
for row in rows:
 p,s=row['perplexity'],row['seed'];a=neigh[(p,s)];bs=[neigh[(p,t)] for t in range(20260930,20260940) if t!=s];row['mean_pairwise_knn_jaccard_k5']=np.mean([len(set(a[i])&set(b[i]))/len(set(a[i])|set(b[i])) for b in bs for i in range(len(a))])
pd.DataFrame(rows).to_csv(O/'random_initialization_stability.csv',index=False)
# Cross-perplexity neighborhood agreement for the three displayed maps.
d=pd.read_csv(O/'sensitivity_coordinates.csv');ns={}
for p in [5,15,30]:
 Z=d.loc[(d.perplexity==p)&(d.seed==20260930),['tSNE1','tSNE2']].to_numpy();v=((Z[:,None]-Z[None,:])**2).sum(2);np.fill_diagonal(v,np.inf);ns[p]=np.argsort(v,axis=1)[:,:5]
pd.DataFrame([dict(perplexity_a=a,perplexity_b=b,mean_knn_jaccard_k5=np.mean([len(set(ns[a][i])&set(ns[b][i]))/len(set(ns[a][i])|set(ns[b][i])) for i in range(len(X))])) for a,b in [(5,15),(5,30),(15,30)]]).to_csv(O/'cross_perplexity_agreement.csv',index=False)
print(pd.DataFrame(rows).groupby('perplexity')[['trustworthiness_k5','mean_pairwise_knn_jaccard_k5']].agg(['min','median','max']).to_string())
