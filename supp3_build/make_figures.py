from pathlib import Path
import itertools,json,hashlib,shutil
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE,trustworthiness
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from vectorize import export_native_geometry
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'supp3_final';R=ROOT/'inputs/fig1_materials/Fig1_submission/results';O.mkdir(exist_ok=True)
p=pd.read_csv(R/'corrected_trajectory_predictions.csv');m=pd.read_csv(R/'modules/EMSA_module_scores_wide_by_branch_module.csv');key=pd.read_csv(ROOT/'supp_final/module_key.csv');cols=key.module_column.tolist();assert len(cols)==19
assert len(p)==152 and p.participant_id.nunique()==66 and p.event_label.sum()==47
assert all(p.groupby('participant_id').outer_fold.nunique()==1)
p['delta12']=p.score_t2-p.score_t1;p['abs_delta12']=abs(p.delta12);p['velocity12']=p.delta12/p.dt12;p['target_delta23']=p.score_t3-p.score_t2
p.to_csv(O/'locked_trajectory_input.csv',index=False)
cur=p.merge(m[['participant_id','sample_id']+cols],left_on=['participant_id','sample_t2'],right_on=['participant_id','sample_id'],validate='many_to_one');complete=cur[cols].notna().all(axis=1);inc=cur.loc[complete].copy();assert len(inc)==111 and inc.participant_id.nunique()==63
assert inc.persistence_class.value_counts().to_dict()=={1:38,0:37,2:36}
cur[['participant_id','sample_t2','persistence_class','score_t2','dt12']].assign(included=complete.to_numpy(),missing_modules=cur[cols].isna().apply(lambda z:'|'.join(z.index[z]),axis=1)).to_csv(O/'map_inclusion.csv',index=False)
inc[['participant_id','sample_t2']+cols].to_csv(O/'current_module_inputs.csv',index=False)
scaler=StandardScaler();X=scaler.fit_transform(inc[cols]);pd.DataFrame({'module':cols,'mean':scaler.mean_,'scale':scaler.scale_}).to_csv(O/'map_scaling.csv',index=False)
seeds=[20260930+i for i in range(10)];coordinates={};stable=[];neighbors={}
for per,seed in itertools.product([5,15,30],seeds):
 ts=TSNE(n_components=2,perplexity=per,init='pca',learning_rate='auto',random_state=seed,max_iter=1000,method='exact',n_jobs=1);Z=ts.fit_transform(X);coordinates[(per,seed)]=Z
 D=((Z[:,None,:]-Z[None,:,:])**2).sum(2);np.fill_diagonal(D,np.inf);neighbors[(per,seed)]=np.argsort(D,axis=1)[:,:5]
 stable.append({'perplexity':per,'seed':seed,'trustworthiness_k5':trustworthiness(X,Z,n_neighbors=5),'kl_divergence':ts.kl_divergence_})
for row in stable:
 per,seed=row['perplexity'],row['seed'];a=neighbors[(per,seed)];others=[neighbors[(per,s)] for s in seeds if s!=seed]
 row['mean_pairwise_knn_jaccard_k5']=np.mean([len(set(a[i])&set(b[i]))/len(set(a[i])|set(b[i])) for b in others for i in range(len(a))])
pd.DataFrame(stable).to_csv(O/'embedding_stability.csv',index=False)
Z=coordinates[(15,seeds[0])];meta=inc[['participant_id','sample_t2','persistence_class','score_t2']].reset_index(drop=True);meta[['tSNE1','tSNE2']]=Z;meta.to_csv(O/'embedding_coordinates.csv',index=False)
ss=[]
for (per,seed),v in coordinates.items():
 q=meta[['participant_id','sample_t2','persistence_class']].copy();q['perplexity']=per;q['seed']=seed;q[['tSNE1','tSNE2']]=v;ss.append(q)
pd.concat(ss).to_csv(O/'sensitivity_coordinates.csv',index=False)
features=['MBX_metabolomics_branch__bile_acid','MBX_metabolomics_branch__oxidative_stress_lipid','MGX_EC_function_branch__SCFA_butyrate','MGX_EC_function_branch__aromatic_amino_acid_indole','MGX_EC_function_branch__mucin_carbohydrate','MGX_EC_function_branch__pathobiont_inflammatory_potential']
over=[]
for letter,col in zip('BCDEFG',features):
 q=meta.copy();q['panel']=letter;q['module']=col;q['raw_score']=inc[col].to_numpy();q['standardized_score']=X[:,cols.index(col)];q['display_clipped']=abs(q.standardized_score)>2;over.append(q)
pd.concat(over).to_csv(O/'overlay_values.csv',index=False)
# Conditional supplementary Ridge search on locked upstream HI scores, keeping locked participant outer folds.
g=p.participant_id.to_numpy();y=p.target_delta23.to_numpy();current=p.score_t2.to_numpy();configs=[]
for history,dynamics,alpha in itertools.product([1,2],[0,1,3],np.logspace(-3,3,13)):
 features2=(['score_t2'] if history==1 else ['score_t1','score_t2'])+{0:[],1:['delta12'],3:['delta12','abs_delta12','velocity12']}[dynamics]+['dt12'];configs.append(dict(config_id=f'R{len(configs)+1:03d}',history=history,dynamics=dynamics,alpha=float(alpha),n_features=len(features2),features=features2))
outer=[(np.flatnonzero(p.outer_fold.to_numpy()!=f),np.flatnonzero(p.outer_fold.to_numpy()==f)) for f in sorted(p.outer_fold.unique())];assert len(outer)==5
A={c['config_id']:p[c['features']].to_numpy() for c in configs}
def fit(c,tr,te):
 model=make_pipeline(StandardScaler(),Ridge(alpha=c['alpha']));model.fit(A[c['config_id']][tr],y[tr]);return model.predict(A[c['config_id']][te])+current[te]
def evaluate(c,splits):
 pred=np.full(len(p),np.nan);errs=[]
 for tr,te in splits:
  assert not set(g[tr])&set(g[te]);v=fit(c,tr,te);pred[te]=v;errs.extend(abs(v-p.score_t3.to_numpy()[te]))
 return np.mean(errs),pred
records=[];predictions={}
for c in configs:
 error,v=evaluate(c,outer);predictions[c['config_id']]=v;records.append({**c,'features':'|'.join(c['features']),'mae':error})
table=pd.DataFrame(records).sort_values(['mae','n_features','alpha','config_id']);best=table.iloc[0];table.to_csv(O/'configuration_results.csv',index=False);pd.DataFrame(predictions).to_csv(O/'configuration_oof_predictions.csv',index_label='row_index')
nested=np.full(len(p),np.nan);selection=[];inner_results=[]
for fold,(tr,te) in zip(sorted(p.outer_fold.unique()),outer):
 inner=[(tr[a],tr[b]) for a,b in GroupKFold(4).split(p.iloc[tr],y[tr],g[tr])];rank=[]
 for c in configs:
  error,_=evaluate(c,inner);rank.append((error,c['n_features'],c['alpha'],c['config_id']));inner_results.append(dict(outer_fold=int(fold),config_id=c['config_id'],inner_mae=error))
 chosen=next(c for c in configs if c['config_id']==min(rank)[3]);nested[te]=fit(chosen,tr,te);selection.append(dict(outer_fold=int(fold),config_id=chosen['config_id'],history=chosen['history'],dynamics=chosen['dynamics'],alpha=chosen['alpha'],features='|'.join(chosen['features']),inner_mae=min(rank)[0],test_windows=len(te),test_participants=len(set(g[te]))))
pd.DataFrame(selection).to_csv(O/'nested_selections.csv',index=False);pd.DataFrame(inner_results).to_csv(O/'nested_inner_search.csv',index=False)
q=p[['participant_id','sample_t1','sample_t2','sample_t3','outer_fold','score_t2','score_t3','pred_HI']].copy();q['nested_prediction']=nested;q['best_grid_prediction']=predictions[best.config_id];q.to_csv(O/'updated_oof_predictions.csv',index=False)
rng=np.random.default_rng(20261002);ids=np.unique(g);idx=[np.concatenate([np.flatnonzero(g==i) for i in rng.choice(ids,len(ids),replace=True)]) for _ in range(3000)];metrics=[]
for name,v in [('nested_regressor_selection',nested),('persistence',current),('descriptive_grid_minimum',predictions[best.config_id]),('locked_core_reference',p.pred_HI.to_numpy())]:
 error=abs(v-p.score_t3.to_numpy());boots=[error[j].mean() for j in idx];lo,hi=np.quantile(boots,[.025,.975]);metrics.append(dict(model=name,mae=error.mean(),mae_ci_low=lo,mae_ci_high=hi,rmse=np.sqrt(mean_squared_error(p.score_t3,v)),r2=r2_score(p.score_t3,v)))
pd.DataFrame(metrics).to_csv(O/'performance.csv',index=False);diff=abs(nested-p.score_t3.to_numpy())-abs(current-p.score_t3.to_numpy());ci=np.quantile([diff[j].mean() for j in idx],[.025,.975]);pd.DataFrame([dict(comparison='nested_minus_persistence',mae_difference=diff.mean(),ci_low=ci[0],ci_high=ci[1],bootstrap=3000)]).to_csv(O/'paired_mae_difference.csv',index=False)
# Inclusion descriptors and cluster confidence intervals.
rows=[]
for status in [True,False]:
 q=cur.loc[complete==status];rows.append(dict(included=status,windows=len(q),people=q.participant_id.nunique(),HI_mean=q.score_t2.mean(),HI_median=q.score_t2.median(),dt12_median=q.dt12.median(),low=sum(q.persistence_class==0),intermediate=sum(q.persistence_class==1),high=sum(q.persistence_class==2)))
pd.DataFrame(rows).to_csv(O/'inclusion_descriptors.csv',index=False)
plt.rcParams.update({'font.family':'Nimbus Roman','font.size':11,'axes.labelsize':11,'axes.titlesize':11,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,'text.color':'black','axes.labelcolor':'black','svg.fonttype':'none','pdf.fonttype':42})
STATE=['#4D8CAD','#62A69D','#D18A56'];names=['Low','Intermediate','High'];W,H=180/25.4,250/25.4
fig=plt.figure(figsize=(W,H),facecolor='white');fig.text(.025,.985,'Supplementary Figure S3',fontsize=14,va='top');fig.text(.025,.958,'Current molecular landscape and embedding sensitivity',fontsize=12,va='top')
def mapax(rect,zz,values=None):
 ax=fig.add_axes(rect);ax.set_aspect('equal',adjustable='box');ax.set_xticks([]);ax.set_yticks([])
 for sp in ax.spines.values():sp.set_visible(False)
 if values is None:
  for k in range(3):
   mask=meta.persistence_class==k;ax.scatter(zz[mask,0],zz[mask,1],s=12,color=STATE[k],linewidths=.25,edgecolors='white')
 else:ax.scatter(zz[:,0],zz[:,1],s=12,color=plt.get_cmap('RdBu_r')((np.clip(values,-2,2)+2)/4),linewidths=.25,edgecolors='white')
 return ax
def panel(x,y,l,t):fig.text(x,y,l,fontsize=16,weight='bold',va='top');fig.text(x+.042,y-.002,t,fontsize=11,va='top')
panel(.03,.912,'A','Current HI state');ax=mapax([.06,.726,.30,.156],Z);ax.set_xlabel('t-SNE 1');ax.set_ylabel('t-SNE 2')
for k,n in enumerate([37,38,36]):fig.text(.45,.871-k*.034,names[k]+f': {n} visits',fontsize=11);fig.add_artist(Line2D([.415],[.876-k*.034],marker='o',color=STATE[k],ls='none',markersize=5,transform=fig.transFigure))
fig.text(.415,.748,'111 complete visits / 63 people\n19 updated MBX + MGX modules\nLabels: observed current t2 state',fontsize=11,linespacing=1.5)
titles=['Bile acid\nMBX','Oxidative lipid\nMBX','SCFA / butyrate\nMGX EC','AA / indole\nMGX EC','Mucin / carbohydrate\nMGX EC','Pathobiont\nMGX EC']
for j,(col,t) in enumerate(zip(features,titles)):
 x=.035+(j%3)*.323;y=.688-(j//3)*.213;panel(x,y,chr(66+j),t);mapax([x+.015,y-.180,.270,.131],Z,X[:,cols.index(col)])
# Explicit vector color key: no raster gradients.
cax=fig.add_axes([.22,.246,.57,.010]);cm=plt.get_cmap('RdBu_r')
for i in range(64):cax.add_patch(Rectangle((-2+4*i/64,0),4/64,1,facecolor=cm((i+.5)/64),edgecolor='none'))
cax.set(xlim=(-2,2),ylim=(0,1),yticks=[],xticks=[-2,0,2]);cax.tick_params(length=2);cax.spines[['top','left','right']].set_visible(False);fig.text(.5,.209,'Module z; colors saturate outside −2 to +2',ha='center',fontsize=11)
for j,per in enumerate([5,15,30]):
 x=.035+j*.323;panel(x,.185,chr(72+j),f'Perplexity {per}');ax=mapax([x+.015,.061,.270,.094],coordinates[(per,seeds[0])]);ax.set_xlabel('t-SNE 1')
fig.text(.025,.025,'Exploratory t-SNE; all module overlays share panel A coordinates',fontsize=11)
export_native_geometry(fig,O/'S3',ROOT/'supp3_build/s3.json');plt.close(fig)
# Separate model exploration figure; core performance is supplied only as a consistency reference.
fig=plt.figure(figsize=(W,H),facecolor='white');fig.text(.025,.985,'Supplementary Figure S4',fontsize=14,va='top');fig.text(.025,.956,'Conditional trajectory-regressor configuration search',fontsize=12,va='top');panel(.025,.918,'A','78 configurations; future interval excluded')
ax=fig.add_axes([.07,.677,.84,.157]);arr=table[['history','dynamics','n_features','alpha','mae']].to_numpy();arr[:,3]=np.log10(arr[:,3]);lo=np.array([1,0,2,-3,np.floor(arr[:,4].min()*100)/100-.01]);hi=np.array([2,3,6,3,np.ceil(arr[:,4].max()*100)/100+.01]);zz=(arr-lo)/(hi-lo);color=plt.get_cmap('viridis_r');norm=matplotlib.colors.Normalize(arr[:,4].min(),arr[:,4].max())
for i in range(len(table)-1,-1,-1):ax.plot(range(5),zz[i],color=color(norm(arr[i,4])),lw=.65,alpha=.25)
ax.plot(range(5),zz[0],color='#006B53',lw=2,marker='o',markersize=3)
ticks=[[1,2],[0,1,3],[2,3,4,5,6],[-3,-2,-1,0,1,2,3],np.linspace(lo[4],hi[4],4)]
for j,ts in enumerate(ticks):
 ax.plot([j,j],[0,1],color='#555555',lw=.8)
 for v in ts:
  lab=f'10^{int(v)}' if j==3 else (f'{v:.3f}' if j==4 else str(int(v)));ax.text(j-.055,(v-lo[j])/(hi[j]-lo[j]),lab,ha='right',va='center',fontsize=11,bbox={'facecolor':'white','edgecolor':'none','pad':.1,'alpha':.85})
for j,label in enumerate(['HI history\n(visits)','Dynamics\n(columns)','Input\ncolumns','Ridge\nalpha','Outer-test\nMAE']):ax.text(j,1.12,label,ha='center',va='bottom',fontsize=11)
ax.set(xlim=(-.29,4.13),ylim=(-.03,1.03));ax.axis('off');fig.text(.055,.64,f'Green: grid minimum {best.config_id}, alpha {best.alpha:g}, MAE {best.mae:.3f}',fontsize=11);fig.text(.055,.612,'152 windows / 66 people; locked participant outer folds',fontsize=11);fig.text(.055,.584,'Dynamics: 0 none; 1 change; 3 change, absolute change, velocity',fontsize=11)
panel(.025,.538,'B','Nested selection and persistence')
met=pd.DataFrame(metrics).set_index('model');ax=fig.add_axes([.39,.363,.53,.118])
for y0,name,co in [(1,'nested_regressor_selection','#62A69D'),(0,'persistence','#4D8CAD')]:
 r=met.loc[name];ax.errorbar(r.mae,y0,xerr=[[r.mae-r.mae_ci_low],[r.mae_ci_high-r.mae]],fmt='o',color=co,ms=4,capsize=3,lw=1);ax.text(.98,y0,f'{r.mae:.3f}',transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=11)
ax.set_yticks([1,0],['Nested Ridge','Persistence']);ax.set_xlabel('MAE (95% participant-bootstrap CI)');ax.set_ylim(-.7,1.7);ax.spines[['top','right']].set_visible(False)
panel(.025,.306,'C','Paired error difference')
ax=fig.add_axes([.17,.208,.71,.056]);ax.axvline(0,color='#777777',ls='--',lw=.8);ax.errorbar(diff.mean(),0,xerr=[[diff.mean()-ci[0]],[ci[1]-diff.mean()]],fmt='o',color='#62A69D',capsize=3,ms=4);ax.set_yticks([]);ax.set_xlabel('MAE: nested Ridge − persistence');ax.spines[['top','right','left']].set_visible(False);ax.set_ylim(-1,1)
fig.text(.5,.145,f'Difference {diff.mean():+.3f}; 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]',ha='center',fontsize=11)
fig.text(.035,.104,'Conditional on locked upstream HI scores; regressor selection only',fontsize=11);fig.text(.035,.074,'Not a replacement for the locked main-model performance',fontsize=11);fig.text(.035,.044,'Participant-grouped inner search; 3,000 bootstrap resamples',fontsize=11)
export_native_geometry(fig,O/'S4',ROOT/'supp3_build/s4.json');plt.close(fig)
clipped=pd.concat(over).groupby('panel').display_clipped.sum().to_dict();summary={'windows':152,'participants':66,'next_high':47,'map_visits':111,'map_people':63,'map_states':[37,38,36],'features':19,'configurations':78,'future_interval_inputs':False,'best_grid':best.to_dict(),'performance':metrics,'paired_difference':{'estimate':diff.mean(),'ci':ci.tolist()},'clipped_points':{k:int(v) for k,v in clipped.items()},'upstream_scope':'Conditional on locked exported HI; no claim of full-pipeline revalidation.'};(O/'analysis_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
