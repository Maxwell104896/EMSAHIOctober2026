from pathlib import Path
import shutil,hashlib,zipfile
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'Fig4C_reference_style_temporal_module_matrix'
OUT=ROOT/'Fig4C_extended_temporal_module_atlas'
for sub in ['figure','source','derived','code']:(OUT/sub).mkdir(parents=True,exist_ok=True)
for p in (BASE/'source').iterdir():shutil.copy2(p,OUT/'source'/p.name)
v=pd.read_csv(BASE/'derived/Fig4C_92_complete_windows.csv')
meta=pd.read_csv(BASE/'derived/Fig4C_module_order.csv');cols=meta.module_column.tolist();labels=meta.label.tolist();groups=meta.branch.tolist()
a=v[['t2__'+c for c in cols]].to_numpy();b=v[['t3__'+c for c in cols]].to_numpy();assert a.shape==(92,19) and np.isfinite(a).all() and np.isfinite(b).all()
def corr(x,y):
 rx=pd.DataFrame(x).rank().to_numpy();ry=pd.DataFrame(y).rank().to_numpy();return np.corrcoef(rx.T,ry.T)[:19,19:]
same=corr(a,a);lag=corr(a,b);delta=lag-same
# Participant bootstrap preserves all windows of a sampled participant; stability is descriptive.
ids=v.participant_id.to_numpy();unique=np.unique(ids);idx={pid:np.flatnonzero(ids==pid) for pid in unique};rng=np.random.default_rng(3486);B=500
boot=np.empty((B,19,19),dtype=np.float32)
for it in range(B):
 sampled=rng.choice(unique,size=len(unique),replace=True);ii=np.concatenate([idx[pid] for pid in sampled]);boot[it]=corr(a[ii],b[ii])-corr(a[ii],a[ii])
lo=np.quantile(boot,.025,axis=0);hi=np.quantile(boot,.975,axis=0);stable=(lo>0)|(hi<0)
assert np.isfinite(boot).all()
for name,mat in [('same_visit_spearman',same),('next_visit_spearman',lag),('lag_minus_same',delta),('cluster_bootstrap_delta_lower',lo),('cluster_bootstrap_delta_upper',hi)]:pd.DataFrame(mat,index=cols,columns=cols).to_csv(OUT/'derived'/f'{name}.csv',float_format='%.8f')
pd.DataFrame(stable,index=cols,columns=cols).to_csv(OUT/'derived'/'bootstrap_direction_stability.csv')
v.to_csv(OUT/'derived'/'paired_window_matrix.csv',index=False);meta.to_csv(OUT/'derived'/'module_order.csv',index=False)
persistence=pd.DataFrame({'module_column':cols,'branch':groups,'same_visit_diagonal':np.diag(same),'cross_visit_same_module_rho':np.diag(lag),'difference_from_same_visit':np.diag(delta),'cluster_bootstrap_lower':np.diag(lo),'cluster_bootstrap_upper':np.diag(hi)})
persistence.to_csv(OUT/'derived'/'same_module_temporal_persistence.csv',index=False,float_format='%.8f')
summary=[]
for g1 in dict.fromkeys(groups):
 for g2 in dict.fromkeys(groups):
  rows=np.flatnonzero(meta.branch==g1);cc=np.flatnonzero(meta.branch==g2)
  summary.append(dict(t2_branch=g1,t3_branch=g2,cells=len(rows)*len(cc),mean_abs_lag_rho=np.mean(np.abs(lag[np.ix_(rows,cc)])),mean_lag_minus_same=np.mean(delta[np.ix_(rows,cc)]),stable_cells=int(stable[np.ix_(rows,cc)].sum())))
pd.DataFrame(summary).to_csv(OUT/'derived'/'branch_pair_summary.csv',index=False,float_format='%.8f')
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Nimbus Roman','DejaVu Serif'],'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
pal={'MBX':'#55aaa5','MGX EC':'#6e8fbd','MGX pathway':'#d5a56d','MGX taxonomy':'#b87d96'}
fig=plt.figure(figsize=(18,10),facecolor='white')
fig.text(.07,.96,'Longitudinal organization of HI molecular modules',fontsize=17,weight='bold')
fig.text(.07,.925,'92 complete t2–t3 windows · 53 participants · 19 MBX/MGX modules · observed 2–30-week intervals',fontsize=10,color='#48555d')
positions=[.10,.40,.70]; titles=['A  Same-visit association (t2 × t2)','B  Next-visit association (t2 × t3)','C  Change in association (B − A)']
for j,(mat,title) in enumerate(zip([same,lag,delta],titles)):
 ax=fig.add_axes([positions[j],.29,.245,.45]);cmap='RdBu_r';scale=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1) if j<2 else TwoSlopeNorm(vmin=-.75,vcenter=0,vmax=.75)
 im=ax.imshow(mat,cmap=cmap,norm=scale,interpolation='nearest')
 ax.set_title(title,loc='left',fontsize=11,pad=25)
 ax.set_xticks(np.arange(19),labels,rotation=90,fontsize=6)
 if j==0:ax.set_yticks(np.arange(19),labels,fontsize=6)
 else:ax.set_yticks(np.arange(19),[])
 ax.tick_params(length=0)
 for k,g in enumerate(groups):
  ax.add_patch(plt.Rectangle((k-.5,-1.18),1,.25,color=pal[g],clip_on=False,lw=0))
  ax.add_patch(plt.Rectangle((-1.18,k-.5),.25,1,color=pal[g],clip_on=False,lw=0))
 for cut in [5,11,17]:ax.axhline(cut-.5,color='#5d6670',lw=.55);ax.axvline(cut-.5,color='#5d6670',lw=.55)
 if j==2:
  yy,xx=np.where(stable)
  ax.scatter(xx,yy,s=5,facecolors='none',edgecolors='#252525',linewidth=.35)
 for sp in ax.spines.values():sp.set_visible(False)
 cax=fig.add_axes([positions[j]+.25,.37,.008,.26]);cb=fig.colorbar(im,cax=cax);cb.ax.tick_params(labelsize=7);cb.set_label('Spearman ρ' if j<2 else 'Δρ',fontsize=8)
fig.legend(handles=[Line2D([0],[0],color=val,lw=7,label=key) for key,val in pal.items()],loc='upper right',bbox_to_anchor=(.94,.91),ncol=4,frameon=False)
ax=fig.add_axes([.12,.045,.78,.085])
x=np.arange(19);y=np.diag(lag);low=np.diag(lo)-np.diag(delta);high=np.diag(hi)-np.diag(delta)
ax.bar(x,y,color=[pal[g] for g in groups],width=.68,alpha=.88)
ax.axhline(0,color='#666',lw=.7);ax.set_ylim(-.2,1.05);ax.set_ylabel('Same-module\nt2→t3 ρ',fontsize=9);ax.set_xticks(x,[str(i) for i in range(1,20)],fontsize=8);ax.set_xlabel('Module order (see matrix labels and module_order.csv)',fontsize=8)
ax.set_title('D  Persistence of each module across visits',loc='left',fontsize=10,pad=2)
ax.spines[['top','right']].set_visible(False)
fig.text(.07,.207,'Open circles in C: participant-bootstrap 95% interval for Δρ excludes zero (500 resamples). Descriptive stability, not FDR significance.',fontsize=8,color='#48555d')
fig.text(.07,.188,'All matrices use identical complete windows. Diagonal in A equals 1 by definition; C reflects both cross-visit persistence and current-visit correlation.',fontsize=8,color='#48555d')
for ext in ['png','pdf','svg']:fig.savefig(OUT/'figure'/f'Fig4C_extended_temporal_module_atlas.{ext}',dpi=600,bbox_inches='tight')
plt.close(fig)
shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
(OUT/'requirements.txt').write_text('numpy\npandas\nmatplotlib\n')
(OUT/'README.md').write_text('''# Fig4C 扩展版：HI 分子模块纵向图谱\n\nA 是 t2 同访视 Spearman 相关矩阵；B 是当前 t2 模块与下一访视 t3 模块的 Spearman 相关矩阵；C 是 B−A 的逐格差异；D 为同一模块跨访视相关系数。均使用相同的 92 个完整配对有序窗口、53 人、19 个 MBX/MGX 模块。t2–t3 为实际 2–30 周，间隔不固定。\n\nC 的空心圈表示按照参与者有放回重抽样 500 次、保留每位参与者所有窗口后计算的 Δρ 百分位 95% 区间未跨零。这是方向稳定性的探索性标记，不是多重检验校正后的统计显著性。A 的对角线按定义为 1；C 对角线同时含有这一数学基线，不能称为模块活性下降。D 报告实际同一模块的跨访视相关。重复窗口、缺失形成的完整病例选择、HI 评分的构成，以及访视间隔不同均限制推断。该图没有 SHAP 值、因果效应或独立预测增益。\n\n图像输出为 600 dpi PNG 与矢量 PDF/SVG。`source/` 保留两份已计算的来源表；`derived/` 包含逐窗口矩阵、三张相关矩阵、逐格区间、方向稳定性、模块序号和分支汇总。`code/make_fig4c_extended.py` 用于重算，解压后将两份来源文件的路径按脚本中的 ROOT/BASE 调整。源表为分析与模块评分衍生数据，不是原始组学检测矩阵。\n\n**Suggested legend:** Fig. 4C | Longitudinal organization of metabolomic and metagenomic modules. Same-visit (A) and cross-visit (B) Spearman associations were calculated on the same 92 complete ordered windows from 53 participants. Panel C shows cross-visit minus same-visit correlation; open circles mark cells whose participant-level bootstrap percentile interval for the difference excludes zero in 500 resamples. Panel D shows within-module cross-visit association. These descriptive associations are not SHAP attributions or causal effects.\n''',encoding='utf-8')
with (OUT/'SHA256SUMS.txt').open('w') as f:
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='SHA256SUMS.txt':f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(OUT))+'\n')
zp=ROOT/'Fig4C_扩展版_纵向模块关联与稳定性_审核包.zip'
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(OUT.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(ROOT))
print('stable cells',stable.sum(),'cross diagonal',np.diag(lag).min(),np.diag(lag).max(),zp)
