from pathlib import Path
import hashlib, shutil, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,ConnectionPatch
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'Fig4A_HI_molecular_state_map'
OUT=ROOT/'Fig4A_reference_style_HI_module_map'
(OUT/'figure').mkdir(parents=True,exist_ok=True)
(OUT/'source').mkdir(parents=True,exist_ok=True)
(OUT/'derived').mkdir(parents=True,exist_ok=True)
(OUT/'code').mkdir(parents=True,exist_ok=True)
p=SOURCE/'derived/Fig4A_MBX_546_coordinates.csv'
d=pd.read_csv(p)
assert len(d)==546 and d.participant_id.nunique()==106 and not d.sample_id.duplicated().any()
assert d[['MBX_PC1','MBX_PC2','EMSA_HI_exact_state_score']].notna().all().all()
colors={'nonIBD':'#83b5a4','UC':'#3695b0','CD':'#d98664'}
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Nimbus Roman','DejaVu Serif'],'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(12,7.5),facecolor='white')
ax=fig.add_axes([.07,.12,.61,.78])
size=12+43*d.EMSA_HI_exact_state_score.rank(pct=True)
for k in colors:
 s=d[d.diagnosis_label==k]; ax.scatter(s.MBX_PC1,s.MBX_PC2,s=size.loc[s.index],c=colors[k],alpha=.72,edgecolor='white',linewidth=.35,label=f'{k} ({len(s)} visits)',rasterized=False)
ax.set(xlabel='PC1 (45.4% variance)',ylabel='PC2 (20.2% variance)',title='MBX molecular module landscape of HI-scored visits')
ax.spines[['top','right']].set_visible(False)
ax.grid(alpha=.1)
# Boxes select real high-density regions, and magnify without changing coordinates.
boxes=[(-2.65,-.85,-.95,.9),(.35,2.2,-1.5,.35)]
positions=[[.74,.52,.23,.28],[.74,.16,.23,.27]]
letters=['I','II']
records=[]
for (xmin,xmax,ymin,ymax),pos,letter in zip(boxes,positions,letters):
 sub=d[d.MBX_PC1.between(xmin,xmax)&d.MBX_PC2.between(ymin,ymax)].copy()
 assert len(sub)>=20
 ax.add_patch(Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,fill=False,edgecolor='#444444',lw=1,linestyle=(0,(3,2))))
 ia=fig.add_axes(pos)
 for k in colors:
  s=sub[sub.diagnosis_label==k];ia.scatter(s.MBX_PC1,s.MBX_PC2,s=17+37*s.EMSA_HI_exact_state_score.rank(pct=True),c=colors[k],alpha=.8,edgecolor='white',linewidth=.35)
 ia.set_xlim(xmin,xmax);ia.set_ylim(ymin,ymax);ia.set_xticks([]);ia.set_yticks([])
 for spine in ia.spines.values():spine.set_linestyle((0,(3,2)));spine.set_color('#555555')
 ia.set_title(f'Zoom {letter}: {len(sub)} visits',loc='left',fontsize=10,pad=5)
 # Labels are real records, chosen deterministically from two tails of HI score.
 chosen=pd.concat([sub.nsmallest(1,'EMSA_HI_exact_state_score'),sub.nlargest(1,'EMSA_HI_exact_state_score')])
 for j,row in enumerate(chosen.itertuples()):
  horiz=-55 if row.MBX_PC1>(xmin+xmax)/2 else 8
  ia.annotate(f'{row.sample_id}\nHI {row.EMSA_HI_exact_state_score:.2f}',(row.MBX_PC1,row.MBX_PC2),xytext=(horiz,12 if j==0 else -24),textcoords='offset points',fontsize=7,annotation_clip=False,arrowprops=dict(arrowstyle='-',lw=.45,color='#555555'))
 sub.assign(zoom=letter).to_csv(OUT/'derived'/f'zoom_{letter}_points.csv',index=False)
 records.append((letter,len(sub),sub.participant_id.nunique()))
fig.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[k],markeredgecolor='white',markersize=9,label=f'{"non-IBD" if k=="nonIBD" else k} (n={sum(d.diagnosis_label==k)})') for k in colors],loc='upper right',bbox_to_anchor=(.98,.99),frameon=False,title='Recorded diagnosis')
fig.text(.735,.46,'Dot area scales with the within-figure\nrank of exact MBX-HI score.',fontsize=9,color='#555555')
fig.text(.07,.035,'Each dot is a sample visit; repeat visits are retained. Zooms show local overlap, not separate diagnostic clusters.',fontsize=9,color='#555555')
for ext in ['png','pdf','svg']:
 fig.savefig(OUT/'figure'/f'Fig4A_reference_style_MBX_HI_map.{ext}',dpi=600,bbox_inches='tight')
plt.close(fig)
shutil.copy2(p,OUT/'source'/p.name)
for fname in ['Fig4A_MBX_546_exact_source.csv','EMSA_module_scores_wide_by_branch_module.csv','EMSA_FINAL_v3_module_score_rerun_resource_fixed_report.md','EMSA_FINAL_v3_module_scoring_branch_resource_manifest.csv']:
 shutil.copy2(SOURCE/'source'/fname,OUT/'source'/fname)
shutil.copy2(SOURCE/'code/make_fig4a.py',OUT/'code/make_original_pca.py')
shutil.copy2(__file__,OUT/'code/make_fig4a_reference_style.py')
(OUT/'README.md').write_text(f'''# Fig4A — 参考图 A 风格的 HI 分子模块地图

总览图：546 次样本访视、106 人，五个 MBX 模块标准化后 PCA。颜色对应记录诊断 non-IBD、UC、CD，点面积按该图内 exact MBX-HI 分数秩缩放（**不是病例数**）。两个虚线框是实际 PCA 坐标的局部放大；Zoom I {records[0][1]} 次访视/{records[0][2]} 人，Zoom II {records[1][1]} 次访视/{records[1][2]} 人。框中标注真实样本 ID 和分数，选择规则为局部最低、最高分。

本图沿用参考 A 的全图、局部放大及分类图例结构，但只展示本研究真正拥有的三种诊断类别。三个诊断组明显重叠；局部放大不能解释为清晰簇，也不能推断诊断性能。点为访视，同一人可能有重复访视。局部框是展示视角，不是独立验证。HI 分数大小仅作描述，不借用未经核实的 low/intermediate/high 阈值。

`figure/` 有 600 dpi PNG、矢量 PDF 与 SVG；`source/` 有 exact HI 来源、五模块所在宽表、上游资源说明与前版 PCA 逐点坐标；`derived/` 为两个放大区逐点数据。`code/make_original_pca.py` 可从来源模块分数重建前版坐标；`code/make_fig4a_reference_style.py` 用前版坐标画新布局。依赖 numpy、pandas、matplotlib、scikit-learn。新脚本的 SOURCE/OUT 路径相对于打包前工作区，解压后按实际目录调整后运行。

**Legend:** Figure 4A | Molecular module landscape of HI-scored visits. Principal components of five standardized metabolomic module scores in 546 visits from 106 participants. Dots are colored by recorded diagnosis and sized by the within-figure rank of the exact MBX-HI score. Dashed boxes magnify two regions of the same embedding and label observed sample IDs. Repeated visits are retained. PCA and the zoom views are descriptive and do not measure diagnostic discrimination.
''',encoding='utf-8')
(OUT/'requirements.txt').write_text('numpy\npandas\nmatplotlib\nscikit-learn\n')
with (OUT/'SHA256SUMS.txt').open('w') as f:
 for q in sorted(OUT.rglob('*')):
  if q.is_file() and q.name!='SHA256SUMS.txt':f.write(hashlib.sha256(q.read_bytes()).hexdigest()+'  '+str(q.relative_to(OUT))+'\n')
zp=ROOT/'Fig4A_参考图A风格_HI模块地图_可复现审核包.zip'
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
 for q in sorted(OUT.rglob('*')):
  if q.is_file():z.write(q,q.relative_to(ROOT))
print(records,zp)
