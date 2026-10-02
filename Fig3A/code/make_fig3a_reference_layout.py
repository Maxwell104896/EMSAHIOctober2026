from pathlib import Path
import hashlib, shutil, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Fig3A_reference_style_submission'
for sub in ['figure','data/source','data/selected','code','audit']:(OUT/sub).mkdir(parents=True,exist_ok=True)
src=ROOT/'Fig3A_submission/data/source/Fig6B_internal_next_HI_score_source_exact.csv'
upstream=ROOT/'Fig3A_submission/data/source/EMSA_HI_step57_event_oriented_oof_predictions.csv'
d=pd.read_csv(src)
assert len(d)==152 and d.participant_id.nunique()==66
keys=['participant_id','sample_t1','sample_t2','sample_t3']
u=pd.read_csv(upstream)
m=d.merge(u[keys+['score_t3','ridge_delta_base_pred_next_HI_score']],on=keys,validate='one_to_one',suffixes=('_plot','_upstream'))
assert len(m)==152
for col in ['score_t3','ridge_delta_base_pred_next_HI_score']:
    assert np.allclose(m[col+'_plot'],m[col+'_upstream'],atol=1e-12)

spec=[('Low',True),('Intermediate',True),('High',True),('Low',False),('Intermediate',False),('High',False)]
rows=[];used=set()
for state,correct in spec:
    a=d[(d.observed_state_label==state)&((d.observed_state_label==d.predicted_state_label)==correct)].copy()
    med=a.absolute_error.median()
    a['stratum_median_absolute_error']=med
    a['distance_to_median_error']=(a.absolute_error-med).abs()
    a=a.sort_values(['distance_to_median_error','participant_id','sample_t3'])
    r=a[~a.participant_id.isin(used)].iloc[0].copy()
    used.add(r.participant_id)
    r['selection_stratum_n']=len(a)
    r['selection_rule']='median absolute-error exemplar per observed-state × agreement stratum; distinct participants; ID/sample tie break'
    rows.append(r)
s=pd.DataFrame(rows).reset_index(drop=True)
s.insert(0,'lane',range(1,7))
s.to_csv(OUT/'data/selected/Fig3A_six_illustrative_windows.csv',index=False)
shutil.copy2(src,OUT/'data/source'/src.name)
shutil.copy2(upstream,OUT/'data/source'/upstream.name)

plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
blue='#28729B';orange='#D37F3F';gray='#505F66';light='#E3EAEC'
fig,ax=plt.subplots(figsize=(9.4,6.2))
fig.subplots_adjust(left=.105,right=.96,bottom=.15,top=.84)
ax.set_xlim(-13,27);ax.set_ylim(-.55,6.85)
ax.set_yticks([]);ax.set_xticks([-10,0,10,20])
ax.tick_params(axis='x',bottom=False,labelbottom=False,top=True,labeltop=True,length=4)
ax.xaxis.set_label_position('top')
ax.set_xlabel('Weeks relative to prediction origin (t2)',labelpad=11,fontsize=9.5)
for spine in ax.spines.values():spine.set_visible(False)
ax.spines['top'].set_visible(True);ax.spines['top'].set_position(('data',6.45))
ax.axvline(0,ymin=.07,ymax=.87,color=gray,lw=1.1,ls=(0,(5,4)))

ax.text(-12.8,6.1,'Observed HI history\n(model input: t1, t2)',color=gray,ha='left',va='top',fontsize=10)
ax.text(3.0,6.1,'Next-visit HI at t3',color=gray,ha='left',va='top',fontsize=10)
ax.text(20.8,6.08,'Predicted',color=orange,ha='left',fontsize=8.6)
ax.text(20.8,5.78,'Observed',color=blue,ha='left',fontsize=8.6)
for i,r in s.iterrows():
    y=5.15-i*.84
    x1=-r.dt12;x3=r.dt23
    ax.plot([x1,0],[y,y],color=gray,lw=1.25,zorder=1)
    ax.plot([0,x3],[y,y+.16],color=orange,lw=1.5,zorder=2)
    ax.plot([0,x3],[y,y-.16],color=blue,lw=1.5,zorder=2)
    ax.plot([x1,0],[y,y],linestyle='none',marker='s',markersize=6,color=blue,zorder=3)
    ax.plot(x3,y+.16,marker='s',markersize=6,color=orange,zorder=4)
    ax.plot(x3,y-.16,marker='o',markersize=5.5,color=blue,zorder=4)
    ax.text(-12.8,y+.28,f'{int(r.lane)}  {r.observed_state_label} observed',fontsize=8.4,color='#334D5B',va='bottom')
    ax.text(x1,y-.32,f'{r.score_t1:+.2f}',ha='center',fontsize=7.7,color=blue)
    ax.text(0,y-.32,f'{r.score_t2:+.2f}',ha='center',fontsize=7.7,color=blue)
    ax.text(x3+.45,y+.25,f'{r.ridge_delta_base_pred_next_HI_score:+.2f}',color=orange,fontsize=7.8,va='center')
    ax.text(x3+.45,y-.25,f'{r.score_t3:+.2f}',color=blue,fontsize=7.8,va='center')

fig.text(.105,.967,'a',fontsize=17,weight='bold',color='#1E303B')
fig.text(.145,.963,'Observed trajectories and one-step next-visit predictions',fontsize=12.4,weight='bold',color='#263E4A')
fig.text(.105,.904,'Six illustrative windows from 152 windows / 66 participants',fontsize=8.8,color='#536A76')
fig.text(.105,.068,'Blue squares: observed t1/t2 HI  ·  Orange squares: out-of-fold predicted t3 HI  ·  Blue circles: observed t3 HI',fontsize=8.3,color='#405D6B')
fig.text(.105,.035,'Horizontal lanes organize examples; their vertical positions do not encode HI. Numbers at markers are HI scores.',fontsize=7.8,color='#667B85')
for ext in ['svg','pdf','png']:
    fig.savefig(OUT/'figure'/f'Fig3A_reference_style.{ext}',dpi=600,facecolor='white')
plt.close(fig)

(OUT/'figure/Fig3A_caption.txt').write_text('Fig. 3A | Observed molecular HI histories and one-step next-visit predictions. Six illustrative participant windows, selected by a reproducible median-error rule within each observed Low, Intermediate, or High state and prediction-agreement stratum, are displayed from 152 ordered windows among 66 participants. Blue squares denote HI measured at t1 and t2, orange squares denote participant-grouped out-of-fold predicted HI at t3, and blue circles denote the observed t3 HI. The vertical dashed line marks the t2 prediction origin. Horizontal position is elapsed weeks relative to t2; vertical lane placement is for display only and does not represent HI magnitude. Numeric labels give HI scores. The six examples are illustrative and do not estimate overall performance. These are single next-visit molecular HI predictions, not simulated longitudinal futures, intervention effects or clinical relapse predictions.\n',encoding='utf-8')
(OUT/'README.md').write_text('''# Fig3A — reference-panel-A style

This revision follows the reference image’s horizontal-lane and branching design. Every lane is one genuine t1/t2/t3 window. The paired branches at t3 depict one out-of-fold model prediction and one actual observation at the same visit. They are **not separate simulated trajectories**. Numeric labels are the actual HI scores; lane height and branch separation are layout devices, not a y-axis.

Source: locked Fig6B 152-window export and Step57 upstream OOF predictions, copied in `data/source/`. These are analysis-level window data, not raw assay measurements. All 152 predictions and observations were matched one-to-one by participant and visit sample IDs and checked numerically. Internal participant-grouped CV does not provide external clinical validation.

Selection: one matching and one mismatching predicted-state example for each observed Low, Intermediate and High state. In each of six strata, choose the absolute HI error closest to the stratum median, retaining distinct participants; sort ties by participant ID and sample_t3. The complete six-row table, error and selection metadata are in `data/selected/`.

Reproduce with `python code/make_fig3a_reference_layout.py` from this package directory. Requirements: pandas, numpy, matplotlib. It recreates SVG (editable text), PDF and 600 dpi PNG. DejaVu Serif was used as Times New Roman is unavailable in this environment; edit the font parameter if installed. The figure caption is in `figure/`.
''',encoding='utf-8')
shutil.copy2(Path(__file__),OUT/'code/make_fig3a_reference_layout.py')
with (OUT/'audit/SHA256SUMS.txt').open('w') as f:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS.txt':f.write(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n')
with zipfile.ZipFile(ROOT/'Fig3A_reference_style_submission.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(ROOT))
print('created',ROOT/'Fig3A_reference_style_submission.zip')
