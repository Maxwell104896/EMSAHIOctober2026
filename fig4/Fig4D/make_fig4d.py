from pathlib import Path
import hashlib
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

def proportion_confint(k,n):
 z=1.959963984540054; p=k/n; den=1+z*z/n
 mid=(p+z*z/(2*n))/den
 half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return mid-half,mid+half

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'attachments/cbd0ca87-9272-4b2b-a8ab-098fec67ce84/EMSA_HI_trajectory_prediction_window_table.csv'
REPORT = ROOT / 'attachments/706fa307-b04f-4c96-9e2e-d1c6472a4dde/EMSA_HI_trajectory_prediction_model_report.md'
OUT = ROOT / 'Fig4D_next_visit_HI_by_interval'
(OUT/'figures').mkdir(parents=True,exist_ok=True)
(OUT/'data'/'source').mkdir(parents=True,exist_ok=True)
(OUT/'data'/'derived').mkdir(parents=True,exist_ok=True)
(OUT/'audit').mkdir(parents=True,exist_ok=True)
shutil.copy2(SOURCE,OUT/'data'/'source'/SOURCE.name)
shutil.copy2(REPORT,OUT/'audit'/REPORT.name)
d=pd.read_csv(SOURCE)
assert len(d)==152 and d.participant_id.nunique()==66
assert np.allclose(d.time_t3-d.time_t2,d.dt23) and d.dt23.between(2,30).all()
assert (d.target_high_state.astype(int)==d.target_next_state.eq('HI_high_state').astype(int)).all()
assert not d.duplicated(['participant_id','sample_t1','sample_t2','sample_t3']).any() if set(['sample_t1','sample_t2','sample_t3']).issubset(d.columns) else True
labels=['2–8','9–12','13–30']
d['interval_weeks']=pd.cut(d.dt23,[1,8,12,30],labels=labels)
d['current_state']=d.state_t2.str.replace('HI_','',regex=False).str.replace('_state','',regex=False).str.title()
v=d[['participant_id','time_t2','time_t3','dt23','interval_weeks','state_t2','current_state','target_next_state','target_high_state']].copy()
v.to_csv(OUT/'data'/'derived'/'Fig4D_window_plotting_data.csv',index=False)
rows=[]
for state in ['Low','Intermediate','High']:
 for interval in labels:
  s=v[(v.current_state==state)&(v.interval_weeks==interval)]
  n=len(s); e=int(s.target_high_state.sum()); lo,hi=proportion_confint(e,n)
  rows.append(dict(current_state=state,interval_weeks=interval,windows=n,participants=s.participant_id.nunique(),high_HI_events=e,observed_fraction=e/n,wilson95_lower=lo,wilson95_upper=hi,median_dt_weeks=s.dt23.median()))
t=pd.DataFrame(rows)
t.to_csv(OUT/'data'/'derived'/'Fig4D_bin_summary.csv',index=False,float_format='%.8f')
rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Nimbus Roman','DejaVu Serif'],'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(8.4,4.8))
colors={'Low':'#54A9A5','Intermediate':'#6586B8','High':'#C3796E'}
for state,off in [('Low',-.18),('Intermediate',0),('High',.18)]:
 s=t[t.current_state==state]
 x=np.arange(3)+off;y=s.observed_fraction.to_numpy();low=s.wilson95_lower.to_numpy();high=s.wilson95_upper.to_numpy()
 ax.errorbar(x,y,yerr=np.vstack([np.maximum(0,y-low),np.maximum(0,high-y)]),fmt='o',ms=7,capsize=3,lw=1.5,color=colors[state],label=f'{state} HI')
 for xi,yi,row in zip(x,y,s.itertuples()):
  ax.annotate(f'{row.high_HI_events}/{row.windows}',(xi,yi),xytext=(0,8 if state!='High' else -17),textcoords='offset points',ha='center',fontsize=8,color=colors[state])
ax.set_xticks(np.arange(3),[f'{lab} wk' for lab in labels]);ax.set_xlim(-.5,2.5);ax.set_ylim(-.07,1.07)
ax.set_ylabel('Observed next-visit high-HI fraction');ax.set_xlabel('Observed interval from t2 to t3 (weeks)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,pos:f'{x:.0%}'))
ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper left',frameon=False,ncol=3,fontsize=9)
fig.text(.12,.015,'Labels: events/windows; bars: descriptive Wilson 95% intervals. 66 participants, 152 overlapping windows.',fontsize=8)
fig.tight_layout(rect=[0,.055,1,1])
for ext in ['png','pdf','svg']:
 fig.savefig(OUT/'figures'/f'Fig4D_next_visit_high_HI_by_interval.{ext}',dpi=600,bbox_inches='tight')
plt.close(fig)
readme='''# Fig4D — 下一访视 high-HI 状态与真实访视间隔

主图以 t2 至 t3 实际间隔（周）分成 2–8、9–12、13–30 周，按 t2 的 low/intermediate/high-HI 分层。点是 t3 观察到 high-HI 的窗口比例；旁边的 e/n 是事件数/窗口数；误差线为描述性的 Wilson 95% 二项区间。`Fig4D_bin_summary.csv` 同时给出每箱独立参与者数。各箱的参与者可重复出现，人数不可跨箱相加。全表为 66 人的 152 个有序且可能重叠的 t1–t2–t3 窗口。

本图以固定分箱统计实际观察状态，不拟合平滑曲线、不展示死亡风险、复发风险或风险倍数。Low-HI 三箱均为零事件，Wilson 上限提醒不能将零事件解释为真实风险恒为零。区间没有校正同一人重复窗口的相关性，仅作描述，不据其进行组间假设检验。间隔组的患者构成可能不同，不能推断风险随时间的因果变化，也不能将间隔曲线解释为诊断后随访风险。

## 文件及运行

- `figures/`: 600 dpi PNG 和矢量 PDF、SVG。
- `data/source/`: 云盘 Step54 的完整逐窗口分析表，属于模型衍生表，不是原始组学检测矩阵。
- `data/derived/`: 作图逐窗口字段和九个分组的统计表。
- `audit/`: 原模型报告、来源信息、校验和。
- `make_fig4d.py`: 复现代码。将原始源表放在脚本中 SOURCE 指定位置，运行 `python make_fig4d.py`；依赖见 `requirements.txt`。

来源： https://drive.google.com/file/d/1j-Xvevz7iDD185L-ieJa5ltyELSEkH_9/view （Step54）；时间单位“周”由已有 Fig3C 真实预测间隔审计说明核对。脚本检查 152 窗口、66 人、时间差及 high-HI 结局编码。

## Suggested legend

**Fig. 4D | Observed next-visit high-HI state across follow-up intervals.** The time from the second to third visit was 2–30 weeks and was grouped into 2–8, 9–12, and 13–30 weeks. Points show the observed fraction of windows with high HI at the next visit, stratified by current HI state. Labels give events/windows; bars indicate descriptive Wilson 95% binomial intervals. The 152 ordered windows came from 66 participants, some of whom contributed multiple windows. This is a molecular state outcome and does not estimate clinical relapse or time-to-event risk.
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
(OUT/'requirements.txt').write_text('numpy\npandas\nmatplotlib\n',encoding='utf-8')
(OUT/'audit'/'source_urls.txt').write_text('Step54 window table: https://drive.google.com/file/d/1j-Xvevz7iDD185L-ieJa5ltyELSEkH_9/view\nStep54 model report: https://drive.google.com/file/d/1KaFm-TCzkhoCoDWS9CLLFz11kcP8RXba/view\nFig3C time-unit audit: https://drive.google.com/file/d/1-qmd2KDZ4ZuNzs13LSX4sLOAGWuqrtO1/view\n',encoding='utf-8')
shutil.copy2(__file__,OUT/'make_fig4d.py')
with (OUT/'audit'/'SHA256SUMS.txt').open('w') as f:
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='SHA256SUMS.txt': f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(OUT))+'\n')
zipname=ROOT/'Fig4D_真实访视间隔与下一访视HI_图数据代码审核包.zip'
with zipfile.ZipFile(zipname,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(OUT.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(ROOT))
print(t.to_string(index=False));print(zipname)
