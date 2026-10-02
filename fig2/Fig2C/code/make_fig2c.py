"""Fig. 2C: participant-cluster bootstrap distributions of four endpoint AUROCs."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

TASKS=[
 ('Next high HI','next_high_state','ridge_delta_base_uncalibrated','#3D7F9C'),
 ('Next high-extreme HI','next_high_extreme','ridge_plus_modules_score_calibrated','#467D8B'),
 ('Meaningful HI improvement','meaningful_improvement_delta','ridge_delta_base_uncalibrated','#D4A05F'),
 ('Meaningful HI worsening','meaningful_worsening_delta','ridge_plus_modules_uncalibrated','#B76B80'),
]

def main(root):
    data=root/'data';out=root/'figure';out.mkdir(exist_ok=True)
    source=pd.read_csv(data/'source_step57_oof_predictions.csv')
    perf=pd.read_csv(data/'source_final_endpoint_performance.csv')
    assert len(source)==152 and source.participant_id.nunique()==66
    columns=['participant_id','sample_t1','sample_t2','sample_t3','time_t3']
    for _,endpoint,model,_ in TASKS:
        columns.extend([f'{model}_{endpoint}_observed',f'{model}_{endpoint}_risk_score'])
    d=source[columns].copy()
    assert d.notna().all().all()
    d.to_csv(data/'four_endpoint_oof_labels_and_scores.csv',index=False)
    people=d.participant_id.unique();groups={p:d[d.participant_id==p] for p in people}
    rng=np.random.default_rng(20260928)
    boot={e:[] for _,e,_,_ in TASKS}
    invalid=0
    for rep in range(2500):
        # Identical participant resamples across endpoints preserve their pairing.
        sampled=pd.concat([groups[p] for p in rng.choice(people,len(people),replace=True)],ignore_index=True)
        values={}
        for _,endpoint,model,_ in TASKS:
            y=sampled[f'{model}_{endpoint}_observed']
            score=sampled[f'{model}_{endpoint}_risk_score']
            if y.nunique()<2:invalid+=1;break
            values[endpoint]=roc_auc_score(y,score)
        else:
            for endpoint,value in values.items():boot[endpoint].append((rep,value))
    assert len(boot[TASKS[0][1]])>=2400
    rows=[];dist=[]
    for label,endpoint,model,color in TASKS:
        y=d[f'{model}_{endpoint}_observed'];score=d[f'{model}_{endpoint}_risk_score']
        auc=roc_auc_score(y,score)
        ref=perf[perf.endpoint==endpoint]
        assert len(ref)==1 and abs(auc-float(ref.AUROC.iloc[0]))<1e-9
        values=np.array([v for _,v in boot[endpoint]])
        lower,upper=np.quantile(values,[.025,.975])
        rows.append(dict(endpoint=endpoint,display_label=label,preferred_model=model,
                         n_windows=len(y),n_participants=len(people),n_events=int(y.sum()),
                         full_sample_AUROC=auc,bootstrap_median=np.median(values),
                         percentile_95ci_lower=lower,percentile_95ci_upper=upper,
                         valid_bootstrap_replicates=len(values)))
        dist.extend(dict(replicate=int(i),endpoint=endpoint,AUROC=float(v)) for i,v in boot[endpoint])
    result=pd.DataFrame(rows);result.to_csv(data/'endpoint_auc_bootstrap_summary.csv',index=False)
    pd.DataFrame(dist).to_csv(data/'participant_bootstrap_auc_replicates.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(10.5,5.7),facecolor='white')
    fig.subplots_adjust(left=.28,right=.78,top=.83,bottom=.20)
    order=list(range(4,0,-1))
    sets=[np.array([v for _,v in boot[e]]) for _,e,_,_ in TASKS]
    viol=ax.violinplot(sets,positions=order,vert=False,showmedians=False,showextrema=False,widths=.78)
    for body,(_,_,_,col) in zip(viol['bodies'],TASKS):
        body.set_facecolor(col);body.set_edgecolor(col);body.set_alpha(.20)
    boxes=ax.boxplot(sets,positions=order,vert=False,showfliers=False,patch_artist=True,widths=.25,
                     manage_ticks=False,medianprops={'color':'#263F4B','linewidth':1.4},
                     whiskerprops={'color':'#637781'},capprops={'color':'#637781'})
    for box,(_,_,_,col) in zip(boxes['boxes'],TASKS):
        box.set_facecolor(col);box.set_edgecolor(col);box.set_alpha(.55)
    for y,(_,_,_,col),r in zip(order,TASKS,rows):
        ax.plot([r['percentile_95ci_lower'],r['percentile_95ci_upper']],[y,y],
                color='#253E4B',lw=1.1,zorder=5)
        ax.scatter([r['full_sample_AUROC']],[y],s=66,marker='D',facecolor='white',
                   edgecolor=col,linewidth=2,zorder=7)
        ax.text(1.025,y,f"{r['full_sample_AUROC']:.3f}   ({r['n_events']}/{r['n_windows']} events)",
                va='center',ha='left',fontsize=9,color='#294958',
                transform=ax.get_yaxis_transform(),clip_on=False)
    ax.set_yticks(order,[x[0] for x in TASKS]);ax.set_xlim(.42,1.005)
    ax.set_xlabel('AUROC');ax.set_xticks(np.arange(.45,1.01,.1))
    ax.grid(axis='x',color='#E8EFF1',zorder=0)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.suptitle('Discrimination across molecular prediction endpoints',
                 x=.08,y=.95,ha='left',fontsize=16,fontweight='bold',color='#254854')
    fig.text(.08,.89,'Participant-level bootstrap AUROC distribution (2,500 resamples; 66 participants)',
             fontsize=10,color='#536A75')
    fig.text(.08,.085,'Boxes and shaded shapes: bootstrap AUROCs. Diamonds: original out-of-fold AUROCs. '
             'Horizontal strokes: bootstrap 95% intervals.',
             fontsize=8.5,color='#536A75')
    fig.text(.08,.052,'All four outcomes are model-derived molecular states or movements, not clinical relapse.',
             fontsize=8.5,color='#536A75')
    for ext in ('svg','pdf','png'):
        fig.savefig(out/f'Fig2C_endpoint_AUROC_bootstrap.{ext}',dpi=600 if ext=='png' else None,facecolor='white')
    plt.close(fig)
    print(result[['endpoint','n_events','full_sample_AUROC','percentile_95ci_lower','percentile_95ci_upper']].to_string(index=False))
    print('invalid replicates',invalid)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().root)
