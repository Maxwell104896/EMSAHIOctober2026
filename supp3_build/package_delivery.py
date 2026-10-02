from pathlib import Path
import pandas as pd,numpy as np,json,zipfile,hashlib,shutil
from pypdf import PdfReader,PdfWriter
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'supp3_final';R=ROOT/'inputs/fig1_materials/Fig1_submission/results'
p=pd.read_csv(O/'locked_trajectory_input.csv');original=pd.read_csv(R/'corrected_trajectory_predictions.csv');q=pd.read_csv(O/'updated_oof_predictions.csv');mods=pd.read_csv(O/'current_module_inputs.csv');source=pd.read_csv(R/'modules/EMSA_module_scores_wide_by_branch_module.csv').set_index(['participant_id','sample_id']);cols=mods.columns[2:]
assert np.allclose(p[['score_t1','score_t2','score_t3']],original[['score_t1','score_t2','score_t3']],rtol=0,atol=1e-12)
assert np.array_equal(p.persistence_class,original.persistence_class) and np.array_equal(p.observed_class,original.observed_class)
assert np.allclose(q.pred_HI,original.pred_HI,rtol=0,atol=1e-12)
assert np.allclose(mods[cols].to_numpy(),source.loc[list(zip(mods.participant_id,mods.sample_t2)),cols].to_numpy(),rtol=0,atol=1e-12)
c=pd.read_csv(O/'configuration_results.csv');assert not c.features.str.contains('dt23').any();v=pd.read_csv(O/'configuration_oof_predictions.csv');maxerr=max(abs(np.mean(abs(v[r.config_id]-p.score_t3))-r.mae) for r in c.itertuples());assert maxerr<1e-12
checks={'locked_windows':152,'locked_people':66,'next_high_events':47,'map_visits':111,'map_people':63,'map_current_state_counts':[37,38,36],'updated_module_columns':19,'source_module_match_max_error':0,'locked_HI_scores_labels_outer_folds_and_main_predictions_preserved':True,'candidate_configs':78,'candidate_future_dt23_input':False,'config_MAE_recomputation_max_error':maxerr,'bootstrap_resamples':3000,'PPTX_native_shapes':True,'PPTX_font':'Times New Roman','final_font_min_points':11,'final_size_mm':[180,250],'scope':'Analysis-level upstream tables; no independent assay raw-data QC; supplementary nested selection conditional on exported HI.'};(O/'consistency_validation.json').write_text(json.dumps(checks,indent=2))
rows=[('A,H–J','S3A,H–J','旧状态35/35/41','定稿t2标签37/38/36；19点改类'),('A–G','S3A–G','旧16模块及坐标','19更新模块、111完整访视、重新标准化及拟合t-SNE'),('H–J','S3H–J及稳定性CSV','单seed敏感性、纵横比例','等比例；3个perplexity×10seed；另补30次随机初始化'),('B–G','S3B–G','色限饱和未说明','完整z值保留；色限和超界点数写入图注'),('K','S4A','旧HI MAE0.413、156配置含未来间隔','改用定稿HI及outer_fold；78组无dt23配置；最小MAE0.461997'),('K','S4B/C','择优误差与嵌套性能混淆','内层4折按人选择；嵌套MAE0.472848、基线0.509483；配对差和CI'),('K','S4图注','全流程验证与上游条件混淆','明确回归器选择条件验证；锁定核心预测和主图性能不替换'),('整图','S3/S4','编号重叠、主题跨越','分子地图及稳定性S3；回归器配置探索S4；各自从A编号'),('整图','两页PPT/PNG/PDF/SVG','备注替代图注、字号过小','独立图注＋PPT备注；180×250mm；TNR≥11pt；黑字和统一蓝绿橙'),('完整病例','纳入清单及描述表','41排除窗口缺少说明','输出逐访视缺失模块；纳入111/63人与排除41/28人描述'),('统计及解释','正式图注及逐行结果','置信区间、独立信息和因果误读','3000次按人bootstrap；说明列冗余、模块分数性质及探索性'),('编辑和AI/溯源','可编辑PPT、代码和校验表','SVG整体嵌入及版本冲突','所有数据标记和文字为原生形状；未使用生成式绘图；从分析源表闭合')]
pd.DataFrame(rows,columns=['原位置','新位置','问题','修订结果']).to_csv(O/'revision_table.csv',index=False,encoding='utf-8-sig')
(O/'README.md').write_text('''# Supplementary Figure S3/S4 submission revision

S3 replaces the old molecular-state maps with updated 19-module input scores and locked current HI labels. S4 replaces the old configuration analysis with a separate 78-configuration supplementary search excluding actual future interval dt23. Main-model HI scores, cutoffs, labels, participant outer folds and predictions are unchanged. See consistency_validation.json and the captions.

## Files
- Supplementary_Figures_S3_S4.pptx: two pages, native editable shapes and text, Times New Roman, minimum11pt, 180×250mm per page.
- S3/S4.png: 600dpi, same final physical dimensions. S3/S4.pdf and .svg: vector versions. PDF raster preview and standalone plots use installed Nimbus Roman as a metric-compatible serif fallback; the PPTX specifies Times New Roman.
- S3_caption.txt / S4_caption.txt: publication captions, also embedded in PPT notes.
- revision_table.csv: issue-to-fix mapping.
- Analysis CSV/JSON tables: inputs, normalization, current state labels, completeness records, embedding coordinates, six overlays, 60 total t-SNE fits (30 fixed-PCA plus30 random initialization), configuration predictions, inner search, selected outer-fold configurations, regression predictions and confidence intervals.

## Interpretation
The maps are exploratory. Modules are model scores, not concentrations or pathway fluxes. Current t2 groups differ from observed next t3 groups. Overlay color saturation is display-only. Absolute coordinates across perplexities are not comparable. Fixed PCA initialization yields the same neighborhoods over the tested seeds; random-initialization results show nontrivial neighborhood variation and do not justify a claim of invariant clustering.

S4 tests regression-layer selection conditional on the exported upstream HI scores. It is not full-pipeline nested validation. The supplementary grid minimum, nested selection and locked core model are distinct analyses. The main model reference remains MAE0.462369, R²0.728695. Bootstrap CIs condition on existing fitted predictions and do not include full upstream refitting.

## Reproduce
Extract the package preserving directories. Install numpy2.3.5, pandas2.2.3, scipy1.17.0, scikit-learn1.8.0, matplotlib3.10.8 and pypdf. Run `python supp3_build/make_figures.py`, then `python supp3_build/add_stability.py`. Plot outputs and analysis tables are written into supp3_final. Native PPT authoring uses the supplied @oai/artifact-tool runtime and its validation helpers; `supp3_build/build_ppt.mjs` includes that workflow. Before rerunning PPT finalization, use a fresh output destination because the finalizer preserves existing deliverables. The included source tables are analysis-level inputs; upstream assay processing and raw-data QC require the corresponding Fig1 source/replay package. Raw spectra/FASTQ files are not included or independently verified here.

## References preserved
The source folder 2026.10.1复核后最终版ppt was checked for Fig1–5 and finalized S1/S2. The same HI308 longitudinal axis and updated module resource layer are used. S3/S4 numbering occupies the next available supplement numbers in that folder; synchronize manuscript references when replacing the old supplementary figure. Upstream Fig1/2/3/4/5 files are not modified.
''',encoding='utf-8')
writer=PdfWriter()
for k in ['S3','S4']:
 for page in PdfReader(O/(k+'.pdf')).pages:writer.add_page(page)
with (O/'Supplementary_Figures_S3_S4.pdf').open('wb') as f:writer.write(f)
files=[(x,'supp3_final/'+x.name) for x in O.iterdir() if x.suffix!='.zip']
for name in ['make_figures.py','add_stability.py','vectorize.py','build_ppt.mjs','package_delivery.py']:files.append((ROOT/'supp3_build'/name,'supp3_build/'+name))
for name in ['corrected_trajectory_predictions.csv','modules/EMSA_module_scores_wide_by_branch_module.csv']:files.append((R/name,'inputs/fig1_materials/Fig1_submission/results/'+name))
files.append((ROOT/'supp_final/module_key.csv','supp_final/module_key.csv'))
manifest='\n'.join(hashlib.sha256(x.read_bytes()).hexdigest()+'  '+arc for x,arc in sorted(files,key=lambda t:t[1]));(O/'SHA256SUMS.txt').write_text(manifest+'\n')
with zipfile.ZipFile(O/'Supplementary_Figures_S3_S4_submission_package.zip','w',zipfile.ZIP_DEFLATED) as z:
 for x,arc in files:z.write(x,arc)
 z.write(O/'SHA256SUMS.txt','supp3_final/SHA256SUMS.txt')
print('consistency passed; ZIP bytes',(O/'Supplementary_Figures_S3_S4_submission_package.zip').stat().st_size)
