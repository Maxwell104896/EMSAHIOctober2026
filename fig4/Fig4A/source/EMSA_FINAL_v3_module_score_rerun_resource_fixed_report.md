# EMSA FINAL v3 module score rerun, resource-fixed report

Generated: 2026-07-01T18:07:13.950726

## Purpose

This step reruns supplementary module-level ridge models using FINAL v3 module annotation. Resource paths are loaded from the 31b exact-complete branch-resource audit, avoiding branch-name errors caused by `_X.npz` filenames. This module layer is interpretability-focused and does not replace the locked primary EMSA stacked fusion model.

## Inputs

- FINAL v3 annotation: `/Users/mac/Downloads/HMP2_processed_by_omics/30b3_refresh_FINAL_v3_module_annotation_with_MPX_EC_KO_20260701_171612/EMSA_FINAL_v3_functional_module_feature_annotation_long.csv`
- 31b resource table: `/Users/mac/Downloads/HMP2_processed_by_omics/31b_FINAL_v3_module_resource_repair_audit_20260701_175622/EMSA_31b_annotated_branch_resource_match.csv`
- Primary EMSA scores: `/Users/mac/Downloads/HMP2_processed_by_omics/26_FINAL_EMSA_HMP2_model_output_package_20260701_110428/EMSA_sample_level_scores.csv`

## Modeled module tasks

- Total annotation branch-module pairs: 50
- Modeled tasks: 50
- Skipped tasks: 0

## Task status counts

- modeled_ok: 50

## Top module model performance by AUROC

- MBX_metabolomics_branch__oxidative_stress_lipid: n=546, participants=106, features=18, AUROC=0.7747, AUPRC=0.9061
- MBX_metabolomics_branch__bile_acid: n=546, participants=106, features=159, AUROC=0.7676, AUPRC=0.9137
- Serology_branch__serology_inflammation_anchor: n=210, participants=67, features=11, AUROC=0.7278, AUPRC=0.9034
- MBX_metabolomics_branch__aromatic_amino_acid_indole: n=546, participants=106, features=16, AUROC=0.7107, AUPRC=0.8637
- 16S_composition_branch__SCFA_butyrate: n=178, participants=81, features=8, AUROC=0.6370, AUPRC=0.8190
- Host_transcriptomics_branch__host_immune_activation: n=249, participants=90, features=43, AUROC=0.6240, AUPRC=0.8736
- MGX_pathway_function_branch__aromatic_amino_acid_indole: n=1631, participants=130, features=179, AUROC=0.6111, AUPRC=0.7951
- MTX_EC_activity_branch__mucin_carbohydrate: n=817, participants=109, features=1809, AUROC=0.6061, AUPRC=0.8034
- Host_transcriptomics_branch__epithelial_barrier_repair: n=249, participants=90, features=3, AUROC=0.6020, AUPRC=0.8492
- MTX_EC_activity_branch__SCFA_butyrate: n=817, participants=109, features=6433, AUROC=0.6010, AUPRC=0.8198
- MPX_KO_function_branch__SCFA_butyrate: n=450, participants=89, features=4, AUROC=0.6008, AUPRC=0.7913
- MGX_pathway_function_branch__mucin_carbohydrate: n=1631, participants=130, features=319, AUROC=0.6005, AUPRC=0.8012
- MTX_EC_activity_branch__aromatic_amino_acid_indole: n=817, participants=109, features=1687, AUROC=0.5926, AUPRC=0.8026
- MGX_EC_function_branch__aromatic_amino_acid_indole: n=1636, participants=130, features=2686, AUROC=0.5873, AUPRC=0.7857
- MGX_taxonomy_branch__pathobiont_inflammatory_potential: n=1638, participants=130, features=151, AUROC=0.5836, AUPRC=0.7995
- MPX_EC_function_branch__SCFA_butyrate: n=450, participants=89, features=7, AUROC=0.5834, AUPRC=0.7747
- MTX_EC_activity_branch__pathobiont_inflammatory_potential: n=817, participants=109, features=10234, AUROC=0.5675, AUPRC=0.8160
- MTX_pathway_activity_branch__aromatic_amino_acid_indole: n=805, participants=109, features=204, AUROC=0.5623, AUPRC=0.8125
- MGX_EC_function_branch__SCFA_butyrate: n=1636, participants=130, features=5023, AUROC=0.5619, AUPRC=0.7556
- MBX_metabolomics_branch__SCFA_butyrate: n=546, participants=106, features=4, AUROC=0.5600, AUPRC=0.8039
- Viromics_branch__phage_bacteria_interaction: n=645, participants=105, features=260, AUROC=0.5492, AUPRC=0.7733
- MGX_taxonomy_branch__SCFA_butyrate: n=1638, participants=130, features=22, AUROC=0.5419, AUPRC=0.7538
- MGX_EC_function_branch__oxidative_stress_lipid: n=1636, participants=130, features=521, AUROC=0.5414, AUPRC=0.7353
- MPX_EC_function_branch__mucin_carbohydrate: n=450, participants=89, features=14, AUROC=0.5392, AUPRC=0.7668
- MGX_EC_function_branch__pathobiont_inflammatory_potential: n=1636, participants=130, features=17822, AUROC=0.5333, AUPRC=0.7907

## Top IBD-minus-nonIBD module score shifts

- Serology_branch__serology_inflammation_anchor: delta=0.9755, mean_nonIBD=-0.7813, mean_IBD=0.1942
- MBX_metabolomics_branch__oxidative_stress_lipid: delta=0.9304, mean_nonIBD=-0.7255, mean_IBD=0.2049
- MBX_metabolomics_branch__bile_acid: delta=0.8627, mean_nonIBD=-0.6527, mean_IBD=0.2100
- MBX_metabolomics_branch__aromatic_amino_acid_indole: delta=0.7724, mean_nonIBD=-0.5762, mean_IBD=0.1962
- 16S_composition_branch__SCFA_butyrate: delta=0.6072, mean_nonIBD=-0.4556, mean_IBD=0.1516
- Host_transcriptomics_branch__host_immune_activation: delta=0.5293, mean_nonIBD=-0.4277, mean_IBD=0.1016
- Host_transcriptomics_branch__epithelial_barrier_repair: delta=0.4285, mean_nonIBD=-0.3459, mean_IBD=0.0825
- MGX_taxonomy_branch__pathobiont_inflammatory_potential: delta=0.4023, mean_nonIBD=-0.1318, mean_IBD=0.2705
- MPX_EC_function_branch__SCFA_butyrate: delta=0.3813, mean_nonIBD=-0.2414, mean_IBD=0.1399
- MPX_KO_function_branch__SCFA_butyrate: delta=0.3685, mean_nonIBD=-0.2495, mean_IBD=0.1190
- MGX_pathway_function_branch__aromatic_amino_acid_indole: delta=0.3665, mean_nonIBD=-0.3013, mean_IBD=0.0651
- MTX_EC_activity_branch__pathobiont_inflammatory_potential: delta=0.3662, mean_nonIBD=-0.2078, mean_IBD=0.1584
- MTX_EC_activity_branch__SCFA_butyrate: delta=0.3593, mean_nonIBD=-0.2626, mean_IBD=0.0967
- MTX_EC_activity_branch__mucin_carbohydrate: delta=0.3451, mean_nonIBD=-0.1995, mean_IBD=0.1456
- MGX_pathway_function_branch__mucin_carbohydrate: delta=0.3372, mean_nonIBD=-0.2927, mean_IBD=0.0445
- MBX_metabolomics_branch__SCFA_butyrate: delta=0.3246, mean_nonIBD=-0.2457, mean_IBD=0.0788
- MGX_EC_function_branch__aromatic_amino_acid_indole: delta=0.2871, mean_nonIBD=-0.1269, mean_IBD=0.1602
- MGX_EC_function_branch__pathobiont_inflammatory_potential: delta=0.2815, mean_nonIBD=-0.0468, mean_IBD=0.2346
- MTX_EC_activity_branch__aromatic_amino_acid_indole: delta=0.2766, mean_nonIBD=-0.2110, mean_IBD=0.0656
- MPX_EC_function_branch__mucin_carbohydrate: delta=0.2252, mean_nonIBD=-0.1765, mean_IBD=0.0487
- MTX_pathway_activity_branch__aromatic_amino_acid_indole: delta=0.2221, mean_nonIBD=-0.2512, mean_IBD=-0.0290
- MPX_KO_function_branch__mucin_carbohydrate: delta=0.2076, mean_nonIBD=-0.1655, mean_IBD=0.0421
- MTX_EC_activity_branch__bile_acid: delta=0.1983, mean_nonIBD=-0.0750, mean_IBD=0.1232
- MGX_EC_function_branch__mucin_carbohydrate: delta=0.1964, mean_nonIBD=-0.1542, mean_IBD=0.0422
- MPX_EC_function_branch__oxidative_stress_lipid: delta=0.1931, mean_nonIBD=-0.1043, mean_IBD=0.0888

## Top module-global score associations

- Serology_branch__serology_inflammation_anchor: n=210, Pearson=0.9247, Spearman=0.9115
- MBX_metabolomics_branch__bile_acid: n=546, Pearson=0.6514, Spearman=0.7784
- MBX_metabolomics_branch__oxidative_stress_lipid: n=546, Pearson=0.4461, Spearman=0.5734
- Host_transcriptomics_branch__host_immune_activation: n=249, Pearson=0.5684, Spearman=0.5680
- MBX_metabolomics_branch__aromatic_amino_acid_indole: n=546, Pearson=0.4592, Spearman=0.5465
- Host_transcriptomics_branch__epithelial_barrier_repair: n=249, Pearson=0.4110, Spearman=0.4601
- MGX_EC_function_branch__aromatic_amino_acid_indole: n=1636, Pearson=0.1566, Spearman=0.3679
- MGX_EC_function_branch__SCFA_butyrate: n=1636, Pearson=0.2140, Spearman=0.3484
- MTX_EC_activity_branch__aromatic_amino_acid_indole: n=817, Pearson=0.1873, Spearman=0.3401
- MGX_EC_function_branch__mucin_carbohydrate: n=1636, Pearson=0.1094, Spearman=0.3359
- MGX_pathway_function_branch__aromatic_amino_acid_indole: n=1631, Pearson=0.1297, Spearman=0.3283
- MBX_metabolomics_branch__SCFA_butyrate: n=546, Pearson=0.3587, Spearman=0.3208
- MGX_pathway_function_branch__mucin_carbohydrate: n=1631, Pearson=0.1615, Spearman=0.3163
- MTX_pathway_activity_branch__aromatic_amino_acid_indole: n=805, Pearson=0.2836, Spearman=0.3132
- MPX_EC_function_branch__SCFA_butyrate: n=450, Pearson=0.3061, Spearman=0.3096
- MGX_EC_function_branch__oxidative_stress_lipid: n=1636, Pearson=0.1588, Spearman=0.3081
- MTX_EC_activity_branch__mucin_carbohydrate: n=817, Pearson=0.1526, Spearman=0.3078
- MPX_KO_function_branch__SCFA_butyrate: n=450, Pearson=0.2975, Spearman=0.3069
- MTX_EC_activity_branch__oxidative_stress_lipid: n=817, Pearson=0.1182, Spearman=0.3001
- MTX_EC_activity_branch__SCFA_butyrate: n=817, Pearson=0.1355, Spearman=0.2814
- 16S_composition_branch__pathobiont_inflammatory_potential: n=178, Pearson=0.3359, Spearman=0.2390
- MTX_pathway_activity_branch__mucin_carbohydrate: n=805, Pearson=0.1411, Spearman=0.2311
- MGX_taxonomy_branch__pathobiont_inflammatory_potential: n=1638, Pearson=0.0463, Spearman=0.2199
- MBX_metabolomics_branch__mucin_carbohydrate: n=546, Pearson=0.1541, Spearman=0.2109
- 16S_composition_branch__SCFA_butyrate: n=178, Pearson=0.1807, Spearman=0.2065

## Interpretation boundary

Module scores are supplementary explanatory scores. They should not be interpreted as standalone diagnostic models, clinical thresholds, or replacements for the primary EMSA global state score.

## Outputs

- `/Users/mac/Downloads/HMP2_processed_by_omics/31c_FINAL_v3_module_score_rerun_resource_fixed_20260701_180236/EMSA_FINAL_v3_module_scoring_branch_resource_manifest.csv`
- `/Users/mac/Downloads/HMP2_processed_by_omics/31c_FINAL_v3_module_score_rerun_resource_fixed_20260701_180236/EMSA_module_scores.csv`
- `/Users/mac/Downloads/HMP2_processed_by_omics/31c_FINAL_v3_module_score_rerun_resource_fixed_20260701_180236/EMSA_module_model_performance.csv`
- `/Users/mac/Downloads/HMP2_processed_by_omics/31c_FINAL_v3_module_score_rerun_resource_fixed_20260701_180236/EMSA_module_score_summary_by_diagnosis.csv`
- `/Users/mac/Downloads/HMP2_processed_by_omics/31c_FINAL_v3_module_score_rerun_resource_fixed_20260701_180236/EMSA_module_binary_contrast_summary.csv`
- `/Users/mac/Downloads/HMP2_processed_by_omics/31c_FINAL_v3_module_score_rerun_resource_fixed_20260701_180236/EMSA_module_global_score_association.csv`
- `/Users/mac/Downloads/HMP2_processed_by_omics/31c_FINAL_v3_module_score_rerun_resource_fixed_20260701_180236/EMSA_module_ridge_C_tuning_audit.csv`
- `/Users/mac/Downloads/HMP2_processed_by_omics/31c_FINAL_v3_module_score_rerun_resource_fixed_20260701_180236/EMSA_module_fold_preprocessing_audit.csv`
- `/Users/mac/Downloads/HMP2_processed_by_omics/31c_FINAL_v3_module_score_rerun_resource_fixed_20260701_180236/EMSA_module_scores_wide_by_branch_module.csv`
- `/Users/mac/Downloads/HMP2_processed_by_omics/31c_FINAL_v3_module_score_rerun_resource_fixed_20260701_180236/EMSA_module_scores_wide_by_module_mean.csv`
- `/Users/mac/Downloads/HMP2_processed_by_omics/31c_FINAL_v3_module_score_rerun_resource_fixed_20260701_180236/EMSA_top_module_state_by_sample.csv`
- `/Users/mac/Downloads/HMP2_processed_by_omics/31c_FINAL_v3_module_score_rerun_resource_fixed_20260701_180236/EMSA_module_modeling_task_table.csv`
