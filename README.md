# EMSA HI October locked analysis verification

Version: 2026-10-01-locked-verification-2026-10-02

This archive accompanies the locked Fig1–5 and S1–S7. It is distinct from the June 2026 Zenodo version (10.5281/zenodo.20771219). No new DOI or published GitHub release is claimed by this file.

## Reproduction levels

1. `python broad_fusion/reproduce.py` reproduces the 11-branch fusion from archived branch scores and original participant folds: AUROC 0.6551461110196917, n=2506 samples/131 participants. The model is StandardScaler plus L2-penalised LogisticRegression, lbfgs, max_iter=5000, C grid [.01,.03,.1,.3,1,3,10]. The Fig1B word ridge denotes the L2 penalty, not a linear Ridge estimator. All five train/test participant-overlap checks are zero. Broad probabilities and fold fits are provided in expected/.

2. `python locked_supplementary_S5/code/verify.py` validates archived saved downstream and selected MBX-module models against the supplied locked data. Tested maximum differences: downstream 4.44e-16, module 8.88e-16; 152 windows, 66 people. This is saved-model verification, not refitting every upstream assay.

3. `python locked_pipeline/code/reproduce_locked_pipeline.py` refits fusion and trajectory layers from the legacy archived branch-score inputs. Fusion AUROCs: Broad .655146, MBX .824367, HI308 .807473. This conditional refit is NOT numerically identical to the current saved October axes: maximum absolute HI score difference .0015814 and next-HI prediction difference .0012813. Continuous metrics differ slightly from the locked figure values. The locked predictions remain authoritative; do not replace them by the refit outputs. The separate high-state event AUROC remains .926241 and three-state balanced accuracy .717924. Results/replay_vs_locked_numeric_differences.json reports these differences explicitly.

4. S3/S4 and S5 archives retain their original plotting and sensitivity inputs. The original folder name Supplementary_Fig4_materials corresponds to CURRENT Figure S5, not current S4. Original scripts may contain historic absolute workspace paths or references to additional source files. They are source/reference records, not a claim that every file is a standalone raw-assay replay.

## Reporting

reporting/ contains cohort, current-visit treatment and clinical missingness summaries plus the selection audit. Clinical metadata source is HMP2 public hmp2_metadata_2018-08-20.csv; its SHA-256 is recorded in selection_and_missingness_audit.json. HI308 has 82 people: 8 with one, 8 with two visits; the remaining 66 yield ALL 152 consecutive three-visit windows, exactly matching the locked list. Specimen receipt dates are not recruitment dates, and the figure lock date is not an independently established original data-download freeze date.

## Scope not included

This archive does not include a full rerun from raw sequence reads, original mass-spectrometry files or all 11 original assay matrices to branch scores. Those data and source processing workflows are attributed to HMP2/IBDMDB and the external source repositories. No claim of full upstream nesting or complete raw-assay-to-model reproducibility is made. Source access and redistribution terms continue to apply. No new licence is assigned to third-party data. The author must choose/confirm a code licence before public release; the existing repository LICENSE_PLACEHOLDER must not be mistaken for a valid software licence.

## Public archiving

GitHub can host a clearly versioned code/data branch or release; it is not the journal's only acceptable repository. Zenodo can publish a new version of the existing record and issue a new version DOI. zenodo_metadata_draft.json supplies corrected creator names and version/scope metadata, but is not a deposition receipt. Public publishing and licence selection require author confirmation. Do not publish the manuscript, cover letter, author-confirmation notes or other internal files as reproducibility data.
