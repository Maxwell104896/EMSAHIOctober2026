# Step 54: HI-axis preliminary trajectory prediction model

Input Step51 directory:

`D:\Research Materials\IBD Item1 Meta-Model\Data\v0HMP2-Data\51_HI_axis_within_subject_state_shift_audit_20260706_202133`

Output directory:

`D:\Research Materials\IBD Item1 Meta-Model\Data\v0HMP2-Data\54_HI_axis_preliminary_trajectory_prediction_model_20260706_204043`

## Data

- HI score column: `emsa_hi_stratification_score`
- Prediction windows: 152
- Participants: 66
- GroupKFold splits: 5
- Stable-zone threshold for direction: 0.149236
- Number of retained features: 70
- Module features initially available: 62
- Dropped all-missing/constant features: 5

## Best-model summary

                    target                             best_model       best_metric  best_value                                                        interpretation
             next_HI_score       last_observation_carried_forward               MAE    0.442510                       Lower MAE indicates better next-score forecast.
             next_HI_state last_observation_carried_forward_state Balanced_accuracy    0.772413 Higher balanced accuracy indicates better state/direction prediction.
next_direction_stable_zone        ridge_logistic_direct_direction Balanced_accuracy    0.440944 Higher balanced accuracy indicates better state/direction prediction.
    next_high_state_binary                        locf_high_state Balanced_accuracy    0.814808 Higher balanced accuracy indicates better state/direction prediction.

## Full performance table

                    target                                  model   n      MAE     RMSE         R2  Pearson_r  Spearman_r  Accuracy  Balanced_accuracy  Macro_F1 notes
             next_HI_score       last_observation_carried_forward 152 0.442510 0.604967   0.663742   0.828913    0.825993       NaN                NaN       NaN      
             next_HI_score          two_point_linear_continuation 152 0.969317 1.391190  -0.778206   0.561559    0.585932       NaN                NaN       NaN      
             next_HI_score       ridge_longitudinal_feature_model 152 1.204397 5.905777 -31.045204   0.120468    0.799790       NaN                NaN       NaN      
             next_HI_state last_observation_carried_forward_state 152      NaN      NaN        NaN        NaN         NaN  0.776316           0.772413  0.774404      
             next_HI_state                 two_point_linear_state 152      NaN      NaN        NaN        NaN         NaN  0.598684           0.586192  0.583995      
             next_HI_state              ridge_score_derived_state 152      NaN      NaN        NaN        NaN         NaN  0.717105           0.711121  0.713789      
             next_HI_state            ridge_logistic_direct_state 152      NaN      NaN        NaN        NaN         NaN  0.631579           0.628616  0.631978      
next_direction_stable_zone                 always_stable_baseline 152      NaN      NaN        NaN        NaN         NaN  0.269737           0.333333  0.141623      
next_direction_stable_zone             two_point_linear_direction 152      NaN      NaN        NaN        NaN         NaN  0.250000           0.250457  0.255207      
next_direction_stable_zone          ridge_score_derived_direction 152      NaN      NaN        NaN        NaN         NaN  0.434211           0.437489  0.433725      
next_direction_stable_zone        ridge_logistic_direct_direction 152      NaN      NaN        NaN        NaN         NaN  0.447368           0.440944  0.440236      
    next_high_state_binary              ridge_logistic_high_state 152      NaN      NaN        NaN        NaN         NaN  0.723684           0.708716  0.705263      
    next_high_state_binary                        locf_high_state 152      NaN      NaN        NaN        NaN         NaN  0.828947           0.814808  0.814808      
    next_high_state_binary                      linear_high_state 152      NaN      NaN        NaN        NaN         NaN  0.723684           0.704780  0.703125      

## Interpretation

This step replaces naive two-point continuation with a participant-grouped longitudinal prediction audit. It evaluates whether t1/t2 HI-axis scores, short-term velocity, time gaps, state strata and module context can predict the next HI score, next HI state, score-movement direction and high-state status.

## Boundary

This is a preliminary molecular trajectory prediction audit. It does not establish clinical relapse prediction, treatment-response prediction or intervention efficacy.
