、# Step 55: Optimized multilevel HI-axis delta trajectory model

Input Step54 directory:

`D:\Research Materials\IBD Item1 Meta-Model\Data\v0HMP2-Data\54_HI_axis_preliminary_trajectory_prediction_model_20260706_204043`

Output directory:

`D:\Research Materials\IBD Item1 Meta-Model\Data\v0HMP2-Data\55_HI_axis_multilevel_delta_trajectory_model_20260706_212053`

## Data

- Prediction windows: 152
- Participants: 66
- GroupKFold splits: 5
- Three-state thresholds:
  - low/intermediate: -0.455622
  - intermediate/high: 0.475316
- Five-state thresholds:
  - q20: -0.895304
  - q40: -0.281487
  - q60: 0.287118
  - q80: 0.878588
- Stable delta threshold: 0.151597
- Base features: 7
- State dummy features: 6
- Priority module features: 62

## Best model summary

                      endpoint                             best_model       best_metric  best_value
             delta_HI_t2_to_t3                       ridge_delta_base               MAE    0.420718
direction_from_predicted_delta ridge_delta_base_plus_priority_modules Balanced_accuracy    0.443860
                 next_HI_score                       ridge_delta_base               MAE    0.420718
               next_five_state                       ridge_delta_base Balanced_accuracy    0.542147
        next_high_state_binary                       ridge_delta_base Balanced_accuracy    0.842005
              next_three_state                       ridge_delta_base Balanced_accuracy    0.804603

## Key performance table

                      endpoint                                  model   n      MAE     RMSE        R2  Pearson_r  Spearman_r  Accuracy  Balanced_accuracy  Macro_F1
                 next_HI_score                            persistence 152 0.442510 0.604967  0.663742   0.828913    0.825993       NaN                NaN       NaN
             delta_HI_t2_to_t3                            persistence 152 0.442510 0.604967 -0.001910        NaN         NaN       NaN                NaN       NaN
              next_three_state                            persistence 152      NaN      NaN       NaN        NaN         NaN  0.802632           0.800286  0.799830
               next_five_state                            persistence 152      NaN      NaN       NaN        NaN         NaN  0.539474           0.533704  0.534872
        next_high_state_binary                            persistence 152      NaN      NaN       NaN        NaN         NaN  0.855263           0.836287  0.839231
direction_from_predicted_delta                            persistence 152      NaN      NaN       NaN        NaN         NaN  0.269737           0.333333  0.141623
                 next_HI_score                    linear_continuation 152 0.969317 1.391190 -0.778206   0.561559    0.585932       NaN                NaN       NaN
             delta_HI_t2_to_t3                    linear_continuation 152 0.969317 1.391190 -4.298328  -0.341671   -0.387472       NaN                NaN       NaN
              next_three_state                    linear_continuation 152      NaN      NaN       NaN        NaN         NaN  0.625000           0.614065  0.613260
               next_five_state                    linear_continuation 152      NaN      NaN       NaN        NaN         NaN  0.414474           0.401350  0.387161
        next_high_state_binary                    linear_continuation 152      NaN      NaN       NaN        NaN         NaN  0.763158           0.739280  0.739280
direction_from_predicted_delta                    linear_continuation 152      NaN      NaN       NaN        NaN         NaN  0.250000           0.250457  0.255207
                 next_HI_score                       ridge_delta_base 152 0.420718 0.566089  0.705572   0.841611    0.843223       NaN                NaN       NaN
             delta_HI_t2_to_t3                       ridge_delta_base 152 0.420718 0.566089  0.122726   0.350693    0.347494       NaN                NaN       NaN
              next_three_state                       ridge_delta_base 152      NaN      NaN       NaN        NaN         NaN  0.802632           0.804603  0.802153
               next_five_state                       ridge_delta_base 152      NaN      NaN       NaN        NaN         NaN  0.539474           0.542147  0.543043
        next_high_state_binary                       ridge_delta_base 152      NaN      NaN       NaN        NaN         NaN  0.868421           0.842005  0.850980
direction_from_predicted_delta                       ridge_delta_base 152      NaN      NaN       NaN        NaN         NaN  0.388158           0.403732  0.378492
                 next_HI_score ridge_delta_base_plus_priority_modules 152 0.449277 0.606338  0.662217   0.822029    0.815769       NaN                NaN       NaN
             delta_HI_t2_to_t3 ridge_delta_base_plus_priority_modules 152 0.449277 0.606338 -0.006456   0.252358    0.379643       NaN                NaN       NaN
              next_three_state ridge_delta_base_plus_priority_modules 152      NaN      NaN       NaN        NaN         NaN  0.776316           0.775357  0.773789
               next_five_state ridge_delta_base_plus_priority_modules 152      NaN      NaN       NaN        NaN         NaN  0.519737           0.520911  0.523646
        next_high_state_binary ridge_delta_base_plus_priority_modules 152      NaN      NaN       NaN        NaN         NaN  0.848684           0.818087  0.827741
direction_from_predicted_delta ridge_delta_base_plus_priority_modules 152      NaN      NaN       NaN        NaN         NaN  0.421053           0.443860  0.418920

## Interpretation

The optimized framework outputs predicted_delta_HI, predicted_next_HI_score, predicted three-state status, predicted five-state intensity and predicted direction derived from predicted delta.

If ridge-delta models do not outperform the persistence baseline, the result should still be interpreted as evidence of short-horizon molecular state persistence rather than failed trajectory modeling. The continuous score and predicted delta remain the correct outputs for distinguishing severity within high-HI states.

## Boundary

This is preliminary molecular trajectory prediction. It is not clinical disease prediction or relapse-risk validation.
