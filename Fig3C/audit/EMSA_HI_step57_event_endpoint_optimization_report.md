
# Step 57: Event-oriented HI-axis trajectory endpoint optimization

Input source:

`D:\Research Materials\IBD Item1 Meta-Model\Data\v0HMP2-Data\56_HI_axis_ordinal_and_direction_calibration_20260706_213625`

Input type:

`Step56 calibrated predictions`

Output directory:

`D:\Research Materials\IBD Item1 Meta-Model\Data\v0HMP2-Data\57_HI_axis_event_oriented_trajectory_endpoint_optimization_20260706_214810`

## Models evaluated

                              model                                                             delta_col                                                                  score_col                            source
      ridge_delta_base_uncalibrated                                        ridge_delta_base_pred_delta_HI                                        ridge_delta_base_pred_next_HI_score               Step55 uncalibrated
  ridge_delta_base_score_calibrated                       ridge_delta_base_score_calibrated_pred_delta_HI                       ridge_delta_base_score_calibrated_pred_next_HI_score Step56 score-shrinkage calibrated
    ridge_plus_modules_uncalibrated                  ridge_delta_base_plus_priority_modules_pred_delta_HI                  ridge_delta_base_plus_priority_modules_pred_next_HI_score               Step55 uncalibrated
ridge_plus_modules_score_calibrated ridge_delta_base_plus_priority_modules_score_calibrated_pred_delta_HI ridge_delta_base_plus_priority_modules_score_calibrated_pred_next_HI_score Step56 score-shrinkage calibrated
                        persistence                                             persistence_pred_delta_HI                                             persistence_pred_next_HI_score              Persistence baseline

## Best event endpoints

                      endpoint                          best_model                      model_source   n  prevalence  accuracy  balanced_accuracy  macro_F1  sensitivity  specificity    AUROC    AUPRC
       persistent_high_extreme ridge_plus_modules_score_calibrated Step56 score-shrinkage calibrated 152    0.118421  0.967105           0.981343  0.929519     1.000000     0.962687 0.981343 0.839203
         persistent_high_state                         persistence              Persistence baseline 152    0.243421  0.921053           0.947826  0.902710     1.000000     0.895652 0.959812 0.859074
               next_high_state       ridge_delta_base_uncalibrated               Step55 uncalibrated 152    0.335526  0.848684           0.818191  0.825888     0.725490     0.910891 0.893807 0.809478
             next_high_extreme ridge_plus_modules_score_calibrated Step56 score-shrinkage calibrated 152    0.210526  0.881579           0.787500  0.808242     0.625000     0.950000 0.910417 0.730150
  meaningful_improvement_delta       ridge_delta_base_uncalibrated               Step55 uncalibrated 152    0.453947  0.684211           0.665619  0.660714     0.463768     0.867470 0.681509 0.621400
five_state_downward_transition       ridge_delta_base_uncalibrated               Step55 uncalibrated 152    0.223684  0.815789           0.640578  0.664882     0.323529     0.957627 0.683574 0.390310
    meaningful_worsening_delta     ridge_plus_modules_uncalibrated               Step55 uncalibrated 152    0.381579  0.690789           0.631145  0.631430     0.379310     0.882979 0.688921 0.563156
  five_state_upward_transition     ridge_plus_modules_uncalibrated               Step55 uncalibrated 152    0.223684  0.815789           0.630110  0.653646     0.294118     0.966102 0.624751 0.370610

## Recommended reporting

                      endpoint recommended_use                            role                          best_model  balanced_accuracy  accuracy    AUROC    AUPRC                                                                                        wording
       persistent_high_extreme          report primary_or_secondary_reportable ridge_plus_modules_score_calibrated           0.981343  0.967105 0.981343 0.839203 Predicts persistence of high-extreme molecular intensity among currently high-extreme windows.
         persistent_high_state          report primary_or_secondary_reportable                         persistence           0.947826  0.921053 0.959812 0.859074               Predicts persistence of high-HI molecular state among currently high-HI windows.
               next_high_state          report primary_or_secondary_reportable       ridge_delta_base_uncalibrated           0.818191  0.848684 0.893807 0.809478                       Predicts whether the next sample will fall in a high-HI molecular state.
             next_high_extreme          report primary_or_secondary_reportable ridge_plus_modules_score_calibrated           0.787500  0.881579 0.910417 0.730150    Predicts whether the next sample will fall in the high-extreme molecular intensity stratum.
  meaningful_improvement_delta          report  secondary_directional_endpoint       ridge_delta_base_uncalibrated           0.665619  0.684211 0.681509 0.621400     Predicts a meaningful negative HI-score decrease beyond a fold-calibrated delta threshold.
five_state_downward_transition          report  secondary_directional_endpoint       ridge_delta_base_uncalibrated           0.640578  0.815789 0.683574 0.390310                          Predicts downward movement across ordered molecular intensity strata.
    meaningful_worsening_delta          report  secondary_directional_endpoint     ridge_plus_modules_uncalibrated           0.631145  0.690789 0.688921 0.563156     Predicts a meaningful positive HI-score increase beyond a fold-calibrated delta threshold.
  five_state_upward_transition          report  secondary_directional_endpoint     ridge_plus_modules_uncalibrated           0.630110  0.815789 0.624751 0.370610                            Predicts upward movement across ordered molecular intensity strata.

## Interpretation

This step converts weak raw direction prediction into event-oriented molecular trajectory endpoints. The goal is not to overclaim clinical relapse prediction, but to identify which molecular state-transition endpoints are stable enough to report.

Preferred outputs after this step:
1. predicted_next_HI_score
2. predicted_delta_HI
3. next_high_state
4. persistent_high_state / high-extreme state if performance supports it
5. meaningful molecular worsening/improvement if performance supports it
6. five-state upward/downward transitions as ordinal molecular movement endpoints

## Boundary

These outputs are molecular trajectory endpoints, not clinical disease, relapse, treatment-response, or intervention predictions.
