#!/usr/bin/env python3
"""Check that plotted Step57 values match the earlier Step55 CV predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
step55 = pd.read_csv(root / "audit/EMSA_HI_multilevel_delta_prediction_cv_predictions.csv")
step57 = pd.read_csv(root / "data/source/EMSA_HI_step57_event_oriented_oof_predictions.csv")
locked = pd.read_csv(root / "audit/Fig6B_internal_next_HI_score_source_exact.csv")
key = ["participant_id", "sample_t1", "sample_t2", "sample_t3"]
for table in [step55, step57, locked]:
    assert not table.duplicated(key).any()
left = step55.merge(step57, on=key, suffixes=("_55", "_57"), validate="one_to_one")
right = locked.merge(step57, on=key, suffixes=("_locked", "_57"), validate="one_to_one")
assert len(left) == len(right) == len(step55) == len(step57) == len(locked) == 152
columns = ["score_t3", "locf_pred_score_t3", "linear_pred_score_t3",
           "ridge_delta_base_pred_next_HI_score"]
diff = {column: float(np.max(np.abs(left[column + "_55"] - left[column + "_57"])))
        for column in columns}
assert all(value < 1e-12 for value in diff.values())
locked_diff = {column: float(np.max(np.abs(right[column + "_locked"] - right[column + "_57"])))
               for column in ["score_t3", "ridge_delta_base_pred_next_HI_score"]}
assert all(value == 0 for value in locked_diff.values())
result = {"matched_windows": 152, "unique_participants": int(step57.participant_id.nunique()),
          "step55_to_step57_max_difference": diff,
          "locked_fig6b_to_step57_max_difference": locked_diff,
          "meaning": "The plotted model and baselines are the same window-level predictions as the earlier CV table."}
(root / "audit/provenance_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
