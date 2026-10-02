#!/usr/bin/env python3
"""Check the source module values and outcome against the earlier window table."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
early = pd.read_csv(root / "data/source/EMSA_HI_trajectory_prediction_window_table.csv")
late = pd.read_csv(root / "data/source/EMSA_HI_step57_event_oriented_oof_predictions.csv")
keys = ["participant_id", "sample_t1", "sample_t2", "sample_t3"]
assert not early.duplicated(keys).any() and not late.duplicated(keys).any()
joined = early.merge(late, on=keys, suffixes=("_step54", "_step57"), validate="one_to_one")
assert len(joined) == len(early) == len(late) == 152
fields = [
    "score_t2", "score_t3", "target_delta23",
    "mbx_metabolomics_branch_bile_acid_module_state_score__delta12",
    "mgx_ec_function_branch_scfa_butyrate_module_state_score__delta12",
    "mbx_metabolomics_branch_oxidative_stress_lipid_module_state_score__delta12",
]
checks = {}
for name in fields:
    a, b = joined[name + "_step54"], joined[name + "_step57"]
    assert a.isna().equals(b.isna())
    maximum = float(np.nanmax(np.abs(a - b)))
    assert maximum == 0.0
    checks[name] = {"available_windows": int(a.notna().sum()), "maximum_absolute_difference": maximum}
record = {"matched_windows": 152, "participants": int(late.participant_id.nunique()),
          "match_key": keys, "source_field_checks": checks}
(root / "audit/provenance_check.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
