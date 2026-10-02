#!/usr/bin/env python3
"""Audit locked Fig6B export against upstream Step57 rows for Fig3E."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
locked = pd.read_csv(root / "data/source/Fig6B_internal_next_HI_score_source_exact.csv")
upstream = pd.read_csv(root / "data/source/EMSA_HI_step57_event_oriented_oof_predictions.csv")
keys = ["participant_id", "sample_t1", "sample_t2", "sample_t3"]
assert not locked.duplicated(keys).any() and not upstream.duplicated(keys).any()
joined = locked.merge(upstream, on=keys, suffixes=("_locked", "_step57"), validate="one_to_one")
assert len(joined) == len(locked) == len(upstream) == 152
columns = ["score_t3", "ridge_delta_base_pred_next_HI_score"]
checks = {}
for col in columns:
    difference = np.abs(joined[col + "_locked"] - joined[col + "_step57"])
    assert difference.max() == 0.0
    checks[col] = float(difference.max())
record = {"matched_windows": 152, "participants": int(locked.participant_id.nunique()),
          "join_columns": keys, "maximum_absolute_differences": checks}
(root / "audit/provenance_check.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
