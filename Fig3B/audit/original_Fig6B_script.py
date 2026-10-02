import io, zipfile, pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

zip_path = r"/mnt/data/DataforManuscript (2)(1).zip"
member = r"Fig6_Forecasting_and_Longitudinal/data_v0hmp2_data_57_hi_axis_event_oriented_trajectory_endpoint_optimization_20260706_214810_emsa_hi_step57_event_oriented_oof_predictions.csv"

with zipfile.ZipFile(zip_path) as zf:
    df = pd.read_csv(io.BytesIO(zf.read(member)))

pred_col = "ridge_delta_base_pred_next_HI_score"
obs_col = "score_t3"
state_col = "observed_three_state_t3"

# Exact metrics
r2 = r2_score(df[obs_col], df[pred_col])
mae = mean_absolute_error(df[obs_col], df[pred_col])
rho = spearmanr(df[obs_col], df[pred_col]).statistic
print("R2", r2, "MAE", mae, "Spearman", rho)
