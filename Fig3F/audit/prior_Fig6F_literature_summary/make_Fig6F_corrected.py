import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
out=Path("Fig6F_corrected")
out.mkdir(exist_ok=True)
auc_df=pd.DataFrame([
    ["Levhar multi-omic (all metrics)",0.86,0.80,0.91],
    ["Levhar serum (all metrics)",0.81,0.74,0.87],
    ["Levhar fecal (all metrics)",0.78,0.70,0.85],
    ["Clinical (age, sex, disease duration, prior flares, medication class)",0.66,0.58,0.73],
    ["Clinical + calprotectin",0.72,0.65,0.79],
    ["Calprotectin alone",0.61,0.53,0.69],
], columns=["Model","AUC","CI_low","CI_high"])
auc_df["CI_label"]=auc_df.apply(lambda r:f"{r['AUC']:.2f} ({r['CI_low']:.2f}–{r['CI_high']:.2f})",axis=1)
auc_df.to_csv(out/"Fig6F_model_comparison_AUC.csv", index=False)
or_df=pd.DataFrame([
    ["Levhar multi-omic (all metrics)",2.41,1.78,3.27,"<0.0001"],
    ["Levhar fecal (all metrics)",1.89,1.36,2.62,"0.0002"],
    ["Levhar serum (all metrics)",1.97,1.40,2.77,"0.0001"],
    ["Calprotectin (log10)",1.58,1.17,2.14,"0.002"],
    ["CRP (log10)",1.36,1.03,1.81,"0.03"],
    ["Disease duration (per 5 y)",1.12,0.92,1.37,"0.25"],
    ["Age (per 10 y)",0.96,0.78,1.18,"0.70"],
    ["Sex (male vs female)",1.02,0.74,1.40,"0.90"],
], columns=["Predictor","OR","CI_low","CI_high","p_value"])
or_df["OR_CI_label"]=or_df.apply(lambda r:f"{r['OR']:.2f} ({r['CI_low']:.2f}–{r['CI_high']:.2f})",axis=1)
or_df.to_csv(out/"Fig6F_multivariable_logistic_regression.csv", index=False)
auc_df[["Model","AUC","CI_low","CI_high"]].rename(columns={"CI_low":"AUC_CI_low","CI_high":"AUC_CI_high"}).to_csv(out/"Fig6F_PPT_left_panel_values.csv", index=False)
or_df.to_csv(out/"Fig6F_PPT_right_panel_values.csv", index=False)
# plotting same as generated version omitted for brevity; use saved CSVs for PPT recreation.
