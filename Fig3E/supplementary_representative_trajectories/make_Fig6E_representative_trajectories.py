import zipfile, io, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

outdir = Path("./Fig6E_exact")
outdir.mkdir(exist_ok=True)
zip_path = "DataforManuscript (2)(1).zip"
source_csv = "Fig6_Forecasting_and_Longitudinal/data_v0hmp2_data_30c3_serology_participant_timepoint_alignment_audit_20260701_174354_emsa_30c3_no_serology_scores_metadata_enriched.csv"
with zipfile.ZipFile(zip_path) as zf:
    df = pd.read_csv(io.BytesIO(zf.read(source_csv)))

src = df[["participant_id_final","diagnosis_label_final","time_value","EMSA_global_state_score"]].dropna().copy()
agg = (src.groupby(["participant_id_final","diagnosis_label_final","time_value"], as_index=False)["EMSA_global_state_score"].mean()
         .rename(columns={"participant_id_final":"participant_id","diagnosis_label_final":"diagnosis","time_value":"time_months","EMSA_global_state_score":"EMSA_HI_score"}))

selection_info = [('H4044', 'UC', 'Example 1', 'Relatively stable trajectory'), ('C3016', 'CD', 'Example 2', 'Progressive worsening trajectory'), ('C3003', 'UC', 'Example 3', 'Relapsing-remitting trajectory')]
low_thr = -0.75
high_thr = 0.75

def state_label(x):
    if x < low_thr:
        return "Low"
    elif x > high_thr:
        return "High"
    return "Intermediate"

sel = []
for pid, diag, ex, ptype in selection_info:
    g = agg[agg["participant_id"] == pid].sort_values("time_months").copy()
    g["example"] = ex
    g["pattern_label"] = ptype
    sel.append(g)
sel_df = pd.concat(sel, ignore_index=True)
sel_df["state"] = sel_df["EMSA_HI_score"].map(state_label)
sel_df.to_csv(outdir / "Fig6E_representative_trajectory_points.csv", index=False)

summary = []
for pid, diag, ex, ptype in selection_info:
    g = sel_df[sel_df["participant_id"] == pid].sort_values("time_months")
    y = g["EMSA_HI_score"].values
    t = g["time_months"].values
    summary.append({"example": ex, "participant_id": pid, "diagnosis": diag, "pattern_label": ptype,
                    "n_timepoints": len(g), "time_min_months": float(t.min()), "time_max_months": float(t.max()),
                    "start_score": float(y[0]), "end_score": float(y[-1]), "min_score": float(y.min()),
                    "max_score": float(y.max()), "mean_score": float(y.mean()), "score_range": float(y.max()-y.min())})
pd.DataFrame(summary).to_csv(outdir / "Fig6E_representative_trajectory_summary.csv", index=False)
sel_df[["example","participant_id","diagnosis","pattern_label","time_months","EMSA_HI_score","state"]].to_csv(outdir / "Fig6E_PPT_ready_values.csv", index=False)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
state_bg = {"Low":"#dce9f9","Intermediate":"#fbe6b8","High":"#f6c4c4"}
line_colors = {"UC":"#E69F00","CD":"#56B4E9","nonIBD":"#009E73"}
for ax, (pid, diag, ex, ptype) in zip(axes, selection_info):
    g = sel_df[sel_df["participant_id"] == pid].sort_values("time_months")
    ax.axhspan(-2.5, low_thr, color=state_bg["Low"], alpha=0.35, zorder=0)
    ax.axhspan(low_thr, high_thr, color=state_bg["Intermediate"], alpha=0.25, zorder=0)
    ax.axhspan(high_thr, 3.5, color=state_bg["High"], alpha=0.35, zorder=0)
    ax.axhline(low_thr, color="gray", lw=1, ls="--")
    ax.axhline(high_thr, color="gray", lw=1, ls="--")
    ax.plot(g["time_months"], g["EMSA_HI_score"], color=line_colors.get(diag, "#333333"), lw=2)
    ax.scatter(g["time_months"], g["EMSA_HI_score"], color=line_colors.get(diag, "#333333"), s=26, zorder=3)
    ax.set_title(f"{ex}\n{ptype}", pad=10)
    ax.text(0.02, 0.97, f"{pid} ({diag})", transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))
    ax.set_xlim(g["time_months"].min()-1, g["time_months"].max()+1)
    ax.set_xlabel("Follow-up time")
    ax.grid(axis="y", alpha=0.2)
axes[0].set_ylabel("EMSA-HI score")
axes[0].set_ylim(-2.5, 3.5)
axes[0].text(axes[0].get_xlim()[0]+0.2, -1.75, "Low state", fontsize=9, color="#3a6ea5")
axes[0].text(axes[0].get_xlim()[0]+0.2, -0.1, "Intermediate state", fontsize=9, color="#8a6d1d")
axes[0].text(axes[0].get_xlim()[0]+0.2, 2.1, "High state", fontsize=9, color="#a94442")
handles = [Patch(facecolor=state_bg["Low"], edgecolor="none", label="Low (< -0.75)"),
           Patch(facecolor=state_bg["Intermediate"], edgecolor="none", label="Intermediate (-0.75 to 0.75)"),
           Patch(facecolor=state_bg["High"], edgecolor="none", label="High (> 0.75)")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5,-0.02))
fig.suptitle("Fig6E. Representative longitudinal EMSA-HI trajectories", y=1.02, fontsize=13)
fig.tight_layout(rect=[0,0.06,1,0.96])
fig.savefig(outdir / "Fig6E_representative_trajectories.png", dpi=300, bbox_inches="tight")
fig.savefig(outdir / "Fig6E_representative_trajectories.pdf", bbox_inches="tight")
fig.savefig(outdir / "Fig6E_representative_trajectories.svg", bbox_inches="tight")
