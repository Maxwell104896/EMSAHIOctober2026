"""Observed next-visit high-HI risk by current state (participant-clustered intervals)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def proportion_confint(k, n, alpha=.05, method="wilson"):
    # Wilson score interval, z=1.959963984540054 for two-sided 95% CI.
    assert alpha == .05 and method == "wilson"
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0., center-half), min(1., center+half)

ORDER = ["HI_low_state", "HI_intermediate_state", "HI_high_state"]
LABELS = ["Low HI", "Intermediate HI", "High HI"]
COLORS = ["#719cc4", "#d6aa56", "#348e85"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, nboot=10000, seed=20260929):
    root = Path(root)
    source = root / "source"
    out = root / "derived"
    figdir = root / "figure"
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    path = source / "EMSA_HI_trajectory_prediction_window_table.csv"
    cvpath = source / "EMSA_HI_multilevel_delta_prediction_cv_predictions.csv"
    d = pd.read_csv(path, dtype={"participant_id": str, "sample_t1": str,
                                 "sample_t2": str, "sample_t3": str})
    cv = pd.read_csv(cvpath, dtype={"participant_id": str, "sample_t1": str,
                                     "sample_t2": str, "sample_t3": str})
    key = ["participant_id", "sample_t1", "sample_t2", "sample_t3"]
    assert not d.duplicated(key).any() and not cv.duplicated(key).any()
    m = d[key + ["state_t2", "target_high_state"]].merge(
        cv[key + ["state_t2", "target_high_state"]], on=key, validate="one_to_one",
        suffixes=("_source", "_cv"))
    assert len(m) == len(d) == len(cv) == 152
    assert (m.state_t2_source == m.state_t2_cv).all()
    assert (m.target_high_state_source == m.target_high_state_cv).all()
    assert d.participant_id.nunique() == 66
    assert set(d.state_t2) == set(ORDER)
    assert d.target_high_state.isin([0, 1]).all()
    assert (d.target_high_state == (d.target_next_state == "HI_high_state").astype(int)).all()
    assert (d.target_high_entry_or_persistence == d.target_high_state).all()

    keep = key + ["time_t1", "time_t2", "time_t3", "dt12", "dt23",
                  "score_t1", "score_t2", "score_t3", "state_t1", "state_t2",
                  "target_next_state", "target_high_state", "target_high_entry_or_persistence"]
    d[keep].to_csv(out / "Fig4B_analysis_windows.csv", index=False)
    rows = []
    for state, label in zip(ORDER, LABELS):
        s = d[d.state_t2 == state]
        k, n = int(s.target_high_state.sum()), len(s)
        lo, hi = proportion_confint(k, n, alpha=.05, method="wilson")
        rows.append({"current_state": state, "label": label, "n_windows": n,
                     "n_participants": int(s.participant_id.nunique()),
                     "next_high_events": k, "next_high_rate": k/n,
                     "wilson_low": lo, "wilson_high": hi,
                     "event_definition": "Observed t3 HI_high_state"})
    rates = pd.DataFrame(rows)

    # Resample independent participant clusters, retaining all of each patient's windows.
    ids = d.participant_id.unique()
    counts = np.zeros((len(ids), 3), dtype=int)
    events = np.zeros((len(ids), 3), dtype=int)
    for k, pid in enumerate(ids):
        sub = d[d.participant_id == pid]
        for j, state in enumerate(ORDER):
            x = sub[sub.state_t2 == state].target_high_state
            counts[k, j], events[k, j] = len(x), int(x.sum())
    rng = np.random.default_rng(seed)
    boot = np.full((nboot, 3), np.nan)
    rr = np.full(nboot, np.nan)
    for i in range(nboot):
        sample = rng.integers(0, len(ids), size=len(ids))
        n_by_state = counts[sample].sum(axis=0)
        e_by_state = events[sample].sum(axis=0)
        for j, state in enumerate(ORDER):
            if n_by_state[j]: boot[i, j] = e_by_state[j] / n_by_state[j]
        if np.isfinite(boot[i, 1]) and boot[i, 1] > 0:
            rr[i] = boot[i, 2] / boot[i, 1]
    rates["cluster_boot_low"] = [np.nanpercentile(boot[:, j], 2.5) for j in range(3)]
    rates["cluster_boot_high"] = [np.nanpercentile(boot[:, j], 97.5) for j in range(3)]
    rates.to_csv(out / "Fig4B_observed_state_rates.csv", index=False)
    pd.DataFrame({"iteration": np.arange(nboot), "rate_low": boot[:, 0],
                  "rate_intermediate": boot[:, 1], "rate_high": boot[:, 2],
                  "RR_high_vs_intermediate": rr}).to_csv(out / "Fig4B_cluster_bootstrap.csv", index=False)
    point_rr = rates.loc[2, "next_high_rate"] / rates.loc[1, "next_high_rate"]
    rr_ci = np.nanpercentile(rr, [2.5, 97.5])
    assert np.isfinite(rr_ci).all()
    summary = {"sample_unit": "ordered t1-t2-t3 prediction window",
               "cluster_unit": "participant_id", "n_participants": len(ids),
               "n_windows": len(d), "endpoint": "observed t3 high-HI state",
               "reference_state": "intermediate HI (26% observed risk)",
               "RR_high_vs_intermediate": point_rr,
               "RR_cluster_bootstrap_95pct": rr_ci.tolist(),
               "n_bootstrap": nboot, "seed": seed,
               "low_vs_intermediate_RR_note": "0 events in low state; no finite CI or RR claim is reported",
               "input_sha256": {p.name: digest(p) for p in (path, cvpath)},
               "boundary": "Observed molecular-state endpoint, not clinical relapse, mortality, or causal effect. Variable follow-up gap; descriptive unadjusted RR."}
    (out / "Fig4B_analysis_summary.json").write_text(json.dumps(summary, indent=2))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "#202c33", "axes.labelcolor": "#202c33"})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.9),
                                  gridspec_kw={"width_ratios": [1.38, 1]}, constrained_layout=True)
    y = np.array([2, 1, 0])
    for j, row in rates.iterrows():
        p, lo, hi = row.next_high_rate, row.wilson_low, row.wilson_high
        ax.plot([lo*100, hi*100], [y[j], y[j]], color=COLORS[j], lw=2)
        ax.plot([lo*100, lo*100], [y[j]-.06, y[j]+.06], color=COLORS[j], lw=1.5)
        ax.plot([hi*100, hi*100], [y[j]-.06, y[j]+.06], color=COLORS[j], lw=1.5)
        ax.scatter([p*100], [y[j]], s=75, color=COLORS[j], zorder=3)
        ax.text(122, y[j]+.16, f"{row.next_high_events}/{row.n_windows} ({p*100:.1f}%)",
                ha="right", va="bottom", fontsize=9)
    ax.set_xlim(-2, 126)
    ax.set_ylim(-.55, 2.6)
    ax.set_yticks(y, LABELS)
    ax.set_xlabel("Observed next-visit high-HI (%)")
    ax.set_title("Observed outcome by current HI state", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=.2)
    ax2.axvline(1, color="#76858c", ls="--", lw=1)
    ax2.errorbar([point_rr], [0], xerr=[[point_rr-rr_ci[0]], [rr_ci[1]-point_rr]],
                 fmt="o", color=COLORS[2], capsize=5, markersize=8, lw=2)
    ax2.text(.03, .87, f"RR {point_rr:.2f} (95% cluster CI {rr_ci[0]:.2f}–{rr_ci[1]:.2f})",
             transform=ax2.transAxes, ha="left", fontsize=9)
    ax2.set_xlim(0, max(5, rr_ci[1]*1.25))
    ax2.set_ylim(-.7, .75)
    ax2.set_yticks([0], ["High vs intermediate"])
    ax2.set_xlabel("Unadjusted risk ratio")
    ax2.set_title("Intermediate HI = reference", loc="left", fontweight="bold")
    ax2.text(.02, .15, "Low HI: 0/47 events; RR not displayed.\nOutcome is molecular high HI, not relapse.",
             transform=ax2.transAxes, fontsize=8, va="bottom", color="#53616a")
    for a in (ax, ax2): a.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figdir / f"Fig4B_next_visit_high_HI.{ext}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"rates": rows, "RR": point_rr, "RR_CI": rr_ci.tolist()}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--nboot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260929)
    a = p.parse_args()
    main(a.root, a.nboot, a.seed)
