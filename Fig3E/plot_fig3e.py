#!/usr/bin/env python3
"""Continuous HI-score calibration from locked, internal next-visit predictions.

Five approximately equal-frequency bins are defined once from model predictions.
Participant bootstrap resamples whole participant trajectories and keeps those
original bin boundaries fixed. This is not binary-event calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

KEY = ["participant_id", "sample_t1", "sample_t2", "sample_t3"]
OBS = "score_t3"
PRED = "ridge_delta_base_pred_next_HI_score"
COLOR = "#217E89"


def load(source: Path) -> pd.DataFrame:
    d = pd.read_csv(source)
    if set(KEY + [OBS, PRED]) - set(d.columns):
        raise ValueError("The locked prediction and observed-score columns are required")
    if d[KEY + [OBS, PRED]].isna().any().any() or d.duplicated(KEY).any():
        raise ValueError("Missing data or duplicate visit windows")
    return d


def analyze(d: pd.DataFrame, nboot: int, seed: int):
    d = d.copy()
    d["predicted_HI"] = d[PRED]
    d["observed_HI"] = d[OBS]
    d["calibration_bin"], edges = pd.qcut(d.predicted_HI, 5, labels=False, retbins=True, duplicates="raise")
    d["calibration_bin"] = d.calibration_bin + 1
    rows = []
    for bin_id, group in d.groupby("calibration_bin", sort=True):
        rows.append({"bin": int(bin_id), "pred_min": float(group.predicted_HI.min()),
                     "pred_max": float(group.predicted_HI.max()),
                     "mean_predicted_HI": float(group.predicted_HI.mean()),
                     "mean_observed_HI": float(group.observed_HI.mean()),
                     "mean_residual": float((group.observed_HI - group.predicted_HI).mean()),
                     "n_windows": len(group), "n_participants": group.participant_id.nunique()})
    bins = pd.DataFrame(rows)
    ids = d.participant_id.unique()
    indices = {p: np.flatnonzero(d.participant_id.to_numpy() == p) for p in ids}
    rng = np.random.default_rng(seed)
    boot_bins = {j: [] for j in range(1, 6)}
    boot_slope, boot_intercept, boot_bias = [], [], []
    for _ in range(nboot):
        sampled = d.iloc[np.concatenate([indices[p] for p in rng.choice(ids, len(ids), replace=True)])]
        for j, group in sampled.groupby("calibration_bin"):
            boot_bins[j].append(float(group.observed_HI.mean()))
        fit = np.polyfit(sampled.predicted_HI, sampled.observed_HI, 1)
        boot_slope.append(float(fit[0])); boot_intercept.append(float(fit[1]))
        boot_bias.append(float((sampled.observed_HI - sampled.predicted_HI).mean()))
    for j in range(1, 6):
        if len(boot_bins[j]) != nboot:
            raise ValueError(f"Bin {j} absent in a bootstrap sample")
        ci = np.percentile(boot_bins[j], [2.5, 97.5])
        bins.loc[bins.bin == j, ["observed_mean_CI_low", "observed_mean_CI_high"]] = ci
    slope, intercept = np.polyfit(d.predicted_HI, d.observed_HI, 1)
    metrics = {
        "n_windows": len(d), "n_participants": d.participant_id.nunique(),
        "n_bins": 5, "bin_edges_predicted_HI": [float(v) for v in edges],
        "calibration_slope": float(slope), "slope_participant_bootstrap_95CI": list(map(float, np.percentile(boot_slope, [2.5, 97.5]))),
        "calibration_intercept": float(intercept), "intercept_participant_bootstrap_95CI": list(map(float, np.percentile(boot_intercept, [2.5, 97.5]))),
        "mean_observed_minus_predicted": float((d.observed_HI - d.predicted_HI).mean()),
        "mean_bias_participant_bootstrap_95CI": list(map(float, np.percentile(boot_bias, [2.5, 97.5]))),
        "MAE": float(mean_absolute_error(d.observed_HI, d.predicted_HI)),
        "R2": float(r2_score(d.observed_HI, d.predicted_HI)),
        "bootstrap_iterations": nboot, "bootstrap_unit": "participant_id", "seed": seed,
        "binning": "Five quantile bins from locked predicted continuous HI; fixed edges during bootstrap",
        "scope": "internal continuous-score calibration; not event-rate calibration"}
    return d, bins, metrics


def draw(d: pd.DataFrame, bins: pd.DataFrame, metrics: dict, out: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 10.5, "axes.linewidth": .8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(5.1, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.25, .82], left=.17, right=.975,
                          bottom=.11, top=.93, hspace=.07)
    ax, hist = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None)
    lo = np.floor(min(d.predicted_HI.min(), d.observed_HI.min()) * 2) / 2
    hi = np.ceil(max(d.predicted_HI.max(), d.observed_HI.max()) * 2) / 2
    ax.plot([lo, hi], [lo, hi], ls="--", color="#5E6267", lw=1.1, label="Identity")
    x, y = bins.mean_predicted_HI.to_numpy(), bins.mean_observed_HI.to_numpy()
    loerr = y - bins.observed_mean_CI_low.to_numpy()
    hierr = bins.observed_mean_CI_high.to_numpy() - y
    ax.errorbar(x, y, yerr=[loerr, hierr], fmt="o-", color=COLOR, lw=1.45,
                markersize=6, capsize=3.3, zorder=4, label="Mean by predicted-HI quintile")
    for _, row in bins.iterrows():
        ax.annotate(f"Q{int(row.bin)} (n={int(row.n_windows)})",
                    (row.mean_predicted_HI, row.mean_observed_HI), xytext=(6, 8),
                    textcoords="offset points", fontsize=8.1, color="#33434A")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel("Mean observed HI at next visit")
    ax.legend(loc="upper left", frameon=False, fontsize=8.3, handlelength=2.2)
    ax.text(.98, .025,
            f"{metrics['n_windows']} windows / {metrics['n_participants']} participants\n"
            f"Slope {metrics['calibration_slope']:.2f} "
            f"[{metrics['slope_participant_bootstrap_95CI'][0]:.2f}, "
            f"{metrics['slope_participant_bootstrap_95CI'][1]:.2f}]\n"
            f"Mean bias {metrics['mean_observed_minus_predicted']:+.3f} HI",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=8.0,
            bbox=dict(boxstyle="round,pad=.28", facecolor="white", edgecolor="none", alpha=.91))
    ax.text(-.18, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=12.5)
    hist.hist(d.predicted_HI, bins=25, color="#A8C8C9", edgecolor="white", linewidth=.4)
    hist.set_xlim(lo, hi)
    hist.set_ylabel("Windows")
    hist.set_xlabel("Predicted HI at next visit")
    hist.text(-.18, 1.04, "b", transform=hist.transAxes, fontweight="bold", fontsize=12.5)
    for a in [ax, hist]:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(direction="out", width=.7, length=3.4)
    ax.tick_params(labelbottom=False)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"Fig3E_continuous_HI_calibration.{ext}", dpi=600)
    plt.close(fig)


def main(source: Path, out: Path, nboot: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    derived = out.parent / "data" / "derived"; derived.mkdir(parents=True, exist_ok=True)
    d, bins, metrics = analyze(load(source), nboot, seed)
    d[KEY + ["predicted_HI", "observed_HI", "calibration_bin"]].to_csv(
        derived / "Fig3E_window_calibration_assignment.csv", index=False)
    bins.to_csv(derived / "Fig3E_quintile_calibration_values.csv", index=False)
    (derived / "Fig3E_calibration_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    draw(d, bins, metrics, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=3486)
    args = parser.parse_args()
    main(args.input, args.out, args.bootstrap, args.seed)
