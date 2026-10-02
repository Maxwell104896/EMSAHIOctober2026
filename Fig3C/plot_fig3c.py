#!/usr/bin/env python3
"""Fig3C: next-visit HI prediction error by observed prediction interval.

This script reuses the Step57 out-of-fold prediction table. It does not train
or tune the ridge model. Baselines are deterministic, prespecified formulas.
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
from sklearn.metrics import r2_score

LABELS = ["2–8", "9–12", "13–30"]
MODELS = {
    "HI model": ("ridge_delta_base_pred_next_HI_score", "#187F86"),
    "Last HI (LOCF)": ("locf_pred_score_t3", "#506A9A"),
    "Linear continuation": ("linear_pred_score_t3", "#B77D4D"),
}
KEY = ["participant_id", "sample_t1", "sample_t2", "sample_t3"]


def load_and_check(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = KEY + ["time_t1", "time_t2", "time_t3", "dt12", "dt23",
                    "score_t1", "score_t2", "score_t3"] + [x[0] for x in MODELS.values()]
    missing = sorted(set(needed) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[needed].isna().any().any() or df.duplicated(KEY).any():
        raise ValueError("Required fields contain missing values or duplicate windows")
    if (df.dt12 <= 0).any() or (df.dt23 <= 0).any():
        raise ValueError("Prediction intervals must be positive")
    if not np.allclose(df.dt23, df.time_t3 - df.time_t2):
        raise ValueError("dt23 differs from t3 minus t2")
    if not np.allclose(df.dt12, df.time_t2 - df.time_t1):
        raise ValueError("dt12 differs from t2 minus t1")
    if not np.allclose(df.locf_pred_score_t3, df.score_t2):
        raise ValueError("LOCF column does not equal HI at t2")
    linear = df.score_t2 + (df.score_t2 - df.score_t1) * df.dt23 / df.dt12
    if not np.allclose(df.linear_pred_score_t3, linear):
        raise ValueError("Linear prediction does not match the stated formula")
    df["horizon_bin"] = pd.cut(df.dt23, [0, 8, 12, np.inf], labels=LABELS,
                               right=True, include_lowest=True)
    if df.horizon_bin.isna().any():
        raise ValueError("An interval was not assigned to a bin")
    return df


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for label, group in [("All", df)] + [(s, df[df.horizon_bin == s]) for s in LABELS]:
        obs = group.score_t3.to_numpy()
        for name, (column, _) in MODELS.items():
            pred = group[column].to_numpy()
            records.append({"horizon_bin_weeks": label, "model": name,
                            "n_windows": len(group), "n_participants": group.participant_id.nunique(),
                            "dt23_median_weeks": float(group.dt23.median()),
                            "MAE": float(np.mean(np.abs(obs - pred))),
                            "RMSE": float(np.sqrt(np.mean((obs - pred) ** 2))),
                            "R2": float(r2_score(obs, pred))})
    rows = pd.DataFrame(records)
    diff = []
    for label in ["All"] + LABELS:
        values = rows[rows.horizon_bin_weeks.eq(label)].set_index("model")
        diff.append({"horizon_bin_weeks": label,
                     "LOCF_minus_HI_MAE": values.loc["Last HI (LOCF)", "MAE"] - values.loc["HI model", "MAE"],
                     "Linear_minus_HI_MAE": values.loc["Linear continuation", "MAE"] - values.loc["HI model", "MAE"]})
    return rows, pd.DataFrame(diff)


def cluster_bootstrap(df: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    """Resample participants, retaining all their windows and fixed bins."""
    participants = df.participant_id.unique()
    indices = {p: np.flatnonzero(df.participant_id.to_numpy() == p) for p in participants}
    rng = np.random.default_rng(seed)
    estimates = {("All", name): [] for name in MODELS}
    estimates.update({(s, name): [] for s in LABELS for name in MODELS})
    differences = {s: [] for s in ["All"] + LABELS}
    for _ in range(iterations):
        drawn = rng.choice(participants, len(participants), replace=True)
        sample = df.iloc[np.concatenate([indices[p] for p in drawn])]
        for label, group in [("All", sample)] + [(s, sample[sample.horizon_bin == s]) for s in LABELS]:
            if len(group) == 0:
                continue
            errors = {name: np.abs(group.score_t3.to_numpy() - group[col].to_numpy()).mean()
                      for name, (col, _) in MODELS.items()}
            for name, value in errors.items():
                estimates[(label, name)].append(value)
            differences[label].append(errors["Last HI (LOCF)"] - errors["HI model"])
    results = []
    for label in ["All"] + LABELS:
        for name in MODELS:
            low, high = np.percentile(estimates[(label, name)], [2.5, 97.5])
            results.append({"horizon_bin_weeks": label, "quantity": "MAE", "model": name,
                            "lower_95": float(low), "upper_95": float(high)})
        low, high = np.percentile(differences[label], [2.5, 97.5])
        results.append({"horizon_bin_weeks": label, "quantity": "LOCF_minus_HI_MAE", "model": "paired difference",
                        "lower_95": float(low), "upper_95": float(high)})
    return pd.DataFrame(results)


def draw(metrics: pd.DataFrame, ci: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 10.5, "axes.linewidth": .8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, (ax, dx) = plt.subplots(1, 2, figsize=(8.2, 3.5), gridspec_kw={"width_ratios": [1.25, .9]})
    fig.subplots_adjust(left=.085, right=.985, bottom=.26, top=.88, wspace=.33)
    x = np.arange(len(LABELS))
    for name, (_, color) in MODELS.items():
        y = []; lower = []; upper = []
        for label in LABELS:
            value = metrics[(metrics.horizon_bin_weeks == label) & (metrics.model == name)].iloc[0]
            interval = ci[(ci.horizon_bin_weeks == label) & (ci.model == name) & (ci.quantity == "MAE")].iloc[0]
            y.append(value.MAE); lower.append(value.MAE - interval.lower_95); upper.append(interval.upper_95 - value.MAE)
        ax.errorbar(x, y, yerr=[lower, upper], marker="o", ms=5.1, lw=1.6, capsize=3.2,
                    color=color, label=name, zorder=3)
    ax.set_ylabel("Mean absolute error (HI units)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False, fontsize=8.1, labelspacing=.25)
    ax.text(-.16, 1.06, "a", transform=ax.transAxes, fontweight="bold", fontsize=13)

    differences = []
    for label in LABELS:
        ridge = metrics[(metrics.horizon_bin_weeks == label) & (metrics.model == "HI model")].iloc[0].MAE
        locf = metrics[(metrics.horizon_bin_weeks == label) & (metrics.model == "Last HI (LOCF)")].iloc[0].MAE
        interval = ci[(ci.horizon_bin_weeks == label) & (ci.quantity == "LOCF_minus_HI_MAE")].iloc[0]
        differences.append((locf - ridge, interval.lower_95, interval.upper_95))
    d = np.array(differences)
    dx.axhline(0, color="#555555", lw=1, ls="--")
    dx.errorbar(x, d[:, 0], yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]],
                fmt="o", ms=6, capsize=4, color="#187F86", lw=1.6)
    dx.set_ylabel("MAE difference: last HI − model")
    dx.text(.97, .96, "Positive favors HI model", ha="right", va="top",
            transform=dx.transAxes, fontsize=8.5, color="#444444")
    dx.text(-.18, 1.06, "b", transform=dx.transAxes, fontweight="bold", fontsize=13)
    descriptions = []
    for label in LABELS:
        row = metrics[(metrics.horizon_bin_weeks == label) & (metrics.model == "HI model")].iloc[0]
        descriptions.append(f"{label}\nn={row.n_windows} ({row.n_participants})")
    for a in [ax, dx]:
        a.set_xticks(x, descriptions)
        a.set_xlabel("Observed t2–t3 interval (weeks)")
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(direction="out", width=.7, length=3.5)
        a.set_xlim(-.35, 2.35)
    fig.text(.535, .035, "n = windows (participants)", ha="center", fontsize=8.5, color="#555555")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"Fig3C_HI_error_by_prediction_interval.{ext}", dpi=600)
    plt.close(fig)


def main(source: Path, out: Path, iterations: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    derived = out.parent / "data" / "derived"; derived.mkdir(parents=True, exist_ok=True)
    df = load_and_check(source)
    cols = KEY + ["time_t1", "time_t2", "time_t3", "dt12", "dt23", "score_t1", "score_t2", "score_t3",
                  "horizon_bin"] + [v[0] for v in MODELS.values()]
    df[cols].to_csv(derived / "Fig3C_window_plotting_data.csv", index=False)
    metrics, differences = evaluate(df)
    ci = cluster_bootstrap(df, iterations, seed)
    metrics.to_csv(derived / "Fig3C_metrics_by_interval.csv", index=False)
    differences.to_csv(derived / "Fig3C_paired_MAE_differences.csv", index=False)
    ci.to_csv(derived / "Fig3C_participant_bootstrap_CI.csv", index=False)
    (derived / "Fig3C_analysis_settings.json").write_text(json.dumps({
        "bins_weeks": ["2–8", "9–12", "13–30"], "interval_column": "dt23",
        "bootstrap_unit": "participant_id", "bootstrap_iterations": iterations, "seed": seed,
        "baseline_LOCF": "score_t2", "baseline_linear": "score_t2 + (score_t2-score_t1)*dt23/dt12",
        "model_prediction": "ridge_delta_base_pred_next_HI_score",
        "difference_sign": "LOCF MAE minus HI model MAE; positive favors HI model"}, indent=2), encoding="utf-8")
    draw(metrics, ci, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=3486)
    args = parser.parse_args()
    main(args.input, args.out, args.bootstrap, args.seed)
