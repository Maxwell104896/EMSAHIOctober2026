#!/usr/bin/env python3
"""Fig3D: temporal module associations with subsequent molecular HI change.

Observational, lagged associations only. No intervention or causal simulation.
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
from scipy.stats import spearmanr

MODULES = [
    ("Bile acid", "MBX", "mbx_metabolomics_branch_bile_acid_module_state_score"),
    ("SCFA / butyrate", "MGX EC", "mgx_ec_function_branch_scfa_butyrate_module_state_score"),
    ("Oxidative lipid", "MBX", "mbx_metabolomics_branch_oxidative_stress_lipid_module_state_score"),
]
KEY = ["participant_id", "sample_t1", "sample_t2", "sample_t3"]
COLORS = {False: "#478E9A", True: "#C8844E"}


def read_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    columns = KEY + ["score_t2", "score_t3", "target_delta23", "dt12", "dt23"]
    columns += [f"{base}__{suffix}" for _, _, base in MODULES for suffix in ["t1", "t2", "delta12"]]
    if set(columns) - set(data.columns):
        raise ValueError(f"Missing columns: {sorted(set(columns) - set(data.columns))}")
    if data[KEY + ["score_t2", "score_t3", "target_delta23", "dt12", "dt23"]].isna().any().any():
        raise ValueError("Missing outcome, identifiers, or interval")
    if data.duplicated(KEY).any() or (data[["dt12", "dt23"]] <= 0).any().any():
        raise ValueError("Duplicate windows or non-positive time gaps")
    if not np.allclose(data.score_t3 - data.score_t2, data.target_delta23):
        raise ValueError("Future HI delta is inconsistent with observed scores")
    for _, _, base in MODULES:
        pair = data[[f"{base}__t1", f"{base}__t2", f"{base}__delta12"]]
        present = pair.notna().all(axis=1)
        # A modality can be available at only one prior visit; its change is
        # undefined and those windows are omitted for that module only.
        if not np.allclose((pair.loc[present, f"{base}__t2"] - pair.loc[present, f"{base}__t1"]),
                           pair.loc[present, f"{base}__delta12"]):
            raise ValueError(f"Module delta is inconsistent for {base}")
    return data


def slope(x: np.ndarray, y: np.ndarray, controls: np.ndarray | None = None) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x] + ([] if controls is None else [controls]))
    return np.linalg.lstsq(design, y, rcond=None)[0]


def prepare_and_analyze(data: pd.DataFrame, nboot: int, seed: int):
    rng = np.random.default_rng(seed)
    all_rows = []
    summary = []
    curves = {}
    missingness = []
    sensitivity = []
    for name, modality, base in MODULES:
        col = f"{base}__delta12"
        part = data[KEY + ["score_t2", "score_t3", "target_delta23", "dt12", "dt23", col]].dropna().copy()
        part["module"] = name
        part["modality"] = modality
        part["prior_delta_module"] = part[col]
        mean, sd = part[col].mean(), part[col].std(ddof=1)
        part["prior_delta_module_z"] = (part[col] - mean) / sd
        part["future_delta_HI"] = part.target_delta23
        part["future_HI_increase"] = part.future_delta_HI > 0
        all_rows.append(part[KEY + ["module", "modality", "prior_delta_module", "prior_delta_module_z",
                              "score_t2", "score_t3", "dt12", "dt23", "future_delta_HI", "future_HI_increase"]])
        participants = part.participant_id.unique()
        indices = {p: np.flatnonzero(part.participant_id.to_numpy() == p) for p in participants}
        x = part.prior_delta_module_z.to_numpy()
        y = part.future_delta_HI.to_numpy()
        controls = part[["score_t2", "dt12", "dt23"]].to_numpy()
        raw = slope(x, y)
        adjusted = slope(x, y, controls)[1]
        rho = float(spearmanr(x, y).statistic)
        grid = np.linspace(x.min(), x.max(), 100)
        boot_raw, boot_adjusted, boot_rho, boot_lines = [], [], [], []
        for _ in range(nboot):
            chosen = rng.choice(participants, len(participants), replace=True)
            take = np.concatenate([indices[p] for p in chosen])
            xb, yb, cb = x[take], y[take], controls[take]
            if np.std(xb) < 1e-12:
                continue
            fit = slope(xb, yb)
            boot_raw.append(fit[1])
            boot_adjusted.append(slope(xb, yb, cb)[1])
            boot_rho.append(float(spearmanr(xb, yb).statistic))
            boot_lines.append(fit[0] + fit[1] * grid)
        ci_raw = np.percentile(boot_raw, [2.5, 97.5])
        ci_adj = np.percentile(boot_adjusted, [2.5, 97.5])
        ci_rho = np.percentile(boot_rho, [2.5, 97.5])
        band = np.percentile(np.array(boot_lines), [2.5, 97.5], axis=0)
        curves[name] = (grid, raw[0] + raw[1] * grid, band)
        summary.append({"module": name, "modality": modality, "source_delta_column": col,
                        "n_windows": len(part), "n_participants": len(participants),
                        "n_future_HI_increase": int(part.future_HI_increase.sum()),
                        "module_delta_mean": mean, "module_delta_sd": sd,
                        "spearman_rho": rho, "spearman_rho_CI_low": ci_rho[0], "spearman_rho_CI_high": ci_rho[1],
                        "unadjusted_beta_per_SD": raw[1], "unadjusted_beta_CI_low": ci_raw[0],
                        "unadjusted_beta_CI_high": ci_raw[1],
                        "adjusted_beta_per_SD": adjusted, "adjusted_beta_CI_low": ci_adj[0],
                        "adjusted_beta_CI_high": ci_adj[1]})
        trimmed = part[part.prior_delta_module_z.abs() <= 3]
        sensitivity.append({"module": name, "rule": "exclude |module delta z| > 3",
                            "n_retained": len(trimmed), "n_excluded": len(part) - len(trimmed),
                            "unadjusted_beta_per_original_SD": float(slope(trimmed.prior_delta_module_z.to_numpy(),
                                                                            trimmed.future_delta_HI.to_numpy())[1]),
                            "spearman_rho": float(spearmanr(trimmed.prior_delta_module_z,
                                                            trimmed.future_delta_HI).statistic)})
        missingness.append({"module": name, "source_column": col, "all_windows": len(data),
                            "available_windows": len(part), "missing_windows": len(data) - len(part),
                            "available_participants": len(participants)})
    return (pd.concat(all_rows, ignore_index=True), pd.DataFrame(summary),
            pd.DataFrame(missingness), pd.DataFrame(sensitivity), curves)


def plot(rows: pd.DataFrame, stats: pd.DataFrame, curves: dict, out: Path):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 10.5, "axes.linewidth": .8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(10.65, 3.55), sharey=True)
    fig.subplots_adjust(left=.075, right=.985, bottom=.23, top=.82, wspace=.14)
    for i, ((name, modality, _), ax) in enumerate(zip(MODULES, axes)):
        part = rows[rows.module == name]
        row = stats[stats.module == name].iloc[0]
        for rising, label in [(False, "No HI increase"), (True, "HI increase")]:
            sub = part[part.future_HI_increase == rising]
            ax.scatter(sub.prior_delta_module_z, sub.future_delta_HI, s=20, alpha=.72,
                       color=COLORS[rising], edgecolor="white", linewidth=.26,
                       label=label, zorder=3)
        xx, yy, bounds = curves[name]
        ax.fill_between(xx, bounds[0], bounds[1], color="#5F6872", alpha=.15, linewidth=0)
        ax.plot(xx, yy, color="#343A40", lw=1.45)
        ax.axhline(0, color="#777777", lw=.8, ls="--")
        ax.axvline(0, color="#777777", lw=.8, ls=":")
        ax.set_title(f"{modality} · {name}", fontsize=10.6, pad=9)
        ax.text(.98, .04, f"n={row.n_windows} windows / {row.n_participants} people\n"
                f"β={row.unadjusted_beta_per_SD:.2f} [{row.unadjusted_beta_CI_low:.2f}, {row.unadjusted_beta_CI_high:.2f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0,
                bbox=dict(boxstyle="round,pad=.25", facecolor="white", alpha=.88, edgecolor="none"))
        ax.set_xlabel("Prior module change, t1–t2 (SD)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3.3, width=.7)
        ax.text(-.16, 1.15, "abc"[i], transform=ax.transAxes, fontweight="bold", fontsize=12.5)
    axes[0].set_ylabel("Subsequent HI change, t2–t3")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.1, handletextpad=.25)
    fig.text(.53, .035, "Observed temporal associations; lines show unadjusted fits and participant-bootstrap 95% bands",
             ha="center", fontsize=8.5, color="#555555")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"Fig3D_prior_modules_vs_subsequent_HI.{ext}", dpi=600)
    plt.close(fig)


def main(source: Path, out: Path, nboot: int, seed: int):
    out.mkdir(parents=True, exist_ok=True)
    derived = out.parent / "data" / "derived"; derived.mkdir(parents=True, exist_ok=True)
    data = read_data(source)
    rows, stats, missingness, sensitivity, curves = prepare_and_analyze(data, nboot, seed)
    rows.to_csv(derived / "Fig3D_plotting_rows.csv", index=False)
    stats.to_csv(derived / "Fig3D_association_estimates.csv", index=False)
    missingness.to_csv(derived / "Fig3D_module_availability.csv", index=False)
    sensitivity.to_csv(derived / "Fig3D_extreme_value_sensitivity.csv", index=False)
    (derived / "Fig3D_analysis_settings.json").write_text(json.dumps({
        "comparison": "module delta t1-to-t2 versus future HI delta t2-to-t3",
        "future_increase": "score_t3 > score_t2",
        "x_standardization": "mean 0 and sample SD 1 within module's available windows",
        "adjusted_sensitivity_covariates": ["score_t2", "dt12", "dt23"],
        "bootstrap": "participant resampling with all windows retained",
        "iterations": nboot, "seed": seed,
        "scope": "observational association, not intervention or causal effect"}, indent=2), encoding="utf-8")
    plot(rows, stats, curves, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=3486)
    args = parser.parse_args()
    main(args.input, args.out, args.bootstrap, args.seed)
