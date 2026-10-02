"""Reproduce Fig4A from archived, sample-level module scores.

Run from any directory: python code/make_fig4a.py --root Fig4A_HI_molecular_state_map
PCA is descriptive, not a diagnostic or prognostic model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def projection(df, columns, prefix, out):
    d = df.dropna(subset=columns).copy()
    z = StandardScaler().fit_transform(d[columns].to_numpy(dtype=float))
    pca = PCA(n_components=2, svd_solver="full")
    xy = pca.fit_transform(z)
    d[f"{prefix}_PC1"] = xy[:, 0]
    d[f"{prefix}_PC2"] = xy[:, 1]
    d[["sample_id", "participant_id", "diagnosis_label", "EMSA_HI_exact_state_score", *columns,
       f"{prefix}_PC1", f"{prefix}_PC2"]].to_csv(out, index=False)
    return d, pca.explained_variance_ratio_


def draw(ax, df, prefix, evr, title, label):
    colors = {"nonIBD": "#637887", "UC": "#1f8f9b", "CD": "#d27a43"}
    for group in ("nonIBD", "UC", "CD"):
        sub = df[df.diagnosis_label == group]
        ax.scatter(sub[f"{prefix}_PC1"], sub[f"{prefix}_PC2"], s=21,
                   color=colors[group], alpha=.66, linewidths=0,
                   label=f"{group if group != 'nonIBD' else 'non-IBD'} (n={len(sub)})",
                   rasterized=True)
    ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}% variance)")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
    ax.text(.01, .98, label, transform=ax.transAxes, va="top", ha="left",
            fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def main(root):
    root = Path(root)
    src = root / "source"
    fig = root / "figure"
    derived = root / "derived"
    fig.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    hi_file = src / "Fig4A_MBX_546_exact_source.csv"
    module_file = src / "EMSA_module_scores_wide_by_branch_module.csv"
    hi = pd.read_csv(hi_file, dtype={"sample_id": str, "participant_id": str})
    modules = pd.read_csv(module_file, dtype={"sample_id": str, "participant_id": str})
    assert hi.sample_id.is_unique and modules.sample_id.is_unique
    merged = hi.merge(modules, on="sample_id", how="left", validate="one_to_one",
                      suffixes=("_hi", "_module"), indicator=True)
    assert (merged._merge == "both").all()
    assert (merged.participant_id_hi == merged.participant_id_module).all()
    assert (merged.diagnosis_label_hi == merged.diagnosis_label_module).all()
    merged = merged.drop(columns=["_merge", "participant_id_module", "diagnosis_label_module",
                                  "disease_group_binary_module"])
    merged = merged.rename(columns={"participant_id_hi": "participant_id",
                                    "diagnosis_label_hi": "diagnosis_label",
                                    "disease_group_binary_hi": "disease_group_binary"})
    mbx = [c for c in merged if c.startswith("MBX_metabolomics_branch__")]
    mgx = [c for c in merged if c.startswith(("MGX_EC_function_branch__",
                                                "MGX_pathway_function_branch__",
                                                "MGX_taxonomy_branch__"))]
    assert len(mbx) == 5 and len(mgx) == 14
    assert merged[mbx].notna().all().all()
    mbx_df, mbx_evr = projection(merged, mbx, "MBX", derived / "Fig4A_MBX_546_coordinates.csv")
    joint_df, joint_evr = projection(merged, mbx + mgx, "MBX_MGX",
                                     derived / "Fig4A_MBX_MGX_386_coordinates.csv")
    assert len(mbx_df) == 546 and len(joint_df) == 386

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelcolor": "#202b32", "text.color": "#202b32"})
    fig_obj, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), constrained_layout=True)
    draw(axes[0], mbx_df, "MBX", mbx_evr, "MBX molecular module space", "A")
    draw(axes[1], joint_df, "MBX_MGX", joint_evr, "Paired MBX + MGX module space", "B")
    fig_obj.text(.5, -.012, "Each point is one sample visit; repeated visits from the same participant are retained.",
                 ha="center", fontsize=8, color="#505d66")
    for ext in ("png", "pdf", "svg"):
        fig_obj.savefig(fig / f"Fig4A_HI_molecular_state_map.{ext}", dpi=600,
                        bbox_inches="tight", facecolor="white")
    plt.close(fig_obj)

    summary = {
        "unit": "sample visit", "participant_id_count_MBX": int(mbx_df.participant_id.nunique()),
        "participant_id_count_MBX_MGX": int(joint_df.participant_id.nunique()),
        "sample_count_MBX": len(mbx_df), "sample_count_MBX_MGX": len(joint_df),
        "diagnoses_MBX": mbx_df.diagnosis_label.value_counts().to_dict(),
        "diagnoses_MBX_MGX": joint_df.diagnosis_label.value_counts().to_dict(),
        "features_MBX": mbx, "features_MGX": mgx,
        "variance_explained_MBX": mbx_evr.tolist(),
        "variance_explained_MBX_MGX": joint_evr.tolist(),
        "algorithm": "StandardScaler on each feature, PCA(n_components=2, svd_solver=full)",
        "input_sha256": {p.name: sha256(p) for p in (hi_file, module_file)},
        "note": "Two independently fitted coordinate systems. Descriptive embedding, not validated classification."
    }
    (derived / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ["sample_count_MBX", "sample_count_MBX_MGX",
                    "participant_id_count_MBX", "participant_id_count_MBX_MGX",
                    "diagnoses_MBX_MGX"]}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    main(p.parse_args().root)
