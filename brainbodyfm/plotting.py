"""Small plotting helpers for the three planned BrainBodyFM figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DISPLAY_NAMES = {
    "physiology_elastic_net": "Physiology elastic net",
    "physiology": "Physiology elastic net",
    "all_head_elastic_net": "All-head elastic net",
    "ridge_841": "841-head ridge", "semantic_841": "841-head ridge",
    "ridge_1024": "1024-latent ridge", "latent_1024": "1024-latent ridge",
    "learned_stable4_4": "Learned Stable4 (prior coordinates)",
    "stable4": "Learned Stable4 (prior coordinates)",
}
_FONTS_READY = False
INTERFACE_COLORS = {
    "physiology_elastic_net": "#0072B2", "physiology": "#0072B2",
    "all_head_elastic_net": "#009E73",
    "ridge_841": "#D55E00", "semantic_841": "#D55E00",
    "ridge_1024": "#CC79A7", "latent_1024": "#CC79A7",
    "learned_stable4_4": "#555555", "stable4": "#555555",
}


def _save_figure(fig, output):
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf", ".svg"):
            fig.savefig(path.with_suffix(suffix), dpi=240, bbox_inches="tight")


def _plt():
    global _FONTS_READY
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to create BrainBodyFM figures") from exc
    if not _FONTS_READY:
        # Homebrew/runtime upgrades can leave a system font cache without the
        # current bundled fonts. Register only in this process; do not change
        # the user's shared Matplotlib cache or system font installation.
        for font in sorted((Path(matplotlib.get_data_path()) / "fonts" / "ttf").glob("*.ttf")):
            font_manager.fontManager.addfont(str(font))
        matplotlib.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
        _FONTS_READY = True
    return plt


def plot_interface_benchmark(
    summary: pd.DataFrame,
    *,
    random_controls: pd.DataFrame | None = None,
    pca_controls: pd.DataFrame | None = None,
    sparse_path: pd.DataFrame | None = None,
    stable4_auc: float | None = None,
    output: str | Path | None = None,
):
    """Plot Figure 1: discrimination against interface dimensionality."""
    plt = _plt()
    has_path = sparse_path is not None and not sparse_path.empty
    fig, axes = plt.subplots(1, 2 if has_path else 1, figsize=(12, 6) if has_path else (8, 6))
    ax = axes[0] if has_path else axes
    required = {"representation", "dimensionality", "macro_age_auc"}
    if not required.issubset(summary.columns):
        raise ValueError(f"summary needs columns {sorted(required)}")
    palette = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#555555"]
    for index, (_, row) in enumerate(summary.iterrows()):
        color = INTERFACE_COLORS.get(str(row["representation"]), palette[index % len(palette)])
        ax.scatter(row["dimensionality"], row["macro_age_auc"], label=DISPLAY_NAMES.get(str(row["representation"]), str(row["representation"])), s=62, color=color, zorder=4)
        if "ci_low" in row and np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"]):
            ax.vlines(row["dimensionality"], row["ci_low"], row["ci_high"], color=color, alpha=.65, linewidth=1.5)
    if pca_controls is not None and not pca_controls.empty:
        ax.plot(pca_controls["k"], pca_controls["macro_age_auc"], "o--", color="0.45", label="PCA-k")
    if random_controls is not None and not random_controls.empty:
        grouped = random_controls.groupby("k")["macro_age_auc"]
        centers = grouped.median()
        lows = grouped.quantile(0.25)
        highs = grouped.quantile(0.75)
        ax.errorbar(centers.index, centers, yerr=[centers - lows, highs - centers], fmt="s", color="0.2", label="random-k IQR")
    if has_path:
        required_path = {"outer_site", "C", "k_C", "V_C"}
        if not required_path.issubset(sparse_path.columns):
            raise ValueError(f"sparse_path needs columns {sorted(required_path)}")
        for site, group in sparse_path.groupby("outer_site", sort=True):
            # Support size need not grow monotonically with C. Keep the
            # actual regularization sequence instead of relinking by size.
            ordered = group.sort_values("C")
            axes[1].plot(
                ordered["k_C"], ordered["V_C"], linewidth=1.0, marker="o", markersize=3, alpha=0.65,
                label=f"Held-out fold: {site}",
            )
        axes[1].set(xlabel="Active coordinates fitted on source pool", ylabel="Source-transfer validation AUROC", title="B  Prespecified regularization path")
        axes[1].set_xscale("symlog", linthresh=1)
        axes[1].set_xlim(0, max(1., float(sparse_path.k_C.max()) * 1.15))
        axes[1].legend(frameon=False, fontsize=8)
        axes[1].grid(alpha=.15)
    if stable4_auc is not None and np.isfinite(stable4_auc):
        ax.axhline(stable4_auc, linestyle="--", color="#E69F00", label="Stable4 reference")
    ax.set_xlabel("Dimensions (mean active coordinates for elastic net)")
    ax.set_ylabel("Three-site macro age-conditioned AUROC")
    ax.set_title("A  Held-out institution performance")
    ax.set_xscale("symlog", linthresh=1)
    dimensions = summary.dimensionality.astype(float).tolist()
    for controls in (random_controls, pca_controls):
        if controls is not None and not controls.empty:
            dimensions.extend(controls.k.astype(float).tolist())
    ax.set_xlim(max(0., min(dimensions) * .8), max(1., max(dimensions) * 1.3))
    ax.grid(alpha=.15)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3 if has_path else 2,
               frameon=False, fontsize=8, bbox_to_anchor=(.5, .01))
    fig.tight_layout(rect=(0, .14 if has_path else .19, 1, 1))
    _save_figure(fig, output)
    return fig, ax


def plot_label_efficiency(
    label_results: pd.DataFrame,
    *,
    fixed_reference: dict[str, float] | None = None,
    output: str | Path | None = None,
):
    """Plot Figure 2 with median and IQR across label draws."""
    plt = _plt()
    required = {"representation", "label_fraction"}
    if not required.issubset(label_results.columns):
        raise ValueError(f"label_results needs columns {sorted(required)}")
    if "median_macro_age_auc" in label_results.columns:
        curve = label_results.copy()
    else:
        from .label_efficiency import summarize_label_efficiency

        _replicates, curve = summarize_label_efficiency(label_results)
    fig, ax = plt.subplots(figsize=(7.2, 5.7))
    for name, stats in curve.groupby("representation", sort=False):
        stats = stats.sort_values("label_fraction")
        line, = ax.plot(
            stats["label_fraction"], stats["median_macro_age_auc"],
            marker="o", label=DISPLAY_NAMES.get(str(name), str(name)),
            color=INTERFACE_COLORS.get(str(name)),
        )
        ax.fill_between(
            stats["label_fraction"], stats["q25_macro_age_auc"],
            stats["q75_macro_age_auc"], alpha=0.15, color=line.get_color(),
        )
    for name, value in (fixed_reference or {}).items():
        ax.axhline(value, linestyle="--", color="#E69F00", label=f"{name} fixed reference")
    ax.set_xlabel("Fraction of source outcome labels")
    ax.set_ylabel("Three-site macro age-conditioned AUROC")
    ax.set_xticks([.25, .5, .75, 1.0], ["25%", "50%", "75%", "100%"])
    ax.grid(alpha=.15)
    ax.set_title("Downstream label efficiency under institutional shift")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .17, 1, 1))
    _save_figure(fig, output)
    return fig, ax


def plot_family_stability(
    family_results: pd.DataFrame,
    *,
    output: str | Path | None = None,
):
    """Plot Figure 3 candidate: family-level bootstrap selection frequency."""
    plt = _plt()
    required = {"outer_site", "physiological_family", "bootstrap_frequency"}
    if not required.issubset(family_results.columns):
        raise ValueError(f"family_results needs columns {sorted(required)}")
    pivot = family_results.pivot(index="physiological_family", columns="outer_site", values="bootstrap_frequency").fillna(0)
    ax = pivot.plot(kind="barh", figsize=(8.4, max(4.8, 0.36 * len(pivot) + .8)))
    ax.set_xlabel("Bootstrap inclusion frequency")
    ax.set_ylabel("Physiological family")
    ax.set_title("Selection stability across physiological families")
    ax.set_yticklabels([str(label).replace("_", " ") for label in pivot.index])
    ax.set_xlim(0, 1.02)
    fig = ax.figure
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(handles, labels, title="Held-out site", ncol=3, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, .16, 1, 1))
    _save_figure(fig, output)
    return fig, ax
