"""Primary figures for the representation-search manuscript pivot."""
import numpy as np
from .plotting import _plt, _save_figure

LABELS = {
    "stable4_fixed": "Fixed Stable4*", "stable4_equal": "Equal Stable4*",
    "learned_stable4_4": "Learned Stable4*", "rem_theta_prior": "REM theta*",
    "physiology_elastic_net": "Physiology elastic net", "all_head_elastic_net": "All-head elastic net",
    "ridge_841": "841-head ridge", "ridge_1024": "1024-latent ridge",
    "source_selected_random4": "Source-selected random-4",
    "pure_l1_sensitivity": "Pure L1 (sensitivity)",
    **{f"pca_{k}": f"PCA-{k}" for k in (4, 8, 16, 32)},
}


def plot_probe_interfaces(site_table, random4, output):
    plt = _plt()
    fig, (ax, hist) = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1.6, 1]})
    table = site_table.copy()
    if "representation" in table:
        table = table.set_index("representation")
    if table.Macro.isna().any():
        raise ValueError("Primary figure requires every interface, including the source-selected winner")
    y = np.arange(len(table))
    ax.axhspan(-.5, 3.5, color="#E69F00", alpha=.10)
    for site, color, marker, offset in (("I0002", "#0072B2", "o", -.16), ("I0006", "#D55E00", "s", 0), ("S0001", "#009E73", "^", .16)):
        ax.scatter(table[site], y+offset, s=28, color=color, marker=marker, alpha=.75, label=site)
    prior = table.evidence_role.eq("historically_discovered_reference").to_numpy()
    ax.scatter(table.Macro.to_numpy()[prior], y[prior], marker="D", facecolors="none", edgecolors="black", s=70, label="Macro: historical reference")
    ax.scatter(table.Macro.to_numpy()[~prior], y[~prior], marker="D", color="black", s=35, label="Macro: nested / sensitivity")
    ax.set_yticks(y, [LABELS.get(name, name) for name in table.index])
    ax.set_ylim(len(table)-.5, -.5)
    ax.set_xlim(.25, .9)
    ax.set(xlabel="Held-site age-conditioned AUROC", title="A  Interfaces under institutional shift")
    ax.axvline(.5, color=".7", linestyle=":")
    ax.grid(axis="x", alpha=.15)
    hist.hist(random4.macro_age_auc, bins=25, color="#8D99AE", edgecolor="white", label="500 fixed random quartets")
    for name, color, style in (("stable4_fixed", "#E69F00", "--"), ("learned_stable4_4", "#777777", ":"), ("source_selected_random4", "#0072B2", "-")):
        if name not in table.index:
            continue
        hist.axvline(table.loc[name, "Macro"], color=color, linestyle=style, linewidth=2, label=LABELS[name])
    hist.set(xlabel="Three-site macro age-conditioned AUROC", ylabel="Number of quartets", title="B  Four-dimensional interfaces")
    hist.legend(frameon=False, fontsize=8, loc="upper left")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(.5, .025))
    fig.text(.5, .01, "* Previously discovered coordinates; discovery was not nested. Shading and hollow diamonds mark references.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, 1))
    _save_figure(fig, output)
    return fig


def _budget_axis(ax, frame, metric, title, color, limits):
    ax.plot(frame.budget, frame[f"{metric}_median"], "o-", color=color)
    ax.fill_between(frame.budget, frame[f"{metric}_low95"], frame[f"{metric}_high95"], color=color, alpha=.18)
    ax.set_xscale("log")
    ax.set_xticks(frame.budget, frame.budget.astype(str))
    ax.set(xlabel="Candidate quartets searched", title=title, ylim=limits)
    ax.grid(alpha=.15)


def plot_search_hero(summary, panel, correlations, output):
    """Canonical Figure 1: all frozen budgets, 95% ordering intervals and gap."""
    plt = _plt()
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 3)
    macro = summary.loc[summary.outer_site.eq("macro")].sort_values("budget")
    for j, (metric, title, color, limits) in enumerate((
            ("source_auc", "A  Source validation", "#0072B2", (.3, 1.)),
            ("target_auc", "B  Held-site transport", "#D55E00", (.3, 1.)),
            ("gap", "C  Source–target gap", "#009E73", (-.15, .35)))):
        ax = fig.add_subplot(grid[0, j])
        _budget_axis(ax, macro, metric, title, color, limits)
        ax.set_ylabel("Macro AUROC" if metric != "gap" else "Source minus target AUROC")
        if metric == "gap":
            ax.axhline(0, color=".6", linestyle=":")
    for j, (site, group) in enumerate(panel.groupby("outer_site", sort=True)):
        ax = fig.add_subplot(grid[1, j])
        ax.scatter(group.source_auc, group.target_auc, s=12, color="#4C78A8", alpha=.35, edgecolors="none")
        winner = group.sort_values(["source_auc", "draw"], ascending=[False, True]).iloc[0]
        ax.scatter([winner.source_auc], [winner.target_auc], s=95, marker="*", color="#D55E00", label="Source-selected best")
        stats = correlations.loc[correlations.outer_site.eq(site)].iloc[0]
        ax.text(.035, .96, f"Spearman ρ = {stats.spearman:.3f}", transform=ax.transAxes, va="top", fontsize=9)
        ax.set(xlabel="Source-validation AUROC", ylabel="Held-site AUROC", title=f"D{j+1}  Held out: {site}", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
        ax.grid(alpha=.15)
        if j == 2:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(.5, .015, "Median and central 95% interval over 100 saved candidate permutations; not patient or new-site confidence intervals.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1), h_pad=2)
    _save_figure(fig, output)
    return fig


def plot_site_budget_robustness(summary, output):
    """Identical site-level source, transport and signed-gap curves."""
    plt = _plt()
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    sites = ("I0002", "I0006", "S0001")
    for i, site in enumerate(sites):
        frame = summary.loc[summary.outer_site.eq(site)].sort_values("budget")
        for j, (metric, title, color) in enumerate((
                ("source_auc", "Source validation", "#0072B2"),
                ("target_auc", "Held-site transport", "#D55E00"),
                ("gap", "Source–target gap", "#009E73"))):
            limits = (-.6, .8) if metric == "gap" else (-.02, 1.02)
            ax = axes[i,j]
            _budget_axis(ax, frame, metric, f"{site}: {title}", color, limits)
            ax.set_ylabel("AUROC" if metric != "gap" else "Source minus target AUROC")
            if metric == "gap":
                ax.axhline(0, color=".6", linestyle=":")
    fig.text(.5, .01, "All sites and all five budgets; median and central 95% candidate-order intervals. B=500 contains the entire bank.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    _save_figure(fig, output)
    return fig


def plot_representation_search(summary, panel, correlations, output):
    plt = _plt()
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 6)
    for metric, title, location, color in (("source_auc", "A  Source validation", grid[0, :3], "#0072B2"), ("target_auc", "B  Transport of the source-selected winner", grid[0, 3:], "#D55E00")):
        ax = fig.add_subplot(location)
        ax.plot(summary.budget, summary[f"{metric}_median"], "o-", color=color)
        ax.fill_between(summary.budget, summary[f"{metric}_q25"], summary[f"{metric}_q75"], color=color, alpha=.18)
        ax.set_xscale("log")
        ax.set_xticks(summary.budget, summary.budget.astype(str))
        ax.set(xlabel="Candidate quartets searched on source institutions", ylabel="Three-fold mean age-conditioned AUROC", title=title, ylim=(.3, 1.))
        ax.grid(alpha=.15)
    for index, (site, group) in enumerate(panel.groupby("outer_site", sort=True)):
        ax = fig.add_subplot(grid[1, index*2:(index+1)*2])
        ax.scatter(group.source_auc, group.target_auc, s=12, color="#4C78A8", alpha=.35, edgecolors="none")
        winner = group.sort_values(["source_auc", "draw"], ascending=[False, True]).iloc[0]
        ax.scatter([winner.source_auc], [winner.target_auc], s=100, marker="*", color="#D55E00", label="Source-selected best")
        stats = correlations.loc[correlations.outer_site.eq(site)].iloc[0]
        ax.text(.04, .94, f"Spearman ρ = {stats.spearman:.2f}\nPearson r = {stats.pearson:.2f}", transform=ax.transAxes, va="top", fontsize=9)
        # Include the complete valid AUROC range: low-scoring candidates are
        # evidence too, and must not disappear below a manually chosen limit.
        ax.set(xlabel="Source-transfer validation AUROC", ylabel="Held-site AUROC", title=f"C{index+1}  Held out: {site}", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
        ax.grid(alpha=.15)
        if index == 2:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(.5, .015, "Curves: median and IQR over 100 predetermined candidate permutations, not patient confidence intervals.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1), h_pad=2)
    _save_figure(fig, output)
    return fig
