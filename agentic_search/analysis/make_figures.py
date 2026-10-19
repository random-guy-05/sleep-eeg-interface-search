"""Render the three prespecified held-site search figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
OUT = PACKAGE / "analysis/output"
FIGS = OUT / "figures"
BLUE = "#2271B2"
ORANGE = "#E57A1F"
GREEN = "#2E8B57"
PURPLE = "#8A63B8"
GRAY = "#666A73"


def _summaries() -> pd.DataFrame:
    path = OUT / "paired_trajectory_summary.csv"
    if not path.is_file():
        raise FileNotFoundError("Run agentic_search.analysis.aggregate first")
    return pd.read_csv(path)


def _method_rows(frame: pd.DataFrame, method: str, selector: str,
                 budget: int, label_contains: str | None = None) -> pd.Series | None:
    rows = frame[(frame.method == method) & (frame.selector == selector) & (frame.budget == budget)]
    if label_contains:
        rows = rows[rows.method_label.str.contains(label_contains, regex=False)]
    return None if rows.empty else rows.iloc[0]


def _line(ax, frame: pd.DataFrame, *, method: str, selector: str,
          label: str, color: str, marker: str, budgets: list[int] = [10, 25, 50],
          label_contains: str | None = None, linestyle: str = "-") -> None:
    x, y, low, high = [], [], [], []
    for budget in budgets:
        row = _method_rows(frame, method, selector, budget, label_contains)
        if row is None:
            continue
        x.append(budget)
        y.append(float(row.macro_mean))
        low.append(float(row.macro_mean - row.macro_p2_5))
        high.append(float(row.macro_p97_5 - row.macro_mean))
    if not x:
        return
    positions = [budgets.index(value) if value in budgets else value for value in x]
    ax.errorbar(positions, y, yerr=np.asarray([low, high]), marker=marker, color=color,
                linewidth=2.1, markersize=6.5, capsize=3, linestyle=linestyle, label=label)


def figure1(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.1), constrained_layout=True)
    ax.set_facecolor("#FBFCFE")
    # Map the four prespecified budgets to equal visual spacing so B=500 remains readable.
    budget_map = {10: 0, 25: 1, 50: 2, 500: 3}
    curves = [
        ("random_source_mean", "source_mean", "Random + source mean", BLUE, "o", None),
        ("random_source_robust", "source_robust", "Random + robust", GRAY, "s", None),
        ("evolutionary_robust", "robust", "Evolutionary robust", GREEN, "D", None),
        ("full_agent", "robust", "Full agent · GPT-6 Sol", ORANGE, "o", "GPT-6 Sol"),
    ]
    for method, selector, label, color, marker, label_match in curves:
        x, y, lo, hi = [], [], [], []
        for budget in (10, 25, 50):
            row = _method_rows(frame, method, selector, budget, label_match)
            if row is None:
                continue
            x.append(budget_map[budget]); y.append(float(row.macro_mean))
            lo.append(float(row.macro_mean - row.macro_p2_5)); hi.append(float(row.macro_p97_5 - row.macro_mean))
        if x:
            ax.errorbar(x, y, yerr=np.asarray([lo, hi]), marker=marker, color=color,
                        linewidth=2.1, markersize=7, capsize=3, label=label)
    luna_path = OUT / "luna_generalizability_macro_bootstrap.csv"
    if luna_path.is_file():
        luna = pd.read_csv(luna_path)
        ax.errorbar([budget_map[int(b)] for b in luna.budget], luna.macro_target_auc,
                    yerr=[luna.macro_target_auc - luna.bootstrap_ci_2_5,
                          luna.bootstrap_ci_97_5 - luna.macro_target_auc],
                    marker="v", color=PURPLE, linestyle=":", linewidth=2.1,
                    markersize=7, capsize=3, label="GPT-6 Luna · final choice (participant CI)")
    zero_path = OUT / "luna_zero_shot_scored.csv"
    if zero_path.is_file():
        zero = pd.read_csv(zero_path)
        zero_macro = float(zero.groupby("held_site").target_auc.mean().mean())
        ax.axhline(zero_macro, color=PURPLE, linestyle="--", linewidth=1.2,
                   label=f"Luna zero-shot · {zero_macro:.3f}")
    random_500 = _method_rows(frame, "random_source_mean", "source_mean", 500)
    if random_500 is not None:
        ax.errorbar([budget_map[500]], [random_500.macro_mean],
                    yerr=[[random_500.macro_mean - random_500.macro_p2_5],
                          [random_500.macro_p97_5 - random_500.macro_mean]],
                    marker="P", markersize=9, color=BLUE, linestyle="--", linewidth=1.8,
                    capsize=3, label="Prospective random bank · B=500")
        ax.annotate("500 random quartets", (budget_map[500], random_500.macro_mean),
                    xytext=(-8, 13), textcoords="offset points", ha="right", fontsize=9, color=GRAY)
    random_500_robust = _method_rows(frame, "random_source_robust", "source_robust", 500)
    if random_500_robust is not None:
        ax.errorbar([budget_map[500]], [random_500_robust.macro_mean],
                    yerr=[[random_500_robust.macro_mean - random_500_robust.macro_p2_5],
                          [random_500_robust.macro_p97_5 - random_500_robust.macro_mean]],
                    marker="X", markersize=9, color=GRAY, linestyle="--", linewidth=1.8,
                    capsize=3, label="Random + robust · B=500")
    ax.set_xticks([0, 1, 2, 3], ["10", "25", "50", "500"])
    ax.set_xlabel("Source-evaluated candidate budget")
    ax.set_ylabel("Macro held-site age-conditioned AUROC")
    ax.set_title("Target-blind search efficiency", loc="left", weight="bold", fontsize=16, pad=15)
    ax.text(0, 1.02, "Equal-weight mean across institutions · run percentiles, except Luna's fixed-choice participant bootstrap CI",
            transform=ax.transAxes, color=GRAY, fontsize=9.5)
    ax.set_ylim(0.35, 0.9)
    ax.grid(axis="y", color="#DCE2EA", linewidth=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"figure1_search_efficiency.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def figure2(frame: pd.DataFrame) -> None:
    chosen = []
    for method, selector, label, label_match in (
        ("random_source_robust", "source_robust", "Random + robust", None),
        ("opaque_iterative", "robust", "Opaque iterative", None),
        ("batch_semantic", "robust", "Batch semantic", None),
        ("full_agent", "robust", "Full agent · GPT-6 Sol", "GPT-6 Sol"),
    ):
        row = _method_rows(frame, method, selector, 25, label_match)
        if row is not None:
            chosen.append((label, row))
    if not chosen:
        return
    labels = [item[0] for item in chosen]
    actual = [float(row.macro_mean) for _, row in chosen]
    actual_low = [float(row.macro_mean - row.macro_p2_5) for _, row in chosen]
    actual_high = [float(row.macro_p97_5 - row.macro_mean) for _, row in chosen]
    oracle = [float(row.target_oracle_mean) for _, row in chosen]
    x = np.arange(len(chosen))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [1.1, 1]}, constrained_layout=True)
    colors = [BLUE, GRAY, GREEN, ORANGE, PURPLE][:len(chosen)]
    axes[0].bar(x, actual, color=colors, width=.67, alpha=.9)
    axes[0].errorbar(x, actual, yerr=[actual_low, actual_high], color="#20242B", fmt="none", capsize=4, linewidth=1.3)
    axes[0].set_title("Source-selected held-site AUROC", loc="left", weight="bold")
    axes[0].set_ylabel("Macro target age-conditioned AUROC")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylim(.35, .9)
    axes[0].grid(axis="y", color="#E1E5EA", linewidth=.8)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].bar(x, oracle, color=colors, width=.67, alpha=.9)
    axes[1].set_title("Best target candidate in the queried set", loc="left", weight="bold")
    axes[1].set_ylabel("Macro target oracle AUROC · post hoc")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylim(.35, .9)
    axes[1].grid(axis="y", color="#E1E5EA", linewidth=.8)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("B=25 ablation: semantics, feedback, and search coverage", x=.04, ha="left", fontsize=16, weight="bold")
    fig.text(.04, .94, "Bars show means across paired site-level trajectories; left uncertainty bars are run percentiles.",
             fontsize=9.5, color=GRAY)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"figure2_semantics_feedback_ablation.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def figure3(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), sharey=True, constrained_layout=True)
    colors = {"Random + robust": BLUE, "Evolutionary robust": GREEN,
              "Full agent · GPT-6 Sol": ORANGE}
    methods = [
        ("random_source_robust", "source_robust", "Random + robust", None),
        ("evolutionary_robust", "robust", "Evolutionary robust", None),
        ("full_agent", "robust", "Full agent · GPT-6 Sol", "GPT-6 Sol"),
    ]
    for ax, site in zip(axes, ("I0002", "I0006", "S0001")):
        for method, selector, label, match in methods:
            x, y = [], []
            for budget in (25, 50):
                row = _method_rows(frame, method, selector, budget, match)
                if row is not None:
                    x.append(budget); y.append(float(row[f"{site}_mean"]))
            if x:
                ax.plot(x, y, marker="o", color=colors[label], linewidth=2.1, markersize=6, label=label)
        ax.set_title(site, loc="left", weight="bold", fontsize=13)
        ax.set_xticks([25, 50])
        ax.set_xlabel("Candidate budget")
        ax.grid(axis="y", color="#E1E5EA", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-site age-conditioned AUROC")
    axes[0].set_ylim(.3, .9)
    axes[-1].legend(frameon=False, fontsize=8.6, loc="lower right")
    fig.suptitle("Institution-specific held-site performance", x=.04, ha="left", fontsize=16, weight="bold")
    fig.text(.04, .94, "Site means are shown separately so the macro result cannot hide institution-level variation.",
             fontsize=9.5, color=GRAY)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"figure3_site_level.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_figures() -> list[str]:
    FIGS.mkdir(parents=True, exist_ok=True)
    frame = _summaries()
    figure1(frame)
    figure2(frame)
    figure3(frame)
    return sorted(path.name for path in FIGS.iterdir() if path.is_file())


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print("\n".join(make_figures()))


if __name__ == "__main__":
    main()
