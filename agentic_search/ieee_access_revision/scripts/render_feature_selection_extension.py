"""Create tables and the updated budget plot from aggregate extension results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
OUT = REV / "results/feature_selection_extension"
PAPER = REV / "paper"
LABELS = {"full_agent_B10": "Language model, ten evaluations",
          "l1_top4": "L1 selection", "univariate_top4": "Univariate top-four",
          "greedy_forward4": "Greedy forward", "llm_zero_evaluation": "One-shot language model"}


def interval(row, signed=True):
    fmt = (lambda v: f"{v:+.3f}") if signed else (lambda v: f"{v:.3f}")
    return f"[{fmt(row.ci_low)}, {fmt(row.ci_high)}]"


def table(name, headers, rows, spec, caption, label, placement="t"):
    text = "\\begin{table}[" + placement + "]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
    text += "\\caption{" + caption + "}\\label{" + label + "}\n"
    text += "\\begin{tabular}{" + spec + "}\n\\toprule\n" + " & ".join(headers) + r" \\" + "\n\\midrule\n"
    text += "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows)
    text += "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (PAPER / (name + ".tex")).write_text(text)


def main():
    summary = pd.read_csv(OUT / "joint_summary.csv")
    macro = summary[summary.site.eq("macro")].set_index("method")
    contrasts = pd.read_csv(OUT / "joint_contrasts.csv")
    rows = []
    counts = {"full_agent_B10": "10", "l1_top4": "L1 path", "univariate_top4": "539",
              "greedy_forward4": "2,150", "llm_zero_evaluation": "0"}
    for method in LABELS:
        rows.append([LABELS[method], counts[method], f"{macro.loc[method, 'auc']:.3f}"])
    table("feature_selection_table", ["Method", "Selection evaluations", "Macro AUROC"], rows,
          "lrr", "Additional four-feature selectors. Selection evaluations count tested feature subsets before the final choice; they exclude probe tuning. L1 uses a regularization path. All results include S0001 development matches.", "tab:featureselection")

    rows = []
    for method in LABELS:
        values = summary[summary.method.eq(method)].set_index("site")
        row = [LABELS[method]] + [f"{values.loc[s, 'auc']:.3f}" for s in ["I0002", "I0006", "S0001", "macro"]]
        row.append(interval(values.loc["macro"], False))
        rows.append(row)
    table("supplement_feature_selection_table", ["Method", "I0002", "I0006", "S0001", "Macro", r"Macro 95\% interval"],
          rows, "lrrrrr", "Additional selectors and macro run/participant intervals. L1, univariate and greedy choices are deterministic on the observed sources, so their intervals resample participants only. All methods retain the original cohort and S0001 development matches.", "tab:featureselection_sites", "H")

    choices = pd.read_csv(OUT / "source_choices.csv")
    rows = []
    for row in choices[~choices.method.eq("llm_zero_evaluation")].itertuples():
        label = {"l1_top4": "L1", "univariate_top4": "Univariate", "greedy_forward4": "Greedy"}[row.method]
        rows.append([row.held_site, label, row.quartet_key.replace("|", ", "), f"{row.selected_C:.4g}"])
    table("supplement_baseline_choices_table", ["Test hospital", "Selector", "Selected feature IDs", "$C$"], rows,
          "lllr", "Source-selected feature sets and final-probe regularization. Target scores were computed after all choices were saved.", "tab:baselinechoices", "H")

    overlap = pd.read_csv(OUT / "one_shot_quartet_overlap.csv")
    rows = [[r.quartet_key.replace("|", ", "), r.n_calls, r.features_in_B10_selected_union,
             r.max_shared_features, f"{r.mean_jaccard:.3f}"] for r in overlap.itertuples()]
    table("supplement_one_shot_quartets_table",
          ["Quartet", "Calls", "Features in search union", "Maximum shared", "Mean Jaccard"], rows,
          "lrrrr", "Distinct one-shot quartets and overlap with the 30 original ten-evaluation selections. The search union contains 29 distinct features selected in those 30 runs; maximum shared is the largest intersection with one selected quartet. Mean Jaccard averages over the 30 comparisons. No exact quartet matches occurred.", "tab:oneshotquartets", "H")

    exclusion = pd.read_csv(OUT / "all_selector_exclusion.csv")
    labels_all = {"full_agent_B10": "Language model, ten evaluations",
        "random_source_robust_B10": "Random, ten evaluations", "evolutionary_robust_B10": "Evolutionary, ten evaluations",
        "taxonomy_diverse_robust_B10": "Taxonomy-diverse, ten evaluations",
        **{k: v for k, v in LABELS.items() if k != "full_agent_B10"}}
    rows = []
    exrows = []
    for key, label in labels_all.items():
        ex = exclusion[exclusion.method.eq(key)].set_index("site")
        full = summary[summary.method.eq(key)].set_index("site")
        values = [full.loc["macro", "auc"], ex.loc["I0002_I0006", "auc"],
                  full.loc["S0001", "auc"], ex.loc["S0001", "auc"]]
        rows.append([label] + [f"{value:.3f}" for value in values])
        exrows.append([label] + [f"{ex.loc[s, 'auc']:.3f}" for s in ["S0001", "macro", "I0002_I0006"]])
    table("evaluation_scope_table", ["Method", "Full macro", "I0002/I0006", "S0001", "S0001 retained"],
          rows, "lrrrr", "AUROC by method and test set. I0002/I0006 tests use S0001 among the sources; S0001 tests use I0002 and I0006. Development exclusions change only test participants, leaving seven S0001 participants and four eligible pairs. Selected features and source fits are unchanged.", "tab:scope")
    path = PAPER / "evaluation_scope_table.tex"
    path.write_text(path.read_text().replace("{table}", "{table*}"))
    table("supplement_exclusion_table", ["Method", "S0001 retained", "Retained macro", "I0002/I0006"],
          exrows, "lrrr", "AUROC after excluding development-list matches from test data. S0001 retains seven participants and four pairs. Source fits and selected features are unchanged. The two-hospital mean is a separate descriptive comparison.", "tab:exclusion", "H")

    comparison_names = {"random_source_robust_B10": "Random, ten evaluations",
        "evolutionary_robust_B10": "Evolutionary, ten evaluations",
        "taxonomy_diverse_robust_B10": "Taxonomy-diverse, ten evaluations",
        "random_source_robust_B500": "Random, 500 evaluations",
        **{k: v for k, v in LABELS.items() if k != "full_agent_B10"}}
    rows = [[r"\multicolumn{4}{l}{\textit{One-shot selection minus comparator}}"]]
    for key in ["l1_top4", "greedy_forward4", "univariate_top4"]:
        row = contrasts[contrasts.left.eq("llm_zero_evaluation") & contrasts.right.eq(key)].iloc[0]
        rows.append([LABELS[key], f"{row.difference:+.3f}", interval(row),
                     f"[{row.participant_ci_low:+.3f}, {row.participant_ci_high:+.3f}]"])
    rows.append([r"\multicolumn{4}{l}{\textit{Ten-evaluation search minus comparator}}"])
    for key, label in comparison_names.items():
        row = contrasts[contrasts.left.eq("full_agent_B10") & contrasts.right.eq(key)].iloc[0]
        rows.append([label, f"{row.difference:+.3f}", interval(row),
                     f"[{row.participant_ci_low:+.3f}, {row.participant_ci_high:+.3f}]"])
    table("supplement_joint_contrast_table", ["Comparator", r"$\Delta$", r"Run/participant 95\%", r"Participant-only 95\%"], rows,
          "lrrr", "One-shot selection and ten-evaluation language-model search minus each comparator. Both interval types use identical participant draws; joint intervals also resample runs. These are exploratory pointwise intervals.", "tab:jointcontrasts", "H")

    rows = []
    for left, right, label in [("feedback", "no_feedback", "Feedback minus no feedback"),
        ("feedback", "full_agent_B10", "Later feedback minus original search")]:
        row = contrasts[contrasts.left.eq(left) & contrasts.right.eq(right)].iloc[0]
        rows.append([label, f"{row.difference:+.3f}", interval(row)])
    table("supplement_joint_feedback_table", ["Comparison", r"$\Delta$", r"Run/participant 95\%"], rows,
          "lrr", "Paired continuation comparisons. Original and later searches share opening batches; they are not independent full reruns.", "tab:jointfeedback", "H")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    metrics = pd.read_csv(REV.parent / "analysis/output/final_selection_metrics.csv")
    selectors = {"full_agent": "robust", "random_source_robust": "source_robust",
        "random_source_mean": "source_mean", "evolutionary_robust": "robust", "taxonomy_diverse_robust": "robust"}
    labels = ["Language model", "Random (min.)", "Random (mean)", "Evolutionary", "Taxonomy-diverse"]
    colors = ["#2166ac", "#b2182b", "#ef8a62", "#555555", "#1b7837"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 4.1), gridspec_kw={"width_ratios": [1.05, 1.15]})
    for (method, selector), label, color in zip(selectors.items(), labels, colors):
        frame = metrics[metrics.method.eq(method) & metrics.selector.eq(selector)]
        means = frame.groupby(["budget", "held_site"]).target_auc.mean().groupby("budget").mean()
        means = means[means.index.isin([10, 25, 50, 500])]
        ax.plot(means.index, means.values, "o-", label=label, color=color, lw=1.5, ms=4)
    ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 500], [10, 25, 50, 500]); ax.set_ylim(.53, .68)
    ax.set(xlabel="Feature sets evaluated", ylabel="Macro age-conditioned AUROC")
    ax.set_title("(a) Search budget", loc="left", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", frameon=False, labelspacing=.3)
    keys = ["random_source_robust_B10", "evolutionary_robust_B10", "taxonomy_diverse_robust_B10",
            "l1_top4", "univariate_top4", "greedy_forward4", "llm_zero_evaluation"]
    focal = contrasts[contrasts.left.eq("full_agent_B10")].set_index("right").loc[keys]
    one_shot_keys = ["l1_top4", "greedy_forward4", "univariate_top4"]
    one_shot = contrasts[contrasts.left.eq("llm_zero_evaluation")].set_index("right").loc[one_shot_keys]
    bx.axvline(0, color="#777777", ls="--", lw=.8)
    for frame, positions, marker, color in [
        (one_shot, np.arange(1., 4.), "D", "#1b7837"),
        (focal, np.arange(5.2, 12.2), "o", "#2166ac")]:
        delta = frame.difference.to_numpy()
        errors = np.vstack([delta - frame.ci_low, frame.ci_high - delta])
        bx.errorbar(delta, positions, xerr=errors, fmt=marker, color=color, capsize=3, lw=1.3, ms=4)
    bx.set_yticks(np.r_[np.arange(1., 4.), np.arange(5.2, 12.2)],
        ["L1 selection", "Greedy", "Univariate", "Random", "Evolutionary", "Taxonomy-diverse",
         "L1 selection", "Univariate", "Greedy", "One-shot LLM"], fontsize=8)
    bx.text(.02, .1, "One-shot minus comparator", transform=bx.get_yaxis_transform(),
            fontsize=8, fontweight="bold", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=1))
    bx.text(.02, 4.3, "Search minus comparator", transform=bx.get_yaxis_transform(),
            fontsize=8, fontweight="bold", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=1))
    bx.set_ylim(11.9, -.65); bx.set_xlim(-.15, .20)
    bx.set_xticks([-.10, 0, .10, .20], ["-0.10", "0", "0.10", "0.20"])
    bx.set_xlabel("AUROC difference", fontsize=9)
    bx.set_title("(b) Selection differences", loc="left", fontsize=10, fontweight="bold")
    for panel in [ax, bx]:
        panel.tick_params(labelsize=8); panel.grid(axis="x", color="#eeeeee", lw=.5); panel.set_axisbelow(True)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(PAPER / "budget.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
