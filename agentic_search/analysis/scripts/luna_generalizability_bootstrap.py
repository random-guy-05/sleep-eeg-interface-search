"""Participant bootstrap for the completed Luna-only controller extension.

The Opus controller run was stopped by the user. This module deliberately
computes only the six prespecified Luna-vs-baseline contrasts and labels their
Holm correction as a partial family; it does not impute or infer Opus results.
Selected quartets stay fixed while participants are resampled within site and
outcome, matching the committed protocol's participant-level estimand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from agentic_search.analysis.scripts import participant_bootstrap as pb

PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / "analysis/output"
SITES = ("I0002", "I0006", "S0001")
BUDGETS = (10, 25, 50)
DRAWS = 50_000
SEED = 20260927


def _luna_choices() -> pd.DataFrame:
    rows = []
    root = PACKAGE / "runs/supplemental/generalizability/luna/pilot/codex"
    for path in sorted(root.glob("**/*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if state.get("supplemental_phase") != "main" or state.get("agent_choice_valid") is not True:
            continue
        site = str(state["held_site_internal"])
        budget = int(state["budget"])
        seed = int(state["seed"])
        ids = sorted(map(str, state.get("agent_choice_quartet_real_head_ids", [])))
        key = "|".join(ids)
        match = next((r for r in state.get("history", [])
                      if str(r.get("real_quartet_key") or
                             "|".join(sorted(map(str, r.get("real_head_ids") or r["head_ids"])))) == key), None)
        if match is None:
            raise ValueError(f"Luna final choice is absent from source history: {path}")
        rows.append({"model": "gpt-6-luna", "method": "luna_final_choice", "held_site": site,
                     "budget": budget, "seed": seed, "quartet_key": key,
                     "selected_C": float(match["selected_C"]),
                     "selection_role": "primary_final_choice"})
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(["held_site", "budget", "seed"]).any():
        raise ValueError("Luna final-choice grid is empty or has duplicate cells")
    return frame


def _luna_robust_choices() -> pd.DataFrame:
    rows = []
    root = PACKAGE / "runs/supplemental/generalizability/luna/pilot/codex"
    for path in sorted(root.glob("**/*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (int(state.get("seed", -1)) not in range(1000, 1010) or
                int(state.get("budget", 0)) not in BUDGETS):
            continue
        site = str(state["held_site_internal"])
        budget = int(state["budget"])
        seed = int(state["seed"])
        key = str(state.get("robust_selected_quartet_key", ""))
        match = next((r for r in state.get("history", [])
                      if str(r.get("quartet_key")) == key or
                      str(r.get("real_quartet_key")) == key), None)
        if not key or match is None:
            raise ValueError(f"Luna robust source selection is absent from source history: {path}")
        ids = sorted(map(str, match.get("real_head_ids") or match["head_ids"]))
        rows.append({"model": "gpt-6-luna", "method": "luna_source_robust", "held_site": site,
                     "budget": budget, "seed": seed, "quartet_key": "|".join(ids),
                     "selected_C": float(match["selected_C"]),
                     "selection_role": "secondary_source_robust_selector"})
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(["held_site", "budget", "seed"]).any():
        raise ValueError("Luna robust selector grid is empty or has duplicate cells")
    return frame


def _fixed_baselines() -> pd.DataFrame:
    source = pd.read_csv(OUT / "final_selection_metrics.csv")
    keep = source[
        ((source.method == "random_source_robust") &
         (source.selector == "source_robust") & source.budget.isin(BUDGETS))
        | ((source.method == "taxonomy_diverse_robust") &
           (source.selector == "robust") & source.budget.isin(BUDGETS))
    ].copy()
    keep["model"] = keep.method.map({"random_source_robust": "random",
                                     "taxonomy_diverse_robust": "taxonomy"})
    keep["selection_role"] = "fixed_robust_baseline"
    return keep[["model", "method", "held_site", "budget", "seed", "quartet_key",
                 "selected_C", "target_auc", "selection_role"]]


def _holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.ones(len(pvalues), float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * float(pvalues[index])))
        adjusted[index] = running
    return adjusted.tolist()


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    luna = _luna_choices()
    if set(luna.budget.astype(int)) != set(BUDGETS):
        raise ValueError("Luna has no valid selected choices at one or more budgets")
    fixed = _fixed_baselines()
    luna_scores = pd.read_csv(OUT / "luna_final_choice_scored.csv")
    luna = luna.merge(luna_scores[["held_site", "budget", "seed", "quartet_key", "target_auc"]],
                      on=["held_site", "budget", "seed", "quartet_key"], how="left", validate="one_to_one")
    if luna.target_auc.isna().any():
        raise ValueError("A valid Luna final choice is missing from its held-site score table")
    robust = _luna_robust_choices()
    target_release = pd.read_parquet(PACKAGE / "runs/supplemental/held_site/generalizability_target_results.parquet")
    target_index = target_release.set_index(["held_site", "quartet_key"])
    robust["target_auc"] = [float(target_index.loc[(r.held_site, r.quartet_key)].target_auc)
                             for r in robust.itertuples(index=False)]
    if not set(robust.budget.astype(int)) == set(BUDGETS):
        raise ValueError("Luna robust selector grid is missing a budget")
    choices = pd.concat([luna, robust, fixed], ignore_index=True, sort=False)
    for model in ("random", "taxonomy"):
        for budget in BUDGETS:
            for site in SITES:
                n = len(fixed[(fixed.model == model) & (fixed.budget == budget) &
                              fixed.held_site.astype(str).eq(site)])
                if n == 0:
                    raise ValueError(f"Missing fixed baseline: {model}/{site}/B{budget}")

    # Use cached primary participant predictions when an exact site/key/C match
    # exists; recompute and validate any Luna quartet absent from that cache.
    cached: dict[tuple[str, str, float, str], float] = {}
    cache_path = OUT / "participant_predictions.parquet"
    if cache_path.exists():
        for r in pd.read_parquet(cache_path).itertuples(index=False):
            cached[(str(r.held_site), str(r.quartet_key), float(r.selected_C),
                    str(r.subject_id))] = float(r.score)
    target_frames = {site: pb._target_frame(site) for site in SITES}
    unique = choices[["held_site", "quartet_key", "selected_C"]].drop_duplicates()
    recomputed: dict[tuple[str, str, float], np.ndarray] = {}
    with threadpool_limits(limits=1):
        for site in SITES:
            target = target_frames[site]
            rows = unique[unique.held_site.astype(str).eq(site)]
            for r in rows.itertuples(index=False):
                key = (site, str(r.quartet_key), float(r.selected_C))
                if not all((*key, str(subject)) in cached for subject in target.subject_id.astype(str)):
                    recomputed[key] = pb._score_one(site, key[1], key[2], target)["scores"]

    prediction_rows = []
    for site in SITES:
        target = target_frames[site]
        for r in choices[choices.held_site.astype(str).eq(site)].itertuples(index=False):
            key = (site, str(r.quartet_key), float(r.selected_C))
            scores = recomputed.get(key)
            if scores is None:
                scores = np.asarray([cached[(*key, str(subject))]
                                     for subject in target.subject_id.astype(str)], float)
            for subject, score in zip(target.subject_id.astype(str), scores):
                prediction_rows.append({"held_site": site, "subject_id": subject,
                                        "quartet_key": key[1], "selected_C": key[2],
                                        "score": float(score)})

    def get_scores(site: str, key: str, c: float) -> np.ndarray:
        cache_key = (site, key, float(c))
        if cache_key in recomputed:
            return recomputed[cache_key]
        target = target_frames[site]
        return np.asarray([cached[(*cache_key, str(subject))]
                           for subject in target.subject_id.astype(str)], float)

    wins: dict[tuple[str, str, float], np.ndarray] = {}
    for r in choices.itertuples(index=False):
        key = (str(r.held_site), str(r.quartet_key), float(r.selected_C))
        if key in wins:
            continue
        win, _, _ = pb._wins(target_frames[key[0]], get_scores(*key))
        wins[key] = win

    eligible: dict[str, np.ndarray] = {}
    pair_rows = []
    for site in SITES:
        target = target_frames[site]
        y = target.outcome.to_numpy(int)
        ages = target.age.to_numpy(float)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        e = (np.abs(ages[pos, None] - ages[None, neg]) <= 2.0).astype(float)
        eligible[site] = e
        pair_rows.append({"held_site": site, "n_participants": len(target),
                          "n_positive": int((y == 1).sum()), "n_negative": int((y == 0).sum()),
                          "age_matched_positive_negative_pairs": int(e.sum())})

    contrasts = [(budget, baseline) for budget in BUDGETS for baseline in ("random", "taxonomy")]
    site_differences: dict[tuple[str, int, str], float] = {}
    site_draws: dict[str, dict[str, np.ndarray]] = {site: {} for site in SITES}
    invalid_draws = {}
    site_points_rows = []
    names = { (budget, baseline): f"luna_B{budget}_minus_{baseline}" for budget, baseline in contrasts }
    for site in SITES:
        local_diffs = {}
        for budget, baseline in contrasts:
            luna_rows = luna[(luna.budget == budget) & luna.held_site.astype(str).eq(site)]
            base_rows = fixed[(fixed.model == baseline) & (fixed.budget == budget) &
                              fixed.held_site.astype(str).eq(site)]
            luna_surface = np.mean([wins[(site, str(r.quartet_key), float(r.selected_C))]
                                    for r in luna_rows.itertuples(index=False)], axis=0)
            base_surface = np.mean([wins[(site, str(r.quartet_key), float(r.selected_C))]
                                    for r in base_rows.itertuples(index=False)], axis=0)
            diff = luna_surface - base_surface
            name = names[(budget, baseline)]
            local_diffs[name] = diff
            point = float(diff.sum() / eligible[site].sum())
            # The paired AUROC is normalized by the number of eligible
            # positive-negative pairs, including zero-win ineligible cells.
            site_differences[(site, budget, baseline)] = point
            site_points_rows.append({"held_site": site, "budget": budget, "comparator": baseline,
                                     "luna_n": int(len(luna_rows)), "comparator_n": int(len(base_rows)),
                                     "luna_macro_auc": float(luna_rows.target_auc.mean()),
                                     "site_difference": point})
        rng = np.random.default_rng(SEED + SITES.index(site))
        draw_result = pb._site_bootstrap_diffs(local_diffs, eligible[site], target_frames[site], rng)
        invalid_draws[site] = int(draw_result.pop("_invalid_resamples")[0])
        site_draws[site] = draw_result

    summary_rows = []
    for budget, baseline in contrasts:
        name = names[(budget, baseline)]
        site_points = [site_differences[(site, budget, baseline)] for site in SITES]
        point = float(np.mean(site_points))
        draws = np.mean(np.stack([site_draws[s][name] for s in SITES]), axis=0)
        pvalue = min(1.0, (1 + float(np.sum(np.abs(draws - point) >= abs(point)))) / (DRAWS + 1))
        ci = np.quantile(draws, [.025, .975])
        summary_rows.append({"budget": budget, "contrast": name, "comparator": baseline,
                             "macro_difference": point, "bootstrap_ci_2_5": float(ci[0]),
                             "bootstrap_ci_97_5": float(ci[1]),
                             "null_centered_two_sided_p": pvalue, "draws": DRAWS,
                             "luna_n": int(len(luna[luna.budget == budget])),
                             "invalid_resamples": int(sum(invalid_draws.values())),
                             "site_differences": json.dumps({s: site_differences[(s, budget, baseline)]
                                                            for s in SITES}, sort_keys=True),
                             "holm_adjusted_p_six_luna_tests": np.nan})
    adjusted = _holm([r["null_centered_two_sided_p"] for r in summary_rows])
    for row, p_adj in zip(summary_rows, adjusted):
        row["holm_adjusted_p_six_luna_tests"] = p_adj

    # Absolute participant-bootstrap percentile intervals for Luna's Figure 1
    # row. These use the same fixed-choice, within-site outcome-stratified
    # resampling design as the paired contrasts.
    luna_ci_rows = []
    absolute_draws: dict[tuple[str, int], np.ndarray] = {}
    for site in SITES:
        local = {}
        for budget in BUDGETS:
            local_rows = luna[(luna.budget == budget) & luna.held_site.astype(str).eq(site)]
            surface = np.mean([wins[(site, str(r.quartet_key), float(r.selected_C))]
                               for r in local_rows.itertuples(index=False)], axis=0)
            local[f"luna_B{budget}"] = surface
        rng = np.random.default_rng(SEED + 100 + SITES.index(site))
        draw_result = pb._site_bootstrap_diffs(local, eligible[site], target_frames[site], rng)
        for budget in BUDGETS:
            absolute_draws[(site, budget)] = draw_result[f"luna_B{budget}"]
    for budget in BUDGETS:
        per_site = [float(luna[(luna.budget == budget) & luna.held_site.astype(str).eq(site)].target_auc.mean())
                    for site in SITES]
        draws = np.mean(np.stack([absolute_draws[(site, budget)] for site in SITES]), axis=0)
        ci = np.quantile(draws, [.025, .975])
        luna_ci_rows.append({"budget": budget, "n_valid_choices": int(len(luna[luna.budget == budget])),
                             "macro_target_auc": float(np.mean(per_site)),
                             "bootstrap_ci_2_5": float(ci[0]), "bootstrap_ci_97_5": float(ci[1]),
                             "draws": DRAWS, "invalid_resamples": int(sum(invalid_draws.values())),
                             "design": "paired participant bootstrap stratified by outcome within site; selected quartets fixed"})

    robust_ci_rows = []
    robust_site_rows = []
    for site in SITES:
        local = {}
        for budget in BUDGETS:
            local_rows = robust[(robust.budget == budget) & robust.held_site.astype(str).eq(site)]
            surface = np.mean([wins[(site, str(r.quartet_key), float(r.selected_C))]
                               for r in local_rows.itertuples(index=False)], axis=0)
            local[f"robust_B{budget}"] = surface
            robust_site_rows.append({"held_site": site, "budget": budget, "n": int(len(local_rows)),
                                     "mean_target_auc": float(local_rows.target_auc.mean()),
                                     "quartet_count": int(local_rows.quartet_key.nunique())})
        rng = np.random.default_rng(SEED + 200 + SITES.index(site))
        draw_result = pb._site_bootstrap_diffs(local, eligible[site], target_frames[site], rng)
        for budget in BUDGETS:
            absolute_draws[(site, -budget)] = draw_result[f"robust_B{budget}"]
    for budget in BUDGETS:
        per_site = [float(robust[(robust.budget == budget) & robust.held_site.astype(str).eq(site)].target_auc.mean())
                    for site in SITES]
        draws = np.mean(np.stack([absolute_draws[(site, -budget)] for site in SITES]), axis=0)
        ci = np.quantile(draws, [.025, .975])
        robust_ci_rows.append({"budget": budget, "n_valid_choices": int(len(robust[robust.budget == budget])),
                               "macro_target_auc": float(np.mean(per_site)),
                               "bootstrap_ci_2_5": float(ci[0]), "bootstrap_ci_97_5": float(ci[1]),
                               "draws": DRAWS,
                               "design": "participant bootstrap stratified by outcome within site; robust choices fixed"})

    out = pd.DataFrame(summary_rows)
    out.to_csv(OUT / "luna_generalizability_participant_bootstrap.csv", index=False, float_format="%.8f")
    sites = pd.DataFrame(site_points_rows)
    sites.to_csv(OUT / "luna_generalizability_site_differences.csv", index=False, float_format="%.8f")
    pd.DataFrame(pair_rows).to_csv(OUT / "luna_generalizability_age_matched_pairs.csv", index=False)
    choices.to_csv(OUT / "luna_generalizability_choices_for_bootstrap.csv", index=False)
    pd.DataFrame(luna_ci_rows).to_csv(OUT / "luna_generalizability_macro_bootstrap.csv", index=False,
                                     float_format="%.8f")
    robust.to_csv(OUT / "luna_generalizability_robust_choices.csv", index=False, float_format="%.8f")
    pd.DataFrame(robust_ci_rows).to_csv(OUT / "luna_generalizability_robust_macro_bootstrap.csv", index=False,
                                       float_format="%.8f")
    pd.DataFrame(robust_site_rows).to_csv(OUT / "luna_generalizability_robust_site_summary.csv", index=False,
                                         float_format="%.8f")
    pd.DataFrame(prediction_rows).drop_duplicates().to_parquet(
        OUT / "luna_generalizability_participant_predictions.parquet", index=False)
    return out, sites, pd.DataFrame(pair_rows)


if __name__ == "__main__":
    summary, site, pairs = main()
    print("Age-matched participant pairs:\n", pairs.to_string(index=False))
    print("Luna paired participant bootstrap (six-test partial family):\n",
          summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("Site differences:\n", site.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
