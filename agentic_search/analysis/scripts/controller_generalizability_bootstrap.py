"""Preregistered 15-contrast paired participant bootstrap for Luna/Opus."""
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


def _controllers() -> pd.DataFrame:
    rows = []
    root = PACKAGE / "runs/supplemental/generalizability"
    for model in ("luna", "opus"):
        for path in sorted((root / model / "pilot").glob("**/*.json")):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if state.get("supplemental_phase") != "main" or not state.get("agent_choice_valid"):
                continue
            budget = int(state["budget"])
            site = str(state["held_site_internal"])
            seed = int(state["seed"])
            history = state.get("history", [])
            choice = str(state["agent_choice_quartet_key"])
            match = next((r for r in history if str(r.get("real_quartet_key") or r["quartet_key"]) == choice), None)
            if match is None:
                match = next((r for r in history if "|".join(sorted(r.get("real_head_ids") or r["head_ids"])) == choice), None)
            if match is None:
                raise ValueError(f"Final choice is absent from source history: {path}")
            ids = sorted(map(str, state.get("agent_choice_quartet_real_head_ids") or choice.split("|")))
            rows.append({"model": model, "method": f"{model}_final_choice", "held_site": site,
                         "budget": budget, "seed": seed, "quartet_key": "|".join(ids),
                         "selected_C": float(match["selected_C"]), "selection_role": "primary_final_choice"})
    return pd.DataFrame(rows)


def _fixed_baselines() -> pd.DataFrame:
    source = pd.read_csv(OUT / "final_selection_metrics.csv")
    keep = source[
        ((source.method == "random_source_robust") & (source.selector == "source_robust") & source.budget.isin(BUDGETS))
        | ((source.method == "taxonomy_diverse_robust") & (source.selector == "robust") & source.budget.isin(BUDGETS))
    ].copy()
    keep["model"] = keep.method.map({"random_source_robust": "random", "taxonomy_diverse_robust": "taxonomy"})
    keep["selection_role"] = "fixed_robust_baseline"
    return keep[["model", "method", "held_site", "budget", "seed", "quartet_key", "selected_C", "target_auc", "selection_role"]]


def main() -> pd.DataFrame:
    models = _controllers()
    if set(models.model) != {"luna", "opus"} or set(models.budget) != set(BUDGETS):
        raise ValueError("Both controller final-choice grids must have at least one valid trajectory per budget")
    fixed = _fixed_baselines()
    all_choices = pd.concat([models, fixed], ignore_index=True, sort=False)
    cache_path = PACKAGE / "analysis/output/controller_generalizability_predictions.parquet"
    target_path = PACKAGE / "runs/supplemental/held_site/generalizability_target_results_v2.parquet"
    if not target_path.is_file():
        target_path = PACKAGE / "runs/supplemental/held_site/generalizability_target_results.parquet"
    targets = pd.read_parquet(target_path).set_index(["held_site", "quartet_key"])
    for row in models.itertuples(index=False):
        if (row.held_site, row.quartet_key) not in targets.index:
            raise ValueError(f"Controller selection absent from frozen scored manifest: {row.held_site}/{row.quartet_key}")
    models["target_auc"] = [float(targets.loc[(r.held_site, r.quartet_key)].target_auc)
                            for r in models.itertuples(index=False)]
    models = models[["model", "method", "held_site", "budget", "seed", "quartet_key",
                     "selected_C", "target_auc", "selection_role"]]
    all_choices = pd.concat([models, fixed], ignore_index=True, sort=False)

    cached = {}
    base_predictions = OUT / "participant_predictions.parquet"
    if base_predictions.exists():
        p = pd.read_parquet(base_predictions)
        for r in p.itertuples(index=False):
            cached[(str(r.held_site), str(r.quartet_key), float(r.selected_C), str(r.subject_id))] = float(r.score)
    unique = all_choices[["held_site", "quartet_key", "selected_C"]].drop_duplicates()
    missing_by_site: dict[str, dict] = {}
    for site in SITES:
        group = unique[unique.held_site.astype(str).eq(site)]
        missing_by_site[site] = {(str(r.quartet_key), float(r.selected_C)) for r in group.itertuples(index=False)
                                 if not any(k[:3] == (site, str(r.quartet_key), float(r.selected_C)) for k in cached)}
    score_cache = {}
    with threadpool_limits(limits=1):
        for site, pairs in missing_by_site.items():
            target = pb._target_frame(site)
            for key, c in sorted(pairs):
                result = pb._score_one(site, key, c, target)
                score_cache[(site, key, c)] = result["scores"]
    prediction_rows = []
    wins = {}
    eligibility = {}
    points = {}
    all_site_rows = []
    for site in SITES:
        target = pb._target_frame(site)
        y = target.outcome.to_numpy(int)
        ages = target.age.to_numpy(float)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        eligible = (np.abs(ages[pos, None] - ages[None, neg]) <= 2.0).astype(float)
        eligibility[site] = eligible
        for row in all_choices[all_choices.held_site.astype(str).eq(site)].itertuples(index=False):
            key = (site, str(row.quartet_key), float(row.selected_C))
            if (site, key[1], key[2]) in score_cache:
                scores = score_cache[(site, key[1], key[2])]
            else:
                scores = np.asarray([cached[(site, key[1], key[2], str(subject))]
                                     for subject in target.subject_id.astype(str)], float)
            win, _, _ = pb._wins(target, scores)
            wins[key] = win
            for subject, score in zip(target.subject_id.astype(str), scores):
                prediction_rows.append({"held_site": site, "subject_id": subject, "quartet_key": key[1],
                                        "selected_C": key[2], "score": float(score)})
        for budget in BUDGETS:
            for model in ("luna_final_choice", "opus_final_choice", "random", "taxonomy"):
                group = all_choices[(all_choices.model == model) & (all_choices.budget == budget) &
                                    all_choices.held_site.astype(str).eq(site)]
                mats = [wins[(site, str(r.quartet_key), float(r.selected_C))] for r in group.itertuples(index=False)]
                if not mats:
                    raise ValueError(f"No selections for {site}/{budget}/{model}")
                points[(site, budget, model)] = float(group.target_auc.mean())
                surface = np.mean(mats, axis=0)
                for contrast in ("luna_minus_random", "luna_minus_taxonomy", "opus_minus_random",
                                 "opus_minus_taxonomy", "luna_minus_opus"):
                    a, b = {
                        "luna_minus_random": ("luna_final_choice", "random"),
                        "luna_minus_taxonomy": ("luna_final_choice", "taxonomy"),
                        "opus_minus_random": ("opus_final_choice", "random"),
                        "opus_minus_taxonomy": ("opus_final_choice", "taxonomy"),
                        "luna_minus_opus": ("luna_final_choice", "opus_final_choice"),
                    }[contrast]
                    if model == a:
                        other = all_choices[(all_choices.model == b) & (all_choices.budget == budget) &
                                            all_choices.held_site.astype(str).eq(site)]
                        other_surface = np.mean([wins[(site, str(r.quartet_key), float(r.selected_C))]
                                                 for r in other.itertuples(index=False)], axis=0)
                        wins[(site, budget, contrast)] = surface - other_surface
                        all_site_rows.append({"held_site": site, "budget": budget, "contrast": contrast,
                                              "site_difference": points[(site, budget, a)] - points[(site, budget, b)]})
    # Participant resampling is common across all method pairs within each site.
    draws_by_site = {s: {} for s in SITES}
    invalid_by_site = {}
    for site in SITES:
        diff_surfaces = {(b, c): wins[(site, b, c)] for b in BUDGETS for c in
                         ("luna_minus_random", "luna_minus_taxonomy", "opus_minus_random",
                          "opus_minus_taxonomy", "luna_minus_opus")}
        named = {f"B{b}_{c}": v for (b, c), v in diff_surfaces.items()}
        rng = np.random.default_rng(SEED + SITES.index(site))
        result = pb._site_bootstrap_diffs(named, eligibility[site], pb._target_frame(site), rng)
        invalid_by_site[site] = int(result.pop("_invalid_resamples")[0])
        draws_by_site[site] = result
    contrast_names = ["luna_minus_random", "luna_minus_taxonomy", "opus_minus_random",
                      "opus_minus_taxonomy", "luna_minus_opus"]
    rows = []
    for budget in BUDGETS:
        for contrast in contrast_names:
            points_by_site = [next(x["site_difference"] for x in all_site_rows
                                   if x["held_site"] == site and x["budget"] == budget and x["contrast"] == contrast)
                              for site in SITES]
            point = float(np.mean(points_by_site))
            draws = np.mean(np.stack([draws_by_site[s][f"B{budget}_{contrast}"] for s in SITES]), axis=0)
            pvalue = min(1.0, (1 + float(np.sum(np.abs(draws - point) >= abs(point)))) / (DRAWS + 1))
            rows.append({"budget": budget, "contrast": contrast, "macro_difference": point,
                         "bootstrap_ci_2_5": float(np.quantile(draws, .025)),
                         "bootstrap_ci_97_5": float(np.quantile(draws, .975)),
                         "null_centered_two_sided_p": pvalue, "draws": DRAWS,
                         "luna_n": int(len(models[(models.model == "luna") & (models.budget == budget)])),
                         "opus_n": int(len(models[(models.model == "opus") & (models.budget == budget)])),
                         "invalid_resamples": int(sum(invalid_by_site.values()))})
    pvalues = np.array([r["null_centered_two_sided_p"] for r in rows])
    order = np.argsort(pvalues)
    adjusted = np.ones(len(rows))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * pvalues[index]))
        adjusted[index] = running
    for i, row in enumerate(rows):
        row["holm_adjusted_p_15_tests"] = float(adjusted[i])
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "controller_generalizability_participant_bootstrap.csv", index=False, float_format="%.8f")
    pd.DataFrame(prediction_rows).drop_duplicates().to_parquet(cache_path, index=False)
    models.to_csv(OUT / "controller_generalizability_final_choices.csv", index=False)
    pd.DataFrame(all_site_rows).to_csv(OUT / "controller_generalizability_site_differences.csv", index=False)
    return result


if __name__ == "__main__":
    print(main().to_string(index=False, float_format=lambda x: f"{x:.5f}"))
