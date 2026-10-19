"""Per-site participant-level bootstrap intervals for the released study.

The participant bootstrap in ``participant_bootstrap.py`` already draws a
per-site distribution for every contrast and then reduces it to the macro mean.
This script replays that exact draw (same seed, same RNG stream, same 50,000
draws) and reports the per-site intervals instead, so the site-level numbers in
the paper come from the same resamples as the macro ones in Table 2.

It reuses the saved fixed-selection participant predictions, so no probe is
refitted and no target value is recomputed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_search.analysis.scripts.participant_bootstrap import (
    DRAWS,
    METHODS,
    OUT,
    SEED,
    SITES,
    _site_bootstrap_diffs,
    _target_frame,
    _wins,
)


def main() -> pd.DataFrame:
    metrics = pd.read_csv(OUT / "final_selection_metrics.csv")
    needed = metrics[
        ((metrics.method == "full_agent") & (metrics.selector == "robust")
         & metrics.budget.isin([10, 25, 50]))
        | ((metrics.method == "random_source_robust") & (metrics.selector == "source_robust")
           & metrics.budget.isin([10, 25, 50, 500]))
        | (metrics.method.isin(["evolutionary_robust", "taxonomy_diverse_robust"])
           & (metrics.selector == "robust") & metrics.budget.isin([10, 25, 50]))
    ].copy()

    predictions = pd.read_parquet(OUT / "participant_predictions.parquet")
    targets = {site: _target_frame(site) for site in SITES}

    candidate_wins: dict[tuple[str, str, float], np.ndarray] = {}
    for (site, key, c), part in predictions.groupby(
            ["held_site", "quartet_key", "selected_C"], sort=False):
        ordered = part.set_index("subject_id").loc[
            targets[str(site)].subject_id.astype(str)]
        win, _, _ = _wins(targets[str(site)], ordered.score.to_numpy(float))
        candidate_wins[(str(site), str(key), float(c))] = win

    # Mean pair-win surfaces = the method mean over its fixed selected runs.
    surfaces: dict[tuple[str, int, str], np.ndarray] = {}
    point_site: dict[tuple[str, int, str], float] = {}
    for method, (selector, _label) in METHODS.items():
        for budget in (10, 25, 50):
            group = needed[(needed.method == method) & (needed.selector == selector)
                           & (needed.budget == budget)]
            for site in SITES:
                g = group[group.held_site.astype(str).eq(site)]
                surfaces[(method, budget, site)] = np.mean(
                    [candidate_wins[(site, str(r.quartet_key), float(r.selected_C))]
                     for r in g.itertuples(index=False)], axis=0)
                point_site[(method, budget, site)] = float(g.target_auc.mean())
    for site in SITES:
        group = needed[(needed.method == "random_source_robust")
                       & (needed.selector == "source_robust") & (needed.budget == 500)
                       & needed.held_site.astype(str).eq(site)]
        surfaces[("random_source_robust", 500, site)] = np.mean(
            [candidate_wins[(site, str(r.quartet_key), float(r.selected_C))]
             for r in group.itertuples(index=False)], axis=0)
        point_site[("random_source_robust", 500, site)] = float(group.target_auc.mean())

    contrasts = [(b, c) for b in (10, 25, 50)
                 for c in ("random_source_robust", "evolutionary_robust",
                           "taxonomy_diverse_robust")]
    contrasts.append((10, "random_source_robust_B500"))
    names = [f"full_agent_B{b}_minus_{c}" for b, c in contrasts]

    # Replay the macro run's RNG stream site by site.
    rng = np.random.default_rng(SEED)
    site_draws: dict[str, dict[str, np.ndarray]] = {}
    for site in SITES:
        diffs = {}
        for (budget, comparator), name in zip(contrasts, names):
            cbudget = 500 if comparator.endswith("_B500") else budget
            cmethod = comparator.removesuffix("_B500")
            diffs[name] = (surfaces[("full_agent", budget, site)]
                           - surfaces[(cmethod, cbudget, site)])
        site_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        result = _site_bootstrap_diffs(diffs, _eligible(site, targets[site]), targets[site], site_rng)
        result.pop("_invalid_resamples")
        site_draws[site] = result

    rows = []
    for (budget, comparator), name in zip(contrasts, names):
        cbudget = 500 if comparator.endswith("_B500") else budget
        cmethod = comparator.removesuffix("_B500")
        row = {"comparison": name, "comparator": cmethod, "agent_budget": budget,
               "comparator_budget": cbudget}
        for site in SITES:
            draws = site_draws[site][name]
            point = point_site[("full_agent", budget, site)] - point_site[(cmethod, cbudget, site)]
            lo, hi = np.quantile(draws, [0.025, 0.975])
            p = min(1.0, (1.0 + float(np.sum(np.abs(draws - point) >= abs(point))))
                    / (DRAWS + 1.0))
            row[f"{site}_difference"] = float(point)
            row[f"{site}_ci_2_5"] = float(lo)
            row[f"{site}_ci_97_5"] = float(hi)
            row[f"{site}_p"] = float(p)
        row["bootstrap_draws"] = DRAWS
        row["bootstrap_seed"] = SEED
        row["design"] = ("paired participant bootstrap stratified by outcome within "
                         "held site; fixed selected quartets")
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "participant_bootstrap_per_site.csv", index=False)
    return out


def _eligible(site: str, target: pd.DataFrame) -> np.ndarray:
    y = target.outcome.to_numpy(int)
    ages = target.age.to_numpy(float)
    p, n = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return (np.abs(ages[p, None] - ages[None, n]) <= 2.0).astype(float)


if __name__ == "__main__":
    table = main()
    keep = ["comparison", "I0002_difference", "I0002_ci_2_5", "I0002_ci_97_5",
            "I0006_difference", "I0006_ci_2_5", "I0006_ci_97_5",
            "S0001_difference", "S0001_ci_2_5", "S0001_ci_97_5"]
    print(table[keep].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
