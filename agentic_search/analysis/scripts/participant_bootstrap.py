"""Participant-level bootstrap for fixed selected quartets in the released study.

The bootstrap resamples participants within site and outcome, retaining all
selected quartets and source-fitted scores. Thus it quantifies participant
sampling uncertainty conditional on the observed search selections.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from brainbodyfm.metrics import age_conditioned_auroc
from brainbodyfm.models import ModelSpec, fit_probe
from agentic_search.evaluator.source_evaluator import C_GRID


PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / "analysis/output"
SITES = ("I0002", "I0006", "S0001")
METHODS = {
    "full_agent": ("robust", "Full agent, GPT-6 Sol"),
    "random_source_robust": ("source_robust", "Random + source robust"),
    "evolutionary_robust": ("robust", "Evolutionary robust"),
    "taxonomy_diverse_robust": ("robust", "Taxonomy-diverse robust"),
}
DRAWS = 50_000
SEED = 20260927


@lru_cache(maxsize=3)
def _target_frame(site: str) -> pd.DataFrame:
    labels = pd.read_parquet(PACKAGE / "sealed/target_labels.parquet")
    features = pd.read_parquet(PACKAGE / "data/participants.parquet")
    labels = labels[labels.held_site.astype(str).eq(site)].copy()
    features = features[features.site.astype(str).eq(site)].copy()
    frame = labels.merge(features, on=["subject_id", "site", "age"], validate="one_to_one")
    if len(frame) != len(labels):
        raise ValueError(f"Target labels/features do not align for {site}")
    return frame.sort_values("subject_id").reset_index(drop=True)


def _wins(target: pd.DataFrame, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    y = target.outcome.to_numpy(int)
    ages = target.age.to_numpy(float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    eligible = np.abs(ages[pos, None] - ages[None, neg]) <= 2.0
    differences = scores[pos, None] - scores[None, neg]
    win = (differences > 0).astype(float) + 0.5 * (differences == 0)
    win *= eligible
    return win, eligible.astype(float), int(eligible.sum())


@lru_cache(maxsize=1)
def _catalog() -> pd.DataFrame:
    return pd.read_csv(PACKAGE / "data/heads.csv", dtype=str).set_index("head_id")


@lru_cache(maxsize=3)
def _source(site: str) -> pd.DataFrame:
    return pd.read_parquet(PACKAGE / f"data/source_folds/{site}.parquet")


def _score_one(site: str, key: str, c: float, target: pd.DataFrame) -> dict:
    ids = key.split("|")
    catalog = _catalog()
    names = catalog.loc[ids, "head_name"].astype(str).tolist()
    source = _source(site)
    spec = ModelSpec(name="prospective_random4_l2", penalty="l2", c_grid=C_GRID,
                     max_iter=10_000, tol=1e-5, random_state=2026090401)
    source_ids = source.subject_id.astype(str).to_numpy()
    target_ids = target.subject_id.astype(str).to_numpy()
    if set(source_ids).intersection(target_ids):
        raise ValueError(f"Source/target subject overlap for {site}")
    model = fit_probe(source[names].apply(pd.to_numeric, errors="coerce").to_numpy(float),
                      source.outcome.to_numpy(int), spec=spec, C=float(c), feature_names=names,
                      source_subject_ids=source_ids, target_subject_ids=target_ids)
    audit = model.preprocessor.audit_
    if audit is None or audit.target_rows_used:
        raise AssertionError("Target participants entered source preprocessing")
    prediction = model.decision_function(target[names].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    auc = age_conditioned_auroc(target.outcome.to_numpy(int), prediction,
                                target.age.to_numpy(float), age_window=2.0)
    return {"held_site": site, "quartet_key": key, "selected_C": float(c),
            "scores": np.asarray(prediction, float), "target_auc_recomputed": float(auc)}


def _holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.ones(len(pvalues), float)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(pvalues[index])))
        adjusted[index] = running
    return adjusted.tolist()


def _site_bootstrap_diffs(differences: dict[str, np.ndarray], eligible: np.ndarray,
                          target: pd.DataFrame, rng: np.random.Generator) -> dict[str, np.ndarray]:
    y = target.outcome.to_numpy(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    # Both ``eligible`` and every comparison surface are already in positive × negative order.
    e = eligible
    denominator = int(e.sum())
    if denominator == 0:
        raise ValueError("No age-matched positive-negative pair in site")
    output = {name: np.empty(DRAWS, float) for name in differences}
    invalid = 0
    cursor = 0
    while cursor < DRAWS:
        size = min(1000, DRAWS - cursor)
        pos_ix = rng.integers(0, len(pos), size=(size, len(pos)))
        neg_ix = rng.integers(0, len(neg), size=(size, len(neg)))
        elig_draw = e[pos_ix[:, :, None], neg_ix[:, None, :]]
        denom_draw = elig_draw.sum(axis=(1, 2))
        good = denom_draw > 0
        invalid += int((~good).sum())
        take = int(good.sum())
        if take == 0:
            continue
        for name, diff in differences.items():
            wins_draw = diff[pos_ix[:, :, None], neg_ix[:, None, :]].sum(axis=(1, 2))
            output[name][cursor:cursor + take] = wins_draw[good] / denom_draw[good]
        cursor += take
    for array in output.values():
        array[:] = (array - 0.0)  # retain draws unchanged; denominator is per draw above
    return {**output, "_invalid_resamples": np.asarray([invalid], int)}


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(OUT / "final_selection_metrics.csv")
    needed = metrics[
        ((metrics.method == "full_agent") & (metrics.selector == "robust") & metrics.budget.isin([10, 25, 50]))
        | ((metrics.method == "random_source_robust") & (metrics.selector == "source_robust") & metrics.budget.isin([10, 25, 50, 500]))
        | (metrics.method.isin(["evolutionary_robust", "taxonomy_diverse_robust"])
           & (metrics.selector == "robust") & metrics.budget.isin([10, 25, 50]))
    ].copy()
    keys = needed[["held_site", "quartet_key", "selected_C"]].drop_duplicates().sort_values(["held_site", "quartet_key"])
    targets = {site: _target_frame(site) for site in SITES}

    predictions: dict[tuple[str, str, float], dict] = {}
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_score_one, str(row.held_site), str(row.quartet_key), float(row.selected_C),
                               targets[str(row.held_site)]): (str(row.held_site), str(row.quartet_key), float(row.selected_C))
                   for row in keys.itertuples(index=False)}
        for future in as_completed(futures):
            key = futures[future]
            result = future.result()
            predictions[key] = result
            if len(predictions) % 25 == 0:
                print(f"Recomputed fixed target scores: {len(predictions)}/{len(keys)}", flush=True)

    pred_rows = []
    candidate_wins: dict[tuple[str, str, float], np.ndarray] = {}
    site_elig: dict[str, np.ndarray] = {}
    site_pair_counts = []
    for site in SITES:
        target = targets[site]
        y = target.outcome.to_numpy(int)
        ages = target.age.to_numpy(float)
        p, n = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        e = (np.abs(ages[p, None] - ages[None, n]) <= 2.0).astype(float)
        site_elig[site] = e
        site_pair_counts.append({"held_site": site, "n_participants": len(target),
                                 "n_positive": int((y == 1).sum()), "n_negative": int((y == 0).sum()),
                                 "age_matched_positive_negative_pairs": int(e.sum())})
    observed = {(str(r.held_site), str(r.quartet_key), float(r.selected_C)): float(r.target_auc)
                for r in needed.itertuples(index=False)}
    for key, result in predictions.items():
        target = targets[key[0]]
        auc = result["target_auc_recomputed"]
        if not np.isclose(auc, observed[key], rtol=0, atol=1e-12):
            raise ValueError(f"Recomputed participant predictions fail evaluator check: {key}: {auc} vs {observed[key]}")
        win, _, _ = _wins(target, result["scores"])
        candidate_wins[key] = win
        for subj, score in zip(target.subject_id.astype(str), result["scores"]):
            pred_rows.append({"held_site": key[0], "subject_id": subj, "quartet_key": key[1],
                              "selected_C": key[2], "score": float(score)})

    # Mean pair-win surfaces correspond to the method mean over its fixed selected runs.
    surfaces: dict[tuple[str, int, str], np.ndarray] = {}
    point_site: dict[tuple[str, int, str], float] = {}
    for (method, (selector, _label)) in METHODS.items():
        for budget in (10, 25, 50):
            group = needed[(needed.method == method) & (needed.selector == selector) & (needed.budget == budget)]
            for site in SITES:
                g = group[group.held_site.astype(str).eq(site)]
                mats = [candidate_wins[(site, str(r.quartet_key), float(r.selected_C))]
                        for r in g.itertuples(index=False)]
                surfaces[(method, budget, site)] = np.mean(mats, axis=0)
                point_site[(method, budget, site)] = float(np.mean(g.target_auc))
    for site in SITES:
        group = needed[(needed.method == "random_source_robust") & (needed.selector == "source_robust")
                       & (needed.budget == 500) & needed.held_site.astype(str).eq(site)]
        mats = [candidate_wins[(site, str(r.quartet_key), float(r.selected_C))]
                for r in group.itertuples(index=False)]
        surfaces[("random_source_robust", 500, site)] = np.mean(mats, axis=0)
        point_site[("random_source_robust", 500, site)] = float(group.target_auc.mean())

    contrasts = []
    for budget in (10, 25, 50):
        for comparator in ("random_source_robust", "evolutionary_robust", "taxonomy_diverse_robust"):
            contrasts.append((budget, comparator))
    contrasts.append((10, "random_source_robust_B500"))
    names = [f"full_agent_B{b}_minus_{c}" for b, c in contrasts]
    rng = np.random.default_rng(SEED)
    site_draws: dict[str, dict[str, np.ndarray]] = {site: {} for site in SITES}
    bootstrap_invalid = {}
    for site in SITES:
        diffs = {}
        target = targets[site]
        for (budget, comparator), name in zip(contrasts, names):
            cbudget = 500 if comparator.endswith("_B500") else budget
            cmethod = comparator.removesuffix("_B500")
            diffs[name] = (surfaces[("full_agent", budget, site)] - surfaces[(cmethod, cbudget, site)])
        site_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        draw_result = _site_bootstrap_diffs(diffs, site_elig[site], target, site_rng)
        bootstrap_invalid[site] = int(draw_result.pop("_invalid_resamples")[0])
        site_draws[site] = draw_result

    summary_rows = []
    tax_raw = []
    for ix, ((budget, comparator), name) in enumerate(zip(contrasts, names)):
        site_points = []
        for site in SITES:
            cm = comparator.removesuffix("_B500")
            cb = 500 if comparator.endswith("_B500") else budget
            site_points.append(point_site[("full_agent", budget, site)] - point_site[(cm, cb, site)])
        macro_point = float(np.mean(site_points))
        draws = np.mean(np.stack([site_draws[s][name] for s in SITES]), axis=0)
        # Null-centered paired-bootstrap test: shift the bootstrap distribution
        # by the observed effect, then compare a zero-null draw to |theta_hat|.
        p = min(1.0, (1.0 + float(np.sum(np.abs(draws - macro_point) >= abs(macro_point)))) /
                (DRAWS + 1.0))
        ci = np.quantile(draws, [0.025, 0.975])
        summary_rows.append({"comparison": name, "budget": budget, "macro_difference": macro_point,
                             "bootstrap_ci_2_5": float(ci[0]), "bootstrap_ci_97_5": float(ci[1]),
                             "bootstrap_two_sided_p": p, "bootstrap_draws": DRAWS, "bootstrap_seed": SEED,
                             "bootstrap_invalid_site_draws": sum(bootstrap_invalid.values()),
                             "design": "paired participant bootstrap stratified by outcome within held site; fixed selected quartets"})
        if comparator == "taxonomy_diverse_robust":
            tax_raw.append(p)
    tax_adjusted = _holm(tax_raw)
    tax_lookup = {10: tax_adjusted[0], 25: tax_adjusted[1], 50: tax_adjusted[2]}
    for row in summary_rows:
        if "taxonomy_diverse_robust" in row["comparison"]:
            row["holm_adjusted_p_taxonomy_budgets"] = tax_lookup[row["budget"]]
        else:
            row["holm_adjusted_p_taxonomy_budgets"] = np.nan
    for row in summary_rows:
        budget = row["budget"]
        comp = "random_source_robust" if "random_source_robust_B500" in row["comparison"] else next(
            (c for c in ("random_source_robust", "evolutionary_robust", "taxonomy_diverse_robust")
             if f"_minus_{c}" in row["comparison"]), None)
        row["full_agent_macro_mean"] = float(np.mean([point_site[("full_agent", budget, s)] for s in SITES]))
        if comp is not None:
            cb = 500 if "B500" in row["comparison"] else budget
            row["comparator_macro_mean"] = float(np.mean([point_site[(comp, cb, s)] for s in SITES]))
        row["site_differences"] = json.dumps({s: point_site[("full_agent", budget, s)] - point_site[(
            "random_source_robust" if "random_source_robust_B500" in row["comparison"] else comp,
            500 if "B500" in row["comparison"] else budget, s)] for s in SITES}, sort_keys=True)

    summary = pd.DataFrame(summary_rows)
    site_rows = []
    for budget in (10, 25, 50):
        for method, (selector, label) in METHODS.items():
            group = needed[(needed.method == method) & (needed.selector == selector) & (needed.budget == budget)]
            for site in SITES:
                g = group[group.held_site.astype(str).eq(site)]
                site_rows.append({"method": method, "method_label": label, "budget": budget,
                                  "held_site": site, "n_fixed_selections": int(len(g)),
                                  "mean_target_auc": float(g.target_auc.mean()),
                                  "quartet_count": int(g.quartet_key.nunique())})
    g = needed[(needed.method == "random_source_robust") & (needed.budget == 500)]
    for site, part in g.groupby("held_site"):
        site_rows.append({"method": "random_source_robust", "method_label": "Random + source robust",
                          "budget": 500, "held_site": site, "n_fixed_selections": len(part),
                          "mean_target_auc": part.target_auc.mean(), "quartet_count": part.quartet_key.nunique()})
    site_table = pd.DataFrame(site_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_parquet(OUT / "participant_predictions.parquet", index=False)
    pd.DataFrame(site_pair_counts).to_csv(OUT / "participant_age_matched_pairs.csv", index=False)
    summary.to_csv(OUT / "participant_bootstrap_contrasts.csv", index=False)
    site_table.to_csv(OUT / "participant_bootstrap_site_means.csv", index=False)
    return summary, site_table, pd.DataFrame(site_pair_counts)


if __name__ == "__main__":
    s, sites, pairs = main()
    print("Age-matched pairs:\n", pairs.to_string(index=False))
    print("Participant bootstrap contrasts:\n", s.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("Per-site fixed-selection means:\n", sites.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
