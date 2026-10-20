"""Run/participant bootstrap for the observed search procedures and new baselines.

Participants and runs are crossed: one participant resample is shared by all
runs/methods at a hospital. Source data/models are not refitted in the bootstrap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from agentic_search.ieee_access_revision.scripts.statistical_analysis import pair_surface
from agentic_search.analysis.scripts.participant_bootstrap import _target_frame

PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / "ieee_access_revision/results/feature_selection_extension"
SITES = ("I0002", "I0006", "S0001")
DRAWS = 50000
SEED = 2026101907


def ci(x):
    return dict(ci_low=float(np.quantile(x, .025)), ci_high=float(np.quantile(x, .975)))


def bootstrap_site(surfaces, eligible, rng, draws=DRAWS, groups=None, shared_weights=None):
    """surfaces[method] is run x positive x negative; keep every method paired."""
    if not eligible.sum():
        raise ValueError("No eligible participant pairs")
    groups = groups or {}
    shared_weights = shared_weights or {}
    sizes = {key: len(value) for key, value in surfaces.items()}
    if any(s.shape[1:] != eligible.shape for s in surfaces.values()):
        raise ValueError("Pair surface shapes differ")
    group_sizes = {}
    for key, size in sizes.items():
        group = groups.get(key, key)
        if group in group_sizes and group_sizes[group] != size:
            raise ValueError("Paired run groups must have equal sizes")
        group_sizes[group] = size
    joint = {key: np.empty(draws) for key in surfaces}
    participant = {key: np.empty(draws) for key in surfaces}
    # Only eligible pairs contribute. Duplicate score-order surfaces share the
    # same participant calculation while retaining separate run multiplicities.
    pp, neg = np.where(eligible > 0)
    flat = np.concatenate([value[:, pp, neg] for value in surfaces.values()])
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    offsets = {}
    offset = 0
    for key, size in sizes.items():
        offsets[key] = inverse[offset:offset+size]
        offset += size
    cursor = rejected = 0
    np_, nn = eligible.shape
    while cursor < draws:
        count = min(500, draws - cursor)
        wp = rng.multinomial(np_, np.full(np_, 1 / np_), size=count)
        wn = rng.multinomial(nn, np.full(nn, 1 / nn), size=count)
        denom = np.einsum("bi,ij,bj->b", wp, eligible, wn, optimize=True)
        good = denom > 0
        rejected += int((~good).sum())
        wp, wn, denom = wp[good], wn[good], denom[good]
        count = len(denom)
        if not count:
            continue
        weights = {}
        for group, size in group_sizes.items():
            weights[group] = (shared_weights[group][cursor:cursor+count]
                if group in shared_weights else
                rng.multinomial(size, np.full(size, 1 / size), size=count) / size)
        pair_weights = wp[:, pp] * wn[:, neg]
        unique_scores = (pair_weights @ unique.T) / denom[:, None]
        for key in surfaces:
            per_run = unique_scores[:, offsets[key]]
            participant[key][cursor:cursor+count] = per_run.mean(axis=1)
            joint[key][cursor:cursor+count] = np.sum(per_run * weights[groups.get(key, key)], axis=1)
        cursor += count
    return joint, participant, rejected


def build_surfaces(site, include_new):
    target = _target_frame(site)
    subjects = target.subject_id.astype(str)
    surfaces = {}
    points = {}
    eligible = None

    def add(name, vectors):
        nonlocal eligible
        stack = []
        for scores in vectors:
            win, e = pair_surface(target.outcome, target.age, scores, 2.)
            if eligible is None:
                eligible = e
            assert np.array_equal(eligible, e)
            stack.append(win)
        surfaces[name] = np.stack(stack)
        points[name] = surfaces[name].sum(axis=(1, 2)) / eligible.sum()

    metrics = pd.read_csv(PACKAGE / "analysis/output/final_selection_metrics.csv")
    pred = pd.read_parquet(PACKAGE / "analysis/output/participant_predictions.parquet")
    pred = pred[pred.held_site.eq(site)].copy()
    lookup = {}
    for (key, c), frame in pred.groupby(["quartet_key", "selected_C"]):
        lookup[(key, round(float(c), 9))] = frame.assign(subject_id=frame.subject_id.astype(str)).set_index("subject_id").score.reindex(subjects).to_numpy()
    selectors = {"full_agent": "robust", "random_source_robust": "source_robust",
                 "evolutionary_robust": "robust", "taxonomy_diverse_robust": "robust"}
    for budget in [10, 25, 50, 500]:
        for method, selector in selectors.items():
            selected = metrics[metrics.held_site.eq(site) & metrics.method.eq(method) &
                metrics.selector.eq(selector) & metrics.budget.eq(budget)].sort_values("seed")
            if selected.empty:
                continue
            scores = [lookup[(row.quartet_key, round(float(row.selected_C), 9))] for row in selected.itertuples()]
            assert all(np.isfinite(x).all() for x in scores)
            name = f"{method}_B{budget}"
            add(name, scores)
            np.testing.assert_allclose(points[name], selected.target_auc, atol=1e-12)

    paired = pd.read_parquet(PACKAGE / "ieee_access_revision/results/paired_predictions_private.parquet")
    paired = paired[paired.site.eq(site)]
    for arm, frame in paired.groupby("arm"):
        vectors = [g.assign(subject_id=g.subject_id.astype(str)).set_index("subject_id").score.reindex(subjects).to_numpy()
                   for _, g in frame.groupby("seed", sort=True)]
        add(arm, vectors)

    if include_new:
        pred = pd.read_parquet(OUT / "predictions_private.parquet")
        pred = pred[pred.held_site.eq(site)]
        for method, frame in pred.groupby("method"):
            add(method, [g.assign(subject_id=g.subject_id.astype(str)).set_index("subject_id").score.reindex(subjects).to_numpy()
                         for _, g in frame.groupby("run", sort=True)])
    return surfaces, eligible, points


def main():
    include_new = (OUT / "predictions_private.parquet").exists()
    rng = np.random.default_rng(SEED)
    # A one-shot quartet is evaluated at all hospitals, preserving that dependence.
    shared = {"llm_zero_evaluation": rng.multinomial(10, np.full(10, .1), size=DRAWS) / 10,
              "random_ordering": rng.multinomial(100, np.full(100, .01), size=DRAWS) / 100}
    groups = {"full_agent_B10": "opening", "feedback": "opening", "no_feedback": "opening"}
    # The original random bank uses the same permutation at all hospitals and
    # nested budgets; preserve this across-hospital and across-budget dependence.
    groups.update({f"random_source_robust_B{b}": "random_ordering" for b in [10, 25, 50, 500]})
    all_joint = {}
    all_participant = {}
    point_estimates = {}
    rows = []
    rejects = []
    for site in SITES:
        surfaces, eligible, points = build_surfaces(site, include_new)
        joint, participant, rejected = bootstrap_site(surfaces, eligible, rng, groups=groups, shared_weights=shared)
        for name, values in points.items():
            row = dict(site=site, method=name, n_runs=len(values), auc=float(values.mean()),
                       run_sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0., **ci(joint[name]))
            row.update(participant_ci_low=ci(participant[name])["ci_low"],
                       participant_ci_high=ci(participant[name])["ci_high"])
            rows.append(row)
            all_joint.setdefault(name, []).append(joint[name])
            all_participant.setdefault(name, []).append(participant[name])
            point_estimates.setdefault(name, []).append(float(values.mean()))
        rejects.append(dict(site=site, accepted=DRAWS, rejected_zero_pair_draws=rejected))
        print("Joint bootstrap complete", site, "rejected", rejected, flush=True)
    macro_joint = {name: np.mean(arr, axis=0) for name, arr in all_joint.items()}
    macro_participant = {name: np.mean(arr, axis=0) for name, arr in all_participant.items()}
    macro_points = {name: float(np.mean(arr)) for name, arr in point_estimates.items()}
    for name, values in macro_joint.items():
        rows.append(dict(site="macro", method=name, auc=macro_points[name], **ci(values),
                         participant_ci_low=ci(macro_participant[name])["ci_low"],
                         participant_ci_high=ci(macro_participant[name])["ci_high"]))
    contrasts = []
    comparisons = [("full_agent_B10", name) for name in macro_joint if name not in
        ["full_agent_B10", "full_agent_B25", "full_agent_B50", "feedback", "no_feedback"]]
    comparisons += [("feedback", "no_feedback"), ("feedback", "full_agent_B10"),
                    ("no_feedback", "full_agent_B10")]
    comparisons += [(f"full_agent_B{b}", f"{method}_B{b}") for b in [25, 50]
                    for method in ["random_source_robust", "evolutionary_robust", "taxonomy_diverse_robust"]]
    if include_new:
        comparisons += [("llm_zero_evaluation", name) for name in
                        ["l1_top4", "greedy_forward4", "univariate_top4"]]
    for left, right in comparisons:
        delta = macro_joint[left] - macro_joint[right]
        part = macro_participant[left] - macro_participant[right]
        contrasts.append(dict(left=left, right=right, difference=macro_points[left] - macro_points[right],
            **ci(delta), participant_ci_low=ci(part)["ci_low"], participant_ci_high=ci(part)["ci_high"]))
    suffix = "" if include_new else "_original"
    pd.DataFrame(rows).to_csv(OUT / f"joint_summary{suffix}.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / f"joint_contrasts{suffix}.csv", index=False)
    np.savez_compressed(OUT / f"macro_bootstrap_draws{suffix}.npz", **macro_joint)
    (OUT / f"joint_resampling{suffix}.json").write_text(json.dumps(dict(seed=SEED, draws=DRAWS,
        sites=rejects, method="Crossed run/participant bootstrap; hospitals fixed.",
        dependence="Participants shared across methods/runs; original opening indices shared with both continuations; one-shot quartets and random-bank orderings shared across hospitals; random orderings shared across nested budgets.",
        limitations="Conditional on observed source cohort, fitted models and empirical run distributions. Source cohort uncertainty, future model-version drift and hospital sampling uncertainty are not estimated."), indent=2) + "\n")
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
