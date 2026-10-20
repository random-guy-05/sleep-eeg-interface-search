"""Extend existing test-only exclusions to all selectors; describe quartet overlap."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from agentic_search.analysis.scripts.participant_bootstrap import _target_frame
from agentic_search.ieee_access_revision.scripts.joint_uncertainty import (
    bootstrap_site, build_surfaces, ci,
)

REV = Path(__file__).resolve().parents[1]
OUT = REV / "results/feature_selection_extension"
SITES = ("I0002", "I0006", "S0001")
METHODS = ["full_agent_B10", "random_source_robust_B10", "evolutionary_robust_B10",
           "taxonomy_diverse_robust_B10", "l1_top4", "univariate_top4",
           "greedy_forward4", "llm_zero_evaluation"]


def main():
    membership = pd.read_parquet(REV / "results/pretraining_membership_private.parquet")
    rng = np.random.default_rng(2026101920)
    draws = 50000
    shared = {"llm_zero_evaluation": rng.multinomial(10, np.full(10, .1), size=draws) / 10,
              "random_source_robust_B10": rng.multinomial(100, np.full(100, .01), size=draws) / 100}
    rows = []
    boot = {}
    points = {}
    rejections = []
    for site in SITES:
        frame = _target_frame(site)
        removed = set(membership[membership.site.eq(site) & membership.development_subject_match].subject_id.astype(str))
        keep = ~frame.subject_id.astype(str).isin(removed).to_numpy()
        surfaces, eligible, _ = build_surfaces(site, True)
        pos = keep[frame.outcome.to_numpy() == 1]
        neg = keep[frame.outcome.to_numpy() == 0]
        kept = {key: surfaces[key][:, pos][:, :, neg] for key in METHODS}
        eligible = eligible[pos][:, neg]
        joint, _, rejected = bootstrap_site(kept, eligible, rng, draws=draws, shared_weights=shared)
        rejections.append(dict(site=site, n_retained=int(keep.sum()), eligible_pairs=int(eligible.sum()),
                               accepted=draws, rejected_zero_pair_draws=rejected))
        for key, tensor in kept.items():
            value = float(tensor.sum(axis=(1, 2)).mean() / eligible.sum())
            points.setdefault(key, []).append(value)
            boot.setdefault(key, []).append(joint[key])
            rows.append(dict(site=site, method=key, auc=value, **ci(joint[key])))
    for key in METHODS:
        rows.append(dict(site="macro", method=key, auc=float(np.mean(points[key])),
                         **ci(np.mean(boot[key], axis=0))))
        rows.append(dict(site="I0002_I0006", method=key, auc=float(np.mean(points[key][:2])),
                         **ci(np.mean(boot[key][:2], axis=0))))
    pd.DataFrame(rows).to_csv(OUT / "all_selector_exclusion.csv", index=False)
    pd.DataFrame(rejections).to_csv(OUT / "all_selector_exclusion_resampling.csv", index=False)
    # All historical point estimates must remain unchanged by the new resampling.
    old = pd.read_csv(REV / "results/overlap_exclusion_sensitivity.csv")
    for method in METHODS[:4]:
        for site in SITES:
            a = old[old.method.eq(method.removesuffix("_B10")) & old.site.eq(site)].iloc[0].auc
            b = next(row["auc"] for row in rows if row["method"] == method and row["site"] == site)
            assert abs(a - b) < 1e-12
    # Quartet overlap is descriptive; it cannot establish why a model chose a feature.
    historical = pd.read_csv(REV.parent / "analysis/output/final_selection_metrics.csv")
    historical = historical[historical.method.eq("full_agent") & historical.selector.eq("robust") & historical.budget.eq(10)]
    hsets = [set(key.split("|")) for key in historical.quartet_key]
    union = set.union(*hsets)
    proposals = [json.loads(p.read_text()) for p in sorted((OUT / "one_shot").glob("run_*.json"))]
    records = []
    for key in sorted({p["quartet_key"] for p in proposals}):
        q = set(key.split("|"))
        jaccard = [len(q & h) / len(q | h) for h in hsets]
        records.append(dict(quartet_key=key,
            n_calls=sum(p["quartet_key"] == key for p in proposals),
            features_in_B10_selected_union=len(q & union),
            max_shared_features=max(len(q & h) for h in hsets),
            exact_B10_matches=sum(q == h for h in hsets),
            mean_jaccard=float(np.mean(jaccard)), max_jaccard=float(max(jaccard))))
    pd.DataFrame(records).to_csv(OUT / "one_shot_quartet_overlap.csv", index=False)
    raw_rows = [next(r for r in records if r["quartet_key"] == p["quartet_key"]) for p in proposals]
    report = dict(n_calls=10, n_distinct_quartets=len(records),
        original_B10_selected_quartets=len(hsets), original_B10_feature_union_size=len(union),
        expected_union_positions_uniform_random=40 * len(union) / 539,
        H0191_original_selected_quartets=sum("H0191" in q for q in hsets),
        H0191_distinct_one_shot_quartets=sum("H0191" in r["quartet_key"].split("|") for r in records),
        H0191_one_shot_calls=sum("H0191" in p["quartet_key"].split("|") for p in proposals),
        original_B10_distinct_quartets=len({tuple(sorted(s)) for s in hsets}),
        one_shot_positions_overlapping_B10_feature_union=sum(r["features_in_B10_selected_union"] for r in raw_rows),
        total_one_shot_feature_positions=40,
        one_shot_calls_with_exact_B10_quartet_match=sum(r["exact_B10_matches"] > 0 for r in raw_rows),
        mean_cross_set_jaccard=float(np.mean([r["mean_jaccard"] for r in raw_rows])),
        interpretation="Descriptive recurrence of feature choices, not proof of a mechanism or independent biological validation.")
    (OUT / "one_shot_overlap_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index="method", columns="site", values="auc").to_string())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
