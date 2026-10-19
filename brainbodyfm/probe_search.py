"""Source-only representation selection and descriptive transport analyses."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def recover_candidate(task):
    """Re-execute one original draw, retaining the previously omitted traces."""
    import hashlib
    from pathlib import Path
    from .controls import run_random_k_controls
    matrix, manifest, names, config, draw, expected_macro, families, output, run_key = task
    directory = Path(output) / "candidates" / f"draw_{draw:03d}"
    marker = directory / "recovery.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["run_key"] != run_key:
            raise ValueError("Recovery inputs changed; refusing stale candidate checkpoint")
        for name, digest in saved["hashes"].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Recovered candidate artifact hash mismatch")
        return saved["panel"]
    summaries, results = run_random_k_controls(
        matrix, manifest, feature_names=names, k_values=(4,), n_draws=500,
        seed=config.random_k_seed, config=config, draw_indices=(draw,), retain_results=True)
    result = results[(4, draw)]
    if not np.isclose(result.macro_auc, expected_macro, rtol=0, atol=1e-12):
        raise ValueError(f"Original draw {draw} was not reproduced: {result.macro_auc} vs {expected_macro}")
    result.save(directory)
    heads = list(result.coefficients.feature.drop_duplicates())
    if len(heads) != 4:
        raise ValueError("Recovered quartet width changed")
    rows = []
    for fold in result.folds:
        if fold.target_rows_in_preprocessing_fit or fold.target_labels_used_in_hyperparameter_selection:
            raise ValueError("Target entered source model selection")
        rows.append(dict(draw=int(draw), outer_site=fold.outer_site, source_sites="+".join(fold.source_sites),
                         source_auc=fold.inner_auc, source_A_to_B_auc=fold.inner_A_to_B_auc,
                         source_B_to_A_auc=fold.inner_B_to_A_auc, target_auc=fold.outer_auc,
                         chosen_C=fold.chosen_C, heads=json.dumps(heads),
                         families=json.dumps([families[name] for name in heads]),
                         target_labels_used_for_selection=0, original_macro_auc=float(expected_macro)))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(run_key=run_key, panel=rows, hashes=hashes), indent=2))
    temporary.replace(marker)
    return rows


def source_winner(source_scores: pd.Series, candidate_ids) -> int:
    """No target-score argument: exact ties resolve to the smallest draw ID."""
    ids = np.asarray(candidate_ids, dtype=int)
    if not len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be nonempty and unique")
    values = source_scores.loc[ids].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("All candidate source scores must be finite")
    return int(ids[values == values.max()].min())


def validate_panel(panel: pd.DataFrame, *, n_candidates=500, sites=("I0002", "I0006", "S0001")):
    required = {"draw", "outer_site", "source_auc", "target_auc", "heads", "families", "chosen_C"}
    if not required.issubset(panel):
        raise ValueError(f"Missing panel columns: {sorted(required.difference(panel))}")
    expected = {(s, d) for s in sites for d in range(n_candidates)}
    if panel.duplicated(["outer_site", "draw"]).any() or set(zip(panel.outer_site, panel.draw)) != expected:
        raise ValueError("Incomplete or duplicated candidate-by-site panel")
    if not np.isfinite(panel[["source_auc", "target_auc", "chosen_C"]].to_numpy(float)).all():
        raise ValueError("Nonfinite panel estimates")
    if not panel[["source_auc", "target_auc"]].apply(lambda x: x.between(0, 1)).all().all():
        raise ValueError("AUROC outside [0,1]")
    for _, group in panel.groupby("draw"):
        if group.heads.nunique() != 1 or group.families.nunique() != 1:
            raise ValueError("A candidate must use the same quartet in every fold")
        if len(set(json.loads(group.heads.iloc[0]))) != 4:
            raise ValueError("Each candidate must have four distinct heads")


def search_budget_analysis(panel, permutations, budgets):
    rows = []
    for site, group in panel.groupby("outer_site", sort=True):
        indexed = group.set_index("draw")
        if not indexed.index.is_unique:
            raise ValueError("Candidate IDs must be unique within each site")
        scores = indexed.source_auc
        for repeat, permutation in enumerate(permutations):
            if set(permutation) != set(indexed.index) or len(permutation) != len(indexed):
                raise ValueError("Every permutation must contain every candidate exactly once")
            for budget in budgets:
                if not 1 <= budget <= len(permutation):
                    raise ValueError("Invalid search budget")
                winner = source_winner(scores, permutation[:budget])
                chosen = indexed.loc[winner]
                rows.append(dict(outer_site=site, permutation=repeat, budget=budget,
                                 selected_draw=winner, source_auc=float(chosen.source_auc),
                                 target_auc=float(chosen.target_auc), chosen_C=float(chosen.chosen_C),
                                 heads=chosen.heads, families=chosen.families,
                                 target_labels_used_for_selection=0))
    detail = pd.DataFrame(rows)
    macro = detail.groupby(["permutation", "budget"], as_index=False)[["source_auc", "target_auc"]].mean()
    if not detail.groupby(["permutation", "budget"]).outer_site.nunique().eq(panel.outer_site.nunique()).all():
        raise ValueError("Missing site in macro summary")
    rows = []
    for budget, group in macro.groupby("budget", sort=True):
        row = dict(budget=int(budget), n_permutations=len(group))
        for metric in ("source_auc", "target_auc"):
            values = group[metric].to_numpy(float)
            row.update({f"{metric}_mean": float(values.mean()), f"{metric}_median": float(np.median(values)),
                        f"{metric}_q25": float(np.quantile(values, .25)), f"{metric}_q75": float(np.quantile(values, .75))})
        rows.append(row)
    return detail, macro, pd.DataFrame(rows)


def source_target_statistics(panel):
    rows = []
    for site, group in panel.groupby("outer_site", sort=True):
        ranked = group.sort_values(["source_auc", "draw"], ascending=[False, True])
        top = ranked.head(max(1, int(np.ceil(.1 * len(ranked)))))
        winner = ranked.iloc[0]
        varying = group.source_auc.nunique() > 1 and group.target_auc.nunique() > 1
        rows.append(dict(outer_site=site, n_candidates=len(group),
                         spearman=float(spearmanr(group.source_auc, group.target_auc).statistic) if varying else np.nan,
                         pearson=float(pearsonr(group.source_auc, group.target_auc).statistic) if varying else np.nan,
                         all_target_mean=group.target_auc.mean(), source_top_decile_n=len(top),
                         source_top_decile_target_mean=top.target_auc.mean(),
                         source_top_decile_target_median=top.target_auc.median(),
                         selected_draw=int(winner.draw), selected_source_auc=winner.source_auc,
                         selected_target_auc=winner.target_auc))
    return pd.DataFrame(rows)


def family_enrichment(panel):
    """Presence per quartet (not coordinate count); overlapping groups, descriptive."""
    candidates = panel.drop_duplicates("draw").set_index("draw")
    family_sets = {int(d): set(json.loads(row.families)) for d, row in candidates.iterrows()}
    families = sorted(set().union(*family_sets.values()))
    groups = []
    for site, group in panel.groupby("outer_site", sort=True):
        ranked = group.sort_values(["source_auc", "draw"], ascending=[False, True])
        for fraction in (.10, .05, .01):
            selected = ranked.head(max(1, int(np.ceil(fraction * len(group)))))
            groups.append(("source_only", site, f"top_{int(fraction*100)}_percent", selected.draw.tolist(),
                           selected.source_auc.mean(), selected.target_auc.mean()))
    macro = panel.groupby("draw").target_auc.mean().reset_index().sort_values(["target_auc", "draw"], ascending=[False, True])
    for name, fraction, bottom in (("top_5_percent", .05, False), ("top_1_percent", .01, False), ("bottom_50_percent", .5, True)):
        n = max(1, int(np.ceil(fraction * len(macro))))
        selected = macro.tail(n) if bottom else macro.head(n)
        groups.append(("target_selected_exploratory", "macro", name, selected.draw.tolist(), np.nan, selected.target_auc.mean()))
    rows = []
    for basis, site, name, ids, source_mean, target_mean in groups:
        for family in families:
            baseline = np.mean([family in values for values in family_sets.values()])
            presence = np.mean([family in family_sets[int(draw)] for draw in ids])
            rows.append(dict(selection_basis=basis, outer_site=site, group=name, family=family,
                             n_quartets=len(ids), group_presence=presence, all_presence=baseline,
                             presence_ratio=presence/baseline if baseline else np.nan,
                             source_auc_mean=source_mean, target_auc_mean=target_mean))
    return pd.DataFrame(rows)
