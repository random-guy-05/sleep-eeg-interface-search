"""Aggregate sealed target results into prespecified tables and search-behavior data."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
OUT = PACKAGE / "analysis/output"
SITES = ("I0002", "I0006", "S0001")


def _read_target() -> dict[tuple[str, str], dict[str, Any]]:
    path = PACKAGE / "sealed/new_target_results.parquet"
    if not path.is_file():
        raise FileNotFoundError("Target aggregation is locked until new_target_results.parquet exists")
    frame = pd.read_parquet(path)
    return {(str(row.held_site), str(row.quartet_key)): row._asdict()
            for row in frame.itertuples(index=False)}


def _identity(method: str, provider: str = "", model: str = "") -> tuple[str, str, str]:
    protocol = yaml.safe_load((PACKAGE / "configs/protocol.yaml").read_text(encoding="utf-8"))
    aliases = protocol.get("agent_models", [])
    match = next((row for row in aliases if row.get("provider") == provider), None)
    model_label = (match or {}).get("label") or (model or provider or method)
    if method == "full_agent":
        display = model_label
    else:
        display = {
            "random_source_mean": "Random + source mean",
            "random_source_robust": "Random + source robust",
            "evolutionary_robust": "Evolutionary robust",
            "taxonomy_diverse_robust": "Taxonomy-diverse robust",
            "batch_semantic": "Batch semantic",
            "opaque_iterative": "Opaque iterative",
        }.get(method, method)
    return method, str(model_label), display


def _target_value(targets: dict, held_site: str, quartet_key: str) -> float:
    result = targets.get((str(held_site), str(quartet_key)))
    if result is None:
        raise ValueError(f"Missing frozen held-site result for {held_site}/{quartet_key}")
    return float(result["target_auc"])


def _selection_row(*, method: str, method_label: str, provider: str, model: str,
                   held_site: str, budget: int, seed: int, selector: str,
                   selected: dict, oracle_target: float, target: float) -> dict[str, Any]:
    return {
        "method": method, "method_label": method_label, "provider": provider, "model": model,
        "held_site": str(held_site), "budget": int(budget), "seed": int(seed),
        "selector": selector, "quartet_key": str(selected["quartet_key"]),
        "experiment_id": str(selected.get("experiment_id", "")),
        "target_auc": float(target), "target_oracle_auc": float(oracle_target),
        "target_regret": float(oracle_target - target),
        "mean_source_auc": float(selected.get("mean_source_auc", selected.get("source_auc", np.nan))),
        "min_source_auc": float(selected.get("min_source_auc", np.nan)),
        "direction_gap": float(selected.get("direction_gap", np.nan)),
        "selected_C": float(selected.get("selected_C", np.nan)),
    }


def aggregate() -> dict[str, int]:
    targets = _read_target()
    catalog = pd.read_csv(PACKAGE / "data/heads.csv", dtype=str).set_index("head_id")
    selections: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    # The prospective random-500 bank: nested prefixes share fixed orderings across sites.
    bank = pd.read_csv(PACKAGE / "data/prospective_random500.csv", dtype=str)
    orderings = pd.read_csv(PACKAGE / "runs/source_only/random_bank_orderings.csv")
    orderings = orderings.sort_values(["repeat", "position"])
    scores = pd.read_parquet(PACKAGE / "runs/source_only/prospective_random500.parquet")
    source_lookup = {(str(row.held_site), str(row.quartet_key)): row._asdict()
                     for row in scores.itertuples(index=False)}
    bank_keys = dict(zip(bank.bank_index.astype(int), bank.quartet_key.astype(str)))
    oracle_by_prefix: dict[tuple[int, int, str], float] = {}
    for repeat, group in orderings.groupby("repeat", sort=False):
        ordered_keys = [bank_keys[int(value)] for value in group.bank_index]
        for budget in (1, 10, 25, 50, 100, 500):
            prefix = ordered_keys[:budget]
            for held_site in SITES:
                oracle_by_prefix[(int(repeat), budget, held_site)] = max(
                    _target_value(targets, held_site, key) for key in prefix)
    prefix_selections = pd.read_csv(PACKAGE / "runs/source_only/random_prefix_selections.csv", dtype={"quartet_key": str})
    for held_site in SITES:
        for row in bank.itertuples(index=False):
            heads = [row.head_1, row.head_2, row.head_3, row.head_4]
            source = dict(source_lookup[(held_site, str(row.quartet_key))])
            candidate_rows.append({**source, "method": "prospective_random_bank",
                                   "method_label": "Prospective random bank", "provider": "", "model": "",
                                   "budget": 500, "held_site": held_site,
                                   "category_count": int(len(set(catalog.loc[heads, "taxonomy_category"])))})
    for row in prefix_selections.itertuples(index=False):
        method, model_label, display = _identity(str(row.method))
        selected = source_lookup[(str(row.held_site), str(row.quartet_key))]
        selected["quartet_key"] = str(row.quartet_key)
        target = _target_value(targets, row.held_site, row.quartet_key)
        oracle = oracle_by_prefix[(int(row.repeat), int(row.budget), str(row.held_site))]
        selections.append(_selection_row(
            method=method, method_label=display, provider="", model="",
            held_site=row.held_site, budget=row.budget, seed=row.repeat,
            selector=str(row.selector), selected=selected, oracle_target=oracle, target=target))

    # Source-only classical baselines.
    classic_path = PACKAGE / "runs/source_only/classical_baselines.parquet"
    if classic_path.exists():
        classic = pd.read_parquet(classic_path)
        for (method, held_site, budget, seed), group in classic.groupby(
                ["method", "held_site", "budget", "seed"], sort=False):
            queried = []
            for row in group.to_dict("records"):
                key = str(row["quartet_key"])
                row["target_auc"] = _target_value(targets, held_site, key)
                queried.append(row)
                candidate_rows.append({**row, "method_label": _identity(method)[2],
                                       "provider": "", "model": "", "category_count": int(len(set(
                                           catalog.loc[list(row["head_ids"]), "taxonomy_category"])))})
            oracle = max(float(row["target_auc"]) for row in queried)
            robust = sorted(queried, key=lambda r: (-float(r["min_source_auc"]), str(r["quartet_key"])))[0]
            mean = sorted(queried, key=lambda r: (-float(r["mean_source_auc"]), str(r["quartet_key"])))[0]
            oracle_selected = max(queried, key=lambda r: (float(r["target_auc"]), str(r["quartet_key"])))
            for selector, selected in (("robust", robust), ("source_mean", mean), ("oracle", oracle_selected)):
                selections.append(_selection_row(
                    method=method, method_label=_identity(method)[2], provider="", model="",
                    held_site=held_site, budget=budget, seed=seed, selector=selector,
                    selected=selected, oracle_target=oracle, target=float(selected["target_auc"])))

    # Full-agent and opaque-iterative runs.
    for path in sorted((PACKAGE / "runs/source_only/agents").glob("**/*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        method = str(state.get("method", ""))
        if method not in {"full_agent", "opaque_iterative"}:
            continue
        site, budget, seed = str(state["held_site_internal"]), int(state["budget"]), int(state["seed"])
        provider, model = str(state.get("provider", "")), str(state.get("model_requested", ""))
        _, model_label, display = _identity(method, provider, model)
        queried = []
        for item in state.get("history", []):
            head_ids = item.get("real_head_ids") or item["head_ids"]
            key = item.get("real_quartet_key") or "|".join(sorted(map(str, head_ids)))
            candidate = {**item, "head_ids": list(sorted(map(str, head_ids))),
                         "quartet_key": key,
                         "experiment_id": str(state["experiment_id"]),
                         "target_auc": _target_value(targets, site, key)}
            queried.append(candidate)
            candidate_rows.append({**candidate, "method": method, "method_label": display,
                                   "provider": provider, "model": model,
                                   "experiment_id": state["experiment_id"], "budget": budget,
                                   "seed": seed, "held_site": site,
                                   "category_count": int(len(set(catalog.loc[candidate["head_ids"], "taxonomy_category"])))})
        if len(queried) != budget:
            raise ValueError(f"Incomplete source-search trajectory: {path}")
        oracle = max(float(row["target_auc"]) for row in queried)
        robust_key = "|".join(sorted(state.get("robust_selected_real_head_ids", [])))
        robust = next(row for row in queried if row["quartet_key"] == robust_key)
        source_mean = sorted(queried, key=lambda r: (-float(r["mean_source_auc"]), str(r["quartet_key"])))[0]
        choices = []
        agent_choice = None
        if state.get("agent_choice_valid") and state.get("agent_choice_real_head_ids"):
            choice_key = "|".join(sorted(state["agent_choice_real_head_ids"]))
            agent_choice = next((row for row in queried if row["quartet_key"] == choice_key), None)
        for selector, selected in (("robust", robust), ("source_mean", source_mean),
                                   ("agent_choice", agent_choice),
                                   ("oracle", max(queried, key=lambda r: (float(r["target_auc"]), str(r["quartet_key"]))))):
            if selected is None:
                continue
            selections.append(_selection_row(
                method=method, method_label=display, provider=provider, model=model,
                held_site=site, budget=budget, seed=seed, selector=selector,
                selected=selected, oracle_target=oracle, target=float(selected["target_auc"])))

    # One-call semantic B=25 ablation.
    semantic_path = PACKAGE / "runs/source_only/batch_semantic.parquet"
    if semantic_path.exists():
        semantic = pd.read_parquet(semantic_path)
        for (experiment_id, held_site, seed), group in semantic.groupby(
                ["experiment_id", "held_site", "seed"], sort=False):
            queried = []
            provider, model = str(group.iloc[0].provider), str(group.iloc[0].model_requested)
            for row in group.to_dict("records"):
                row["target_auc"] = _target_value(targets, held_site, row["quartet_key"])
                queried.append(row)
                candidate_rows.append({**row, "method": "batch_semantic",
                                       "method_label": _identity("batch_semantic")[2],
                                       "provider": provider, "model": model,
                                       "category_count": int(len(set(catalog.loc[list(row["head_ids"]), "taxonomy_category"])))})
            if len(queried) != 25:
                raise ValueError(f"Batch Semantic run should query 25 quartets: {experiment_id}")
            oracle = max(float(row["target_auc"]) for row in queried)
            robust = sorted(queried, key=lambda r: (-float(r["min_source_auc"]), str(r["quartet_key"])))[0]
            mean = sorted(queried, key=lambda r: (-float(r["mean_source_auc"]), str(r["quartet_key"])))[0]
            oracle_selected = max(queried, key=lambda r: (float(r["target_auc"]), str(r["quartet_key"])))
            for selector, selected in (("robust", robust), ("source_mean", mean),
                                       ("oracle", oracle_selected)):
                selections.append(_selection_row(
                    method="batch_semantic", method_label=_identity("batch_semantic")[2],
                    provider=provider, model=model, held_site=held_site, budget=25, seed=seed,
                    selector=selector, selected=selected, oracle_target=oracle,
                    target=float(selected["target_auc"])))

    if not selections:
        raise RuntimeError("No source-only selections found for post-target aggregation")
    selection_frame = pd.DataFrame(selections)
    OUT.mkdir(parents=True, exist_ok=True)
    selection_frame.to_csv(OUT / "final_selection_metrics.csv", index=False)
    candidates_frame = pd.DataFrame(candidate_rows)
    if not candidates_frame.empty:
        candidates_frame.to_parquet(OUT / "queried_candidate_results.parquet", index=False)

    # Paired runs across the three sites form each macro trial. These intervals describe
    # trajectory/run variability; they are not participant-level confidence intervals.
    group_columns = ["method", "method_label", "provider", "model", "budget", "selector", "seed"]
    macro_trial = selection_frame.groupby(group_columns, dropna=False).agg(
        n_sites=("held_site", "nunique"), macro_target_auc=("target_auc", "mean"),
        macro_source_auc=("mean_source_auc", "mean"), macro_optimism_gap=("mean_source_auc", lambda v: float(v.mean())),
        macro_target_regret=("target_regret", "mean"), macro_oracle_auc=("target_oracle_auc", "mean")).reset_index()
    macro_trial = macro_trial[macro_trial.n_sites.eq(len(SITES))].copy()
    # source-to-target optimism is defined per selection before macro-averaging.
    optimism = selection_frame.assign(optimism_gap=selection_frame.mean_source_auc - selection_frame.target_auc)
    optimism_by_trial = optimism.groupby(group_columns, dropna=False).optimism_gap.mean().rename("macro_optimism_gap").reset_index()
    macro_trial = macro_trial.drop(columns="macro_optimism_gap").merge(optimism_by_trial, on=group_columns, how="left")
    trial_summary = macro_trial.groupby(group_columns[:-1], dropna=False).agg(
        n_trajectory_repetitions=("seed", "nunique"), macro_mean=("macro_target_auc", "mean"),
        macro_median=("macro_target_auc", "median"), macro_p2_5=("macro_target_auc", lambda x: x.quantile(.025)),
        macro_p97_5=("macro_target_auc", lambda x: x.quantile(.975)),
        optimism_gap_mean=("macro_optimism_gap", "mean"), target_regret_mean=("macro_target_regret", "mean"),
        target_oracle_mean=("macro_oracle_auc", "mean")).reset_index()
    site_means = selection_frame.groupby(["method", "method_label", "provider", "model", "budget", "selector", "held_site"], dropna=False).target_auc.mean().unstack("held_site")
    site_means = site_means.rename(columns={site: f"{site}_mean" for site in SITES}).reset_index()
    site_means = site_means[[column for column in site_means.columns if column in {
        "method", "method_label", "provider", "model", "budget", "selector", "I0002_mean", "I0006_mean", "S0001_mean"}]]
    trial_summary = trial_summary.merge(site_means, on=["method", "method_label", "provider", "model", "budget", "selector"], how="left")
    trial_summary.to_csv(OUT / "paired_trajectory_summary.csv", index=False)
    macro_trial.to_csv(OUT / "paired_macro_trials.csv", index=False)

    primary = trial_summary[trial_summary.selector.eq("robust")].copy()
    random_mean = trial_summary[(trial_summary.method.eq("random_source_mean")) & trial_summary.selector.eq("source_mean")].copy()
    random_robust = trial_summary[(trial_summary.method.eq("random_source_robust")) & trial_summary.selector.eq("source_robust")].copy()
    main = pd.concat([random_mean, random_robust, primary], ignore_index=True)
    main = main.sort_values(["budget", "method_label", "provider"], kind="stable")
    main.to_csv(OUT / "main_results_table.csv", index=False)

    # Round-by-round adaptive search diagnostics (target oracle is explicitly post hoc).
    trace_rows = []
    for path in sorted((PACKAGE / "runs/source_only/agents").glob("**/*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if state.get("method") not in {"full_agent", "opaque_iterative"} or not state.get("history"):
            continue
        history = sorted(state["history"], key=lambda row: (int(row["round"]), int(row["batch_index"])))
        best_source = -np.inf
        best_target = -np.inf
        for row in history:
            real_ids = row.get("real_head_ids") or row["head_ids"]
            key = row.get("real_quartet_key") or "|".join(sorted(real_ids))
            target = _target_value(targets, state["held_site_internal"], key)
            best_source = max(best_source, float(row["min_source_auc"]))
            best_target = max(best_target, target)
            trace_rows.append({"experiment_id": state["experiment_id"], "method": state["method"],
                               "provider": state.get("provider"), "model": state.get("model_requested"),
                               "method_label": _identity(state["method"], state.get("provider", ""),
                                                          state.get("model_requested", ""))[2],
                               "held_site": state["held_site_internal"], "budget": state["budget"],
                               "seed": state["seed"], "round": row["round"],
                               "batch_index": row["batch_index"], "quartet_key": key,
                               "source_robust": row["min_source_auc"], "source_mean": row["mean_source_auc"],
                               "direction_gap": row["direction_gap"], "target_auc_posthoc": target,
                               "best_source_robust_so_far": best_source,
                               "best_target_oracle_so_far_posthoc": best_target})
    if trace_rows:
        pd.DataFrame(trace_rows).to_csv(OUT / "agent_round_diagnostics.csv", index=False)

    # Candidate-level biological search behavior.
    if not candidates_frame.empty:
        head_rows = []
        for row in candidates_frame.itertuples(index=False):
            for head in row.head_ids:
                head_rows.append({"method": row.method, "method_label": row.method_label,
                                  "provider": getattr(row, "provider", ""), "model": getattr(row, "model", ""),
                                  "budget": row.budget, "head_id": str(head),
                                  "head_name": str(catalog.loc[str(head), "head_name"]),
                                  "taxonomy_category": str(catalog.loc[str(head), "taxonomy_category"])})
        head_frame = pd.DataFrame(head_rows)
        frequency = head_frame.groupby(["method", "method_label", "provider", "model", "budget",
                                        "head_id", "head_name", "taxonomy_category"], dropna=False).size().rename("query_count").reset_index()
        frequency.to_csv(OUT / "head_and_category_frequency.csv", index=False)
        category_frequency = head_frame.groupby(["method", "method_label", "provider", "model", "budget",
                                                 "taxonomy_category"], dropna=False).size().rename("head_occurrences").reset_index()
        category_frequency.to_csv(OUT / "taxonomy_frequency.csv", index=False)
        diversity = candidates_frame.groupby(["method", "method_label", "provider", "model", "budget"], dropna=False).agg(
            mean_unique_categories=("category_count", "mean"), min_unique_categories=("category_count", "min"),
            candidate_count=("quartet_key", "size")).reset_index()
        diversity.to_csv(OUT / "quartet_category_diversity.csv", index=False)

    # Convergence of robust-selected head sets across independent trajectories, within site.
    robust_rows = selection_frame[selection_frame.selector.eq("robust")].to_dict("records")
    convergence = []
    for keys, group in pd.DataFrame(robust_rows).groupby(
            ["method", "method_label", "provider", "model", "budget", "held_site"], dropna=False):
        sets = [set(str(row["quartet_key"]).split("|")) for row in group.to_dict("records")]
        values = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
        if values:
            convergence.append({**dict(zip(["method", "method_label", "provider", "model", "budget", "held_site"], keys)),
                                "n_pairs": len(values), "mean_jaccard_selected_heads": float(np.mean(values))})
    if convergence:
        pd.DataFrame(convergence).to_csv(OUT / "selected_quartet_convergence.csv", index=False)

    # Choose illustrative full-agent traces using a declared target-outcome ranking rule.
    full = selection_frame[(selection_frame.method.eq("full_agent")) & selection_frame.selector.eq("robust")].copy()
    if not full.empty:
        full = full.sort_values(["target_auc", "provider", "held_site", "budget", "seed"]).reset_index(drop=True)
        picked = {"worst": full.iloc[0], "median": full.iloc[(len(full) - 1) // 2], "best": full.iloc[-1]}
        traces = []
        by_experiment = {}
        for path in (PACKAGE / "runs/source_only/agents").glob("**/*.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if state.get("experiment_id"):
                by_experiment[state["experiment_id"]] = state
        for label, selected in picked.items():
            state = by_experiment.get(selected.get("experiment_id"))
            if not state:
                continue
            events = []
            for row in sorted(state["history"], key=lambda r: (r["round"], r["batch_index"])):
                ids = row.get("real_head_ids") or row["head_ids"]
                key = row.get("real_quartet_key") or "|".join(sorted(ids))
                cats = sorted(set(catalog.loc[list(ids), "taxonomy_category"].astype(str)))
                events.append({"round": row["round"], "batch_index": row["batch_index"],
                               "quartet_key": key, "taxonomy_categories": cats,
                               "auc_source1_to_source2": row["auc_source1_to_source2"],
                               "auc_source2_to_source1": row["auc_source2_to_source1"],
                               "mean_source_auc": row["mean_source_auc"], "min_source_auc": row["min_source_auc"],
                               "target_auc_posthoc": _target_value(targets, state["held_site_internal"], key)})
            traces.append({"rank_rule": "robust-selected held-site AUROC pooled across full-agent fold-trajectories",
                           "selection": label, "experiment_id": state["experiment_id"],
                           "held_site": state["held_site_internal"], "budget": state["budget"],
                           "seed": state["seed"], "provider": state.get("provider"),
                           "model": state.get("model_requested"), "robust_selected_target_auc": float(selected.target_auc),
                           "events": events})
        (OUT / "manual_trajectory_review.json").write_text(json.dumps(traces, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    return {"selections": len(selection_frame), "candidate_rows": len(candidates_frame),
            "summaries": len(trial_summary), "paired_macro_trials": len(macro_trial)}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(aggregate(), indent=2))


if __name__ == "__main__":
    main()
