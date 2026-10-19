"""Recover GPT-6 Sol's secondary final picks from frozen run states.

This post-target descriptive analysis is separate from the frozen primary
aggregator. The primary aggregator expects an older field name for the model's
final choice; completed run states store the choice under
``agent_choice_quartet_real_head_ids``. No search or scoring code is changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / "analysis/output"
STATE_ROOT = PACKAGE / "runs/source_only/agents"
SITES = ("I0002", "I0006", "S0001")
DRAWS = 50_000
SEED = 20260926


def _bootstrap_paired_difference(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError("At least two paired seeds are required for a run-level interval")
    indices = rng.integers(0, len(values), size=(DRAWS, len(values)))
    draws = values[indices].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = pd.read_parquet(PACKAGE / "sealed/new_target_results.parquet")
    target_lookup = {
        (str(row.held_site), str(row.quartet_key)): float(row.target_auc)
        for row in targets.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for path in sorted(STATE_ROOT.glob("**/*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if state.get("method") != "full_agent":
            continue
        if state.get("agent_choice_valid") is not True:
            raise ValueError(f"Invalid or missing final model choice in {path}")
        choice_ids = state.get("agent_choice_quartet_real_head_ids")
        choice_key = state.get("agent_choice_quartet_key")
        if not choice_ids or not choice_key:
            raise ValueError(f"Final model choice fields are incomplete in {path}")
        choice_ids = sorted(map(str, choice_ids))
        canonical_key = "|".join(choice_ids)
        if canonical_key != str(choice_key):
            raise ValueError(f"Final choice IDs and key disagree in {path}")
        history = state.get("history", [])
        selected = next((row for row in history
                         if str(row.get("real_quartet_key", row.get("quartet_key"))) == canonical_key), None)
        if selected is None:
            raise ValueError(f"Final model choice was not source-evaluated in {path}")
        site = str(state["held_site_internal"])
        target_auc = target_lookup.get((site, canonical_key))
        if target_auc is None:
            raise ValueError(f"Final model choice is absent from the frozen target release: {path}")
        robust_key = str(state["robust_selected_quartet_key"])
        robust_target = target_lookup.get((site, robust_key))
        if robust_target is None:
            raise ValueError(f"Robust-selected quartet is absent from target release: {path}")
        candidate_targets = [target_lookup[(site, str(row.get("real_quartet_key", row["quartet_key"]))) ]
                             for row in history]
        rows.append({
            "experiment_id": str(state["experiment_id"]),
            "held_site": site,
            "budget": int(state["budget"]),
            "seed": int(state["seed"]),
            "agent_choice_quartet_key": canonical_key,
            "agent_choice_target_auc": target_auc,
            "agent_choice_source_mean_auc": float(selected["mean_source_auc"]),
            "agent_choice_min_source_auc": float(selected["min_source_auc"]),
            "agent_choice_selected_C": float(selected["selected_C"]),
            "robust_selected_quartet_key": robust_key,
            "robust_selected_target_auc": float(robust_target),
            "agent_choice_minus_robust_target_auc": target_auc - float(robust_target),
            "agent_choice_target_regret_within_trajectory": max(candidate_targets) - target_auc,
            "agent_choice_differs_from_robust": canonical_key != robust_key,
        })

    detail = pd.DataFrame(rows)
    if len(detail) != 90 or detail.experiment_id.nunique() != 90:
        raise ValueError(f"Expected 90 unique primary model trajectories, found {len(detail)}")
    macro = detail.groupby(["budget", "seed"], as_index=False).agg(
        macro_agent_choice_target_auc=("agent_choice_target_auc", "mean"),
        macro_robust_target_auc=("robust_selected_target_auc", "mean"),
        macro_choice_minus_robust=("agent_choice_minus_robust_target_auc", "mean"),
        macro_choice_regret=("agent_choice_target_regret_within_trajectory", "mean"),
        changed_site_picks=("agent_choice_differs_from_robust", "sum"),
        n_sites=("held_site", "nunique"),
    )
    macro = macro[macro.n_sites.eq(len(SITES))].copy()
    if macro.groupby("budget").size().to_dict() != {10: 10, 25: 10, 50: 10}:
        raise ValueError("Expected ten complete three-site macro trials at each budget")

    rng = np.random.default_rng(SEED)
    summaries: list[dict[str, object]] = []
    for budget in (10, 25, 50):
        selected = detail[detail.budget.eq(budget)]
        paired = macro[macro.budget.eq(budget)].sort_values("seed")
        lo, hi = _bootstrap_paired_difference(paired.macro_choice_minus_robust.to_numpy(float), rng)
        macro_values = paired.macro_agent_choice_target_auc.to_numpy(float)
        row: dict[str, object] = {
            "budget": budget,
            "n_trajectories": int(len(selected)),
            "macro_mean_target_auc": float(macro_values.mean()),
            "macro_median_target_auc": float(np.median(macro_values)),
            "macro_run_percentile_2_5": float(np.quantile(macro_values, 0.025)),
            "macro_run_percentile_97_5": float(np.quantile(macro_values, 0.975)),
            "mean_target_regret_within_queried_set": float(selected.agent_choice_target_regret_within_trajectory.mean()),
            "mean_site_picks_differing_from_robust": float(selected.agent_choice_differs_from_robust.mean()),
            "paired_delta_vs_robust": float(paired.macro_choice_minus_robust.mean()),
            "paired_delta_bootstrap_run_ci_2_5": lo,
            "paired_delta_bootstrap_run_ci_97_5": hi,
            "paired_seed_win_fraction_vs_robust": float((paired.macro_choice_minus_robust > 0).mean()),
            "bootstrap_draws": DRAWS,
            "bootstrap_seed": SEED,
            "interpretation": "post-hoc trajectory-level comparison; not a patient-level confidence interval or confirmatory test",
        }
        for site in SITES:
            row[f"{site}_mean_target_auc"] = float(
                selected.loc[selected.held_site.eq(site), "agent_choice_target_auc"].mean()
            )
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT / "agent_choice_secondary_per_trajectory.csv", index=False)
    summary.to_csv(OUT / "agent_choice_secondary_summary.csv", index=False)
    return detail, summary


if __name__ == "__main__":
    detail, summary = main()
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
