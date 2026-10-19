"""Freeze every unique source-evaluated candidate and its experiment roles."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from agentic_search.freeze_protocol import current_execution_hashes, current_input_hashes


PACKAGE = Path(__file__).resolve().parent


def _hash_json(value: Any) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def _source_record(candidates: dict, held_site: str, quartet_key: str, source_auc: float,
                   min_source_auc: float, direction_gap: float, selected_C: float,
                   head_ids: list[str], membership: str) -> None:
    key = (str(held_site), str(quartet_key))
    if key not in candidates:
        candidates[key] = {"held_site": key[0], "quartet_key": key[1],
                           "head_ids": sorted(map(str, head_ids)),
                           "mean_source_auc": float(source_auc), "min_source_auc": float(min_source_auc),
                           "direction_gap": float(direction_gap), "selected_C": float(selected_C),
                           "memberships": []}
    else:
        row = candidates[key]
        for column, value in (("mean_source_auc", source_auc), ("min_source_auc", min_source_auc),
                              ("direction_gap", direction_gap), ("selected_C", selected_C)):
            if abs(float(row[column]) - float(value)) > 1e-10:
                raise ValueError(f"Conflicting source result for {key}: {column}")
        if row["head_ids"] != sorted(map(str, head_ids)):
            raise ValueError(f"Quartet key and head IDs disagree for {key}")
    if membership not in candidates[key]["memberships"]:
        candidates[key]["memberships"].append(membership)


def _load_agent_states() -> list[Path]:
    root = PACKAGE / "runs/source_only/agents"
    states = []
    for path in root.glob("**/*.json"):
        if path.name == "protocol_freeze.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if data.get("method") not in {"full_agent", "opaque_iterative"}:
            continue
        states.append(path)
    return sorted(states)


def build() -> dict[str, Any]:
    destination = PACKAGE / "runs/frozen_manifest.json"
    if any((PACKAGE / f"sealed/{name}").exists() for name in
           ("new_target_results.parquet", "new_target_results_progress.parquet")):
        raise RuntimeError("Target scoring has started; refusing to rebuild the frozen manifest")
    freeze_path = PACKAGE / "runs/protocol_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("phase") != "final" or not freeze.get("prompt_sha256"):
        raise RuntimeError("Freeze final prompts after the source-only pilot before building the manifest")
    protocol = yaml.safe_load((PACKAGE / "configs/protocol.yaml").read_text(encoding="utf-8"))
    if hashlib.sha256((PACKAGE / "configs/protocol.yaml").read_bytes()).hexdigest() != freeze.get("protocol_sha256"):
        raise RuntimeError("Protocol changed after the final freeze")
    if current_execution_hashes() != freeze.get("execution_sha256"):
        raise RuntimeError("Executable pipeline code changed after the final freeze")
    if current_input_hashes() != freeze.get("input_sha256"):
        raise RuntimeError("Source-only experiment inputs changed after the final freeze")
    current_prompts = {name: hashlib.sha256((PACKAGE / f"configs/prompts/{name}").read_bytes()).hexdigest()
                       for name in ("full_agent.txt", "opaque_agent.txt", "batch_semantic.txt")}
    if current_prompts != freeze.get("prompt_sha256"):
        raise RuntimeError("Prompts changed after the final freeze")
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    historical = set(pd.read_csv(PACKAGE / "data/historical_quartets.csv", dtype=str).quartet_key)
    catalog_ids = set(pd.read_csv(PACKAGE / "data/heads.csv", dtype=str).head_id)

    bank = pd.read_parquet(PACKAGE / "runs/source_only/prospective_random500.parquet")
    if len(bank) != 1500 or bank.duplicated(["held_site", "quartet_key"]).any():
        raise RuntimeError("Prospective source-only bank is incomplete or duplicated")
    for row in bank.itertuples(index=False):
        _source_record(candidates, row.held_site, row.quartet_key, row.mean_source_auc,
                       row.min_source_auc, row.direction_gap, row.selected_C, list(row.head_ids),
                       f"prospective_random500:{row.held_site}")
    classical_path = PACKAGE / "runs/source_only/classical_baselines.parquet"
    if not classical_path.exists():
        raise RuntimeError("Complete source-only classical baseline runs are required before manifest freeze")
    classical = pd.read_parquet(classical_path)
    expected_classical = 2 * 3 * 3 * 20
    groups = classical.groupby(["method", "held_site", "budget", "seed"], sort=False)
    if groups.ngroups != expected_classical:
        raise RuntimeError(f"Expected {expected_classical} complete classical trajectories, found {groups.ngroups}")
    for keys, group in groups:
        if len(group) != int(keys[2]) or group.quartet_key.duplicated().any():
            raise RuntimeError(f"Incomplete/repeated classical trajectory: {keys}")
    for row in classical.itertuples(index=False):
        membership = f"{row.method}:{row.held_site}:B{row.budget}:seed{row.seed}:query{row.candidate_index}"
        if bool(row.selected_by_robust):
            membership += ":robust_selected"
        _source_record(candidates, row.held_site, row.quartet_key, row.mean_source_auc,
                       row.min_source_auc, row.direction_gap, row.selected_C, list(row.head_ids), membership)

    for path in _load_agent_states():
        state = json.loads(path.read_text(encoding="utf-8"))
        budget = int(state["budget"])
        if int(state.get("budget_used", -1)) != budget or len(state.get("history", [])) != budget:
            raise RuntimeError(f"Incomplete agent trajectory cannot enter the final manifest: {path}")
        history = state["history"]
        keys = [row["quartet_key"] for row in history]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Repeated candidate within trajectory: {path}")
        robust_key = state.get("robust_selected_quartet_key")
        choice_key = state.get("agent_choice_quartet_key") if state.get("agent_choice_valid") else None
        for row in history:
            membership = f"{state['method']}:{state['experiment_id']}:query{int(row['round']):02d}-{int(row['batch_index'])}"
            if row["quartet_key"] == robust_key:
                membership += ":robust_selected"
            if row["quartet_key"] == choice_key:
                membership += ":agent_choice"
            real_ids = row.get("real_head_ids") or row["head_ids"]
            _source_record(candidates, state["held_site_internal"], "|".join(sorted(real_ids)),
                           row["mean_source_auc"], row["min_source_auc"], row["direction_gap"],
                           row["selected_C"], real_ids, membership)

    primary_full = [state for _, state in ((path, json.loads(path.read_text(encoding="utf-8")))
                   for path in _load_agent_states()) if state.get("method") == "full_agent"]
    full_models = {(row["provider"], row["model"]) for row in protocol["agent_models"]}
    expected_full = len(full_models) * 3 * 3 * 10
    if len(primary_full) != expected_full:
        raise RuntimeError(f"Expected {expected_full} complete full-agent trajectories, found {len(primary_full)}")
    full_ids = [(state["provider"], state["model_requested"], state["held_site_internal"],
                 int(state["budget"]), int(state["seed"])) for state in primary_full]
    if len(set(full_ids)) != expected_full:
        raise RuntimeError("Full-agent grid contains duplicate or missing experiment identities")
    expected_full_ids = {(provider, model, site, int(budget), int(seed))
                         for provider, model in full_models
                         for site in json.loads((PACKAGE / "data/folds.json").read_text())["site_order"]
                         for budget in protocol["main_budgets"]
                         for seed in protocol["trajectory_seeds"]}
    if set(full_ids) != expected_full_ids:
        raise RuntimeError("Full-agent runs do not cover the frozen model/site/budget/seed grid")
    for state in primary_full:
        model = (state["provider"], state["model_requested"])
        if state.get("protocol_sha256") != freeze["protocol_sha256"]:
            raise RuntimeError(f"Full-agent run used a different protocol: {state['experiment_id']}")
        if state.get("prompt_sha256") != freeze["prompt_sha256"]["full_agent.txt"]:
            raise RuntimeError(f"Full-agent run used a different prompt: {state['experiment_id']}")

    primary_opaque = [json.loads(path.read_text(encoding="utf-8")) for path in
                      (PACKAGE / "runs/source_only/agents").glob("**/*.json")
                      if path.is_file() and path.suffix == ".json" and
                      json.loads(path.read_text(encoding="utf-8")).get("method") == "opaque_iterative"]
    default_model = (protocol["agent_models"][0]["provider"], protocol["agent_models"][0]["model"])
    expected_opaque = {(default_model[0], default_model[1], site, 25, int(seed))
                       for site in json.loads((PACKAGE / "data/folds.json").read_text())["site_order"]
                       for seed in protocol["trajectory_seeds"]}
    opaque_ids = {(state["provider"], state["model_requested"], state["held_site_internal"],
                   int(state["budget"]), int(state["seed"])) for state in primary_opaque}
    if opaque_ids != expected_opaque:
        raise RuntimeError("Opaque iterative runs do not cover the frozen B=25 ablation grid")

    pilot_protocol = protocol["pilot"]
    pilot_states = []
    for path in (PACKAGE / "runs/pilot/pilot").glob("**/*.json"):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if state.get("provider") == pilot_protocol["provider"] and state.get("model_requested") == pilot_protocol["model"]:
            pilot_states.append(state)
    pilot_ids = {(state.get("held_site_internal"), int(state.get("budget", -1)), int(state.get("seed", -1)))
                 for state in pilot_states if state.get("completed_at_utc")}
    expected_pilot = {(pilot_protocol["held_site"], int(pilot_protocol["budget"]), int(seed))
                      for seed in pilot_protocol["trajectory_seeds"]}
    if pilot_ids != expected_pilot:
        raise RuntimeError("The five source-only pilot trajectories are not all complete")

    semantic_path = PACKAGE / "runs/source_only/batch_semantic.parquet"
    if not semantic_path.exists():
        raise RuntimeError("Complete B=25 Batch Semantic ablation runs are required")
    semantic = pd.read_parquet(semantic_path)
    if semantic.groupby("experiment_id").size().ne(25).any() or semantic.experiment_id.nunique() != 30:
        raise RuntimeError("Batch Semantic runs do not cover the 30 complete B=25 trajectories")
    semantic_ids = set()
    for row in semantic.itertuples(index=False):
        semantic_ids.add((str(row.held_site), int(row.seed)))
        membership = f"batch_semantic:{row.experiment_id}:query{row.candidate_index}"
        if bool(row.selected_by_robust):
            membership += ":robust_selected"
        _source_record(candidates, row.held_site, row.quartet_key, row.mean_source_auc,
                       row.min_source_auc, row.direction_gap, row.selected_C, list(row.head_ids), membership)
    expected_semantic_ids = {(site, int(seed))
                             for site in json.loads((PACKAGE / "data/folds.json").read_text())["site_order"]
                             for seed in protocol["trajectory_seeds"]}
    if semantic_ids != expected_semantic_ids:
        raise RuntimeError("Batch Semantic output is missing held-site/seed pairs")

    # Random-prefix selectors use only candidates already in the prospective bank.
    random_selection_path = PACKAGE / "runs/source_only/random_prefix_selections.csv"
    if not random_selection_path.exists():
        raise RuntimeError("Random-prefix source-only selector results are missing")
    selections = pd.read_csv(random_selection_path, dtype={"quartet_key": str})
    expected_selection_rows = 100 * 6 * 3 * 2
    if len(selections) != expected_selection_rows:
        raise RuntimeError(f"Expected {expected_selection_rows} random-prefix selector rows, found {len(selections)}")
    for row in selections.itertuples(index=False):
        key = (str(row.held_site), str(row.quartet_key))
        if key not in candidates:
            raise ValueError("Random-prefix selection is absent from the prospective random bank")
        candidates[key]["memberships"].append(
            f"{row.method}:repeat{int(row.repeat)}:B{int(row.budget)}:selected")

    for key, row in candidates.items():
        if len(row["head_ids"]) != 4 or len(set(row["head_ids"])) != 4 or set(row["head_ids"]).difference(catalog_ids):
            raise ValueError(f"Invalid quartet in candidate manifest: {key}")
        if row["quartet_key"] in historical:
            raise ValueError(f"Historical target-inspected quartet entered the new search: {key}")
        if not 1e-4 <= float(row["selected_C"]) <= 1e4:
            raise ValueError(f"selected_C outside the frozen regularization grid for {key}")
        row["memberships"] = sorted(set(row["memberships"]))
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": freeze["protocol_sha256"],
        "prompt_sha256": freeze["prompt_sha256"],
        "execution_sha256": freeze["execution_sha256"],
        "input_sha256": freeze["input_sha256"],
        "candidate_rule": "every unique source-scored candidate/fold pair in prospective bank, classical baselines, full agents, and ablations; pilot excluded",
        "candidate_count": len(candidates),
        "candidates": [candidates[key] for key in sorted(candidates)],
    }
    manifest = {**payload, "manifest_sha256": _hash_json(payload)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    (PACKAGE / "runs/frozen_manifest.sha256").write_text(manifest["manifest_sha256"] + "  frozen_manifest.json\n", encoding="utf-8")
    return {"candidate_count": len(candidates), "manifest_sha256": manifest["manifest_sha256"],
            "protocol_sha256": manifest["protocol_sha256"], "prompt_sha256": manifest["prompt_sha256"]}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()
