"""Run the B=25 semantic, no-feedback ablation."""
from __future__ import annotations

import argparse
import fcntl
import json
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from agentic_search.agents.model_cli import call_static
from agentic_search.evaluator.source_evaluator import evaluate_source


PACKAGE = Path(__file__).resolve().parents[1]


def _validate(candidates: Any, ids: set[str], forbidden: set[str]) -> tuple[list[list[str] | None], list[str]]:
    errors = []
    values = candidates if isinstance(candidates, list) else []
    if len(values) != 25:
        errors.append(f"Expected exactly 25 candidates, got {len(values)}")
    normalized: list[list[str] | None] = []
    seen = set()
    for index in range(25):
        if index >= len(values):
            normalized.append(None)
            continue
        raw = values[index]
        reason = []
        if not isinstance(raw, list) or len(raw) != 4:
            normalized.append(None)
            errors.append(f"candidate {index + 1}: exactly four IDs required")
            continue
        quartet = sorted(str(value) for value in raw)
        if len(set(quartet)) != 4:
            reason.append("duplicate IDs")
        if set(quartet).difference(ids):
            reason.append("unknown head ID")
        key = "|".join(quartet)
        if key in seen:
            reason.append("duplicate quartet")
        if key in forbidden:
            reason.append("historically target-inspected quartet")
        if reason:
            normalized.append(None)
            errors.append(f"candidate {index + 1}: " + "; ".join(reason))
        else:
            normalized.append(quartet)
            seen.add(key)
    return normalized, errors


def _catalog_prompt(catalog: pd.DataFrame) -> str:
    return "\n".join(f"{row.head_id}\t{row.head_name}\t{row.taxonomy_category}"
                     for row in catalog.itertuples(index=False))


def run_one(provider: str, model: str, held_site: str, seed: int) -> list[dict[str, Any]]:
    catalog = pd.read_csv(PACKAGE / "data/heads.csv", dtype=str)
    ids = set(catalog.head_id)
    forbidden = set(pd.read_csv(PACKAGE / "data/historical_quartets.csv", dtype=str).quartet_key)
    protocol = yaml.safe_load((PACKAGE / "configs/protocol.yaml").read_text(encoding="utf-8"))
    freeze = json.loads((PACKAGE / "runs/protocol_freeze.json").read_text(encoding="utf-8"))
    prompt_file = PACKAGE / "configs/prompts/batch_semantic.txt"
    system = prompt_file.read_text(encoding="utf-8")
    schema = {"type": "object", "required": ["candidates"], "additionalProperties": False,
              "properties": {"candidates": {"type": "array", "minItems": 25, "maxItems": 25,
                  "items": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "string"}}}}}
    experiment_id = f"batch_semantic__{provider}-{model}__{held_site}__B25__seed{seed}"
    state_path = PACKAGE / f"runs/source_only/batch_semantic/{provider}/{held_site}/seed{seed}.json"
    if state_path.exists():
        old = json.loads(state_path.read_text(encoding="utf-8"))
        if len(old.get("history", [])) == 25:
            return old["history"]
        raise ValueError(f"Partial batch-semantic trajectory exists: {state_path}")

    catalogue = _catalog_prompt(catalog)
    prompt = (f"Candidate budget: 25. Before receiving any performance results, generate 25 distinct quartets.\n"
              f"Frozen catalog (head ID, name, pre-existing family):\n{catalogue}\n"
              "Use only IDs from this catalog. Do not infer or request held-site information. "
              "Return the 25 quartets in the required JSON shape.")
    call = call_static(provider, model, system, prompt, schema,
                       seed=int(seed) + zlib.crc32(held_site.encode()),
                       reasoning_effort=protocol.get("reasoning_effort"))
    candidates, errors = _validate(call["answer"].get("candidates"), ids, forbidden)
    invalid_attempts = []
    if errors:
        invalid_attempts.append({"errors": errors, "attempt": 1})
        repair = (f"Your previous list was invalid. Errors: {json.dumps(errors)}\n"
                  f"Previous proposal: {json.dumps(call['answer'].get('candidates'))}\n"
                  f"Catalog (ID, name, category):\n{catalogue}\n"
                  "Return exactly 25 distinct, valid quartets and repair all errors. Use no held-site data.")
        repaired_call = call_static(provider, model, system, repair, schema,
                                     seed=int(seed) + zlib.crc32(held_site.encode()) + 1)
        candidates, errors = _validate(repaired_call["answer"].get("candidates"), ids, forbidden)
        if errors:
            invalid_attempts.append({"errors": errors, "attempt": 2})
            rng = np.random.default_rng(seed + zlib.crc32(held_site.encode()))
            seen = {"|".join(row) for row in candidates if row is not None}
            for index, candidate in enumerate(candidates):
                if candidate is not None:
                    continue
                for _ in range(100_000):
                    random_candidate = sorted(rng.choice(sorted(ids), size=4, replace=False).tolist())
                    key = "|".join(random_candidate)
                    if key not in forbidden and key not in seen:
                        candidates[index] = random_candidate
                        seen.add(key)
                        break
                if candidates[index] is None:
                    raise RuntimeError("Could not fill an invalid semantic-ablation slot")
            repair_metadata = repaired_call["metadata"]
            auto_repair = True
        else:
            repair_metadata = repaired_call["metadata"]
            auto_repair = False
    else:
        repair_metadata = None
        auto_repair = False
    if len(candidates) != 25 or any(value is None for value in candidates):
        raise AssertionError("Batch-semantic validator failed to produce 25 valid quartets")

    history = []
    for index, quartet in enumerate(candidates, start=1):
        result = evaluate_source(held_site, quartet, cache_path=PACKAGE / "cache/source.sqlite")
        history.append({"candidate_index": index, "head_ids": quartet,
                        "quartet_key": "|".join(quartet),
                        "mean_source_auc": result["mean_source_auc"],
                        "min_source_auc": result["min_source_auc"],
                        "direction_gap": result["direction_gap"], "selected_C": result["selected_C"],
                        "auto_repair": bool(auto_repair)})
    selected = sorted(history, key=lambda row: (-float(row["min_source_auc"]), row["quartet_key"]))[0]
    for row in history:
        row["selected_by_robust"] = row["quartet_key"] == selected["quartet_key"]
    state = {"experiment_id": experiment_id, "method": "batch_semantic", "provider": provider,
             "model_requested": model, "model_reported": call["metadata"].get("model_reported"),
             "held_site_internal": held_site, "budget": 25, "seed": int(seed),
             "budget_used": 25, "history": history, "invalid_attempts": invalid_attempts,
             "auto_repair": auto_repair, "robust_selected_quartet_key": selected["quartet_key"],
             "temperature_requested": float(protocol["agent_temperature"]),
             "temperature_controlled": bool(call["metadata"].get("temperature_controlled", False)),
             "temperature_applied": call["metadata"].get("temperature"),
             "reasoning_effort": call["metadata"].get("reasoning_effort"),
             "protocol_sha256": freeze["protocol_sha256"],
             "prompt_sha256": __import__("hashlib").sha256(prompt_file.read_bytes()).hexdigest(),
             "provider_calls": [call["metadata"]] + ([repair_metadata] if repair_metadata else []),
             "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    destination = PACKAGE / "runs/source_only/batch_semantic.parquet"
    with destination.with_suffix(".lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        previous = pd.read_parquet(destination).to_dict("records") if destination.exists() else []
        previous = [row for row in previous if row.get("experiment_id") != experiment_id]
        rows = [{**row, "experiment_id": experiment_id, "method": "batch_semantic",
                 "provider": provider, "model_requested": model, "held_site": held_site, "seed": seed}
                for row in history]
        temporary = destination.with_suffix(".tmp.parquet")
        pd.DataFrame(previous + rows).sort_values(["experiment_id", "candidate_index"]).to_parquet(temporary, index=False)
        temporary.replace(destination)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return history


def run_grid(provider: str, model: str, sites: list[str] | None = None,
             seeds: list[int] | None = None, workers: int = 1) -> None:
    protocol = yaml.safe_load((PACKAGE / "configs/protocol.yaml").read_text())
    folds = json.loads((PACKAGE / "data/folds.json").read_text())
    jobs = [(held_site, int(seed)) for held_site in sites or folds["site_order"]
            for seed in seeds or protocol["trajectory_seeds"]]
    if int(workers) <= 1:
        for held_site, seed in jobs:
            run_one(provider, model, held_site, seed)
            print(f"Batch semantic: {held_site} seed {seed}/25 source evaluations", flush=True)
        return
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(run_one, provider, model, held_site, seed): (held_site, seed)
                   for held_site, seed in jobs}
        for future in as_completed(futures):
            held_site, seed = futures[future]
            future.result()
            print(f"Batch semantic: {held_site} seed {seed}/25 source evaluations", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("codex", "claude", "ollama"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--site", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent semantic ablations to run concurrently")
    args = parser.parse_args()
    run_grid(args.provider, args.model, args.site, args.seed, args.workers)


if __name__ == "__main__":
    main()
