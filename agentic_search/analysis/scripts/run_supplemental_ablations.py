"""Run missing B=10 and B=50 source-only ablation trajectories.

This leaves the frozen primary runner and B=25 ablation artifacts untouched.
All supplemental output is stored under ``runs/supplemental``.
"""
from __future__ import annotations

import argparse
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
from agentic_search.agents.run_trajectories import run_trajectory
from agentic_search.evaluator.source_evaluator import evaluate_source


PACKAGE = Path(__file__).resolve().parents[2]
SITES = ("I0002", "I0006", "S0001")
SEEDS = tuple(range(1000, 1010))


def _validate(candidates: Any, budget: int, ids: set[str], forbidden: set[str]):
    errors: list[str] = []
    values = candidates if isinstance(candidates, list) else []
    if len(values) != budget:
        errors.append(f"Expected exactly {budget} candidates, got {len(values)}")
    normalized: list[list[str] | None] = []
    seen: set[str] = set()
    for index in range(budget):
        if index >= len(values):
            normalized.append(None)
            continue
        raw = values[index]
        if not isinstance(raw, list) or len(raw) != 4:
            normalized.append(None)
            errors.append(f"candidate {index + 1}: exactly four IDs required")
            continue
        quartet = sorted(map(str, raw))
        reasons = []
        if len(set(quartet)) != 4:
            reasons.append("duplicate IDs")
        if set(quartet).difference(ids):
            reasons.append("unknown head ID")
        key = "|".join(quartet)
        if key in seen:
            reasons.append("duplicate quartet")
        if key in forbidden:
            reasons.append("historically target-inspected quartet")
        if reasons:
            normalized.append(None)
            errors.append(f"candidate {index + 1}: " + "; ".join(reasons))
        else:
            normalized.append(quartet)
            seen.add(key)
    return normalized, errors


def run_batch_semantic(site: str, seed: int, budget: int) -> dict[str, Any]:
    root = PACKAGE / "runs/supplemental/batch_semantic"
    state_path = root / site / f"B{budget}_seed{seed}.json"
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        if saved.get("completed_at_utc") and saved.get("budget_used") == budget:
            return saved
        raise RuntimeError(f"Partial semantic trajectory exists; inspect before resume: {state_path}")

    catalog = pd.read_csv(PACKAGE / "data/heads.csv", dtype=str)
    ids = set(catalog.head_id.astype(str))
    forbidden = set(pd.read_csv(PACKAGE / "data/historical_quartets.csv", dtype=str).quartet_key.astype(str))
    protocol = yaml.safe_load((PACKAGE / "configs/protocol.yaml").read_text())
    freeze = json.loads((PACKAGE / "runs/protocol_freeze.json").read_text())
    prompt_path = PACKAGE / "configs/prompts/batch_semantic.txt"
    system = prompt_path.read_text()
    schema = {"type": "object", "required": ["candidates"], "additionalProperties": False,
              "properties": {"candidates": {"type": "array", "minItems": budget, "maxItems": budget,
                  "items": {"type": "array", "minItems": 4, "maxItems": 4,
                            "items": {"type": "string"}}}}}
    catalogue = "\n".join(f"{row.head_id}\t{row.head_name}\t{row.taxonomy_category}"
                         for row in catalog.itertuples(index=False))
    base_seed = int(seed) + zlib.crc32(site.encode())
    user = (f"Candidate budget: {budget}. Before receiving any performance results, generate {budget} distinct quartets.\n"
            f"Frozen catalog (head ID, name, pre-existing family):\n{catalogue}\n"
            "Use only IDs from this catalog. Do not infer or request held-site information. "
            "Return the quartets in the required JSON shape.")
    calls = []
    proposal = call_static("codex", "gpt-6-sol", system, user, schema, seed=base_seed,
                           reasoning_effort=protocol.get("reasoning_effort"))
    candidates, errors = _validate(proposal["answer"].get("candidates"), budget, ids, forbidden)
    if errors:
        calls.append({"attempt": 1, "errors": errors, "metadata": proposal["metadata"]})
        repair = (f"Your previous list was invalid. Errors: {json.dumps(errors)}\n"
                  f"Previous proposal: {json.dumps(proposal['answer'].get('candidates'))}\n"
                  f"Catalog (ID, name, category):\n{catalogue}\n"
                  f"Return exactly {budget} distinct, valid quartets and repair all errors. Use no held-site data.")
        repair_call = call_static("codex", "gpt-6-sol", system, repair, schema, seed=base_seed + 1,
                                  reasoning_effort=protocol.get("reasoning_effort"))
        candidates, repair_errors = _validate(repair_call["answer"].get("candidates"), budget, ids, forbidden)
        calls.append({"attempt": 2, "errors": repair_errors, "metadata": repair_call["metadata"]})
        if repair_errors:
            rng = np.random.default_rng(base_seed)
            seen = {"|".join(x) for x in candidates if x is not None}
            for index, candidate in enumerate(candidates):
                if candidate is not None:
                    continue
                for _ in range(100_000):
                    value = sorted(rng.choice(sorted(ids), size=4, replace=False).tolist())
                    key = "|".join(value)
                    if key not in forbidden and key not in seen:
                        candidates[index] = value
                        seen.add(key)
                        break
                if candidates[index] is None:
                    raise RuntimeError("Unable to fill invalid semantic proposal slots")
            chosen_call = repair_call
            auto_repair = True
        else:
            chosen_call = repair_call
            auto_repair = False
    else:
        chosen_call = proposal
        auto_repair = False

    history = []
    for ix, quartet in enumerate(candidates, start=1):
        score = evaluate_source(site, quartet, cache_path=PACKAGE / "cache/source.sqlite")
        history.append({"candidate_index": ix, "head_ids": quartet,
                        "quartet_key": "|".join(quartet), **score,
                        "auto_repair": bool(auto_repair)})
    selected = sorted(history, key=lambda row: (-float(row["min_source_auc"]), row["quartet_key"]))[0]
    for row in history:
        row["selected_by_robust"] = row["quartet_key"] == selected["quartet_key"]
    state = {"experiment_id": f"batch_semantic__codex-gpt-6-sol__{site}__B{budget}__seed{seed}",
             "method": "batch_semantic", "provider": "codex", "model_requested": "gpt-6-sol",
             "model_reported": chosen_call["metadata"].get("model_reported"),
             "held_site_internal": site, "budget": budget, "seed": seed, "budget_used": budget,
             "history": history, "invalid_attempts": calls, "auto_repair": auto_repair,
             "robust_selected_quartet_key": selected["quartet_key"],
             "reasoning_effort": chosen_call["metadata"].get("reasoning_effort", protocol.get("reasoning_effort")),
             "protocol_sha256": freeze["protocol_sha256"],
             "prompt_sha256": __import__("hashlib").sha256(prompt_path.read_bytes()).hexdigest(),
             "provider_calls": [proposal["metadata"]] + ([repair_call["metadata"]] if errors else []),
             "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return state


def run_one(method: str, site: str, budget: int, seed: int) -> dict[str, Any]:
    if method == "batch_semantic":
        return run_batch_semantic(site, seed, budget)
    return run_trajectory(provider="codex", model="gpt-6-sol", held_site=site,
                          budget=budget, seed=seed, method="opaque_iterative",
                          output_dir=PACKAGE / "runs/supplemental")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("batch_semantic", "opaque_iterative"), required=True)
    parser.add_argument("--budget", choices=(10, 50), type=int, action="append", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = [(site, budget, seed) for budget in args.budget for site in SITES for seed in SEEDS]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, args.method, site, budget, seed): (site, budget, seed)
                   for site, budget, seed in jobs}
        for future in as_completed(futures):
            site, budget, seed = futures[future]
            result = future.result()
            print(f"{args.method} B={budget} {site} seed={seed}: "
                  f"{result.get('budget_used')} source candidates complete", flush=True)


if __name__ == "__main__":
    main()
