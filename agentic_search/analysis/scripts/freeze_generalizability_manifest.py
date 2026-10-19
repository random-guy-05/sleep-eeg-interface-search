"""Freeze an append-only source manifest for all usable controller candidates."""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE / "runs/supplemental/generalizability"
OUT = PACKAGE / "runs/supplemental/held_site"


def _hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main(manifest_name: str = "generalizability_manifest.json") -> None:
    catalog = set(pd.read_csv(PACKAGE / "data/heads.csv", dtype=str).head_id)
    history = set(pd.read_csv(PACKAGE / "data/historical_quartets.csv", dtype=str).quartet_key)
    candidates, run_inventory, invalid_records = {}, [], []
    parent_hash = None
    if manifest_name != "generalizability_manifest.json":
        parent_path = OUT / "generalizability_manifest.json"
        if not parent_path.is_file():
            raise FileNotFoundError("Versioned generalizability manifest requires the first frozen manifest")
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        parent_hash = parent["manifest_sha256"]
        for row in parent["candidates"]:
            candidates[(str(row["held_site"]), str(row["quartet_key"]))] = dict(row)
    for model in ("luna", "opus"):
        for path in sorted((ROOT / model).glob("**/*.json")):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(state, dict):
                continue
            phase = str(state.get("supplemental_phase", ""))
            if phase not in {"main", "pilot"}:
                # zero-shot JSON records have no phase field
                if "/zero_shot/" in path.as_posix():
                    phase = "zero-shot"
                elif int(state.get("seed", -1)) in range(1000, 1010) and int(state.get("budget", 0)) in (10, 25, 50):
                    # Failed main runs may have a fully source-scored history but
                    # no final-choice response, so the runner never stamps phase.
                    phase = "main"
                else:
                    continue
            site = str(state.get("held_site_internal", state.get("held_site", "")))
            if site not in {"I0002", "I0006", "S0001"}:
                continue
            budget = int(state.get("budget", 0))
            seed = int(state.get("seed", -1))
            complete = False
            added = 0
            if phase == "zero-shot":
                key = str(state.get("quartet_key", ""))
                ids = sorted(map(str, state.get("head_ids", key.split("|"))))
                c = float(state["selected_C"])
                entries = [(ids, c, "zero_shot")]
                complete = True
            else:
                rows = state.get("history", [])
                complete = bool(budget and len(rows) == budget)
                entries = []
                for row in rows:
                    ids = sorted(map(str, row.get("real_head_ids") or row["head_ids"]))
                    entries.append((ids, float(row["selected_C"]), f"{phase}:B{budget}"))
                for attempt in state.get("invalid_attempts", []):
                    invalid_records.append({"model": model, "phase": phase, "site": site,
                                            "budget": budget, "seed": seed, "diagnostic": attempt})
            for ids, c, condition in entries:
                key = "|".join(ids)
                if len(ids) != 4 or len(set(ids)) != 4 or set(ids) - catalog:
                    raise ValueError(f"Invalid source proposal in {path}: {ids}")
                if key in history:
                    raise ValueError(f"Forbidden historical candidate in {path}: {key}")
                ident = (site, key)
                row = candidates.get(ident)
                member = {"model": model, "phase": phase, "condition": condition,
                          "budget": budget, "seed": seed,
                          "selection_role": ("model_final_choice" if state.get("agent_choice_valid") and
                                             state.get("agent_choice_quartet_key") == key else
                                             "zero_shot_pick" if phase == "zero-shot" else
                                             "source_robust_candidate" if state.get("robust_selected_quartet_key") == key else "queried_candidate")}
                if row is None:
                    candidates[ident] = {"held_site": site, "quartet_key": key, "head_ids": ids,
                                         "selected_C": c, "memberships": [member]}
                else:
                    if abs(float(row["selected_C"]) - c) > 1e-12:
                        raise ValueError(f"Source-selected C differs for duplicate {ident}")
                    row["memberships"].append(member)
                added += 1
            run_inventory.append({"model": model, "phase": phase, "site": site, "budget": budget,
                                  "seed": seed, "source_budget_complete": complete,
                                  "valid_final_choice": bool(state.get("agent_choice_valid", phase == "zero-shot")),
                                  "source_candidate_occurrences": added,
                                  "invalid_batch_events": len(state.get("invalid_attempts", []))})
    for row in candidates.values():
        row["memberships"] = sorted(row["memberships"], key=lambda x: (x["model"], x["phase"], x["budget"], x["seed"]))
    failure_path = ROOT / "failures.jsonl"
    provider_failures = []
    if failure_path.exists():
        for line in failure_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("phase") in {"main", "zero-shot", "pilot"} and entry.get("model") in {"luna", "opus"}:
                provider_failures.append(entry)
    freeze = json.loads((PACKAGE / "runs/protocol_freeze.json").read_text(encoding="utf-8"))
    primary = json.loads((PACKAGE / "runs/frozen_manifest.json").read_text(encoding="utf-8"))
    payload = {"manifest_type": "append-only generalizability candidate manifest",
               "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
               "primary_manifest_sha256": primary["manifest_sha256"],
               "protocol_sha256": freeze["protocol_sha256"],
               "prompt_sha256": freeze["prompt_sha256"],
               "candidate_rule": "every source-scored quartet in retained Luna and Opus pilot/main/zero-shot artifacts; failed calls without accepted source-scored candidates remain in raw failure logs",
               "candidate_count": len(candidates), "run_inventory": run_inventory,
               "invalid_batch_diagnostics": invalid_records, "provider_failures": provider_failures,
               "candidates": [candidates[k] for k in sorted(candidates)]}
    if parent_hash is not None:
        payload["parent_manifest_sha256"] = parent_hash
    manifest = {**payload, "manifest_sha256": _hash(payload)}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / manifest_name
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / f"{Path(manifest_name).stem}.sha256").write_text(manifest["manifest_sha256"] + f"  {manifest_name}\n")
    print(json.dumps({"manifest": str(path), "candidate_count": len(candidates),
                      "model_phase_runs": {k: sum(r['model']==k for r in run_inventory) for k in ('luna','opus')},
                      "manifest_sha256": manifest["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="generalizability_manifest.json")
    main(parser.parse_args().manifest)
