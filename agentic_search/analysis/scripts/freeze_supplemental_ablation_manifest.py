"""Freeze a separate manifest for completed post-freeze B=10/50 ablation runs."""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / "runs/supplemental/held_site"


def digest(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main(manifest_name: str = "ablation_manifest.json") -> None:
    catalog = set(pd.read_csv(PACKAGE / "data/heads.csv", dtype=str).head_id)
    historical = set(pd.read_csv(PACKAGE / "data/historical_quartets.csv", dtype=str).quartet_key)
    candidates: dict[tuple[str, str], dict] = {}
    parent_hash = None
    if manifest_name != "ablation_manifest.json":
        parent_path = OUT / "ablation_manifest.json"
        if not parent_path.is_file():
            raise FileNotFoundError("Versioned ablation manifest requires the first frozen manifest")
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        parent_hash = parent["manifest_sha256"]
        for row in parent["candidates"]:
            candidates[(str(row["held_site"]), str(row["quartet_key"]))] = dict(row)
    included = {"batch_semantic": 0, "opaque_iterative": 0}
    for method in included:
        for path in sorted((PACKAGE / "runs/supplemental" / method).glob("**/*.json")):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            budget = int(state.get("budget", 0))
            history = state.get("history", [])
            if budget not in (10, 50) or len(history) != budget:
                continue
            site = str(state.get("held_site_internal", ""))
            if site not in {"I0002", "I0006", "S0001"}:
                raise ValueError(f"Unexpected held site in {path}")
            seed = int(state["seed"])
            for row in history:
                ids = sorted(map(str, row.get("real_head_ids") or row["head_ids"]))
                key = "|".join(ids)
                if len(ids) != 4 or len(set(ids)) != 4 or set(ids) - catalog:
                    raise ValueError(f"Invalid quartet in {path}: {ids}")
                if key in historical:
                    raise ValueError(f"Historical excluded quartet in {path}: {key}")
                identity = (site, key)
                membership = f"{method}:B{budget}:seed{seed}"
                row_out = {"held_site": site, "quartet_key": key, "head_ids": ids,
                           "selected_C": float(row["selected_C"]), "memberships": [membership]}
                if identity in candidates:
                    prior = candidates[identity]
                    if prior["selected_C"] != row_out["selected_C"]:
                        raise ValueError(f"Source-selected C differs for duplicate {identity}")
                    prior["memberships"].append(membership)
                else:
                    candidates[identity] = row_out
            included[method] += 1
    for row in candidates.values():
        row["memberships"] = sorted(set(row["memberships"]))
    freeze = json.loads((PACKAGE / "runs/protocol_freeze.json").read_text(encoding="utf-8"))
    payload = {
        "manifest_type": "append-only supplemental ablation candidate manifest",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "primary_manifest_sha256": json.loads((PACKAGE / "runs/frozen_manifest.json").read_text())["manifest_sha256"],
        "protocol_sha256": freeze["protocol_sha256"],
        "prompt_sha256": freeze["prompt_sha256"],
        "included_complete_trajectories": included,
        "candidate_count": len(candidates),
        "candidate_rule": "every source-evaluated quartet in each complete supplemental batch-semantic or opaque-iterative B=10/B=50 trajectory; failed or partial trajectories are excluded from the analysis manifest and remain in raw logs",
        "candidates": [candidates[key] for key in sorted(candidates)],
    }
    if parent_hash is not None:
        payload["parent_manifest_sha256"] = parent_hash
    manifest = {**payload, "manifest_sha256": digest(payload)}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / manifest_name
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / f"{Path(manifest_name).stem}.sha256").write_text(manifest["manifest_sha256"] + f"  {manifest_name}\n")
    print(json.dumps({"path": str(path), "candidate_count": len(candidates),
                      "included_complete_trajectories": included,
                      "manifest_sha256": manifest["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="ablation_manifest.json")
    main(parser.parse_args().manifest)
