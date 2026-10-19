"""Automated target-blindness, freeze, candidate, and scoring integrity checks."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from agentic_search.evaluator.source_evaluator import C_GRID
from agentic_search.freeze_protocol import current_execution_hashes, current_input_hashes


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent


def _verify_manifest(path: Path) -> tuple[dict, datetime]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if actual != manifest.get("manifest_sha256"):
        raise AssertionError("Frozen manifest content hash does not match")
    created = datetime.fromisoformat(str(manifest["created_at_utc"]).replace("Z", "+00:00"))
    return manifest, created


def _file_datetime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _check_quartet(ids: list[str], catalog_ids: set[str], location: str) -> None:
    normalized = list(map(str, ids))
    if len(normalized) != 4 or len(set(normalized)) != 4:
        raise AssertionError(f"Candidate must contain four unique heads: {location}")
    if set(normalized).difference(catalog_ids):
        raise AssertionError(f"Candidate contains unknown head IDs: {location}")


def _in_grid(value: float) -> bool:
    return any(np.isclose(float(value), candidate, rtol=1e-8, atol=1e-12) for candidate in C_GRID)


def _scan_target_blind_files() -> dict[str, int]:
    agent_files = [PACKAGE / "agents/mcp_server.py", PACKAGE / "agents/model_cli.py",
                   PACKAGE / "agents/run_trajectories.py", PACKAGE / "agents/run_batch_semantic.py",
                   PACKAGE / "evaluator/source_evaluator.py"]
    forbidden_imports = ("target_evaluator", "target_labels.parquet", "new_target_results.parquet")
    for path in agent_files:
        content = path.read_text(encoding="utf-8")
        if any(value in content for value in forbidden_imports):
            raise AssertionError(f"Source-only agent path references target-evaluation data: {path}")
    feature_table = pd.read_parquet(PACKAGE / "data/participants.parquet")
    if "outcome" in feature_table.columns:
        raise AssertionError("Agent-visible participant feature table contains outcomes")
    fold_sites = json.loads((PACKAGE / "data/folds.json").read_text(encoding="utf-8"))["site_order"]
    for held_site in fold_sites:
        fold = pd.read_parquet(PACKAGE / f"data/source_folds/{held_site}.parquet")
        if "outcome" not in fold or held_site in set(fold.site.astype(str)):
            raise AssertionError(f"Agent-visible source fold includes held-site outcomes: {held_site}")
        if fold.site.nunique() != 2:
            raise AssertionError(f"Source fold has other than two institutions: {held_site}")
    literal_targets = ("held_auc", "target_auc", "target_rank", f"{fold_sites[-1]} target")
    source_only = PACKAGE / "runs/source_only/agents"
    scanned_logs = 0
    if source_only.exists():
        for path in source_only.glob("**/*"):
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".txt"}:
                continue
            try:
                content = path.read_text(encoding="utf-8").casefold()
            except UnicodeDecodeError:
                continue
            scanned_logs += 1
            forbidden = [value for value in literal_targets if value.casefold() in content]
            if forbidden:
                raise AssertionError(f"Target-result string appeared in an agent input/output log ({path}): {forbidden}")
    return {"agent_source_files_scanned": len(agent_files), "source_only_logs_scanned": scanned_logs}


def _load_primary_trajectories() -> list[tuple[Path, dict]]:
    root = PACKAGE / "runs/source_only/agents"
    outputs = []
    if root.exists():
        for path in sorted(root.glob("**/*.json")):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if state.get("method") in {"full_agent", "opaque_iterative"}:
                outputs.append((path, state))
    return outputs


def _check_preprocessing_fit_scope() -> None:
    target_path = PACKAGE / "evaluator/target_evaluator.py"
    target_text = target_path.read_text(encoding="utf-8")
    if "fit_probe(" not in target_text or "target_rows_used" not in target_text:
        raise AssertionError("Target evaluator does not expose its source-only preprocessing assertion")
    if "age_conditioned_auroc" not in target_text:
        raise AssertionError("Target evaluator does not use the shared age-conditioned AUROC")
    from brainbodyfm import tuning
    import inspect

    tuning_text = inspect.getsource(tuning)
    if "age_conditioned_auroc" not in tuning_text:
        raise AssertionError("Source-side tuning does not use the shared age-conditioned AUROC")


def audit() -> dict:
    protocol_path = PACKAGE / "configs/protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    freeze = json.loads((PACKAGE / "runs/protocol_freeze.json").read_text(encoding="utf-8"))
    if freeze.get("phase") != "final":
        raise AssertionError("Final protocol and prompts are not frozen")
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if protocol_hash != freeze.get("protocol_sha256"):
        raise AssertionError("The frozen protocol hash differs from the current protocol")
    for name, expected in freeze.get("prompt_sha256", {}).items():
        path = PACKAGE / f"configs/prompts/{name}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"A prompt changed after the final freeze: {name}")
    if current_execution_hashes() != freeze.get("execution_sha256"):
        raise AssertionError("Executable pipeline code differs from the final frozen code")
    if current_input_hashes() != freeze.get("input_sha256"):
        raise AssertionError("Source-only input data differs from the final frozen inputs")
    blind_counts = _scan_target_blind_files()
    _check_preprocessing_fit_scope()

    catalog_ids = set(pd.read_csv(PACKAGE / "data/heads.csv", dtype=str).head_id)
    historical = set(pd.read_csv(PACKAGE / "data/historical_quartets.csv", dtype=str).quartet_key)
    source_path = PACKAGE / "runs/source_only/prospective_random500.parquet"
    source_bank = pd.read_parquet(source_path)
    if len(source_bank) != 1500:
        raise AssertionError(f"Expected 1,500 source-only random-bank rows; got {len(source_bank)}")
    if any("target" in str(column).casefold() for column in source_bank.columns):
        raise AssertionError("The source-only random bank contains a target-named column")
    source_records = 0
    for row in source_bank.itertuples(index=False):
        ids = list(row.head_ids)
        _check_quartet(ids, catalog_ids, f"prospective bank/{row.held_site}/{row.quartet_key}")
        if row.quartet_key in historical:
            raise AssertionError(f"Historical quartet was evaluated prospectively: {row.quartet_key}")
        if not _in_grid(float(row.selected_C)):
            raise AssertionError(f"Source C is off-grid: {row.selected_C}")
        source_records += 1

    trajectories = _load_primary_trajectories()
    for path, state in trajectories:
        budget = int(state["budget"])
        if int(state.get("budget_used", -1)) != budget or len(state.get("history", [])) != budget:
            raise AssertionError(f"Agent trajectory budget is incomplete or exceeded: {path}")
        keys = []
        for item in state["history"]:
            ids = item.get("real_head_ids") or item.get("head_ids")
            key = item.get("real_quartet_key") or "|".join(sorted(map(str, ids)))
            _check_quartet(list(ids), catalog_ids, f"{path}/{key}")
            if key in historical:
                raise AssertionError(f"Agent queried a historically target-inspected quartet: {path}/{key}")
            if not _in_grid(float(item["selected_C"])):
                raise AssertionError(f"Agent source C is off-grid: {path}/{key}")
            keys.append(key)
        if len(keys) != len(set(keys)):
            raise AssertionError(f"Agent repeated a quartet within one trajectory: {path}")
        if state.get("protocol_sha256") != protocol_hash:
            raise AssertionError(f"Agent trajectory used a different protocol version: {path}")

    # Manifest presence and integrity are required before the held labels/results can be read.
    manifest_path = PACKAGE / "runs/frozen_manifest.json"
    if manifest_path.exists():
        manifest, manifest_time = _verify_manifest(manifest_path)
        if manifest.get("execution_sha256") != freeze.get("execution_sha256"):
            raise AssertionError("Manifest execution-code hashes differ from the final freeze")
        if manifest.get("input_sha256") != freeze.get("input_sha256"):
            raise AssertionError("Manifest source-input hashes differ from the final freeze")
        for record in manifest["candidates"]:
            ids = list(record["head_ids"])
            _check_quartet(ids, catalog_ids, f"manifest/{record['held_site']}/{record['quartet_key']}")
            if record["quartet_key"] in historical:
                raise AssertionError(f"Historical quartet entered the manifest: {record['quartet_key']}")
            if not _in_grid(float(record["selected_C"])):
                raise AssertionError(f"Manifest C is off-grid: {record['selected_C']}")
        if (PACKAGE / "sealed/new_target_results.parquet").exists():
            target_path = PACKAGE / "sealed/new_target_results.parquet"
            if _file_datetime(target_path) < manifest_time:
                raise AssertionError("Target results predate the frozen candidate manifest")
            targets = pd.read_parquet(target_path)
            if len(targets) != int(manifest["candidate_count"]):
                raise AssertionError("Target result row count differs from the frozen manifest")
            manifest_keys = {(row["held_site"], row["quartet_key"]): row for row in manifest["candidates"]}
            for row in targets.itertuples(index=False):
                key = (str(row.held_site), str(row.quartet_key))
                if key not in manifest_keys:
                    raise AssertionError(f"Unfrozen candidate appears in target results: {key}")
                expected_c = float(manifest_keys[key]["selected_C"])
                if not np.isclose(float(row.selected_C), expected_c, rtol=0, atol=1e-12):
                    raise AssertionError(f"Target evaluator used a C different from the frozen source C: {key}")
                if int(row.target_labels_used_for_model_or_C_selection) != 0:
                    raise AssertionError(f"Target outcomes were used for model/C selection: {key}")
            release = PACKAGE / "sealed/target_scoring_release.json"
            if not release.is_file() or _file_datetime(release) < manifest_time:
                raise AssertionError("Target release record is absent or predates the manifest")
        if (PACKAGE / "sealed/new_target_results_progress.parquet").exists():
            if _file_datetime(PACKAGE / "sealed/new_target_results_progress.parquet") < manifest_time:
                raise AssertionError("Partial target results predate the frozen manifest")
    else:
        manifest_time = None

    _check_preprocessing_fit_scope()
    return {"status": "passed", "source_bank_rows": source_records,
            "primary_agent_trajectories": len(trajectories),
            "manifest_candidates": int(manifest["candidate_count"]) if manifest_path.exists() else 0,
            "target_scored": bool((PACKAGE / "sealed/new_target_results.parquet").exists()),
            **blind_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()
