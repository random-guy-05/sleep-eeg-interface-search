"""Hash frozen protocol/prompts and append an immutable freeze record."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent

FROZEN_EXECUTION_FILES = (
    "freeze_protocol.py",
    "regression_test_existing.py",
    "prepare_search_space.py",
    "generate_prospective_bank.py",
    "generate_opaque_mapping.py",
    "score_source_bank.py",
    "run_classical_baselines.py",
    "baselines/random_orderings.py",
    "agents/model_cli.py",
    "agents/mcp_server.py",
    "agents/run_trajectories.py",
    "agents/run_batch_semantic.py",
    "evaluator/source_evaluator.py",
    "evaluator/cache.py",
    "evaluator/target_evaluator.py",
    "build_manifest.py",
    "score_targets.py",
    "audit.py",
    "analysis/aggregate.py",
    "analysis/make_figures.py",
    "../brainbodyfm/__init__.py",
    "../brainbodyfm/config.py",
    "../brainbodyfm/metrics.py",
    "../brainbodyfm/preprocess.py",
    "../brainbodyfm/models.py",
    "../brainbodyfm/tuning.py",
)

FROZEN_INPUT_FILES = (
    "data/heads.csv",
    "data/participants.parquet",
    "data/historical_quartets.csv",
    "data/folds.json",
    "data/prospective_random500.csv",
    "data/opaque_head_mapping.csv",
    "data/opaque_mapping_provenance.json",
    "data/source_folds/I0002.parquet",
    "data/source_folds/I0006.parquet",
    "data/source_folds/S0001.parquet",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def current_execution_hashes() -> dict[str, str]:
    return {name: sha256(PACKAGE / name) for name in FROZEN_EXECUTION_FILES}


def current_input_hashes() -> dict[str, str]:
    return {name: sha256(PACKAGE / name) for name in FROZEN_INPUT_FILES}


def freeze(phase: str, amendment: str | None = None) -> dict[str, object]:
    protocol = PACKAGE / "configs/protocol.yaml"
    required = [protocol]
    if phase == "final":
        required += [PACKAGE / f"configs/prompts/{name}" for name in
                     ("full_agent.txt", "opaque_agent.txt", "batch_semantic.txt")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing inputs: {missing}")
    protocol_hash = sha256(protocol)
    record_path = PACKAGE / "runs/protocol_freeze.json"
    previous = json.loads(record_path.read_text()) if record_path.exists() else None
    changed = bool(previous and previous["protocol_sha256"] != protocol_hash)
    if changed and not amendment:
        raise ValueError("Protocol changed after its initial freeze; create a versioned amendment instead")
    if phase == "final" and previous is None:
        raise ValueError("Freeze the protocol before freezing prompts")
    record = {
        "phase": phase,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": "configs/protocol.yaml",
        "protocol_sha256": protocol_hash,
        "prompt_sha256": {p.name: sha256(p) for p in required if p != protocol},
    }
    if phase == "final":
        record["execution_sha256"] = current_execution_hashes()
        record["input_sha256"] = current_input_hashes()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if previous:
        history = PACKAGE / "runs/protocol_freeze_history.jsonl"
        previous["superseded_at_utc"] = datetime.now(timezone.utc).isoformat()
        previous["amendment"] = amendment or "Prompt freeze advanced the recorded phase without changing protocol."
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(previous, sort_keys=True) + "\n")
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("protocol", "final"), required=True)
    parser.add_argument("--amendment", help="Document a pre-target protocol correction; required when the protocol hash changes")
    args = parser.parse_args()
    print(json.dumps(freeze(args.phase, args.amendment), indent=2))


if __name__ == "__main__":
    main()
