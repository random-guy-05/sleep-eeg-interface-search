"""Score the unique frozen candidate manifest on held sites exactly once."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from agentic_search.evaluator.target_evaluator import evaluate_target
from agentic_search.freeze_protocol import current_execution_hashes, current_input_hashes


PACKAGE = Path(__file__).resolve().parent


def _score(record):
    return evaluate_target(record["held_site"], record["head_ids"], record["selected_C"])


def score(workers: int = 4) -> pd.DataFrame:
    manifest_path = PACKAGE / "runs/frozen_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("No frozen manifest exists. Finish all source-only searches first.")
    final_path = PACKAGE / "sealed/new_target_results.parquet"
    if final_path.exists():
        raise RuntimeError("Target results already exist; no second target-scoring pass is permitted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads((PACKAGE / "runs/protocol_freeze.json").read_text(encoding="utf-8"))
    if freeze.get("phase") != "final" or manifest.get("protocol_sha256") != freeze.get("protocol_sha256"):
        raise RuntimeError("Target scoring requires the current final frozen protocol and manifest")
    if current_execution_hashes() != freeze.get("execution_sha256"):
        raise RuntimeError("Executable pipeline code changed after the final freeze")
    if current_input_hashes() != freeze.get("input_sha256"):
        raise RuntimeError("Source-only experiment inputs changed after the final freeze")
    existing_progress = PACKAGE / "sealed/new_target_results_progress.parquet"
    existing = {}
    if existing_progress.exists():
        previous = pd.read_parquet(existing_progress)
        if "manifest_sha256" not in previous or set(previous.manifest_sha256.astype(str)) != {manifest["manifest_sha256"]}:
            raise ValueError("Partial target results belong to a different or unverifiable manifest")
        if set(previous.quartet_key).difference({row["quartet_key"] for row in manifest["candidates"]}):
            raise ValueError("Partial target scoring contains candidates outside the frozen manifest")
        existing = {(str(row.held_site), str(row.quartet_key)): row._asdict()
                    for row in previous.itertuples(index=False)}
    records = manifest["candidates"]
    pending = [row for row in records if (row["held_site"], row["quartet_key"]) not in existing]
    results = list(existing.values())
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {pool.submit(_score, row): row for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append({**result, "manifest_sha256": manifest["manifest_sha256"]})
                if index % 100 == 0 or index == len(pending):
                    progress = pd.DataFrame(results).sort_values(["held_site", "quartet_key"]).reset_index(drop=True)
                    temp = existing_progress.with_suffix(".tmp.parquet")
                    progress.to_parquet(temp, index=False)
                    temp.replace(existing_progress)
                    print(f"Held-site scoring complete {index}/{len(pending)}; metrics remain sealed", flush=True)
    frame = pd.DataFrame(results).sort_values(["held_site", "quartet_key"]).reset_index(drop=True)
    if len(frame) != len(records) or frame.duplicated(["held_site", "quartet_key"]).any():
        raise ValueError("Target result set is incomplete or contains duplicates")
    final_frame = frame.drop(columns=["manifest_sha256"], errors="ignore")
    final_frame.to_parquet(final_path, index=False)
    existing_progress.unlink(missing_ok=True)
    (PACKAGE / "sealed/target_scoring_release.json").write_text(json.dumps({
        "manifest_sha256": manifest["manifest_sha256"],
        "candidate_pairs_scored": len(final_frame),
        "result_path": "sealed/new_target_results.parquet",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = score(args.workers)
    print(f"Scored {len(result)} unique candidate/fold pairs under the frozen manifest")


if __name__ == "__main__":
    main()
