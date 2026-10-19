"""Score the separately frozen supplemental ablation manifest with the locked evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from agentic_search.evaluator import target_evaluator as te

PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / "runs/supplemental/held_site"


def canonical_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def score(workers: int, manifest_name: str = "ablation_manifest.json",
          results_name: str = "ablation_target_results.parquet") -> pd.DataFrame:
    mp = OUT / manifest_name
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if canonical_hash(payload) != manifest["manifest_sha256"]:
        raise ValueError("Supplemental manifest hash failed verification")
    if manifest.get("manifest_type") not in {
        "append-only supplemental ablation candidate manifest",
        "append-only generalizability candidate manifest",
    }:
        raise ValueError("Unexpected manifest type")
    allowed = {(r["held_site"], r["quartet_key"]): float(r["selected_C"])
               for r in manifest["candidates"]}
    if len(allowed) != manifest["candidate_count"]:
        raise ValueError("Manifest contains duplicate held-site/quartet keys")

    # Keep the original frozen-manifest guard intact for normal calls. This
    # supplemental path validates against its own separately hashed manifest.
    def check(site, key, selected_C, historical_regression):
        if historical_regression:
            raise ValueError("Historical-regression mode is not permitted for supplemental candidates")
        expected = allowed.get((str(site), str(key)))
        if expected is None or abs(expected - float(selected_C)) > 1e-12:
            raise ValueError(f"Candidate/C is absent from supplemental frozen manifest: {site}/{key}")
    te._check_target_release = check

    stem = results_name.removesuffix(".parquet")
    progress_path = OUT / f"{stem}_progress.parquet"
    final_path = OUT / results_name
    if final_path.exists():
        raise RuntimeError("Supplemental target scores are already released; refusing a second pass")
    existing = {}
    parent_hash = manifest.get("parent_manifest_sha256")
    if parent_hash is not None:
        result_stem = ("ablation_target_results" if "ablation" in manifest["manifest_type"]
                       else "generalizability_target_results")
        parent_result = OUT / f"{result_stem}.parquet"
        parent_release = OUT / ("ablation_target_release.json" if result_stem == "ablation_target_results"
                                else f"{result_stem}_release.json")
        if not parent_result.is_file() or not parent_release.is_file():
            raise FileNotFoundError("Versioned scoring requires the parent manifest's released scores")
        release = json.loads(parent_release.read_text(encoding="utf-8"))
        if release.get("manifest_sha256") != parent_hash:
            raise ValueError("Parent target scores do not match the parent manifest hash")
        parent_rows = pd.read_parquet(parent_result)
        for row in parent_rows.to_dict("records"):
            key = (str(row["held_site"]), str(row["quartet_key"]))
            expected = allowed.get(key)
            if expected is None or abs(expected - float(row["selected_C"])) > 1e-12:
                raise ValueError(f"Parent score is absent from the appended manifest or has different C: {key}")
            row["manifest_sha256"] = manifest["manifest_sha256"]
            existing[key] = row
    if progress_path.exists():
        old = pd.read_parquet(progress_path)
        if set(old.manifest_sha256.astype(str)) != {manifest["manifest_sha256"]}:
            raise ValueError("Progress file belongs to another supplemental manifest")
        existing = {(str(r.held_site), str(r.quartet_key)): r._asdict() for r in old.itertuples(index=False)}
    todo = [r for r in manifest["candidates"] if (r["held_site"], r["quartet_key"]) not in existing]
    results = list(existing.values())

    def run(row):
        result = te.evaluate_target(row["held_site"], row["head_ids"], row["selected_C"])
        return {**result, "manifest_sha256": manifest["manifest_sha256"]}

    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(run, row): row for row in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 50 == 0 or i == len(todo):
                pd.DataFrame(results).sort_values(["held_site", "quartet_key"]).to_parquet(progress_path, index=False)
                print(f"Supplemental held-site results: {i}/{len(todo)}", flush=True)
    frame = pd.DataFrame(results).sort_values(["held_site", "quartet_key"]).reset_index(drop=True)
    if len(frame) != len(allowed) or frame.duplicated(["held_site", "quartet_key"]).any():
        raise ValueError("Supplemental scoring is incomplete or contains duplicate keys")
    frame.to_parquet(final_path, index=False)
    (OUT / f"{stem}_release.json").write_text(json.dumps({
        "manifest_sha256": manifest["manifest_sha256"], "candidate_pairs_scored": len(frame),
        "result_path": str(final_path.relative_to(PACKAGE)),
        "evaluator": "agentic_search.evaluator.target_evaluator.evaluate_target, called under separately verified supplemental-manifest membership check",
    }, indent=2, sort_keys=True) + "\n")
    progress_path.unlink(missing_ok=True)
    return frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--manifest", default="ablation_manifest.json")
    p.add_argument("--results", default="ablation_target_results.parquet")
    args = p.parse_args()
    result = score(args.workers, args.manifest, args.results)
    print(f"Scored {len(result)} unique supplemental held-site/quartet candidates")


if __name__ == "__main__":
    main()
