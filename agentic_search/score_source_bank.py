"""Source-score the prospective random-500 bank for all frozen site folds."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from agentic_search.evaluator.source_evaluator import evaluate_source


PACKAGE = Path(__file__).resolve().parent


def _evaluate(task):
    held_site, quartet, cache_path = task
    return evaluate_source(held_site, quartet, cache_path=cache_path)


def score_bank(workers: int = 4, output: Path | None = None) -> pd.DataFrame:
    bank = pd.read_csv(PACKAGE / "data/prospective_random500.csv", dtype=str)
    folds = json.loads((PACKAGE / "data/folds.json").read_text(encoding="utf-8"))
    sites = list(folds["site_order"])
    destination = output or PACKAGE / "runs/source_only/prospective_random500.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[str, str], dict] = {}
    if destination.exists():
        prior = pd.read_parquet(destination)
        existing = {(str(row["held_site"]), str(row["quartet_key"])): row
                    for row in prior.to_dict(orient="records")}
    tasks = []
    for held_site in sites:
        for row in bank.itertuples(index=False):
            key = str(row.quartet_key)
            if (held_site, key) not in existing:
                tasks.append((held_site, [getattr(row, f"head_{i}") for i in range(1, 5)],
                              str(PACKAGE / "cache/source.sqlite")))
    records = list(existing.values())
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
            future_to_task = {pool.submit(_evaluate, task): task for task in tasks}
            for done, future in enumerate(as_completed(future_to_task), start=1):
                result = future.result()
                records.append(result)
                if done % 50 == 0 or done == len(tasks):
                    print(f"Source scored {done}/{len(tasks)} uncached candidates", flush=True)
    frame = pd.DataFrame(records).sort_values(["held_site", "quartet_key"]).reset_index(drop=True)
    if len(frame) != 3 * len(bank) or frame.duplicated(["held_site", "quartet_key"]).any():
        raise ValueError("Incomplete or duplicated source-score bank")
    if "target_auc" in frame or any("target" in str(c).lower() for c in frame.columns):
        raise AssertionError("Source-score artifact contains target information")
    frame.to_parquet(destination, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_bank(args.workers, args.output)
    print(f"Wrote {len(result)} source-only candidate/fold rows")


if __name__ == "__main__":
    main()
