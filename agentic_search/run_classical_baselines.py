"""Run evolutionary-robust and taxonomy-diverse source-only search baselines."""
from __future__ import annotations

import argparse
import json
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_search.evaluator.source_evaluator import evaluate_source


PACKAGE = Path(__file__).resolve().parent


def _random_quartet(rng: np.random.Generator, ids: list[str], forbidden: set[str], seen: set[str]) -> tuple[str, ...]:
    for _ in range(100_000):
        candidate = tuple(sorted(rng.choice(ids, size=4, replace=False).tolist()))
        key = "|".join(candidate)
        if key not in forbidden and key not in seen:
            return candidate
    raise RuntimeError("Unable to sample a new valid quartet")


def _mutate(elite: tuple[str, ...], rng: np.random.Generator, ids: list[str],
            forbidden: set[str], seen: set[str]) -> tuple[str, ...]:
    for _ in range(10_000):
        candidate = list(elite)
        replace_at = int(rng.integers(0, 4))
        unused = [head for head in ids if head not in candidate]
        candidate[replace_at] = str(rng.choice(unused))
        quartet = tuple(sorted(candidate))
        key = "|".join(quartet)
        if key not in forbidden and key not in seen:
            return quartet
    return _random_quartet(rng, ids, forbidden, seen)


def _distinct_family_quartet(rng: np.random.Generator, groups: dict[str, list[str]], ids: list[str],
                             forbidden: set[str], seen: set[str]) -> tuple[tuple[str, ...], bool]:
    usable = [category for category, members in groups.items() if members]
    if len(usable) >= 4:
        for _ in range(512):
            categories = rng.choice(usable, size=4, replace=False)
            quartet = tuple(sorted(str(rng.choice(groups[str(category)])) for category in categories))
            key = "|".join(quartet)
            if key not in forbidden and key not in seen:
                return quartet, True
    return _random_quartet(rng, ids, forbidden, seen), False


def _run_one(task):
    method, held_site, budget, seed, package_path = task
    package = Path(package_path)
    catalog = pd.read_csv(package / "data/heads.csv", dtype=str)
    ids = catalog.head_id.tolist()
    forbidden = set(pd.read_csv(package / "data/historical_quartets.csv", dtype=str).quartet_key)
    groups = {str(category): group.head_id.tolist()
              for category, group in catalog.groupby("taxonomy_category", sort=True)}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(budget), zlib.crc32(held_site.encode())]))
    seen: set[str] = set()
    evaluations: list[dict[str, object]] = []
    n = int(budget)
    if method == "evolutionary_robust":
        initial_n = min(5, n)
        proposed = []
        for _ in range(initial_n):
            quartet = _random_quartet(rng, ids, forbidden, seen)
            seen.add("|".join(quartet))
            proposed.append(quartet)
        seen.clear()
    elif method == "taxonomy_diverse_robust":
        proposed = []
    else:
        raise ValueError(method)

    while len(evaluations) < n:
        if method == "taxonomy_diverse_robust":
            candidate, used_diversity = _distinct_family_quartet(rng, groups, ids, forbidden, seen)
            batch = [(candidate, used_diversity)]
        elif proposed:
            batch = [(proposed.pop(0), False)]
        else:
            ranked = sorted(evaluations, key=lambda row: (-float(row["min_source_auc"]), str(row["quartet_key"])))
            elites = [tuple(row["head_ids"]) for row in ranked[:3]]
            replacements = [elites[0], elites[0], elites[1], elites[2]]
            batch = []
            reserved = set(seen)
            for elite in replacements:
                q = _mutate(elite, rng, ids, forbidden, reserved)
                reserved.add("|".join(q))
                batch.append((q, False))
            restart = _random_quartet(rng, ids, forbidden, reserved)
            batch.append((restart, False))

        for quartet, diversity_used in batch:
            if len(evaluations) >= n:
                break
            key = "|".join(quartet)
            if key in seen or key in forbidden:
                raise AssertionError("Baseline emitted a repeated or forbidden quartet")
            seen.add(key)
            result = evaluate_source(held_site, quartet, cache_path=package / "cache/source.sqlite")
            evaluations.append({**result,
                                "method": method,
                                "held_site": held_site,
                                "budget": n,
                                "seed": int(seed),
                                "candidate_index": len(evaluations) + 1,
                                "diversity_quartet": bool(diversity_used)})

    chosen = sorted(evaluations, key=lambda row: (-float(row["min_source_auc"]), str(row["quartet_key"])))
    selected = chosen[0]
    for row in evaluations:
        row["selected_by_robust"] = row["quartet_key"] == selected["quartet_key"]
    return evaluations


def run(workers: int = 4) -> pd.DataFrame:
    protocol = __import__("yaml").safe_load((PACKAGE / "configs/protocol.yaml").read_text())
    folds = json.loads((PACKAGE / "data/folds.json").read_text())
    tasks = []
    for held_site in folds["site_order"]:
        for seed in range(1000, 1020):
            for budget in protocol["evolutionary"]["budgets"]:
                tasks.append(("evolutionary_robust", held_site, budget, seed, str(PACKAGE)))
            for budget in protocol["taxonomy_diverse"]["budgets"]:
                tasks.append(("taxonomy_diverse_robust", held_site, budget, seed, str(PACKAGE)))
    destination = PACKAGE / "runs/source_only/classical_baselines.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = PACKAGE / "runs/source_only/classical_trajectories"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[tuple[str, str, int, int], list[dict]] = {}
    if destination.exists():
        old = pd.read_parquet(destination)
        for keys, group in old.groupby(["method", "held_site", "budget", "seed"], sort=False):
            completed[tuple([str(keys[0]), str(keys[1]), int(keys[2]), int(keys[3])])] = group.to_dict("records")
    for path in checkpoint_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"]
        if not rows:
            continue
        row = rows[0]
        key = (str(row["method"]), str(row["held_site"]), int(row["budget"]), int(row["seed"]))
        completed.setdefault(key, rows)
    pending = [task for task in tasks if (task[0], task[1], int(task[2]), int(task[3])) not in completed]
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(_run_one, task): task for task in pending}
        for done, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            key = (task[0], task[1], int(task[2]), int(task[3]))
            completed[key] = future.result()
            checkpoint = checkpoint_dir / f"{key[0]}__{key[1]}__B{key[2]}__seed{key[3]}.json"
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps({"rows": completed[key]}, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(checkpoint)
            if done % 5 == 0 or done == len(pending):
                print(f"Classical trajectories complete {done}/{len(pending)}", flush=True)
    rows = [row for key in sorted(completed) for row in completed[key]]
    frame = pd.DataFrame(rows)
    frame.to_parquet(destination, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run(args.workers)
    print(f"Saved {result.groupby(['method','held_site','budget','seed']).ngroups} source-only trajectories")


if __name__ == "__main__":
    main()
