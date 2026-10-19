"""Generate the deterministic, target-blind 500-quartet random bank."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parent


def generate(n: int = 500, seed: int = 2026092401, output: Path | None = None) -> pd.DataFrame:
    catalog = pd.read_csv(PACKAGE / "data/heads.csv", dtype=str)
    historical = pd.read_csv(PACKAGE / "data/historical_quartets.csv", dtype=str)
    forbidden = set(historical.quartet_key)
    ids = catalog.head_id.tolist()
    rng = np.random.default_rng(seed)
    found: set[tuple[str, ...]] = set()
    rows = []
    while len(rows) < n:
        quartet = tuple(sorted(rng.choice(ids, size=4, replace=False).tolist()))
        key = "|".join(quartet)
        if key in forbidden or quartet in found:
            continue
        found.add(quartet)
        rows.append({"bank_index": len(rows), "seed": seed, "quartet_key": key,
                     **{f"head_{i + 1}": value for i, value in enumerate(quartet)}})
    frame = pd.DataFrame(rows)
    if frame.quartet_key.nunique() != n or set(frame.quartet_key).intersection(forbidden):
        raise AssertionError("Prospective random bank violates uniqueness or history exclusion")
    path = output or PACKAGE / "data/prospective_random500.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026092401)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = generate(args.n, args.seed, args.output)
    print(f"Saved {len(result)} unique target-blind quartets; seed={args.seed}")


if __name__ == "__main__":
    main()
