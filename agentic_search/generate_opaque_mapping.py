"""Freeze the opaque ID permutation for the semantics ablation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parent


def run() -> pd.DataFrame:
    config = __import__("yaml").safe_load((PACKAGE / "configs/protocol.yaml").read_text())
    catalog = pd.read_csv(PACKAGE / "data/heads.csv", dtype=str)
    seed = int(config["opaque_mapping_seed"])
    rng = np.random.default_rng(seed)
    values = catalog.head_id.to_numpy(copy=True)
    rng.shuffle(values)
    mapping = pd.DataFrame({"opaque_id": [f"X{i:04d}" for i in range(1, len(values) + 1)],
                            "head_id": values})
    path = PACKAGE / "data/opaque_head_mapping.csv"
    if path.exists():
        previous = pd.read_csv(path, dtype=str)
        if not previous.equals(mapping):
            raise ValueError("Existing opaque mapping differs from the frozen seed")
        return previous
    mapping.to_csv(path, index=False)
    (PACKAGE / "data/opaque_mapping_provenance.json").write_text(
        json.dumps({"seed": seed, "algorithm": "numpy.default_rng(seed).shuffle(head_ids)",
                    "mapping_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return mapping


if __name__ == "__main__":
    result = run()
    print(f"Saved fixed opaque mapping for {len(result)} heads")
