"""One-shot held-site evaluator; call only after the candidate manifest freezes."""
from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from brainbodyfm.metrics import age_conditioned_auroc
from brainbodyfm.config import DEFAULT_CONFIG
from brainbodyfm.models import ModelSpec, fit_probe
from agentic_search.evaluator.source_evaluator import C_GRID, canonical_quartet


PACKAGE = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=3)
def _source(held_site: str) -> pd.DataFrame:
    return pd.read_parquet(PACKAGE / f"data/source_folds/{held_site}.parquet")


@lru_cache(maxsize=1)
def _participants() -> pd.DataFrame:
    return pd.read_parquet(PACKAGE / "data/participants.parquet")


@lru_cache(maxsize=1)
def _sealed_labels() -> pd.DataFrame:
    return pd.read_parquet(PACKAGE / "sealed/target_labels.parquet")


def _check_target_release(held_site: str, quartet_key: str, selected_C: float,
                         historical_regression: bool) -> None:
    if historical_regression:
        history = pd.read_csv(PACKAGE / "data/historical_quartets.csv", dtype=str)
        if quartet_key not in set(history.quartet_key):
            raise ValueError("The pre-freeze regression path is restricted to previously target-inspected quartets")
        return
    manifest_path = PACKAGE / "runs/frozen_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Target evaluation is locked until all source-only searches are frozen")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if expected != manifest.get("manifest_sha256"):
        raise ValueError("Frozen candidate manifest hash does not verify")
    matches = [row for row in manifest.get("candidates", [])
               if row["held_site"] == str(held_site) and row["quartet_key"] == quartet_key]
    if not matches:
        raise ValueError("Target candidate is absent from the frozen manifest")
    if any(not np.isclose(float(row["selected_C"]), float(selected_C), rtol=0, atol=1e-12)
           for row in matches):
        raise ValueError("Target selected_C disagrees with the frozen source-selected C")


def evaluate_target(
    held_site: str,
    quartet: Sequence[str],
    selected_C: float,
    *,
    cache_path: str | Path | None = None,
    historical_regression: bool = False,
) -> dict[str, object]:
    """Fit on both sources and score this held site once with the source C."""
    key = canonical_quartet(quartet)
    quartet_key = "|".join(key)
    _check_target_release(str(held_site), quartet_key, float(selected_C), historical_regression)
    permitted_C = (*C_GRID, *DEFAULT_CONFIG.c_grid)
    if not any(np.isclose(float(selected_C), value, rtol=1e-8, atol=1e-12) for value in permitted_C):
        raise ValueError("selected_C is not one of the frozen 17 regularization values")
    catalog = pd.read_csv(PACKAGE / "data/heads.csv", dtype=str).set_index("head_id", drop=False)
    if set(key).difference(catalog.index):
        raise ValueError("Target candidate includes an unknown physiology head")
    names = catalog.loc[list(key), "head_name"].astype(str).tolist()
    source = _source(str(held_site))
    participants = _participants()
    target_labels = _sealed_labels()
    target_labels = target_labels.loc[target_labels.held_site.astype(str).eq(str(held_site))].copy()
    target_features = participants.loc[participants.site.astype(str).eq(str(held_site))].copy()
    target = target_labels.merge(target_features, on=["subject_id", "site", "age"], validate="one_to_one")
    if len(target) != len(target_labels) or len(target) == 0:
        raise ValueError("Sealed labels and held-site feature rows do not align")
    source_sites = sorted(source.site.astype(str).unique())
    if str(held_site) in source_sites or len(source_sites) != 2:
        raise ValueError("Target evaluation source pool does not match the frozen fold")
    for column in names:
        if column not in source or column not in target:
            raise ValueError(f"Missing candidate feature {column}")
    source_ids = source.subject_id.astype(str).to_numpy()
    target_ids = target.subject_id.astype(str).to_numpy()
    if set(source_ids).intersection(target_ids):
        raise ValueError("Source and target participants overlap")
    spec = ModelSpec(name="prospective_random4_l2", penalty="l2", c_grid=C_GRID,
                     max_iter=10_000, tol=1e-5, random_state=2026090401)
    probe = fit_probe(
        source[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
        source.outcome.to_numpy(dtype=int), spec=spec, C=float(selected_C),
        feature_names=names, source_subject_ids=source_ids, target_subject_ids=target_ids,
    )
    if probe.preprocessor.audit_ is None or probe.preprocessor.audit_.target_rows_used:
        raise AssertionError("Held-site participants entered a source preprocessing fit")
    scores = probe.decision_function(target[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
    auc = age_conditioned_auroc(target.outcome.to_numpy(dtype=int), scores,
                                target.age.to_numpy(dtype=float), age_window=2.0)
    if not np.isfinite(auc):
        raise ValueError(f"No finite held-site age-conditioned AUROC for {held_site}/{key}")
    return {
        "held_site": str(held_site),
        "quartet_key": quartet_key,
        "head_ids": list(key),
        "head_names": names,
        "selected_C": float(selected_C),
        "target_auc": float(auc),
        "n_target": int(len(target)),
        "target_prediction_sha256": __import__("hashlib").sha256(np.asarray(scores, dtype="<f8").tobytes()).hexdigest(),
        "target_labels_used_for_model_or_C_selection": 0,
    }
