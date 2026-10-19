"""Core BrainBodyFM benchmark orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BrainBodyFMConfig, DEFAULT_CONFIG
from .evaluation import NestedEvaluationResult, run_nested_loso
from .models import ModelSpec, make_model_spec
from .taxonomy import coerce_bool_series


def physiology_columns_from_taxonomy(
    feature_names: Sequence[str], taxonomy: pd.DataFrame
) -> list[str]:
    required = {"head_name", "include_physiology_only"}
    if not required.issubset(taxonomy.columns):
        raise ValueError(f"taxonomy needs columns {sorted(required)}")
    allowed = set(taxonomy.loc[coerce_bool_series(taxonomy["include_physiology_only"]), "head_name"].astype(str))
    selected = [str(name) for name in feature_names if str(name) in allowed]
    if not selected:
        raise ValueError("Taxonomy selected zero physiology-only features.")
    return selected


def core_model_specs(config: BrainBodyFMConfig = DEFAULT_CONFIG) -> dict[str, ModelSpec]:
    return {
        "physiology_elastic_net": make_model_spec(
            "physiology_elastic_net", c_grid=config.c_grid, alpha=config.alpha,
            active_threshold=config.active_threshold, random_state=config.random_state,
            max_iter=config.max_iter_elastic_net,
        ),
        "all_head_elastic_net": make_model_spec(
            "all_head_elastic_net", c_grid=config.c_grid, alpha=config.alpha,
            active_threshold=config.active_threshold, random_state=config.random_state,
            max_iter=config.max_iter_elastic_net,
        ),
        "ridge_841": make_model_spec(
            "ridge_841", c_grid=config.c_grid, random_state=config.random_state,
            max_iter=config.max_iter_ridge,
        ),
        "ridge_1024": make_model_spec(
            "ridge_1024", c_grid=config.c_grid, random_state=config.random_state,
            max_iter=config.max_iter_ridge,
        ),
        "learned_stable4_4": make_model_spec(
            "learned_stable4_4", c_grid=config.c_grid, random_state=config.random_state,
            max_iter=config.max_iter_ridge,
        ),
    }


def run_core_benchmark(
    representations: Mapping[str, object],
    manifest: pd.DataFrame,
    *,
    feature_names: Mapping[str, Sequence[str]],
    config: BrainBodyFMConfig = DEFAULT_CONFIG,
    checkpoint_dir: str | Path | None = None,
    model_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, NestedEvaluationResult]]:
    """Run the prespecified learned interfaces, keeping Stable4 marked as prior."""
    required = {"semantic_841", "physiology", "stable4"}
    missing = sorted(required.difference(representations))
    if missing:
        raise ValueError(f"Core BrainBodyFM benchmark is missing required interfaces: {missing}")
    specs = core_model_specs(config)
    aliases = {
        "physiology_elastic_net": "physiology",
        "all_head_elastic_net": "semantic_841",
        "ridge_841": "semantic_841",
        "ridge_1024": "latent_1024",
        "learned_stable4_4": "stable4",
    }
    results: dict[str, NestedEvaluationResult] = {}
    summary_rows: list[pd.DataFrame] = []
    for model_name, representation_name in aliases.items():
        if model_names is not None and model_name not in model_names:
            continue
        if representation_name not in representations:
            continue
        result = run_nested_loso(
            representations[representation_name],
            manifest,
            model_spec=specs[model_name],
            representation=model_name,
            feature_names=feature_names[representation_name],
            config=config,
        )
        results[model_name] = result
        if checkpoint_dir is not None:
            result.save(Path(checkpoint_dir) / model_name)
            print(f"MODEL complete {model_name}: macro_auc={result.macro_auc:.8f}", flush=True)
        model_summary = result.summary()
        model_summary["evidence_section"] = (
            "previously_discovered_reference"
            if model_name == "learned_stable4_4"
            else "properly_nested_interface"
        )
        model_summary["target_site_labels_used"] = False
        summary_rows.append(model_summary)
    summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    return summary, results


def run_pure_l1_sensitivity(
    X: object,
    manifest: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    config: BrainBodyFMConfig = DEFAULT_CONFIG,
) -> NestedEvaluationResult:
    """Run the prespecified alpha=1 pure-L1 sensitivity on one interface."""
    spec = make_model_spec(
        "pure_l1_sensitivity",
        c_grid=config.c_grid,
        alpha=1.0,
        active_threshold=config.active_threshold,
        random_state=config.random_state,
        max_iter=config.max_iter_elastic_net,
    )
    return run_nested_loso(
        X,
        manifest,
        model_spec=spec,
        representation="pure_l1_sensitivity",
        feature_names=feature_names,
        config=config,
    )


def empirical_tail_probability(null_aucs: Sequence[float], observed_auc: float) -> float:
    """Finite-sample tail statistic for the dimension-matched random null."""
    values = np.asarray(null_aucs, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or not np.isfinite(observed_auc):
        return float("nan")
    return float((1 + np.count_nonzero(values >= float(observed_auc))) / (len(values) + 1))
