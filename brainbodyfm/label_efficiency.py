"""Downstream-label budget experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import os

import numpy as np
import pandas as pd

from .config import BrainBodyFMConfig, DEFAULT_CONFIG
from .execution import run_bounded_fit_tasks
from .metrics import age_conditioned_auroc
from .models import ModelSpec, fit_probe
from .preprocess import SourcePreprocessor
from .tuning import tune_source_transfer


def stratified_source_sample_indices(
    manifest: pd.DataFrame,
    *,
    source_sites: Sequence[str],
    fraction: float,
    seed: int,
    outcome_column: str = "outcome",
    site_column: str = "site",
    max_retries: int = 100,
) -> np.ndarray:
    """Sample labels within site×outcome strata without target-site rows."""
    q = float(fraction)
    if not 0 < q <= 1:
        raise ValueError("fraction must be in (0, 1].")
    frame = manifest.reset_index(drop=True).copy()
    frame["_row_index"] = np.arange(len(frame), dtype=int)
    source = frame[frame[site_column].astype(str).isin([str(x) for x in source_sites])]
    if source.empty:
        raise ValueError("No source rows available for label sampling.")
    rng = np.random.default_rng(int(seed))
    for attempt in range(max_retries):
        chosen: list[int] = []
        for (_site, outcome), group in source.groupby([site_column, outcome_column], sort=True):
            indices = group["_row_index"].to_numpy(dtype=int)
            if q >= 1.0:
                take = len(indices)
            else:
                take = max(1, int(np.ceil(q * len(indices))))
                take = min(take, len(indices))
            chosen.extend(rng.choice(indices, size=take, replace=False).tolist())
        chosen_arr = np.asarray(sorted(set(chosen)), dtype=int)
        if len(np.unique(manifest.iloc[chosen_arr][outcome_column])) >= 2:
            return chosen_arr
        # One-class samples are not discarded as valid draws; advance the
        # deterministic generator and redraw as specified by the protocol.
        rng = np.random.default_rng(int(seed) + attempt + 1)
    raise ValueError("Unable to draw a two-class source label sample after retries.")


def _fit_label_budget_task(task):
    (row, source_x, source_y, source_ids, target_x, target_y, target_age,
     target_ids, spec, names, preprocessor, age_window) = task
    probe = fit_probe(source_x, source_y, spec=spec, C=row["chosen_C"],
                      feature_names=names, source_subject_ids=source_ids,
                      target_subject_ids=target_ids, preprocessor=preprocessor)
    target_scores = probe.decision_function(target_x)
    return {**row, "n_active": int(probe.n_active),
            "age_auc": float(age_conditioned_auroc(target_y, target_scores, target_age, age_window))}


def run_label_efficiency(
    representations: Mapping[str, object],
    manifest: pd.DataFrame,
    *,
    model_specs: Mapping[str, ModelSpec],
    feature_names: Mapping[str, Sequence[str]] | None = None,
    config: BrainBodyFMConfig = DEFAULT_CONFIG,
    chosen_C_by_interface_site: Mapping[tuple[str, str], float] | None = None,
    checkpoint_path: str | Path | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Compare interfaces at 25/50/75/100% source-label budgets.

    Regularization is selected once per outer fold and representation on the
    full source labels.  At reduced budgets all source rows remain available to
    unsupervised preprocessing, while only the sampled labels fit the decoder.
    """
    sites = manifest[config.site_column].astype(str).to_numpy()
    outcome_column = config.primary_outcome
    if outcome_column not in manifest.columns:
        if outcome_column == "outcome" and "target" in manifest.columns:
            outcome_column = "target"
        elif outcome_column == "target" and "outcome" in manifest.columns:
            outcome_column = "outcome"
        else:
            raise ValueError(f"Manifest is missing outcome column {config.primary_outcome!r}.")
    y = pd.to_numeric(manifest[outcome_column], errors="raise").astype(int).to_numpy()
    ages = pd.to_numeric(manifest[config.age_column]).to_numpy(dtype=float)
    subjects = manifest[config.subject_column].astype(str).to_numpy()
    matrices = {
        name: np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value, dtype=float)
        for name, value in representations.items()
    }
    for name, matrix in matrices.items():
        if matrix.ndim == 1:
            matrices[name] = matrix.reshape(-1, 1)
        if len(matrices[name]) != len(manifest):
            raise ValueError(f"Representation {name!r} and manifest have different row counts.")
        if name not in model_specs:
            raise ValueError(f"Missing ModelSpec for representation {name!r}.")

    def tasks():
        for target_site in config.sites:
            target_idx = np.flatnonzero(sites == target_site)
            source_sites = tuple(site for site in config.sites if site != target_site)
            source_idx = np.flatnonzero(np.isin(sites, source_sites))
            for name, matrix in matrices.items():
                spec = model_specs[name]
                names = tuple(str(x) for x in (feature_names or {}).get(name, [f"feature_{i}" for i in range(matrix.shape[1])]))
                key = (name, str(target_site))
                if chosen_C_by_interface_site is not None:
                    if key not in chosen_C_by_interface_site:
                        raise ValueError(f"Missing frozen C for {key}")
                    chosen_C = float(chosen_C_by_interface_site[key])
                else:
                    tuning = tune_source_transfer(
                        matrix[source_idx], y[source_idx], ages[source_idx], sites[source_idx],
                        model_spec=spec, feature_names=names, subject_ids=subjects[source_idx],
                        age_window=config.age_window, tie_tolerance=config.tie_tolerance,
                    )
                    chosen_C = tuning.chosen_C
                # All draws use exactly the same source-only unsupervised fit.
                # Each process receives its own serialized copy, never a target fit.
                preprocessor = SourcePreprocessor(n_components=spec.n_components, random_state=spec.random_state)
                preprocessor.fit(matrix[source_idx], subject_ids=subjects[source_idx], target_subject_ids=subjects[target_idx])
                for fraction in config.label_fractions:
                    repetitions = 1 if float(fraction) >= 1.0 else config.label_repetitions
                    for repeat in range(repetitions):
                        site_seed = sum((position + 1) * ord(char) for position, char in enumerate(str(target_site)))
                        draw_seed = int(config.label_seed + repeat * 1009 + round(float(fraction) * 1000) + sum(ord(c) for c in name) + site_seed)
                        labeled_idx = stratified_source_sample_indices(
                            manifest, source_sites=source_sites, fraction=float(fraction),
                            seed=draw_seed, outcome_column=outcome_column, site_column=config.site_column,
                        )
                        row = {"outer_site": target_site, "representation": name,
                               "label_fraction": float(fraction), "repeat": int(repeat),
                               "n_labeled": int(len(labeled_idx)), "n_positive": int(y[labeled_idx].sum()),
                               "chosen_C": chosen_C}
                        yield (row, matrix[labeled_idx], y[labeled_idx], subjects[labeled_idx],
                               matrix[target_idx], y[target_idx], ages[target_idx], subjects[target_idx],
                               spec, names, preprocessor, config.age_window)
    rows: list[dict[str, object]] = []
    order = ["outer_site", "representation", "label_fraction", "repeat"]
    for row in run_bounded_fit_tasks(_fit_label_budget_task, tasks(), workers):
        rows.append(row)
        if checkpoint_path is not None:
            destination = Path(checkpoint_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            pd.DataFrame(rows).sort_values(order).to_csv(temporary, index=False)
            temporary.replace(destination)
        if os.environ.get("BBFM_PROGRESS"):
            print(f"LABEL done site={row['outer_site']} interface={row['representation']} fraction={row['label_fraction']} repeat={row['repeat']} rows={len(rows)}", flush=True)
    return pd.DataFrame(rows, columns=[
        "outer_site", "representation", "label_fraction", "repeat", "n_labeled",
        "n_positive", "chosen_C", "n_active", "age_auc",
    ]).sort_values(order).reset_index(drop=True)


def summarize_label_efficiency(
    results: pd.DataFrame,
    *,
    site_order: Sequence[str] = ("I0002", "I0006", "S0001"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create replicate-level macro AUCs and median/IQR curve summaries.

    Site-specific AUCs are never pooled as if they were exchangeable draws.
    Each repeat is first macro-averaged across the three held-out sites, exactly
    matching the primary endpoint, and only then summarized across label draws.
    """
    required = {"outer_site", "representation", "label_fraction", "repeat", "age_auc"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"label-efficiency table is missing columns: {missing}")
    expected_sites = {str(site) for site in site_order}
    replicate_rows: list[dict[str, object]] = []
    grouped = results.groupby(
        ["representation", "label_fraction", "repeat"], sort=True, dropna=False
    )
    for (representation, fraction, repeat), group in grouped:
        sites = set(group["outer_site"].astype(str))
        values = pd.to_numeric(group["age_auc"], errors="coerce").to_numpy(dtype=float)
        complete = sites == expected_sites and len(group) == len(expected_sites)
        macro = float(np.mean(values)) if complete and np.all(np.isfinite(values)) else float("nan")
        replicate_rows.append(
            {
                "representation": str(representation),
                "label_fraction": float(fraction),
                "repeat": int(repeat),
                "n_sites": int(len(sites)),
                "macro_age_auc": macro,
            }
        )
    replicates = pd.DataFrame(replicate_rows)
    summary_rows: list[dict[str, object]] = []
    for (representation, fraction), group in replicates.groupby(
        ["representation", "label_fraction"], sort=True
    ):
        values = pd.to_numeric(group["macro_age_auc"], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        summary_rows.append(
            {
                "representation": str(representation),
                "label_fraction": float(fraction),
                "n_repeats": int(len(group)),
                "n_finite": int(len(finite)),
                "median_macro_age_auc": float(np.median(finite)) if len(finite) else float("nan"),
                "q25_macro_age_auc": float(np.quantile(finite, 0.25)) if len(finite) else float("nan"),
                "q75_macro_age_auc": float(np.quantile(finite, 0.75)) if len(finite) else float("nan"),
            }
        )
    return replicates, pd.DataFrame(summary_rows)
