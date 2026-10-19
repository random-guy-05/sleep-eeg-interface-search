"""Independent consistency checks for completed, frozen-protocol artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BrainBodyFMConfig
from .metrics import age_conditioned_auroc
from .representations import STABLE4_COLUMNS
from .taxonomy import coerce_bool_series


def reconstruct_source_normalized_scores(source, target, coefficients, intercept):
    """Reference calculation using only source medians, means and population SD."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    medians = np.array([np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0. for column in source.T])
    filled = np.where(np.isfinite(source), source, medians)
    mean, scale = filled.mean(axis=0), filled.std(axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.
    target_filled = np.where(np.isfinite(target), target, medians)
    return ((target_filled - mean)/scale) @ np.asarray(coefficients) + float(intercept)


def verify_tuning_group(tuning, fold, config):
    """Check all directions, means and the stronger-regularization tie rule."""
    expected = {(float(c), d) for c in config.c_grid for d in ("A_to_B", "B_to_A", "mean")}
    if set(zip(tuning.C, tuning.direction)) != expected or len(tuning) != len(expected):
        raise ValueError("Incomplete or duplicated source-transfer tuning grid")
    a, b = sorted(set(config.sites) - {str(fold.outer_site)})
    for direction, source, target in (("A_to_B", a, b), ("B_to_A", b, a), ("mean", f"{a}+{b}", "paired")):
        selected = tuning.loc[tuning.direction.eq(direction)]
        if not selected.source_site.eq(source).all() or not selected.target_site.eq(target).all():
            raise ValueError("Tuning direction includes the wrong institution")
    pivot = tuning.pivot(index="C", columns="direction", values="age_auc")
    if not np.isfinite(pivot.to_numpy()).all():
        raise ValueError("Undefined tuning endpoint")
    if not np.allclose(pivot["mean"], (pivot.A_to_B + pivot.B_to_A)/2, rtol=0, atol=1e-12):
        raise ValueError("Stored source-transfer means are inconsistent")
    eligible = pivot.index[pivot["mean"] >= pivot["mean"].max() - config.tie_tolerance]
    if float(fold.chosen_C) != float(min(eligible)):
        raise ValueError("Chosen C violates the source-only tie rule")
    selected = pivot.loc[float(fold.chosen_C)]
    if not np.allclose([fold.inner_A_to_B_auc, fold.inner_B_to_A_auc, fold.inner_mean_auc],
                       selected[["A_to_B", "B_to_A", "mean"]], rtol=0, atol=1e-12):
        raise ValueError("Fold tuning summary differs from the selected grid row")


def verify_core_results(directory, manifest, taxonomy, config: BrainBodyFMConfig, *, cache_paths=None):
    """Recompute endpoints and verify all core artifact relationships."""
    root = Path(directory)
    files = [root/name for name in ("core_outer_summary.csv", "core_site_summary.csv", "core_outer_predictions.parquet", "tuning_logs.parquet", "elasticnet_coefficients.parquet")]
    summary, folds = (pd.read_csv(p) for p in files[:2])
    predictions, tuning, coefficients = (pd.read_parquet(p) for p in files[2:])
    semantic = set(taxonomy.head_name.astype(str))
    physiology = set(taxonomy.loc[coerce_bool_series(taxonomy.include_physiology_only), "head_name"].astype(str))
    names = {"physiology_elastic_net": physiology, "all_head_elastic_net": semantic,
             "ridge_841": semantic, "ridge_1024": None, "learned_stable4_4": set(STABLE4_COLUMNS)}
    aliases = {"physiology_elastic_net": "physiology", "all_head_elastic_net": "semantic_841", "ridge_841": "semantic_841", "ridge_1024": "latent_1024", "learned_stable4_4": "stable4"}
    if cache_paths is not None:
        from .extraction import assemble_cached_representations
        matrices, columns, _ = assemble_cached_representations(manifest, semantic_path=str(cache_paths["semantic"]), latent_path=str(cache_paths["latent"]), taxonomy_path=str(cache_paths["taxonomy"]), require_complete=True, strict_dimensions=True)
        files.extend(Path(p) for p in cache_paths.values())
    if set(summary.representation) != set(names) or len(summary) != len(names):
        raise ValueError("Invalid core representation panel")
    expected_folds = {(name, site) for name in names for site in config.sites}
    if set(zip(folds.representation, folds.outer_site)) != expected_folds or len(folds) != len(expected_folds):
        raise ValueError("Invalid outer fold panel")
    for table in (folds, tuning):
        flags = [c for c in table if c.startswith("target_") and c != "target_site"]
        if not flags or table[flags].isna().any().any() or table[flags].to_numpy().any():
            raise ValueError("Nonzero or missing leakage audit flag")
    if tuning.fit_error.fillna("").astype(str).str.strip().ne("").any():
        raise ValueError("An inner fit failed")
    warnings = tuning.convergence_warning.fillna("").astype(str).tolist()
    patient_index = manifest.assign(subject_id=manifest[config.subject_column].astype(str)).set_index("subject_id")
    if not patient_index.index.is_unique:
        raise ValueError("Duplicate manifest subject")
    for name, allowed in names.items():
        model_predictions = predictions.loc[predictions.representation.eq(name)].copy()
        model_predictions["subject_id"] = model_predictions.subject_id.astype(str)
        if model_predictions.subject_id.duplicated().any() or set(model_predictions.subject_id) != set(patient_index.index):
            raise ValueError(f"Incorrect patient prediction panel: {name}")
        aligned = model_predictions.set_index("subject_id").loc[patient_index.index]
        for column, manifest_column in (("site", config.site_column), ("age", config.age_column), ("outcome", config.primary_outcome)):
            if not np.array_equal(aligned[column].to_numpy(), patient_index[manifest_column].to_numpy()):
                raise ValueError(f"Prediction metadata mismatch: {name}/{column}")
        if not np.isfinite(aligned.score.to_numpy()).all():
            raise ValueError("Nonfinite prediction score")
        model_folds = folds.loc[folds.representation.eq(name)]
        fold_file = root/name/"fold_summary.csv"
        files.append(fold_file)
        warnings.extend(pd.read_csv(fold_file).convergence_warning.fillna("").astype(str).tolist())
        recomputed = []
        for fold in model_folds.itertuples():
            group = tuning.loc[tuning.representation.eq(name) & tuning.outer_site.eq(fold.outer_site)]
            verify_tuning_group(group, fold, config)
            p = aligned.loc[aligned.site.eq(fold.outer_site)]
            value = age_conditioned_auroc(p.outcome, p.score, p.age, config.age_window)
            if not np.isclose(value, fold.outer_auc, rtol=0, atol=1e-12):
                raise ValueError("Held-site endpoint does not reproduce predictions")
            recomputed.append(value)
            c = coefficients.loc[coefficients.representation.eq(name) & coefficients.outer_site.eq(fold.outer_site)]
            expected_width = 1024 if allowed is None else len(allowed)
            if len(c) != expected_width or c.feature.duplicated().any() or (allowed is not None and set(c.feature) != allowed):
                raise ValueError("Coefficient feature set differs from its frozen interface")
            if not np.isfinite(c.coefficient.to_numpy()).all() or not c.chosen_C.eq(fold.chosen_C).all():
                raise ValueError("Invalid coefficient values or regularization")
            active = c.coefficient.abs().gt(config.active_threshold)
            if not np.array_equal(active, coerce_bool_series(c.active)) or int(active.sum()) != int(fold.n_active):
                raise ValueError("Active-coordinate counts are inconsistent")
            if cache_paths is not None:
                if c.intercept.nunique() != 1 or not np.isfinite(c.intercept.iloc[0]):
                    raise ValueError("Invalid saved intercept")
                alias = aliases[name]
                indices = {feature: i for i, feature in enumerate(columns[alias])}
                matrix = np.asarray(matrices[alias], dtype=float)[:, [indices[feature] for feature in c.feature]]
                held = manifest[config.site_column].astype(str).eq(fold.outer_site).to_numpy()
                reconstructed = reconstruct_source_normalized_scores(matrix[~held], matrix[held], c.coefficient.to_numpy(), c.intercept.iloc[0])
                if not np.allclose(reconstructed, p.score.to_numpy(), rtol=1e-10, atol=1e-10):
                    raise ValueError(f"Saved predictions do not reproduce source-normalized coefficients: {name}/{fold.outer_site}")
        model_summary = summary.loc[summary.representation.eq(name)].iloc[0]
        if not np.isclose(np.mean(recomputed), model_summary.macro_age_auc, rtol=0, atol=1e-12):
            raise ValueError("Macro endpoint differs from the three held-site endpoints")
        width = 1024 if allowed is None else len(allowed)
        dimension = model_folds.n_active.mean() if name in ("physiology_elastic_net", "all_head_elastic_net") else width
        if model_summary.input_dimensionality != width or not np.isclose(model_summary.dimensionality, dimension, rtol=0, atol=1e-12):
            raise ValueError("Reported interface dimension is inconsistent")
    for message in warnings:
        lowered = message.lower()
        if "converg" in lowered and ("max_iter" in lowered or "failed" in lowered):
            raise ValueError("Unresolved convergence warning")
    if set(predictions.representation) != set(names) or set(tuning.representation) != set(names) or set(coefficients.representation) != set(names):
        raise ValueError("Unexpected representation rows in a core artifact")
    if len(tuning) != len(expected_folds)*len(config.c_grid)*3:
        raise ValueError("Unexpected tuning records")
    return {"status": "verified", "source_normalized_scores_reconstructed": cache_paths is not None, "n_interfaces": len(names), "n_outer_folds": len(folds),
            "n_predictions": len(predictions), "n_tuning_records": len(tuning), "n_coefficients": len(coefficients),
            "checks": ["patient_alignment", "recomputed_held_site_AUC", "three_site_macro", "full_source_transfer_grid", "source_institution_isolation", "source_means", "stronger_regularization_tie_rule", "frozen_coordinate_sets", "active_coordinate_counts", "reported_dimensions", "zero_leakage_flags", "no_failed_or_unconverged_fits"],
            "artifact_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
