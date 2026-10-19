"""Patient-level uncertainty and source-fold selection stability."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import os

import numpy as np
import pandas as pd

from .config import BrainBodyFMConfig, DEFAULT_CONFIG
from .evaluation import _as_matrix
from .execution import run_bounded_fit_tasks
from .metrics import age_conditioned_auroc, macro_site_auc
from .models import ModelSpec, fit_probe
from .preprocess import SourcePreprocessor
from .tuning import tune_source_transfer


def bootstrap_macro_auc(
    predictions: pd.DataFrame,
    *,
    n_replicates: int = 20_000,
    seed: int = 2026090404,
    score_column: str = "score",
    outcome_column: str = "outcome",
    age_column: str = "age",
    site_column: str = "site",
    subject_column: str = "subject_id",
    site_order: Iterable[str] = ("I0002", "I0006", "S0001"),
    age_window: float = 2.0,
) -> pd.DataFrame:
    """Resample patients within site and recompute macro age-AUROC."""
    required = {score_column, outcome_column, age_column, site_column, subject_column}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    frame = predictions.copy()
    if frame[subject_column].astype(str).duplicated().any():
        raise ValueError("patient-level bootstrap requires one prediction per subject")
    grouped = {str(site): group.reset_index(drop=True) for site, group in frame.groupby(site_column, sort=False)}
    order = [str(site) for site in site_order]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for replicate in range(int(n_replicates)):
        row: dict[str, object] = {"replicate": replicate}
        site_values: list[float] = []
        for site in order:
            group = grouped.get(site)
            if group is None or group.empty:
                value = float("nan")
            else:
                take = rng.integers(0, len(group), size=len(group))
                sample = group.iloc[take]
                value = age_conditioned_auroc(sample[outcome_column], sample[score_column], sample[age_column], age_window)
            row[f"site_auc_{site}"] = float(value)
            site_values.append(float(value))
        row["macro_auc"] = (
            float(np.mean(site_values))
            if len(site_values) == len(order) and np.all(np.isfinite(site_values))
            else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_differences(
    predictions: pd.DataFrame,
    *,
    representation_a: str,
    representation_b: str,
    n_replicates: int = 20_000,
    seed: int = 2026090404,
    age_window: float = 2.0,
    site_order: Iterable[str] = ("I0002", "I0006", "S0001"),
) -> pd.DataFrame:
    """Bootstrap paired macro-AUC differences on identical patients."""
    required = {"subject_id", "site", "age", "outcome", "representation", "score"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    columns = ["subject_id", "site", "age", "outcome", "score"]
    a_rows = predictions[predictions["representation"].eq(representation_a)][columns].copy()
    b_rows = predictions[predictions["representation"].eq(representation_b)][columns].copy()
    if a_rows.empty or b_rows.empty:
        raise ValueError("Both requested representations must be present.")
    if a_rows["subject_id"].astype(str).duplicated().any() or b_rows["subject_id"].astype(str).duplicated().any():
        raise ValueError("Each representation must have exactly one prediction per subject.")
    if set(a_rows["subject_id"].astype(str)) != set(b_rows["subject_id"].astype(str)):
        raise ValueError("Paired bootstrap requires identical subject sets.")
    wide = a_rows.merge(
        b_rows,
        on="subject_id",
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if not (
        wide["site_a"].astype(str).eq(wide["site_b"].astype(str)).all()
        and pd.to_numeric(wide["age_a"]).eq(pd.to_numeric(wide["age_b"])).all()
        and pd.to_numeric(wide["outcome_a"]).eq(pd.to_numeric(wide["outcome_b"])).all()
    ):
        raise ValueError("Subject site, age, or outcome metadata differ across representations.")
    wide = wide.rename(
        columns={
            "site_a": "site",
            "age_a": "age",
            "outcome_a": "outcome",
            "score_a": representation_a,
            "score_b": representation_b,
        }
    )
    grouped = {str(site): group.reset_index(drop=True) for site, group in wide.groupby("site", sort=False)}
    order = [str(site) for site in site_order]
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(int(n_replicates)):
        auc_a: list[float] = []
        auc_b: list[float] = []
        site_rows: dict[str, float] = {}
        for site in order:
            group = grouped.get(site)
            if group is None or group.empty:
                continue
            take = rng.integers(0, len(group), size=len(group))
            sample = group.iloc[take]
            a = age_conditioned_auroc(sample["outcome"], sample[representation_a], sample["age"], age_window)
            b = age_conditioned_auroc(sample["outcome"], sample[representation_b], sample["age"], age_window)
            auc_a.append(float(a))
            auc_b.append(float(b))
            site_rows[f"site_diff_{site}"] = float(a - b)
        n_sites = len(order)
        valid = (
            len(auc_a) == n_sites
            and len(auc_b) == n_sites
            and np.all(np.isfinite(auc_a))
            and np.all(np.isfinite(auc_b))
        )
        macro_a = float(np.mean(auc_a)) if valid else float("nan")
        macro_b = float(np.mean(auc_b)) if valid else float("nan")
        rows.append({
            "replicate": replicate,
            "representation_a": representation_a,
            "representation_b": representation_b,
            "macro_auc_a": macro_a,
            "macro_auc_b": macro_b,
            "difference": float(macro_a - macro_b),
            **site_rows,
        })
    return pd.DataFrame(rows)


def bootstrap_interval_summary(
    replicates: pd.DataFrame,
    *,
    value_column: str,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Return a percentile interval from a machine-readable bootstrap table."""
    if value_column not in replicates.columns:
        raise ValueError(f"bootstrap table is missing {value_column!r}")
    if not 0 < float(confidence) < 1:
        raise ValueError("confidence must be between 0 and 1")
    values = pd.to_numeric(replicates[value_column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "n_replicates": int(len(replicates)),
        "n_finite": int(len(values)),
        "bootstrap_median": float(np.median(values)) if len(values) else float("nan"),
        "ci_low": float(np.quantile(values, alpha)) if len(values) else float("nan"),
        "ci_high": float(np.quantile(values, 1.0 - alpha)) if len(values) else float("nan"),
        "confidence": float(confidence),
    }


def _bootstrap_indices_by_site(sites: np.ndarray, source_sites: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    parts: list[np.ndarray] = []
    for site in source_sites:
        indices = np.flatnonzero(sites == str(site))
        if len(indices):
            parts.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(parts) if parts else np.empty(0, dtype=int)


def _fit_stability_task(task):
    bootstrap, matrix, labels, spec, chosen_C, names, source_ids, target_ids = task
    try:
        probe = fit_probe(matrix, labels, spec=spec, C=chosen_C, feature_names=names,
                          source_subject_ids=source_ids, target_subject_ids=target_ids)
    except ValueError:
        return bootstrap, None
    return bootstrap, tuple(probe.active_feature_names)


def run_selection_stability(
    X: object,
    manifest: pd.DataFrame,
    *,
    model_spec: ModelSpec,
    feature_names: Sequence[str],
    family_by_feature: Mapping[str, str] | None = None,
    config: BrainBodyFMConfig = DEFAULT_CONFIG,
    bootstraps_per_outer_fold: int | None = None,
    seed: int | None = None,
    chosen_C_by_outer_site: Mapping[str, float] | None = None,
    checkpoint_path: str | Path | None = None,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bootstrap active heads and aggregate coordinate/family frequencies."""
    matrix = _as_matrix(X)
    if len(feature_names) != matrix.shape[1]:
        raise ValueError("feature_names length does not match X")
    subjects = manifest[config.subject_column].astype(str).to_numpy()
    sites = manifest[config.site_column].astype(str).to_numpy()
    y = pd.to_numeric(manifest[config.primary_outcome]).astype(int).to_numpy()
    ages = pd.to_numeric(manifest[config.age_column]).to_numpy(dtype=float)
    n_boot = int(bootstraps_per_outer_fold or config.stability_bootstraps_per_outer_fold)
    if n_boot <= 0:
        raise ValueError("bootstraps_per_outer_fold must be positive")
    if not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("workers must be an integer from 1 to 4")
    rng = np.random.default_rng(int(seed if seed is not None else config.stability_seed))
    families = dict(family_by_feature or {})
    head_counts: Counter[tuple[str, str]] = Counter()
    family_counts: Counter[tuple[str, str]] = Counter()
    outer_selected_heads: dict[str, set[str]] = {}
    outer_selected_families: dict[str, set[str]] = {}
    audit_rows: list[dict[str, object]] = []
    def checkpoint():
        if checkpoint_path is not None:
            destination = Path(checkpoint_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            pd.DataFrame(audit_rows).sort_values(["outer_site", "bootstrap"]).to_csv(temporary, index=False)
            temporary.replace(destination)
    for target_site in config.sites:
        target_idx = np.flatnonzero(sites == target_site)
        source_sites = tuple(site for site in config.sites if site != target_site)
        source_idx = np.flatnonzero(np.isin(sites, source_sites))
        if chosen_C_by_outer_site and target_site in chosen_C_by_outer_site:
            chosen_C = float(chosen_C_by_outer_site[target_site])
        else:
            tuning = tune_source_transfer(
                matrix[source_idx], y[source_idx], ages[source_idx], sites[source_idx],
                model_spec=model_spec, feature_names=feature_names, subject_ids=subjects[source_idx],
                age_window=config.age_window, tie_tolerance=config.tie_tolerance,
            )
            chosen_C = tuning.chosen_C
        try:
            outer_probe = fit_probe(
                matrix[source_idx], y[source_idx], spec=model_spec, C=chosen_C,
                feature_names=feature_names, source_subject_ids=subjects[source_idx],
                target_subject_ids=subjects[target_idx],
            )
            selected_heads = set(outer_probe.active_feature_names)
        except ValueError:
            selected_heads = set()
        outer_selected_heads[target_site] = selected_heads
        outer_selected_families[target_site] = {
            families[feature]
            for feature in selected_heads
            if feature in families and families[feature]
        }
        def tasks():
            for bootstrap in range(n_boot):
                sampled = _bootstrap_indices_by_site(sites, source_sites, rng)
                yield (bootstrap, matrix[sampled], y[sampled], model_spec, chosen_C,
                       feature_names, subjects[sampled], subjects[target_idx])
        for bootstrap, selected in run_bounded_fit_tasks(_fit_stability_task, tasks(), workers):
            if selected is None:
                audit_rows.append({"outer_site": target_site, "bootstrap": bootstrap, "chosen_C": chosen_C, "valid_fit": False, "n_active": np.nan})
                checkpoint()
                continue
            active = set(selected)
            for feature in active:
                head_counts[(target_site, feature)] += 1
            for family in set(families.get(feature, "") for feature in active if families.get(feature, "")):
                family_counts[(target_site, family)] += 1
            audit_rows.append({"outer_site": target_site, "bootstrap": bootstrap, "chosen_C": chosen_C, "valid_fit": True, "n_active": len(active)})
            checkpoint()
            if os.environ.get("BBFM_PROGRESS"):
                print(f"STABILITY done outer={target_site} draw={bootstrap} C={chosen_C:g} active={len(active)} completed={len(audit_rows)}", flush=True)
        # Counts are normalized below by the requested number, preserving the
        # fact that failed one-class draws were attempted rather than silently
        # discarded.
        for feature in feature_names:
            head_counts.setdefault((target_site, feature), 0)
        for family in set(families.values()):
            if family:
                family_counts.setdefault((target_site, family), 0)

    head_rows = []
    for (outer_site, feature), count in sorted(head_counts.items()):
        head_rows.append({
            "outer_site": outer_site,
            "feature": feature,
            "selected_in_outer_fit": feature in outer_selected_heads.get(outer_site, set()),
            "outer_folds_selected": int(sum(feature in selected for selected in outer_selected_heads.values())),
            "selection_count": int(count),
            "bootstrap_frequency": float(count / n_boot),
        })
    family_rows = []
    for (outer_site, family), count in sorted(family_counts.items()):
        family_rows.append({
            "outer_site": outer_site,
            "physiological_family": family,
            "selected_in_outer_fit": family in outer_selected_families.get(outer_site, set()),
            "outer_folds_selected": int(sum(family in selected for selected in outer_selected_families.values())),
            "selection_count": int(count),
            "bootstrap_frequency": float(count / n_boot),
        })
    return pd.DataFrame(head_rows), pd.DataFrame(family_rows), pd.DataFrame(audit_rows).sort_values(["outer_site", "bootstrap"]).reset_index(drop=True)


# Friendly alias used by analysis scripts.
selection_stability = run_selection_stability


def bootstrap_prediction_panel(
    predictions: pd.DataFrame,
    *,
    n_replicates: int = 20_000,
    seed: int = 2026090404,
    site_order: Sequence[str] = ("I0002", "I0006", "S0001"),
    age_window: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """Bootstrap all interfaces on identical patient draws with exact pair counts.

    Repeated patients contribute multiplicities to eligible positive-negative
    pairs. This is algebraically identical to expanding resampled patient rows,
    including half credit for score ties, but avoids repeated dataframe joins.
    The RNG order is replicate, site, patient, matching bootstrap_macro_auc.
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    required = {"subject_id", "site", "age", "outcome", "representation", "score"}
    if not required.issubset(predictions):
        raise ValueError(f"Missing prediction columns: {sorted(required-set(predictions))}")
    names = predictions.representation.drop_duplicates().astype(str).tolist()
    if not names:
        raise ValueError("Empty prediction panel")
    first = predictions[predictions.representation.eq(names[0])].copy()
    if first.subject_id.astype(str).duplicated().any():
        raise ValueError("Duplicate subjects in prediction panel")
    ids = first.subject_id.astype(str).tolist()
    reference = first.assign(subject_id=first.subject_id.astype(str)).set_index("subject_id")
    matrices = []
    for name in names:
        part = predictions[predictions.representation.eq(name)].copy()
        part["subject_id"] = part.subject_id.astype(str)
        if part.subject_id.duplicated().any() or set(part.subject_id) != set(ids):
            raise ValueError("Every interface must cover identical unique subjects")
        aligned = part.set_index("subject_id").loc[ids]
        for column in ("site", "age", "outcome"):
            if not np.array_equal(aligned[column].to_numpy(), reference[column].to_numpy()):
                raise ValueError(f"Inconsistent patient {column} metadata")
        matrices.append(aligned.score.to_numpy(dtype=float))
    scores = np.column_stack(matrices)
    if not np.isfinite(scores).all():
        raise ValueError("Bootstrap scores must be finite")
    site = reference.site.astype(str).to_numpy()
    age = reference.age.to_numpy(dtype=float)
    labels = reference.outcome.to_numpy(dtype=int)
    if set(site) != set(site_order) or not np.isin(labels, [0, 1]).all() or not np.isfinite(age).all():
        raise ValueError("Invalid site, age or label metadata")
    indices = [np.flatnonzero(site == name) for name in site_order]
    counts = [np.empty((n_replicates, len(idx)), dtype=np.int32) for idx in indices]
    rng = np.random.default_rng(seed)
    for replicate in range(n_replicates):
        for count, idx in zip(counts, indices):
            take = rng.integers(0, len(idx), size=len(idx))
            count[replicate] = np.bincount(take, minlength=len(idx))
    site_auc = np.full((n_replicates, len(site_order), len(names)), np.nan)
    for s, (idx, count) in enumerate(zip(indices, counts)):
        positive = np.flatnonzero(labels[idx] == 1)
        negative = np.flatnonzero(labels[idx] == 0)
        left, right = np.where(np.abs(age[idx][positive, None] - age[idx][negative][None, :]) <= age_window)
        p = positive[left]
        n = negative[right]
        difference = scores[idx][p] - scores[idx][n]
        credit = (difference > 0).astype(float) + .5 * (difference == 0)
        for start in range(0, n_replicates, 256):
            batch = count[start:start+256]
            weights = batch[:, p].astype(float) * batch[:, n]
            denominator = weights.sum(axis=1)
            numerator = weights @ credit
            site_auc[start:start+len(batch), s] = np.divide(
                numerator, denominator[:, None], out=np.full_like(numerator, np.nan),
                where=denominator[:, None] > 0,
            )
    macro = site_auc.mean(axis=1)
    return {
        name: pd.DataFrame({
            "replicate": np.arange(n_replicates),
            **{f"site_auc_{s}": site_auc[:, j, k] for j, s in enumerate(site_order)},
            "macro_auc": macro[:, k],
        })
        for k, name in enumerate(names)
    }
