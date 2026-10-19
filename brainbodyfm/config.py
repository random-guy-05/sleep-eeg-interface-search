"""Frozen protocol configuration.

All values in :data:`DEFAULT_CONFIG` are intentionally explicit.  A caller may
override them for a smoke test, but the published analysis should load the
committed ``frozen/analysis_config.yaml`` without changing the grids after
seeing held-site results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_SITES: tuple[str, ...] = ("I0002", "I0006", "S0001")
DEFAULT_C_GRID: tuple[float, ...] = (
    0.0001,
    0.0003162278,
    0.001,
    0.003162278,
    0.01,
    0.03162278,
    0.1,
    0.3162278,
    1.0,
    3.16227766,
    10.0,
    31.6227766,
    100.0,
    316.227766,
    1000.0,
    3162.27766,
    10000.0,
)


@dataclass(frozen=True)
class BrainBodyFMConfig:
    """Prespecified BrainBodyFM analysis choices.

    The dataclass is immutable so a running analysis cannot accidentally
    mutate a grid or seed from inside a fold.  ``with_overrides`` can be used
    explicitly for small smoke tests and records the resulting values.
    """

    primary_outcome: str = "outcome"
    age_column: str = "age"
    site_column: str = "site"
    subject_column: str = "subject_id"
    age_window: float = 2.0
    sites: tuple[str, ...] = DEFAULT_SITES
    alpha: float = 0.5
    active_threshold: float = 1e-8
    tie_tolerance: float = 1e-4
    c_grid: tuple[float, ...] = DEFAULT_C_GRID
    random_k_values: tuple[int, ...] = (4, 8, 16, 32)
    random_k_draws: int = 500
    random_k_seed: int = 2026090401
    label_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    label_repetitions: int = 30
    label_seed: int = 2026090402
    stability_bootstraps_per_outer_fold: int = 200
    stability_seed: int = 2026090403
    headline_bootstrap_replicates: int = 20_000
    headline_bootstrap_seed: int = 2026090404
    max_iter_elastic_net: int = 20_000
    max_iter_ridge: int = 10_000
    random_state: int = 2026090401
    imputation_policy: str = "training_median"
    normalization_policy: str = "training_mean_sd"
    primary_sparse: str = "physiology_elastic_net"
    dense_semantic: str = "ridge_841"
    dense_latent: str = "ridge_1024"
    stable4_active_threshold: float = 1e-8

    def __post_init__(self) -> None:
        if len(self.sites) < 2:
            raise ValueError("At least two source sites are required.")
        if not self.sites or any(not str(site) for site in self.sites):
            raise ValueError("sites must contain non-empty identifiers.")
        if self.age_window < 0:
            raise ValueError("age_window must be non-negative.")
        if not self.c_grid or any(float(c) <= 0 for c in self.c_grid):
            raise ValueError("c_grid must contain positive values.")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")
        if self.active_threshold < 0:
            raise ValueError("active_threshold must be non-negative.")
        if self.tie_tolerance < 0:
            raise ValueError("tie_tolerance must be non-negative.")
        if any(k <= 0 for k in self.random_k_values):
            raise ValueError("random_k_values must be positive.")
        if any(not 0 < q <= 1 for q in self.label_fractions):
            raise ValueError("label_fractions must be in (0, 1].")
        if self.random_k_draws <= 0 or self.label_repetitions <= 0:
            raise ValueError("draw/repetition counts must be positive.")
        if self.stability_bootstraps_per_outer_fold <= 0:
            raise ValueError("stability bootstrap count must be positive.")
        if self.headline_bootstrap_replicates <= 0:
            raise ValueError("headline bootstrap count must be positive.")

    def to_mapping(self) -> dict[str, Any]:
        """Return the human-readable nested YAML representation."""
        return {
            "primary_metric": {
                "name": "age_conditioned_auc",
                "age_window": float(self.age_window),
            },
            "overall_summary": {"type": "unweighted_macro_site_mean"},
            "sites": list(self.sites),
            "outer_folds": "leave_one_site_out",
            "inner_tuning": "paired_source_transfer_A_to_B_and_B_to_A",
            "elastic_net": {
                "alpha": float(self.alpha),
                "active_threshold": float(self.active_threshold),
                "C_grid": list(self.c_grid),
                "max_iter": int(self.max_iter_elastic_net),
            },
            "ridge": {"max_iter": int(self.max_iter_ridge)},
            "random_k": {
                "values": list(self.random_k_values),
                "n_draws": int(self.random_k_draws),
                "seed": int(self.random_k_seed),
            },
            "label_efficiency": {
                "fractions": list(self.label_fractions),
                "repetitions": int(self.label_repetitions),
                "seed": int(self.label_seed),
            },
            "selection_stability": {
                "bootstraps_per_outer_fold": int(self.stability_bootstraps_per_outer_fold),
                "seed": int(self.stability_seed),
            },
            "headline_uncertainty": {
                "bootstrap_replicates": int(self.headline_bootstrap_replicates),
                "seed": int(self.headline_bootstrap_seed),
            },
            "hyperparameter_ties": {
                "preference": "stronger_regularization",
                "tolerance": float(self.tie_tolerance),
            },
            "models": {
                "primary_sparse": self.primary_sparse,
                "dense_semantic": self.dense_semantic,
                "dense_latent": self.dense_latent,
            },
            "columns": {
                "outcome": self.primary_outcome,
                "age": self.age_column,
                "site": self.site_column,
                "subject_id": self.subject_column,
            },
            "missing_value_policy": self.imputation_policy,
            "normalization_policy": self.normalization_policy,
            "random_state": int(self.random_state),
            "stopping_rules": {
                "no_post_hoc_grid_refinement": True,
                "latent_runtime_failure": "drop_latent_interface_if_reliable_cache_cannot_be_obtained",
                "convergence": "increase_max_iter_and_record_warning",
                "prohibited_additions": [
                    "AURORA_TRIAD",
                    "new_neural_architecture",
                    "new_foundation_model",
                    "new_dataset",
                    "new_k_values",
                    "clinical_subgroup_fishing",
                    "new_Stable4_coefficient_scheme",
                    "threshold_classification",
                ],
            },
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BrainBodyFMConfig":
        """Build a config from the nested YAML shape used by the protocol."""
        metric = raw.get("primary_metric", {}) or {}
        enet = raw.get("elastic_net", {}) or {}
        ridge = raw.get("ridge", {}) or {}
        random_k = raw.get("random_k", {}) or {}
        labels = raw.get("label_efficiency", {}) or {}
        stability = raw.get("selection_stability", {}) or {}
        uncertainty = raw.get("headline_uncertainty", {}) or {}
        ties = raw.get("hyperparameter_ties", {}) or {}
        columns = raw.get("columns", {}) or {}
        models = raw.get("models", {}) or {}
        return cls(
            primary_outcome=str(columns.get("outcome", raw.get("primary_outcome", "outcome"))),
            age_column=str(columns.get("age", raw.get("age_column", "age"))),
            site_column=str(columns.get("site", raw.get("site_column", "site"))),
            subject_column=str(columns.get("subject_id", raw.get("subject_column", "subject_id"))),
            age_window=float(metric.get("age_window", raw.get("age_window", 2.0))),
            sites=tuple(str(x) for x in raw.get("sites", DEFAULT_SITES)),
            alpha=float(enet.get("alpha", 0.5)),
            active_threshold=float(enet.get("active_threshold", 1e-8)),
            tie_tolerance=float(ties.get("tolerance", 1e-4)),
            c_grid=tuple(float(x) for x in enet.get("C_grid", DEFAULT_C_GRID)),
            random_k_values=tuple(int(x) for x in random_k.get("values", (4, 8, 16, 32))),
            random_k_draws=int(random_k.get("n_draws", 500)),
            random_k_seed=int(random_k.get("seed", 2026090401)),
            label_fractions=tuple(float(x) for x in labels.get("fractions", (0.25, 0.5, 0.75, 1.0))),
            label_repetitions=int(labels.get("repetitions", 30)),
            label_seed=int(labels.get("seed", 2026090402)),
            stability_bootstraps_per_outer_fold=int(stability.get("bootstraps_per_outer_fold", 200)),
            stability_seed=int(stability.get("seed", 2026090403)),
            headline_bootstrap_replicates=int(uncertainty.get("bootstrap_replicates", 20_000)),
            headline_bootstrap_seed=int(uncertainty.get("seed", 2026090404)),
            max_iter_elastic_net=int(enet.get("max_iter", 20_000)),
            max_iter_ridge=int(ridge.get("max_iter", 10_000)),
            imputation_policy=str(raw.get("missing_value_policy", "training_median")),
            normalization_policy=str(raw.get("normalization_policy", "training_mean_sd")),
            primary_sparse=str(models.get("primary_sparse", "physiology_elastic_net")),
            dense_semantic=str(models.get("dense_semantic", "ridge_841")),
            dense_latent=str(models.get("dense_latent", "ridge_1024")),
            random_state=int(raw.get("random_state", 2026090401)),
        )

    def with_overrides(self, **kwargs: Any) -> "BrainBodyFMConfig":
        values = asdict(self)
        values.update(kwargs)
        for key in ("sites", "c_grid", "random_k_values", "label_fractions"):
            if key in values and not isinstance(values[key], tuple):
                values[key] = tuple(values[key])
        return type(self)(**values)


DEFAULT_CONFIG = BrainBodyFMConfig()


def load_config(path: str | Path) -> BrainBodyFMConfig:
    """Load a frozen YAML config and validate it eagerly."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("BrainBodyFM config must be a YAML mapping.")
    return BrainBodyFMConfig.from_mapping(payload)


def save_config(config: BrainBodyFMConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.to_mapping(), sort_keys=False), encoding="utf-8"
    )
