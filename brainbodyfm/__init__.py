"""Prespecified sparse semantic transfer analyses for BrainBodyFM.

The package deliberately keeps the pretrained model frozen.  It provides the
downstream interfaces, source-only nested site-transfer evaluation, controls,
and uncertainty helpers described by the BrainBodyFM analysis protocol.
"""

from .config import (
    DEFAULT_C_GRID,
    DEFAULT_SITES,
    BrainBodyFMConfig,
    load_config,
    save_config,
)
from .evaluation import (
    FoldResult,
    NestedEvaluationResult,
    fit_source_normalization_by_outer_site,
    run_nested_loso,
    run_fixed_interface_loso,
)
from .metrics import (
    age_conditioned_auc,
    age_conditioned_auroc,
    macro_site_auc,
    paired_macro_auc,
)
from .models import ModelSpec, FittedProbe
from .preprocess import SourcePreprocessor
from .splits import SiteSplit, assert_disjoint_subject_ids, iter_site_loso, source_transfer_directions
from .site_decoding import dimension_matched_site_information_analysis, site_information_analysis
from .bootstrap import (
    bootstrap_interval_summary,
    bootstrap_macro_auc,
    paired_bootstrap_differences,
    run_selection_stability,
)
from .controls import (
    random_feature_subsets,
    run_pca_controls,
    run_random_k_controls,
    summarize_random_k_controls,
    sparse_regularization_path,
)
from .benchmark import empirical_tail_probability, run_pure_l1_sensitivity
from .extraction import assemble_cached_representations, write_cached_bundle
from .label_efficiency import run_label_efficiency, stratified_source_sample_indices
from .manifest import (
    MANIFEST_COLUMNS,
    audit_locked_replication,
    build_subject_manifest,
    cohort_by_site,
    read_manifest,
    validate_subject_manifest,
    write_manifest,
)
from .representations import (
    find_subject_ids_sidecar,
    load_cached_representation,
    STABLE4_COLUMNS,
    Representation,
    align_representation,
    infer_latent_columns,
    infer_semantic_columns,
    load_npy_representation,
    load_table_representation,
    representation_qc,
    save_representation,
)
from .taxonomy import (
    PHYSIOLOGY_FAMILIES,
    build_family_taxonomy,
    build_head_taxonomy,
    classify_head,
    coerce_bool_series,
    load_taxonomy,
    write_taxonomy,
)

__all__ = [
    "DEFAULT_C_GRID",
    "DEFAULT_SITES",
    "BrainBodyFMConfig",
    "MANIFEST_COLUMNS",
    "PHYSIOLOGY_FAMILIES",
    "Representation",
    "STABLE4_COLUMNS",
    "FittedProbe",
    "FoldResult",
    "fit_source_normalization_by_outer_site",
    "ModelSpec",
    "NestedEvaluationResult",
    "SourcePreprocessor",
    "SiteSplit",
    "assert_disjoint_subject_ids",
    "age_conditioned_auroc",
    "age_conditioned_auc",
    "assemble_cached_representations",
    "write_cached_bundle",
    "audit_locked_replication",
    "bootstrap_macro_auc",
    "bootstrap_interval_summary",
    "build_family_taxonomy",
    "build_head_taxonomy",
    "build_subject_manifest",
    "classify_head",
    "coerce_bool_series",
    "cohort_by_site",
    "empirical_tail_probability",
    "run_pure_l1_sensitivity",
    "infer_latent_columns",
    "infer_semantic_columns",
    "find_subject_ids_sidecar",
    "load_cached_representation",
    "load_npy_representation",
    "load_table_representation",
    "load_config",
    "macro_site_auc",
    "paired_macro_auc",
    "paired_bootstrap_differences",
    "random_feature_subsets",
    "read_manifest",
    "representation_qc",
    "run_label_efficiency",
    "run_fixed_interface_loso",
    "run_nested_loso",
    "run_pca_controls",
    "run_random_k_controls",
    "summarize_random_k_controls",
    "run_selection_stability",
    "iter_site_loso",
    "save_config",
    "save_representation",
    "sparse_regularization_path",
    "source_transfer_directions",
    "site_information_analysis",
    "dimension_matched_site_information_analysis",
    "stratified_source_sample_indices",
    "validate_subject_manifest",
    "write_manifest",
    "write_taxonomy",
]
