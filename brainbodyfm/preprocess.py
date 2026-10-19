"""Fold-local unsupervised preprocessing.

The transformer has no access to outcomes.  It is nevertheless fit separately
for every source-to-source direction and every outer source pool so that
imputation, scaling, and PCA can never learn from the held-out institution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # scikit-learn is a declared project dependency, but keep imports clear.
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover - import guard for lightweight installs
    PCA = None


@dataclass(frozen=True)
class PreprocessAudit:
    """Evidence about which rows were used to fit a transform."""

    n_rows_fit: int
    fit_subject_ids: tuple[str, ...] = ()
    target_subject_ids: tuple[str, ...] = ()

    @property
    def target_rows_used(self) -> int:
        return len(set(self.fit_subject_ids).intersection(self.target_subject_ids))


class SourcePreprocessor:
    """Median-impute, standardize, and optionally PCA-compress a matrix."""

    def __init__(
        self,
        *,
        n_components: int | None = None,
        random_state: int = 0,
        allow_pca_shrink: bool = True,
    ) -> None:
        if n_components is not None and int(n_components) <= 0:
            raise ValueError("n_components must be positive when provided.")
        self.n_components = None if n_components is None else int(n_components)
        self.random_state = int(random_state)
        self.allow_pca_shrink = bool(allow_pca_shrink)
        self.medians_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.scales_: np.ndarray | None = None
        self.pca_: object | None = None
        self.n_features_in_: int | None = None
        self.output_feature_names_: tuple[str, ...] = ()
        self.audit_: PreprocessAudit | None = None

    @staticmethod
    def _as_matrix(X: object) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            matrix = np.asarray(X.to_numpy(), dtype=float)
        else:
            matrix = np.asarray(X, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if matrix.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {matrix.shape}.")
        if matrix.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        return matrix

    def fit(
        self,
        X: object,
        *,
        subject_ids: Sequence[object] | None = None,
        target_subject_ids: Sequence[object] | None = None,
    ) -> "SourcePreprocessor":
        matrix = self._as_matrix(X)
        finite = np.isfinite(matrix)
        medians = np.zeros(matrix.shape[1], dtype=float)
        for j in range(matrix.shape[1]):
            column = matrix[finite[:, j], j]
            medians[j] = float(np.median(column)) if len(column) else 0.0
        filled = np.where(finite, matrix, medians)
        means = np.mean(filled, axis=0)
        scales = np.std(filled, axis=0, ddof=0)
        scales[~np.isfinite(scales) | (scales < 1e-12)] = 1.0
        standardized = (filled - means) / scales

        self.medians_ = medians
        self.means_ = means
        self.scales_ = scales
        self.n_features_in_ = matrix.shape[1]
        self.pca_ = None
        if self.n_components is None:
            self.output_feature_names_ = tuple(f"feature_{i}" for i in range(matrix.shape[1]))
        else:
            if PCA is None:
                raise RuntimeError("scikit-learn is required for PCA preprocessing.")
            max_components = min(matrix.shape[0], matrix.shape[1])
            if max_components <= 0:
                raise ValueError("Cannot fit PCA with no rows or features.")
            if self.n_components > max_components and not self.allow_pca_shrink:
                raise ValueError(
                    f"n_components={self.n_components} exceeds fit limit {max_components}."
                )
            effective = min(self.n_components, max_components)
            self.pca_ = PCA(n_components=effective, random_state=self.random_state)
            self.pca_.fit(standardized)
            self.output_feature_names_ = tuple(f"PC{i + 1}" for i in range(effective))

        ids = tuple(str(x) for x in subject_ids) if subject_ids is not None else ()
        target_ids = tuple(str(x) for x in target_subject_ids) if target_subject_ids is not None else ()
        self.audit_ = PreprocessAudit(
            n_rows_fit=matrix.shape[0], fit_subject_ids=ids, target_subject_ids=target_ids
        )
        if self.audit_.target_rows_used:
            raise AssertionError("Target subjects were used while fitting a source transform.")
        return self

    def _check_fitted(self) -> None:
        if self.medians_ is None or self.means_ is None or self.scales_ is None or self.n_features_in_ is None:
            raise RuntimeError("SourcePreprocessor must be fit before transform.")

    def transform(self, X: object) -> np.ndarray:
        self._check_fitted()
        matrix = self._as_matrix(X)
        assert self.medians_ is not None and self.means_ is not None and self.scales_ is not None and self.n_features_in_ is not None
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {matrix.shape[1]}.")
        filled = np.where(np.isfinite(matrix), matrix, self.medians_)
        transformed = (filled - self.means_) / self.scales_
        if self.pca_ is not None:
            transformed = self.pca_.transform(transformed)
        return np.asarray(transformed, dtype=float)

    def fit_transform(
        self,
        X: object,
        *,
        subject_ids: Sequence[object] | None = None,
        target_subject_ids: Sequence[object] | None = None,
    ) -> np.ndarray:
        return self.fit(
            X, subject_ids=subject_ids, target_subject_ids=target_subject_ids
        ).transform(X)

    @property
    def output_dim(self) -> int:
        self._check_fitted()
        return len(self.output_feature_names_)

    def inverse_standardize(self, X: object) -> np.ndarray:
        """Convert standardized (pre-PCA) values back to original units."""
        self._check_fitted()
        assert self.means_ is not None and self.scales_ is not None
        matrix = self._as_matrix(X)
        return matrix * self.scales_ + self.means_
