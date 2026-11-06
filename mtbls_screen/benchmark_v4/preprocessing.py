"""Fold-fitted raw transforms and preprocessing for the v4 benchmark."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import StandardScaler


TRANSFORM_KINDS = (
    "already_scaled",
    "auto_nonnegative",
    "intensity",
    "counts",
    "microbiome_counts",
)
AUTO_NONNEGATIVE_RULE = "train_finite_nonnegative_max_gt_50_v2"


def _freeze_float(values: NDArray[np.float64]) -> NDArray[np.float64]:
    copied = np.asarray(values, dtype=float).copy()
    copied.setflags(write=False)
    return copied


def _relative_abundance(X: NDArray[np.float64]) -> NDArray[np.float64]:
    finite = np.isfinite(X)
    if np.any(X[finite] < 0):
        raise ValueError("count transforms require non-negative finite values")
    totals = np.where(finite, X, 0.0).sum(axis=1)
    safe_totals = np.where(totals > 0, totals, 1.0)
    return np.where(finite, X / safe_totals[:, None], np.nan)


def _raw_transform(
    X: NDArray[np.float64],
    kind: str,
    *,
    positive_floor: float | None,
    pseudocount: float | None,
    log_applied: bool | None,
) -> NDArray[np.float64]:
    clean = np.where(np.isfinite(X), X, np.nan)
    if kind == "already_scaled":
        return clean
    if kind in {"intensity", "auto_nonnegative"}:
        if kind == "auto_nonnegative" and not log_applied:
            return clean
        if positive_floor is None:
            raise ValueError("log transform requires a training-derived floor")
        return np.where(
            np.isfinite(clean), np.log2(np.maximum(clean, positive_floor)), np.nan
        )
    relative = _relative_abundance(clean)
    if kind == "counts":
        return np.where(
            np.isfinite(relative), np.log2(relative * 1_000_000.0 + 1.0), np.nan
        )
    if kind == "microbiome_counts":
        if pseudocount is None:
            raise ValueError("microbiome transform requires a training pseudocount")
        return np.where(
            np.isfinite(relative), np.log2(relative + pseudocount), np.nan
        )
    raise ValueError(f"transform_kind must be one of {TRANSFORM_KINDS!r}")


@dataclass(frozen=True)
class FoldPreprocessor:
    """Fitted raw transform, feature filter, imputer and scaler."""

    n_features_in: int
    original_feature_indices: tuple[int, ...]
    medians: NDArray[np.float64]
    means: NDArray[np.float64]
    scales: NDArray[np.float64]
    transform_kind: str
    positive_floor: float | None = None
    pseudocount: float | None = None
    log_applied: bool | None = None

    @property
    def n_features_out(self) -> int:
        return len(self.original_feature_indices)

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in:
            raise ValueError(
                f"X must have shape (n, {self.n_features_in}); got {values.shape}"
            )
        transformed = _raw_transform(
            values,
            self.transform_kind,
            positive_floor=self.positive_floor,
            pseudocount=self.pseudocount,
            log_applied=self.log_applied,
        )
        selected = transformed[:, self.original_feature_indices]
        imputed = np.where(np.isfinite(selected), selected, self.medians)
        return np.asarray((imputed - self.means) / self.scales, dtype=float)


def fit_fold_preprocessor(
    X_train: ArrayLike,
    *,
    completeness_threshold: float,
    variance_cap: int | None,
    transform_kind: str,
    auto_nonnegative_rule: str = AUTO_NONNEGATIVE_RULE,
) -> FoldPreprocessor:
    """Fit all data-dependent operations using one training fold only."""
    values = np.asarray(X_train, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("X_train must be a non-empty two-dimensional matrix")
    if transform_kind not in TRANSFORM_KINDS:
        raise ValueError(f"transform_kind must be one of {TRANSFORM_KINDS!r}")
    if auto_nonnegative_rule != AUTO_NONNEGATIVE_RULE:
        raise ValueError(
            f"auto_nonnegative_rule must be {AUTO_NONNEGATIVE_RULE!r}"
        )

    floor = pseudo = None
    auto_log: bool | None = None
    finite = values[np.isfinite(values)]
    if transform_kind == "auto_nonnegative":
        auto_log = bool(
            finite.size
            and np.all(finite >= 0.0)
            and finite.max() > 50.0
        )
    finite_positive = values[np.isfinite(values) & (values > 0)]
    if transform_kind == "intensity" or auto_log:
        if finite_positive.size == 0:
            raise ValueError("intensity training fold has no positive values")
        floor = float(finite_positive.min() / 2.0)
    if transform_kind == "microbiome_counts":
        relative = _relative_abundance(values)
        positive = relative[np.isfinite(relative) & (relative > 0)]
        if positive.size == 0:
            raise ValueError("microbiome training fold has no positive abundance")
        pseudo = float(positive.min() / 2.0)

    transformed = _raw_transform(
        values,
        transform_kind,
        positive_floor=floor,
        pseudocount=pseudo,
        log_applied=auto_log,
    )
    completeness = np.mean(np.isfinite(transformed), axis=0)
    complete_indices = np.flatnonzero(completeness >= completeness_threshold)
    if complete_indices.size == 0:
        raise ValueError("preprocessing removed every feature by completeness")
    complete = transformed[:, complete_indices]
    medians = np.nanmedian(complete, axis=0)
    imputed = np.where(np.isfinite(complete), complete, medians)
    variances = np.var(imputed, axis=0)
    nonconstant = np.isfinite(variances) & (variances > 1e-12)
    indices = complete_indices[nonconstant]
    imputed = imputed[:, nonconstant]
    variances = variances[nonconstant]
    if indices.size == 0:
        raise ValueError("preprocessing removed every constant feature")
    if variance_cap is not None and indices.size > variance_cap:
        ranked = np.lexsort((indices, -variances))[:variance_cap]
        retained = np.sort(ranked)
        indices = indices[retained]
        imputed = imputed[:, retained]
    scaler = StandardScaler().fit(imputed)
    scales = np.asarray(scaler.scale_, dtype=float)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("training scaler encountered a non-positive scale")
    final_medians = np.nanmedian(transformed[:, indices], axis=0)
    return FoldPreprocessor(
        n_features_in=values.shape[1],
        original_feature_indices=tuple(map(int, indices)),
        medians=_freeze_float(final_medians),
        means=_freeze_float(np.asarray(scaler.mean_, dtype=float)),
        scales=_freeze_float(scales),
        transform_kind=transform_kind,
        positive_floor=floor,
        pseudocount=pseudo,
        log_applied=auto_log,
    )


__all__ = [
    "AUTO_NONNEGATIVE_RULE",
    "FoldPreprocessor",
    "TRANSFORM_KINDS",
    "fit_fold_preprocessor",
]
