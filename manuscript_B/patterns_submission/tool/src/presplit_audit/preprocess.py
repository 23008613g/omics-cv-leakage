"""Label-free preprocessing fitted on training rows only."""
from __future__ import annotations

import numpy as np


class StandardPreprocessor:
    """Completeness filter, median imputation, constant-feature removal, variance cap and z-scoring.

    Every parameter is estimated from the rows passed to ``fit`` and applied
    unchanged by ``transform``. Data should already be on an appropriate scale
    (for example, log-transformed intensities); scale transforms whose constants
    depend on the data can be supplied as a custom preprocessor instead.
    """

    def __init__(self, completeness: float = 0.5, variance_cap: int | None = 5000):
        if not 0.0 <= completeness <= 1.0:
            raise ValueError("completeness must lie in [0, 1]")
        if variance_cap is not None and variance_cap < 1:
            raise ValueError("variance_cap must be positive or None")
        self.completeness = completeness
        self.variance_cap = variance_cap

    def fit(self, X) -> "StandardPreprocessor":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be a two-dimensional matrix with at least two rows")
        finite = np.isfinite(X)
        keep = np.flatnonzero(finite.mean(axis=0) >= self.completeness)
        if keep.size == 0:
            raise ValueError("no feature meets the completeness threshold in this training fold")
        sub = np.where(finite[:, keep], X[:, keep], np.nan)
        medians = np.nanmedian(sub, axis=0)
        imputed = np.where(np.isfinite(sub), sub, medians)
        sd = imputed.std(axis=0)
        varying = np.flatnonzero(sd > 0)
        if varying.size == 0:
            raise ValueError("every feature is constant in this training fold")
        if self.variance_cap is not None and varying.size > self.variance_cap:
            order = np.lexsort((varying, -sd[varying]))          # largest variance first, ties by column
            varying = np.sort(varying[order[: self.variance_cap]])
        self.features_ = keep[varying]
        self.medians_ = medians[varying]
        self.means_ = imputed[:, varying].mean(axis=0)
        self.scales_ = sd[varying]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} columns")
        sub = X[:, self.features_]
        sub = np.where(np.isfinite(sub), sub, self.medians_)
        return (sub - self.means_) / self.scales_
