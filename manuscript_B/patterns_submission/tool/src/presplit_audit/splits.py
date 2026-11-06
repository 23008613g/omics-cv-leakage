"""Outer and inner split plans shared by the two protocols.

Both protocols are scored on the same outer folds, and the fold-contained
protocol tunes on the same inner folds, so their AUCs form a paired comparison.
All indices are raw row indices into the full data matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold


@dataclass(frozen=True)
class OuterFold:
    """One outer fold: training and test rows, and inner folds within the training rows."""

    repeat: int
    train: np.ndarray
    test: np.ndarray
    inner: tuple[tuple[np.ndarray, np.ndarray], ...]


Splitter = Callable[[np.ndarray, "np.ndarray | None"], Sequence[OuterFold]]


def _check_classes(y: np.ndarray, rows: np.ndarray, what: str) -> None:
    if np.unique(y[rows]).size < 2:
        raise ValueError(f"{what} contains a single class; use fewer folds or more samples")


def _inner(y: np.ndarray, groups: np.ndarray | None, train: np.ndarray, n_splits: int,
           seed: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    y_train = y[train]
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        parts = cv.split(np.zeros(train.size), y_train)
    else:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        parts = cv.split(np.zeros(train.size), y_train, groups[train])
    out = []
    for a, b in parts:
        tr, te = train[a], train[b]
        _check_classes(y, tr, "an inner training fold")
        _check_classes(y, te, "an inner test fold")
        out.append((tr, te))
    return tuple(out)


def make_splits(y, groups=None, *, n_splits: int = 5, n_repeats: int = 3, inner_splits: int = 3,
                random_state: int = 0) -> list[OuterFold]:
    """Repeated stratified k-fold outer folds, each with stratified inner folds.

    With ``groups``, every group (for example, a participant) stays within one
    fold, in both the outer and the inner loop.
    """
    y = np.asarray(y)
    if np.unique(y).size != 2:
        raise ValueError("y must contain exactly two classes")
    g = None if groups is None else np.asarray(groups)
    if g is None and np.bincount(np.unique(y, return_inverse=True)[1]).min() < n_splits:
        raise ValueError(f"each class needs at least {n_splits} samples for {n_splits}-fold splitting")
    folds: list[OuterFold] = []
    if g is None:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        pairs = [(k // n_splits, tr, te) for k, (tr, te) in enumerate(cv.split(np.zeros(y.size), y))]
    else:
        pairs = []
        for r in range(n_repeats):
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state + r)
            pairs += [(r, tr, te) for tr, te in cv.split(np.zeros(y.size), y, g)]
    for k, (repeat, train, test) in enumerate(pairs):
        _check_classes(y, train, "an outer training fold")
        _check_classes(y, test, "an outer test fold")
        inner = _inner(y, g, train, inner_splits, random_state + 10_000 + k)
        folds.append(OuterFold(repeat, np.asarray(train), np.asarray(test), inner))
    validate_splits(folds, y.size, groups=g)
    return folds


def default_splitter(*, n_splits: int = 5, n_repeats: int = 3, inner_splits: int = 3,
                     random_state: int = 0) -> Splitter:
    """A splitter that rebuilds the plan for each label vector, as the permutation test requires."""
    def split(y, groups=None):
        return make_splits(y, groups, n_splits=n_splits, n_repeats=n_repeats, inner_splits=inner_splits,
                           random_state=random_state)
    return split


def validate_splits(folds: Sequence[OuterFold], n_samples: int, groups=None) -> None:
    """Check disjointness, one test appearance per sample per repeat, and group integrity."""
    repeats = sorted({f.repeat for f in folds})
    if repeats != list(range(len(repeats))):
        raise ValueError("repeat indices must be 0, 1, ..., R-1")
    for r in repeats:
        seen = np.concatenate([f.test for f in folds if f.repeat == r])
        if seen.size != n_samples or np.unique(seen).size != n_samples:
            raise ValueError(f"repeat {r}: every sample must be tested exactly once")
    g = None if groups is None else np.asarray(groups)
    for f in folds:
        if np.intersect1d(f.train, f.test).size:
            raise ValueError("an outer fold reuses a training sample for testing")
        for tr, te in f.inner:
            if np.setdiff1d(np.concatenate([tr, te]), f.train).size or np.intersect1d(tr, te).size:
                raise ValueError("inner folds must partition the outer training rows")
            if g is not None and np.intersect1d(g[tr], g[te]).size:
                raise ValueError("a group is split across inner folds")
        if g is not None and np.intersect1d(g[f.train], g[f.test]).size:
            raise ValueError("a group is split across an outer training and test fold")
