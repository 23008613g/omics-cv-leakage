"""Full-pipeline permutation test and the ``audit`` entry point."""
from __future__ import annotations

from typing import Callable

import numpy as np
from joblib import Parallel, delayed

from .core import AuditResult, evaluate_protocols, logistic_l2
from .preprocess import StandardPreprocessor
from .splits import Splitter, default_splitter
from .steps import BUILTIN, SupervisedStep


def permute_labels(y, groups, rng: np.random.Generator) -> np.ndarray:
    """Permute labels at the level of the exchangeable unit.

    Without groups, samples are exchanged. With groups whose label is constant,
    group labels are exchanged across groups. When labels vary within groups
    (for example, paired samples), labels are exchanged within each group.
    """
    y = np.asarray(y)
    if groups is None:
        return rng.permutation(y)
    g = np.asarray(groups)
    units, inverse = np.unique(g, return_inverse=True)
    per_unit = [np.unique(y[inverse == u]) for u in range(units.size)]
    if all(v.size == 1 for v in per_unit):
        unit_labels = np.array([v[0] for v in per_unit])
        return rng.permutation(unit_labels)[inverse]
    out = y.copy()
    for u in range(units.size):
        rows = np.flatnonzero(inverse == u)
        out[rows] = rng.permutation(y[rows])
    return out


def empirical_p(observed: float, null) -> float:
    """Right-tailed empirical P, (b + 1) / (B + 1), counting ties against detection."""
    null = np.asarray(null, dtype=float)
    return float((1 + np.count_nonzero(null >= observed)) / (null.size + 1))


def _resolve(step) -> SupervisedStep:
    if isinstance(step, str):
        if step not in BUILTIN:
            raise ValueError(f"unknown step {step!r}; choose from {sorted(BUILTIN)} or pass a step object")
        return BUILTIN[step]()
    return step


def audit(X, y, step="top_q", *, groups=None, splitter: Splitter | None = None,
          preprocessor: Callable[[], object] = StandardPreprocessor,
          classifier: Callable[[int], object] = logistic_l2, n_permutations: int = 199,
          random_state: int = 0, n_jobs: int = 1) -> AuditResult:
    """Compare the two protocols on one dataset and, optionally, test each against its own null.

    Every permutation reruns the complete pipeline of both arms, including a new
    split plan for the permuted labels, so each arm is compared with a null that
    carries the same optimism as the arm itself.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("y must contain exactly two classes")
    y01 = (y == classes[1]).astype(int)
    if X.ndim != 2 or X.shape[0] != y01.size:
        raise ValueError("X must be a matrix with one row per label")
    step = _resolve(step)
    splitter = splitter or default_splitter(random_state=random_state)
    kw = dict(preprocessor=preprocessor, classifier=classifier, seed=random_state)
    observed = evaluate_protocols(X, y01, splitter(y01, groups), step, **kw)
    result = AuditResult(step=step.name, n_samples=X.shape[0], n_features=X.shape[1],
                         pre_split_auc=observed["pre_split_auc"],
                         fold_contained_auc=observed["fold_contained_auc"], gap=observed["gap"],
                         chosen_candidates=observed["chosen_candidates"])
    if n_permutations <= 0:
        return result
    seeds = np.random.SeedSequence(random_state).spawn(n_permutations)

    def one(seed):
        yp = permute_labels(y01, groups, np.random.default_rng(seed))
        r = evaluate_protocols(X, yp, splitter(yp, groups), step, **kw)
        return r["pre_split_auc"], r["fold_contained_auc"]

    nulls = Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in seeds)
    result.n_permutations = n_permutations
    result.pre_split_null = [float(a) for a, _ in nulls]
    result.fold_contained_null = [float(b) for _, b in nulls]
    result.p_pre_split = empirical_p(result.pre_split_auc, result.pre_split_null)
    result.p_fold_contained = empirical_p(result.fold_contained_auc, result.fold_contained_null)
    return result
