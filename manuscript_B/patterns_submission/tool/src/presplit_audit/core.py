"""Paired evaluation of the pre-split and fold-contained protocols."""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .preprocess import StandardPreprocessor
from .splits import OuterFold, validate_splits
from .steps import SupervisedStep

ARMS = ("pre_split", "fold_contained")
SeedSchedule = Callable[..., int]
"""Maps a fitting context to an integer seed. Contexts: ("global",) for the pre-split supervised fit;
("inner_representation", outer, inner) and ("inner_classifier", outer, inner, candidate) in the tuning
loop; ("pre_split_classifier", outer), ("outer_representation", outer) and
("fold_contained_classifier", outer) in each outer fold. The default gives every context the same seed."""


def logistic_l2(seed: int = 0) -> LogisticRegression:
    """Default classifier: L2-penalised logistic regression (C = 1, liblinear, no class weights)."""
    return LogisticRegression(C=1.0, solver="liblinear", max_iter=5000, random_state=seed)


def _fit_predict(classifier: Callable[[int], object], seed: int, Xtr, ytr, Xte) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"^sklearn\.")
        warnings.filterwarnings("error", category=ConvergenceWarning)
        model = classifier(seed).fit(Xtr, ytr)
    return np.asarray(model.predict_proba(Xte)[:, 1], dtype=float)


def select_best(mean_auc: Sequence[float]) -> int:
    """Highest mean inner AUC; exact ties go to the earlier candidate."""
    scores = np.asarray(mean_auc, dtype=float)
    finite = np.flatnonzero(np.isfinite(scores))
    if finite.size == 0:
        raise ValueError("no candidate produced a finite inner AUC")
    best = int(finite[0])
    for i in finite[1:]:
        if scores[i] > scores[best]:
            best = int(i)
    return best


def _tune(X, y, fold: OuterFold, step: SupervisedStep, preprocessor, classifier, seeds: SeedSchedule,
          outer: int) -> int:
    """Choose a candidate by inner cross-validation inside the outer training rows."""
    per_candidate: list[list[float]] = [[] for _ in step.candidates]
    for k, (tr, te) in enumerate(fold.inner):
        prep = preprocessor().fit(X[tr])
        a, b = prep.transform(X[tr]), prep.transform(X[te])
        for c, t in enumerate(step.fit_bundle(a, y[tr], seed=seeds("inner_representation", outer, k))):
            scores = _fit_predict(classifier, seeds("inner_classifier", outer, k, c), t(a), y[tr], t(b))
            per_candidate[c].append(float(roc_auc_score(y[te], scores)))
    return select_best([float(np.mean(v)) for v in per_candidate])


def pooled_auc(y: np.ndarray, folds: Sequence[OuterFold], fold_scores: Sequence[np.ndarray]) -> float:
    """AUC of out-of-fold scores after averaging each sample's score over repeats."""
    total, count = np.zeros(y.size), np.zeros(y.size)
    for f, s in zip(folds, fold_scores, strict=True):
        total[f.test] += s
        count[f.test] += 1
    n_repeats = len({f.repeat for f in folds})
    if not np.all(count == n_repeats):
        raise ValueError("every sample must be scored once per repeat")
    return float(roc_auc_score(y, total / count))


def evaluate_protocols(X, y, folds: Sequence[OuterFold], step: SupervisedStep, *,
                       preprocessor: Callable[[], object] = StandardPreprocessor,
                       classifier: Callable[[int], object] = logistic_l2, seed: int = 0,
                       seeds: SeedSchedule | None = None) -> dict:
    """Score both protocols on the same folds with the same tuning choice and classifier.

    pre-split       preprocessing and the supervised step are fitted once on all
                    rows; only the classifier is refitted in each outer fold.
    fold-contained  preprocessing, the supervised step and the classifier are all
                    refitted on each outer training partition.
    The candidate is chosen by the fold-contained inner loop and used by both arms,
    so the arms differ only in where the label-free and supervised steps are fitted.
    ``seeds`` optionally maps each fitting context to its own seed (see ``SeedSchedule``);
    by default every fit uses ``seed``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    validate_splits(folds, y.size)
    seeds = seeds or (lambda *context: seed)
    global_prep = preprocessor().fit(X)
    gx = global_prep.transform(X)
    global_z = [t(gx) for t in step.fit_bundle(gx, y, seed=seeds("global"))]
    scores = {arm: [] for arm in ARMS}
    chosen = []
    for o, fold in enumerate(folds):
        tr, te = fold.train, fold.test
        c = _tune(X, y, fold, step, preprocessor, classifier, seeds, o)
        chosen.append(c)
        z = global_z[c]
        scores["pre_split"].append(_fit_predict(classifier, seeds("pre_split_classifier", o), z[tr], y[tr], z[te]))
        prep = preprocessor().fit(X[tr])
        a, b = prep.transform(X[tr]), prep.transform(X[te])
        t = step.fit_bundle(a, y[tr], seed=seeds("outer_representation", o))[c]
        scores["fold_contained"].append(_fit_predict(classifier, seeds("fold_contained_classifier", o), t(a), y[tr],
                                                     t(b)))
    out = {f"{arm}_auc": pooled_auc(y, folds, scores[arm]) for arm in ARMS}
    out["gap"] = out["pre_split_auc"] - out["fold_contained_auc"]
    out["chosen_candidates"] = chosen
    return out


@dataclass
class AuditResult:
    step: str
    n_samples: int
    n_features: int
    pre_split_auc: float
    fold_contained_auc: float
    gap: float
    chosen_candidates: list[int]
    n_permutations: int = 0
    pre_split_null: list[float] = field(default_factory=list)
    fold_contained_null: list[float] = field(default_factory=list)
    p_pre_split: float | None = None
    p_fold_contained: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Supervised step: {self.step} (n = {self.n_samples} samples, p = {self.n_features} features)",
            f"Fold-contained AUC (performance estimate): {self.fold_contained_auc:.3f}",
            f"Pre-split AUC (supervised step fitted to all samples): {self.pre_split_auc:.3f}",
            f"Gap (pre-split minus fold-contained): {self.gap:+.3f}",
        ]
        if self.n_permutations:
            lines += [
                f"Full-pipeline permutation test, B = {self.n_permutations}:",
                f"  fold-contained P = {self.p_fold_contained:.4f}; null median AUC "
                f"{np.median(self.fold_contained_null):.3f}",
                f"  pre-split P = {self.p_pre_split:.4f}; null median AUC {np.median(self.pre_split_null):.3f}",
            ]
        lines.append("Report the fold-contained AUC as the performance estimate. A pre-split AUC is not a "
                     "performance estimate; its permutation P value tests association only.")
        return "\n".join(lines)
