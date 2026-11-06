"""Shared code for cross-study and sensitivity analyses.

Binding. Shared formal algorithms use the released evaluator sources, hash-checked through
``validation/validate_against_formal.py``: the configuration, the binary split-plan generator, the seed
derivation and the fold preprocessor, and, through presplit-audit, the five supervised steps that reproduce the
formal methods exactly (Note S5). Multiclass and regression tasks need plans, steps and an evaluator that the
binary benchmark does not have. They are defined here and follow the binary code operation by operation;
``evaluate_generic`` with the binary model and score reproduces ``presplit_audit.evaluate_protocols`` exactly
(tests/test_common.py).

Grouping. Folds keep groups together wherever a study records them: participants, as found by the formal
loader, and families of twins in two MicrobiomeHD obesity studies (ob_goodrich, 532 families; ob_gordon_2008_v2,
54), which the formal loader does not detect. Labels are permuted between groups when they are constant within
every group and between samples otherwise, so that permuted labels are independent of the features either way;
for datasets without groups this is the formal permutation, seed for seed.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "validation"))
import validate_against_formal as V                       # noqa: E402  (binds benchmark_v4, presplit-audit, loaders)

from presplit_audit import HTSPLS, OPLSDA, PLSDA, RFE, OuterFold, TopQ      # noqa: E402
from presplit_audit.core import _fit_predict, logistic_l2, select_best      # noqa: E402
from presplit_audit.splits import validate_splits                          # noqa: E402
from presplit_audit.steps import _columns, _stable_largest                 # noqa: E402

RESULTS = HERE / "results"
ARMS = ("pre_split", "fold_contained")
FAMILY = {"ob_goodrich": "familyid", "ob_gordon_2008_v2": "Family"}
Seeds = Callable[..., int]


def formal_config():
    """The formal configuration, as the formal runners construct it."""
    from benchmark_v4.protocol import ProtocolConfig
    return ProtocolConfig(n_permutations=199)


def derive_seed(base: int, *parts) -> int:
    from benchmark_v4.splitting import derive_seed as formal_derive_seed
    return formal_derive_seed(base, *parts)


def check_formal_code() -> None:
    code = V.verify_formal_code()
    if not all(code.values()):
        raise RuntimeError(f"formal code mismatch: {[k for k, v in code.items() if not v]}")


# ------------------------------------------------------------------ MicrobiomeHD

@dataclass(frozen=True)
class Study:
    """One MicrobiomeHD archive, read and aligned as benchmark_v4.raw_loaders._load_mbhd reads it."""

    study: str
    sample_ids: tuple[str, ...]
    state: np.ndarray            # DiseaseState per sample; the formal healthy labels are mapped to "H"
    counts: np.ndarray           # samples x OTUs
    taxonomy: tuple[str, ...]    # RDP lineage of each OTU
    metadata: pd.DataFrame       # one row per sample, aligned to sample_ids
    groups: np.ndarray | None

    def rows(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        return self.counts[mask], None if self.groups is None else self.groups[mask]


def load_study(study: str) -> Study:
    from benchmark_v4 import raw_loaders as R
    entry = R._catalog_index()["mbhd"][study]
    metadata, counts, reason = R._read_mbhd_tables(entry)
    if reason != "ok" or metadata is None or counts is None:
        raise RuntimeError(f"{study}: {reason}")
    metadata, counts = metadata.copy(), counts.copy()
    metadata.index = metadata.index.astype(str).str.strip()
    counts.columns = [str(column).strip() for column in counts.columns]
    if metadata.index.duplicated().any() or pd.Index(counts.columns).duplicated().any():
        raise RuntimeError(f"{study}: duplicated sample identifiers")
    column = next(c for c in metadata.columns if str(c).lower() in ("diseasestate", "disease_state"))
    raw_state = metadata[column].astype(str).str.strip()
    sample_by_taxon = counts.T
    ids = [str(sample) for sample in sample_by_taxon.index if sample in raw_state.index]
    values = sample_by_taxon.loc[ids].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise RuntimeError(f"{study}: counts must be finite and non-negative")
    state = raw_state.loc[ids].to_numpy(dtype=object)
    healthy = np.array([str(s).lower() in R.mbhd_loader.HEALTHY for s in state])
    state = np.where(healthy, "H", state).astype(object)
    meta = metadata.loc[ids]
    if study in FAMILY:
        groups = meta[FAMILY[study]].astype(str).str.strip().to_numpy(dtype=object)
    else:
        found = R._groups_from_frame(metadata, ids)
        groups = None if found is None else np.asarray(found, dtype=object)
    if groups is not None and np.unique(groups).size == groups.size:
        groups = None                                   # one sample per group: nothing to keep together
    return Study(study, tuple(ids), state, values, tuple(str(t) for t in counts.index), meta, groups)


def genus_lineage(taxon: str) -> str | None:
    """Lineage down to genus ("k__...;g__Name"), or None when the genus is unassigned."""
    parts = [p.strip() for p in str(taxon).split(";")]
    at = next((i for i, p in enumerate(parts) if p.startswith("g__")), None)
    if at is None or parts[at] == "g__":
        return None
    return ";".join(parts[: at + 1])


def genus_counts(study: Study, mask: np.ndarray) -> pd.DataFrame:
    """Read counts summed over the OTUs of each named genus (samples x genera)."""
    keys = [genus_lineage(t) for t in study.taxonomy]
    named = np.flatnonzero([k is not None for k in keys])
    names, codes = np.unique([keys[i] for i in named], return_inverse=True)
    indicator = sp.csr_matrix((np.ones(named.size), (named, codes)), shape=(len(keys), names.size))
    summed = np.asarray((indicator.T @ study.counts[mask].T).T)
    return pd.DataFrame(summed, index=np.asarray(study.sample_ids)[mask], columns=names)


# ------------------------------------------------------------------ plans and permutations

def binary_plan(y, groups, unit_id: str, cfg) -> list[OuterFold]:
    """The formal split plan (benchmark_v4.splitting.materialize_plan) for a binary label vector."""
    from benchmark_v4.splitting import materialize_plan
    return V.formal_folds(materialize_plan(y, cfg, unit_id, groups=groups))


class FoldClassError(ValueError):
    """A stratified fold lacks a class."""


def _parts(y, groups, n_splits: int, seed: int, stratify: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    index = np.zeros(y.size)
    if groups is None:
        cv = (StratifiedKFold if stratify else KFold)(n_splits, shuffle=True, random_state=seed)
        parts = cv.split(index, y)
    else:
        cv = (StratifiedGroupKFold if stratify else GroupKFold)(n_splits, shuffle=True, random_state=seed)
        parts = cv.split(index, y, groups)
    out = [(np.asarray(a, dtype=int), np.asarray(b, dtype=int)) for a, b in parts]
    if stratify:
        classes = np.unique(y)
        for a, b in out:
            if np.unique(y[a]).size != classes.size or np.unique(y[b]).size != classes.size:
                raise FoldClassError("a fold lacks a class")
    return out


def _n_splits(y, groups, wanted: int, stratify: bool) -> int:
    """Folds feasible with groups: at most the number of groups (per class when stratified), as the formal plan
    reduces its outer folds when groups are few."""
    if groups is None:
        return wanted
    cap = (min(np.unique(groups[y == c]).size for c in np.unique(y)) if stratify else np.unique(groups).size)
    if cap < 2:
        raise ValueError("insufficient groups for a split plan")
    return min(wanted, cap)


def _parts_retry(y, groups, wanted: int, stratify: bool, cfg, unit_id: str, *namespace):
    """The first seed is derive_seed(base, unit_id, *namespace); if a stratified fold lacks a class, further
    seeds are derived deterministically (the formal plan retries group splits the same way), up to 256."""
    n_splits = _n_splits(y, groups, wanted, stratify)
    for attempt in range(256):
        extra = () if attempt == 0 else ("attempt", attempt)
        try:
            return _parts(y, groups, n_splits, derive_seed(cfg.seed, unit_id, *namespace, *extra), stratify)
        except FoldClassError:
            continue
    raise ValueError("no split plan with every class in every fold after 256 seeds")


def generic_plan(y, groups, unit_id: str, cfg, *, stratify: bool) -> list[OuterFold]:
    """Repeated k-fold plan with the formal sizes (5 outer folds x 3 repeats, 3 inner folds), stratified on the
    class when ``stratify`` (multiclass) and group-aware when groups are given; seeds derive from ``unit_id``."""
    y = np.asarray(y)
    folds = []
    for r in range(cfg.outer_repeats):
        outer = _parts_retry(y, groups, cfg.outer_splits, stratify, cfg, unit_id, "outer", r)
        for k, (train, test) in enumerate(outer):
            g = None if groups is None else groups[train]
            inner = _parts_retry(y[train], g, cfg.inner_splits, stratify, cfg, unit_id, "inner", r, k)
            folds.append(OuterFold(r, train, test, tuple((train[a], train[b]) for a, b in inner)))
    validate_splits(folds, y.size, groups=groups)
    return folds


def permuted(y, groups, unit_id: str, index: int, cfg) -> np.ndarray:
    """Permuted labels: between groups if labels are constant within every group, else between samples."""
    y = np.asarray(y)
    rng = np.random.default_rng(derive_seed(cfg.seed, unit_id, "label_permutation", index))
    if groups is not None:
        unique, inverse = np.unique(groups, return_inverse=True)
        per_group = [np.unique(y[inverse == g]) for g in range(unique.size)]
        if all(v.size == 1 for v in per_group):
            labels = np.array([v[0] for v in per_group])
            return labels[rng.permutation(unique.size)][inverse]
    return y[rng.permutation(y.size)]


# ------------------------------------------------------------------ steps for other tasks

def _one_hot(y) -> np.ndarray:
    y = np.asarray(y)
    return (y[:, None] == np.unique(y)[None, :]).astype(float)


def _elimination_path(X, requested: Sequence[int], weights_of) -> list:
    """The presplit_audit.RFE path: prune to each requested size in decreasing order, refitting at each stage and
    keeping the largest weights (ties to the lower index)."""
    targets = sorted({min(v, X.shape[1]) for v in requested}, reverse=True)
    current = np.arange(X.shape[1])
    path = {}
    for target in targets:
        if current.size > target:
            current = np.sort(current[_stable_largest(np.asarray(weights_of(X[:, current]), float), target)])
        path[target] = current.copy()
    return [_columns(path[min(v, X.shape[1])]) for v in requested]


def _svc_weights(Z, y, c: float, max_iter: int, seed: int) -> np.ndarray:
    estimator = LinearSVC(C=c, dual="auto", max_iter=max_iter, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(Z, y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError(f"RFE LinearSVC did not converge at max_iter={max_iter}")
    coef = np.asarray(estimator.coef_)
    return np.abs(coef.reshape(-1)) if coef.shape[0] == 1 else np.linalg.norm(coef, axis=0)


class RFEMulticlass(RFE):
    """RFE with the benchmark's linear SVM, one-vs-rest; a feature's weight is the Euclidean norm of its
    coefficients over the classes (with two classes this is the binary RFE)."""

    def fit_bundle(self, X, y, seed=0):
        return _elimination_path(X, [c["n_features"] for c in self.candidates],
                                 lambda Z: _svc_weights(Z, y, self.svc_c, self.max_iter, seed))


class PLSDAMulticlass(PLSDA):
    """PLS-DA on the one-hot class matrix (PLS2); passes the first k latent scores on."""

    def _fit(self, X, y, k_max):
        return super()._fit(X, _one_hot(y), k_max)


class TopQRegression(TopQ):
    """Keep the q features with the largest univariate regression F statistic (largest |correlation|)."""

    def fit_bundle(self, X, y, seed=0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = np.asarray(f_regression(X, np.asarray(y, float))[0], dtype=float)
        ranking = _stable_largest(scores, X.shape[1])
        return [_columns(ranking[: min(c["q"], X.shape[1])]) for c in self.candidates]


class RFERegression(RFE):
    """RFE with ridge regression, alpha = 1/(2C): the squared-loss counterpart of the benchmark's linear SVM (C =
    0.1). Deterministic, so the seed is unused."""

    def fit_bundle(self, X, y, seed=0):
        alpha = 1.0 / (2.0 * self.svc_c)
        return _elimination_path(X, [c["n_features"] for c in self.candidates],
                                 lambda Z: np.abs(Ridge(alpha=alpha).fit(Z, np.asarray(y, float)).coef_))


def make_step(task: str, method: str, cfg):
    """The supervised step of a formal method for a task. HT-sPLS and OPLS-DA are defined for one response, so
    the multiclass task uses top-q, RFE and PLS-DA; for regression PLS-DA, HT-sPLS and OPLS-DA take the continuous
    response as it is, which makes them PLS, sparse PLS and OPLS regression."""
    if task == "binary":
        return V.make_step(method, cfg)
    if task == "multiclass":
        return {"top_k": lambda: TopQ(), "rfe": lambda: RFEMulticlass(svc_c=cfg.rfe_svc_c, max_iter=cfg.rfe_max_iter),
                "pls_da": lambda: PLSDAMulticlass(max_iter=cfg.max_iter)}[method]()
    if task == "regression":
        return {"top_k": lambda: TopQRegression(), "rfe": lambda: RFERegression(svc_c=cfg.rfe_svc_c),
                "spls": lambda: HTSPLS(), "pls_da": lambda: PLSDA(max_iter=cfg.max_iter),
                "opls_da": lambda: OPLSDA()}[method]()
    raise ValueError(task)


TASK_METHODS = {"binary": V.METHODS, "multiclass": ("top_k", "rfe", "pls_da"), "regression": V.METHODS}


# ------------------------------------------------------------------ models and scores

def _multinomial(seed, a, y, b):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"^sklearn\.")
        warnings.filterwarnings("error", category=ConvergenceWarning)
        model = LogisticRegression(C=1.0, max_iter=5000, random_state=seed).fit(a, y)
    return np.asarray(model.predict_proba(b), dtype=float)


def _macro_auc(y, scores):
    return float(roc_auc_score(y, scores, multi_class="ovr", average="macro", labels=np.arange(scores.shape[1])))


MODELS = {
    # binary: the benchmark's L2 logistic regression (C = 1, liblinear) and AUC
    "binary": (lambda seed, a, y, b: _fit_predict(logistic_l2, seed, a, y, b), lambda y, s: float(roc_auc_score(y, s))),
    # multiclass: multinomial L2 logistic regression (C = 1) and macro-averaged one-vs-rest AUC
    "multiclass": (_multinomial, _macro_auc),
    # regression: ridge regression (alpha = 1) and R2 about the mean of all scored samples
    "regression": (lambda seed, a, y, b: Ridge(alpha=1.0).fit(a, y).predict(b), lambda y, s: float(r2_score(y, s))),
}


# ------------------------------------------------------------------ evaluator

def _prepared(X, tr, te, preprocessor, cache: dict | None):
    """Label-free preprocessing fitted on rows ``tr`` and applied to ``tr`` and ``te``. With ``cache``, a fit is
    reused for the same pair of row sets (methods sharing one split plan), which leaves every result unchanged."""
    key = None
    if cache is not None:
        key = hashlib.sha1(np.asarray(tr).tobytes() + b"|" + np.asarray(te).tobytes()).hexdigest()
        if key in cache:
            return cache[key]
    prep = preprocessor().fit(X[tr])
    out = (prep.transform(X[tr]), prep.transform(X[te]))
    if cache is not None:
        cache[key] = out
    return out


def _tune(X, y, fold, step, preprocessor, fit_predict, score, seeds: Seeds, outer: int, cache=None) -> int:
    per_candidate: list[list[float]] = [[] for _ in step.candidates]
    for k, (tr, te) in enumerate(fold.inner):
        a, b = _prepared(X, tr, te, preprocessor, cache)
        for c, t in enumerate(step.fit_bundle(a, y[tr], seed=seeds("inner_representation", outer, k))):
            pred = fit_predict(seeds("inner_classifier", outer, k, c), t(a), y[tr], t(b))
            per_candidate[c].append(float(score(y[te], pred)))
    return select_best([float(np.mean(v)) for v in per_candidate])


def pooled_scores(y, folds, predictions) -> np.ndarray:
    """Out-of-fold predictions with each sample's prediction averaged over repeats."""
    first = np.asarray(predictions[0])
    total, count = np.zeros((y.size,) + first.shape[1:]), np.zeros(y.size)
    for f, s in zip(folds, predictions, strict=True):
        total[f.test] += s
        count[f.test] += 1
    if not np.all(count == len({f.repeat for f in folds})):
        raise ValueError("every sample must be scored once per repeat")
    return total / (count[:, None] if total.ndim == 2 else count)


def pooled(y, folds, predictions, score) -> float:
    """Score of out-of-fold predictions after averaging each sample's prediction over repeats."""
    return float(score(y, pooled_scores(y, folds, predictions)))


def evaluate_generic(X, y, folds, step, *, preprocessor, task: str, seeds: Seeds, cache: dict | None = None,
                     return_scores: bool = False) -> dict:
    """presplit_audit.evaluate_protocols for any task: the same two protocols, the same tuning shared by both
    arms and the same seed contexts, with the task's model and score."""
    fit_predict, score = MODELS[task]
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    validate_splits(folds, y.size)
    every = np.arange(y.size)
    gx = _prepared(X, every, every, preprocessor, cache)[0]
    global_z = [t(gx) for t in step.fit_bundle(gx, y, seed=seeds("global"))]
    predictions = {arm: [] for arm in ARMS}
    chosen = []
    for o, fold in enumerate(folds):
        tr, te = fold.train, fold.test
        c = _tune(X, y, fold, step, preprocessor, fit_predict, score, seeds, o, cache)
        chosen.append(c)
        z = global_z[c]
        predictions["pre_split"].append(fit_predict(seeds("pre_split_classifier", o), z[tr], y[tr], z[te]))
        a, b = _prepared(X, tr, te, preprocessor, cache)
        t = step.fit_bundle(a, y[tr], seed=seeds("outer_representation", o))[c]
        predictions["fold_contained"].append(fit_predict(seeds("fold_contained_classifier", o), t(a), y[tr], t(b)))
    out = {arm: pooled(y, folds, predictions[arm], score) for arm in ARMS}
    out["gap"] = out["pre_split"] - out["fold_contained"]
    out["chosen_candidates"] = chosen
    if return_scores:                                      # pooled out-of-fold predictions, for other metrics
        out["scores"] = {arm: pooled_scores(y, folds, predictions[arm]) for arm in ARMS}
    return out


def final_model_predictions(X_train, y_train, X_test, step, chosen: Sequence[int], *, preprocessor, task: str,
                            rep_seed: int, clf_seed: int) -> tuple[np.ndarray, int]:
    """The hold-out analysis's final model (control_arm/run_v2_arms.mode_holdout): preprocessing, supervised step
    and classifier fitted on all training samples, with the candidate chosen most often across the outer folds
    (ties to the earlier candidate), then applied unchanged to the test samples."""
    counts = Counter(chosen)
    best = min(c for c, v in counts.items() if v == max(counts.values()))
    prep = preprocessor().fit(X_train)
    a, b = prep.transform(X_train), prep.transform(X_test)
    t = step.fit_bundle(a, np.asarray(y_train), seed=rep_seed)[best]
    return MODELS[task][0](clf_seed, t(a), np.asarray(y_train), t(b)), best


def preprocessor_for(cfg, transform_kind: str):
    return lambda: V.FormalPreprocessor(cfg, transform_kind)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)
