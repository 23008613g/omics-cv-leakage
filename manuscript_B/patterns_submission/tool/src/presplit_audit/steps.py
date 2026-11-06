"""Supervised steps whose placement relative to the split is audited.

A step fits a *bundle*: one transform per candidate setting, all learned from
the same labelled rows. Candidates are tried in order and the inner loop picks
the one with the highest mean inner AUC (ties go to the earlier candidate).

The five built-in methods of the benchmark (univariate top-q, RFE, HT-sPLS,
PLS-DA and OPLS-DA) follow the formal benchmark code operation by operation, so
that, given the same split plan, preprocessing and seeds, they return the same
numbers. Only RFE uses its seed (liblinear's dual solver shuffles samples).
"""
from __future__ import annotations

import warnings
from typing import Callable, Protocol, Sequence

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC

Transform = Callable[[np.ndarray], np.ndarray]


class SupervisedStep(Protocol):
    name: str
    candidates: tuple[dict, ...]

    def fit_bundle(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> list[Transform]:
        """Fit every candidate on (X, y); return one transform per candidate."""


def _stable_largest(values: np.ndarray, count: int) -> np.ndarray:
    """Indices of the ``count`` largest finite values; ties go to the lower index."""
    finite = np.where(np.isfinite(values), values, -np.inf)
    return np.lexsort((np.arange(values.size), -finite))[:count]


def _columns(selected: np.ndarray) -> Transform:
    cols = np.sort(selected)
    return lambda X: np.asarray(X)[:, cols]


class TopQ:
    """Keep the q features with the largest two-sample F statistic."""

    name = "top_q"

    def __init__(self, qs: Sequence[int] = (5, 10, 20, 50)):
        self.candidates = tuple({"q": int(q)} for q in qs)

    def fit_bundle(self, X, y, seed=0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = np.asarray(f_classif(X, y)[0], dtype=float)
        ranking = _stable_largest(scores, X.shape[1])
        return [_columns(ranking[: min(c["q"], X.shape[1])]) for c in self.candidates]


def _vip(X: np.ndarray, scores: np.ndarray, weights: np.ndarray, y_loadings: np.ndarray) -> np.ndarray:
    explained = np.sum(scores * scores, axis=0) * np.sum(y_loadings * y_loadings, axis=0)
    total = float(np.sum(explained))
    if total <= 1e-12:
        return np.sum(np.abs(weights), axis=1)
    return np.sqrt(X.shape[1] * np.sum(weights * weights * explained[None, :], axis=1) / total)


class PLSDA:
    """Partial least squares discriminant analysis; passes the first k latent scores on."""

    name = "pls_da"

    def __init__(self, components: Sequence[int] = (1, 2, 3, 5), max_iter: int = 5000):
        self.candidates = tuple({"n_components": int(k)} for k in components)
        self.max_iter = max_iter

    def _fit(self, X, y, k_max):
        k = min(k_max, X.shape[1], max(1, X.shape[0] - 1))
        model = PLSRegression(n_components=k, scale=False, max_iter=self.max_iter)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"^y residual is constant", category=UserWarning)
            model.fit(X, np.asarray(y, dtype=float))
        t, w = np.asarray(model.x_scores_), np.asarray(model.x_weights_)
        zero = np.flatnonzero(~((np.linalg.norm(t, axis=0) > 1e-12) & (np.linalg.norm(w, axis=0) > 1e-12)))
        usable = int(zero[0]) if zero.size else k
        if usable < 1:
            raise RuntimeError("PLS-DA produced no nonzero component")
        return model, usable

    def fit_bundle(self, X, y, seed=0):
        model, usable = self._fit(X, y, max(c["n_components"] for c in self.candidates))
        out = []
        for c in self.candidates:
            count = min(c["n_components"], usable)
            out.append(lambda Z, count=count: np.asarray(model.transform(Z)[:, :count], dtype=float))
        return out


class VIPSelect:
    """Keep features with variable importance in projection above a threshold (one candidate)."""

    name = "vip"

    def __init__(self, n_components: int = 2, threshold: float = 1.0, max_iter: int = 5000):
        self.candidates = ({"n_components": int(n_components), "threshold": float(threshold)},)
        self.pls = PLSDA((n_components,), max_iter=max_iter)
        self.threshold = threshold

    def fit_bundle(self, X, y, seed=0):
        model, usable = self.pls._fit(X, y, self.candidates[0]["n_components"])
        vip = _vip(X, np.asarray(model.x_scores_)[:, :usable], np.asarray(model.x_weights_)[:, :usable],
                   np.asarray(model.y_loadings_)[:, :usable])
        keep = np.flatnonzero(vip > self.threshold)
        return [_columns(keep if keep.size else np.array([int(np.argmax(vip))]))]


class RFE:
    """Recursive feature elimination with a linear support-vector classifier.

    One elimination path serves all candidates: the features are pruned to each
    requested size in decreasing order, refitting the classifier at every stage
    and keeping the largest absolute weights (ties to the lower index).
    """

    name = "rfe"

    def __init__(self, n_features: Sequence[int] = (5, 10, 20, 50), svc_c: float = 0.1, max_iter: int = 50_000):
        self.candidates = tuple({"n_features": int(k)} for k in n_features)
        self.svc_c, self.max_iter = float(svc_c), int(max_iter)

    def fit_bundle(self, X, y, seed=0):
        requested = [c["n_features"] for c in self.candidates]
        targets = sorted({min(value, X.shape[1]) for value in requested}, reverse=True)
        current = np.arange(X.shape[1])
        path: dict[int, np.ndarray] = {}
        for target in targets:
            if current.size > target:
                estimator = LinearSVC(C=self.svc_c, dual="auto", max_iter=self.max_iter, random_state=seed)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator.fit(X[:, current], y)
                if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                    raise RuntimeError(f"RFE LinearSVC did not converge at max_iter={self.max_iter}")
                weights = np.abs(np.asarray(estimator.coef_).reshape(-1))
                current = np.sort(current[_stable_largest(weights, target)])
            path[target] = current.copy()
        return [_columns(path[min(value, X.shape[1])]) for value in requested]


def _sparse_pls(X: np.ndarray, y: np.ndarray, n_components: int, keep_x: int) -> Transform:
    """Hard-threshold sparse PLS by sequential deflation (deterministic)."""
    residual_x = X.copy()
    residual_y = y.astype(float) - float(np.mean(y))
    weights: list[np.ndarray] = []
    loadings: list[np.ndarray] = []
    feasible = min(n_components, max(1, X.shape[0] - 1), X.shape[1])
    for _ in range(feasible):
        covariance = residual_x.T @ residual_y
        support = _stable_largest(np.abs(covariance), min(keep_x, X.shape[1]))
        weight = np.zeros(X.shape[1], dtype=float)
        weight[support] = covariance[support]
        norm = float(np.linalg.norm(weight))
        if norm <= 1e-12:
            break
        weight /= norm
        score = residual_x @ weight
        denominator = float(score @ score)
        if denominator <= 1e-12:
            break
        loading = residual_x.T @ score / denominator
        response_loading = float(residual_y @ score / denominator)
        weights.append(weight)
        loadings.append(loading)
        residual_x = residual_x - np.outer(score, loading)
        residual_y = residual_y - score * response_loading

    def transform(values: np.ndarray) -> np.ndarray:
        residual = np.asarray(values, dtype=float).copy()
        columns = []
        for weight, loading in zip(weights, loadings):
            score = residual @ weight
            columns.append(score)
            residual = residual - np.outer(score, loading)
        return np.column_stack(columns) if columns else np.zeros((residual.shape[0], 1), dtype=float)

    return transform


class HTSPLS:
    """Hard-threshold sparse PLS: per component, keep the keep_x features with the largest
    covariance with the centred binary response, then deflate; passes latent scores on.
    Not the penalised sPLS-DA estimator of mixOmics."""

    name = "ht_spls"

    def __init__(self, settings: Sequence[tuple[int, int]] = ((1, 5), (1, 20), (2, 20), (2, 50))):
        self.candidates = tuple({"n_components": int(k), "keep_x": int(q)} for k, q in settings)

    def fit_bundle(self, X, y, seed=0):
        return [_sparse_pls(X, y, c["n_components"], c["keep_x"]) for c in self.candidates]


class OPLSDA:
    """Orthogonal PLS-DA with one predictive component and 0-3 orthogonal components removed by
    deflation through the orthogonal loadings; passes the predictive score on."""

    name = "opls_da"

    def __init__(self, orthogonal: Sequence[int] = (0, 1, 2, 3)):
        self.candidates = tuple({"n_orthogonal": int(k)} for k in orthogonal)

    def fit_bundle(self, X, y, seed=0):
        centered_y = y.astype(float) - float(np.mean(y))
        predictive_weight = X.T @ centered_y
        norm = float(np.linalg.norm(predictive_weight))
        predictive_weight = predictive_weight / norm if norm > 1e-12 else np.zeros(X.shape[1], dtype=float)
        residual = X.copy()
        orthogonal: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(max(c["n_orthogonal"] for c in self.candidates)):
            predictive_score = residual @ predictive_weight
            denominator = float(predictive_score @ predictive_score)
            if denominator <= 1e-12:
                break
            predictive_loading = residual.T @ predictive_score / denominator
            orthogonal_weight = predictive_loading - (predictive_weight @ predictive_loading) * predictive_weight
            orthogonal_norm = float(np.linalg.norm(orthogonal_weight))
            if orthogonal_norm <= 1e-12:
                break
            orthogonal_weight /= orthogonal_norm
            score = residual @ orthogonal_weight
            score_denominator = float(score @ score)
            if score_denominator <= 1e-12:
                break
            loading = residual.T @ score / score_denominator
            orthogonal.append((orthogonal_weight, loading))
            residual = residual - np.outer(score, loading)

        def make(count: int) -> Transform:
            effective = min(count, len(orthogonal))

            def transform(values: np.ndarray) -> np.ndarray:
                corrected = np.asarray(values, dtype=float).copy()
                for weight, loading in orthogonal[:effective]:
                    corrected = corrected - np.outer(corrected @ weight, loading)
                return np.asarray(corrected @ predictive_weight, dtype=float)[:, None]

            return transform

        return [make(c["n_orthogonal"]) for c in self.candidates]


class CustomStep:
    """Wrap any supervised step: ``fit(X, y, **params)`` must return a transform."""

    def __init__(self, name: str, fit: Callable[..., Transform], candidates: Sequence[dict] = ({},)):
        self.name, self._fit = name, fit
        self.candidates = tuple(dict(c) for c in candidates)

    def fit_bundle(self, X, y, seed=0):
        return [self._fit(X, y, **c) for c in self.candidates]


BUILTIN = {"top_q": TopQ, "rfe": RFE, "ht_spls": HTSPLS, "pls_da": PLSDA, "opls_da": OPLSDA, "vip": VIPSelect}
