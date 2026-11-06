"""Five supervised representations used by the v4 benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping
import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC


METHODS = ("top_k", "rfe", "spls", "pls_da", "opls_da")
_GRIDS: Mapping[str, tuple[tuple[tuple[str, int], ...], ...]] = MappingProxyType(
    {
        "top_k": tuple((("k", k),) for k in (5, 10, 20, 50)),
        "rfe": tuple((("n_features", k),) for k in (5, 10, 20, 50)),
        "spls": (
            (("n_components", 1), ("keep_x", 5)),
            (("n_components", 1), ("keep_x", 20)),
            (("n_components", 2), ("keep_x", 20)),
            (("n_components", 2), ("keep_x", 50)),
        ),
        "pls_da": tuple((("n_components", k),) for k in (1, 2, 3, 5)),
        "opls_da": tuple((("n_orthogonal", k),) for k in (0, 1, 2, 3)),
    }
)


def candidate_grid(method: str) -> tuple[dict[str, int], ...]:
    """Return a fresh copy of a method's ordered four-candidate grid."""
    try:
        return tuple(dict(candidate) for candidate in _GRIDS[method])
    except KeyError as exc:
        raise ValueError(f"unknown supervised method: {method!r}") from exc


@dataclass(frozen=True)
class FittedRepresentation:
    transform: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    selected: tuple[int, ...]
    effective_params: tuple[tuple[str, int], ...]
    importance_scores: tuple[float, ...] | None = None


def _stable_largest(values: NDArray[np.float64], count: int) -> NDArray[np.int_]:
    finite = np.where(np.isfinite(values), values, -np.inf)
    return np.lexsort((np.arange(values.size), -finite))[:count]


def _top_k_bundle(
    X: NDArray[np.float64], y: NDArray[np.int_]
) -> tuple[FittedRepresentation, ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = np.asarray(f_classif(X, y)[0], dtype=float)
    ranking = _stable_largest(scores, X.shape[1])
    output = []
    for params in candidate_grid("top_k"):
        count = min(params["k"], X.shape[1])
        selected = tuple(sorted(map(int, ranking[:count])))
        output.append(
            FittedRepresentation(
                lambda values, selected=selected: values[:, selected],
                selected,
                (("effective_k", count),),
            )
        )
    return tuple(output)


def _rfe_bundle(
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    *,
    svc_c: float,
    max_iter: int,
    seed: int,
) -> tuple[FittedRepresentation, ...]:
    requested = [params["n_features"] for params in candidate_grid("rfe")]
    targets = sorted({min(value, X.shape[1]) for value in requested}, reverse=True)
    current = np.arange(X.shape[1])
    path: dict[int, tuple[int, ...]] = {}
    for target in targets:
        if current.size > target:
            estimator = LinearSVC(
                C=svc_c,
                dual="auto",
                max_iter=max_iter,
                random_state=seed,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(X[:, current], y)
            if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                raise RuntimeError(
                    "RFE LinearSVC did not converge at the prespecified "
                    f"max_iter={max_iter}"
                )
            weights = np.abs(np.asarray(estimator.coef_).reshape(-1))
            current = np.sort(current[_stable_largest(weights, target)])
        path[target] = tuple(map(int, current))
    return tuple(
        FittedRepresentation(
            lambda values, selected=path[min(value, X.shape[1])]: values[
                :, selected
            ],
            path[min(value, X.shape[1])],
            (("effective_n_features", min(value, X.shape[1])),),
        )
        for value in requested
    )


def _sparse_pls_representation(
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    n_components: int,
    keep_x: int,
) -> FittedRepresentation:
    """Fit deterministic hard-threshold sparse PLS by sequential deflation."""
    residual_x = X.copy()
    residual_y = y.astype(float) - float(np.mean(y))
    weights: list[NDArray[np.float64]] = []
    loadings: list[NDArray[np.float64]] = []
    selected: set[int] = set()
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
        selected.update(map(int, support))
        residual_x = residual_x - np.outer(score, loading)
        residual_y = residual_y - score * response_loading

    def transform(values: NDArray[np.float64]) -> NDArray[np.float64]:
        residual = values.copy()
        columns = []
        for weight, loading in zip(weights, loadings):
            score = residual @ weight
            columns.append(score)
            residual = residual - np.outer(score, loading)
        return (
            np.column_stack(columns)
            if columns
            else np.zeros((values.shape[0], 1), dtype=float)
        )

    return FittedRepresentation(
        transform,
        tuple(sorted(selected)),
        (
            ("effective_n_components", len(weights)),
            ("effective_keep_x", min(keep_x, X.shape[1])),
        ),
    )


def _spls_bundle(
    X: NDArray[np.float64], y: NDArray[np.int_]
) -> tuple[FittedRepresentation, ...]:
    return tuple(
        _sparse_pls_representation(
            X, y, params["n_components"], params["keep_x"]
        )
        for params in candidate_grid("spls")
    )


def _pls_bundle(
    X: NDArray[np.float64], y: NDArray[np.int_], *, max_iter: int
) -> tuple[FittedRepresentation, ...]:
    requested = [params["n_components"] for params in candidate_grid("pls_da")]
    maximum = min(max(requested), X.shape[1], max(1, X.shape[0] - 1))
    model = PLSRegression(n_components=maximum, scale=False, max_iter=max_iter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"^y residual is constant at iteration \d+$",
                category=UserWarning,
                module=r"^sklearn\.cross_decomposition\._pls$",
            )
            warnings.filterwarnings("error", category=ConvergenceWarning)
            model.fit(X, y.astype(float))
    except ConvergenceWarning as exc:
        raise RuntimeError(
            "PLS-DA PLSRegression did not converge at the prespecified "
            f"max_iter={max_iter}"
        ) from exc
    scores_all = np.asarray(model.x_scores_, dtype=float)
    weights_all = np.asarray(model.x_weights_, dtype=float)
    nonzero = (np.linalg.norm(scores_all, axis=0) > 1e-12) & (
        np.linalg.norm(weights_all, axis=0) > 1e-12
    )
    zero_components = np.flatnonzero(~nonzero)
    effective_maximum = (
        int(zero_components[0]) if zero_components.size else maximum
    )
    if effective_maximum < 1:
        raise RuntimeError("PLS-DA produced no nonzero predictive component")
    output = []
    for value in requested:
        count = min(value, effective_maximum)
        scores = scores_all[:, :count]
        weights = weights_all[:, :count]
        loadings_y = np.asarray(model.y_loadings_[:, :count], dtype=float)
        explained = np.sum(scores * scores, axis=0) * np.sum(
            loadings_y * loadings_y, axis=0
        )
        denominator = float(np.sum(explained))
        if denominator > 1e-12:
            vip = np.sqrt(
                X.shape[1]
                * np.sum(weights * weights * explained[None, :], axis=1)
                / denominator
            )
        else:
            vip = np.sum(np.abs(weights), axis=1)
        selected = tuple(
            sorted(map(int, _stable_largest(vip, min(20, X.shape[1]))))
        )
        output.append(
            FittedRepresentation(
                lambda values, count=count: np.asarray(
                    model.transform(values)[:, :count], dtype=float
                ),
                selected,
                (("effective_n_components", count), ("importance_q", len(selected))),
                tuple(map(float, vip)),
            )
        )
    return tuple(output)


def _opls_bundle(
    X: NDArray[np.float64], y: NDArray[np.int_]
) -> tuple[FittedRepresentation, ...]:
    centered_y = y.astype(float) - float(np.mean(y))
    predictive_weight = X.T @ centered_y
    norm = float(np.linalg.norm(predictive_weight))
    predictive_weight = (
        predictive_weight / norm
        if norm > 1e-12
        else np.zeros(X.shape[1], dtype=float)
    )
    residual = X.copy()
    orthogonal: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    maximum = max(params["n_orthogonal"] for params in candidate_grid("opls_da"))
    for _ in range(maximum):
        predictive_score = residual @ predictive_weight
        denominator = float(predictive_score @ predictive_score)
        if denominator <= 1e-12:
            break
        predictive_loading = residual.T @ predictive_score / denominator
        orthogonal_weight = predictive_loading - (
            predictive_weight @ predictive_loading
        ) * predictive_weight
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

    def make_transform(count: int) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        effective = min(count, len(orthogonal))

        def transform(values: NDArray[np.float64]) -> NDArray[np.float64]:
            corrected = values.copy()
            for weight, loading in orthogonal[:effective]:
                corrected = corrected - np.outer(corrected @ weight, loading)
            # Orthogonal scores are diagnostics, never classifier inputs.
            return np.asarray(corrected @ predictive_weight, dtype=float)[:, None]

        return transform

    output = []
    for params in candidate_grid("opls_da"):
        effective = min(params["n_orthogonal"], len(orthogonal))
        effective_predictive = predictive_weight.copy()
        for weight, loading in reversed(orthogonal[:effective]):
            effective_predictive = effective_predictive - weight * (
                loading @ effective_predictive
            )
        selected = tuple(
            sorted(
                map(
                    int,
                    _stable_largest(
                        np.abs(effective_predictive), min(20, X.shape[1])
                    ),
                )
            )
        )
        output.append(
            FittedRepresentation(
                make_transform(params["n_orthogonal"]),
                selected,
                (
                    ("effective_n_orthogonal", effective),
                    ("importance_q", len(selected)),
                ),
                tuple(map(float, np.abs(effective_predictive))),
            )
        )
    return tuple(output)


def fit_candidate_bundle(
    method: str,
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    *,
    rfe_svc_c: float,
    max_iter: int,
    seed: int,
) -> tuple[FittedRepresentation, ...]:
    """Fit all four candidates, sharing rankings/component paths where safe."""
    if method == "top_k":
        return _top_k_bundle(X, y)
    if method == "rfe":
        return _rfe_bundle(
            X, y, svc_c=rfe_svc_c, max_iter=max_iter, seed=seed
        )
    if method == "spls":
        return _spls_bundle(X, y)
    if method == "pls_da":
        return _pls_bundle(X, y, max_iter=max_iter)
    if method == "opls_da":
        return _opls_bundle(X, y)
    raise ValueError(f"unknown supervised method: {method!r}")


__all__ = [
    "METHODS",
    "FittedRepresentation",
    "candidate_grid",
    "fit_candidate_bundle",
]
