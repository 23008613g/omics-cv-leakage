"""Label-free preprocessing caches for nested benchmark evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .preprocessing import FoldPreprocessor
from .splitting import SplitPlan


class PreprocessingConfig(Protocol):
    completeness_threshold: float
    variance_cap: int | None


PreprocessorFit = Callable[..., FoldPreprocessor]


@dataclass(frozen=True)
class PreparedInner:
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_x: NDArray[np.float64]
    test_x: NDArray[np.float64]


@dataclass(frozen=True)
class PreparedOuter:
    train_x: NDArray[np.float64]
    test_x: NDArray[np.float64]
    eligible_features: tuple[int, ...]
    inner: tuple[PreparedInner, ...]


@dataclass(frozen=True)
class GlobalPreprocessingWorkspace:
    x_sha256: str
    config_sha256: str
    transform_kind: str
    preprocessor: FoldPreprocessor
    transformed_x: NDArray[np.float64]


@dataclass(frozen=True)
class PreprocessingWorkspace:
    plan_sha256: str
    workspace_sha256: str
    global_workspace: GlobalPreprocessingWorkspace
    outer: tuple[PreparedOuter, ...]


def x_sha256(X: NDArray[np.float64]) -> str:
    values = np.ascontiguousarray(X, dtype="<f8")
    payload = np.asarray(values.shape, dtype="<i8").tobytes() + values.tobytes()
    return sha256(payload).hexdigest()


def _readonly(X: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(X, dtype=float)
    values.setflags(write=False)
    return values


def build_global_workspace(
    X: ArrayLike,
    config: PreprocessingConfig,
    *,
    config_sha256: str,
    transform_kind: str,
    fit_preprocessor: PreprocessorFit,
) -> GlobalPreprocessingWorkspace:
    """Fit the label-free full-matrix transform once for pre-split arms."""
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError("X must be two-dimensional")
    prep = fit_preprocessor(values, config, transform_kind=transform_kind)
    return GlobalPreprocessingWorkspace(
        x_sha256(values),
        config_sha256,
        transform_kind,
        prep,
        _readonly(prep.transform(values)),
    )


def build_workspace(
    X: ArrayLike,
    plan: SplitPlan,
    config: PreprocessingConfig,
    *,
    config_sha256: str,
    transform_kind: str,
    fit_preprocessor: PreprocessorFit,
    global_workspace: GlobalPreprocessingWorkspace | None = None,
) -> PreprocessingWorkspace:
    """Materialize outer/inner transforms once for a given split plan."""
    values = np.asarray(X, dtype=float)
    global_cache = global_workspace or build_global_workspace(
        values,
        config,
        config_sha256=config_sha256,
        transform_kind=transform_kind,
        fit_preprocessor=fit_preprocessor,
    )
    if (
        global_cache.x_sha256 != x_sha256(values)
        or global_cache.config_sha256 != config_sha256
        or global_cache.transform_kind != transform_kind
    ):
        raise ValueError("global preprocessing workspace does not match X/config")

    prepared_outer = []
    for split in plan.outer_splits:
        train = np.asarray(split.train_indices, dtype=int)
        test = np.asarray(split.test_indices, dtype=int)
        outer_prep = fit_preprocessor(
            values[train], config, transform_kind=transform_kind
        )
        prepared_inner = []
        for inner in split.inner_splits:
            inner_train = train[np.asarray(inner.train_indices, dtype=int)]
            inner_test = train[np.asarray(inner.test_indices, dtype=int)]
            inner_prep = fit_preprocessor(
                values[inner_train], config, transform_kind=transform_kind
            )
            prepared_inner.append(
                PreparedInner(
                    tuple(map(int, inner_train)),
                    tuple(map(int, inner_test)),
                    _readonly(inner_prep.transform(values[inner_train])),
                    _readonly(inner_prep.transform(values[inner_test])),
                )
            )
        prepared_outer.append(
            PreparedOuter(
                _readonly(outer_prep.transform(values[train])),
                _readonly(outer_prep.transform(values[test])),
                outer_prep.original_feature_indices,
                tuple(prepared_inner),
            )
        )

    workspace_payload = {
        "x_sha256": global_cache.x_sha256,
        "config_sha256": global_cache.config_sha256,
        "transform_kind": transform_kind,
        "plan_sha256": plan.sha256,
        "eligible": [item.eligible_features for item in prepared_outer],
    }
    workspace_hash = sha256(
        json.dumps(workspace_payload, separators=(",", ":")).encode()
    ).hexdigest()
    return PreprocessingWorkspace(
        plan.sha256, workspace_hash, global_cache, tuple(prepared_outer)
    )


def validate_workspace(
    values: NDArray[np.float64],
    plan: SplitPlan,
    *,
    config_sha256: str,
    transform_kind: str,
    workspace: PreprocessingWorkspace,
) -> None:
    global_cache = workspace.global_workspace
    if workspace.plan_sha256 != plan.sha256:
        raise ValueError("preprocessing workspace does not match split plan")
    if (
        global_cache.x_sha256 != x_sha256(values)
        or global_cache.config_sha256 != config_sha256
        or global_cache.transform_kind != transform_kind
    ):
        raise ValueError("preprocessing workspace does not match X/config")
    if len(workspace.outer) != len(plan.outer_splits):
        raise ValueError("preprocessing workspace has the wrong outer-fold count")


__all__ = [
    "GlobalPreprocessingWorkspace",
    "PreparedInner",
    "PreparedOuter",
    "PreprocessingWorkspace",
    "build_global_workspace",
    "build_workspace",
    "validate_workspace",
    "x_sha256",
]
