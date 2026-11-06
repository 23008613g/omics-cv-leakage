"""Deterministic nested split materialization for the v4 protocol."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)


class SplitConfig(Protocol):
    seed: int
    outer_splits: int
    outer_repeats: int
    inner_splits: int


@dataclass(frozen=True)
class IndexSplit:
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


@dataclass(frozen=True)
class OuterSplit:
    repeat: int
    fold: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    inner_splits: tuple[IndexSplit, ...]
    inner_split_strategy: str
    effective_inner_splits: int


@dataclass(frozen=True)
class SplitPlan:
    outer_splits: tuple[OuterSplit, ...]
    split_strategy: str
    sha256: str
    effective_outer_splits: int
    requested_outer_splits: int
    group_adjustment_reason: str | None = None
    # Retained for schema compatibility. Supplied groups never fall back.
    group_fallback_reason: str | None = None


def derive_seed(base_seed: int, *namespace: object) -> int:
    payload = json.dumps(
        [int(base_seed), *(str(part) for part in namespace)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "big", signed=False)


def binary_labels(y: ArrayLike) -> NDArray[np.int_]:
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be one-dimensional")
    try:
        levels = np.unique(labels)
    except TypeError as exc:
        raise ValueError("y must contain comparable labels") from exc
    if levels.size != 2:
        raise ValueError("y must be binary with exactly two observed classes")
    return np.asarray(labels == levels[1], dtype=int)


def _missing_group(value: object) -> bool:
    return value is None or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    )


def _unique(values: NDArray[np.object_]) -> NDArray[np.object_]:
    try:
        return np.unique(values)
    except TypeError as exc:
        raise ValueError("groups must contain comparable identifiers") from exc


def _group_capacity(y: NDArray[np.int_], groups: NDArray[np.object_]) -> int:
    return min(_unique(groups[y == label]).size for label in (0, 1))


def validate_groups(
    groups: NDArray[np.object_] | None, y: NDArray[np.int_], n_splits: int
) -> tuple[bool, str | None]:
    """Validate supplied groups without ever silently discarding them."""
    if groups is None:
        return False, None
    if groups.ndim != 1 or groups.size != y.size:
        raise ValueError("groups must be one-dimensional and match y")
    if any(_missing_group(value) for value in groups):
        raise ValueError("groups contain missing values")
    _unique(groups)
    capacity = _group_capacity(y, groups)
    if capacity < 2:
        raise ValueError(
            "insufficient_groups: each class needs at least two independent groups"
        )
    adjustment = None
    if capacity < n_splits:
        adjustment = f"outer_splits_reduced_from_{n_splits}_to_{capacity}"
    return True, adjustment


def _ordinary_splits(
    y: NDArray[np.int_], n_splits: int, seed: int
) -> tuple[IndexSplit, ...]:
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return tuple(
        IndexSplit(tuple(map(int, train)), tuple(map(int, test)))
        for train, test in cv.split(np.zeros(y.size), y)
    )


def _group_splits(
    y: NDArray[np.int_], groups: NDArray[np.object_], n_splits: int, seed: int
) -> tuple[IndexSplit, ...]:
    cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    return tuple(
        IndexSplit(tuple(map(int, train)), tuple(map(int, test)))
        for train, test in cv.split(np.zeros(y.size), y, groups)
    )


def _splits_have_both_classes(
    y: NDArray[np.int_], splits: Sequence[IndexSplit]
) -> bool:
    return all(
        np.unique(y[np.asarray(split.train_indices, dtype=int)]).size == 2
        and np.unique(y[np.asarray(split.test_indices, dtype=int)]).size == 2
        for split in splits
    )


def _retry_group_splits(
    y: NDArray[np.int_],
    groups: NDArray[np.object_],
    n_splits: int,
    base_seed: int,
    *namespace: object,
) -> tuple[IndexSplit, ...]:
    for attempt in range(256):
        seed = derive_seed(base_seed, *namespace, "attempt", attempt)
        splits = _group_splits(y, groups, n_splits, seed)
        if _splits_have_both_classes(y, splits):
            return splits
    raise ValueError(
        "group_split_failed: could not produce folds containing both classes "
        "after 256 deterministic seeds"
    )


def materialize_plan(
    y: ArrayLike,
    config: SplitConfig,
    dataset_id: str,
    *,
    groups: ArrayLike | None = None,
    analysis_tag: str = "observed",
) -> SplitPlan:
    """Create one plan shared by all methods for the supplied label vector."""
    del analysis_tag  # RNG rules are identical for every label vector.
    labels = binary_labels(y)
    group_array = None if groups is None else np.asarray(groups, dtype=object)
    use_groups, adjustment = validate_groups(
        group_array, labels, config.outer_splits
    )

    raw_outer: list[IndexSplit] = []
    if use_groups:
        assert group_array is not None
        effective_outer = min(
            config.outer_splits, _group_capacity(labels, group_array)
        )
        for repeat in range(config.outer_repeats):
            raw_outer.extend(
                _retry_group_splits(
                    labels,
                    group_array,
                    effective_outer,
                    config.seed,
                    dataset_id,
                    "outer",
                    repeat,
                )
            )
        strategy = "repeated_stratified_group_kfold"
    else:
        effective_outer = config.outer_splits
        if int(np.bincount(labels, minlength=2).min()) < effective_outer:
            raise ValueError(
                f"each class needs at least {effective_outer} observations"
            )
        cv = RepeatedStratifiedKFold(
            n_splits=effective_outer,
            n_repeats=config.outer_repeats,
            random_state=derive_seed(config.seed, dataset_id, "outer"),
        )
        raw_outer = [
            IndexSplit(tuple(map(int, train)), tuple(map(int, test)))
            for train, test in cv.split(np.zeros(labels.size), labels)
        ]
        strategy = "repeated_stratified_kfold"

    outer: list[OuterSplit] = []
    for outer_index, split in enumerate(raw_outer):
        train = np.asarray(split.train_indices, dtype=int)
        inner_y = labels[train]
        if use_groups:
            assert group_array is not None
            inner_groups = group_array[train]
            effective_inner = min(
                config.inner_splits, _group_capacity(inner_y, inner_groups)
            )
            if effective_inner < 2:
                raise ValueError(
                    "insufficient_groups: an outer training fold cannot support "
                    "at least two group-aware inner folds"
                )
            inner = _retry_group_splits(
                inner_y,
                inner_groups,
                effective_inner,
                config.seed,
                dataset_id,
                "inner",
                outer_index,
            )
            inner_strategy = "stratified_group_kfold"
        else:
            effective_inner = config.inner_splits
            if int(np.bincount(inner_y, minlength=2).min()) < effective_inner:
                raise ValueError(
                    "outer training fold has too few observations for inner CV"
                )
            inner = _ordinary_splits(
                inner_y,
                effective_inner,
                derive_seed(config.seed, dataset_id, "inner", outer_index),
            )
            inner_strategy = "stratified_kfold"
        outer.append(
            OuterSplit(
                outer_index // effective_outer,
                outer_index % effective_outer,
                split.train_indices,
                split.test_indices,
                inner,
                inner_strategy,
                effective_inner,
            )
        )

    serialized: Sequence[dict] = [
        {
            "repeat": item.repeat,
            "fold": item.fold,
            "train": item.train_indices,
            "test": item.test_indices,
            "inner_strategy": item.inner_split_strategy,
            "effective_inner_splits": item.effective_inner_splits,
            "inner": [
                (split.train_indices, split.test_indices)
                for split in item.inner_splits
            ],
        }
        for item in outer
    ]
    hash_payload = {
        "requested_outer_splits": config.outer_splits,
        "effective_outer_splits": effective_outer,
        "strategy": strategy,
        "splits": serialized,
    }
    digest = sha256(
        json.dumps(hash_payload, separators=(",", ":")).encode()
    ).hexdigest()
    return SplitPlan(
        tuple(outer),
        strategy,
        digest,
        effective_outer,
        config.outer_splits,
        adjustment,
        None,
    )


__all__ = [
    "IndexSplit",
    "OuterSplit",
    "SplitPlan",
    "binary_labels",
    "derive_seed",
    "materialize_plan",
    "validate_groups",
]
