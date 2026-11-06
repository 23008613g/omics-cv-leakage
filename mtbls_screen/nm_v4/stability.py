"""Feature-list stability estimands for the v4 benchmark."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


_NAN = float("nan")


def mean_pairwise_jaccard(feature_sets: Sequence[Iterable[int]]) -> float:
    sets = tuple(frozenset(values) for values in feature_sets)
    if len(sets) < 2:
        return _NAN
    values = []
    for left, right in combinations(sets, 2):
        union = left | right
        values.append(1.0 if not union else len(left & right) / len(union))
    return float(np.mean(values))


def mean_pairwise_kuncheva(
    feature_sets: Sequence[Iterable[int]], *, n_features: int
) -> float:
    """Mean Kuncheva only for equal, nonempty, non-saturated q."""
    sets = tuple(frozenset(values) for values in feature_sets)
    sizes = {len(values) for values in sets}
    if len(sets) < 2 or n_features < 1 or len(sizes) != 1:
        return _NAN
    size = next(iter(sizes))
    if size == 0 or size >= n_features:
        return _NAN
    expected = size * size / n_features
    denominator = size - expected
    return float(
        np.mean(
            [
                (len(left & right) - expected) / denominator
                for left, right in combinations(sets, 2)
            ]
        )
    )


def nogueira_stability(
    feature_sets: Sequence[Iterable[int]], *, n_features: int
) -> float:
    """Nogueira chance-corrected multi-set stability for variable q."""
    sets = tuple(frozenset(values) for values in feature_sets)
    if len(sets) < 2 or n_features < 1:
        return _NAN
    if any(
        any(index < 0 or index >= n_features for index in values)
        for values in sets
    ):
        raise ValueError("feature index outside [0, n_features)")
    selections = np.zeros((len(sets), n_features), dtype=float)
    for row, values in enumerate(sets):
        if values:
            selections[row, list(values)] = 1.0
    fraction = float(selections.sum(axis=1).mean() / n_features)
    denominator = fraction * (1.0 - fraction)
    if denominator <= 0:
        return _NAN
    return float(
        1.0 - np.var(selections, axis=0, ddof=1).mean() / denominator
    )


def feature_stability_summary(
    feature_sets: Sequence[Iterable[int]], *, n_features: int
) -> dict:
    sets = tuple(frozenset(values) for values in feature_sets)
    sizes = [len(values) for values in sets]
    applicable = (
        len(sets) >= 2
        and len(set(sizes)) == 1
        and sizes[0] not in (0, n_features)
    )
    return {
        "n_sets": len(sets),
        "n_features": int(n_features),
        "mean_q": float(np.mean(sizes)) if sizes else _NAN,
        "q_equals_p_any": any(size == n_features for size in sizes),
        "q_equals_p_all": bool(sizes)
        and all(size == n_features for size in sizes),
        "nogueira": nogueira_stability(sets, n_features=n_features),
        "mean_jaccard": mean_pairwise_jaccard(sets),
        "mean_kuncheva": (
            mean_pairwise_kuncheva(sets, n_features=n_features)
            if applicable
            else _NAN
        ),
        "kuncheva_applicable": applicable,
    }


def summarize_feature_stability(
    feature_sets: Sequence[Iterable[int]],
    *,
    eligible_feature_sets: Sequence[Iterable[int]],
    raw_p: int,
) -> dict:
    """Summarize selections on the common fold-eligible raw-feature universe."""
    if raw_p < 1:
        raise ValueError("raw_p must be positive")
    selected = tuple(frozenset(map(int, values)) for values in feature_sets)
    eligible = tuple(
        frozenset(map(int, values)) for values in eligible_feature_sets
    )
    if len(selected) != len(eligible):
        raise ValueError("selected and eligible feature sets must have equal length")
    for fold_index, (chosen, available) in enumerate(
        zip(selected, eligible, strict=True)
    ):
        if any(index < 0 or index >= raw_p for index in available):
            raise ValueError("eligible feature index outside the raw universe")
        if not chosen.issubset(available):
            raise ValueError(
                f"selected feature is not eligible in outer fold {fold_index}"
            )

    common = frozenset.intersection(*eligible) if eligible else frozenset()
    common_order = tuple(sorted(common))
    remap = {feature: index for index, feature in enumerate(common_order)}
    restricted = tuple(
        frozenset(remap[index] for index in values if index in remap)
        for values in selected
    )
    primary = feature_stability_summary(
        restricted, n_features=len(common_order)
    )
    raw_sensitivity = feature_stability_summary(selected, n_features=raw_p)
    saturated = [
        len(chosen) == len(available)
        for chosen, available in zip(selected, eligible, strict=True)
    ]
    coverage = len(common_order) / raw_p
    return {
        **primary,
        # Jaccard needs no chance universe and remains directly interpretable.
        "mean_jaccard": mean_pairwise_jaccard(selected),
        "common_eligible_p": len(common_order),
        "raw_p": int(raw_p),
        "common_eligible_coverage": float(coverage),
        "coverage": float(coverage),
        "q_equals_p_after_preprocessing_any": any(saturated),
        "q_equals_p_after_preprocessing_all": bool(saturated)
        and all(saturated),
        "saturation_any": any(saturated),
        "saturation_all": bool(saturated) and all(saturated),
        "raw_universe_sensitivity": raw_sensitivity,
    }


__all__ = [
    "feature_stability_summary",
    "mean_pairwise_jaccard",
    "mean_pairwise_kuncheva",
    "nogueira_stability",
    "summarize_feature_stability",
]
