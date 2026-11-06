"""Pure record and estimand helpers for the supervised protocol."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from .splitting import OuterSplit


def _fold_record(
    outer_index: int,
    split: OuterSplit,
    y: NDArray[np.int_],
    scores: NDArray[np.float64],
    auc: float,
    params: dict,
    features: Iterable[int],
    p_after: int,
    inner_scores: Sequence[float],
) -> dict:
    feature_list = list(map(int, features))
    return {
        "outer_index": outer_index,
        "repeat": split.repeat,
        "fold": split.fold,
        "train_size": len(split.train_indices),
        "test_size": len(split.test_indices),
        "p_after_preprocessing": int(p_after),
        "saturated_after_preprocessing": len(set(feature_list)) == int(p_after),
        "test_indices": list(split.test_indices),
        "y_true": list(map(int, y)),
        "y_score": list(map(float, scores)),
        "auc": float(auc),
        "chosen_params": params,
        "selected_features": feature_list,
        "inner_candidate_scores": list(map(float, inner_scores)),
    }


def _repeat_averaged_oof_vectors(
    folds: Sequence[dict],
    *,
    n_samples: int,
    outer_repeats: int,
    context: str,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Pool OOF scores after averaging each raw sample across repeats."""
    if n_samples < 2 or outer_repeats < 1:
        raise ValueError(f"{context}: invalid sample/repeat dimensions")
    expected_repeats = set(range(outer_repeats))
    by_index: dict[int, tuple[int, dict[int, float]]] = {}
    for position, fold in enumerate(folds):
        fold_context = f"{context}.folds[{position}]"
        repeat = fold.get("repeat")
        if (
            isinstance(repeat, bool)
            or not isinstance(repeat, int)
            or repeat not in expected_repeats
        ):
            raise ValueError(f"{fold_context}: repeat is outside the frozen design")
        indices = fold.get("test_indices")
        labels = fold.get("y_true")
        scores = fold.get("y_score")
        if not all(isinstance(value, (list, tuple)) for value in (indices, labels, scores)):
            raise ValueError(f"{fold_context}: OOF vectors must be sequences")
        if not (len(indices) == len(labels) == len(scores)):
            raise ValueError(f"{fold_context}: OOF vector lengths differ")
        for raw_index, label, score in zip(indices, labels, scores, strict=True):
            if (
                isinstance(raw_index, bool)
                or not isinstance(raw_index, (int, np.integer))
                or not 0 <= int(raw_index) < n_samples
            ):
                raise ValueError(f"{fold_context}: raw test index is outside the dataset")
            if isinstance(label, bool) or int(label) not in (0, 1):
                raise ValueError(f"{fold_context}: OOF labels must be binary")
            numeric_score = float(score)
            if not np.isfinite(numeric_score):
                raise ValueError(f"{fold_context}: OOF score must be finite")
            index, binary_label = int(raw_index), int(label)
            if index not in by_index:
                by_index[index] = binary_label, {repeat: numeric_score}
                continue
            previous_label, repeat_scores = by_index[index]
            if previous_label != binary_label:
                raise ValueError(
                    f"{context}: OOF label changed for raw test index {index}"
                )
            if repeat in repeat_scores:
                raise ValueError(
                    f"{context}: raw test index {index} occurs more than once "
                    f"in repeat {repeat}"
                )
            repeat_scores[repeat] = numeric_score
    if set(by_index) != set(range(n_samples)):
        raise ValueError(f"{context}: OOF predictions do not cover every raw test index")
    for index, (_, repeat_scores) in by_index.items():
        if set(repeat_scores) != expected_repeats:
            raise ValueError(
                f"{context}: raw test index {index} must occur exactly "
                "outer_repeats times, once per repeat"
            )
    labels = np.asarray([by_index[index][0] for index in range(n_samples)], dtype=int)
    scores = np.asarray(
        [
            np.mean([by_index[index][1][repeat] for repeat in range(outer_repeats)])
            for index in range(n_samples)
        ],
        dtype=float,
    )
    return labels, scores


def _paired_pooled_oof_metrics(
    pre_split_folds: Sequence[dict],
    fold_contained_folds: Sequence[dict],
    *,
    n_samples: int,
    outer_repeats: int,
    context: str,
) -> dict[str, float]:
    """Return arm-paired pooled OOF AUCs under the frozen estimand."""
    pre_y, pre_scores = _repeat_averaged_oof_vectors(
        pre_split_folds,
        n_samples=n_samples,
        outer_repeats=outer_repeats,
        context=f"{context}.pre_split",
    )
    contained_y, contained_scores = _repeat_averaged_oof_vectors(
        fold_contained_folds,
        n_samples=n_samples,
        outer_repeats=outer_repeats,
        context=f"{context}.fold_contained",
    )
    if not np.array_equal(pre_y, contained_y):
        raise ValueError(f"{context}: paired OOF arms have different labels")
    pre_auc = float(roc_auc_score(pre_y, pre_scores))
    contained_auc = float(roc_auc_score(contained_y, contained_scores))
    return {
        "pre_split_oof_auc": pre_auc,
        "fold_contained_oof_auc": contained_auc,
        "gap_oof": pre_auc - contained_auc,
    }


def empirical_p_value(observed: float, null: Sequence[float]) -> float:
    """Right-tailed empirical P with the Phipson-Smyth +1 correction."""
    values = np.asarray(null, dtype=float)
    finite = values[np.isfinite(values)]
    return float((1 + np.count_nonzero(finite >= observed)) / (finite.size + 1))


__all__ = ["empirical_p_value"]
