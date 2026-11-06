"""Paired OOF bootstrap uncertainty for the v4 aggregation tables."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from benchmark_v4.aggregation_schema import base_row, error
from benchmark_v4.checkpoints import sha256_json


def _auc(labels: Any, scores: Any) -> float:
    import numpy as np

    y = np.asarray(labels, dtype=int)
    prediction = np.asarray(scores, dtype=float)
    positives, negatives = int(np.sum(y == 1)), int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both classes")
    order = np.argsort(prediction, kind="mergesort")
    sorted_scores = prediction[order]
    ranks = np.empty(prediction.size, dtype=float)
    start = 0
    while start < prediction.size:
        stop = start + 1
        while stop < prediction.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = float(ranks[y == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _oof_vectors(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_samples: int,
    context: str,
    expected_repeats: int | None = None,
) -> tuple[list[int], list[float], int]:
    import numpy as np

    by_index: dict[int, tuple[int, dict[int, float]]] = {}
    for row in rows:
        index = int(row["test_index"])
        repeat = int(row["repeat"])
        label, score = int(row["y_true"]), float(row["y_score"])
        if index not in by_index:
            by_index[index] = label, {repeat: score}
            continue
        previous_label, scores_by_repeat = by_index[index]
        if previous_label != label:
            raise error(context, f"OOF label changed for raw test index {index}")
        if repeat in scores_by_repeat:
            raise error(
                context,
                f"raw test index {index} occurs more than once in repeat {repeat}",
            )
        scores_by_repeat[repeat] = score
    if set(by_index) != set(range(n_samples)):
        raise error(context, "OOF predictions do not cover every raw test index")
    repeat_sets = {tuple(sorted(scores)) for _, scores in by_index.values()}
    if len(repeat_sets) != 1:
        raise error(context, "OOF predictions have unequal repeat coverage")
    observed_repeats = set(next(iter(repeat_sets)))
    repeats = len(observed_repeats)
    if expected_repeats is not None and observed_repeats != set(
        range(expected_repeats)
    ):
        raise error(
            context,
            "each raw test index must occur exactly outer_repeats times, "
            "once per repeat",
        )
    return (
        [by_index[index][0] for index in range(n_samples)],
        [
            float(np.mean([
                by_index[index][1][repeat]
                for repeat in sorted(observed_repeats)
            ]))
            for index in range(n_samples)
        ],
        repeats,
    )


def _bootstrap_gap(
    labels: list[int],
    pre_scores: list[float],
    contained_scores: list[float],
    group_codes: list[int] | None,
    *,
    requested: int,
    seed: int,
) -> tuple[list[float], str, int, str | None]:
    import numpy as np

    y = np.asarray(labels, dtype=int)
    pre = np.asarray(pre_scores, dtype=float)
    contained = np.asarray(contained_scores, dtype=float)
    rng, bootstraps, reason = np.random.default_rng(seed), [], None
    if group_codes is None:
        class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
        unit, n_units = "class_stratified_sample", y.size
        if min(map(len, class_indices)) < 2:
            reason = "fewer_than_two_independent_samples_per_class"

        def draw() -> Any:
            return np.concatenate([
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices
            ])
    else:
        codes = np.asarray(group_codes, dtype=int)
        unique_groups = np.unique(codes)
        members = {group: np.flatnonzero(codes == group) for group in unique_groups}
        group_labels = {group: np.unique(y[index]) for group, index in members.items()}
        constant = all(values.size == 1 for values in group_labels.values())
        n_units = unique_groups.size
        if constant:
            unit = "class_stratified_participant_cluster"
            class_groups = [
                np.asarray([
                    group for group in unique_groups
                    if group_labels[group][0] == label
                ], dtype=int)
                for label in (0, 1)
            ]
            if min(map(len, class_groups)) < 2:
                reason = "fewer_than_two_participant_clusters_per_class"

            def draw() -> Any:
                sampled = np.concatenate([
                    rng.choice(groups, size=len(groups), replace=True)
                    for groups in class_groups
                ])
                return np.concatenate([members[group] for group in sampled])
        else:
            unit = "participant_cluster"
            if n_units < 4:
                reason = "fewer_than_four_participant_clusters"

            def draw() -> Any:
                sampled = rng.choice(unique_groups, size=n_units, replace=True)
                return np.concatenate([members[group] for group in sampled])
    if reason is None:
        for _ in range(requested):
            indices = draw()
            if np.unique(y[indices]).size != 2:
                continue
            bootstraps.append(
                _auc(y[indices], pre[indices])
                - _auc(y[indices], contained[indices])
            )
    return bootstraps, unit, int(n_units), reason


def append_uncertainty(
    checkpoint: Mapping[str, Any],
    methods: Sequence[str],
    provenance: Mapping[str, Any] | None,
    tables: dict[str, list[dict[str, Any]]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> None:
    """Append paired OOF uncertainty without changing mean-fold summaries."""
    if provenance is None:
        return
    n_samples = checkpoint["data_fingerprint"]["x_shape"][0]
    dataset, predictions = checkpoint["dataset"], tables["outer_fold_predictions"]
    for method in sorted(methods):
        arms = {
            arm: [
                row for row in predictions
                if row["dataset"] == dataset and row["method"] == method
                and row["arm"] == arm
            ]
            for arm in ("pre_split", "fold_contained")
        }
        pre_y, pre_scores, pre_repeats = _oof_vectors(
            arms["pre_split"], n_samples=n_samples,
            context=f"{checkpoint['path']}.{method}.pre_split",
            expected_repeats=provenance["outer_repeats"],
        )
        contained_y, contained_scores, contained_repeats = _oof_vectors(
            arms["fold_contained"], n_samples=n_samples,
            context=f"{checkpoint['path']}.{method}.fold_contained",
            expected_repeats=provenance["outer_repeats"],
        )
        if pre_y != contained_y or pre_repeats != contained_repeats:
            raise error(str(checkpoint["path"]), f"{method} paired OOF arms mismatch")
        derived_seed = int(sha256_json({
            "seed": bootstrap_seed, "dataset": dataset, "method": method,
        })[:16], 16)
        bootstraps, unit, n_units, reason = _bootstrap_gap(
            pre_y, pre_scores, contained_scores, provenance["group_codes"],
            requested=bootstrap_replicates, seed=derived_seed,
        )
        valid = len(bootstraps)
        if reason is None and valid < math.ceil(0.9 * bootstrap_replicates):
            reason = "valid_bootstrap_fraction_below_0.90"
        if reason is None:
            import numpy as np

            standard_error = float(np.std(bootstraps, ddof=1))
            ci_low, ci_high = map(float, np.quantile(bootstraps, [0.025, 0.975]))
            variance = standard_error**2
        else:
            standard_error = ci_low = ci_high = variance = None
        pre_auc, contained_auc = _auc(pre_y, pre_scores), _auc(
            contained_y, contained_scores
        )
        tables["dataset_method_uncertainty"].append({
            **base_row(checkpoint), "method": method,
            "comparison": "pre_split_minus_fold_contained",
            "pre_split_oof_auc": pre_auc,
            "fold_contained_oof_auc": contained_auc,
            "gap_oof": pre_auc - contained_auc,
            "standard_error": standard_error, "ci_low": ci_low,
            "ci_high": ci_high, "variance": variance,
            "uncertainty_unit": unit, "n_units": n_units,
            "n_samples": n_samples, "n_repeats": pre_repeats,
            "requested_bootstrap_replicates": bootstrap_replicates,
            "valid_bootstrap_replicates": valid, "bootstrap_seed": bootstrap_seed,
            "uncertainty_reason": reason or "ok",
            "groups_available": provenance["groups_available"],
        })


__all__ = ["append_uncertainty"]
