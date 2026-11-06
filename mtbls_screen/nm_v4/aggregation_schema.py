"""Strict schema-v2 loading and observed-result flattening."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from nm_v4.aggregation_checkpoint_schema import (
    AggregationError,
    CHECKPOINT_SUFFIX as CHECKPOINT_SUFFIX,
    OBSERVED_METHOD_METRICS,
    SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS,
    as_auc,
    as_gap,
    as_hash,
    as_int,
    as_list,
    as_mapping,
    as_number,
    as_text,
    error,
    json_text,
    load_checkpoints,
)
from nm_v4.checkpoints import sha256_json
from nm_v4.configuration import STATISTIC_RULE
from nm_v4.stability import summarize_feature_stability


def base_row(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    metadata = as_mapping(
        checkpoint["state"].get("metadata"), f"{checkpoint['path']}.metadata"
    )
    source = metadata.get(
        "source_key", metadata.get("source", metadata.get("platform", ""))
    )
    if not isinstance(source, str):
        raise error(str(checkpoint["path"]), "metadata source must be text")
    return {
        "dataset": checkpoint["dataset"],
        "dataset_id": checkpoint["dataset"],
        "source": source,
    }


def analysis_metadata(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    context = f"{checkpoint['path']}.metadata"
    metadata = as_mapping(checkpoint["state"].get("metadata"), context)
    fingerprint = checkpoint["data_fingerprint"]
    if fingerprint is None:
        raise error(context, "observed result lacks data fingerprint metadata")
    n_samples = as_int(metadata.get("n_samples"), f"{context}.n_samples", minimum=1)
    n_features = as_int(
        metadata.get("n_features_raw"), f"{context}.n_features_raw", minimum=1
    )
    if [n_samples, n_features] != fingerprint["x_shape"]:
        raise error(context, "metadata dimensions/data fingerprint mismatch")
    if metadata.get("transform_kind") != fingerprint["transform_kind"]:
        raise error(context, "transform kind/data fingerprint mismatch")
    if not as_text(metadata.get("source_key"), f"{context}.source_key"):
        raise error(context, "source_key is empty")
    counts = as_mapping(metadata.get("class_counts"), f"{context}.class_counts")
    if set(counts) != {"0", "1"}:
        raise error(context, "class_counts must contain binary labels 0 and 1")
    normalized = {
        label: as_int(value, f"{context}.class_counts.{label}", minimum=1)
        for label, value in counts.items()
    }
    if sum(normalized.values()) != n_samples:
        raise error(context, "class_counts do not sum to n_samples")
    return metadata


def provenance(
    checkpoint: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    context = f"{checkpoint['path']}.observed.data_provenance"
    value = as_mapping(observed.get("data_provenance"), context)
    raw_mapping = as_mapping(
        value.get("selected_feature_ids"), f"{context}.selected_feature_ids"
    )
    feature_map: dict[int, str] = {}
    for raw_index, feature_id in raw_mapping.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise error(context, f"invalid raw feature index {raw_index!r}") from exc
        if index < 0 or str(index) != str(raw_index):
            raise error(context, f"invalid raw feature index {raw_index!r}")
        if not isinstance(feature_id, str) or not feature_id:
            raise error(context, f"invalid feature ID for raw index {index}")
        feature_map[index] = feature_id
    available = value.get("groups_available")
    if not isinstance(available, bool):
        raise error(context, "groups_available must be boolean")
    n_groups = as_int(value.get("n_groups"), f"{context}.n_groups")
    codes = value.get("group_codes")
    if available != (n_groups > 0) or available != (codes is not None):
        raise error(context, "group availability/count/codes disagree")
    if codes is not None:
        codes = as_list(codes, f"{context}.group_codes")
        if any(isinstance(code, bool) or not isinstance(code, int) or code < 0 for code in codes):
            raise error(context, "group_codes must be non-negative integers")
        if sorted(set(codes)) != list(range(n_groups)):
            raise error(context, "group_codes are not canonical first-seen codes")
    codes_hash = as_hash(
        value.get("group_codes_sha256"), f"{context}.group_codes_hash",
        nullable=not available,
    )
    if codes is not None and sha256_json(codes) != codes_hash:
        raise error(context, "group_codes SHA256 mismatch")
    fingerprint = checkpoint["data_fingerprint"]
    if fingerprint is None:
        raise error(context, "observed provenance lacks data fingerprint")
    raw_p = fingerprint["x_shape"][1]
    if any(index >= raw_p for index in feature_map):
        raise error(context, "selected feature mapping index is outside raw p")
    if len(codes or []) not in (0, fingerprint["x_shape"][0]):
        raise error(context, "group_codes length does not match raw samples")
    for key in ("feature_ids_sha256", "sample_ids_sha256", "groups_sha256"):
        if value.get(key) != fingerprint[key]:
            raise error(context, f"provenance/data fingerprint {key} mismatch")
    if codes_hash != fingerprint["group_codes_sha256"]:
        raise error(context, "provenance/data fingerprint group_codes SHA256 mismatch")
    if codes is not None:
        first_seen = list(dict.fromkeys(codes))
        if first_seen != list(range(n_groups)):
            raise error(context, "group_codes are not canonical first-seen codes")
    return {
        "feature_ids_sha256": as_hash(
            value.get("feature_ids_sha256"), f"{context}.feature_ids_hash"
        ),
        "sample_ids_sha256": as_hash(
            value.get("sample_ids_sha256"), f"{context}.sample_ids_hash"
        ),
        "groups_sha256": as_hash(
            value.get("groups_sha256"), f"{context}.groups_hash",
            nullable=not available,
        ),
        "group_codes_sha256": codes_hash, "selected_map": feature_map,
        "groups_available": available, "n_groups": n_groups,
        "group_codes": codes,
    }


def _outer_preprocessing(
    checkpoint: Mapping[str, Any], observed: Mapping[str, Any]
) -> tuple[dict[tuple[int, int, int], tuple[int, tuple[int, ...]]], dict[str, Any]]:
    context = f"{checkpoint['path']}.observed"
    requested = as_int(
        observed.get("requested_outer_splits"), f"{context}.requested_outer_splits",
        minimum=2,
    )
    effective = as_int(
        observed.get("effective_outer_splits"), f"{context}.effective_outer_splits",
        minimum=2,
    )
    if effective > requested:
        raise error(context, "effective_outer_splits exceeds requested")
    details = {
        "observed_split_plan_sha256": as_hash(
            observed.get("split_plan_sha256"),
            f"{context}.split_plan_sha256",
        ),
        "split_strategy": as_text(
            observed.get("split_strategy"), f"{context}.split_strategy"
        ),
        "requested_outer_splits": requested,
        "effective_outer_splits": effective,
        "group_adjustment_reason": as_text(
            observed.get("group_adjustment_reason"),
            f"{context}.group_adjustment_reason", nullable=True,
        ),
        "preprocessing_workspace_sha256": as_hash(
            observed.get("preprocessing_workspace_sha256"),
            f"{context}.preprocessing_workspace_sha256",
        ),
        "group_fallback_reason": as_text(
            observed.get("group_fallback_reason"),
            f"{context}.group_fallback_reason", nullable=True,
        ),
    }
    records = as_list(
        observed.get("outer_fold_preprocessing"),
        f"{context}.outer_fold_preprocessing",
    )
    if not records:
        raise error(context, "outer_fold_preprocessing is empty")
    raw_p = checkpoint["data_fingerprint"]["x_shape"][1]
    indexed: dict[tuple[int, int, int], tuple[int, tuple[int, ...]]] = {}
    for position, raw_record in enumerate(records):
        record_context = f"{context}.outer_fold_preprocessing[{position}]"
        record = as_mapping(raw_record, record_context)
        key = (
            as_int(record.get("outer_index"), f"{record_context}.outer_index"),
            as_int(record.get("repeat"), f"{record_context}.repeat"),
            as_int(record.get("fold"), f"{record_context}.fold"),
        )
        eligible = tuple(
            as_int(value, f"{record_context}.eligible_features")
            for value in as_list(
                record.get("eligible_features"), f"{record_context}.eligible_features"
            )
        )
        p_after = as_int(
            record.get("p_after_preprocessing"), f"{record_context}.p_after",
            minimum=1,
        )
        if key in indexed or len(eligible) != p_after:
            raise error(record_context, "duplicate fold or p_after/eligible mismatch")
        if tuple(sorted(set(eligible))) != eligible or any(index >= raw_p for index in eligible):
            raise error(record_context, "eligible features are not sorted unique raw indices")
        indexed[key] = p_after, eligible
    outer_indices = sorted(key[0] for key in indexed)
    if outer_indices != list(range(len(indexed))):
        raise error(context, "outer preprocessing indices are not contiguous")
    repeat_folds: dict[int, set[int]] = {}
    for _, repeat, fold in indexed:
        repeat_folds.setdefault(repeat, set()).add(fold)
    if sorted(repeat_folds) != list(range(len(repeat_folds))) or any(
        folds != set(range(effective)) for folds in repeat_folds.values()
    ):
        raise error(context, "outer preprocessing repeat/fold design is malformed")
    details["outer_repeats"] = len(repeat_folds)
    return indexed, details


def _fold_rows(
    checkpoint: Mapping[str, Any], observed: Mapping[str, Any],
    provenance_value: Mapping[str, Any], method: str, arm: str, folds: Any,
    tables: dict[str, list[dict[str, Any]]],
    signatures: dict[tuple[int, int, int], tuple[tuple[int, ...], tuple[int, ...]]],
    eligible_by_fold: Mapping[
        tuple[int, int, int], tuple[int, tuple[int, ...]]
    ] | None,
) -> list[dict[str, Any]]:
    context = f"{checkpoint['path']}.observed.{method}.{arm}_folds"
    records, seen, validated = as_list(folds, context), set(), []
    common = base_row(checkpoint)
    for position, raw_fold in enumerate(records):
        fold_context = f"{context}[{position}]"
        fold = as_mapping(raw_fold, fold_context)
        outer = as_int(fold.get("outer_index"), f"{fold_context}.outer_index")
        repeat = as_int(fold.get("repeat"), f"{fold_context}.repeat")
        fold_index = as_int(fold.get("fold"), f"{fold_context}.fold")
        key = (outer, repeat, fold_index)
        if key in seen:
            raise error(context, f"duplicate outer fold {key}")
        seen.add(key)
        indices = [as_int(v, f"{fold_context}.test_indices") for v in as_list(
            fold.get("test_indices"), f"{fold_context}.test_indices"
        )]
        labels = [as_int(v, f"{fold_context}.y_true") for v in as_list(
            fold.get("y_true"), f"{fold_context}.y_true"
        )]
        scores = [as_number(v, f"{fold_context}.y_score") for v in as_list(
            fold.get("y_score"), f"{fold_context}.y_score"
        )]
        test_size = as_int(
            fold.get("test_size"), f"{fold_context}.test_size", minimum=1
        )
        if any(label not in (0, 1) for label in labels):
            raise error(fold_context, "y_true must contain binary labels")
        if len(indices) != len(set(indices)) or not (
            len(indices) == len(labels) == len(scores) == test_size
        ):
            raise error(fold_context, "test indices/labels/scores/test_size differ")
        signature = (tuple(indices), tuple(labels))
        if signatures.setdefault(key, signature) != signature:
            raise error(fold_context, "shared outer-fold membership/labels mismatch")
        stored_auc = as_auc(fold.get("auc"), f"{fold_context}.auc")
        recomputed_auc = pooled_auc(labels, scores, f"{fold_context}.recomputed_auc")
        if not math.isclose(
            stored_auc, recomputed_auc, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise error(
                fold_context,
                "stored AUC differs from y_true/y_score",
            )
        auc = recomputed_auc
        p_after = as_int(
            fold.get("p_after_preprocessing"), f"{fold_context}.p_after",
            minimum=1,
        )
        selected = [as_int(v, f"{fold_context}.selected") for v in as_list(
            fold.get("selected_features"), f"{fold_context}.selected"
        )]
        n_samples, raw_p = checkpoint["data_fingerprint"]["x_shape"]
        if any(index >= n_samples for index in indices):
            raise error(fold_context, "test index is outside raw n")
        if p_after > raw_p or any(index >= raw_p for index in selected):
            raise error(fold_context, "p_after or selected index is outside raw p")
        if len(selected) != len(set(selected)):
            raise error(fold_context, "selected raw feature indices are not unique")
        missing = [i for i in selected if i not in provenance_value["selected_map"]]
        if missing:
            raise error(fold_context, f"selected feature mapping is missing raw indices {missing}")
        eligible: tuple[int, ...] | None = None
        if eligible_by_fold is not None:
            if key not in eligible_by_fold:
                raise error(fold_context, "outer preprocessing record is missing")
            expected_p_after, eligible = eligible_by_fold[key]
            if p_after != expected_p_after:
                raise error(
                    fold_context,
                    "fold/preprocessing p_after_preprocessing mismatch",
                )
            if not set(selected).issubset(eligible):
                raise error(fold_context, "selected feature is not fold-eligible")
        saturated = fold.get("saturated_after_preprocessing")
        if not isinstance(saturated, bool) or saturated != (len(selected) == p_after):
            raise error(fold_context, "saturation flag/p_after/selection mismatch")
        params = as_mapping(fold.get("chosen_params"), f"{fold_context}.params")
        inner = as_list(fold.get("inner_candidate_scores"), f"{fold_context}.inner")
        for item in inner:
            as_number(item, f"{fold_context}.inner", nullable=True)
        selected_ids = [provenance_value["selected_map"][i] for i in selected]
        feature_map = {
            str(i): provenance_value["selected_map"][i] for i in sorted(selected)
        }
        train_size = as_int(
            fold.get("train_size"), f"{fold_context}.train", minimum=1
        )
        if train_size + test_size != n_samples:
            raise error(fold_context, "train_size + test_size does not equal raw n")
        tables["outer_fold_diagnostics"].append({
            **common, "method": method, "arm": arm, "outer_index": outer,
            "repeat": repeat, "fold": fold_index, "auc": auc,
            "train_size": train_size,
            "test_size": test_size, "p_after": p_after,
            "p_after_preprocessing": p_after,
            "saturated_after_preprocessing": saturated,
            "selected_raw_indices_json": json_text(selected),
            "eligible_raw_indices_json": (
                None if eligible is None else json_text(eligible)
            ),
            "selected_feature_ids_json": json_text(selected_ids),
            "selected_feature_map_json": json_text(feature_map),
            "chosen_params_json": json_text(dict(params)),
            "inner_candidate_scores_json": json_text(inner),
            "groups_available": provenance_value["groups_available"],
            "n_groups": provenance_value["n_groups"],
            "split_strategy": observed.get("split_strategy"),
            "group_fallback_reason": observed.get("group_fallback_reason"),
            "requested_outer_splits": observed.get("requested_outer_splits"),
            "effective_outer_splits": observed.get("effective_outer_splits"),
            "group_adjustment_reason": observed.get("group_adjustment_reason"),
            "preprocessing_workspace_sha256": observed.get(
                "preprocessing_workspace_sha256"
            ),
        })
        codes = provenance_value["group_codes"]
        for prediction_position, (index, label, score) in enumerate(
            zip(indices, labels, scores, strict=True)
        ):
            tables["outer_fold_predictions"].append({
                **common, "method": method, "arm": arm, "outer_index": outer,
                "repeat": repeat, "fold": fold_index,
                "prediction_position": prediction_position, "test_index": index,
                "group_code": None if codes is None else codes[index],
                "y_true": label, "y_score": score, "auc": auc,
                "groups_available": provenance_value["groups_available"],
                "split_strategy": observed.get("split_strategy"),
            })
        validated.append({
            **dict(fold),
            "outer_index": outer,
            "repeat": repeat,
            "fold": fold_index,
            "test_indices": indices,
            "y_true": labels,
            "y_score": scores,
            "auc": auc,
            "selected_features": selected,
            "eligible_features": None if eligible is None else list(eligible),
        })
    if eligible_by_fold is not None and seen != set(eligible_by_fold):
        raise error(context, "folds do not cover every outer preprocessing record")
    return validated


def pooled_auc(labels: list[int], scores: list[float], context: str) -> float:
    """Compute binary AUC for a validated pooled OOF vector."""
    import numpy as np

    y = np.asarray(labels, dtype=int)
    prediction = np.asarray(scores, dtype=float)
    positives, negatives = int(np.sum(y == 1)), int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        raise error(context, "pooled OOF AUC requires both classes")
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


def pooled_oof_vectors(
    folds: list[dict[str, Any]],
    *,
    n_samples: int,
    outer_repeats: int,
    context: str,
) -> tuple[list[int], list[float]]:
    """Average OOF scores by raw index and validate exact repeat coverage."""
    import numpy as np

    expected_repeats = set(range(outer_repeats))
    by_index: dict[int, tuple[int, dict[int, float]]] = {}
    for position, fold in enumerate(folds):
        fold_context = f"{context}[{position}]"
        repeat = as_int(fold.get("repeat"), f"{fold_context}.repeat")
        if repeat not in expected_repeats:
            raise error(fold_context, "repeat is outside the observed outer design")
        indices = [
            as_int(value, f"{fold_context}.test_indices")
            for value in as_list(fold.get("test_indices"), f"{fold_context}.test_indices")
        ]
        labels = [
            as_int(value, f"{fold_context}.y_true")
            for value in as_list(fold.get("y_true"), f"{fold_context}.y_true")
        ]
        scores = [
            as_number(value, f"{fold_context}.y_score")
            for value in as_list(fold.get("y_score"), f"{fold_context}.y_score")
        ]
        if not (len(indices) == len(labels) == len(scores)):
            raise error(fold_context, "OOF index/label/score lengths differ")
        for index, label, score in zip(indices, labels, scores, strict=True):
            assert score is not None
            if index >= n_samples:
                raise error(fold_context, "raw OOF test index is outside n_samples")
            if label not in (0, 1):
                raise error(fold_context, "OOF labels must be binary")
            if index not in by_index:
                by_index[index] = label, {repeat: score}
                continue
            previous_label, repeat_scores = by_index[index]
            if previous_label != label:
                raise error(context, f"OOF label changed for raw test index {index}")
            if repeat in repeat_scores:
                raise error(
                    context,
                    f"raw test index {index} occurs more than once in repeat {repeat}",
                )
            repeat_scores[repeat] = score
    if set(by_index) != set(range(n_samples)):
        raise error(context, "OOF predictions do not cover every raw test index")
    for index, (_, scores) in by_index.items():
        if set(scores) != expected_repeats:
            raise error(
                context,
                f"raw test index {index} must occur exactly outer_repeats times, "
                "once per repeat",
            )
    return (
        [by_index[index][0] for index in range(n_samples)],
        [
            float(np.mean([
                by_index[index][1][repeat]
                for repeat in range(outer_repeats)
            ]))
            for index in range(n_samples)
        ],
    )


def paired_pooled_oof_metrics(
    pre_split_folds: list[dict[str, Any]],
    fold_contained_folds: list[dict[str, Any]],
    *,
    n_samples: int,
    outer_repeats: int,
    context: str,
) -> dict[str, float]:
    """Recompute the paired pooled OOF estimand from strict fold records."""
    pre_y, pre_scores = pooled_oof_vectors(
        pre_split_folds,
        n_samples=n_samples,
        outer_repeats=outer_repeats,
        context=f"{context}.pre_split",
    )
    contained_y, contained_scores = pooled_oof_vectors(
        fold_contained_folds,
        n_samples=n_samples,
        outer_repeats=outer_repeats,
        context=f"{context}.fold_contained",
    )
    if pre_y != contained_y:
        raise error(context, "paired OOF arms have different labels")
    pre_auc = pooled_auc(pre_y, pre_scores, f"{context}.pre_split")
    contained_auc = pooled_auc(
        contained_y, contained_scores, f"{context}.fold_contained"
    )
    return {
        "pre_split_oof_auc": pre_auc,
        "fold_contained_oof_auc": contained_auc,
        "gap_oof": pre_auc - contained_auc,
    }


def _finite_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _assert_stability_value(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, Mapping):
        value = as_mapping(actual, context)
        if set(value) != set(expected):
            raise error(context, "fields do not match schema v2")
        for key, expected_item in expected.items():
            _assert_stability_value(value[key], expected_item, f"{context}.{key}")
        return
    if expected is None:
        if actual is not None:
            raise error(context, "expected null for an undefined stability estimand")
        return
    if isinstance(expected, bool):
        if not isinstance(actual, bool) or actual != expected:
            raise error(context, f"expected {expected!r}")
        return
    if isinstance(expected, int):
        if as_int(actual, context) != expected:
            raise error(context, f"expected {expected}")
        return
    expected_number = float(expected)
    actual_number = as_number(actual, context)
    if actual_number is None or not math.isclose(
        actual_number, expected_number, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise error(context, "does not match selections/eligible feature sets")


def _feature_stability(
    checkpoint: Mapping[str, Any],
    method: str,
    raw_stability: Any,
    contained_folds: list[dict[str, Any]],
    eligible_by_fold: Mapping[
        tuple[int, int, int], tuple[int, tuple[int, ...]]
    ],
) -> dict[str, Any]:
    context = f"{checkpoint['path']}.{method}.feature_stability"
    stability = as_mapping(raw_stability, context)
    selected = [fold["selected_features"] for fold in contained_folds]
    eligible = [
        eligible_by_fold[(fold["outer_index"], fold["repeat"], fold["fold"])][1]
        for fold in contained_folds
    ]
    expected = _finite_json(summarize_feature_stability(
        selected,
        eligible_feature_sets=eligible,
        raw_p=checkpoint["data_fingerprint"]["x_shape"][1],
    ))
    _assert_stability_value(stability, expected, context)
    return dict(stability)


def aggregate_observed(
    checkpoint: Mapping[str, Any], tables: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, dict[str, float]], dict[str, Any] | None]:
    state = checkpoint["state"]
    if state.get("observed") is None:
        return {}, None
    observed = as_mapping(state["observed"], f"{checkpoint['path']}.observed")
    if as_text(
        observed.get("statistic_rule"),
        f"{checkpoint['path']}.observed.statistic_rule",
    ) != STATISTIC_RULE:
        raise error(str(checkpoint["path"]), "observed statistic_rule mismatch")
    provenance_value = provenance(checkpoint, observed)
    eligible_by_fold, design = _outer_preprocessing(checkpoint, observed)
    if not design["split_strategy"]:
        raise error(str(checkpoint["path"]), "observed split_strategy is empty")
    methods = as_mapping(observed.get("methods"), f"{checkpoint['path']}.methods")
    if not methods:
        raise error(str(checkpoint["path"]), "observed methods are empty")
    common, metadata = base_row(checkpoint), analysis_metadata(checkpoint)
    signatures: dict[
        tuple[int, int, int], tuple[tuple[int, ...], tuple[int, ...]]
    ] = {}
    observed_metrics: dict[str, dict[str, float]] = {}
    for method in sorted(methods):
        wrapper = as_mapping(methods[method], f"{checkpoint['path']}.{method}")
        result = as_mapping(wrapper.get("observed"), f"{checkpoint['path']}.{method}.observed")
        expected_fields = set(OBSERVED_METHOD_METRICS) | {
            "pre_split_folds", "fold_contained_folds", "feature_stability",
        }
        if set(result) != expected_fields:
            raise error(
                str(checkpoint["path"]),
                f"{method} observed estimand fields do not match statistic_rule",
            )
        metrics = {
            "pre_split_mean_fold_auc": as_auc(
                result.get("pre_split_mean_fold_auc"),
                f"{checkpoint['path']}.{method}.pre_mean_fold",
            ),
            "fold_contained_mean_fold_auc": as_auc(
                result.get("fold_contained_mean_fold_auc"),
                f"{checkpoint['path']}.{method}.contained_mean_fold",
            ),
            "gap_mean_fold": as_gap(
                result.get("gap_mean_fold"),
                f"{checkpoint['path']}.{method}.gap_mean_fold",
            ),
            "pre_split_oof_auc": as_auc(
                result.get("pre_split_oof_auc"),
                f"{checkpoint['path']}.{method}.pre_oof",
            ),
            "fold_contained_oof_auc": as_auc(
                result.get("fold_contained_oof_auc"),
                f"{checkpoint['path']}.{method}.contained_oof",
            ),
            "gap_oof": as_gap(
                result.get("gap_oof"), f"{checkpoint['path']}.{method}.gap_oof"
            ),
        }
        assert metrics["gap_mean_fold"] is not None and metrics["gap_oof"] is not None
        if not math.isclose(
            metrics["pre_split_mean_fold_auc"]
            - metrics["fold_contained_mean_fold_auc"],
            metrics["gap_mean_fold"], rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise error(
                str(checkpoint["path"]), f"{method} mean-fold AUC gap is inconsistent"
            )
        contained_folds: list[dict[str, Any]] = []
        pre_split_folds: list[dict[str, Any]] = []
        for arm, key, metric in (
            ("pre_split", "pre_split_folds", "pre_split_mean_fold_auc"),
            (
                "fold_contained",
                "fold_contained_folds",
                "fold_contained_mean_fold_auc",
            ),
        ):
            folds = _fold_rows(
                checkpoint, observed, provenance_value, method, arm,
                result.get(key), tables, signatures,
                eligible_by_fold if arm == "fold_contained" else None,
            )
            if not folds:
                raise error(str(checkpoint["path"]), f"{method}.{key} is empty")
            fold_keys = {
                (fold["outer_index"], fold["repeat"], fold["fold"])
                for fold in folds
            }
            if fold_keys != set(eligible_by_fold):
                raise error(
                    str(checkpoint["path"]),
                    f"{method}.{key} does not cover the shared outer folds",
                )
            fold_mean = sum(float(fold["auc"]) for fold in folds) / len(folds)
            if not math.isclose(fold_mean, metrics[metric], rel_tol=1e-12, abs_tol=1e-12):
                raise error(str(checkpoint["path"]), f"{method}.{metric} differs from folds")
            tables["dataset_method_summary"].append({
                **common, "method": method, "arm": arm, "auc": metrics[metric],
                "gap": metrics["gap_mean_fold"], "n_outer_folds": len(folds),
                "split_strategy": design["split_strategy"],
                "group_fallback_reason": design["group_fallback_reason"],
                "requested_outer_splits": design["requested_outer_splits"],
                "effective_outer_splits": design["effective_outer_splits"],
                "group_adjustment_reason": design["group_adjustment_reason"],
                "preprocessing_workspace_sha256": (
                    design["preprocessing_workspace_sha256"]
                ),
                "groups_available": provenance_value["groups_available"],
                "n_groups": provenance_value["n_groups"],
                "n_samples": metadata.get("n_samples"),
                "n_features_raw": metadata.get("n_features_raw"),
                "analysis_config_sha256": checkpoint["analysis_config_sha256"],
                "code_sha256": checkpoint["code_sha256"],
                "environment_sha256": checkpoint["environment_sha256"],
            })
            if arm == "fold_contained":
                contained_folds = folds
            else:
                pre_split_folds = folds
        recomputed_oof = paired_pooled_oof_metrics(
            pre_split_folds,
            contained_folds,
            n_samples=checkpoint["data_fingerprint"]["x_shape"][0],
            outer_repeats=design["outer_repeats"],
            context=f"{checkpoint['path']}.{method}.pooled_oof",
        )
        for metric, expected in recomputed_oof.items():
            if not math.isclose(
                metrics[metric], expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise error(
                    str(checkpoint["path"]),
                    f"{method}.{metric} differs from pooled OOF fold records",
                )
        if not math.isclose(
            metrics["pre_split_oof_auc"] - metrics["fold_contained_oof_auc"],
            metrics["gap_oof"], rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise error(str(checkpoint["path"]), f"{method} pooled OOF gap is inconsistent")
        observed_metrics[method] = metrics
        stability = _feature_stability(
            checkpoint,
            method,
            result.get("feature_stability"),
            contained_folds,
            eligible_by_fold,
        )
        selected_by_fold = [fold["selected_features"] for fold in contained_folds]
        feature_ids_by_fold = [
            [provenance_value["selected_map"][i] for i in selected]
            for selected in selected_by_fold
        ]
        tables["feature_stability"].append({
            **common,
            **{
                key: value for key, value in stability.items()
                if key != "raw_universe_sensitivity"
            },
            "method": method, "arm": "fold_contained",
            "selected_raw_indices_by_fold_json": json_text(selected_by_fold),
            "selected_feature_ids_by_fold_json": json_text(feature_ids_by_fold),
            "raw_universe_sensitivity_json": json_text(
                stability["raw_universe_sensitivity"]
            ),
            "groups_available": provenance_value["groups_available"],
        })
    baseline = as_mapping(observed.get("baseline_l1"), f"{checkpoint['path']}.baseline")
    baseline_auc = as_auc(baseline.get("auc"), f"{checkpoint['path']}.baseline.auc")
    baseline_folds = _fold_rows(
        checkpoint, observed, provenance_value, "baseline_l1", "fold_contained",
        baseline.get("folds"), tables, signatures, eligible_by_fold,
    )
    if not baseline_folds or not math.isclose(
        sum(float(fold["auc"]) for fold in baseline_folds) / len(baseline_folds),
        baseline_auc, rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise error(str(checkpoint["path"]), "baseline_l1 AUC/folds mismatch")
    pooled_oof_vectors(
        baseline_folds,
        n_samples=checkpoint["data_fingerprint"]["x_shape"][0],
        outer_repeats=design["outer_repeats"],
        context=f"{checkpoint['path']}.baseline_l1.pooled_oof",
    )
    tables["dataset_method_summary"].append({
        **common, "method": "baseline_l1", "arm": "fold_contained",
        "auc": baseline_auc, "gap": None, "n_outer_folds": len(baseline_folds),
        "split_strategy": design["split_strategy"],
        "group_fallback_reason": design["group_fallback_reason"],
        "requested_outer_splits": design["requested_outer_splits"],
        "effective_outer_splits": design["effective_outer_splits"],
        "group_adjustment_reason": design["group_adjustment_reason"],
        "preprocessing_workspace_sha256": design["preprocessing_workspace_sha256"],
        "groups_available": provenance_value["groups_available"],
        "n_groups": provenance_value["n_groups"],
        "n_samples": metadata.get("n_samples"),
        "n_features_raw": metadata.get("n_features_raw"),
        "analysis_config_sha256": checkpoint["analysis_config_sha256"],
        "code_sha256": checkpoint["code_sha256"],
        "environment_sha256": checkpoint["environment_sha256"],
    })
    if set(signatures) != set(eligible_by_fold):
        raise error(
            str(checkpoint["path"]),
            "outer folds and preprocessing records do not have identical coverage",
        )
    provenance_value = {**provenance_value, **design}
    return observed_metrics, provenance_value


__all__ = [
    "AggregationError", "OBSERVED_METHOD_METRICS", "SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS", "aggregate_observed",
    "analysis_metadata", "as_hash", "as_int", "as_list", "as_mapping",
    "as_number", "as_text", "base_row", "error", "load_checkpoints",
    "paired_pooled_oof_metrics", "pooled_auc", "pooled_oof_vectors",
]
