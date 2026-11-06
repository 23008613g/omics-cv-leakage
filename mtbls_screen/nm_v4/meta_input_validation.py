"""Recompute OOF uncertainty to bind aggregate effects to runner checkpoints."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
import pandas as pd

from nm_v4.aggregation_schema import as_list, as_mapping
from nm_v4.aggregation_uncertainty import _auc, _bootstrap_gap, _oof_vectors
from nm_v4.checkpoints import sha256_json
from nm_v4.meta_qc_provenance import (
    QCProvenanceError,
    QC_CLASSIFICATIONS,
    QC_ELIGIBLE_CLASSIFICATIONS,
    validate_qc_artifacts,
)
from nm_v4.methods import METHODS


_FLOAT_COLUMNS = (
    "pre_split_oof_auc", "fold_contained_oof_auc", "gap_oof",
    "standard_error", "ci_low", "ci_high", "variance",
)
_EXACT_COLUMNS = (
    "dataset", "dataset_id", "source", "method", "comparison",
    "uncertainty_unit", "n_units", "n_samples", "n_repeats",
    "requested_bootstrap_replicates", "valid_bootstrap_replicates",
    "bootstrap_seed", "uncertainty_reason", "groups_available",
)
_TOLERANCE = 1e-10
UNCERTAINTY_REQUIRED_COLUMNS = frozenset(_FLOAT_COLUMNS + _EXACT_COLUMNS)


class UncertaintyProvenanceError(ValueError):
    """An uncertainty row differs from the checkpoint-derived value."""


def _text(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise UncertaintyProvenanceError(f"{column} must be non-missing")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise UncertaintyProvenanceError(f"{column} must be non-empty")


def _numbers(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    nullable: frozenset[str] = frozenset(),
) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise UncertaintyProvenanceError(f"{column} must be numeric") from exc
        if column not in nullable and values.isna().any():
            raise UncertaintyProvenanceError(f"{column} must not be missing")
        if not values.loc[values.notna()].map(math.isfinite).all():
            raise UncertaintyProvenanceError(f"{column} must be finite")
        frame[column] = values


def _integers(frame: pd.DataFrame, column: str, minimum: int) -> None:
    if not frame[column].map(lambda value: float(value).is_integer()).all():
        raise UncertaintyProvenanceError(f"{column} must contain integers")
    frame[column] = frame[column].astype(int)
    if (frame[column] < minimum).any():
        raise UncertaintyProvenanceError(f"{column} must be >= {minimum}")


def _boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise UncertaintyProvenanceError("groups_available must be boolean")


def validate_uncertainty_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the complete canonical uncertainty-table contract."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise UncertaintyProvenanceError("uncertainty table must be non-empty")
    data = frame.copy(deep=True)
    missing = sorted(UNCERTAINTY_REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        raise UncertaintyProvenanceError(
            f"uncertainty table is missing required columns: {missing}"
        )
    _text(data, (
        "dataset", "dataset_id", "source", "method", "comparison",
        "uncertainty_unit", "uncertainty_reason",
    ))
    if not data["dataset"].equals(data["dataset_id"]):
        raise UncertaintyProvenanceError("dataset and dataset_id disagree")
    if data.duplicated(["dataset_id", "method"]).any():
        raise UncertaintyProvenanceError("duplicate (dataset_id, method) rows")
    _numbers(
        data,
        _FLOAT_COLUMNS + (
            "n_units", "n_samples", "n_repeats",
            "requested_bootstrap_replicates", "valid_bootstrap_replicates",
            "bootstrap_seed",
        ),
        nullable=frozenset({
            "standard_error", "ci_low", "ci_high", "variance",
        }),
    )
    for column, minimum in (
        ("n_units", 1), ("n_samples", 1), ("n_repeats", 1),
        ("requested_bootstrap_replicates", 2),
        ("valid_bootstrap_replicates", 0), ("bootstrap_seed", 0),
    ):
        _integers(data, column, minimum)
    data["groups_available"] = data["groups_available"].map(_boolean)
    if (
        data["valid_bootstrap_replicates"]
        > data["requested_bootstrap_replicates"]
    ).any():
        raise UncertaintyProvenanceError("valid bootstrap count exceeds requested")
    for column in ("pre_split_oof_auc", "fold_contained_oof_auc"):
        if not data[column].between(0, 1).all():
            raise UncertaintyProvenanceError(f"{column} must lie in [0, 1]")
    paired = data["pre_split_oof_auc"] - data["fold_contained_oof_auc"]
    if not all(
        math.isclose(float(left), float(right), rel_tol=_TOLERANCE,
                     abs_tol=_TOLERANCE)
        for left, right in zip(paired, data["gap_oof"], strict=True)
    ):
        raise UncertaintyProvenanceError("gap_oof is inconsistent")
    if not data["comparison"].eq("pre_split_minus_fold_contained").all():
        raise UncertaintyProvenanceError("comparison is not the paired contrast")
    if (data[["standard_error", "variance"]].dropna() < 0).any().any():
        raise UncertaintyProvenanceError("uncertainty spread cannot be negative")
    expected_methods = set(METHODS)
    for dataset_id, rows in data.groupby("dataset_id", sort=True):
        if set(rows.method) != expected_methods or len(rows) != len(METHODS):
            raise UncertaintyProvenanceError(
                f"{dataset_id}: uncertainty requires exactly five supervised methods"
            )
        for column in ("dataset", "source", "n_samples"):
            if rows[column].nunique(dropna=False) != 1:
                raise UncertaintyProvenanceError(
                    f"{dataset_id}: {column} varies across uncertainty rows"
                )
    inferential = ("standard_error", "ci_low", "ci_high", "variance")
    valid_rows = data["uncertainty_reason"].eq("ok")
    if data.loc[valid_rows, inferential].isna().any().any():
        raise UncertaintyProvenanceError(
            "ok uncertainty rows require all inferential statistics"
        )
    if data.loc[~valid_rows, inferential].notna().any().any():
        raise UncertaintyProvenanceError(
            "failed uncertainty rows cannot carry inferential statistics"
        )
    ok = data.loc[valid_rows]
    if not all(
        math.isclose(float(variance), float(se) ** 2,
                     rel_tol=_TOLERANCE, abs_tol=_TOLERANCE)
        for variance, se in zip(ok["variance"], ok["standard_error"], strict=True)
    ):
        raise UncertaintyProvenanceError("variance must equal standard_error squared")
    if (ok["ci_low"] > ok["ci_high"]).any():
        raise UncertaintyProvenanceError("uncertainty CI bounds are reversed")
    return data


def _fold_prediction_rows(
    observed: Mapping[str, Any],
    *,
    dataset_id: str,
    method: str,
    arm: str,
    n_samples: int,
) -> list[dict[str, Any]]:
    wrapper = as_mapping(observed["methods"][method], f"{dataset_id}.{method}")
    result = as_mapping(wrapper.get("observed"), f"{dataset_id}.{method}.observed")
    key = "pre_split_folds" if arm == "pre_split" else "fold_contained_folds"
    folds = as_list(result.get(key), f"{dataset_id}.{method}.{key}")
    if not folds:
        raise UncertaintyProvenanceError(f"{dataset_id}.{method}.{key} is empty")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    coverage: dict[int, set[int]] = {}
    for position, raw_fold in enumerate(folds):
        context = f"{dataset_id}.{method}.{key}[{position}]"
        fold = as_mapping(raw_fold, context)
        repeat = fold.get("repeat")
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
            raise UncertaintyProvenanceError(f"{context}.repeat is invalid")
        indices = as_list(fold.get("test_indices"), f"{context}.test_indices")
        labels = as_list(fold.get("y_true"), f"{context}.y_true")
        scores = as_list(fold.get("y_score"), f"{context}.y_score")
        if not indices or not (len(indices) == len(labels) == len(scores)):
            raise UncertaintyProvenanceError(f"{context} prediction lengths disagree")
        for raw_index, raw_label, raw_score in zip(
            indices, labels, scores, strict=True
        ):
            if (
                isinstance(raw_index, bool) or not isinstance(raw_index, int)
                or not 0 <= raw_index < n_samples
            ):
                raise UncertaintyProvenanceError(f"{context} test index is invalid")
            if raw_label not in (0, 1) or isinstance(raw_label, bool):
                raise UncertaintyProvenanceError(f"{context} label is not binary")
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(float(raw_score))
            ):
                raise UncertaintyProvenanceError(f"{context} score is invalid")
            signature = repeat, raw_index
            if signature in seen:
                raise UncertaintyProvenanceError(
                    f"{context} repeats raw test index {raw_index}"
                )
            seen.add(signature)
            coverage.setdefault(repeat, set()).add(raw_index)
            rows.append({
                "repeat": repeat,
                "test_index": raw_index,
                "y_true": raw_label,
                "y_score": float(raw_score),
            })
    repeats = sorted(coverage)
    if repeats != list(range(len(repeats))) or any(
        indices != set(range(n_samples)) for indices in coverage.values()
    ):
        raise UncertaintyProvenanceError(
            f"{dataset_id}.{method}.{arm} does not cover every sample per repeat"
        )
    return rows


def independent_unit_summary(
    labels: Sequence[int], group_codes: Sequence[int] | None
) -> dict[str, Any]:
    """Derive observation and independent-unit counts without pseudo-counting."""
    normalized_labels = list(labels)
    if not normalized_labels or any(
        isinstance(value, bool) or value not in (0, 1)
        for value in normalized_labels
    ):
        raise UncertaintyProvenanceError(
            "observed labels must be a non-empty binary sequence"
        )
    class_counts = Counter(normalized_labels)
    if set(class_counts) != {0, 1}:
        raise UncertaintyProvenanceError(
            "observed labels must contain both outcome classes"
        )
    base = {
        "n_observations": len(normalized_labels),
        "minority_observations": min(class_counts.values()),
    }
    if group_codes is None:
        return {
            **base,
            "n_independent_units": len(normalized_labels),
            "minority_independent_units": min(class_counts.values()),
            "independent_unit_type": "observation",
            "minority_independent_units_reason": "row_level_no_groups",
        }
    codes = list(group_codes)
    if len(codes) != len(normalized_labels):
        raise UncertaintyProvenanceError(
            "group-code length differs from observed labels"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in codes
    ):
        raise UncertaintyProvenanceError(
            "group codes must be non-negative integers"
        )
    unique_groups = list(dict.fromkeys(codes))
    if unique_groups != list(range(len(unique_groups))):
        raise UncertaintyProvenanceError(
            "group codes must use canonical first-seen integer labels"
        )
    labels_by_group: dict[int, set[int]] = {
        group: set() for group in unique_groups
    }
    for group, label in zip(codes, normalized_labels, strict=True):
        labels_by_group[group].add(label)
    if all(len(values) == 1 for values in labels_by_group.values()):
        unit_class_counts = Counter(
            next(iter(values)) for values in labels_by_group.values()
        )
        if set(unit_class_counts) != {0, 1}:
            raise UncertaintyProvenanceError(
                "constant-label groups must represent both outcome classes"
            )
        minority_units: int | None = min(unit_class_counts.values())
        reason = "constant_label_groups"
    else:
        minority_units = None
        reason = "mixed_label_groups_no_unique_minority_unit_definition"
    return {
        **base,
        "n_independent_units": len(unique_groups),
        "minority_independent_units": minority_units,
        "independent_unit_type": "group",
        "minority_independent_units_reason": reason,
    }


def derive_independent_unit_summary(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive counts from checkpoint OOF labels and audited group codes."""
    checkpoint = record["checkpoint"]
    dataset_id = str(checkpoint["dataset"])
    rows = _fold_prediction_rows(
        record["observed"],
        dataset_id=dataset_id,
        method=METHODS[0],
        arm="pre_split",
        n_samples=int(record["n_samples"]),
    )
    labels, _, _ = _oof_vectors(
        rows,
        n_samples=int(record["n_samples"]),
        context=f"{dataset_id}.independent_unit_labels",
    )
    summary = independent_unit_summary(
        labels, record["provenance"]["group_codes"]
    )
    expected_counts = {
        0: int(record["n_neg"]),
        1: int(record["n_pos"]),
    }
    if Counter(labels) != expected_counts:
        raise UncertaintyProvenanceError(
            f"{dataset_id}: observed labels differ from metadata class_counts"
        )
    return summary


def _expected_row(
    record: Mapping[str, Any],
    method: str,
    *,
    requested: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    checkpoint = record["checkpoint"]
    dataset_id = str(checkpoint["dataset"])
    observed = record["observed"]
    n_samples = int(record["n_samples"])
    arms = {
        arm: _fold_prediction_rows(
            observed,
            dataset_id=dataset_id,
            method=method,
            arm=arm,
            n_samples=n_samples,
        )
        for arm in ("pre_split", "fold_contained")
    }
    pre_y, pre_scores, pre_repeats = _oof_vectors(
        arms["pre_split"],
        n_samples=n_samples,
        context=f"{dataset_id}.{method}.pre_split",
    )
    contained_y, contained_scores, contained_repeats = _oof_vectors(
        arms["fold_contained"],
        n_samples=n_samples,
        context=f"{dataset_id}.{method}.fold_contained",
    )
    if pre_y != contained_y or pre_repeats != contained_repeats:
        raise UncertaintyProvenanceError(
            f"{dataset_id}.{method} checkpoint OOF arms are not paired"
        )
    derived_seed = int(sha256_json({
        "seed": bootstrap_seed, "dataset": dataset_id, "method": method,
    })[:16], 16)
    bootstraps, unit, n_units, reason = _bootstrap_gap(
        pre_y,
        pre_scores,
        contained_scores,
        record["provenance"]["group_codes"],
        requested=requested,
        seed=derived_seed,
    )
    valid = len(bootstraps)
    if reason is None and valid < math.ceil(0.9 * requested):
        reason = "valid_bootstrap_fraction_below_0.90"
    if reason is None:
        standard_error = float(np.std(bootstraps, ddof=1))
        ci_low, ci_high = map(float, np.quantile(bootstraps, [0.025, 0.975]))
        variance = standard_error**2
    else:
        standard_error = ci_low = ci_high = variance = None
    pre_auc = _auc(pre_y, pre_scores)
    contained_auc = _auc(contained_y, contained_scores)
    return {
        "dataset": dataset_id,
        "dataset_id": dataset_id,
        "source": record["source_key"],
        "method": method,
        "comparison": "pre_split_minus_fold_contained",
        "pre_split_oof_auc": pre_auc,
        "fold_contained_oof_auc": contained_auc,
        "gap_oof": pre_auc - contained_auc,
        "standard_error": standard_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "variance": variance,
        "uncertainty_unit": unit,
        "n_units": n_units,
        "n_samples": n_samples,
        "n_repeats": pre_repeats,
        "requested_bootstrap_replicates": requested,
        "valid_bootstrap_replicates": valid,
        "bootstrap_seed": bootstrap_seed,
        "uncertainty_reason": reason or "ok",
        "groups_available": record["provenance"]["groups_available"],
    }


def _same_float(actual: Any, expected: Any) -> bool:
    if pd.isna(actual) or expected is None:
        return pd.isna(actual) and expected is None
    return math.isclose(
        float(actual), float(expected), rel_tol=_TOLERANCE, abs_tol=_TOLERANCE
    )


def verify_uncertainty_against_checkpoints(
    uncertainty: pd.DataFrame,
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    """Fail unless every uncertainty field matches deterministic recomputation."""
    requested_values = uncertainty["requested_bootstrap_replicates"].unique()
    seed_values = uncertainty["bootstrap_seed"].unique()
    if len(requested_values) != 1 or len(seed_values) != 1:
        raise UncertaintyProvenanceError(
            "uncertainty bootstrap settings must be identical across all rows"
        )
    requested, bootstrap_seed = int(requested_values[0]), int(seed_values[0])
    indexed = uncertainty.set_index(["dataset_id", "method"], verify_integrity=True)
    for dataset_id in sorted(records):
        for method in METHODS:
            actual = indexed.loc[(dataset_id, method)]
            expected = _expected_row(
                records[dataset_id], method,
                requested=requested, bootstrap_seed=bootstrap_seed,
            )
            for column in _EXACT_COLUMNS:
                actual_value = (
                    dataset_id if column == "dataset_id"
                    else (method if column == "method" else actual[column])
                )
                if actual_value != expected[column]:
                    raise UncertaintyProvenanceError(
                        f"{dataset_id}.{method}.{column} differs from checkpoint"
                    )
            for column in _FLOAT_COLUMNS:
                if not _same_float(actual[column], expected[column]):
                    category = (
                        "checkpoint OOF" if column in _FLOAT_COLUMNS[:3]
                        else "checkpoint bootstrap"
                    )
                    raise UncertaintyProvenanceError(
                        f"{dataset_id}.{method}.{column} differs from {category}"
                    )


__all__ = [
    "QCProvenanceError", "QC_CLASSIFICATIONS", "QC_ELIGIBLE_CLASSIFICATIONS",
    "UNCERTAINTY_REQUIRED_COLUMNS", "UncertaintyProvenanceError",
    "derive_independent_unit_summary", "independent_unit_summary",
    "validate_qc_artifacts", "validate_uncertainty_frame",
    "verify_uncertainty_against_checkpoints",
]
