"""Validate QC provenance before a meta-analysis cohort is authorized."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from nm_v4.methods import METHODS
from nm_v4.qc_validation import FLOW_COLUMNS


QC_ELIGIBLE_CLASSIFICATIONS = frozenset({
    "eligible_primary_formal",
    "eligible_paired_sensitivity_nonprimary",
    "eligible_pilot_nonformal",
    "eligible_observed_only_nonformal",
})
QC_CLASSIFICATIONS = QC_ELIGIBLE_CLASSIFICATIONS | frozenset({
    "design_ineligible_group_cv",
    "design_ineligible_preprocessing",
    "null_ineligible_preprocessing",
    "blocking_failure",
})
_QC_BOOLEAN_COLUMNS = (
    "selected_whitelist",
    "checkpoint_present",
    "attempted",
    "observed_complete",
    "five_method_two_arm_complete",
    "shared_fold_repeat_complete",
    "oof_uncertainty_complete",
    "feature_stability_complete",
    "permutation_complete",
    "primary_null_eligible",
    "sensitivity_null_eligible",
    "formal_inference_eligible",
    "analysis_eligible",
    "blocking",
)
_QC_COMPLETE_COLUMNS = (
    "observed_complete",
    "five_method_two_arm_complete",
    "shared_fold_repeat_complete",
    "oof_uncertainty_complete",
    "feature_stability_complete",
    "permutation_complete",
)
_QC_COUNT_KEYS = (
    "eligible_primary_formal",
    "eligible_paired_sensitivity_nonprimary",
    "eligible_pilot_nonformal",
    "eligible_observed_only_nonformal",
    "design_ineligible_group_cv",
    "design_ineligible_preprocessing",
    "null_ineligible_preprocessing",
    "blocking_failure",
)


class QCProvenanceError(ValueError):
    """QC artifacts do not authorize the proposed meta-analysis cohort."""


def _fail(message: str) -> QCProvenanceError:
    return QCProvenanceError(message)


def _boolean(value: Any, context: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise _fail(f"{context} must be boolean")


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise _fail(f"{context} must be an integer >= {minimum}")
    try:
        number = int(str(value))
    except (TypeError, ValueError) as exc:
        raise _fail(f"{context} must be an integer >= {minimum}") from exc
    if str(value).strip() not in {str(number), f"{number}.0"} or number < minimum:
        raise _fail(f"{context} must be an integer >= {minimum}")
    return number


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    return str(value).strip()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{context} must be an object")
    return value


def _summary_counts(
    summary: Mapping[str, Any], actual: Mapping[str, int]
) -> None:
    recorded = _mapping(summary.get("counts"), "QC summary counts")
    if set(recorded) != set(_QC_COUNT_KEYS):
        raise _fail("QC summary classification count keys are incomplete")
    normalized = {
        key: _integer(recorded[key], f"QC summary counts.{key}")
        for key in _QC_COUNT_KEYS
    }
    if normalized != dict(actual):
        raise _fail("QC summary classification counts differ from the flow")


def _reason_counts(
    summary: Mapping[str, Any], flow: pd.DataFrame
) -> None:
    blocking = _mapping(
        summary.get("blocking_reason_counts"), "QC blocking reason counts"
    )
    if blocking:
        raise _fail("QC contains a blocking failure and cannot authorize analysis")
    expected = Counter(
        str(row.reason_code)
        for row in flow.itertuples(index=False)
        if row.classification
        in {"design_ineligible_group_cv", "design_ineligible_preprocessing"}
    )
    recorded = _mapping(
        summary.get("design_ineligible_reason_counts"),
        "QC design-ineligible reason counts",
    )
    normalized = {
        str(key): _integer(
            value, f"QC design-ineligible reason counts.{key}", minimum=1
        )
        for key, value in recorded.items()
    }
    if normalized != dict(sorted(expected.items())):
        raise _fail(
            "QC design-ineligible reason counts differ from the attempted flow"
        )
    expected_null = Counter(
        str(row.reason_code)
        for row in flow.itertuples(index=False)
        if row.classification == "null_ineligible_preprocessing"
    )
    recorded_null = _mapping(
        summary.get("null_ineligible_reason_counts"),
        "QC null-ineligible reason counts",
    )
    normalized_null = {
        str(key): _integer(
            value, f"QC null-ineligible reason counts.{key}", minimum=1
        )
        for key, value in recorded_null.items()
    }
    if normalized_null != dict(sorted(expected_null.items())):
        raise _fail(
            "QC null-ineligible reason counts differ from the attempted flow"
        )


def _validate_eligible_row(row: Any, mode: str, target: int) -> None:
    dataset_id = str(row.dataset_id)
    if row.runner_status != "ok":
        raise _fail(
            f"{dataset_id}: analysis-eligible checkpoint runner status is not ok"
        )
    if not all(bool(getattr(row, column)) for column in _QC_COMPLETE_COLUMNS):
        raise _fail(
            f"{dataset_id}: analysis-eligible QC row lacks complete evidence"
        )
    if row.blocking:
        raise _fail(f"{dataset_id}: analysis-eligible QC row is blocking")
    if row.completed_permutations != target:
        raise _fail(
            f"{dataset_id}: eligible checkpoint has incomplete permutations"
        )
    classification = str(row.classification)
    protocol = _optional_text(row.null_protocol)
    if classification == "eligible_primary_formal":
        valid = (
            mode == "formal"
            and row.primary_null_eligible
            and not row.sensitivity_null_eligible
            and row.formal_inference_eligible
            and protocol == "primary_exchangeable"
        )
    elif classification == "eligible_paired_sensitivity_nonprimary":
        valid = (
            mode != "observed-only"
            and not row.primary_null_eligible
            and row.sensitivity_null_eligible
            and not row.formal_inference_eligible
            and protocol == "paired_repeated_measures_sensitivity"
        )
    elif classification == "eligible_pilot_nonformal":
        valid = (
            mode == "pilot"
            and row.primary_null_eligible
            and not row.sensitivity_null_eligible
            and not row.formal_inference_eligible
            and protocol == "primary_exchangeable"
        )
    else:
        valid = (
            classification == "eligible_observed_only_nonformal"
            and mode == "observed-only"
            and not row.primary_null_eligible
            and not row.sensitivity_null_eligible
            and not row.formal_inference_eligible
            and protocol is None
            and target == 0
        )
    if not valid:
        raise _fail(
            f"{dataset_id}: QC eligibility classification/scope is inconsistent"
        )


def _validate_design_ineligible_row(row: Any) -> None:
    dataset_id = str(row.dataset_id)
    if row.runner_status != "evaluation_error":
        raise _fail(
            f"{dataset_id}: design-ineligible row has an invalid runner status"
        )
    classification = str(row.classification)
    if classification == "design_ineligible_group_cv":
        valid_reason = row.reason_code in {
            "insufficient_groups",
            "group_split_failed",
        }
        reason_error = "design-ineligible reason is not group-CV infeasibility"
    else:
        valid_reason = row.reason_code == "no_feature_meets_completeness"
        reason_error = (
            "design-ineligible preprocessing reason is not exact completeness "
            "infeasibility"
        )
    if not valid_reason:
        raise _fail(
            f"{dataset_id}: {reason_error}"
        )
    if row.blocking or any(
        bool(getattr(row, column)) for column in _QC_COMPLETE_COLUMNS
    ):
        raise _fail(
            f"{dataset_id}: design-ineligible row cannot contain success evidence"
        )
    if any((
        row.primary_null_eligible,
        row.sensitivity_null_eligible,
        row.formal_inference_eligible,
    )):
        raise _fail(
            f"{dataset_id}: design-ineligible row cannot be inference eligible"
        )
    if row.completed_permutations != 0:
        raise _fail(
            f"{dataset_id}: design-ineligible pre-observed failure has permutations"
        )


def _validate_null_ineligible_row(row: Any) -> None:
    dataset_id = str(row.dataset_id)
    if row.runner_status != "evaluation_error":
        raise _fail(
            f"{dataset_id}: null-ineligible row has an invalid runner status"
        )
    if row.reason_code != "permutation_no_feature_meets_completeness":
        raise _fail(
            f"{dataset_id}: null-ineligible preprocessing reason is not exact"
        )
    observed_columns = tuple(
        column for column in _QC_COMPLETE_COLUMNS if column != "permutation_complete"
    )
    if not all(bool(getattr(row, column)) for column in observed_columns):
        raise _fail(
            f"{dataset_id}: null-ineligible row lacks complete observed evidence"
        )
    if row.permutation_complete:
        raise _fail(
            f"{dataset_id}: null-ineligible row cannot have complete permutations"
        )
    if row.p_value_status != "incomplete_no_p":
        raise _fail(
            f"{dataset_id}: null-ineligible row must have incomplete-no-p p-value status"
        )
    if row.target_permutations < 1 or not (
        0 <= row.completed_permutations < row.target_permutations
    ):
        raise _fail(
            f"{dataset_id}: null-ineligible permutation counts are inconsistent"
        )
    if _optional_text(row.null_protocol) not in {
        None,
        "primary_exchangeable",
        "paired_repeated_measures_sensitivity",
    }:
        raise _fail(f"{dataset_id}: null-ineligible null protocol is unsupported")
    if row.blocking:
        raise _fail(f"{dataset_id}: null-ineligible row cannot be blocking")
    if any((
        row.primary_null_eligible,
        row.sensitivity_null_eligible,
        row.formal_inference_eligible,
    )):
        raise _fail(
            f"{dataset_id}: null-ineligible row cannot be inference eligible"
        )


def validate_qc_artifacts(
    flow: pd.DataFrame,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    selected_ids: Sequence[str],
    checkpoint_statuses: Mapping[str, str],
    target_permutations: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate the complete attempted cohort and return eligibility counts."""
    if not isinstance(flow, pd.DataFrame) or flow.empty:
        raise _fail("QC eligibility flow must be non-empty")
    if tuple(flow.columns) != tuple(FLOW_COLUMNS):
        raise _fail("QC eligibility flow columns/order do not match the schema")
    data = flow.copy(deep=True)
    for column in ("dataset", "dataset_id", "source", "runner_status"):
        if data[column].isna().any():
            raise _fail(f"QC flow {column} must be non-missing")
        data[column] = data[column].astype(str).str.strip()
        if data[column].eq("").any():
            raise _fail(f"QC flow {column} must be non-empty")
    for column in ("classification", "reason_code"):
        if data[column].isna().any():
            raise _fail(f"QC flow {column} must be non-missing")
        data[column] = data[column].astype(str).str.strip()
    if not data["dataset"].equals(data["dataset_id"]):
        raise _fail("QC flow dataset and dataset_id disagree")
    if data["dataset_id"].duplicated().any():
        raise _fail("QC flow contains duplicate dataset IDs")
    selected = list(selected_ids)
    if len(selected) != len(set(selected)) or set(data.dataset_id) != set(selected):
        raise _fail("runner selected IDs and full attempted QC flow differ")
    if set(checkpoint_statuses) != set(selected):
        raise _fail("checkpoint statuses do not cover every attempted dataset")
    for column in _QC_BOOLEAN_COLUMNS:
        data[column] = [
            _boolean(value, f"QC flow {column}") for value in data[column]
        ]
    for column, minimum in (
        ("oof_uncertainty_estimable_methods", 0),
        ("completed_permutations", 0),
        ("target_permutations", 0),
    ):
        data[column] = [
            _integer(value, f"QC flow {column}", minimum=minimum)
            for value in data[column]
        ]
    if not (
        data["selected_whitelist"]
        & data["checkpoint_present"]
        & data["attempted"]
    ).all():
        raise _fail("QC flow does not mark every selected dataset as attempted")
    if not data["target_permutations"].eq(target_permutations).all():
        raise _fail("QC flow target differs from the runner protocol")
    if not data["oof_uncertainty_estimable_methods"].between(0, len(METHODS)).all():
        raise _fail("QC flow estimable-method count is outside [0, 5]")

    mode = str(summary.get("mode", ""))
    if mode not in {"formal", "pilot", "observed-only"}:
        raise _fail("QC summary mode is invalid")
    for row in data.itertuples(index=False):
        if checkpoint_statuses[row.dataset_id] != row.runner_status:
            raise _fail(
                f"{row.dataset_id}: QC runner status differs from checkpoint"
            )
        if row.classification not in QC_CLASSIFICATIONS:
            raise _fail(
                f"{row.dataset_id}: QC classification is not recognized"
            )
        if row.analysis_eligible:
            if row.classification not in QC_ELIGIBLE_CLASSIFICATIONS:
                raise _fail(
                    f"{row.dataset_id}: excluded QC class was marked eligible"
                )
            _validate_eligible_row(row, mode, target_permutations)
        elif row.classification in {
            "design_ineligible_group_cv",
            "design_ineligible_preprocessing",
        }:
            _validate_design_ineligible_row(row)
        elif row.classification == "null_ineligible_preprocessing":
            _validate_null_ineligible_row(row)
        else:
            raise _fail(
                f"{row.dataset_id}: blocking or non-design failure cannot be excluded"
            )

    actual_counts = Counter(data["classification"])
    counts = {key: int(actual_counts[key]) for key in _QC_COUNT_KEYS}
    _summary_counts(summary, counts)
    _reason_counts(summary, data)
    if _integer(summary.get("schema_version"), "QC summary schema") != 1:
        raise _fail("QC summary schema version is unsupported")
    if _integer(
        summary.get("selected_whitelist_count"), "QC selected count", minimum=1
    ) != len(selected):
        raise _fail("QC summary selected count differs from attempted flow")
    if _integer(
        summary.get("target_permutations"), "QC target permutations"
    ) != target_permutations:
        raise _fail("QC summary target differs from runner protocol")
    if summary.get("required_methods") != list(METHODS):
        raise _fail("QC summary does not require the exact five methods")
    if summary.get("required_arms") != ["pre_split", "fold_contained"]:
        raise _fail("QC summary does not require both analysis arms")
    if summary.get("primary_and_sensitivity_nulls_separated") is not True:
        raise _fail("QC summary does not separate primary and sensitivity nulls")
    if counts["null_ineligible_preprocessing"]:
        expected_status = "pass_with_ineligible"
    elif (
        counts["design_ineligible_group_cv"]
        or counts["design_ineligible_preprocessing"]
    ):
        expected_status = "pass_with_design_ineligible"
    else:
        expected_status = "pass"
    if summary.get("qc_status") != expected_status:
        raise _fail("QC status is not compatible with the attempted flow")
    formal_allowed = (
        mode == "formal" and counts["eligible_primary_formal"] > 0
    )
    if summary.get("formal_claims_allowed") is not formal_allowed:
        raise _fail("QC formal-claims flag is inconsistent")
    if manifest.get("mode") != mode:
        raise _fail("QC manifest mode differs from QC summary")
    if manifest.get("qc_status") != expected_status:
        raise _fail("QC manifest status differs from QC summary")
    if manifest.get("formal_claims_allowed") is not formal_allowed:
        raise _fail("QC manifest formal-claims flag is inconsistent")
    provenance_counts = {
        "attempted": len(selected),
        "analysis_eligible": int(data["analysis_eligible"].sum()),
        "design_ineligible_group_cv": counts["design_ineligible_group_cv"],
        "design_ineligible_preprocessing": counts[
            "design_ineligible_preprocessing"
        ],
        "null_ineligible_preprocessing": counts[
            "null_ineligible_preprocessing"
        ],
        "eligible_observed_only_nonformal": counts[
            "eligible_observed_only_nonformal"
        ],
        "eligible_paired_sensitivity_nonprimary": counts[
            "eligible_paired_sensitivity_nonprimary"
        ],
        "eligible_pilot_nonformal": counts["eligible_pilot_nonformal"],
        "eligible_primary_formal": counts["eligible_primary_formal"],
        "sensitivity_only": counts["eligible_paired_sensitivity_nonprimary"],
    }
    if provenance_counts["analysis_eligible"] != sum(
        counts[key] for key in QC_ELIGIBLE_CLASSIFICATIONS
    ):
        raise _fail("QC analysis-eligible count is internally inconsistent")
    return data, provenance_counts


__all__ = [
    "QCProvenanceError",
    "QC_CLASSIFICATIONS",
    "QC_ELIGIBLE_CLASSIFICATIONS",
    "validate_qc_artifacts",
]
