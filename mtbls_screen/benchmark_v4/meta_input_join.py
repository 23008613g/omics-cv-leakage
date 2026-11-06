"""Strict aggregate-to-checkpoint joins for meta-analysis inputs."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import pandas as pd

from benchmark_v4.checkpoints import sha256_json
from benchmark_v4.aggregation_schema import OBSERVED_METHOD_METRICS, as_mapping
from benchmark_v4.configuration import STATISTIC_RULE
from benchmark_v4.meta_input_contracts import (
    BASELINE_SOURCE,
    EFFECT_SOURCE,
    SUMMARY_FILENAME,
    UNCERTAINTY_FILENAME,
    VARIANCE_SOURCE,
    _AUC_TOLERANCE,
    _fail,
)
from benchmark_v4.methods import METHODS


_ALLOWED_NON_ANALYSIS_CLASSIFICATIONS = frozenset({
    "design_ineligible_group_cv",
    "design_ineligible_preprocessing",
    "null_ineligible_preprocessing",
})


def _filter_analysis_eligible_frames(
    summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    qc_flow: pd.DataFrame,
    records: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | str]]:
    """Validate the observed/QC ID partition, then return analysis-only frames."""
    summary_ids = set(summary["dataset_id"])
    uncertainty_ids = set(uncertainty["dataset_id"])
    if summary_ids != uncertainty_ids:
        raise _fail(
            "summary and uncertainty dataset IDs differ before eligibility filtering"
        )

    flow_by_id = {
        str(row.dataset_id): row for row in qc_flow.itertuples(index=False)
    }
    flow_ids = set(flow_by_id)
    eligible_ids = {
        dataset_id
        for dataset_id, row in flow_by_id.items()
        if bool(row.analysis_eligible)
    }
    observed_ids = {
        dataset_id
        for dataset_id, row in flow_by_id.items()
        if bool(row.observed_complete)
    }
    record_ids = set(records)
    if record_ids != eligible_ids:
        raise _fail(
            "analysis-eligible QC dataset IDs differ from runner checkpoint records"
        )
    if not eligible_ids:
        raise _fail("validated QC flow contains no analysis-eligible datasets")

    aggregate_ids = summary_ids
    if aggregate_ids - flow_ids:
        raise _fail(
            "aggregation contains unknown dataset IDs absent from validated QC flow"
        )
    stale_ids = aggregate_ids.intersection(flow_ids - observed_ids)
    if stale_ids:
        raise _fail(
            "aggregation contains stale rows for datasets without observed QC evidence"
        )
    if observed_ids - aggregate_ids:
        raise _fail("aggregation is missing QC-observed dataset IDs")

    excluded_ids = flow_ids - eligible_ids
    if any(
        flow_by_id[dataset_id].classification
        not in _ALLOWED_NON_ANALYSIS_CLASSIFICATIONS
        for dataset_id in excluded_ids
    ):
        raise _fail("QC flow contains an unsupported non-analysis classification")
    excluded_observed_ids = aggregate_ids - eligible_ids
    if any(
        flow_by_id[dataset_id].classification
        not in _ALLOWED_NON_ANALYSIS_CLASSIFICATIONS
        or bool(flow_by_id[dataset_id].analysis_eligible)
        for dataset_id in excluded_observed_ids
    ):
        raise _fail(
            "excluded observed aggregation IDs lack permitted QC provenance"
        )

    excluded_preobserved_ids = excluded_ids - excluded_observed_ids
    audit: dict[str, int | str] = {
        "dataset_id_hash_order": "lexicographically_sorted",
        "attempted_dataset_count": len(flow_ids),
        "observed_aggregation_dataset_count": len(aggregate_ids),
        "analysis_eligible_dataset_count": len(eligible_ids),
        "analysis_eligible_ids_sha256": sha256_json(sorted(eligible_ids)),
        "excluded_from_meta_analysis_count": len(excluded_ids),
        "excluded_from_meta_analysis_ids_sha256": sha256_json(
            sorted(excluded_ids)
        ),
        "excluded_observed_audit_count": len(excluded_observed_ids),
        "excluded_observed_audit_ids_sha256": sha256_json(
            sorted(excluded_observed_ids)
        ),
        "excluded_preobserved_count": len(excluded_preobserved_ids),
        "excluded_preobserved_ids_sha256": sha256_json(
            sorted(excluded_preobserved_ids)
        ),
    }
    filtered_summary = summary.loc[
        summary["dataset_id"].isin(eligible_ids)
    ].copy(deep=True)
    filtered_uncertainty = uncertainty.loc[
        uncertainty["dataset_id"].isin(eligible_ids)
    ].copy(deep=True)
    return filtered_summary, filtered_uncertainty, audit


def _single_value(rows: pd.DataFrame, column: str, context: str) -> Any:
    if len(rows) != 1:
        raise _fail(f"{context}: expected exactly one row")
    return rows.iloc[0][column]


def _validate_checkpoint_metric(
    checkpoint_value: float,
    aggregate_value: float,
    *,
    dataset_id: str,
    method: str,
    key: str,
    aggregate_source: str,
) -> None:
    if not math.isclose(
        checkpoint_value,
        aggregate_value,
        rel_tol=_AUC_TOLERANCE,
        abs_tol=_AUC_TOLERANCE,
    ):
        raise _fail(
            f"{dataset_id}.{method}.{key} differs between checkpoint and "
            f"{aggregate_source}"
        )


def _joined_rows(
    summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summary_ids = set(summary["dataset_id"])
    uncertainty_ids = set(uncertainty["dataset_id"])
    checkpoint_ids = set(records)
    if summary_ids != uncertainty_ids or summary_ids != checkpoint_ids:
        raise _fail(
            "dataset IDs differ across summary, uncertainty, and runner checkpoints"
        )
    output: list[dict[str, Any]] = []
    for dataset_id in sorted(checkpoint_ids):
        record = records[dataset_id]
        summary_rows = summary.loc[summary.dataset_id == dataset_id]
        uncertainty_rows = uncertainty.loc[uncertainty.dataset_id == dataset_id]
        if set(summary_rows["source"]) != {record["source_key"]}:
            raise _fail(f"{dataset_id}: summary source does not match checkpoint")
        if set(uncertainty_rows["source"]) != {record["source_key"]}:
            raise _fail(f"{dataset_id}: uncertainty source does not match checkpoint")
        if set(summary_rows["n_samples"]) != {record["n_samples"]}:
            raise _fail(f"{dataset_id}: summary n_samples does not match checkpoint")
        if set(summary_rows["n_features_raw"]) != {record["n_features"]}:
            raise _fail(f"{dataset_id}: summary n_features_raw does not match checkpoint")
        if set(uncertainty_rows["n_samples"]) != {record["n_samples"]}:
            raise _fail(f"{dataset_id}: uncertainty n_samples does not match checkpoint")
        checkpoint = record["checkpoint"]
        expected_hashes = {
            "analysis_config_sha256": checkpoint["analysis_config_sha256"],
            "code_sha256": checkpoint["code_sha256"],
            "environment_sha256": checkpoint["environment_sha256"],
        }
        for column, expected in expected_hashes.items():
            if set(summary_rows[column]) != {expected}:
                raise _fail(f"{dataset_id}: {column} does not match checkpoint")
        baseline_rows = summary_rows.loc[summary_rows.method == "baseline_l1"]
        baseline_auc = float(_single_value(
            baseline_rows, "auc", f"{dataset_id}.baseline_l1"
        ))
        baseline_checkpoint = as_mapping(
            record["observed"].get("baseline_l1"),
            f"{dataset_id}.baseline_l1 checkpoint",
        )
        if not math.isclose(
            float(baseline_checkpoint.get("auc")), baseline_auc,
            rel_tol=_AUC_TOLERANCE, abs_tol=_AUC_TOLERANCE,
        ):
            raise _fail(f"{dataset_id}: baseline_l1 AUC differs from checkpoint")

        for method in METHODS:
            method_summary = summary_rows.loc[summary_rows.method == method]
            pre_mean = float(_single_value(
                method_summary.loc[method_summary.arm == "pre_split"],
                "auc", f"{dataset_id}.{method}.pre_split",
            ))
            contained_mean = float(_single_value(
                method_summary.loc[method_summary.arm == "fold_contained"],
                "auc", f"{dataset_id}.{method}.fold_contained",
            ))
            mean_gap = pre_mean - contained_mean
            uncertainty_row = uncertainty_rows.loc[
                uncertainty_rows.method == method
            ].iloc[0]
            checkpoint_metrics = record["observed_metrics"][method]
            aggregate_metrics = {
                "pre_split_mean_fold_auc": pre_mean,
                "fold_contained_mean_fold_auc": contained_mean,
                "gap_mean_fold": mean_gap,
                "pre_split_oof_auc": float(
                    uncertainty_row["pre_split_oof_auc"]
                ),
                "fold_contained_oof_auc": float(
                    uncertainty_row["fold_contained_oof_auc"]
                ),
                "gap_oof": float(uncertainty_row["gap_oof"]),
            }
            for key in OBSERVED_METHOD_METRICS:
                _validate_checkpoint_metric(
                    checkpoint_metrics[key],
                    aggregate_metrics[key],
                    dataset_id=dataset_id,
                    method=method,
                    key=key,
                    aggregate_source=(
                        SUMMARY_FILENAME
                        if key.endswith("mean_fold_auc")
                        or key == "gap_mean_fold"
                        else UNCERTAINTY_FILENAME
                    ),
                )
            raw_variance = uncertainty_row["variance"]
            usable_variance = (
                float(raw_variance)
                if pd.notna(raw_variance) and float(raw_variance) > 0
                else math.nan
            )
            variance_status = (
                "weighted"
                if pd.notna(raw_variance) and float(raw_variance) > 0
                else (
                    "nonpositive_not_weighted"
                    if pd.notna(raw_variance) else "unavailable_not_weighted"
                )
            )
            fingerprint = checkpoint["data_fingerprint"]
            provenance_value = record["provenance"]
            qc_row = record["qc"]
            unit_summary = record["independent_units"]
            output.append({
                "dataset_id": dataset_id,
                "method": method,
                "pre_split_auc": aggregate_metrics["pre_split_oof_auc"],
                "fold_contained_auc": aggregate_metrics[
                    "fold_contained_oof_auc"
                ],
                "gap": aggregate_metrics["gap_oof"],
                "baseline_l1_auc": baseline_auc,
                "repository": record["repository"],
                "platform": record["platform"],
                "n_samples": record["n_samples"],
                "n_features": record["n_features"],
                "n_pos": record["n_pos"],
                "n_neg": record["n_neg"],
                "n_observations": unit_summary["n_observations"],
                "minority_observations": unit_summary[
                    "minority_observations"
                ],
                "n_independent_units": unit_summary["n_independent_units"],
                "minority_independent_units": unit_summary[
                    "minority_independent_units"
                ],
                "independent_unit_type": unit_summary[
                    "independent_unit_type"
                ],
                "minority_independent_units_reason": unit_summary[
                    "minority_independent_units_reason"
                ],
                "gap_variance": usable_variance,
                "gap_variance_raw": (
                    float(raw_variance) if pd.notna(raw_variance) else math.nan
                ),
                "gap_variance_status": variance_status,
                "summary_pre_split_mean_fold_auc": pre_mean,
                "summary_fold_contained_mean_fold_auc": contained_mean,
                "summary_gap_mean_fold": mean_gap,
                "statistic_rule": STATISTIC_RULE,
                "effect_estimate_source": EFFECT_SOURCE,
                "baseline_auc_source": BASELINE_SOURCE,
                "gap_variance_source": VARIANCE_SOURCE,
                "uncertainty_unit": uncertainty_row["uncertainty_unit"],
                "uncertainty_reason": uncertainty_row["uncertainty_reason"],
                "n_repeats": int(uncertainty_row["n_repeats"]),
                "requested_bootstrap_replicates": int(
                    uncertainty_row["requested_bootstrap_replicates"]
                ),
                "valid_bootstrap_replicates": int(
                    uncertainty_row["valid_bootstrap_replicates"]
                ),
                "source_repository": record["repository"],
                "assay_platform": record["metadata"].get("assay_platform"),
                "analysis_platform": record["platform"],
                "analysis_config_sha256": checkpoint["analysis_config_sha256"],
                "target_config_sha256": checkpoint["target_config_sha256"],
                "code_sha256": checkpoint["code_sha256"],
                "environment_sha256": checkpoint["environment_sha256"],
                "data_fingerprint_sha256": fingerprint["sha256"],
                "feature_ids_sha256": provenance_value["feature_ids_sha256"],
                "sample_ids_sha256": provenance_value["sample_ids_sha256"],
                "groups_sha256": provenance_value["groups_sha256"],
                "group_codes_sha256": provenance_value["group_codes_sha256"],
                "qc_analysis_eligible": qc_row["analysis_eligible"],
                "qc_classification": qc_row["classification"],
                "qc_primary_null_eligible": qc_row[
                    "primary_null_eligible"
                ],
                "qc_sensitivity_null_eligible": qc_row[
                    "sensitivity_null_eligible"
                ],
                "qc_formal_inference_eligible": qc_row[
                    "formal_inference_eligible"
                ],
            })
    return output
