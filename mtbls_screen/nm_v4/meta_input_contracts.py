"""Constants and table contracts for provenance-checked meta inputs."""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path

import pandas as pd

from nm_v4 import checkpoints
from nm_v4.aggregation_schema import OBSERVED_METHOD_METRICS
from nm_v4.methods import METHODS


SUMMARY_FILENAME = "dataset_method_summary.csv"
UNCERTAINTY_FILENAME = "dataset_method_uncertainty.csv"
OUTPUT_FILENAME = "meta_input.csv"
MANIFEST_FILENAME = "meta_input_manifest.json"
QC_FLOW_FILENAME = "dataset_eligibility_flow.csv"
QC_SUMMARY_FILENAME = "qc_summary.json"
QC_MANIFEST_FILENAME = "qc_manifest.json"
EFFECT_SOURCE = (
    f"{UNCERTAINTY_FILENAME}:paired_repeat_averaged_oof_auc"
)
BASELINE_SOURCE = f"{SUMMARY_FILENAME}:mean_outer_fold_auc"
VARIANCE_SOURCE = f"{UNCERTAINTY_FILENAME}:paired_bootstrap_variance"

_SUMMARY_COLUMNS = frozenset({
    "dataset", "dataset_id", "source", "method", "arm", "auc", "gap",
    "n_samples", "n_features_raw", "analysis_config_sha256", "code_sha256",
    "environment_sha256",
})
_HASH_COLUMNS = (
    "analysis_config_sha256", "code_sha256", "environment_sha256"
)
_AUC_TOLERANCE = 1e-10
_OBSERVED_RESULT_FIELDS = frozenset(OBSERVED_METHOD_METRICS) | frozenset({
    "pre_split_folds", "fold_contained_folds", "feature_stability",
})
_META_CODE_DEPENDENCIES = (
    "nm_v4/meta_input.py",
    "nm_v4/meta_input_contracts.py",
    "nm_v4/meta_input_sources.py",
    "nm_v4/meta_input_join.py",
    "nm_v4/meta_input_validation.py",
    "nm_v4/meta_qc_provenance.py",
    "nm_v4/aggregate_results.py",
    "nm_v4/aggregation_schema.py",
    "nm_v4/aggregation_checkpoint_schema.py",
    "nm_v4/aggregation_uncertainty.py",
    "nm_v4/checkpoints.py",
    "nm_v4/configuration.py",
    "nm_v4/meta_analysis.py",
    "nm_v4/meta_models.py",
    "nm_v4/meta_model_common.py",
    "nm_v4/meta_preparation.py",
    "nm_v4/meta_sensitivity.py",
    "nm_v4/methods.py",
    "nm_v4/qc_report.py",
    "nm_v4/qc_validation.py",
    "nm_v4/runner_support.py",
)
_QC_CODE_DEPENDENCIES = (
    "nm_v4/qc_report.py",
    "nm_v4/qc_validation.py",
    "nm_v4/qc_checkpoint_validation.py",
    "nm_v4/qc_contracts.py",
    "nm_v4/qc_nulls.py",
    "nm_v4/aggregate_results.py",
    "nm_v4/aggregation_schema.py",
    "nm_v4/aggregation_checkpoint_schema.py",
    "nm_v4/aggregation_uncertainty.py",
    "nm_v4/checkpoints.py",
    "nm_v4/configuration.py",
    "nm_v4/methods.py",
    "nm_v4/runner_support.py",
    "nm_v4/stability.py",
)


class MetaInputError(ValueError):
    """The aggregate tables cannot be safely joined to their runner inputs."""


def _fail(message: str) -> MetaInputError:
    return MetaInputError(message)


def _regular_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if path.is_symlink() or not path.is_file():
        raise _fail(f"{label} must be a regular non-symlink file: {path}")
    return path


def _code_dependency_hashes(relative_paths: Sequence[str]) -> dict[str, str]:
    workspace_root = Path(__file__).resolve().parent.parent
    return {
        relative_path: checkpoints.sha256_file(
            _regular_file(
                workspace_root / relative_path,
                f"code dependency {relative_path}",
            )
        )
        for relative_path in relative_paths
    }


def _runner_directory(checkpoint_input: Path | str) -> Path:
    source = Path(checkpoint_input)
    if (source / "checkpoints").is_dir():
        return source
    if source.name == "checkpoints" and source.is_dir():
        return source.parent
    raise _fail(
        "checkpoint_input must be a runner directory or its checkpoints directory"
    )


def _read_csv(path: Path, required: frozenset[str], label: str) -> pd.DataFrame:
    _regular_file(path, label)
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise _fail(f"could not read {label}: {exc}") from exc
    if frame.empty:
        raise _fail(f"{label} must not be empty")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise _fail(f"{label} is missing required columns: {missing}")
    return frame.copy(deep=True)


def _normalize_text(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise _fail(f"{label}.{column} must be non-missing")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise _fail(f"{label}.{column} must be non-empty")


def _numeric(
    frame: pd.DataFrame,
    columns: Sequence[str],
    label: str,
    *,
    nullable: frozenset[str] = frozenset(),
) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise _fail(f"{label}.{column} must be numeric") from exc
        if column not in nullable and values.isna().any():
            raise _fail(f"{label}.{column} must not be missing")
        finite = values.notna()
        if not values.loc[finite].map(math.isfinite).all():
            raise _fail(f"{label}.{column} must contain only finite values")
        frame[column] = values


def _integer_series(values: pd.Series, label: str, *, minimum: int = 0) -> pd.Series:
    if not values.map(lambda value: float(value).is_integer()).all():
        raise _fail(f"{label} must contain integers")
    converted = values.astype(int)
    if (converted < minimum).any():
        raise _fail(f"{label} must be >= {minimum}")
    return converted


def _validate_row_identity(frame: pd.DataFrame, label: str) -> None:
    _normalize_text(frame, ("dataset", "dataset_id", "source", "method"), label)
    if not frame["dataset"].equals(frame["dataset_id"]):
        raise _fail(f"{label} dataset and dataset_id columns disagree")


def _validate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    label = SUMMARY_FILENAME
    _validate_row_identity(summary, label)
    _normalize_text(summary, ("arm", *_HASH_COLUMNS), label)
    if summary.duplicated(["dataset_id", "method", "arm"]).any():
        raise _fail(f"{label} contains duplicate (dataset_id, method, arm) rows")
    _numeric(
        summary,
        ("auc", "gap", "n_samples", "n_features_raw"),
        label,
        nullable=frozenset({"gap"}),
    )
    summary["n_samples"] = _integer_series(
        summary["n_samples"], f"{label}.n_samples", minimum=1
    )
    summary["n_features_raw"] = _integer_series(
        summary["n_features_raw"], f"{label}.n_features_raw", minimum=1
    )
    if not summary["auc"].between(0, 1).all():
        raise _fail(f"{label}.auc must lie in [0, 1]")

    expected_methods = set(METHODS)
    for dataset_id, rows in summary.groupby("dataset_id", sort=True):
        supervised = rows.loc[rows.method != "baseline_l1"]
        if set(supervised.method) != expected_methods:
            raise _fail(
                f"{dataset_id}: exactly five supervised methods are required: "
                f"{list(METHODS)}"
            )
        for method in METHODS:
            arms = supervised.loc[supervised.method == method]
            if set(arms.arm) != {"pre_split", "fold_contained"} or len(arms) != 2:
                raise _fail(
                    f"{dataset_id}.{method}: exactly two arms are required"
                )
            pre = float(arms.loc[arms.arm == "pre_split", "auc"].iloc[0])
            contained = float(
                arms.loc[arms.arm == "fold_contained", "auc"].iloc[0]
            )
            gaps = arms["gap"].to_numpy(float)
            if not math.isclose(
                pre - contained, float(gaps[0]),
                rel_tol=_AUC_TOLERANCE, abs_tol=_AUC_TOLERANCE,
            ) or not math.isclose(
                float(gaps[0]), float(gaps[1]),
                rel_tol=_AUC_TOLERANCE, abs_tol=_AUC_TOLERANCE,
            ):
                raise _fail(f"{dataset_id}.{method}: summary gap is inconsistent")
        baseline = rows.loc[rows.method == "baseline_l1"]
        if len(baseline) != 1 or baseline.iloc[0]["arm"] != "fold_contained":
            raise _fail(
                f"{dataset_id}: exactly one fold-contained baseline_l1 row is required"
            )
        if pd.notna(baseline.iloc[0]["gap"]):
            raise _fail(f"{dataset_id}: baseline_l1 gap must be missing")
        if len(rows) != 2 * len(METHODS) + 1:
            raise _fail(f"{dataset_id}: unexpected summary rows or methods")
        invariant_columns = (
            "dataset", "source", "n_samples", "n_features_raw", *_HASH_COLUMNS
        )
        for column in invariant_columns:
            if rows[column].nunique(dropna=False) != 1:
                raise _fail(f"{dataset_id}: {column} varies across summary rows")
    return summary
