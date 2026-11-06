"""Build a provenance-checked meta-analysis input from aggregate CSV files."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math as math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd

from benchmark_v4 import checkpoints
from benchmark_v4.aggregate_results import TABLE_COLUMNS as TABLE_COLUMNS
from benchmark_v4.aggregation_schema import (
    AggregationError,
    OBSERVED_METHOD_METRICS,
    analysis_metadata as analysis_metadata,
    as_hash as as_hash,
    as_mapping as as_mapping,
    load_checkpoints as load_checkpoints,
    provenance as provenance,
)
from benchmark_v4.configuration import STATISTIC_RULE
from benchmark_v4.meta_analysis import prepare_tidy_summary
from benchmark_v4.meta_input_contracts import (
    BASELINE_SOURCE as BASELINE_SOURCE,
    EFFECT_SOURCE as EFFECT_SOURCE,
    MANIFEST_FILENAME as MANIFEST_FILENAME,
    OUTPUT_FILENAME as OUTPUT_FILENAME,
    QC_FLOW_FILENAME as QC_FLOW_FILENAME,
    QC_MANIFEST_FILENAME as QC_MANIFEST_FILENAME,
    QC_SUMMARY_FILENAME as QC_SUMMARY_FILENAME,
    SUMMARY_FILENAME as SUMMARY_FILENAME,
    UNCERTAINTY_FILENAME as UNCERTAINTY_FILENAME,
    VARIANCE_SOURCE as VARIANCE_SOURCE,
    MetaInputError as MetaInputError,
    _AUC_TOLERANCE as _AUC_TOLERANCE,
    _HASH_COLUMNS as _HASH_COLUMNS,
    _META_CODE_DEPENDENCIES as _META_CODE_DEPENDENCIES,
    _OBSERVED_RESULT_FIELDS as _OBSERVED_RESULT_FIELDS,
    _QC_CODE_DEPENDENCIES as _QC_CODE_DEPENDENCIES,
    _SUMMARY_COLUMNS as _SUMMARY_COLUMNS,
    _code_dependency_hashes as _code_dependency_hashes,
    _fail as _fail,
    _integer_series as _integer_series,
    _normalize_text as _normalize_text,
    _numeric as _numeric,
    _read_csv as _read_csv,
    _regular_file as _regular_file,
    _runner_directory as _runner_directory,
    _validate_row_identity as _validate_row_identity,
    _validate_summary as _validate_summary,
)
from benchmark_v4.meta_input_join import (
    _filter_analysis_eligible_frames as _filter_analysis_eligible_frames,
    _joined_rows as _joined_rows,
    _single_value as _single_value,
    _validate_checkpoint_metric as _validate_checkpoint_metric,
)
from benchmark_v4.meta_input_sources import (
    _aggregation_table_paths as _aggregation_table_paths,
    _checkpoint_records as _checkpoint_records,
    _cohort_lock_audit as _cohort_lock_audit,
    _json_object as _json_object,
    _load_qc_artifacts as _load_qc_artifacts,
    _load_runner_manifest as _load_runner_manifest,
    _observed_method_metrics as _observed_method_metrics,
)
from benchmark_v4.meta_input_validation import (
    QCProvenanceError as QCProvenanceError,
    UNCERTAINTY_REQUIRED_COLUMNS,
    UncertaintyProvenanceError,
    derive_independent_unit_summary as derive_independent_unit_summary,
    validate_qc_artifacts as validate_qc_artifacts,
    validate_uncertainty_frame,
    verify_uncertainty_against_checkpoints,
)
from benchmark_v4.methods import METHODS
from benchmark_v4.runner_support import (
    CANONICAL_DATASET_COUNT as CANONICAL_DATASET_COUNT,
    CANONICAL_ORDERED_ID_SHA256 as CANONICAL_ORDERED_ID_SHA256,
    WhitelistError as WhitelistError,
    load_whitelist as load_whitelist,
)


def build_meta_input(
    aggregation_directory: Path | str,
    checkpoint_input: Path | str,
    qc_directory: Path | str,
) -> pd.DataFrame:
    """Return one validated row per dataset and supervised method.

    Effect estimates use paired repeat-averaged out-of-fold predictions from
    ``dataset_method_uncertainty.csv``. Mean fold AUCs and the L1 baseline from
    ``dataset_method_summary.csv`` are retained as explicit audit columns.
    """
    aggregation_dir = Path(aggregation_directory)
    if not aggregation_dir.is_dir() or aggregation_dir.is_symlink():
        raise _fail(
            f"aggregation_directory must be a regular directory: {aggregation_dir}"
        )
    summary = _validate_summary(_read_csv(
        aggregation_dir / SUMMARY_FILENAME, _SUMMARY_COLUMNS, SUMMARY_FILENAME
    ))
    raw_uncertainty = _read_csv(
        aggregation_dir / UNCERTAINTY_FILENAME,
        UNCERTAINTY_REQUIRED_COLUMNS,
        UNCERTAINTY_FILENAME,
    )
    try:
        uncertainty = validate_uncertainty_frame(raw_uncertainty)
    except UncertaintyProvenanceError as exc:
        raise _fail(f"{UNCERTAINTY_FILENAME}: {exc}") from exc
    records, _, _, _, qc_audit, qc_flow = _checkpoint_records(
        checkpoint_input, aggregation_dir, qc_directory
    )
    summary, uncertainty, aggregation_eligibility_audit = (
        _filter_analysis_eligible_frames(
            summary,
            uncertainty,
            qc_flow,
            records,
        )
    )
    qc_audit = {
        **qc_audit,
        "aggregation_eligibility_audit": aggregation_eligibility_audit,
    }
    try:
        verify_uncertainty_against_checkpoints(uncertainty, records)
    except (AggregationError, UncertaintyProvenanceError) as exc:
        raise _fail(f"uncertainty/checkpoint provenance mismatch: {exc}") from exc
    try:
        joined = _joined_rows(summary, uncertainty, records)
    except MetaInputError:
        raise
    except (AggregationError, TypeError, ValueError) as exc:
        raise _fail(f"aggregate/checkpoint metric provenance mismatch: {exc}") from exc
    result = pd.DataFrame(joined)
    if result.empty or result.duplicated(["dataset_id", "method"]).any():
        raise _fail("meta input must contain unique non-empty dataset-method rows")
    try:
        prepare_tidy_summary(result)
    except ValueError as exc:
        raise _fail(f"meta input is incompatible with prepare_tidy_summary: {exc}") from exc
    result = result.reset_index(drop=True)
    result.attrs["qc_audit"] = qc_audit
    return result


def _timestamp(generated_at_utc: str | None) -> str:
    if generated_at_utc is not None:
        return generated_at_utc
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _input_snapshot(
    aggregation_dir: Path, run_dir: Path, qc_dir: Path
) -> dict[str, dict[str, str]]:
    table_paths = _aggregation_table_paths(aggregation_dir)
    inputs: dict[str, Path] = {
        SUMMARY_FILENAME: _regular_file(
            aggregation_dir / SUMMARY_FILENAME, SUMMARY_FILENAME
        ),
        UNCERTAINTY_FILENAME: _regular_file(
            aggregation_dir / UNCERTAINTY_FILENAME, UNCERTAINTY_FILENAME
        ),
        "runner_manifest.json": _regular_file(
            run_dir / "manifest.json", "runner manifest"
        ),
        f"qc/{QC_FLOW_FILENAME}": _regular_file(
            qc_dir / QC_FLOW_FILENAME, "QC eligibility flow"
        ),
        f"qc/{QC_SUMMARY_FILENAME}": _regular_file(
            qc_dir / QC_SUMMARY_FILENAME, "QC summary"
        ),
        f"qc/{QC_MANIFEST_FILENAME}": _regular_file(
            qc_dir / QC_MANIFEST_FILENAME, "QC manifest"
        ),
    }
    runner_manifest = _json_object(
        run_dir / "manifest.json", "runner manifest snapshot"
    )
    raw_whitelist_path = runner_manifest.get("whitelist_path")
    if isinstance(raw_whitelist_path, str) and raw_whitelist_path.strip():
        inputs["runner_whitelist.csv"] = _regular_file(
            Path(raw_whitelist_path), "runner whitelist snapshot"
        )
    inputs.update({
        f"aggregation/{path.name}": path
        for name, path in table_paths.items()
        if name not in {"dataset_method_summary", "dataset_method_uncertainty"}
    })
    checkpoint_paths = sorted((run_dir / "checkpoints").glob("*.json.gz"))
    if not checkpoint_paths:
        raise _fail("runner contains no checkpoint files")
    for path in checkpoint_paths:
        _regular_file(path, "runner checkpoint")
    return {
        "input_sha256": {
            name: checkpoints.sha256_file(path) for name, path in inputs.items()
        },
        "checkpoint_files_sha256": {
            path.name: checkpoints.sha256_file(path) for path in checkpoint_paths
        },
        "code_files_sha256": _code_dependency_hashes(tuple(dict.fromkeys(
            (*_META_CODE_DEPENDENCIES, *_QC_CODE_DEPENDENCIES)
        ))),
    }


def _meta_input_manifest(
    table_path: Path,
    table: pd.DataFrame,
    input_snapshot: Mapping[str, Mapping[str, str]],
    qc_audit: Mapping[str, Any],
    *,
    generated_at_utc: str | None,
) -> dict[str, Any]:
    code_files = _code_dependency_hashes(_META_CODE_DEPENDENCIES)
    code_combined_sha256 = checkpoints.sha256_json(code_files)
    return {
        "schema_version": 1,
        "generated_at_utc": _timestamp(generated_at_utc),
        "n_rows": len(table),
        "n_datasets": int(table["dataset_id"].nunique()),
        "methods": list(METHODS),
        "statistic_rule": STATISTIC_RULE,
        "checkpoint_estimand_fields": list(OBSERVED_METHOD_METRICS),
        "row_unit": "one row per dataset_id and supervised method",
        "effect_estimate_source": EFFECT_SOURCE,
        "baseline_auc_source": BASELINE_SOURCE,
        "gap_variance_source": VARIANCE_SOURCE,
        "zero_variance_policy": (
            "raw zero is retained in gap_variance_raw; gap_variance is missing "
            "to prevent infinite inverse-variance weight"
        ),
        "independent_unit_count_policy": {
            "observations": "checkpoint OOF y_true indexed by raw observation",
            "no_groups": "each observation is one independent unit",
            "constant_label_groups": (
                "groups are independent units and minority units are counted "
                "from the group-level binary outcome"
            ),
            "mixed_label_groups": (
                "n_independent_units is the number of groups; "
                "minority_independent_units is missing because no unique "
                "group-level minority outcome is defined"
            ),
        },
        "input_sha256": dict(input_snapshot["input_sha256"]),
        "checkpoint_files_sha256": dict(
            input_snapshot["checkpoint_files_sha256"]
        ),
        "qc_manifest_sha256": qc_audit["qc_manifest_sha256"],
        "qc_flow_sha256": qc_audit["qc_flow_sha256"],
        "qc_summary_sha256": qc_audit["qc_summary_sha256"],
        "qc_code_sha256": qc_audit["qc_code_sha256"],
        "qc_declared_code_sha256": qc_audit[
            "qc_declared_code_sha256"
        ],
        "qc_declared_code_file_sha256": dict(
            qc_audit["qc_declared_code_file_sha256"]
        ),
        "qc_code_file_sha256": dict(qc_audit["qc_code_file_sha256"]),
        "qc_semantic_code_sha256": qc_audit[
            "qc_semantic_code_sha256"
        ],
        "qc_semantic_code_binding_sha256": qc_audit[
            "qc_semantic_code_binding_sha256"
        ],
        "qc_mode": qc_audit["qc_mode"],
        "qc_status": qc_audit["qc_status"],
        "cohort_lock": dict(qc_audit["cohort_lock"]),
        "eligibility_provenance": dict(
            qc_audit["eligibility_provenance"]
        ),
        "aggregation_eligibility_audit": dict(
            qc_audit["aggregation_eligibility_audit"]
        ),
        "meta_input_py_sha256": code_files["benchmark_v4/meta_input.py"],
        "meta_input_validation_py_sha256": code_files[
            "benchmark_v4/meta_input_validation.py"
        ],
        "code_file_sha256": code_files,
        "code_combined_sha256": code_combined_sha256,
        "output_csv_sha256": checkpoints.sha256_file(table_path),
        "provenance_hashes": {
            column: sorted(table[column].dropna().astype(str).unique().tolist())
            for column in (
                "analysis_config_sha256", "target_config_sha256", "code_sha256",
                "environment_sha256",
            )
        },
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory(staging: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    descriptor, backup_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".backup", dir=destination.parent
    )
    os.close(descriptor)
    backup = Path(backup_name)
    backup.unlink()
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup) if backup.is_dir() else backup.unlink(missing_ok=True)


def publish_meta_input(
    aggregation_directory: Path | str,
    checkpoint_input: Path | str,
    qc_directory: Path | str,
    output_directory: Path | str,
    *,
    overwrite: bool = False,
    generated_at_utc: str | None = None,
) -> dict[str, Path]:
    """Atomically publish the validated CSV and its provenance manifest."""
    aggregation_dir = Path(aggregation_directory)
    run_dir = _runner_directory(checkpoint_input)
    qc_dir = Path(qc_directory)
    destination = Path(output_directory)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    if destination.resolve() in {
        aggregation_dir.resolve(), run_dir.resolve(), qc_dir.resolve()
    }:
        raise _fail("output_directory must differ from all input directories")
    input_snapshot = _input_snapshot(aggregation_dir, run_dir, qc_dir)
    table = build_meta_input(aggregation_dir, checkpoint_input, qc_dir)
    qc_audit = table.attrs.get("qc_audit")
    if not isinstance(qc_audit, Mapping):
        raise _fail("meta input lacks validated QC audit provenance")
    if input_snapshot != _input_snapshot(aggregation_dir, run_dir, qc_dir):
        raise _fail("input files changed during meta-input construction")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    ))
    try:
        table_path = staging / OUTPUT_FILENAME
        _write_bytes(
            table_path,
            table.to_csv(
                index=False, lineterminator="\n", float_format="%.17g", na_rep=""
            ).encode("utf-8"),
        )
        manifest = _meta_input_manifest(
            table_path,
            table,
            input_snapshot,
            qc_audit,
            generated_at_utc=generated_at_utc,
        )
        manifest_path = staging / MANIFEST_FILENAME
        _write_bytes(
            manifest_path,
            (json.dumps(
                manifest, sort_keys=True, indent=2, ensure_ascii=False,
                allow_nan=False,
            ) + "\n").encode("utf-8"),
        )
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if input_snapshot != _input_snapshot(
            aggregation_dir, run_dir, qc_dir
        ):
            raise _fail("input files changed during final publication staging")
        _publish_directory(staging, destination, overwrite=overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "table": destination / OUTPUT_FILENAME,
        "manifest": destination / MANIFEST_FILENAME,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a provenance-checked meta-analysis input from benchmark_v4 "
            "aggregate CSV files and runner checkpoints."
        )
    )
    parser.add_argument("aggregation_directory", type=Path)
    parser.add_argument("checkpoint_input", type=Path)
    parser.add_argument("qc_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = publish_meta_input(
        args.aggregation_directory,
        args.checkpoint_input,
        args.qc_directory,
        args.output_directory,
        overwrite=args.overwrite,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_SOURCE", "EFFECT_SOURCE", "MANIFEST_FILENAME", "METHODS",
    "MetaInputError", "OUTPUT_FILENAME", "SUMMARY_FILENAME",
    "UNCERTAINTY_FILENAME", "VARIANCE_SOURCE", "build_meta_input",
    "build_parser", "main", "publish_meta_input",
]
