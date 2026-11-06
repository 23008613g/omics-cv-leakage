"""Runner, aggregation, and QC source validation for meta inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark_v4 import checkpoints
from benchmark_v4.aggregate_results import TABLE_COLUMNS
from benchmark_v4.aggregation_schema import (
    AggregationError,
    OBSERVED_METHOD_METRICS,
    analysis_metadata,
    as_hash,
    as_mapping,
    load_checkpoints,
    provenance,
)
from benchmark_v4.configuration import STATISTIC_RULE
from benchmark_v4.meta_input_contracts import (
    MetaInputError,
    QC_FLOW_FILENAME,
    QC_MANIFEST_FILENAME,
    QC_SUMMARY_FILENAME,
    _AUC_TOLERANCE,
    _OBSERVED_RESULT_FIELDS,
    _QC_CODE_DEPENDENCIES,
    _code_dependency_hashes,
    _fail,
    _regular_file,
    _runner_directory,
)
from benchmark_v4.meta_input_validation import (
    QCProvenanceError,
    derive_independent_unit_summary,
    validate_qc_artifacts,
)
from benchmark_v4.methods import METHODS
from benchmark_v4.runner_support import (
    CANONICAL_DATASET_COUNT,
    CANONICAL_ORDERED_ID_SHA256,
    WhitelistError,
    load_whitelist,
)


def _load_runner_manifest(
    run_dir: Path, loaded: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], Path]:
    path = _regular_file(run_dir / "manifest.json", "runner manifest")
    try:
        with path.open(encoding="utf-8") as handle:
            manifest = as_mapping(json.load(handle), str(path))
    except (OSError, ValueError, json.JSONDecodeError, AggregationError) as exc:
        raise _fail(f"invalid runner manifest: {exc}") from exc
    first = loaded[0]
    if manifest.get("schema_version") != first["schema_version"]:
        raise _fail("runner manifest/checkpoint schema version mismatch")
    for key in (
        "analysis_config_sha256", "target_config_sha256", "code_sha256",
        "environment_sha256",
    ):
        if manifest.get(key) != first[key]:
            raise _fail(f"runner manifest {key} does not match checkpoints")
    protocol = as_mapping(manifest.get("protocol_config"), "protocol_config")
    if protocol.get("statistic_rule") != STATISTIC_RULE:
        raise _fail("runner manifest statistic_rule does not match the six-estimand contract")
    if checkpoints.sha256_json(protocol) != first["target_config_sha256"]:
        raise _fail("runner manifest protocol_config target hash mismatch")
    analysis_payload = {
        key: value for key, value in protocol.items() if key != "n_permutations"
    }
    if checkpoints.sha256_json(analysis_payload) != first["analysis_config_sha256"]:
        raise _fail("runner manifest protocol_config analysis hash mismatch")
    if protocol.get("n_permutations") != first["target"]:
        raise _fail("runner manifest permutation target does not match checkpoints")
    code_files = as_mapping(manifest.get("code_file_sha256"), "code_file_sha256")
    versions = as_mapping(manifest.get("package_versions"), "package_versions")
    for filename, digest in code_files.items():
        if not isinstance(filename, str) or not filename:
            raise _fail("runner manifest code filename is malformed")
        as_hash(digest, f"code_file_sha256.{filename}")
    if checkpoints.sha256_json(code_files) != first["code_sha256"]:
        raise _fail("runner manifest code_file_sha256 does not match code_sha256")
    if checkpoints.sha256_json(versions) != first["environment_sha256"]:
        raise _fail("runner manifest package_versions do not match environment hash")

    raw_selected = manifest.get("selected_dataset_ids")
    if not isinstance(raw_selected, list) or not all(
        isinstance(value, str) and value for value in raw_selected
    ):
        raise _fail("runner manifest selected_dataset_ids is malformed")
    if len(raw_selected) != len(set(raw_selected)):
        raise _fail("runner manifest selected_dataset_ids contains duplicates")
    if manifest.get("selected_dataset_count") != len(raw_selected):
        raise _fail("runner manifest selected_dataset_count is inconsistent")
    by_dataset = {str(checkpoint["dataset"]): checkpoint for checkpoint in loaded}
    if set(raw_selected) != set(by_dataset):
        raise _fail("runner manifest selection does not match checkpoint files")
    raw_summaries = as_mapping(manifest.get("datasets"), "manifest.datasets")
    if set(raw_summaries) != set(raw_selected):
        raise _fail("runner manifest dataset summaries are incomplete")
    for dataset_id in raw_selected:
        checkpoint = by_dataset[dataset_id]
        recorded = as_mapping(
            raw_summaries[dataset_id], f"manifest.datasets.{dataset_id}"
        )
        expected = checkpoints.summary_from_state(checkpoint["state"])
        for key in (
            "dataset_id", "status", "completed_permutations",
            "target_permutations", "observed_split_plan_sha256",
            "permutation_plan_sha256", "data_fingerprint_sha256",
            "permutation_records_sha256", "permutation_block_sha256",
            "checkpoint", "errors",
        ):
            if recorded.get(key) != expected[key]:
                raise _fail(
                    f"runner manifest dataset {dataset_id} field {key} is stale"
                )
    expected_status = (
        "complete"
        if all(
            checkpoint["state"].get("status") == "ok"
            for checkpoint in loaded
        )
        else "running_or_incomplete"
    )
    if manifest.get("status") != expected_status:
        raise _fail("runner manifest status disagrees with checkpoint statuses")
    return manifest, path


def _json_object(path: Path, label: str) -> Mapping[str, Any]:
    _regular_file(path, label)
    try:
        with path.open(encoding="utf-8") as handle:
            return as_mapping(json.load(handle), label)
    except (OSError, ValueError, json.JSONDecodeError, AggregationError) as exc:
        raise _fail(f"invalid {label}: {exc}") from exc


def _aggregation_table_paths(aggregation_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for table_name in TABLE_COLUMNS:
        candidates = [
            path
            for suffix in ("csv", "parquet")
            if (path := aggregation_dir / f"{table_name}.{suffix}").exists()
        ]
        if len(candidates) != 1:
            raise _fail(
                "aggregation directory must contain exactly one CSV or Parquet "
                f"file for {table_name}"
            )
        paths[table_name] = _regular_file(
            candidates[0], f"aggregation table {table_name}"
        )
    return paths


def _cohort_lock_audit(
    runner_manifest: Mapping[str, Any], qc_mode: str
) -> dict[str, Any]:
    selected = runner_manifest.get("selected_dataset_ids")
    if not isinstance(selected, list):
        raise _fail("runner selected_dataset_ids is malformed")
    selected_hash = checkpoints.sha256_json(selected)
    audit = {
        "formal_canonical_cohort_locked": False,
        "canonical_dataset_count": CANONICAL_DATASET_COUNT,
        "canonical_ordered_id_sha256": CANONICAL_ORDERED_ID_SHA256,
        "selected_dataset_count": len(selected),
        "selected_ordered_id_sha256": selected_hash,
    }
    if qc_mode != "formal":
        return audit
    if (
        runner_manifest.get("canonical_whitelist_count")
        != CANONICAL_DATASET_COUNT
        or len(selected) != CANONICAL_DATASET_COUNT
    ):
        raise _fail(
            "formal meta-analysis requires the canonical ordered 1,102-dataset "
            "cohort"
        )
    whitelist_path = Path(str(runner_manifest.get("whitelist_path", "")))
    _regular_file(whitelist_path, "formal canonical whitelist")
    whitelist_hash = checkpoints.sha256_file(whitelist_path)
    if runner_manifest.get("whitelist_sha256") != whitelist_hash:
        raise _fail("formal canonical whitelist SHA256 differs from runner manifest")
    try:
        records = load_whitelist(whitelist_path)
    except (OSError, ValueError, WhitelistError) as exc:
        raise _fail(f"formal canonical whitelist is invalid: {exc}") from exc
    canonical_ids = [record.dataset_id for record in records]
    if selected != canonical_ids:
        raise _fail(
            "formal selected IDs do not match the canonical 1,102 IDs in order"
        )
    if selected_hash != CANONICAL_ORDERED_ID_SHA256:
        raise _fail("formal selected ordered-ID SHA256 is not canonical")
    return {
        **audit,
        "formal_canonical_cohort_locked": True,
        "whitelist_path": str(whitelist_path.resolve()),
        "whitelist_sha256": whitelist_hash,
    }


def _load_qc_artifacts(
    qc_directory: Path | str,
    aggregation_dir: Path,
    run_dir: Path,
    runner_manifest: Mapping[str, Any],
    loaded: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    qc_dir = Path(qc_directory)
    if not qc_dir.is_dir() or qc_dir.is_symlink():
        raise _fail(f"QC directory is missing or unsafe: {qc_dir}")
    flow_path = _regular_file(qc_dir / QC_FLOW_FILENAME, "QC eligibility flow")
    summary_path = _regular_file(qc_dir / QC_SUMMARY_FILENAME, "QC summary")
    manifest_path = _regular_file(qc_dir / QC_MANIFEST_FILENAME, "QC manifest")
    qc_manifest = _json_object(manifest_path, "QC manifest")
    qc_summary = _json_object(summary_path, "QC summary")
    if qc_manifest.get("schema_version") != 1:
        raise _fail("QC manifest schema version is unsupported")

    table_paths = _aggregation_table_paths(aggregation_dir)
    expected_inputs = {
        "run_manifest": checkpoints.sha256_file(run_dir / "manifest.json"),
        **{
            f"aggregation/{path.name}": checkpoints.sha256_file(path)
            for path in table_paths.values()
        },
    }
    recorded_inputs = as_mapping(
        qc_manifest.get("input_files"), "QC manifest.input_files"
    )
    if dict(recorded_inputs) != expected_inputs:
        raise _fail("QC manifest input hashes do not match runner/aggregation files")

    checkpoint_hashes = {
        f"checkpoints/{Path(checkpoint['path']).name}": checkpoints.sha256_file(
            Path(checkpoint["path"])
        )
        for checkpoint in loaded
    }
    if qc_manifest.get("checkpoint_file_set_sha256") != checkpoints.sha256_json(
        checkpoint_hashes
    ):
        raise _fail("QC checkpoint-file-set hash does not match runner checkpoints")

    expected_outputs = {
        QC_FLOW_FILENAME: checkpoints.sha256_file(flow_path),
        QC_SUMMARY_FILENAME: checkpoints.sha256_file(summary_path),
    }
    recorded_outputs = as_mapping(
        qc_manifest.get("output_files"), "QC manifest.output_files"
    )
    if dict(recorded_outputs) != expected_outputs:
        raise _fail("QC output hash does not match the QC flow/summary artifacts")

    expected_qc_code_files = _code_dependency_hashes(_QC_CODE_DEPENDENCIES)
    recorded_qc_code_files = qc_manifest.get("qc_code_file_sha256")
    if not isinstance(recorded_qc_code_files, Mapping):
        raise _fail("QC semantic dependency code-file map is missing or malformed")
    if dict(recorded_qc_code_files) != expected_qc_code_files:
        raise _fail(
            "QC semantic dependency code-file map does not exactly match the "
            "validated QC implementation"
        )
    expected_qc_code_hash = checkpoints.sha256_json(expected_qc_code_files)
    if qc_manifest.get("qc_code_sha256") != expected_qc_code_hash:
        raise _fail("QC code hash does not match the validated QC implementation")
    qc_semantic_binding_hash = checkpoints.sha256_json({
        "qc_manifest_declared_code_sha256": qc_manifest["qc_code_sha256"],
        "semantic_dependency_file_sha256": expected_qc_code_files,
    })

    selected_ids = runner_manifest.get("selected_dataset_ids")
    assert isinstance(selected_ids, list)
    if qc_manifest.get("selected_ids_sha256") != checkpoints.sha256_json(
        selected_ids
    ):
        raise _fail("QC selected-ID hash differs from the runner selection")
    try:
        flow = pd.read_csv(flow_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise _fail(f"could not read QC eligibility flow: {exc}") from exc
    checkpoint_statuses = {
        str(checkpoint["dataset"]): str(checkpoint["state"].get("status", ""))
        for checkpoint in loaded
    }
    protocol = as_mapping(
        runner_manifest.get("protocol_config"), "runner protocol_config"
    )
    target = protocol.get("n_permutations")
    if isinstance(target, bool) or not isinstance(target, int) or target < 0:
        raise _fail("runner permutation target is invalid")
    try:
        validated_flow, eligibility_counts = validate_qc_artifacts(
            flow,
            qc_summary,
            qc_manifest,
            selected_ids=selected_ids,
            checkpoint_statuses=checkpoint_statuses,
            target_permutations=target,
        )
    except QCProvenanceError as exc:
        raise _fail(f"QC eligibility provenance is invalid: {exc}") from exc
    cohort_lock = _cohort_lock_audit(
        runner_manifest, str(qc_summary["mode"])
    )
    audit = {
        "qc_directory": str(qc_dir.resolve()),
        "qc_manifest_sha256": checkpoints.sha256_file(manifest_path),
        "qc_flow_sha256": checkpoints.sha256_file(flow_path),
        "qc_summary_sha256": checkpoints.sha256_file(summary_path),
        "qc_code_sha256": expected_qc_code_hash,
        "qc_declared_code_sha256": qc_manifest["qc_code_sha256"],
        "qc_declared_code_file_sha256": dict(recorded_qc_code_files),
        "qc_code_file_sha256": expected_qc_code_files,
        "qc_semantic_code_sha256": expected_qc_code_hash,
        "qc_semantic_code_binding_sha256": qc_semantic_binding_hash,
        "qc_mode": qc_summary["mode"],
        "qc_status": qc_summary["qc_status"],
        "eligibility_provenance": eligibility_counts,
        "cohort_lock": cohort_lock,
    }
    return validated_flow, audit


def _observed_method_metrics(
    observed_methods: Mapping[str, Any], method: str, dataset_id: str
) -> dict[str, float]:
    wrapper = as_mapping(observed_methods[method], f"{dataset_id}.{method}")
    observed = as_mapping(
        wrapper.get("observed"), f"{dataset_id}.{method}.observed"
    )
    if set(observed) != set(_OBSERVED_RESULT_FIELDS):
        raise _fail(
            f"{dataset_id}.{method}: observed estimand fields do not match "
            "the six estimands required by statistic_rule"
        )
    metrics: dict[str, float] = {}
    for key in OBSERVED_METHOD_METRICS:
        raw = observed.get(key)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            raise _fail(f"{dataset_id}.{method}.{key} must be finite numeric")
        value = float(raw)
        if key.endswith("_auc") and not 0 <= value <= 1:
            raise _fail(f"{dataset_id}.{method}.{key} must lie in [0, 1]")
        if key.startswith("gap_") and not -1 <= value <= 1:
            raise _fail(f"{dataset_id}.{method}.{key} must lie in [-1, 1]")
        metrics[key] = value
    for prefix in ("mean_fold", "oof"):
        pre = metrics[
            "pre_split_mean_fold_auc"
            if prefix == "mean_fold"
            else "pre_split_oof_auc"
        ]
        contained = metrics[
            "fold_contained_mean_fold_auc"
            if prefix == "mean_fold"
            else "fold_contained_oof_auc"
        ]
        gap_key = f"gap_{prefix}"
        if not math.isclose(
            pre - contained,
            metrics[gap_key],
            rel_tol=_AUC_TOLERANCE,
            abs_tol=_AUC_TOLERANCE,
        ):
            raise _fail(
                f"{dataset_id}.{method}.{gap_key} is arithmetically inconsistent"
            )
    return metrics


def _checkpoint_records(
    checkpoint_input: Path | str,
    aggregation_dir: Path,
    qc_directory: Path | str,
) -> tuple[
    dict[str, dict[str, Any]],
    Mapping[str, Any],
    Path,
    Path,
    dict[str, Any],
    pd.DataFrame,
]:
    run_dir = _runner_directory(checkpoint_input)
    try:
        loaded = load_checkpoints(checkpoint_input)
        manifest, manifest_path = _load_runner_manifest(run_dir, loaded)
        flow, qc_audit = _load_qc_artifacts(
            qc_directory, aggregation_dir, run_dir, manifest, loaded
        )
        flow_by_id = {
            str(row.dataset_id): row._asdict()
            for row in flow.itertuples(index=False)
        }
        eligible_ids = {
            dataset_id
            for dataset_id, row in flow_by_id.items()
            if bool(row["analysis_eligible"])
        }
        records: dict[str, dict[str, Any]] = {}
        for checkpoint in loaded:
            dataset_id = str(checkpoint["dataset"])
            if dataset_id not in eligible_ids:
                continue
            metadata = analysis_metadata(checkpoint)
            observed = as_mapping(
                checkpoint["state"].get("observed"),
                f"{checkpoint['path']}.observed",
            )
            if observed.get("statistic_rule") != STATISTIC_RULE:
                raise _fail(
                    f"{dataset_id}: observed statistic_rule does not match "
                    "the six-estimand contract"
                )
            provenance_value = provenance(checkpoint, observed)
            methods = as_mapping(
                observed.get("methods"), f"{checkpoint['path']}.observed.methods"
            )
            if set(methods) != set(METHODS):
                raise _fail(
                    f"{dataset_id}: checkpoint does not contain exactly five methods"
                )
            observed_metrics = {
                method: _observed_method_metrics(methods, method, dataset_id)
                for method in METHODS
            }
            counts = as_mapping(
                metadata.get("class_counts"), f"{checkpoint['path']}.class_counts"
            )
            source_key = str(metadata.get("source_key", "")).strip()
            repository = str(metadata.get("source_repository", "")).strip()
            platform = str(metadata.get("analysis_platform", "")).strip()
            if not source_key or not repository or not platform:
                raise _fail(
                    f"{dataset_id}: source repository and broad platform are required"
                )
            records[dataset_id] = {
                "checkpoint": checkpoint,
                "metadata": metadata,
                "observed": observed,
                "methods": methods,
                "observed_metrics": observed_metrics,
                "provenance": provenance_value,
                "source_key": source_key,
                "repository": repository,
                "platform": platform,
                "n_samples": int(metadata["n_samples"]),
                "n_features": int(metadata["n_features_raw"]),
                "n_neg": int(counts["0"]),
                "n_pos": int(counts["1"]),
                "qc": flow_by_id[dataset_id],
            }
            records[dataset_id]["independent_units"] = (
                derive_independent_unit_summary(records[dataset_id])
            )
    except MetaInputError:
        raise
    except (AggregationError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise _fail(f"invalid runner checkpoint provenance: {exc}") from exc
    return records, manifest, manifest_path, run_dir, qc_audit, flow
