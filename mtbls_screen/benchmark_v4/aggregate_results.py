"""Publish deterministic audit tables from strict schema-v2 checkpoints."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from benchmark_v4.aggregation_schema import (
    AggregationError,
    OBSERVED_METHOD_METRICS,
    SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS,
    aggregate_observed,
    as_hash,
    as_int,
    as_list,
    as_mapping,
    as_number,
    as_text,
    base_row,
    error,
    load_checkpoints,
)
from benchmark_v4.aggregation_uncertainty import append_uncertainty
from benchmark_v4.checkpoints import sha256_file, sha256_json
from benchmark_v4.configuration import STATISTIC_RULE


FORMAL_MIN_PERMUTATIONS = 199
_METRICS = OBSERVED_METHOD_METRICS
_ARMS = {
    "pre_split_mean_fold_auc": "pre_split",
    "fold_contained_mean_fold_auc": "fold_contained",
    "gap_mean_fold": "pre_split_minus_fold_contained_mean_fold",
    "pre_split_oof_auc": "pre_split",
    "fold_contained_oof_auc": "fold_contained",
    "gap_oof": "pre_split_minus_fold_contained_oof",
}

TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "dataset_method_summary": (
        "dataset", "dataset_id", "source", "method", "arm", "auc", "gap",
        "n_outer_folds", "split_strategy", "group_fallback_reason",
        "requested_outer_splits", "effective_outer_splits",
        "group_adjustment_reason", "preprocessing_workspace_sha256",
        "groups_available", "n_groups", "n_samples", "n_features_raw",
        "analysis_config_sha256", "code_sha256", "environment_sha256",
    ),
    "dataset_method_uncertainty": (
        "dataset", "dataset_id", "source", "method", "comparison",
        "pre_split_oof_auc", "fold_contained_oof_auc", "gap_oof",
        "standard_error", "ci_low", "ci_high", "variance",
        "uncertainty_unit", "n_units", "n_samples", "n_repeats",
        "requested_bootstrap_replicates", "valid_bootstrap_replicates",
        "bootstrap_seed", "uncertainty_reason", "groups_available",
    ),
    "outer_fold_predictions": (
        "dataset", "dataset_id", "source", "method", "arm", "outer_index",
        "repeat", "fold", "prediction_position", "test_index", "group_code",
        "y_true", "y_score", "auc", "groups_available", "split_strategy",
    ),
    "outer_fold_diagnostics": (
        "dataset", "dataset_id", "source", "method", "arm", "outer_index",
        "repeat", "fold", "auc", "train_size", "test_size", "p_after",
        "p_after_preprocessing", "saturated_after_preprocessing",
        "selected_raw_indices_json", "eligible_raw_indices_json",
        "selected_feature_ids_json",
        "selected_feature_map_json", "chosen_params_json",
        "inner_candidate_scores_json", "groups_available", "n_groups",
        "split_strategy", "group_fallback_reason", "requested_outer_splits",
        "effective_outer_splits", "group_adjustment_reason",
        "preprocessing_workspace_sha256",
    ),
    "feature_stability": (
        "dataset", "dataset_id", "source", "method", "arm", "n_sets",
        "n_features", "mean_q", "q_equals_p_any", "q_equals_p_all",
        "nogueira", "mean_jaccard", "mean_kuncheva",
        "kuncheva_applicable", "common_eligible_p", "raw_p",
        "common_eligible_coverage", "coverage",
        "q_equals_p_after_preprocessing_any",
        "q_equals_p_after_preprocessing_all",
        "saturation_any", "saturation_all",
        "raw_universe_sensitivity_json",
        "selected_raw_indices_by_fold_json",
        "selected_feature_ids_by_fold_json", "groups_available",
    ),
    "permutation_summary": (
        "dataset", "dataset_id", "source", "method", "arm", "metric",
        "observed_value", "auc", "gap", "target_permutations",
        "completed_permutations", "valid_null_values", "exceedances",
        "empirical_p_plus_one", "bh_fdr", "fdr_family", "null_mean",
        "null_standard_deviation", "null_min", "null_max", "is_formal",
        "p_value_status", "permutation_protocol", "permutation_unit",
        "formal_null_scope", "permutation_plan_sha256",
    ),
    "failures": (
        "dataset", "dataset_id", "source", "status", "phase", "error_type",
        "message", "traceback", "recorded_at", "completed_permutations",
        "target_permutations", "checkpoint_path",
    ),
    "run_inventory": (
        "dataset", "dataset_id", "source", "status", "schema_version",
        "checkpoint_path", "checkpoint_sha256", "checkpoint_size_bytes",
        "created_at", "updated_at", "completed_at", "n_samples",
        "n_features_raw", "transform_kind", "groups_available", "n_groups",
        "observed_available", "completed_permutations", "target_permutations",
        "target_complete", "formal_permutation_eligible",
        "analysis_config_sha256", "target_config_sha256", "code_sha256",
        "environment_sha256", "evaluator_config_sha256",
        "observed_split_plan_sha256", "permutation_plan_sha256",
        "feature_ids_sha256", "sample_ids_sha256", "groups_sha256",
        "group_codes_sha256", "data_fingerprint_sha256",
    ),
}

_SORT_COLUMNS = {
    "dataset_method_summary": ("dataset", "method", "arm"),
    "dataset_method_uncertainty": ("dataset", "method"),
    "outer_fold_predictions": (
        "dataset", "method", "arm", "repeat", "fold", "prediction_position",
    ),
    "outer_fold_diagnostics": (
        "dataset", "method", "arm", "repeat", "fold", "outer_index",
    ),
    "feature_stability": ("dataset", "method", "arm"),
    "permutation_summary": ("dataset", "method", "metric"),
    "failures": ("dataset", "recorded_at", "phase", "error_type"),
    "run_inventory": ("dataset",),
}


def empirical_p_value_plus_one(
    observed: float, null_values: Sequence[float]
) -> tuple[float, int]:
    """Return right-tailed empirical P and exceedances with the +1 rule."""
    finite = [float(value) for value in null_values if math.isfinite(float(value))]
    exceedances = sum(value >= observed for value in finite)
    return (1.0 + exceedances) / (len(finite) + 1.0), exceedances


def _validate_manifest(
    checkpoint_input: Path | str, checkpoints: Sequence[Mapping[str, Any]]
) -> None:
    source = Path(checkpoint_input)
    run_dir = source if (source / "checkpoints").is_dir() else source.parent
    path = run_dir / "manifest.json"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        manifest = as_mapping(json.load(handle), str(path))
    first = checkpoints[0]
    if manifest.get("schema_version") != first["schema_version"]:
        raise error(str(path), "manifest/checkpoint schema version mismatch")
    for key in (
        "analysis_config_sha256", "target_config_sha256", "code_sha256",
        "environment_sha256",
    ):
        if as_hash(manifest.get(key), f"{path}.{key}") != first[key]:
            raise error(str(path), f"manifest/checkpoint {key} mismatch")
    protocol = as_mapping(manifest.get("protocol_config"), f"{path}.protocol_config")
    if protocol.get("statistic_rule") != STATISTIC_RULE:
        raise error(str(path), "protocol_config statistic_rule mismatch")
    if sha256_json(protocol) != first["target_config_sha256"]:
        raise error(str(path), "target config SHA256 does not match protocol_config")
    analysis_payload = {key: value for key, value in protocol.items() if key != "n_permutations"}
    if sha256_json(analysis_payload) != first["analysis_config_sha256"]:
        raise error(str(path), "analysis config SHA256 does not match protocol_config")
    if protocol.get("n_permutations") != first["target"]:
        raise error(str(path), "protocol target B differs from checkpoints")
    code_files = as_mapping(manifest.get("code_file_sha256"), f"{path}.code_files")
    for filename, digest in code_files.items():
        as_hash(digest, f"{path}.code_files.{filename}")
    if sha256_json(code_files) != first["code_sha256"]:
        raise error(str(path), "code SHA256 does not match code_file_sha256")
    versions = as_mapping(manifest.get("package_versions"), f"{path}.versions")
    if sha256_json(versions) != first["environment_sha256"]:
        raise error(str(path), "environment SHA256 does not match package_versions")
    selected = as_list(manifest.get("selected_dataset_ids"), f"{path}.selected_ids")
    if len(selected) != len(set(selected)):
        raise error(str(path), "selected dataset IDs are not unique")
    unknown = {checkpoint["dataset"] for checkpoint in checkpoints}.difference(selected)
    if unknown:
        raise error(str(path), f"checkpoints outside manifest selection: {sorted(unknown)}")
    summaries = as_mapping(manifest.get("datasets"), f"{path}.datasets")
    for checkpoint in checkpoints:
        summary = summaries.get(checkpoint["dataset"])
        if summary is None:
            continue  # An in-progress manifest may not yet contain the worker result.
        summary = as_mapping(summary, f"{path}.datasets.{checkpoint['dataset']}")
        if summary.get("permutation_plan_sha256") != checkpoint["permutation_plan_sha256"]:
            raise error(str(path), "manifest/checkpoint permutation-plan SHA256 mismatch")


def _permutation_values(
    checkpoint: Mapping[str, Any], expected_methods: set[str]
) -> tuple[list[int], dict[str, dict[str, list[float | None]]], str | None, str | None]:
    state, context = checkpoint["state"], str(checkpoint["path"])
    blocks = as_list(state.get("permutation_blocks"), f"{context}.blocks")
    completed: list[int] = []
    values = {method: {metric: [] for metric in _METRICS} for method in expected_methods}
    protocols: set[str] = set()
    units: set[str] = set()
    previous_start = -1
    for position, raw_block in enumerate(blocks):
        block_context = f"{context}.permutation_blocks[{position}]"
        block = as_mapping(raw_block, block_context)
        indices = [as_int(value, f"{block_context}.indices") for value in as_list(
            block.get("indices"), f"{block_context}.indices"
        )]
        if not indices or indices != sorted(set(indices)) or indices[0] <= previous_start:
            raise error(block_context, "indices/blocks must be sorted and unique")
        previous_start = indices[0]
        if block.get("start") != indices[0] or block.get("stop") != indices[-1] + 1:
            raise error(block_context, "start/stop do not match indices")
        if any(index >= checkpoint["target"] for index in indices):
            raise error(block_context, "permutation index exceeds target B")
        if set(indices).intersection(completed):
            raise error(block_context, "permutation blocks overlap")
        result = as_mapping(block.get("result"), f"{block_context}.result")
        if sha256_json(result) != as_hash(
            block.get("result_sha256"), f"{block_context}.result_hash"
        ):
            raise error(block_context, "result SHA256 mismatch")
        if result.get("dataset_id") != checkpoint["dataset"]:
            raise error(block_context, "permutation result dataset ID mismatch")
        if result.get("statistic_rule") != STATISTIC_RULE:
            raise error(block_context, "permutation statistic_rule mismatch")
        if as_hash(result.get("config_sha256"), f"{block_context}.config_hash") != checkpoint["analysis_config_sha256"]:
            raise error(block_context, "permutation evaluator config SHA256 mismatch")
        if result.get("permutation_indices") != indices:
            raise error(block_context, "result permutation indices mismatch")
        records = as_list(result.get("permutations"), f"{block_context}.records")
        if len(records) != len(indices):
            raise error(block_context, "permutation record count mismatch")
        normalized = []
        for index, raw_record in zip(indices, records, strict=True):
            record = as_mapping(raw_record, f"{block_context}.record")
            if record.get("index") != index:
                raise error(block_context, "permutation record index mismatch")
            normalized.append({
                "index": index,
                "label_sha256": as_hash(record.get("label_sha256"), f"{block_context}.label_hash"),
                "split_plan_sha256": as_hash(record.get("split_plan_sha256"), f"{block_context}.split_hash"),
            })
            protocols.add(as_text(record.get("permutation_protocol"), f"{block_context}.protocol") or "")
            units.add(as_text(record.get("permutation_unit"), f"{block_context}.unit") or "")
        if block.get("permutations") != normalized:
            raise error(block_context, "stored permutation audit records mismatch")
        if sha256_json(normalized) != as_hash(
            block.get("permutation_sha256"), f"{block_context}.permutation_hash"
        ):
            raise error(block_context, "permutation SHA256 mismatch")
        result_methods = as_mapping(result.get("methods"), f"{block_context}.methods")
        if set(result_methods) != expected_methods:
            raise error(block_context, "observed/permutation method sets mismatch")
        for method in sorted(expected_methods):
            method_values = as_mapping(result_methods[method], f"{block_context}.{method}")
            if set(method_values) != set(_METRICS):
                raise error(
                    block_context,
                    f"{method} null estimand fields do not match statistic_rule",
                )
            parsed: dict[str, list[float]] = {}
            for metric in _METRICS:
                items = as_list(method_values.get(metric), f"{block_context}.{method}.{metric}")
                if len(items) != len(indices):
                    raise error(block_context, f"{method}.{metric} length mismatch")
                parsed[metric] = []
                for item in items:
                    number = as_number(item, f"{block_context}.{metric}")
                    assert number is not None
                    parsed[metric].append(number)
                values[method][metric].extend(parsed[metric])
            for index_position in range(len(indices)):
                mean_gap = (
                    parsed["pre_split_mean_fold_auc"][index_position]
                    - parsed["fold_contained_mean_fold_auc"][index_position]
                )
                if not math.isclose(
                    mean_gap,
                    parsed["gap_mean_fold"][index_position],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise error(
                        block_context,
                        f"{method}.gap_mean_fold is inconsistent at permutation "
                        f"{indices[index_position]}",
                    )
                oof_gap = (
                    parsed["pre_split_oof_auc"][index_position]
                    - parsed["fold_contained_oof_auc"][index_position]
                )
                if not math.isclose(
                    oof_gap,
                    parsed["gap_oof"][index_position],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise error(
                        block_context,
                        f"{method}.gap_oof is inconsistent at permutation "
                        f"{indices[index_position]}",
                    )
        completed.extend(indices)
    if len(protocols) > 1 or len(units) > 1:
        raise error(context, "permutation protocols/units cannot be pooled")
    protocol, unit = next(iter(protocols), None), next(iter(units), None)
    valid_scope = (
        protocol == "primary_exchangeable" and unit in {"sample", "group"}
    ) or (
        protocol == "paired_repeated_measures_sensitivity" and unit == "within_group"
    )
    if completed and not valid_scope:
        raise error(context, "unsupported permutation protocol/unit combination")
    return completed, values, protocol, unit


def _append_permutation_rows(
    checkpoint: Mapping[str, Any],
    observed_metrics: Mapping[str, Mapping[str, float]],
    completed: Sequence[int],
    values: Mapping[str, Mapping[str, Sequence[float | None]]],
    protocol: str | None,
    unit: str | None,
    tables: dict[str, list[dict[str, Any]]],
) -> bool:
    target = checkpoint["target"]
    target_complete = list(completed) == list(range(target))
    primary = protocol == "primary_exchangeable"
    scope = (
        "primary_full_pipeline" if primary
        else "sensitivity_only_not_primary"
        if protocol == "paired_repeated_measures_sensitivity" else "none"
    )
    all_formal = bool(observed_metrics) and target >= FORMAL_MIN_PERMUTATIONS and primary
    for method in sorted(observed_metrics):
        for metric in _METRICS:
            observed = float(observed_metrics[method][metric])
            finite = [float(value) for value in values[method][metric] if value is not None]
            metric_complete = target_complete and len(finite) == target
            formal = metric_complete and primary and target >= FORMAL_MIN_PERMUTATIONS
            all_formal = all_formal and formal
            p_value = exceedances = None
            if target == 0:
                status = "no_permutations"
            elif not metric_complete:
                status = "incomplete_no_p"
            else:
                p_value, exceedances = empirical_p_value_plus_one(observed, finite)
                if not primary:
                    status = "sensitivity_not_formal"
                elif target < FORMAL_MIN_PERMUTATIONS:
                    status = "pilot_preliminary"
                else:
                    status = "formal"
            null_mean = sum(finite) / len(finite) if finite else None
            null_sd = (
                math.sqrt(sum((value - null_mean) ** 2 for value in finite) / (len(finite) - 1))
                if len(finite) > 1 and null_mean is not None else 0.0 if finite else None
            )
            tables["permutation_summary"].append({
                **base_row(checkpoint), "method": method, "arm": _ARMS[metric],
                "metric": metric, "observed_value": observed,
                "auc": observed if metric.endswith("_auc") else None,
                "gap": observed if metric.startswith("gap_") else None,
                "target_permutations": target,
                "completed_permutations": len(completed),
                "valid_null_values": len(finite), "exceedances": exceedances,
                "empirical_p_plus_one": p_value, "bh_fdr": None,
                "fdr_family": None, "null_mean": null_mean,
                "null_standard_deviation": null_sd,
                "null_min": min(finite) if finite else None,
                "null_max": max(finite) if finite else None,
                "is_formal": formal, "p_value_status": status,
                "permutation_protocol": protocol, "permutation_unit": unit,
                "formal_null_scope": scope,
                "permutation_plan_sha256": checkpoint["permutation_plan_sha256"],
            })
    return all_formal


def _add_bh_fdr(rows: list[dict[str, Any]]) -> None:
    families: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["empirical_p_plus_one"] is None:
            continue
        prefix = (
            "primary_formal"
            if row["p_value_status"] == "formal"
            else "sensitivity"
            if row["formal_null_scope"] == "sensitivity_only_not_primary"
            else "primary_preliminary"
        )
        family = f"{prefix}:{row['p_value_status']}:{row['metric']}"
        if prefix == "primary_formal":
            family = f"primary_formal:{row['metric']}"
        row["fdr_family"] = family
        families.setdefault(family, []).append(row)
    for family_rows in families.values():
        ordered = sorted(
            family_rows,
            key=lambda row: (row["empirical_p_plus_one"], row["dataset"], row["method"]),
        )
        adjusted, running, count = {}, 1.0, len(ordered)
        for rank in range(count, 0, -1):
            row = ordered[rank - 1]
            running = min(running, row["empirical_p_plus_one"] * count / rank)
            adjusted[id(row)] = running
        for row in family_rows:
            row["bh_fdr"] = adjusted[id(row)]


def _append_failures(
    checkpoint: Mapping[str, Any], completed: Sequence[int],
    tables: dict[str, list[dict[str, Any]]],
) -> None:
    state, context = checkpoint["state"], str(checkpoint["path"])
    status = as_text(state.get("status"), f"{context}.status")
    errors = as_list(state.get("errors"), f"{context}.errors")
    for position, raw_error in enumerate(errors):
        item = as_mapping(raw_error, f"{context}.errors[{position}]")
        tables["failures"].append({
            **base_row(checkpoint), "status": status, "phase": item.get("phase"),
            "error_type": item.get("type"), "message": item.get("message"),
            "traceback": item.get("traceback"), "recorded_at": item.get("recorded_at"),
            "completed_permutations": len(completed),
            "target_permutations": checkpoint["target"],
            "checkpoint_path": str(checkpoint["path"]),
        })
    if status != "ok" and not errors:
        tables["failures"].append({
            **base_row(checkpoint), "status": status, "phase": "checkpoint",
            "error_type": status, "message": "checkpoint is not complete",
            "traceback": "", "recorded_at": state.get("updated_at"),
            "completed_permutations": len(completed),
            "target_permutations": checkpoint["target"],
            "checkpoint_path": str(checkpoint["path"]),
        })


def _append_inventory(
    checkpoint: Mapping[str, Any], provenance: Mapping[str, Any] | None,
    completed: Sequence[int], formal: bool, tables: dict[str, list[dict[str, Any]]],
) -> None:
    state, path = checkpoint["state"], checkpoint["path"]
    metadata = as_mapping(state["metadata"], f"{path}.metadata")
    observed = state.get("observed")
    observed_map = as_mapping(observed, f"{path}.observed") if observed is not None else {}
    complete = observed is not None and list(completed) == list(range(checkpoint["target"]))
    if state.get("status") == "ok" and not complete:
        raise error(str(path), "status ok but observed/target B is incomplete")
    fingerprint = checkpoint["data_fingerprint"]
    tables["run_inventory"].append({
        **base_row(checkpoint), "status": state.get("status"),
        "schema_version": checkpoint["schema_version"], "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path), "checkpoint_size_bytes": path.stat().st_size,
        "created_at": state.get("created_at"), "updated_at": state.get("updated_at"),
        "completed_at": state.get("completed_at"), "n_samples": metadata.get("n_samples"),
        "n_features_raw": metadata.get("n_features_raw"),
        "transform_kind": metadata.get("transform_kind"),
        "groups_available": None if provenance is None else provenance["groups_available"],
        "n_groups": None if provenance is None else provenance["n_groups"],
        "observed_available": observed is not None,
        "completed_permutations": len(completed),
        "target_permutations": checkpoint["target"], "target_complete": complete,
        "formal_permutation_eligible": formal,
        "analysis_config_sha256": checkpoint["analysis_config_sha256"],
        "target_config_sha256": checkpoint["target_config_sha256"],
        "code_sha256": checkpoint["code_sha256"],
        "environment_sha256": checkpoint["environment_sha256"],
        "evaluator_config_sha256": checkpoint["evaluator_hash"],
        "observed_split_plan_sha256": observed_map.get("split_plan_sha256"),
        "permutation_plan_sha256": checkpoint["permutation_plan_sha256"],
        "feature_ids_sha256": None if provenance is None else provenance["feature_ids_sha256"],
        "sample_ids_sha256": None if provenance is None else provenance["sample_ids_sha256"],
        "groups_sha256": None if provenance is None else provenance["groups_sha256"],
        "group_codes_sha256": None if provenance is None else provenance["group_codes_sha256"],
        "data_fingerprint_sha256": None if fingerprint is None else fingerprint["sha256"],
    })


def build_tidy_tables(
    checkpoint_input: Path | str, *, bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 20_260_814,
) -> dict[str, list[dict[str, Any]]]:
    """Validate all inputs before returning stable, tidy table rows."""
    if bootstrap_replicates < 2 or bootstrap_seed < 0:
        raise ValueError("bootstrap_replicates >= 2 and bootstrap_seed >= 0 required")
    checkpoints = load_checkpoints(checkpoint_input)
    _validate_manifest(checkpoint_input, checkpoints)
    tables = {name: [] for name in TABLE_COLUMNS}
    for checkpoint in checkpoints:
        metrics, provenance = aggregate_observed(checkpoint, tables)
        append_uncertainty(
            checkpoint, tuple(metrics), provenance, tables,
            bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed,
        )
        completed, values, protocol, unit = _permutation_values(checkpoint, set(metrics))
        formal = _append_permutation_rows(
            checkpoint, metrics, completed, values, protocol, unit, tables
        )
        _append_failures(checkpoint, completed, tables)
        _append_inventory(checkpoint, provenance, completed, formal, tables)
    _add_bh_fdr(tables["permutation_summary"])
    for name, rows in tables.items():
        rows.sort(key=lambda row: tuple(row.get(column) for column in _SORT_COLUMNS[name]))
    return tables


def _pyarrow_available() -> bool:
    return importlib.util.find_spec("pyarrow") is not None


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)
        handle.flush()
        os.fsync(handle.fileno())


def _write_parquet(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pydict({column: [row.get(column) for row in rows] for column in columns})
    pq.write_table(table, path, compression="zstd")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _remove_path(path: Path) -> None:
    shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink(missing_ok=True)


def _publish_directory(staging: Path, output: Path, *, overwrite: bool) -> None:
    if not output.exists():
        os.replace(staging, output)
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".backup", dir=output.parent
    )
    os.close(descriptor)
    backup = Path(name)
    backup.unlink()
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        os.replace(backup, output)
        raise
    _remove_path(backup)


def aggregate_checkpoints(
    checkpoint_input: Path | str, output_dir: Path | str, *,
    output_format: str = "auto", overwrite: bool = False,
    bootstrap_replicates: int = 2_000, bootstrap_seed: int = 20_260_814,
) -> dict[str, Path]:
    """Atomically publish CSV, or Parquet when requested/available."""
    destination = Path(output_dir)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    if output_format not in {"auto", "csv", "parquet"}:
        raise ValueError("output_format must be one of: auto, csv, parquet")
    parquet_available = _pyarrow_available()
    if output_format == "parquet" and not parquet_available:
        raise RuntimeError("Parquet output requires pyarrow; use CSV or install pyarrow")
    selected_format = "parquet" if output_format == "auto" and parquet_available else output_format
    selected_format = "csv" if selected_format == "auto" else selected_format
    tables = build_tidy_tables(
        checkpoint_input, bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    ))
    try:
        for name, columns in TABLE_COLUMNS.items():
            path = staging / f"{name}.{selected_format}"
            writer = _write_parquet if selected_format == "parquet" else _write_csv
            writer(path, columns, tables[name])
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _publish_directory(staging, destination, overwrite=overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {name: destination / f"{name}.{selected_format}" for name in TABLE_COLUMNS}


def aggregate_results(
    checkpoint_input: Path | str, output_dir: Path | str, **kwargs: Any
) -> dict[str, Path]:
    """Readable alias for :func:`aggregate_checkpoints`."""
    return aggregate_checkpoints(checkpoint_input, output_dir, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate benchmark_v4 checkpoints.")
    parser.add_argument("checkpoint_input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--format", dest="output_format", choices=("auto", "csv", "parquet"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_814)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = aggregate_checkpoints(
        args.checkpoint_input, args.output_dir, output_format=args.output_format,
        overwrite=args.overwrite, bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AggregationError", "FORMAL_MIN_PERMUTATIONS",
    "SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS", "TABLE_COLUMNS",
    "aggregate_checkpoints", "aggregate_results", "build_parser",
    "build_tidy_tables", "empirical_p_value_plus_one", "main",
]
