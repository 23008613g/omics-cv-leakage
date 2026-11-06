"""Primitive validators and strict schema-v2 checkpoint loading."""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
import re
from typing import Any

from benchmark_v4.checkpoints import (
    METHOD_ESTIMAND_FIELDS,
    read_gzip_json,
    sha256_json,
)


SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = frozenset({2})
CHECKPOINT_SUFFIX = ".json.gz"
_HASH = re.compile(r"^[0-9a-f]{64}$")
OBSERVED_METHOD_METRICS = METHOD_ESTIMAND_FIELDS
_AUC_ESTIMANDS = frozenset(
    {
        "pre_split_mean_fold_auc",
        "fold_contained_mean_fold_auc",
        "pre_split_oof_auc",
        "fold_contained_oof_auc",
    }
)
_GAP_ESTIMANDS = frozenset({"gap_mean_fold", "gap_oof"})


class AggregationError(ValueError):
    """A checkpoint cannot be safely interpreted or combined."""


def error(context: str, message: str) -> AggregationError:
    return AggregationError(f"{context}: {message}")


def as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error(context, "expected an object")
    return value


def as_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise error(context, "expected a list")
    return value


def as_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise error(context, f"expected an integer >= {minimum}")
    return value


def as_number(
    value: Any, context: str, *, nullable: bool = False
) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(context, "expected a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise error(context, "expected a finite number")
    return number


def as_auc(value: Any, context: str) -> float:
    number = as_number(value, context)
    assert number is not None
    if not 0.0 <= number <= 1.0:
        raise error(context, "AUC is outside [0, 1]")
    return number


def as_gap(value: Any, context: str) -> float:
    number = as_number(value, context)
    assert number is not None
    if not -1.0 <= number <= 1.0:
        raise error(context, "gap is outside [-1, 1]")
    return number


def as_hash(value: Any, context: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise error(context, "expected a lowercase 64-character SHA256")
    return value


def as_text(value: Any, context: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise error(context, "expected text")
    return value


def json_text(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _checkpoint_paths(path: Path) -> list[Path]:
    checkpoint_dir = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    paths = sorted(checkpoint_dir.glob(f"*{CHECKPOINT_SUFFIX}"))
    if not paths:
        raise AggregationError(f"No gzip JSON checkpoints found in {checkpoint_dir}")
    if any(item.is_symlink() or not item.is_file() for item in paths):
        raise AggregationError(f"Unsafe checkpoint path in {checkpoint_dir}")
    return paths


def _data_fingerprint(
    state: Mapping[str, Any], context: str, *, required: bool
) -> dict[str, Any] | None:
    raw = state.get("data_fingerprint")
    if raw is None:
        if required:
            raise error(context, "observed result lacks a data fingerprint")
        return None
    fingerprint = dict(as_mapping(raw, f"{context}.data_fingerprint"))
    components = {
        "x_shape", "x_float64_sha256", "y_shape", "y_int64_sha256",
        "feature_ids_sha256", "sample_ids_sha256", "groups_sha256",
        "group_codes_sha256", "transform_kind",
    }
    if set(fingerprint) != components | {"sha256"}:
        raise error(context, "data fingerprint fields do not match schema v2")
    x_shape = as_list(fingerprint["x_shape"], f"{context}.fingerprint.x_shape")
    y_shape = as_list(fingerprint["y_shape"], f"{context}.fingerprint.y_shape")
    if len(x_shape) != 2 or len(y_shape) != 1:
        raise error(context, "data fingerprint dimensions are malformed")
    fingerprint["x_shape"] = [as_int(v, f"{context}.x_shape") for v in x_shape]
    fingerprint["y_shape"] = [as_int(v, f"{context}.y_shape") for v in y_shape]
    if fingerprint["x_shape"][0] != fingerprint["y_shape"][0]:
        raise error(context, "data fingerprint X/y sample counts disagree")
    for key in (
        "x_float64_sha256", "y_int64_sha256", "feature_ids_sha256",
        "sample_ids_sha256",
    ):
        as_hash(fingerprint[key], f"{context}.fingerprint.{key}")
    for key in ("groups_sha256", "group_codes_sha256"):
        as_hash(fingerprint[key], f"{context}.fingerprint.{key}", nullable=True)
    if (fingerprint["groups_sha256"] is None) != (
        fingerprint["group_codes_sha256"] is None
    ):
        raise error(context, "data fingerprint group hashes disagree")
    as_text(fingerprint["transform_kind"], f"{context}.fingerprint.transform_kind")
    digest = as_hash(fingerprint["sha256"], f"{context}.fingerprint.sha256")
    payload = {key: fingerprint[key] for key in fingerprint if key != "sha256"}
    if sha256_json(payload) != digest:
        raise error(context, "data fingerprint SHA256 mismatch")
    return fingerprint


def _validate_permutation_estimand_ranges(
    state: Mapping[str, Any], context: str
) -> None:
    """Independently bound every stored permutation scalar during loading."""
    blocks = as_list(state.get("permutation_blocks"), f"{context}.blocks")
    for block_position, raw_block in enumerate(blocks):
        block_context = f"{context}.permutation_blocks[{block_position}]"
        block = as_mapping(raw_block, block_context)
        result = as_mapping(block.get("result"), f"{block_context}.result")
        methods = as_mapping(result.get("methods"), f"{block_context}.methods")
        for method, raw_payload in methods.items():
            payload = as_mapping(raw_payload, f"{block_context}.{method}")
            for metric in _AUC_ESTIMANDS | _GAP_ESTIMANDS:
                if metric not in payload:
                    continue
                values = as_list(payload[metric], f"{block_context}.{method}.{metric}")
                for position, value in enumerate(values):
                    value_context = f"{block_context}.{method}.{metric}[{position}]"
                    if metric in _AUC_ESTIMANDS:
                        as_auc(value, value_context)
                    else:
                        as_gap(value, value_context)


def _validate_top_level(state: Mapping[str, Any], path: Path) -> dict[str, Any]:
    context = str(path)
    version = as_int(state.get("schema_version"), f"{context}.schema_version")
    if version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
        supported = ", ".join(map(str, sorted(SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS)))
        raise error(
            context,
            f"unsupported checkpoint schema version {version}; supported: {supported}",
        )
    dataset = as_text(state.get("dataset_id"), f"{context}.dataset_id")
    assert dataset is not None
    if dataset != path.name[: -len(CHECKPOINT_SUFFIX)]:
        raise error(context, "checkpoint dataset ID does not match filename")
    hashes = {
        key: as_hash(state.get(key), f"{context}.{key}")
        for key in (
            "analysis_config_sha256", "target_config_sha256", "code_sha256",
            "environment_sha256", "permutation_plan_sha256",
        )
    }
    target = as_int(state.get("target_permutations"), f"{context}.target")
    _validate_permutation_estimand_ranges(state, context)
    history = as_list(state.get("target_config_history"), f"{context}.history")
    if not history:
        raise error(context, "target_config_history is empty")
    previous = -1
    for position, raw_entry in enumerate(history):
        entry = as_mapping(raw_entry, f"{context}.history[{position}]")
        entry_target = as_int(entry.get("target_permutations"), f"{context}.history.target")
        entry_hash = as_hash(entry.get("target_config_sha256"), f"{context}.history.hash")
        if entry_target <= previous:
            raise error(context, "permutation targets in history are not increasing")
        previous = entry_target
    if entry_target != target or entry_hash != hashes["target_config_sha256"]:
        raise error(context, "latest target config/history mismatch")
    observed = state.get("observed")
    evaluator = as_hash(
        state.get("evaluator_config_sha256"), f"{context}.evaluator_hash",
        nullable=observed is None,
    )
    if observed is not None:
        observed_map = as_mapping(observed, f"{context}.observed")
        observed_hash = as_hash(
            observed_map.get("config_sha256"), f"{context}.observed.config_hash"
        )
        if observed_map.get("dataset_id") != dataset:
            raise error(context, "observed dataset ID mismatch")
        if observed_hash != evaluator or observed_hash != hashes["analysis_config_sha256"]:
            raise error(context, "observed/evaluator/analysis config SHA256 mismatch")
    return {
        "state": state, "path": path, "dataset": dataset,
        "schema_version": version, "target": target, "evaluator_hash": evaluator,
        "data_fingerprint": _data_fingerprint(
            state, context, required=observed is not None
        ),
        **hashes,
    }


def load_checkpoints(checkpoint_input: Path | str) -> list[dict[str, Any]]:
    loaded = [
        _validate_top_level(read_gzip_json(path), path)
        for path in _checkpoint_paths(Path(checkpoint_input))
    ]
    first = loaded[0]
    labels = {
        "analysis_config_sha256": "analysis config SHA256",
        "target_config_sha256": "target config SHA256",
        "code_sha256": "code SHA256",
        "environment_sha256": "environment SHA256",
        "target": "target permutation count",
        "schema_version": "schema version",
    }
    for checkpoint in loaded[1:]:
        for key, label in labels.items():
            if checkpoint[key] != first[key]:
                raise AggregationError(
                    f"Cross-checkpoint {label} mismatch: "
                    f"{first['dataset']} vs {checkpoint['dataset']}"
                )
    return loaded


__all__ = [
    "AggregationError",
    "CHECKPOINT_SUFFIX",
    "OBSERVED_METHOD_METRICS",
    "SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS",
    "as_auc",
    "as_gap",
    "as_hash",
    "as_int",
    "as_list",
    "as_mapping",
    "as_number",
    "as_text",
    "error",
    "json_text",
    "load_checkpoints",
]
