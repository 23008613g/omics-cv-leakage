"""Atomic, hash-guarded checkpoints for the v4 benchmark runner."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence

import fcntl

from benchmark_v4.configuration import STATISTIC_RULE


SCHEMA_VERSION = 2
METHOD_ESTIMAND_FIELDS = (
    "pre_split_mean_fold_auc",
    "fold_contained_mean_fold_auc",
    "gap_mean_fold",
    "pre_split_oof_auc",
    "fold_contained_oof_auc",
    "gap_oof",
)
PERMUTATION_INFERENCE_AUDIT_SCOPE = (
    "hash_verified_permutation_scalar_estimands_for_p_and_bh_"
    "without_prediction_replay_v1"
)
_SAFE_DATASET_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PERMUTATION_PREDICTION_FIELDS = frozenset(
    {
        "fold_contained_folds",
        "fold_predictions",
        "folds",
        "pre_split_folds",
        "predictions",
        "y_score",
        "y_true",
    }
)


class CheckpointMismatchError(RuntimeError):
    """A checkpoint was produced by different code or analysis settings."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    """Convert scientific Python values to deterministic strict JSON values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_json_safe(item) for item in value]
        return sorted(converted, key=lambda item: repr(item))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload_writer: Callable[[Any], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as raw_handle:
            temporary_path = Path(raw_handle.name)
            payload_writer(raw_handle)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is unavailable on some filesystems.  The file
            # itself was already fsynced and atomically replaced.
            pass
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def atomic_write_gzip_json(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value)

    def writer(raw_handle: Any) -> None:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as compressed:
            compressed.write(payload)

    _atomic_write(path, writer)


def atomic_write_json(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    _atomic_write(path, lambda handle: handle.write(payload))


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Checkpoint {path} is not a JSON object")
    return loaded


class CheckpointStore:
    """Atomic per-dataset checkpoints guarded by exact analysis/code hashes."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        analysis_config_sha256: str,
        code_sha256: str,
        environment_sha256: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.analysis_config_sha256 = str(analysis_config_sha256)
        self.code_sha256 = str(code_sha256)
        self.environment_sha256 = str(environment_sha256)

    def path_for(self, dataset_id: str) -> Path:
        if not _SAFE_DATASET_ID.fullmatch(dataset_id):
            raise ValueError(f"Unsafe dataset ID: {dataset_id!r}")
        return self.checkpoint_dir / f"{dataset_id}.json.gz"

    @contextmanager
    def dataset_lock(self, dataset_id: str) -> Any:
        """Serialize independent runner processes targeting the same ID."""
        if not _SAFE_DATASET_ID.fullmatch(dataset_id):
            raise ValueError(f"Unsafe dataset ID: {dataset_id!r}")
        lock_dir = self.output_dir / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{dataset_id}.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _validate(
        self,
        state: Mapping[str, Any],
        dataset_id: str,
        permutation_plan_sha256: str | None,
    ) -> None:
        if state.get("schema_version") != SCHEMA_VERSION:
            raise CheckpointMismatchError("checkpoint schema version mismatch")
        if state.get("dataset_id") != dataset_id:
            raise CheckpointMismatchError("checkpoint dataset ID mismatch")
        if state.get("analysis_config_sha256") != self.analysis_config_sha256:
            raise CheckpointMismatchError("checkpoint config SHA256 mismatch")
        if state.get("code_sha256") != self.code_sha256:
            raise CheckpointMismatchError("checkpoint code SHA256 mismatch")
        if state.get("environment_sha256") != self.environment_sha256:
            raise CheckpointMismatchError("checkpoint environment SHA256 mismatch")
        if (
            permutation_plan_sha256 is not None
            and state.get("permutation_plan_sha256") != permutation_plan_sha256
        ):
            raise CheckpointMismatchError("checkpoint permutation-plan SHA256 mismatch")

    def load(
        self,
        dataset_id: str,
        permutation_plan_sha256: str | None = None,
    ) -> dict[str, Any] | None:
        path = self.path_for(dataset_id)
        if not path.exists():
            return None
        try:
            state = read_gzip_json(path)
            self._validate(state, dataset_id, permutation_plan_sha256)
            _validate_checkpoint_permutation_payloads(state)
        except CheckpointMismatchError:
            raise
        except (EOFError, KeyError, OSError, TypeError, UnicodeError, ValueError) as exc:
            raise CheckpointMismatchError(
                f"checkpoint permutation payload mismatch: {exc}"
            ) from exc
        return state

    def save(self, state: Mapping[str, Any]) -> None:
        dataset_id = str(state.get("dataset_id", ""))
        self._validate(
            state,
            dataset_id,
            str(state.get("permutation_plan_sha256", "")),
        )
        _validate_checkpoint_permutation_payloads(state)
        atomic_write_gzip_json(self.path_for(dataset_id), state)


def new_checkpoint(
    *,
    dataset_id: str,
    analysis_config_sha256: str,
    target_config_sha256: str,
    code_sha256: str,
    environment_sha256: str,
    permutation_plan_sha256: str,
    target_permutations: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "status": "running",
        "analysis_config_sha256": analysis_config_sha256,
        "target_config_sha256": target_config_sha256,
        "target_config_history": [
            {
                "target_permutations": int(target_permutations),
                "target_config_sha256": target_config_sha256,
                "recorded_at": now,
            }
        ],
        "code_sha256": code_sha256,
        "environment_sha256": environment_sha256,
        "permutation_plan_sha256": permutation_plan_sha256,
        "data_fingerprint": None,
        "target_permutations": int(target_permutations),
        "evaluator_config_sha256": None,
        "metadata": dict(metadata or {}),
        "observed": None,
        "permutation_blocks": [],
        "errors": [],
        "created_at": now,
        "updated_at": now,
    }


def _with_evaluator_config(
    state: dict[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    evaluator_hash = result.get("config_sha256")
    if not evaluator_hash:
        raise ValueError("evaluator result is missing config_sha256")
    previous = state.get("evaluator_config_sha256")
    if previous is not None and previous != evaluator_hash:
        raise CheckpointMismatchError("evaluator config SHA256 mismatch")
    state["evaluator_config_sha256"] = evaluator_hash
    return state


def merge_observed(
    checkpoint: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    current = checkpoint.get("observed")
    candidate = copy.deepcopy(dict(result))
    if current is not None:
        if sha256_json(current) != sha256_json(candidate):
            raise CheckpointMismatchError("observed-result hash mismatch")
        return dict(checkpoint)
    updated = _with_evaluator_config(copy.deepcopy(dict(checkpoint)), candidate)
    updated["observed"] = candidate
    updated["updated_at"] = utc_now()
    return updated


def _permutation_records(
    result: Mapping[str, Any], expected_indices: Sequence[int]
) -> list[dict[str, Any]]:
    records = result.get("permutations")
    if not isinstance(records, list):
        raise ValueError("permutation result is missing a permutations list")
    normalized: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("permutation record must be an object")
        index = record.get("index")
        if type(index) is not int:
            raise ValueError(
                f"permutation record {position} index must be a native int"
            )
        label_hash = record.get("label_sha256")
        split_hash = record.get("split_plan_sha256")
        if (
            not isinstance(label_hash, str)
            or not _SHA256.fullmatch(label_hash)
            or not isinstance(split_hash, str)
            or not _SHA256.fullmatch(split_hash)
        ):
            raise ValueError(f"permutation {index} lacks label or split-plan SHA256")
        normalized.append(
            {
                "index": index,
                "label_sha256": label_hash,
                "split_plan_sha256": split_hash,
            }
        )
    if [record["index"] for record in normalized] != list(expected_indices):
        raise ValueError("evaluator returned unexpected permutation indices")
    return normalized


def _reject_permutation_prediction_payloads(
    value: Any, *, context: str = "permutation result"
) -> None:
    """Reject fold/sample predictions; checkpoints retain scalar nulls only."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            field = str(key).casefold()
            if field in _PERMUTATION_PREDICTION_FIELDS:
                raise ValueError(
                    f"{context}: permutation prediction payloads are not stored "
                    f"({key!r})"
                )
            _reject_permutation_prediction_payloads(
                item, context=f"{context}.{key}"
            )
        return
    if isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            _reject_permutation_prediction_payloads(
                item, context=f"{context}[{position}]"
            )


def _validate_permutation_estimands(
    result: Mapping[str, Any], expected_indices: Sequence[int]
) -> None:
    """Validate the six scalar null estimands stored for each method.

    Per-permutation fold or sample predictions are deliberately rejected to
    keep checkpoints bounded. Downstream QC may recompute P values and BH FDR
    from these hash-verified scalar statistics; it cannot replay or recompute
    model-prediction statistics from the checkpoint.
    """
    methods = result.get("methods")
    if not isinstance(methods, Mapping) or not methods:
        raise ValueError("permutation methods must be a non-empty object")
    expected_count = len(expected_indices)
    auc_fields = frozenset(
        {
            "pre_split_mean_fold_auc",
            "fold_contained_mean_fold_auc",
            "pre_split_oof_auc",
            "fold_contained_oof_auc",
        }
    )
    for method, payload in methods.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"permutation method {method!r} must be an object")
        if set(payload) != set(METHOD_ESTIMAND_FIELDS):
            raise ValueError(
                f"permutation method {method!r} estimand fields must exactly "
                f"match {METHOD_ESTIMAND_FIELDS!r}"
            )
        parsed: dict[str, list[float]] = {}
        for metric in METHOD_ESTIMAND_FIELDS:
            values = payload[metric]
            if not isinstance(values, list):
                raise ValueError(f"permutation metric {method}.{metric} must be a list")
            if len(values) != expected_count:
                raise ValueError(
                    f"permutation metric {method}.{metric} length "
                    f"{len(values)} does not match {expected_count} indices"
                )
            parsed[metric] = []
            for position, value in enumerate(values):
                if type(value) not in (int, float):
                    raise ValueError(
                        f"permutation metric {method}.{metric}[{position}] "
                        "must be a native int or float"
                    )
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(
                        f"permutation metric {method}.{metric}[{position}] "
                        "must be finite"
                    )
                lower, upper = ((0.0, 1.0) if metric in auc_fields else (-1.0, 1.0))
                if not lower <= number <= upper:
                    raise ValueError(
                        f"permutation metric {method}.{metric}[{position}] is "
                        f"outside [{lower:g}, {upper:g}]"
                    )
                parsed[metric].append(number)
        gap_pairs = (
            (
                "gap_mean_fold",
                "pre_split_mean_fold_auc",
                "fold_contained_mean_fold_auc",
            ),
            ("gap_oof", "pre_split_oof_auc", "fold_contained_oof_auc"),
        )
        for position, permutation_index in enumerate(expected_indices):
            for gap, pre_split, fold_contained in gap_pairs:
                expected_gap = (
                    parsed[pre_split][position] - parsed[fold_contained][position]
                )
                if not math.isclose(
                    parsed[gap][position],
                    expected_gap,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"permutation metric {method}.{gap} is inconsistent at "
                        f"permutation {permutation_index}"
                    )


def _validate_permutation_result(
    result: Mapping[str, Any],
    expected_indices: Sequence[int],
    *,
    dataset_id: str | None = None,
    config_sha256: str | None = None,
    expected_methods: set[str] | None = None,
) -> list[dict[str, Any]]:
    _reject_permutation_prediction_payloads(result)
    records = _permutation_records(result, expected_indices)
    _validate_permutation_estimands(result, expected_indices)
    raw_indices = result.get("permutation_indices")
    if not isinstance(raw_indices, list) or any(
        type(index) is not int for index in raw_indices
    ):
        raise ValueError("result permutation_indices must be native-int list")
    if raw_indices != list(expected_indices):
        raise ValueError("result permutation_indices do not match block indices")
    if dataset_id is not None and result.get("dataset_id") != dataset_id:
        raise ValueError("permutation result dataset identity mismatch")
    if config_sha256 is not None and result.get("config_sha256") != config_sha256:
        raise ValueError("permutation result config SHA256 mismatch")
    if result.get("statistic_rule") != STATISTIC_RULE:
        raise ValueError("permutation result statistic_rule mismatch")
    result_methods = result.get("methods")
    if expected_methods is not None and set(result_methods or ()) != expected_methods:
        raise ValueError("observed/permutation method sets mismatch")
    return records


def _stored_indices(raw_indices: Any, context: str) -> list[int]:
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError(f"{context} indices must be a non-empty list")
    if any(type(index) is not int for index in raw_indices):
        raise ValueError(f"{context} indices must contain native ints")
    indices = list(raw_indices)
    if indices != sorted(set(indices)):
        raise ValueError(f"{context} indices must be sorted and unique")
    if indices[0] < 0:
        raise ValueError(f"{context} indices must be non-negative")
    return indices


def _checkpoint_method_set(state: Mapping[str, Any], has_blocks: bool) -> set[str]:
    observed = state.get("observed")
    if observed is None:
        if has_blocks:
            raise ValueError("permutation blocks require an observed result")
        return set()
    if not isinstance(observed, Mapping):
        raise ValueError("checkpoint observed result must be an object")
    dataset_id = state.get("dataset_id")
    analysis_hash = state.get("analysis_config_sha256")
    if observed.get("dataset_id") != dataset_id:
        raise ValueError("observed/checkpoint dataset identity mismatch")
    if observed.get("config_sha256") != analysis_hash:
        raise ValueError("observed/checkpoint config SHA256 mismatch")
    if state.get("evaluator_config_sha256") != analysis_hash:
        raise ValueError("evaluator/analysis config SHA256 mismatch")
    if observed.get("statistic_rule") != STATISTIC_RULE:
        raise ValueError("observed statistic_rule mismatch")
    methods = observed.get("methods")
    if not isinstance(methods, Mapping) or (has_blocks and not methods):
        raise ValueError("observed methods must be a non-empty object")
    return {str(method) for method in methods}


def _validate_checkpoint_permutation_payloads(state: Mapping[str, Any]) -> None:
    """Validate the full block envelope and its checkpoint identity chain."""
    blocks = state.get("permutation_blocks", [])
    if not isinstance(blocks, list):
        raise ValueError("checkpoint permutation_blocks must be a list")
    target = state.get("target_permutations")
    if type(target) is not int or target < 0:
        raise ValueError("checkpoint permutation target must be a non-negative int")
    dataset_id = state.get("dataset_id")
    analysis_hash = state.get("analysis_config_sha256")
    if not isinstance(dataset_id, str) or not isinstance(analysis_hash, str):
        raise ValueError("checkpoint dataset/config identity is malformed")
    expected_methods = _checkpoint_method_set(state, bool(blocks))
    previous_last = -1
    for position, block in enumerate(blocks):
        context = f"permutation block {position}"
        if not isinstance(block, Mapping):
            raise ValueError(f"{context} must be an object")
        indices = _stored_indices(block.get("indices"), context)
        if indices[0] <= previous_last:
            raise ValueError(f"{context} block order/overlap is invalid")
        if indices[-1] >= target:
            raise ValueError(f"{context} index exceeds target permutations")
        if (
            block.get("start") != indices[0]
            or block.get("stop") != indices[-1] + 1
        ):
            raise ValueError(f"{context} start/stop do not match indices")
        if (
            block.get("permutation_inference_audit_scope")
            != PERMUTATION_INFERENCE_AUDIT_SCOPE
        ):
            raise ValueError(f"{context} permutation audit scope mismatch")
        result = block.get("result")
        if not isinstance(result, Mapping):
            raise ValueError(f"{context} result must be an object")
        if block.get("result_sha256") != sha256_json(result):
            raise ValueError(f"{context} result SHA256 mismatch")
        normalized = _validate_permutation_result(
            result,
            indices,
            dataset_id=dataset_id,
            config_sha256=analysis_hash,
            expected_methods=expected_methods,
        )
        if block.get("permutations") != normalized:
            raise ValueError(f"{context} normalized audit records mismatch")
        if block.get("permutation_sha256") != sha256_json(normalized):
            raise ValueError(f"{context} permutation SHA256 mismatch")
        previous_last = indices[-1]


def merge_permutation_block(
    checkpoint: Mapping[str, Any],
    indices: Sequence[int],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge scalar null estimands without retaining prediction vectors."""
    _validate_checkpoint_permutation_payloads(checkpoint)
    normalized_indices = [int(index) for index in indices]
    if (
        not normalized_indices
        or normalized_indices != sorted(set(normalized_indices))
        or normalized_indices[0] < 0
    ):
        raise ValueError("permutation block indices must be sorted and unique")
    records = _validate_permutation_result(result, normalized_indices)
    result_copy = copy.deepcopy(dict(result))
    result_hash = sha256_json(result_copy)
    permutation_hash = sha256_json(records)
    existing_blocks = list(checkpoint.get("permutation_blocks", []))
    requested_set = set(normalized_indices)
    for existing in existing_blocks:
        existing_indices = set(map(int, existing["indices"]))
        if existing_indices == requested_set:
            if (
                existing.get("permutation_sha256") == permutation_hash
                and existing.get("result_sha256") == result_hash
            ):
                return dict(checkpoint)
            raise CheckpointMismatchError(
                "overlapping permutation block has inconsistent hashes"
            )
        if existing_indices.intersection(requested_set):
            raise CheckpointMismatchError(
                "overlap between permutation blocks is not mergeable"
            )
    updated = _with_evaluator_config(copy.deepcopy(dict(checkpoint)), result_copy)
    updated.setdefault("permutation_blocks", []).append(
        {
            "start": normalized_indices[0],
            "stop": normalized_indices[-1] + 1,
            "indices": normalized_indices,
            "permutation_sha256": permutation_hash,
            "result_sha256": result_hash,
            "permutations": records,
            "result": result_copy,
            "permutation_inference_audit_scope": (
                PERMUTATION_INFERENCE_AUDIT_SCOPE
            ),
            "completed_at": utc_now(),
        }
    )
    updated["permutation_blocks"].sort(key=lambda block: block["indices"][0])
    updated["updated_at"] = utc_now()
    _validate_checkpoint_permutation_payloads(updated)
    return updated


def extend_target(
    checkpoint: Mapping[str, Any],
    target_permutations: int,
    target_config_sha256: str,
) -> dict[str, Any]:
    old_target = int(checkpoint["target_permutations"])
    new_target = int(target_permutations)
    if new_target < old_target:
        raise CheckpointMismatchError(
            "cannot resume a checkpoint with a smaller permutation target"
        )
    if new_target == old_target:
        if checkpoint.get("target_config_sha256") != target_config_sha256:
            raise CheckpointMismatchError("target config SHA256 mismatch")
        return dict(checkpoint)
    updated = copy.deepcopy(dict(checkpoint))
    updated["target_permutations"] = new_target
    updated["target_config_sha256"] = target_config_sha256
    updated.setdefault("target_config_history", []).append(
        {
            "target_permutations": new_target,
            "target_config_sha256": target_config_sha256,
            "recorded_at": utc_now(),
        }
    )
    updated["updated_at"] = utc_now()
    return updated


def completed_permutation_indices(
    checkpoint: Mapping[str, Any],
) -> list[int]:
    indices = [
        int(index)
        for block in checkpoint.get("permutation_blocks", [])
        for index in block.get("indices", [])
    ]
    if len(indices) != len(set(indices)):
        raise CheckpointMismatchError("checkpoint contains overlapping blocks")
    return sorted(indices)


def missing_permutation_blocks(
    completed: Iterable[int],
    target_permutations: int,
    block_size: int,
) -> list[list[int]]:
    """Return missing indices, retaining canonical block boundaries."""
    if target_permutations < 0 or block_size < 1:
        raise ValueError("invalid permutation target or block size")
    completed_set = {int(index) for index in completed}
    blocks: list[list[int]] = []
    for start in range(0, target_permutations, block_size):
        stop = min(start + block_size, target_permutations)
        missing = [index for index in range(start, stop) if index not in completed_set]
        if not missing:
            continue
        contiguous: list[int] = []
        for index in missing:
            if contiguous and index != contiguous[-1] + 1:
                blocks.append(contiguous)
                contiguous = []
            contiguous.append(index)
        if contiguous:
            blocks.append(contiguous)
    return blocks


def checkpoint_complete(state: Mapping[str, Any]) -> bool:
    target = int(state["target_permutations"])
    return state.get("observed") is not None and completed_permutation_indices(
        state
    ) == list(range(target))


def summary_from_state(
    state: Mapping[str, Any], *, resumed_without_load: bool = False
) -> dict[str, Any]:
    observed = state.get("observed") or {}
    permutation_records = [
        record
        for block in state.get("permutation_blocks", [])
        for record in block.get("permutations", [])
    ]
    return {
        "dataset_id": state["dataset_id"],
        "status": state["status"],
        "completed_permutations": len(completed_permutation_indices(state)),
        "target_permutations": int(state["target_permutations"]),
        "observed_split_plan_sha256": observed.get("split_plan_sha256"),
        "permutation_plan_sha256": state["permutation_plan_sha256"],
        "data_fingerprint_sha256": (
            (state.get("data_fingerprint") or {}).get("sha256")
        ),
        "permutation_records_sha256": sha256_json(permutation_records),
        "permutation_inference_audit_scope": (
            PERMUTATION_INFERENCE_AUDIT_SCOPE
        ),
        "permutation_block_sha256": [
            block["permutation_sha256"] for block in state.get("permutation_blocks", [])
        ],
        "checkpoint": f"checkpoints/{state['dataset_id']}.json.gz",
        "resumed_without_load": resumed_without_load,
        "errors": copy.deepcopy(state.get("errors", [])),
    }


def record_error(
    state: Mapping[str, Any],
    *,
    status: str,
    phase: str,
    error: BaseException | str,
) -> dict[str, Any]:
    updated = copy.deepcopy(dict(state))
    message = str(error)
    error_type = type(error).__name__ if isinstance(error, BaseException) else status
    updated["status"] = status
    updated.setdefault("errors", []).append(
        {
            "phase": phase,
            "type": error_type,
            "message": message[:2_000],
            "traceback": (
                "".join(traceback.format_exception(error))[-8_000:]
                if isinstance(error, BaseException)
                else ""
            ),
            "recorded_at": utc_now(),
        }
    )
    updated["updated_at"] = utc_now()
    return updated


__all__ = [
    "CheckpointMismatchError",
    "CheckpointStore",
    "METHOD_ESTIMAND_FIELDS",
    "PERMUTATION_INFERENCE_AUDIT_SCOPE",
    "SCHEMA_VERSION",
    "atomic_write_gzip_json",
    "atomic_write_json",
    "canonical_json_bytes",
    "checkpoint_complete",
    "completed_permutation_indices",
    "extend_target",
    "merge_observed",
    "merge_permutation_block",
    "missing_permutation_blocks",
    "new_checkpoint",
    "read_gzip_json",
    "record_error",
    "sha256_file",
    "sha256_json",
    "summary_from_state",
    "utc_now",
]
