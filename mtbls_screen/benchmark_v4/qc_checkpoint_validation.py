"""Checkpoint, permutation-design, and failure contracts for independent QC."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from benchmark_v4.aggregation_schema import OBSERVED_METHOD_METRICS
from benchmark_v4.checkpoints import sha256_file, sha256_json
from benchmark_v4.configuration import STATISTIC_RULE
from benchmark_v4.methods import METHODS
from benchmark_v4.qc_contracts import (
    _boolean,
    _fail,
    _hash,
    _integer,
    _number,
    _rows_for,
)


FORMAL_PERMUTATIONS = 199
METRICS = OBSERVED_METHOD_METRICS
OBSERVED_METHOD_FIELDS = frozenset(METRICS) | {
    "pre_split_folds",
    "fold_contained_folds",
    "feature_stability",
}
DESIGN_FAILURE = re.compile(r"^(insufficient_groups|group_split_failed):")
PERMUTATION_FAILURE_PHASE = re.compile(r"^permutations:(\d+)-(\d+)$")
PREPROCESSING_COMPLETENESS_FAILURE = (
    "preprocessing removed every feature by completeness"
)


@dataclass(frozen=True)
class _DerivedNullDesign:
    protocol: str
    unit: str
    primary: bool
    labels: tuple[int, ...]
    group_codes: tuple[int, ...] | None


def _observed_oof_labels(checkpoint: Mapping[str, Any]) -> tuple[int, ...]:
    dataset, state = checkpoint["dataset"], checkpoint["state"]
    observed = state.get("observed")
    if not isinstance(observed, Mapping):
        raise _fail(dataset, "completed permutations lack observed OOF labels")
    if observed.get("statistic_rule") != STATISTIC_RULE:
        raise _fail(dataset, "observed statistic_rule is missing or not current")
    methods = observed.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(METHODS):
        raise _fail(dataset, "observed OOF labels lack the exact five methods")
    fingerprint = checkpoint.get("data_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise _fail(dataset, "observed OOF labels lack a data fingerprint")
    shape = fingerprint.get("x_shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise _fail(dataset, "observed OOF sample count is unavailable")
    n_samples = _integer(shape[0], f"{dataset}.raw n", minimum=1)
    reference: tuple[int, ...] | None = None
    for method in METHODS:
        result = methods.get(method)
        measured = result.get("observed") if isinstance(result, Mapping) else None
        if not isinstance(measured, Mapping):
            raise _fail(dataset, f"observed OOF labels are missing for {method}")
        if set(measured) != OBSERVED_METHOD_FIELDS:
            raise _fail(
                dataset,
                f"{method} observed estimand fields do not match the current statistic_rule",
            )
        for arm_field in ("pre_split_folds", "fold_contained_folds"):
            folds = measured.get(arm_field)
            if not isinstance(folds, list) or not folds:
                raise _fail(
                    dataset,
                    f"observed OOF labels are missing from {method}.{arm_field}",
                )
            by_repeat: dict[int, dict[int, int]] = {}
            for position, fold in enumerate(folds):
                context = f"{dataset}.{method}.{arm_field}[{position}]"
                if not isinstance(fold, Mapping):
                    raise _fail(context, "fold is not an object")
                repeat = _integer(fold.get("repeat"), f"{context}.repeat")
                indices, labels = fold.get("test_indices"), fold.get("y_true")
                if not isinstance(indices, list) or not isinstance(labels, list):
                    raise _fail(context, "test indices/labels are not auditable lists")
                if len(indices) != len(labels):
                    raise _fail(context, "test indices/labels differ in length")
                repeated = by_repeat.setdefault(repeat, {})
                for raw_index, raw_label in zip(indices, labels, strict=True):
                    index = _integer(raw_index, f"{context}.test_index")
                    label = _integer(raw_label, f"{context}.y_true")
                    if index >= n_samples or label not in (0, 1):
                        raise _fail(context, "test index or binary label is invalid")
                    if index in repeated:
                        raise _fail(
                            context,
                            "sample appears more than once within an OOF repeat",
                        )
                    repeated[index] = label
            vectors: set[tuple[int, ...]] = set()
            for repeat, labels_by_index in by_repeat.items():
                if set(labels_by_index) != set(range(n_samples)):
                    raise _fail(
                        f"{dataset}.{method}.{arm_field}.repeat[{repeat}]",
                        "OOF labels do not cover every sample",
                    )
                vectors.add(
                    tuple(labels_by_index[index] for index in range(n_samples))
                )
            if len(vectors) != 1:
                raise _fail(dataset, "observed OOF labels change across repeats")
            labels_vector = next(iter(vectors))
            if reference is None:
                reference = labels_vector
            elif labels_vector != reference:
                raise _fail(dataset, "observed OOF labels differ across methods/arms")
    assert reference is not None
    return reference


def _derive_null_design(
    checkpoint: Mapping[str, Any],
    tables: Mapping[str, list[dict[str, Any]]],
    inventory: Mapping[str, Any],
) -> _DerivedNullDesign:
    dataset, state = checkpoint["dataset"], checkpoint["state"]
    labels = _observed_oof_labels(checkpoint)
    observed = state.get("observed")
    assert isinstance(observed, Mapping)
    provenance = observed.get("data_provenance")
    if not isinstance(provenance, Mapping):
        raise _fail(dataset, "null design lacks auditable group provenance")
    available = provenance.get("groups_available")
    if not isinstance(available, bool):
        raise _fail(dataset, "groups_available is not an auditable boolean")
    n_groups = _integer(provenance.get("n_groups"), f"{dataset}.n_groups")
    fingerprint = checkpoint.get("data_fingerprint")
    assert isinstance(fingerprint, Mapping)
    groups_hash = provenance.get("groups_sha256")
    codes_hash = provenance.get("group_codes_sha256")
    if (
        groups_hash != fingerprint.get("groups_sha256")
        or codes_hash != fingerprint.get("group_codes_sha256")
    ):
        raise _fail(dataset, "group provenance differs from the data fingerprint")
    raw_codes = provenance.get("group_codes")
    codes: tuple[int, ...] | None
    if not available:
        if (
            n_groups != 0
            or raw_codes is not None
            or groups_hash is not None
            or codes_hash is not None
        ):
            raise _fail(dataset, "no-group provenance contains group evidence")
        codes = None
    else:
        if not isinstance(raw_codes, list):
            raise _fail(dataset, "grouped null lacks auditable group_codes")
        if len(raw_codes) != len(labels):
            raise _fail(dataset, "auditable group_codes do not cover every OOF label")
        if any(
            isinstance(code, bool) or not isinstance(code, int) or code < 0
            for code in raw_codes
        ):
            raise _fail(dataset, "auditable group_codes must be non-negative integers")
        codes = tuple(raw_codes)
        if n_groups < 1 or sorted(set(codes)) != list(range(n_groups)):
            raise _fail(dataset, "auditable group_codes are not canonical")
        if list(dict.fromkeys(codes)) != list(range(n_groups)):
            raise _fail(dataset, "auditable group_codes are not first-seen canonical")
        if not isinstance(codes_hash, str) or sha256_json(list(codes)) != codes_hash:
            raise _fail(dataset, "auditable group_codes SHA256 mismatch")
        _hash(groups_hash, f"{dataset}.groups SHA256")
    if (
        _boolean(inventory.get("groups_available"), f"{dataset}.inventory groups")
        != available
    ):
        raise _fail(
            dataset,
            "inventory groups_available differs from auditable provenance",
        )
    if _integer(inventory.get("n_groups"), f"{dataset}.inventory n_groups") != n_groups:
        raise _fail(dataset, "inventory n_groups differs from auditable provenance")
    if (
        (inventory.get("groups_sha256") or None) != groups_hash
        or (inventory.get("group_codes_sha256") or None) != codes_hash
    ):
        raise _fail(dataset, "inventory group hashes differ from auditable provenance")
    predictions = _rows_for(tables, "outer_fold_predictions", dataset)
    if not predictions:
        raise _fail(dataset, "null design lacks observed OOF prediction evidence")
    for position, row in enumerate(predictions):
        context = f"{dataset}.outer_fold_predictions[{position}]"
        index = _integer(row.get("test_index"), f"{context}.test_index")
        label = _integer(row.get("y_true"), f"{context}.y_true")
        if index >= len(labels) or label != labels[index]:
            raise _fail(
                dataset,
                "aggregation OOF labels differ from checkpoint OOF labels",
            )
        if (
            _boolean(row.get("groups_available"), f"{context}.groups_available")
            != available
        ):
            raise _fail(
                dataset,
                "OOF groups_available differs from auditable provenance",
            )
        raw_group = row.get("group_code")
        if codes is None:
            if raw_group is not None and str(raw_group).strip():
                raise _fail(dataset, "no-group OOF row contains a group code")
        elif _integer(raw_group, f"{context}.group_code") != codes[index]:
            raise _fail(dataset, "OOF group code differs from auditable provenance")
    if codes is None:
        return _DerivedNullDesign(
            "primary_exchangeable", "sample", True, labels, None
        )
    labels_by_group: dict[int, set[int]] = {}
    for code, label in zip(codes, labels, strict=True):
        labels_by_group.setdefault(code, set()).add(label)
    if all(len(group_labels) == 1 for group_labels in labels_by_group.values()):
        return _DerivedNullDesign(
            "primary_exchangeable", "group", True, labels, codes
        )
    return _DerivedNullDesign(
        "paired_repeated_measures_sensitivity", "within_group", False, labels, codes
    )


def _checkpoint_permutations(
    checkpoint: Mapping[str, Any],
    expected_design: _DerivedNullDesign | None = None,
    *,
    require_design: bool = False,
) -> tuple[list[int], str | None, str | None, bool]:
    state, dataset = checkpoint["state"], checkpoint["dataset"]
    completed: list[int] = []
    protocols: set[str] = set()
    units: set[str] = set()
    primary_flags: set[bool] = set()
    for position, block in enumerate(state.get("permutation_blocks", [])):
        context = f"{dataset}.permutation_blocks[{position}]"
        if not isinstance(block, dict) or not isinstance(block.get("indices"), list):
            raise _fail(context, "malformed block")
        indices = [_integer(item, f"{context}.indices") for item in block["indices"]]
        if not indices or indices != sorted(set(indices)):
            raise _fail(context, "indices are not sorted unique")
        if block.get("start") != indices[0] or block.get("stop") != indices[-1] + 1:
            raise _fail(context, "start/stop mismatch")
        if set(indices).intersection(completed) or any(
            i >= checkpoint["target"] for i in indices
        ):
            raise _fail(context, "overlapping or out-of-target permutation indices")
        result = block.get("result")
        if (
            not isinstance(result, dict)
            or sha256_json(result) != block.get("result_sha256")
        ):
            raise _fail(context, "result SHA256 mismatch")
        if (
            result.get("dataset_id") != dataset
            or result.get("config_sha256") != checkpoint["analysis_config_sha256"]
        ):
            raise _fail(context, "result dataset/config mismatch")
        if result.get("statistic_rule") != STATISTIC_RULE:
            raise _fail(context, "permutation statistic_rule is missing or not current")
        if result.get("permutation_indices") != indices:
            raise _fail(context, "result indices mismatch")
        records = result.get("permutations")
        if not isinstance(records, list) or len(records) != len(indices):
            raise _fail(context, "permutation record count mismatch")
        audit = []
        protocol, unit = (
            result.get("permutation_protocol"),
            result.get("permutation_unit"),
        )
        primary = result.get("primary_null_eligible")
        if not isinstance(primary, bool):
            raise _fail(context, "primary-null eligibility is not boolean")
        expected_counts = (
            None
            if expected_design is None
            else [expected_design.labels.count(0), expected_design.labels.count(1)]
        )
        for index, record in zip(indices, records, strict=True):
            if not isinstance(record, dict) or record.get("index") != index:
                raise _fail(context, "permutation record index mismatch")
            for key in ("label_sha256", "split_plan_sha256"):
                _hash(record.get(key), f"{context}.{key}")
            if (
                record.get("permutation_protocol") != protocol
                or record.get("permutation_unit") != unit
            ):
                raise _fail(context, "permutation protocol/unit changed within block")
            if (
                expected_counts is not None
                and record.get("original_class_counts") != expected_counts
            ):
                raise _fail(
                    context,
                    "original class counts differ from observed OOF labels",
                )
            audit.append(
                {
                    key: record[key]
                    for key in ("index", "label_sha256", "split_plan_sha256")
                }
            )
        if (
            block.get("permutations") != audit
            or sha256_json(audit) != block.get("permutation_sha256")
        ):
            raise _fail(context, "permutation audit SHA256 mismatch")
        methods = result.get("methods")
        if not isinstance(methods, dict) or set(methods) != set(METHODS):
            raise _fail(context, "permutation result lacks the exact five methods")
        for method in METHODS:
            if not isinstance(methods[method], dict):
                raise _fail(context, f"malformed method {method}")
            if set(methods[method]) != set(METRICS):
                raise _fail(
                    context,
                    "permutation estimand fields do not match the current "
                    f"statistic_rule for {method}",
                )
            if any(
                not isinstance(methods[method].get(metric), list)
                or len(methods[method][metric]) != len(indices)
                for metric in METRICS
            ):
                raise _fail(
                    context,
                    f"permutation metric length mismatch for {method}",
                )
            for metric in METRICS:
                for value in methods[method][metric]:
                    if value is None:
                        raise _fail(
                            context,
                            f"non-finite permutation value for {method}.{metric}",
                        )
                    _number(value, f"{context}.{method}.{metric}")
        completed.extend(indices)
        protocols.add(str(protocol))
        units.add(str(unit))
        primary_flags.add(primary)
    if len(protocols) > 1 or len(units) > 1 or len(primary_flags) > 1:
        raise _fail(dataset, "primary and paired-sensitivity nulls were pooled")
    protocol = next(iter(protocols), None)
    unit = next(iter(units), None)
    primary = next(iter(primary_flags), False)
    valid = (protocol, unit, primary) in {
        ("primary_exchangeable", "sample", True),
        ("primary_exchangeable", "group", True),
        ("paired_repeated_measures_sensitivity", "within_group", False),
    }
    if completed and not valid:
        raise _fail(dataset, "unsupported permutation protocol/unit/scope")
    declared = (protocol, unit, primary)
    if completed and require_design and expected_design is None:
        raise _fail(
            dataset,
            "completed permutations lack evidence for an independent null-design audit",
        )
    if completed and expected_design is not None and declared != (
        expected_design.protocol,
        expected_design.unit,
        expected_design.primary,
    ):
        raise _fail(
            dataset,
            "declared null differs from independently derived permutation design",
        )
    return sorted(completed), protocol, unit, primary


def _validate_inventory(
    checkpoint: Mapping[str, Any],
    row: Mapping[str, Any],
    completed: Sequence[int],
    protocol: str | None,
    primary: bool,
) -> None:
    dataset, state, path = (
        checkpoint["dataset"],
        checkpoint["state"],
        checkpoint["path"],
    )
    if Path(str(row["checkpoint_path"])).resolve() != Path(path).resolve():
        raise _fail(dataset, "inventory checkpoint path mismatch")
    if (
        _hash(row["checkpoint_sha256"], f"{dataset}.checkpoint SHA256")
        != sha256_file(path)
    ):
        raise _fail(dataset, "checkpoint SHA256 mismatch")
    if (
        _integer(row["checkpoint_size_bytes"], f"{dataset}.checkpoint size")
        != path.stat().st_size
    ):
        raise _fail(dataset, "checkpoint size mismatch")
    expected = {
        "status": state.get("status"),
        "schema_version": checkpoint["schema_version"],
        "target_permutations": checkpoint["target"],
        "completed_permutations": len(completed),
        "analysis_config_sha256": checkpoint["analysis_config_sha256"],
        "target_config_sha256": checkpoint["target_config_sha256"],
        "code_sha256": checkpoint["code_sha256"],
        "environment_sha256": checkpoint["environment_sha256"],
        "permutation_plan_sha256": checkpoint["permutation_plan_sha256"],
    }
    for key, value in expected.items():
        actual: Any = row.get(key)
        if isinstance(value, int):
            actual = _integer(actual, f"{dataset}.{key}")
        if actual != value:
            raise _fail(dataset, f"inventory/checkpoint {key} mismatch")
    observed = state.get("observed") is not None
    complete = observed and list(completed) == list(range(checkpoint["target"]))
    if _boolean(row["observed_available"], f"{dataset}.observed") != observed:
        raise _fail(dataset, "inventory observed flag mismatch")
    if _boolean(row["target_complete"], f"{dataset}.target complete") != complete:
        raise _fail(dataset, "inventory target-complete flag mismatch")
    formal = complete and primary and checkpoint["target"] >= FORMAL_PERMUTATIONS
    if _boolean(row["formal_permutation_eligible"], f"{dataset}.formal") != formal:
        raise _fail(dataset, "inventory formal eligibility mismatch")
    observed_map = state.get("observed") or {}
    fingerprint = state.get("data_fingerprint") or {}
    optional = {
        "evaluator_config_sha256": state.get("evaluator_config_sha256"),
        "observed_split_plan_sha256": observed_map.get("split_plan_sha256"),
        "data_fingerprint_sha256": fingerprint.get("sha256"),
    }
    for key, value in optional.items():
        actual = row.get(key) or None
        if actual != value:
            raise _fail(dataset, f"inventory/checkpoint {key} mismatch")
    if completed and protocol is None:
        raise _fail(dataset, "completed permutations lack a null protocol")


def _validate_failures(
    checkpoint: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    completed: Sequence[int] | None = None,
) -> tuple[str, str, bool] | None:
    dataset, state = checkpoint["dataset"], checkpoint["state"]
    errors = state.get("errors")
    if not isinstance(errors, list):
        raise _fail(dataset, "checkpoint errors is not a list")
    if state.get("status") == "ok":
        if errors or rows:
            raise _fail(dataset, "successful checkpoint contains failure records")
        return None
    if not rows:
        raise _fail(dataset, "failed checkpoint is absent from failures table")
    if errors and len(rows) != len(errors):
        raise _fail(dataset, "checkpoint/failure-table error count mismatch")
    for error, row in zip(errors, rows, strict=False):
        for state_key, table_key in (
            ("phase", "phase"),
            ("type", "error_type"),
            ("message", "message"),
        ):
            if row.get(table_key) != error.get(state_key):
                raise _fail(dataset, f"failure table differs at {table_key}")
    matches = [
        DESIGN_FAILURE.fullmatch(str(item.get("message", "")))
        or DESIGN_FAILURE.match(str(item.get("message", "")))
        for item in errors
    ]
    if (
        state.get("observed") is None
        and state.get("status") == "evaluation_error"
        and errors
        and all(item.get("phase") == "observed" for item in errors)
        and all(matches)
    ):
        assert matches[0] is not None
        return "design_ineligible_group_cv", matches[0].group(1), False
    if (
        state.get("observed") is None
        and state.get("status") == "evaluation_error"
        and errors
        and all(item.get("phase") == "observed" for item in errors)
        and all(
            item.get("message") == PREPROCESSING_COMPLETENESS_FAILURE
            for item in errors
        )
    ):
        return (
            "design_ineligible_preprocessing",
            "no_feature_meets_completeness",
            False,
        )
    permutation_phases = [
        PERMUTATION_FAILURE_PHASE.fullmatch(str(item.get("phase", "")))
        for item in errors
    ]
    raw_target = checkpoint.get("target", state.get("target_permutations"))
    target = (
        raw_target
        if isinstance(raw_target, int) and not isinstance(raw_target, bool)
        else 0
    )
    completed_indices = set(completed or ())
    latest_phase = permutation_phases[-1] if permutation_phases else None
    valid_permutation_phases = (
        target > 0
        and bool(permutation_phases)
        and all(
            match is not None
            and 0 <= int(match.group(1)) <= int(match.group(2)) < target
            for match in permutation_phases
        )
        and latest_phase is not None
        and completed_indices.isdisjoint(
            range(
                int(latest_phase.group(1)),
                int(latest_phase.group(2)) + 1,
            )
        )
    )
    if (
        state.get("observed") is not None
        and state.get("status") == "evaluation_error"
        and errors
        and valid_permutation_phases
        and all(
            item.get("message") == PREPROCESSING_COMPLETENESS_FAILURE
            for item in errors
        )
    ):
        return (
            "null_ineligible_preprocessing",
            "permutation_no_feature_meets_completeness",
            False,
        )
    combined = " ".join(
        f"{item.get('type', '')} {item.get('message', '')}" for item in errors
    ).lower()
    phases = {str(item.get("phase", "")) for item in errors}
    if "converg" in combined:
        reason = "convergence_error"
    elif state.get("status") == "load_error" or "load" in phases:
        reason = "load_error"
    elif not errors:
        reason = "missing_structured_error"
    else:
        reason = f"unknown_{state.get('status', 'error')}"
    return "blocking_failure", reason, True


__all__: list[str] = []
