"""Strict input validation and eligibility decisions for final-run QC."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import json
import math
from pathlib import Path
from typing import Any

from nm_v4.aggregate_results import TABLE_COLUMNS
from nm_v4.aggregation_schema import (
    AggregationError,
    load_checkpoints,
)
from nm_v4.checkpoints import sha256_file, sha256_json, utc_now
from nm_v4.configuration import STATISTIC_RULE
from nm_v4.methods import METHODS
from nm_v4.qc_checkpoint_validation import (
    DESIGN_FAILURE,  # noqa: F401 - compatibility re-export
    FORMAL_PERMUTATIONS,
    METRICS,
    OBSERVED_METHOD_FIELDS,  # noqa: F401 - compatibility re-export
    _DerivedNullDesign,  # noqa: F401 - compatibility re-export
    _checkpoint_permutations,
    _derive_null_design,
    _observed_oof_labels,  # noqa: F401 - compatibility re-export
    _validate_failures,
    _validate_inventory,
)
from nm_v4.qc_contracts import (
    HASH,  # noqa: F401 - compatibility re-export
    QCError,
    _blank,
    _boolean,
    _fail,
    _hash,  # noqa: F401 - compatibility re-export
    _integer,
    _json_cell,
    _number,
    _rows_for,
)
from nm_v4.qc_nulls import checkpoint_null_summary, phipson_smyth_plus_one
from nm_v4.runner_support import CANONICAL_DATASET_COUNT, load_whitelist
from nm_v4.stability import feature_stability_summary


MODES = ("formal", "pilot", "observed-only")
METRIC_ARMS = {
    "pre_split_mean_fold_auc": "pre_split",
    "fold_contained_mean_fold_auc": "fold_contained",
    "gap_mean_fold": "pre_split_minus_fold_contained_mean_fold",
    "pre_split_oof_auc": "pre_split",
    "fold_contained_oof_auc": "fold_contained",
    "gap_oof": "pre_split_minus_fold_contained_oof",
}
PERMUTATION_INFERENCE_AUDIT = {
    "scope": (
        "independent_recalculation_from_hash_bound_permutation_scalar_statistics"
    ),
    "limitation": (
        "does_not_recompute_permutation_model_predictions_or_"
        "scalar_statistics_from_predictions"
    ),
}
METHOD_ARMS = frozenset([(method, arm) for method in METHODS
                         for arm in ("pre_split", "fold_contained")] + [("baseline_l1", "fold_contained")])
KNOWN_UNCERTAINTY_REASONS = frozenset({"ok", "fewer_than_two_independent_samples_per_class",
                                       "fewer_than_two_participant_clusters_per_class",
                                       "fewer_than_four_participant_clusters",
                                       "valid_bootstrap_fraction_below_0.90"})
SUCCESS_TABLES = tuple(name for name in TABLE_COLUMNS if name not in {"failures", "run_inventory"})
_CSV_FIELD_CHARACTER_LIMIT = 1_048_576

FLOW_COLUMNS = (
    "dataset", "dataset_id", "source", "selected_whitelist", "checkpoint_present", "attempted",
    "runner_status", "observed_complete", "five_method_two_arm_complete", "shared_fold_repeat_complete",
    "oof_uncertainty_complete", "oof_uncertainty_estimable_methods", "feature_stability_complete",
    "permutation_complete", "completed_permutations", "target_permutations", "null_protocol",
    "p_value_status", "primary_null_eligible", "sensitivity_null_eligible", "formal_inference_eligible",
    "analysis_eligible", "classification", "reason_code", "blocking")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise _fail(str(path), "required regular file is missing")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise _fail(str(path), "expected a JSON object")
    return value


def _table_path(table_dir: Path, name: str) -> Path:
    candidates = [path for suffix in ("csv", "parquet")
                  if (path := table_dir / f"{name}.{suffix}").exists()]
    if len(candidates) != 1:
        raise _fail(str(table_dir), f"requires exactly one {name}.csv or .parquet")
    return candidates[0]


def _read_table(path: Path, name: str) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        previous_limit = csv.field_size_limit(_CSV_FIELD_CHARACTER_LIMIT)
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != TABLE_COLUMNS[name]:
                    raise _fail(
                        str(path), "columns/order do not match aggregation schema"
                    )
                return [dict(row) for row in reader]
        except csv.Error as exc:
            raise _fail(
                str(path),
                "CSV field exceeds the explicit 1,048,576-character limit "
                f"or the CSV is malformed: {exc}",
            ) from exc
        finally:
            csv.field_size_limit(previous_limit)
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - only for optional Parquet input
        raise _fail(str(path), "Parquet input requires pandas/pyarrow") from exc
    frame = pd.read_parquet(path)
    if tuple(frame.columns) != TABLE_COLUMNS[name]:
        raise _fail(str(path), "columns/order do not match aggregation schema")
    return [{key: ("" if pd.isna(value) else value) for key, value in row.items()}
            for row in frame.to_dict(orient="records")]


def _read_tables(table_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Path]]:
    if not table_dir.is_dir():
        raise _fail(str(table_dir), "aggregation table directory is missing")
    paths = {name: _table_path(table_dir, name) for name in TABLE_COLUMNS}
    tables = {name: _read_table(path, name) for name, path in paths.items()}
    return tables, paths


def _whitelist_ids(manifest: Mapping[str, Any], expected_count: int) -> list[str]:
    if manifest.get("canonical_whitelist_count") != expected_count:
        label = "1,102" if expected_count == CANONICAL_DATASET_COUNT else str(expected_count)
        raise _fail("manifest", f"canonical whitelist count must be {label}")
    path = Path(str(manifest.get("whitelist_path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != manifest.get("whitelist_sha256"):
        raise _fail("manifest", "whitelist path/SHA256 mismatch")
    if expected_count == CANONICAL_DATASET_COUNT:
        return [record.dataset_id for record in load_whitelist(path)]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "id" not in (reader.fieldnames or ()):
            raise _fail(str(path), "test whitelist lacks id column")
        ids = [str(row["id"]).strip() for row in reader]
    if len(ids) != expected_count or len(set(ids)) != len(ids) or any(not item for item in ids):
        raise _fail(str(path), "whitelist count/uniqueness mismatch")
    return ids


def _validate_manifest(run_dir: Path, mode: str, expected_count: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    manifest = _read_json(run_dir / "manifest.json")
    whitelist_ids = _whitelist_ids(manifest, expected_count)
    selected = manifest.get("selected_dataset_ids")
    if not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
        raise _fail("manifest", "selected_dataset_ids must be a string list")
    if len(selected) != len(set(selected)) or manifest.get("selected_dataset_count") != len(selected):
        raise _fail("manifest", "selected dataset count/uniqueness mismatch")
    if not set(selected).issubset(whitelist_ids):
        raise _fail("manifest", "selected IDs fall outside the locked whitelist")
    protocol = manifest.get("protocol_config")
    if not isinstance(protocol, dict):
        raise _fail("manifest", "protocol_config is missing")
    if protocol.get("statistic_rule") != STATISTIC_RULE:
        raise _fail(
            "manifest.protocol_config",
            "does not use the current statistic_rule",
        )
    target = _integer(protocol.get("n_permutations"), "manifest target")
    if mode == "formal":
        if expected_count != CANONICAL_DATASET_COUNT and len(selected) != expected_count:
            raise _fail("manifest", "formal selection does not cover the test whitelist")
        if expected_count == CANONICAL_DATASET_COUNT and selected != whitelist_ids:
            raise _fail("manifest", "formal selected whitelist must contain all 1,102 IDs")
        if target != FORMAL_PERMUTATIONS or manifest.get("pilot") is not False:
            raise _fail("manifest", "formal mode requires B=199 and pilot=false")
    elif mode == "pilot":
        if manifest.get("pilot") is not True or not 0 < target < FORMAL_PERMUTATIONS:
            raise _fail("manifest", "pilot mode requires pilot=true and 0 < B < 199")
    elif target != 0:
        raise _fail("manifest", "observed-only mode requires B=0")
    methods = protocol.get("methods")
    if methods is not None and tuple(methods) != METHODS:
        raise _fail("manifest", "protocol does not contain the exact five methods")
    hashes = {
        "target_config_sha256": sha256_json(protocol),
        "analysis_config_sha256": sha256_json(
            {key: value for key, value in protocol.items() if key != "n_permutations"}
        ),
        "code_sha256": sha256_json(manifest.get("code_file_sha256")),
        "environment_sha256": sha256_json(manifest.get("package_versions")),
    }
    for key, expected in hashes.items():
        if manifest.get(key) != expected:
            raise _fail("manifest", f"{key} does not match its hashed payload")
    try:
        checkpoints = load_checkpoints(run_dir)
    except (AggregationError, FileNotFoundError) as exc:
        raise _fail("checkpoints", str(exc)) from exc
    by_id = {checkpoint["dataset"]: checkpoint for checkpoint in checkpoints}
    if set(by_id) != set(selected) or len(by_id) != len(selected):
        raise _fail("manifest", "selected IDs and checkpoint files differ")
    summaries = manifest.get("datasets")
    if not isinstance(summaries, dict) or set(summaries) != set(selected):
        raise _fail("manifest", "runner dataset summaries do not cover selected IDs")
    for dataset in selected:
        checkpoint, summary = by_id[dataset], summaries[dataset]
        if not isinstance(summary, dict):
            raise _fail(dataset, "runner summary is not an object")
        state = checkpoint["state"]
        for key in (
            "analysis_config_sha256", "target_config_sha256", "code_sha256",
            "environment_sha256",
        ):
            if checkpoint[key] != manifest[key]:
                raise _fail(dataset, f"checkpoint/manifest {key} mismatch")
        completed = _checkpoint_permutations(checkpoint)[0]
        comparisons = {
            "status": state.get("status"),
            "target_permutations": checkpoint["target"],
            "completed_permutations": len(completed),
            "permutation_plan_sha256": checkpoint["permutation_plan_sha256"],
        }
        for key, expected in comparisons.items():
            if summary.get(key) != expected:
                raise _fail(dataset, f"runner summary/checkpoint {key} mismatch")
        relative = summary.get("checkpoint")
        if not isinstance(relative, str) or (run_dir / relative).resolve() != Path(checkpoint["path"]).resolve():
            raise _fail(dataset, "runner summary checkpoint path mismatch")
    all_ok = all(checkpoint["state"].get("status") == "ok" for checkpoint in checkpoints)
    expected_manifest_status = "complete" if all_ok else "running_or_incomplete"
    if manifest.get("status") != expected_manifest_status:
        raise _fail("manifest", "run status disagrees with checkpoint statuses")
    return manifest, [by_id[item] for item in selected], selected


def _validate_table_identities(tables: Mapping[str, list[dict[str, Any]]], selected: Sequence[str]) -> None:
    allowed = set(selected)
    sources: dict[str, str] = {}
    for name, rows in tables.items():
        for position, row in enumerate(rows):
            dataset = row.get("dataset")
            if dataset not in allowed or row.get("dataset_id") != dataset:
                raise _fail(f"{name}[{position}]", "dataset identity is absent or unselected")
            source = row.get("source")
            if not isinstance(source, str) or not source:
                raise _fail(f"{name}[{position}]", "source is empty")
            if sources.setdefault(dataset, source) != source:
                raise _fail(dataset, "source differs across aggregation tables")


def _one_inventory(tables: Mapping[str, list[dict[str, Any]]], dataset: str) -> dict[str, Any]:
    rows = _rows_for(tables, "run_inventory", dataset)
    if len(rows) != 1:
        raise _fail(dataset, "run inventory must contain exactly one row")
    return rows[0]


def _validate_summary_and_folds(dataset: str, tables: Mapping[str, list[dict[str, Any]]], manifest: Mapping[str, Any], checkpoint: Mapping[str, Any], inventory: Mapping[str, Any]) -> int:
    summaries = _rows_for(tables, "dataset_method_summary", dataset)
    pairs = [(row["method"], row["arm"]) for row in summaries]
    if len(pairs) != len(METHOD_ARMS) or set(pairs) != METHOD_ARMS:
        raise _fail(dataset, "successful dataset lacks five methods x two arms")
    state_methods = (checkpoint["state"].get("observed") or {}).get("methods")
    if not isinstance(state_methods, dict) or set(state_methods) != set(METHODS):
        raise _fail(dataset, "checkpoint observed result lacks the exact five methods")
    for row in summaries:
        for key in ("analysis_config_sha256", "code_sha256", "environment_sha256"):
            if row[key] != manifest[key]:
                raise _fail(dataset, f"summary {key} mismatch")
        if _integer(row["n_samples"], f"{dataset}.n") != _integer(inventory["n_samples"], f"{dataset}.inventory n"):
            raise _fail(dataset, "summary/inventory n mismatch")
    diagnostics = _rows_for(tables, "outer_fold_diagnostics", dataset)
    predictions = _rows_for(tables, "outer_fold_predictions", dataset)
    diag_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    pred_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in diagnostics:
        diag_by_pair.setdefault((row["method"], row["arm"]), []).append(row)
    for row in predictions:
        pred_by_pair.setdefault((row["method"], row["arm"]), []).append(row)
    if set(diag_by_pair) != METHOD_ARMS or set(pred_by_pair) != METHOD_ARMS:
        raise _fail(dataset, "fold tables lack the required method/arm pairs")
    reference_keys: set[tuple[int, int, int]] | None = None
    reference_predictions: set[tuple[int, int, int, int, int]] | None = None
    for pair in sorted(METHOD_ARMS):
        keys = {
            (_integer(row["outer_index"], "outer"), _integer(row["repeat"], "repeat"), _integer(row["fold"], "fold"))
            for row in diag_by_pair[pair]
        }
        if len(keys) != len(diag_by_pair[pair]):
            raise _fail(dataset, "duplicate fold diagnostics")
        signatures = {
            (
                _integer(row["outer_index"], "outer"),
                _integer(row["repeat"], "repeat"),
                _integer(row["fold"], "fold"),
                _integer(row["test_index"], "test index"),
                _integer(row["y_true"], "label"),
            )
            for row in pred_by_pair[pair]
        }
        if len(signatures) != len(pred_by_pair[pair]):
            raise _fail(dataset, "duplicate OOF prediction")
        if reference_keys is None:
            reference_keys, reference_predictions = keys, signatures
        elif keys != reference_keys or signatures != reference_predictions:
            raise _fail(dataset, "shared fold/repeat membership differs across methods/arms")
    assert reference_keys is not None and reference_predictions is not None
    repeats = _integer(manifest["protocol_config"]["outer_repeats"], "outer repeats", minimum=1)
    effective_values = {_integer(row["effective_outer_splits"], "effective folds", minimum=2) for row in summaries}
    if len(effective_values) != 1:
        raise _fail(dataset, "effective outer splits differ across methods/arms")
    effective = next(iter(effective_values))
    expected_keys = {
        (repeat * effective + fold, repeat, fold)
        for repeat in range(repeats) for fold in range(effective)
    }
    if reference_keys != expected_keys:
        raise _fail(dataset, "shared fold/repeat coverage is incomplete")
    if any(_integer(row["n_outer_folds"], "n outer folds") != len(expected_keys) for row in summaries):
        raise _fail(dataset, "summary outer-fold count mismatch")
    n_samples = _integer(inventory["n_samples"], f"{dataset}.n", minimum=1)
    for repeat in range(repeats):
        indices = [item[3] for item in reference_predictions if item[1] == repeat]
        if sorted(indices) != list(range(n_samples)):
            raise _fail(dataset, "OOF predictions do not cover every sample index per repeat")
    return len(expected_keys)


def _validate_uncertainty(dataset: str, tables: Mapping[str, list[dict[str, Any]]], manifest: Mapping[str, Any], inventory: Mapping[str, Any]) -> int:
    rows = _rows_for(tables, "dataset_method_uncertainty", dataset)
    if len(rows) != len(METHODS) or {row["method"] for row in rows} != set(METHODS):
        raise _fail(dataset, "OOF uncertainty is incomplete for the five methods")
    estimable = 0
    requested_values: set[int] = set()
    units: set[str] = set()
    seeds: set[int] = set()
    groups = _boolean(inventory["groups_available"], "inventory groups_available")
    expected_units = {"class_stratified_participant_cluster", "participant_cluster"} if groups else {"class_stratified_sample"}
    expected_n_units = _integer(inventory["n_groups" if groups else "n_samples"], "inventory uncertainty units", minimum=1)
    for row in rows:
        reason = row["uncertainty_reason"]
        if reason not in KNOWN_UNCERTAINTY_REASONS:
            raise _fail(dataset, f"unknown OOF uncertainty reason {reason!r}")
        if row["comparison"] != "pre_split_minus_fold_contained":
            raise _fail(dataset, "OOF uncertainty comparison mismatch")
        if _integer(row["n_samples"], "uncertainty n") != _integer(inventory["n_samples"], "inventory n"):
            raise _fail(dataset, "OOF uncertainty sample count mismatch")
        if _integer(row["n_repeats"], "uncertainty repeats", minimum=1) != _integer(manifest["protocol_config"]["outer_repeats"], "outer repeats", minimum=1):
            raise _fail(dataset, "OOF uncertainty repeat count mismatch")
        requested = _integer(row["requested_bootstrap_replicates"], "requested bootstraps", minimum=2)
        valid = _integer(row["valid_bootstrap_replicates"], "valid bootstraps")
        if valid > requested:
            raise _fail(dataset, "valid bootstrap count exceeds requested count")
        requested_values.add(requested)
        unit = str(row["uncertainty_unit"])
        units.add(unit)
        seeds.add(_integer(row["bootstrap_seed"], "uncertainty bootstrap seed"))
        if (_boolean(row["groups_available"], "uncertainty groups_available") != groups
                or unit not in expected_units
                or _integer(row["n_units"], "uncertainty n_units", minimum=1) != expected_n_units):
            raise _fail(dataset, "uncertainty unit/n_units/group metadata mismatch")
        for key in ("pre_split_oof_auc", "fold_contained_oof_auc", "gap_oof"):
            _number(row[key], f"{dataset}.{key}")
        if reason == "ok":
            for key in ("standard_error", "ci_low", "ci_high", "variance"):
                _number(row[key], f"{dataset}.{key}")
            if valid < math.ceil(0.9 * requested):
                raise _fail(dataset, "OOF uncertainty marked ok below 90% valid")
            estimable += 1
        elif any(str(row[key]).strip() for key in ("standard_error", "ci_low", "ci_high", "variance")):
            raise _fail(dataset, "non-estimable OOF uncertainty contains inferential values")
    if len(requested_values) != 1 or len(units) != 1 or len(seeds) != 1:
        raise _fail(dataset, "OOF uncertainty bootstrap/unit settings differ by method")
    return estimable


def _validate_stability(dataset: str, tables: Mapping[str, list[dict[str, Any]]], fold_count: int, inventory: Mapping[str, Any]) -> None:
    rows = _rows_for(tables, "feature_stability", dataset)
    if len(rows) != len(METHODS) or {row["method"] for row in rows} != set(METHODS):
        raise _fail(dataset, "feature stability is incomplete for the five methods")
    diagnostics = _rows_for(tables, "outer_fold_diagnostics", dataset)
    for row in rows:
        method = row["method"]
        if row["arm"] != "fold_contained" or _integer(row["n_sets"], "stability n sets") != fold_count:
            raise _fail(dataset, "feature stability fold count/arm mismatch")
        if _integer(row["raw_p"], "stability raw p", minimum=1) != _integer(inventory["n_features_raw"], "inventory p", minimum=1):
            raise _fail(dataset, "feature stability raw p mismatch")
        selected = _json_cell(row["selected_raw_indices_by_fold_json"], "selected features by fold")
        feature_ids = _json_cell(row["selected_feature_ids_by_fold_json"], "selected feature IDs by fold")
        raw_sensitivity = _json_cell(row["raw_universe_sensitivity_json"], "raw-universe stability")
        if not isinstance(selected, list) or not isinstance(feature_ids, list) or not isinstance(raw_sensitivity, dict):
            raise _fail(dataset, "feature stability audit payload is malformed")
        contained = sorted((item for item in diagnostics
                            if item["method"] == method and item["arm"] == "fold_contained"),
                           key=lambda item: (_integer(item["outer_index"], "outer"), _integer(item["repeat"], "repeat"), _integer(item["fold"], "fold")))
        expected = [_json_cell(item["selected_raw_indices_json"], "fold selection") for item in contained]
        expected_ids = [_json_cell(item["selected_feature_ids_json"], "fold feature IDs") for item in contained]
        raw_expected = {key: None if isinstance(value, float) and not math.isfinite(value) else value
                        for key, value in feature_stability_summary(
                            selected, n_features=_integer(row["raw_p"], "stability raw p")).items()}
        if selected != expected:
            raise _fail(dataset, "feature stability selections do not match fold diagnostics")
        if feature_ids != expected_ids:
            raise _fail(dataset, "feature IDs do not match fold diagnostics")
        if raw_sensitivity != raw_expected:
            raise _fail(dataset, "raw-universe stability does not match fold selections")


def _validate_permutation_rows(dataset: str, tables: Mapping[str, list[dict[str, Any]]], mode: str, checkpoint: Mapping[str, Any], completed: Sequence[int], protocol: str | None, unit: str | None, primary: bool) -> str:
    rows = _rows_for(tables, "permutation_summary", dataset)
    keys = [(row["method"], row["metric"]) for row in rows]
    expected_keys = {(method, metric) for method in METHODS for metric in METRICS}
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise _fail(dataset, "permutation summary is incomplete for five methods x six metrics")
    target = checkpoint["target"]
    complete = list(completed) == list(range(target))
    if not complete:
        expected_status, expected_formal = "incomplete_no_p", False
        expected_scope = ("primary_full_pipeline" if primary else
                          "sensitivity_only_not_primary" if protocol == "paired_repeated_measures_sensitivity" else "none")
    elif mode == "observed-only":
        expected_status, expected_scope, expected_formal = "no_permutations", "none", False
    elif protocol == "paired_repeated_measures_sensitivity":
        expected_status = "sensitivity_not_formal"
        expected_scope, expected_formal = "sensitivity_only_not_primary", False
    elif mode == "pilot":
        expected_status, expected_scope, expected_formal = "pilot_preliminary", "primary_full_pipeline", False
    else:
        expected_status, expected_scope, expected_formal = "formal", "primary_full_pipeline", True
    observed_result = checkpoint["state"].get("observed")
    observed_methods = (
        observed_result.get("methods")
        if isinstance(observed_result, Mapping) else None
    )
    if not isinstance(observed_methods, Mapping):
        raise _fail(dataset, "permutation inference lacks checkpoint observed values")
    for row in rows:
        metric = row["metric"]
        if row["arm"] != METRIC_ARMS[metric]:
            raise _fail(dataset, "permutation metric/arm mismatch")
        if _integer(row["target_permutations"], "permutation target") != target or _integer(row["completed_permutations"], "completed permutations") != len(completed):
            raise _fail(dataset, "permutation row/checkpoint completion mismatch")
        if row["permutation_plan_sha256"] != checkpoint["permutation_plan_sha256"]:
            raise _fail(dataset, "permutation-plan SHA256 mismatch")
        actual_protocol = row["permutation_protocol"] or None
        actual_unit = row["permutation_unit"] or None
        if actual_protocol != protocol or actual_unit != unit:
            raise _fail(dataset, "primary and paired-sensitivity null scopes differ")
        if row["p_value_status"] != expected_status:
            raise _fail(dataset, "formal p-value status is inconsistent")
        if row["formal_null_scope"] != expected_scope or _boolean(row["is_formal"], "formal p flag") != expected_formal:
            raise _fail(dataset, "formal null scope/status is inconsistent")
        method_result = observed_methods.get(row["method"])
        observed_metrics = (
            method_result.get("observed")
            if isinstance(method_result, Mapping) else None
        )
        if not isinstance(observed_metrics, Mapping):
            raise _fail(dataset, "checkpoint observed method result is malformed")
        checkpoint_observed = _number(
            observed_metrics.get(metric), f"{dataset}.{row['method']}.{metric}"
        )
        table_observed = _number(row["observed_value"], "observed value")
        if not math.isclose(
            table_observed, checkpoint_observed, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise _fail(dataset, "permutation observed value differs from checkpoint")
        metric_column, empty_column = (
            ("gap", "auc") if metric.startswith("gap_") else ("auc", "gap")
        )
        if not math.isclose(
            _number(row[metric_column], f"permutation {metric_column}"),
            checkpoint_observed,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not _blank(row[empty_column]):
            raise _fail(dataset, "permutation observed metric columns are inconsistent")
        valid = _integer(row["valid_null_values"], "valid null values")
        null_values, expected_nulls = checkpoint_null_summary(checkpoint, row["method"], metric)
        if valid != len(null_values):
            raise _fail(dataset, "permutation null count differs from checkpoint")
        for key, expected in expected_nulls.items():
            actual = row[key]
            if ((expected is None and str(actual).strip()) or
                    (expected is not None and not math.isclose(_number(actual, key), expected, rel_tol=1e-12, abs_tol=1e-12))):
                raise _fail(dataset, "permutation null summary differs from checkpoint")
        if target == 0:
            if valid != 0 or any(not _blank(row[key]) for key in ("exceedances", "empirical_p_plus_one", "bh_fdr", "fdr_family")):
                raise _fail(dataset, "observed-only row contains permutation inference")
        elif not complete:
            if any(not _blank(row[key]) for key in ("exceedances", "empirical_p_plus_one", "bh_fdr", "fdr_family")):
                raise _fail(dataset, "incomplete permutation row contains inference or wrong null count")
        else:
            if valid != target or _blank(row["empirical_p_plus_one"]) or _blank(row["bh_fdr"]):
                raise _fail(dataset, "complete permutation row lacks p/FDR values")
            expected_valid, expected_exceedances, expected_p = phipson_smyth_plus_one(
                checkpoint_observed, null_values
            )
            if valid != expected_valid:
                raise _fail(dataset, "valid B differs from checkpoint null")
            if _integer(row["exceedances"], "permutation exceedances") != expected_exceedances:
                raise _fail(dataset, "permutation exceedance count differs from checkpoint null")
            p_value = _number(row["empirical_p_plus_one"], "empirical p")
            fdr = _number(row["bh_fdr"], "BH FDR")
            if not 0.0 <= p_value <= 1.0 or not 0.0 <= fdr <= 1.0:
                raise _fail(dataset, "permutation p/FDR is outside [0, 1]")
            if not math.isclose(p_value, expected_p, rel_tol=1e-12, abs_tol=1e-12):
                raise _fail(dataset, "permutation empirical p differs from checkpoint null")
            expected_family = (
                f"primary_formal:{metric}" if expected_formal
                else f"sensitivity:{expected_status}:{metric}" if not primary
                else f"primary_preliminary:{expected_status}:{metric}"
            )
            if row["fdr_family"] != expected_family:
                raise _fail(dataset, "primary and sensitivity FDR families are not separate")
    return expected_status


def _validate_bh_fdr(rows: Sequence[Mapping[str, Any]]) -> None:
    families: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if _blank(row.get("empirical_p_plus_one")):
            continue
        family = str(row.get("fdr_family", ""))
        if not family:
            raise _fail(str(row.get("dataset", "permutation row")), "p-value lacks an FDR family")
        families.setdefault(family, []).append(row)
    for family, family_rows in families.items():
        ordered = sorted(
            family_rows,
            key=lambda row: (
                _number(row["empirical_p_plus_one"], f"{family}.p"),
                str(row["dataset"]),
                str(row["method"]),
            ),
        )
        expected: dict[int, float] = {}
        running, count = 1.0, len(ordered)
        for rank in range(count, 0, -1):
            row = ordered[rank - 1]
            p_value = _number(row["empirical_p_plus_one"], f"{family}.p")
            running = min(running, p_value * count / rank)
            expected[id(row)] = running
        for row in family_rows:
            actual = _number(row["bh_fdr"], f"{family}.BH FDR")
            if not math.isclose(actual, expected[id(row)], rel_tol=1e-12, abs_tol=1e-12):
                raise _fail(
                    f"{row['dataset']}.{row['method']}.{row['metric']}",
                    "BH FDR differs from independent recomputation",
                )


def _base_flow(dataset: str, source: str, state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset, "dataset_id": dataset, "source": source,
        "selected_whitelist": True, "checkpoint_present": True, "attempted": True,
        "runner_status": state.get("status"),
        "observed_complete": False, "five_method_two_arm_complete": False,
        "shared_fold_repeat_complete": False, "oof_uncertainty_complete": False,
        "oof_uncertainty_estimable_methods": 0, "feature_stability_complete": False,
        "permutation_complete": False, "completed_permutations": 0,
        "target_permutations": state.get("target_permutations"), "null_protocol": None,
        "p_value_status": None, "primary_null_eligible": False,
        "sensitivity_null_eligible": False, "formal_inference_eligible": False,
        "analysis_eligible": False, "classification": None,
        "reason_code": None, "blocking": False,
    }


def build_qc(run_dir: Path | str, table_dir: Path | str, *, mode: str = "formal", expected_whitelist_count: int = CANONICAL_DATASET_COUNT) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Path], dict[str, Any]]:
    """Validate a run and return publishable eligibility rows plus audit inputs."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    run_path, table_path = Path(run_dir).resolve(), Path(table_dir).resolve()
    manifest, checkpoint_list, selected = _validate_manifest(run_path, mode, expected_whitelist_count)
    tables, table_paths = _read_tables(table_path)
    _validate_table_identities(tables, selected)
    if {row["dataset"] for row in tables["run_inventory"]} != set(selected):
        raise _fail("run_inventory", "does not cover every selected dataset")
    flows: list[dict[str, Any]] = []
    for checkpoint in checkpoint_list:
        dataset, state = checkpoint["dataset"], checkpoint["state"]
        inventory = _one_inventory(tables, dataset)
        expected_design = (
            _derive_null_design(checkpoint, tables, inventory)
            if state.get("observed") is not None else None
        )
        completed, protocol, unit, primary = _checkpoint_permutations(
            checkpoint, expected_design, require_design=True
        )
        _validate_inventory(checkpoint, inventory, completed, protocol, primary)
        failure = _validate_failures(
            checkpoint,
            _rows_for(tables, "failures", dataset),
            completed=completed,
        )
        flow = _base_flow(dataset, str(inventory["source"]), state)
        flow.update(
            completed_permutations=len(completed),
            null_protocol=protocol,
            permutation_complete=(state.get("observed") is not None
                                  and completed == list(range(checkpoint["target"]))),
        )
        if failure is not None:
            classification, reason, blocking = failure
            if state.get("observed") is None:
                stale = [name for name in SUCCESS_TABLES if _rows_for(tables, name, dataset)]
                if stale:
                    raise _fail(dataset, f"failed dataset contains stale success rows: {stale}")
            else:
                fold_count = _validate_summary_and_folds(dataset, tables, manifest, checkpoint, inventory)
                estimable = _validate_uncertainty(dataset, tables, manifest, inventory)
                _validate_stability(dataset, tables, fold_count, inventory)
                p_status = _validate_permutation_rows(
                    dataset, tables, mode, checkpoint, completed, protocol, unit, primary)
                flow.update(observed_complete=True, five_method_two_arm_complete=True,
                            shared_fold_repeat_complete=True, oof_uncertainty_complete=True,
                            oof_uncertainty_estimable_methods=estimable,
                            feature_stability_complete=True, p_value_status=p_status)
                if (
                    classification == "null_ineligible_preprocessing"
                    and flow["permutation_complete"]
                ):
                    classification = "blocking_failure"
                    reason = "stale_complete_permutation_output"
                    blocking = True
            flow.update(classification=classification, reason_code=reason, blocking=blocking)
            flows.append(flow)
            continue
        if state.get("status") != "ok":
            raise _fail(dataset, "non-ok checkpoint escaped failure classification")
        if not flow["permutation_complete"]:
            raise _fail(dataset, "status ok but eligible checkpoint is incomplete")
        fold_count = _validate_summary_and_folds(dataset, tables, manifest, checkpoint, inventory)
        estimable = _validate_uncertainty(dataset, tables, manifest, inventory)
        _validate_stability(dataset, tables, fold_count, inventory)
        p_status = _validate_permutation_rows(dataset, tables, mode, checkpoint, completed, protocol, unit, primary)
        if mode == "formal" and primary:
            classification, reason, formal = "eligible_primary_formal", "ok", True
        elif protocol == "paired_repeated_measures_sensitivity":
            classification, reason, formal = (
                "eligible_paired_sensitivity_nonprimary",
                "paired_repeated_measures_sensitivity",
                False,
            )
        elif mode == "pilot":
            classification, reason, formal = "eligible_pilot_nonformal", "pilot_B_below_199", False
        else:
            classification, reason, formal = "eligible_observed_only_nonformal", "B_equals_0", False
        flow.update(
            observed_complete=True,
            five_method_two_arm_complete=True,
            shared_fold_repeat_complete=True,
            oof_uncertainty_complete=True,
            oof_uncertainty_estimable_methods=estimable,
            feature_stability_complete=True,
            p_value_status=p_status,
            primary_null_eligible=primary,
            sensitivity_null_eligible=(protocol == "paired_repeated_measures_sensitivity"),
            formal_inference_eligible=formal,
            analysis_eligible=True,
            classification=classification,
            reason_code=reason,
        )
        flows.append(flow)
    _validate_bh_fdr(tables["permutation_summary"])
    flows.sort(key=lambda row: row["dataset"])
    classifications = (
        "eligible_primary_formal",
        "eligible_paired_sensitivity_nonprimary",
        "eligible_pilot_nonformal",
        "eligible_observed_only_nonformal",
        "design_ineligible_group_cv",
        "design_ineligible_preprocessing",
        "null_ineligible_preprocessing",
        "blocking_failure",
    )
    raw_counts = Counter(str(row["classification"]) for row in flows)
    counts = {classification: raw_counts[classification] for classification in classifications}
    blocking_reasons = Counter(str(row["reason_code"]) for row in flows if row["blocking"])
    design_reasons = Counter(
        str(row["reason_code"])
        for row in flows
        if row["classification"]
        in {"design_ineligible_group_cv", "design_ineligible_preprocessing"}
    )
    null_ineligible_reasons = Counter(
        str(row["reason_code"])
        for row in flows
        if row["classification"] == "null_ineligible_preprocessing"
    )
    if counts["blocking_failure"]:
        status = "fail"
    elif counts["null_ineligible_preprocessing"]:
        status = "pass_with_ineligible"
    elif (
        counts["design_ineligible_group_cv"]
        or counts["design_ineligible_preprocessing"]
    ):
        status = "pass_with_design_ineligible"
    else:
        status = "pass"
    formal_allowed = mode == "formal" and status != "fail" and counts["eligible_primary_formal"] > 0
    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "statistic_rule": STATISTIC_RULE,
        "permutation_inference_audit": dict(PERMUTATION_INFERENCE_AUDIT),
        "qc_status": status,
        "mode": mode,
        "formal_claims_allowed": formal_allowed,
        "nonformal_reason": None if mode == "formal" else (
            "pilot_B_below_199" if mode == "pilot" else "observed_only_B_equals_0"),
        "selected_whitelist_count": len(selected),
        "target_permutations": manifest["protocol_config"]["n_permutations"],
        "counts": counts,
        "blocking_reason_counts": dict(sorted(blocking_reasons.items())),
        "design_ineligible_reason_counts": dict(sorted(design_reasons.items())),
        "null_ineligible_reason_counts": dict(
            sorted(null_ineligible_reasons.items())
        ),
        "required_methods": list(METHODS),
        "required_arms": ["pre_split", "fold_contained"],
        "primary_and_sensitivity_nulls_separated": True,
    }
    inputs = {"run_manifest": run_path / "manifest.json",
              **{f"aggregation/{name}": path for name, path in table_paths.items()}}
    checkpoint_hashes = {checkpoint["dataset"]: sha256_file(checkpoint["path"]) for checkpoint in checkpoint_list}
    audit = {"run_dir": run_path, "table_dir": table_path, "inputs": inputs,
             "checkpoint_hashes": checkpoint_hashes, "selected_ids_sha256": sha256_json(selected)}
    return flows, summary, table_paths, audit


__all__ = ["FLOW_COLUMNS", "FORMAL_PERMUTATIONS", "MODES", "QCError", "build_qc"]
