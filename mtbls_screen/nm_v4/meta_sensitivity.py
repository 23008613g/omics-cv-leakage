"""Repository-cluster and cohort sensitivity analyses for meta-analysis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

from .human_cohort_adjudication import (
    CANONICAL_MANUAL_FIELDS,
    CANONICAL_STATUSES,
    COMPONENT_VALUES,
    DEFAULT_CANONICAL_ROWS,
    DEFAULT_PENDING_ROWS,
    OUTPUT_FIELDS,
    OUTPUT_AUDIT_FIELDS,
    REVIEW_COMPONENT_FIELDS,
    SCHEMA_VERSION as COHORT_SCHEMA_VERSION,
    TOOL_VERSION as COHORT_TOOL_VERSION,
    merge_human_cohort_adjudication,
)
from .meta_preparation import prepare_tidy_summary

if TYPE_CHECKING:
    from .meta_analysis import MetaAnalysisConfig


BOOTSTRAP_COLUMNS = tuple(
    """analysis effect_column estimand method n_rows n_datasets n_repositories
    estimate ci_low ci_high bootstrap_replicates successful_bootstraps cluster_unit
    exploratory robust_ci inference_status minimum_repository_clusters_for_interval
    minimum_repository_clusters_for_robust_ci threshold_basis minimum_total_units
    minimum_minority_units n_excluded_missing_minority_units exclusion_reason
    independent_unit_definition participant_level_claim""".split()
)
LORO_COLUMNS = tuple(
    """analysis left_out_repository method estimand estimate n_datasets
    n_remaining_repositories external_validation interpretation
    left_out_repository_specific_estimate n_left_out_datasets loro_estimate_min
    loro_estimate_max loro_estimate_range exploratory range_scope""".split()
)
COHORT_COMPARISON_COLUMNS = tuple(
    """analysis subgroup method effect_column estimand estimate ci_low ci_high
    n_datasets n_repositories included_dataset_ids n_resolved_datasets
    n_excluded_unresolved excluded_unresolved_dataset_ids inference_status robust_ci
    ci_method cluster_unit bootstrap_replicates exclusion_reason exploratory
    minimum_repository_clusters_for_robust_ci""".split()
)
COHORT_AUDIT_CONTRACT = "human_cohort_adjudication_v2_manual/v1"
COHORT_CSV_NAME = "human_cohort_adjudication_v2_manual.csv"
_COHORT_SOURCE_PATHS = {
    "canonical": Path("human_cohort_adjudication.csv"),
    **{
        f"reviewer_{reviewer}_chunk_{chunk}": Path(f"manual_reviews_v4/chunk{chunk}_reviewer{reviewer}.csv")
        for reviewer in (1, 2) for chunk in (1, 2, 3)
    },
    "adjudication_1": Path("manual_reviews_v4/adjudication_chunk12.csv"),
    "adjudication_2": Path("manual_reviews_v4/adjudication_chunk3.csv"),
}


@dataclass(frozen=True, slots=True)
class _ValidatedCohort:
    path: Path
    manifest_path: Path
    frame: pd.DataFrame
    audit: Mapping[str, Any]


def _default_config() -> MetaAnalysisConfig:
    # Import lazily so the public configuration class remains owned by the facade.
    from .meta_analysis import MetaAnalysisConfig

    return MetaAnalysisConfig()


def _derived_seed(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _bootstrap_prepared(
    data: pd.DataFrame,
    *,
    config: MetaAnalysisConfig,
    analysis: str,
    effect_column: str,
    threshold_basis: str = "none",
    minimum_total_units: int | None = None,
    minimum_minority_units: int | None = None,
    n_excluded_missing_minority_units: int = 0,
    exclusion_reason: str = "",
    independent_unit_definition: str = "not_applicable",
) -> pd.DataFrame:
    if effect_column not in data:
        raise ValueError(f"unknown effect column: {effect_column}")
    usable = data[np.isfinite(data[effect_column])].copy()
    rows: list[dict[str, Any]] = []
    for method in sorted(usable["method"].unique()):
        method_data = usable[usable.method == method]
        repositories = sorted(method_data["repository"].unique())
        by_repository = {
            repository: method_data.loc[
                method_data.repository == repository, effect_column
            ].to_numpy(float)
            for repository in repositories
        }
        rng = np.random.default_rng(
            _derived_seed(config.seed, analysis, effect_column, str(method))
        )
        means: list[float] = []
        medians: list[float] = []
        interval_estimable = (
            len(repositories) >= config.minimum_repository_clusters_for_interval
        )
        if interval_estimable:
            for _ in range(config.bootstrap_replicates):
                sampled = rng.choice(repositories, size=len(repositories), replace=True)
                values = np.concatenate([by_repository[item] for item in sampled])
                means.append(float(np.mean(values)))
                medians.append(float(np.median(values)))
        robust_ci = (
            interval_estimable
            and len(repositories) >= config.minimum_repository_clusters_for_robust_ci
        )
        if not interval_estimable:
            inference_status = "insufficient_repository_clusters_for_interval"
        elif robust_ci:
            inference_status = "exploratory_repository_cluster_interval"
        else:
            inference_status = "exploratory_interval_insufficient_repository_clusters"
        for name, estimate, distribution in (
            (f"mean_{effect_column}", method_data[effect_column].mean(), means),
            (f"median_{effect_column}", method_data[effect_column].median(), medians),
        ):
            low, high = (
                np.quantile(distribution, [config.alpha / 2, 1 - config.alpha / 2])
                if distribution
                else (np.nan, np.nan)
            )
            rows.append(
                {
                    "analysis": analysis,
                    "effect_column": effect_column,
                    "estimand": name,
                    "method": method,
                    "n_rows": len(method_data),
                    "n_datasets": method_data["dataset_id"].nunique(),
                    "n_repositories": len(repositories),
                    "estimate": float(estimate),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_replicates": config.bootstrap_replicates,
                    "successful_bootstraps": len(distribution),
                    "cluster_unit": "repository",
                    "exploratory": True,
                    "robust_ci": robust_ci,
                    "inference_status": inference_status,
                    "minimum_repository_clusters_for_interval": (
                        config.minimum_repository_clusters_for_interval
                    ),
                    "minimum_repository_clusters_for_robust_ci": (
                        config.minimum_repository_clusters_for_robust_ci
                    ),
                    "threshold_basis": threshold_basis,
                    "minimum_total_units": minimum_total_units,
                    "minimum_minority_units": minimum_minority_units,
                    "n_excluded_missing_minority_units": (
                        n_excluded_missing_minority_units
                    ),
                    "exclusion_reason": exclusion_reason,
                    "independent_unit_definition": independent_unit_definition,
                    "participant_level_claim": False,
                }
            )
    return (
        pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)
        .sort_values(["method", "estimand"], kind="stable")
        .reset_index(drop=True)
    )


def repository_cluster_bootstrap(
    summary: pd.DataFrame,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    chosen = config or _default_config()
    return _bootstrap_prepared(
        prepare_tidy_summary(summary),
        config=chosen,
        analysis="overall_repository_cluster_bootstrap",
        effect_column="gap",
    )


def leave_one_repository_out(summary: pd.DataFrame) -> pd.DataFrame:
    """Descriptive LORO sensitivity; this is not external validation."""
    data = prepare_tidy_summary(summary)
    rows: list[dict[str, Any]] = []
    for excluded in sorted(data["repository"].unique()):
        retained = data[data.repository != excluded]
        for method in sorted(retained["method"].unique()):
            values = retained.loc[retained.method == method, "gap"]
            excluded_values = data.loc[
                (data.repository == excluded) & (data.method == method), "gap"
            ]
            for estimand, estimate, repository_estimate in (
                ("mean_gap", values.mean(), excluded_values.mean()),
                ("median_gap", values.median(), excluded_values.median()),
            ):
                rows.append(
                    {
                        "analysis": "leave_one_repository_out_sensitivity",
                        "left_out_repository": excluded,
                        "method": method,
                        "estimand": estimand,
                        "estimate": float(estimate),
                        "n_datasets": retained.loc[
                            retained.method == method, "dataset_id"
                        ].nunique(),
                        "n_remaining_repositories": retained["repository"].nunique(),
                        "external_validation": False,
                        "interpretation": (
                            "sensitivity analysis; not external validation"
                        ),
                        "left_out_repository_specific_estimate": float(
                            repository_estimate
                        ),
                        "n_left_out_datasets": int(excluded_values.size),
                        "loro_estimate_min": np.nan,
                        "loro_estimate_max": np.nan,
                        "loro_estimate_range": np.nan,
                        "exploratory": True,
                        "range_scope": ("across_leave_one_repository_out_estimates"),
                    }
                )
    result = pd.DataFrame(rows, columns=LORO_COLUMNS)
    if result.empty:
        return result
    grouped = result.groupby(["method", "estimand"], sort=True)["estimate"]
    result["loro_estimate_min"] = grouped.transform("min")
    result["loro_estimate_max"] = grouped.transform("max")
    result["loro_estimate_range"] = (
        result["loro_estimate_max"] - result["loro_estimate_min"]
    )
    return result.sort_values(
        ["left_out_repository", "method", "estimand"], kind="stable"
    ).reset_index(drop=True)


def sample_size_sensitivity(
    summary: pd.DataFrame,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    data, chosen = prepare_tidy_summary(summary), config or _default_config()
    frames = []
    for total, minority in chosen.observation_count_thresholds:
        name = f"observation_count_n_ge_{total}_minority_observations_ge_{minority}"
        mask = (data.n_observations >= total) & (data.minority_observations >= minority)
        if mask.any():
            frames.append(
                _bootstrap_prepared(
                    data[mask],
                    config=chosen,
                    analysis=name,
                    effect_column="gap",
                    threshold_basis="observation_count",
                    minimum_total_units=total,
                    minimum_minority_units=minority,
                )
            )
    if not frames:
        return pd.DataFrame(columns=BOOTSTRAP_COLUMNS)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["analysis", "method", "estimand"], kind="stable")
        .reset_index(drop=True)
    )


def _no_eligible_independent_unit_rows(
    data: pd.DataFrame,
    *,
    config: MetaAnalysisConfig,
    analysis: str,
    minimum_total_units: int,
    minimum_minority_units: int,
    n_excluded_missing: int,
    exclusion_reason: str,
) -> pd.DataFrame:
    rows = []
    for method in sorted(data["method"].unique()):
        for estimand in ("mean_gap", "median_gap"):
            rows.append(
                {
                    "analysis": analysis,
                    "effect_column": "gap",
                    "estimand": estimand,
                    "method": method,
                    "n_rows": 0,
                    "n_datasets": 0,
                    "n_repositories": 0,
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "bootstrap_replicates": config.bootstrap_replicates,
                    "successful_bootstraps": 0,
                    "cluster_unit": "repository",
                    "exploratory": True,
                    "robust_ci": False,
                    "inference_status": "no_eligible_datasets",
                    "minimum_repository_clusters_for_interval": (
                        config.minimum_repository_clusters_for_interval
                    ),
                    "minimum_repository_clusters_for_robust_ci": (
                        config.minimum_repository_clusters_for_robust_ci
                    ),
                    "threshold_basis": "independent_unit_count",
                    "minimum_total_units": minimum_total_units,
                    "minimum_minority_units": minimum_minority_units,
                    "n_excluded_missing_minority_units": n_excluded_missing,
                    "exclusion_reason": exclusion_reason,
                    "independent_unit_definition": (
                        "observation_or_group_from_audited_group_codes"
                    ),
                    "participant_level_claim": False,
                }
            )
    return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)


def independent_unit_sensitivity(
    summary: pd.DataFrame,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    """Apply thresholds to audited independent units, never assumed participants."""
    data, chosen = prepare_tidy_summary(summary), config or _default_config()
    datasets = data.drop_duplicates("dataset_id")
    missing = datasets["minority_independent_units"].isna()
    n_excluded_missing = int(missing.sum())
    reasons = sorted(
        datasets.loc[missing, "minority_independent_units_reason"].astype(str).unique()
    )
    exclusion_reason = (
        "excluded_missing_minority_independent_units:" + "|".join(reasons)
        if reasons
        else ""
    )
    frames = []
    for total, minority in chosen.independent_unit_thresholds:
        analysis = f"independent_unit_n_ge_{total}_minority_units_ge_{minority}"
        eligible = (
            data["independent_unit_metadata_available"]
            & data["n_independent_units"].ge(total)
            & data["minority_independent_units"].ge(minority)
        )
        if eligible.any():
            frames.append(
                _bootstrap_prepared(
                    data[eligible],
                    config=chosen,
                    analysis=analysis,
                    effect_column="gap",
                    threshold_basis="independent_unit_count",
                    minimum_total_units=total,
                    minimum_minority_units=minority,
                    n_excluded_missing_minority_units=n_excluded_missing,
                    exclusion_reason=exclusion_reason,
                    independent_unit_definition=(
                        "observation_or_group_from_audited_group_codes"
                    ),
                )
            )
        else:
            frames.append(
                _no_eligible_independent_unit_rows(
                    data,
                    config=chosen,
                    analysis=analysis,
                    minimum_total_units=total,
                    minimum_minority_units=minority,
                    n_excluded_missing=n_excluded_missing,
                    exclusion_reason=(
                        exclusion_reason or "no_dataset_met_prespecified_threshold"
                    ),
                )
            )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["analysis", "method", "estimand"], kind="stable")
        .reset_index(drop=True)
    )


def _cohort_error(message: str) -> ValueError:
    return ValueError(f"audited cohort contract failed: {message}")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise _cohort_error(f"cannot read provenance file {path}: {exc}") from exc
    return digest.hexdigest()


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype(str).str.strip()


def _validate_cohort_rows(frame: pd.DataFrame) -> dict[str, Any]:
    if tuple(frame.columns) != OUTPUT_FIELDS:
        raise _cohort_error("CSV must have the exact frozen 41-column schema")
    identifiers = _text(frame, "id")
    invalid_ids = identifiers.eq("").any() or identifiers.duplicated().any()
    if len(frame) != DEFAULT_CANONICAL_ROWS or invalid_ids:
        raise _cohort_error("CSV must contain 1,102 unique non-empty dataset IDs")
    strict = _text(frame, "strict_observational_status")
    if not set(strict).issubset(CANONICAL_STATUSES):
        raise _cohort_error("CSV has invalid automatic strict statuses")
    pending = strict.eq("needs_manual_review")
    if int(pending.sum()) != DEFAULT_PENDING_ROWS:
        raise _cohort_error("CSV must contain the complete 258-row manual queue")
    final = _text(frame, "manual_final_status")
    final_components = [_text(frame, field) for field in REVIEW_COMPONENT_FIELDS]
    if not set(final[pending]).issubset({"include", "exclude"}):
        raise _cohort_error("manual final status is unresolved")
    if any(not set(values[pending]).issubset(COMPONENT_VALUES) for values in final_components):
        raise _cohort_error("manual final evidence is invalid")
    final_includable = (final_components[0].eq("yes")
                        & final_components[1].eq("yes") & final_components[2].eq("no"))
    if ((final.eq("include") != final_includable) & pending).any():
        raise _cohort_error("manual final status conflicts with evidence")
    decision_1 = _text(frame, "reviewer_1_decision")
    decision_2 = _text(frame, "reviewer_2_decision")
    component_mismatch = pd.concat([
        _text(frame, f"reviewer_1_{field}").ne(_text(frame, f"reviewer_2_{field}"))
        for field in REVIEW_COMPONENT_FIELDS
    ], axis=1).any(axis=1)
    requires_adjudication = pending & (
        decision_1.ne(decision_2)
        | decision_1.eq("needs_adjudication")
        | decision_2.eq("needs_adjudication")
        | component_mismatch
    )
    adjudication_fields = tuple(
        "adjudicator adjudication_decision adjudication_notes adjudication_date".split()
    )
    supplied = pd.concat(
        [_text(frame, field).ne("") for field in adjudication_fields], axis=1
    )
    adjudication = _text(frame, "adjudication_decision")
    if (
        not supplied.loc[requires_adjudication].all(axis=None)
        or supplied.loc[~requires_adjudication].any(axis=None)
    ):
        raise _cohort_error("adjudication coverage is incomplete")
    if not set(adjudication[requires_adjudication]).issubset({"include", "exclude"}):
        raise _cohort_error("adjudication decision is unresolved")
    if (adjudication.ne(final) & requires_adjudication).any():
        raise _cohort_error("adjudication decision conflicts with final status")
    dates = _text(frame, "adjudication_date")
    if not dates[requires_adjudication].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise _cohort_error("adjudication date is invalid")
    manual_fields = (*CANONICAL_MANUAL_FIELDS, *OUTPUT_AUDIT_FIELDS)
    altered = frame.loc[~pending, manual_fields].apply(lambda column: column.str.strip())
    if altered.ne("").to_numpy().any():
        raise _cohort_error("automatic cohort rows contain manual audit data")
    effective = adjudication.where(adjudication.ne(""), final.where(final.ne(""), strict))
    if not set(effective).issubset({"include", "exclude"}):
        raise _cohort_error("one or more effective cohort statuses are unresolved")
    counts = {status: int(effective.eq(status).sum()) for status in ("include", "exclude")}
    return {
        "effective_status_counts": {**counts, "needs_adjudication": 0},
        "adjudicated_count": int(requires_adjudication.sum()),
        "ordered_ids_sha256": hashlib.sha256(
            ("\n".join(identifiers) + "\n").encode()
        ).hexdigest(),
    }


def _verify_cohort_replay(path: Path, expected_hash: str) -> None:
    root = path.parent
    sources = {key: root / relative for key, relative in _COHORT_SOURCE_PATHS.items()}
    try:
        with TemporaryDirectory(prefix="human-cohort-replay-") as staging:
            replay = Path(staging) / COHORT_CSV_NAME
            merge_human_cohort_adjudication(
                canonical_csv=sources["canonical"],
                reviewer_1_csvs=tuple(sources[f"reviewer_1_chunk_{i}"] for i in (1, 2, 3)),
                reviewer_2_csvs=tuple(sources[f"reviewer_2_chunk_{i}"] for i in (1, 2, 3)),
                adjudication_csvs=tuple(sources[f"adjudication_{i}"] for i in (1, 2)),
                output_csv=replay,
            )
            replay_hash = _sha256_path(replay)
    except (OSError, ValueError) as exc:
        raise _cohort_error(f"deterministic source replay failed: {exc}") from exc
    if replay_hash != expected_hash:
        raise _cohort_error("deterministic source replay does not match published output")


def _cohort_manifest_audit(
    path: Path, manifest_path: Path, manifest: Mapping[str, Any], semantics: Mapping[str, Any],
) -> dict[str, Any]:
    csv_hash, manifest_hash = _sha256_path(path), _sha256_path(manifest_path)
    output_contract = (
        manifest.get("schema_version") == COHORT_SCHEMA_VERSION
        and manifest.get("tool_version") == COHORT_TOOL_VERSION
        and manifest.get("output_file") == path.name
        and manifest.get("output_csv_name") == path.name
        and manifest.get("output_sha256") == csv_hash
        and manifest.get("output_csv_sha256") == csv_hash
    )
    if not output_contract:
        raise _cohort_error("manifest output name, version, or SHA-256 hash is invalid")
    scalar_counts = {
        "row_count": DEFAULT_CANONICAL_ROWS,
        "pending_input_count": DEFAULT_PENDING_ROWS,
        "reviewed_count": DEFAULT_PENDING_ROWS,
        "reviewer1_coverage": DEFAULT_PENDING_ROWS,
        "reviewer2_coverage": DEFAULT_PENDING_ROWS,
        "adjudicated_count": semantics["adjudicated_count"],
    }
    if any(
        type(manifest.get(key)) is not int or manifest[key] != value
        for key, value in scalar_counts.items()
    ):
        raise _cohort_error("manifest audit counts are incomplete")
    if (
        manifest.get("final_status_counts") != semantics["effective_status_counts"]
        or manifest.get("ordered_ids_sha256") != semantics["ordered_ids_sha256"]
    ):
        raise _cohort_error("manifest status counts or ordered-ID hash disagree with CSV")
    counts, inputs = manifest.get("counts"), manifest.get("input_sha256")
    if not isinstance(counts, dict) or counts.get("effective_unresolved") != 0:
        raise _cohort_error("manifest does not prove a fully resolved cohort")
    if not isinstance(inputs, dict) or set(inputs) != set(_COHORT_SOURCE_PATHS):
        raise _cohort_error("manifest source hash inventory is incomplete")
    if manifest.get("base_input_sha256") != inputs["canonical"]:
        raise _cohort_error("base-input source hash link is invalid")
    for key, relative_path in _COHORT_SOURCE_PATHS.items():
        source = path.parent / relative_path
        if not source.is_file() or inputs[key] != _sha256_path(source):
            raise _cohort_error(f"source SHA-256 hash mismatch: {key}")
    code_hash = _sha256_path(Path(__file__).with_name("human_cohort_adjudication.py"))
    if manifest.get("code_sha256") != code_hash:
        raise _cohort_error("merge-code SHA-256 hash mismatch")
    _verify_cohort_replay(path, csv_hash)
    return {
        "contract": COHORT_AUDIT_CONTRACT, "status": "validated_resolved",
        "strict_human_tables_generated": True, "csv_name": path.name,
        "csv_sha256": csv_hash, "manifest_name": manifest_path.name,
        "manifest_sha256": manifest_hash, "schema_version": COHORT_SCHEMA_VERSION,
        **scalar_counts,
        "effective_status_counts": semantics["effective_status_counts"],
        "effective_unresolved": 0, "ordered_ids_sha256": semantics["ordered_ids_sha256"],
        "source_input_sha256": dict(sorted(inputs.items())),
        "merge_code_sha256": code_hash,
    }


def _validated_cohort(cohort: str | Path | _ValidatedCohort) -> _ValidatedCohort:
    if isinstance(cohort, _ValidatedCohort):
        return cohort
    if not isinstance(cohort, (str, Path)):
        raise _cohort_error(
            "supply a CSV path; DataFrame input cannot prove audit provenance"
        )
    path = Path(cohort)
    if path.name != COHORT_CSV_NAME or not path.is_file():
        raise _cohort_error(f"expected readable {COHORT_CSV_NAME}")
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise _cohort_error("adjacent adjudication manifest sidecar is missing")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise _cohort_error(f"cannot read CSV/manifest bundle: {exc}") from exc
    if not isinstance(manifest, dict):
        raise _cohort_error("manifest must contain a JSON object")
    semantics = _validate_cohort_rows(frame)
    audit = _cohort_manifest_audit(path, manifest_path, manifest, semantics)
    validated_frame = frame.assign(dataset_id=_text(frame, "id"))
    return _ValidatedCohort(path, manifest_path, validated_frame, audit)


def _cohort_audit_payload(cohort: _ValidatedCohort | None) -> dict[str, Any]:
    if cohort is None:
        return {"contract": COHORT_AUDIT_CONTRACT, "status": "not_provided", "strict_human_tables_generated": False}
    return dict(cohort.audit)


def _cohort_membership(
    data: pd.DataFrame, cohort: str | Path | _ValidatedCohort | None
) -> pd.DataFrame:
    datasets = data[["dataset_id"]].drop_duplicates().sort_values("dataset_id")
    if cohort is None:
        result = datasets.assign(
            strict_observational_status="missing",
            strict_human_observational_included=False,
            strict_inclusion_basis="no_cohort_record",
        )
        return result.reset_index(drop=True)
    review = _validated_cohort(cohort).frame.set_index("dataset_id")
    missing = sorted(set(datasets["dataset_id"]).difference(review.index))
    if missing:
        detail = "bundle does not cover every meta-analysis dataset: "
        raise _cohort_error(detail + "|".join(missing))
    rows = []
    for dataset_id in datasets["dataset_id"]:
        record = review.loc[dataset_id]
        automatic = str(record["strict_observational_status"]).strip()
        manual = str(record["manual_final_status"]).strip()
        adjudication = str(record["adjudication_decision"]).strip()
        if adjudication:
            final, basis = adjudication, "adjudication_decision"
        elif manual:
            final, basis = manual, "manual_final_status"
        else:
            final, basis = automatic, "automatic_status"
        rows.append((dataset_id, final, final == "include", basis))
    return pd.DataFrame(
        rows,
        columns=(
            "dataset_id",
            "strict_observational_status",
            "strict_human_observational_included",
            "strict_inclusion_basis",
        ),
    )


def strict_human_observational(
    summary: pd.DataFrame,
    cohort: str | Path | _ValidatedCohort,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    data, chosen = prepare_tidy_summary(summary), config or _default_config()
    membership = _cohort_membership(data, cohort)
    identifiers = membership.loc[
        membership.strict_human_observational_included, "dataset_id"
    ].tolist()
    result = _bootstrap_prepared(
        data[data.dataset_id.isin(identifiers)],
        config=chosen,
        analysis="strict_human_observational",
        effect_column="gap",
    )
    result["included_dataset_ids"] = "|".join(sorted(identifiers))
    return result


def cohort_subgroup_comparison(
    summary: pd.DataFrame,
    cohort: str | Path | _ValidatedCohort | None,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    """Compare disjoint resolved cohort groups; unresolved records enter neither."""
    data, chosen = prepare_tidy_summary(summary), config or _default_config()
    membership = _cohort_membership(data, cohort)
    resolved = membership.strict_observational_status.isin(["include", "exclude"])
    unresolved = membership[~resolved]
    unresolved_ids = sorted(unresolved["dataset_id"].tolist())
    resolved_count = int(resolved.sum())
    exclusion_reason = (
        "cohort not provided; all datasets excluded from both resolved subgroups"
        if cohort is None
        else "needs_manual_review or missing cohort classification excluded from both subgroups"
    )
    methods = sorted(data["method"].unique())
    rows: list[dict[str, Any]] = []
    for subgroup, decision in (
        ("strict_human_observational", "include"),
        ("other_resolved_or_non_strict", "exclude"),
    ):
        selected = membership.strict_observational_status == decision
        identifiers = sorted(membership.loc[selected, "dataset_id"].tolist())
        subset = data[data.dataset_id.isin(identifiers)]
        bootstrap = _bootstrap_prepared(
            subset,
            config=chosen,
            analysis=f"cohort_subgroup_{subgroup}",
            effect_column="gap",
        )
        for method in methods:
            method_data = subset[subset.method == method]
            n_datasets = int(method_data["dataset_id"].nunique())
            n_repositories = int(method_data["repository"].nunique())
            estimate = float(method_data["gap"].mean()) if n_datasets else np.nan
            ci_low = ci_high = np.nan
            robust_ci = False
            if not n_datasets:
                status = "no_eligible_datasets"
            elif n_repositories < chosen.minimum_repository_clusters_for_interval:
                status = "insufficient_repositories_for_cluster_ci"
            else:
                match = bootstrap[
                    (bootstrap.method == method) & (bootstrap.estimand == "mean_gap")
                ]
                if match.empty:
                    status = "bootstrap_unavailable"
                else:
                    record = match.iloc[0]
                    ci_low, ci_high = float(record.ci_low), float(record.ci_high)
                    robust_ci = bool(record.robust_ci)
                    status = str(record.inference_status)
            rows.append(
                {
                    "analysis": "resolved_cohort_subgroup_comparison",
                    "subgroup": subgroup,
                    "method": method,
                    "effect_column": "gap",
                    "estimand": "mean_paired_gap",
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_datasets": n_datasets,
                    "n_repositories": n_repositories,
                    "included_dataset_ids": "|".join(identifiers),
                    "n_resolved_datasets": resolved_count,
                    "n_excluded_unresolved": len(unresolved_ids),
                    "excluded_unresolved_dataset_ids": "|".join(unresolved_ids),
                    "inference_status": status,
                    "robust_ci": robust_ci,
                    "ci_method": (
                        "repository_cluster_percentile_bootstrap_exploratory"
                        if np.isfinite([ci_low, ci_high]).all()
                        else "not_estimable"
                    ),
                    "cluster_unit": "repository",
                    "bootstrap_replicates": chosen.bootstrap_replicates,
                    "exclusion_reason": exclusion_reason,
                    "exploratory": True,
                    "minimum_repository_clusters_for_robust_ci": (
                        chosen.minimum_repository_clusters_for_robust_ci
                    ),
                }
            )
    result = pd.DataFrame(rows, columns=COHORT_COMPARISON_COLUMNS)
    return result.sort_values(["subgroup", "method"], kind="stable").reset_index(
        drop=True
    )


def ceiling_sensitivity(
    summary: pd.DataFrame,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    data, chosen = prepare_tidy_summary(summary), config or _default_config()
    adjusted = _bootstrap_prepared(
        data[data.headroom > 0],
        config=chosen,
        analysis="headroom_adjusted_gap_sensitivity",
        effect_column="headroom_adjusted_gap",
    )
    excluded = _bootstrap_prepared(
        data[data.fold_contained_auc < chosen.ceiling_auc],
        config=chosen,
        analysis=f"exclude_fold_contained_auc_ge_{chosen.ceiling_auc:.2f}",
        effect_column="gap",
    )
    result = pd.concat([adjusted, excluded], ignore_index=True)
    result["ceiling_definition"] = (
        f"method-specific fold_contained_auc >= {chosen.ceiling_auc:.2f}"
    )
    return result.sort_values(
        ["analysis", "method", "estimand"], kind="stable"
    ).reset_index(drop=True)


__all__ = [
    "BOOTSTRAP_COLUMNS",
    "COHORT_COMPARISON_COLUMNS",
    "LORO_COLUMNS",
    "_bootstrap_prepared",
    "_cohort_membership",
    "ceiling_sensitivity",
    "cohort_subgroup_comparison",
    "independent_unit_sensitivity",
    "leave_one_repository_out",
    "repository_cluster_bootstrap",
    "sample_size_sensitivity",
    "strict_human_observational",
]
