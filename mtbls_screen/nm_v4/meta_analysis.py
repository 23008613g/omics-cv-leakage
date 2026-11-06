"""Auditable heterogeneity analyses for tidy dataset-method summaries."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import statsmodels
from .meta_models import (
    PLATFORM_MARGINAL_ESTIMAND,
    PLATFORM_REFERENCE_DISTRIBUTION,
    fit_gee_prepared,
    fit_mixedlm_prepared,
    marginal_predictions_prepared,
    standardized_coefficients_prepared,
    weighting_label,
)
from .meta_preparation import (
    INDEPENDENT_UNIT_COLUMNS,
    REQUIRED_COLUMNS,
    prepare_tidy_summary as _prepare_tidy_summary,
)
from .meta_sensitivity import (
    BOOTSTRAP_COLUMNS,
    _ValidatedCohort,
    _cohort_audit_payload,
    _cohort_membership,
    _validated_cohort,
    ceiling_sensitivity as _ceiling_sensitivity,
    cohort_subgroup_comparison as _cohort_subgroup_comparison,
    independent_unit_sensitivity as _independent_unit_sensitivity,
    leave_one_repository_out as _leave_one_repository_out,
    repository_cluster_bootstrap as _repository_cluster_bootstrap,
    sample_size_sensitivity as _sample_size_sensitivity,
    strict_human_observational as _strict_human_observational,
)


@dataclass(frozen=True, slots=True)
class MetaAnalysisConfig:
    bootstrap_replicates: int = 2_000
    seed: int = 20_260_814
    alpha: float = 0.05
    ceiling_auc: float = 0.95
    prediction_points: int = 51
    observation_count_thresholds: tuple[tuple[int, int], ...] = (
        (50, 20),
        (100, 30),
    )
    independent_unit_thresholds: tuple[tuple[int, int], ...] = (
        (50, 20),
        (100, 30),
    )
    minimum_repository_clusters_for_interval: int = 2
    minimum_repository_clusters_for_robust_ci: int = 20
    gee_max_condition_number: float = 1e12
    difficulty_spline_min_unique_values: int = 6
    difficulty_spline_min_datasets: int = 12
    platform_marginal_min_datasets: int = 2
    mixedlm_singular_variance_threshold: float = 1e-8

    def __post_init__(self) -> None:
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if not 0.5 < self.ceiling_auc < 1:
            raise ValueError("ceiling_auc must lie strictly between 0.5 and one")
        if self.prediction_points < 2:
            raise ValueError("prediction_points must be at least two")
        _validate_threshold_pairs(
            self.observation_count_thresholds,
            name="observation_count_thresholds",
        )
        _validate_threshold_pairs(
            self.independent_unit_thresholds,
            name="independent_unit_thresholds",
        )
        if self.minimum_repository_clusters_for_interval < 2:
            raise ValueError(
                "minimum_repository_clusters_for_interval must be at least two"
            )
        if (
            self.minimum_repository_clusters_for_robust_ci
            < self.minimum_repository_clusters_for_interval
        ):
            raise ValueError(
                "minimum_repository_clusters_for_robust_ci cannot be smaller "
                "than the interval threshold"
            )
        if (
            not np.isfinite(self.gee_max_condition_number)
            or self.gee_max_condition_number < 1
        ):
            raise ValueError("gee_max_condition_number must be finite and >= 1")
        for name in (
            "difficulty_spline_min_unique_values",
            "difficulty_spline_min_datasets",
            "platform_marginal_min_datasets",
        ):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least two")
        if (
            not np.isfinite(self.mixedlm_singular_variance_threshold)
            or self.mixedlm_singular_variance_threshold <= 0
        ):
            raise ValueError("mixedlm_singular_variance_threshold must be positive")


def _validate_threshold_pairs(
    values: tuple[tuple[int, int], ...],
    *,
    name: str,
) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    previous: tuple[int, int] | None = None
    for value in values:
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in value
            )
            or value[0] <= 0
            or value[1] <= 0
        ):
            raise ValueError(f"{name} must contain positive integer pairs")
        if value[1] > value[0]:
            raise ValueError(f"{name} minority threshold exceeds total threshold")
        if previous is not None and not (
            value[0] > previous[0] and value[1] > previous[1]
        ):
            raise ValueError(f"{name} pairs must be strictly increasing")
        previous = value


def prepare_tidy_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Validate and derive covariates without mutating the caller's table."""
    return _prepare_tidy_summary(summary)


def repository_cluster_bootstrap(
    summary: pd.DataFrame,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    return _repository_cluster_bootstrap(summary, config=config)


def leave_one_repository_out(summary: pd.DataFrame) -> pd.DataFrame:
    """Descriptive LORO sensitivity; this is not external validation."""
    return _leave_one_repository_out(summary)


def sample_size_sensitivity(
    summary: pd.DataFrame,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    return _sample_size_sensitivity(summary, config=config)


def independent_unit_sensitivity(
    summary: pd.DataFrame,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    """Apply thresholds to audited independent units, never assumed participants."""
    return _independent_unit_sensitivity(summary, config=config)


def strict_human_observational(
    summary: pd.DataFrame,
    cohort: str | Path,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    """Analyze only a fully resolved, provenance-validated cohort bundle."""
    return _strict_human_observational(summary, cohort, config=config)


def cohort_subgroup_comparison(
    summary: pd.DataFrame,
    cohort: str | Path | None,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    """Compare a validated cohort, or emit no claims when cohort is ``None``."""
    return _cohort_subgroup_comparison(summary, cohort, config=config)


def ceiling_sensitivity(
    summary: pd.DataFrame,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    return _ceiling_sensitivity(summary, config=config)


def fit_gee_models(
    summary: pd.DataFrame,
    *,
    config: MetaAnalysisConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen = config or MetaAnalysisConfig()
    coefficients, diagnostics, _ = fit_gee_prepared(
        prepare_tidy_summary(summary),
        alpha=chosen.alpha,
        max_condition_number=chosen.gee_max_condition_number,
        difficulty_spline_min_unique_values=(
            chosen.difficulty_spline_min_unique_values
        ),
        difficulty_spline_min_datasets=chosen.difficulty_spline_min_datasets,
    )
    return coefficients, diagnostics


def standardized_continuous_coefficients(
    summary: pd.DataFrame,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    chosen = config or MetaAnalysisConfig()
    coefficients, _ = fit_standardized_gee_models(summary, config=chosen)
    return coefficients


def fit_standardized_gee_models(
    summary: pd.DataFrame,
    *,
    config: MetaAnalysisConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen = config or MetaAnalysisConfig()
    return standardized_coefficients_prepared(
        prepare_tidy_summary(summary),
        alpha=chosen.alpha,
        max_condition_number=chosen.gee_max_condition_number,
    )


def difficulty_marginal_predictions(
    summary: pd.DataFrame,
    *,
    difficulty_grid: Sequence[float] | None = None,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    chosen, data = config or MetaAnalysisConfig(), prepare_tidy_summary(summary)
    grid = np.asarray(
        (
            np.linspace(0, 1, chosen.prediction_points)
            if difficulty_grid is None
            else difficulty_grid
        ),
        dtype=float,
    )
    if grid.ndim != 1 or not grid.size or not np.isfinite(grid).all():
        raise ValueError("difficulty_grid must be a finite one-dimensional sequence")
    if np.any((grid < 0) | (grid > 1)):
        raise ValueError("difficulty_grid must lie in [0, 1]")
    grid = np.unique(np.sort(grid))
    predictions, _, _, _ = marginal_predictions_prepared(
        data,
        grid=grid,
        alpha=chosen.alpha,
        max_condition_number=chosen.gee_max_condition_number,
        difficulty_spline_min_unique_values=(
            chosen.difficulty_spline_min_unique_values
        ),
        difficulty_spline_min_datasets=chosen.difficulty_spline_min_datasets,
        minimum_platform_datasets=chosen.platform_marginal_min_datasets,
    )
    return predictions


def platform_marginal_predictions(
    summary: pd.DataFrame,
    *,
    config: MetaAnalysisConfig | None = None,
) -> pd.DataFrame:
    chosen, data = config or MetaAnalysisConfig(), prepare_tidy_summary(summary)
    _, predictions, _, _ = marginal_predictions_prepared(
        data,
        grid=np.asarray([0.5]),
        alpha=chosen.alpha,
        max_condition_number=chosen.gee_max_condition_number,
        difficulty_spline_min_unique_values=(
            chosen.difficulty_spline_min_unique_values
        ),
        difficulty_spline_min_datasets=chosen.difficulty_spline_min_datasets,
        minimum_platform_datasets=chosen.platform_marginal_min_datasets,
    )
    return predictions


def fit_mixedlm_models(
    summary: pd.DataFrame,
    *,
    config: MetaAnalysisConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen = config or MetaAnalysisConfig()
    return fit_mixedlm_prepared(
        prepare_tidy_summary(summary),
        alpha=chosen.alpha,
        singular_variance_threshold=(chosen.mixedlm_singular_variance_threshold),
    )


def _input_sha256(data: pd.DataFrame) -> str:
    columns = list(REQUIRED_COLUMNS)
    columns.extend(
        column
        for column in (
            *INDEPENDENT_UNIT_COLUMNS,
            "uncertainty_unit",
            "uncertainty_reason",
        )
        if column in data
    )
    if "gap_variance" in data:
        columns.append("gap_variance")
    hashed = pd.util.hash_pandas_object(data[columns], index=False).to_numpy()
    digest = hashlib.sha256("\0".join(columns).encode())
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def _config_payload(config: MetaAnalysisConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _config_sha256(config: MetaAnalysisConfig) -> str:
    payload = json.dumps(
        _config_payload(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _threshold_payload(config: MetaAnalysisConfig) -> dict[str, Any]:
    payload = _config_payload(config)
    names = (
        "alpha",
        "ceiling_auc",
        "observation_count_thresholds",
        "independent_unit_thresholds",
        "minimum_repository_clusters_for_interval",
        "minimum_repository_clusters_for_robust_ci",
        "gee_max_condition_number",
        "difficulty_spline_min_unique_values",
        "difficulty_spline_min_datasets",
        "platform_marginal_min_datasets",
        "mixedlm_singular_variance_threshold",
    )
    return {name: payload[name] for name in names}


def _file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_file_sha256() -> dict[str, str]:
    public_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("meta_models.py").resolve(),
    )
    dependency_paths = (
        *public_paths,
        Path(__file__).with_name("meta_model_common.py").resolve(),
        Path(__file__).with_name("meta_preparation.py").resolve(),
        Path(__file__).with_name("meta_sensitivity.py").resolve(),
        Path(__file__).with_name("human_cohort_adjudication.py").resolve(),
    )
    dependency_hashes: dict[str, str] = {}
    for path in dependency_paths:
        value = _file_sha256(path)
        if value is None:
            raise RuntimeError(f"could not hash analysis code file: {path.name}")
        dependency_hashes[path.name] = value
    file_hashes = {path.name: dependency_hashes[path.name] for path in public_paths}
    combined = hashlib.sha256()
    for name in sorted(dependency_hashes):
        combined.update(name.encode())
        combined.update(b"\0")
        combined.update(dependency_hashes[name].encode())
        combined.update(b"\n")
    return {**file_hashes, "combined_sha256": combined.hexdigest()}


def _summary_source(
    summary: pd.DataFrame | str | Path,
) -> tuple[pd.DataFrame, Path | None]:
    if isinstance(summary, pd.DataFrame):
        return summary, None
    path = Path(summary)
    return pd.read_csv(path), path


def _analysis_manifest(
    *,
    input_path: Path | None,
    cohort: _ValidatedCohort | None,
    data: pd.DataFrame,
    config: MetaAnalysisConfig,
    generated_at_utc: str | None,
) -> dict[str, Any]:
    cohort_path = cohort.path if cohort is not None else None
    if cohort is not None and _validated_cohort(cohort.path).audit != cohort.audit:
        raise RuntimeError("audited cohort bundle changed during meta-analysis")
    code_hashes = _code_file_sha256()
    timestamp = generated_at_utc or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "schema_version": 2,
        "input_csv_name": input_path.name if input_path else None,
        "input_csv_sha256": _file_sha256(input_path),
        "cohort_csv_name": cohort_path.name if cohort_path else None,
        "cohort_csv_sha256": (
            cohort.audit["csv_sha256"] if cohort is not None else None
        ),
        "cohort_audit": _cohort_audit_payload(cohort),
        "canonical_input_table_sha256": _input_sha256(data),
        "meta_analysis_py_sha256": code_hashes["meta_analysis.py"],
        "code_file_sha256": code_hashes,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "config": _config_payload(config),
        "config_sha256": _config_sha256(config),
        "seed": config.seed,
        "generated_at_utc": timestamp,
        "estimands": {
            "platform_marginal_predictions": {
                "estimand": PLATFORM_MARGINAL_ESTIMAND,
                "reference_distribution": PLATFORM_REFERENCE_DISTRIBUTION,
                "adjusted_covariates": [
                    "difficulty",
                    "log2_n",
                    "log10_p_over_n",
                    "log2_imbalance",
                ],
                "inference": "delta_method_GEE_robust_sandwich",
                "unadjusted_fallback": False,
            },
            "cohort_subgroup_comparison": {
                "estimand": "mean_paired_gap",
                "inference": ("exploratory_repository_cluster_percentile_bootstrap"),
                "unresolved_policy": "exclude_from_both_subgroups",
            },
        },
        "model_specification": {
            "method_specific_moderators": ["difficulty"],
            "common_adjustment_terms": [
                "log2_n",
                "log10_p_over_n",
                "log2_imbalance",
                "platform_or_repository",
            ],
            "source_models": ("platform and repository adjusted in separate models"),
        },
        "uncertainty_scope": {
            "fixed_oof": (
                "paired sample-or-cluster bootstrap of fixed "
                "repeat-averaged OOF predictions"
            ),
            "full_pipeline_refit_uncertainty_estimated": False,
        },
    }


def _subset_audit(
    data: pd.DataFrame,
    cohort: _ValidatedCohort | None,
    config: MetaAnalysisConfig,
) -> pd.DataFrame:
    datasets = data.drop_duplicates("dataset_id")[
        [
            "dataset_id",
            "repository",
            "platform",
            "n_samples",
            "n_features",
            "n_pos",
            "n_neg",
            "minority_n",
            "baseline_l1_auc",
            "difficulty",
            "n_observations",
            "minority_observations",
            "n_independent_units",
            "minority_independent_units",
            "independent_unit_type",
            "minority_independent_units_reason",
            "independent_unit_metadata_available",
        ]
    ].copy()
    membership = _cohort_membership(data, cohort)
    datasets = datasets.merge(membership, on="dataset_id", how="left", validate="1:1")
    for total, minority in config.observation_count_thresholds:
        column = f"observation_count_n_ge_{total}_minority_observations_ge_{minority}"
        datasets[column] = (datasets.n_observations >= total) & (
            datasets.minority_observations >= minority
        )
    for total, minority in config.independent_unit_thresholds:
        column = f"independent_unit_n_ge_{total}_minority_units_ge_{minority}"
        datasets[column] = (
            datasets.independent_unit_metadata_available
            & datasets.n_independent_units.ge(total)
            & datasets.minority_independent_units.ge(minority)
        )
    ceiling = (
        data.assign(ceiling=data.fold_contained_auc >= config.ceiling_auc)
        .groupby("dataset_id", sort=True)["ceiling"]
        .agg(["any", "all"])
    )
    ceiling.columns = ["any_method_at_ceiling", "all_methods_at_ceiling"]
    return (
        datasets.merge(ceiling, left_on="dataset_id", right_index=True, validate="1:1")
        .sort_values("dataset_id")
        .reset_index(drop=True)
    )


def run_meta_analysis(
    summary: pd.DataFrame | str | Path,
    *,
    cohort: str | Path | None = None,
    config: MetaAnalysisConfig | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    source, input_path = _summary_source(summary)
    chosen, data = config or MetaAnalysisConfig(), prepare_tidy_summary(source)
    cohort_bundle = _validated_cohort(cohort) if cohort is not None else None
    grid = np.linspace(0, 1, chosen.prediction_points)
    predictions, platform_predictions, gee_coefficients, gee_diagnostics = (
        marginal_predictions_prepared(
            data,
            grid=grid,
            alpha=chosen.alpha,
            max_condition_number=chosen.gee_max_condition_number,
            difficulty_spline_min_unique_values=(
                chosen.difficulty_spline_min_unique_values
            ),
            difficulty_spline_min_datasets=(chosen.difficulty_spline_min_datasets),
            minimum_platform_datasets=chosen.platform_marginal_min_datasets,
        )
    )
    mixed_coefficients, mixed_diagnostics = fit_mixedlm_models(data, config=chosen)
    standardized_coefficients, standardized_diagnostics = fit_standardized_gee_models(
        data, config=chosen
    )
    uncertainty_protocol_present = "uncertainty_unit" in data
    n_fixed_oof_uncertainty_rows = int(
        data["gap_variance"].notna().sum() if "gap_variance" in data else 0
    )
    metadata = {
        "schema_version": 2,
        "seed": chosen.seed,
        "alpha": chosen.alpha,
        "bootstrap_replicates": chosen.bootstrap_replicates,
        "ceiling_auc": chosen.ceiling_auc,
        "difficulty_definition": ("1 - 2 * abs(baseline_l1_mean_fold_auc - 0.5)"),
        "input_sha256": _input_sha256(data),
        "n_rows": len(data),
        "n_datasets": data["dataset_id"].nunique(),
        "methods": sorted(data["method"].unique()),
        "repositories": sorted(data["repository"].unique()),
        "platforms": sorted(data["platform"].unique()),
        "loro_is_external_validation": False,
        "repository_cluster_bootstrap_scope": "exploratory",
        "fixed_oof_paired_sample_or_cluster_bootstrap_uncertainty_estimated": (
            uncertainty_protocol_present and n_fixed_oof_uncertainty_rows > 0
        ),
        "n_dataset_method_fixed_oof_uncertainty_estimates": (
            n_fixed_oof_uncertainty_rows
        ),
        "fixed_oof_uncertainty_scope": (
            "paired bootstrap of fixed repeat-averaged OOF predictions"
        ),
        "full_pipeline_refit_uncertainty_estimated": False,
        "full_pipeline_refit_uncertainty_scope": "not_estimated",
        "gee_covariance": "dataset-clustered Exchangeable robust sandwich",
        "gee_weighting": weighting_label(data),
        "platform_marginal_estimand": PLATFORM_MARGINAL_ESTIMAND,
        "platform_marginal_reference_distribution": (PLATFORM_REFERENCE_DISTRIBUTION),
        "platform_marginal_adjustment": (
            "difficulty|log2_n|log10_p_over_n|log2_imbalance"
        ),
        "cohort_subgroup_estimand": "mean_paired_gap",
        "cohort_unresolved_policy": "exclude_from_both_subgroups",
        "mixedlm_scope": (
            "dataset random-intercept sensitivity; repository/platform fixed "
            "in separate models"
        ),
        "method_specific_moderators": ["difficulty"],
        "common_adjustment_terms": [
            "log2_n",
            "log10_p_over_n",
            "log2_imbalance",
            "platform_or_repository",
        ],
        "thresholds": _threshold_payload(chosen),
        "config_sha256": _config_sha256(chosen),
        "cohort_audit": _cohort_audit_payload(cohort_bundle),
    }
    strict = (
        _strict_human_observational(data, cohort_bundle, config=chosen)
        if cohort_bundle is not None
        else pd.DataFrame(columns=BOOTSTRAP_COLUMNS)
    )
    manifest = _analysis_manifest(
        input_path=input_path,
        cohort=cohort_bundle,
        data=data,
        config=chosen,
        generated_at_utc=generated_at_utc,
    )
    return {
        "overall_gap": repository_cluster_bootstrap(data, config=chosen),
        "leave_one_repository_out": leave_one_repository_out(data),
        "sample_size_sensitivity": sample_size_sensitivity(data, config=chosen),
        "independent_unit_sensitivity": independent_unit_sensitivity(
            data, config=chosen
        ),
        "strict_human_observational": strict,
        "cohort_subgroup_comparison": _cohort_subgroup_comparison(
            data, cohort_bundle, config=chosen
        ),
        "ceiling_sensitivity": ceiling_sensitivity(data, config=chosen),
        "gee_coefficients": gee_coefficients,
        "gee_diagnostics": gee_diagnostics,
        "gee_standardized_coefficients": standardized_coefficients,
        "gee_standardized_diagnostics": standardized_diagnostics,
        "difficulty_marginal_predictions": predictions,
        "platform_marginal_predictions": platform_predictions,
        "mixedlm_coefficients": mixed_coefficients,
        "mixedlm_diagnostics": mixed_diagnostics,
        "subset_audit": _subset_audit(data, cohort_bundle, chosen),
        "analysis_manifest": manifest,
        "metadata": metadata,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_meta_analysis_bundle(
    bundle: Mapping[str, Any],
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write each table and refuse existing targets by default."""
    destination = Path(output_directory)
    paths = {
        key: destination
        / (f"{key}.json" if key in {"metadata", "analysis_manifest"} else f"{key}.csv")
        for key in bundle
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {len(existing)} existing meta-analysis outputs"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for key in sorted(bundle):
        value, path = bundle[key], paths[key]
        if key in {"metadata", "analysis_manifest"}:
            payload = (
                json.dumps(
                    value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
                )
                + "\n"
            ).encode()
        elif isinstance(value, pd.DataFrame):
            payload = value.to_csv(
                index=False, lineterminator="\n", float_format="%.12g"
            ).encode()
        else:
            raise TypeError(f"bundle item {key!r} must be a DataFrame or JSON metadata")
        _atomic_write(path, payload)
    return {key: paths[key] for key in sorted(paths)}


__all__ = [
    "MetaAnalysisConfig",
    "ceiling_sensitivity",
    "cohort_subgroup_comparison",
    "difficulty_marginal_predictions",
    "fit_gee_models",
    "fit_mixedlm_models",
    "fit_standardized_gee_models",
    "independent_unit_sensitivity",
    "leave_one_repository_out",
    "platform_marginal_predictions",
    "prepare_tidy_summary",
    "repository_cluster_bootstrap",
    "run_meta_analysis",
    "sample_size_sensitivity",
    "standardized_continuous_coefficients",
    "strict_human_observational",
    "write_meta_analysis_bundle",
]
