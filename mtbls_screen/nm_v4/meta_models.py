"""Regression models and marginal predictions for the meta-analysis."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.genmod.cov_struct import Exchangeable

from .meta_model_common import (
    CONTINUOUS_COVARIATES,
    GEE_MAX_CONDITION_NUMBER,
    _bh_fdr,
    _coefficient_rows,
    _common_adjustment_terms,
    _design_audit,
    _difficulty_term,
    _finite_inference,
    _fixed_oof_uncertainty_available,
    _formula,
    _model_frame,
    _repository_determines_platform,
    _term_scope,
    _warnings_text,
    _weight_values,
    weighting_label,
)
PLATFORM_MARGINAL_ESTIMAND = "covariate_adjusted_marginal_mean_gap"
PLATFORM_REFERENCE_DISTRIBUTION = (
    "empirical_all_dataset_distribution_equal_dataset_weight"
)
PLATFORM_MARGINAL_COLUMNS = (
    "model",
    "method",
    "platform",
    "estimand",
    "estimate",
    "std_error",
    "ci_low",
    "ci_high",
    "alpha",
    "robust_ci",
    "n_dataset",
    "n_platform_datasets",
    "weighting",
    "model_status",
    "source_model_status",
    "audit_reason",
    "reference_distribution",
    "adjusted_covariates",
    "marginalization",
    "ci_method",
    "unadjusted_fallback_used",
    "source_model_formula",
    "moderation_scope",
    "method_specific_moderators",
    "common_adjustment_terms",
)


def fit_gee_prepared(
    data: pd.DataFrame,
    *,
    alpha: float,
    max_condition_number: float = GEE_MAX_CONDITION_NUMBER,
    difficulty_spline_min_unique_values: int = 6,
    difficulty_spline_min_datasets: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit the two pre-specified GEE models to already validated data."""
    model_data = _model_frame(data)
    difficulty, basis, fallback = _difficulty_term(
        model_data,
        minimum_unique_values=difficulty_spline_min_unique_values,
        minimum_datasets=difficulty_spline_min_datasets,
    )
    weights, weighting = _weight_values(model_data)
    coefficient_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for source, name in (
        ("platform", "platform_adjusted"),
        ("repository", "repository_adjusted"),
    ):
        formula = _formula(source, difficulty, interaction=True)
        common = {
            "model": name,
            "formula": formula,
            "difficulty_basis": basis,
            "difficulty_fallback_reason": fallback,
            "covariance_structure": "Exchangeable",
            "cluster_unit": "dataset_id",
            "weighting": weighting,
            "repository_determines_platform": _repository_determines_platform(
                model_data
            ),
            "n_rows": len(model_data),
            "n_datasets": model_data["dataset_id"].nunique(),
            "n_methods": model_data["method"].nunique(),
            "n_repositories": model_data["repository"].nunique(),
            "n_platforms": model_data["platform"].nunique(),
            "fixed_oof_paired_bootstrap_uncertainty_available": (
                _fixed_oof_uncertainty_available(model_data)
            ),
            "full_pipeline_refit_uncertainty_estimated": False,
            "moderation_scope": "difficulty_by_method_only",
            "method_specific_moderators": "difficulty",
            "common_adjustment_terms": _common_adjustment_terms(source),
        }
        records: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                model = smf.gee(
                    formula,
                    groups="dataset_id",
                    data=model_data,
                    cov_struct=Exchangeable(),
                    family=sm.families.Gaussian(),
                    weights=weights,
                )
                result = model.fit(maxiter=200, cov_type="robust")
                coefficients = _coefficient_rows(result, model=name, alpha=alpha)
            warning_text = _warnings_text(records)
            design = _design_audit(
                result.model.exog,
                max_condition_number=max_condition_number,
            )
            valid = (
                bool(result.converged)
                and _finite_inference(coefficients)
                and bool(design["design_full_rank"])
                and bool(design["condition_number_acceptable"])
            )
            status = "ok" if valid else "model_invalid"
            invalid_reason = (
                ""
                if valid
                else (
                    "model did not converge, returned non-finite inference, or "
                    "failed design rank/condition diagnostics"
                )
            )
            coefficients["n_rows"] = len(model_data)
            coefficients["n_datasets"] = model_data["dataset_id"].nunique()
            coefficients["fit_status"] = status
            coefficients["fit_warnings"] = warning_text
            coefficient_frames.append(coefficients)
            if valid:
                fitted[name] = result
            sizes = model_data.groupby("dataset_id", observed=True).size()
            diagnostics.append(
                {
                    **common,
                    "status": status,
                    "fit_status": status,
                    "converged": bool(result.converged),
                    "robust_covariance": True,
                    "cluster_size_min": int(sizes.min()),
                    "cluster_size_median": float(sizes.median()),
                    "cluster_size_max": int(sizes.max()),
                    **design,
                    "fit_warnings": warning_text,
                    "warning_count": len(records),
                    "error": invalid_reason,
                }
            )
        except Exception as exc:
            diagnostics.append(
                {
                    **common,
                    "status": "error",
                    "fit_status": "error",
                    "converged": False,
                    "robust_covariance": True,
                    "cluster_size_min": np.nan,
                    "cluster_size_median": np.nan,
                    "cluster_size_max": np.nan,
                    "design_rank": np.nan,
                    "design_columns": np.nan,
                    "design_full_rank": False,
                    "design_condition_number": np.nan,
                    "condition_number_limit": max_condition_number,
                    "condition_number_acceptable": False,
                    "fit_warnings": _warnings_text(records),
                    "warning_count": len(records),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    coefficient_columns = (
        "model",
        "term",
        "estimate",
        "std_error",
        "ci_low",
        "ci_high",
        "p_value",
        "p_fdr_bh",
        "n_rows",
        "n_datasets",
        "fit_status",
        "fit_warnings",
        "term_scope",
    )
    coefficients = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame(columns=coefficient_columns)
    )
    return (
        coefficients.sort_values(["model", "term"], kind="stable").reset_index(
            drop=True
        ),
        pd.DataFrame(diagnostics).sort_values("model").reset_index(drop=True),
        fitted,
    )


def standardized_coefficients_prepared(
    data: pd.DataFrame,
    *,
    alpha: float,
    max_condition_number: float = GEE_MAX_CONDITION_NUMBER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit additive sensitivity models on standardized dataset covariates."""
    frame = _model_frame(data)
    dataset_covariates = frame.drop_duplicates("dataset_id")
    standard_deviations: dict[str, float] = {}
    constant_covariates: list[str] = []
    for column in CONTINUOUS_COVARIATES:
        mean = float(dataset_covariates[column].mean())
        sd = float(dataset_covariates[column].std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            constant_covariates.append(column)
            continue
        frame[f"{column}_z"] = (frame[column] - mean) / sd
        standard_deviations[column] = sd
    weights, weighting = _weight_values(frame)
    outputs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    z_terms = " + ".join(f"{column}_z" for column in CONTINUOUS_COVARIATES)
    for source, name in (
        ("platform", "platform_adjusted_standardized"),
        ("repository", "repository_adjusted_standardized"),
    ):
        formula = f"gap ~ C(method) + {z_terms} + C({source})"
        records: list[warnings.WarningMessage] = []
        common = {
            "model": name,
            "formula": formula,
            "covariance_structure": "Exchangeable",
            "cluster_unit": "dataset_id",
            "weighting": weighting,
            "n_rows": len(frame),
            "n_datasets": frame["dataset_id"].nunique(),
            "moderation_scope": (
                "additive_common_slope_sensitivity; primary_model_uses_"
                "difficulty_by_method_only"
            ),
            "method_specific_moderators": "difficulty_in_primary_model_only",
            "common_adjustment_terms": _common_adjustment_terms(source),
            "full_pipeline_refit_uncertainty_estimated": False,
        }
        if constant_covariates:
            diagnostics.append(
                {
                    **common,
                    "status": "model_invalid",
                    "fit_status": "model_invalid",
                    "converged": False,
                    "robust_covariance": True,
                    "design_rank": np.nan,
                    "design_columns": np.nan,
                    "design_full_rank": False,
                    "design_condition_number": np.nan,
                    "condition_number_limit": max_condition_number,
                    "condition_number_acceptable": False,
                    "fit_warnings": "",
                    "warning_count": 0,
                    "error": (
                        "cannot standardize constant covariate "
                        + "|".join(constant_covariates)
                    ),
                }
            )
            continue
        try:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                result = smf.gee(
                    formula,
                    groups="dataset_id",
                    data=frame,
                    cov_struct=Exchangeable(),
                    family=sm.families.Gaussian(),
                    weights=weights,
                ).fit(maxiter=200, cov_type="robust")
                all_coefficients = _coefficient_rows(result, model=name, alpha=alpha)
            warning_text = _warnings_text(records)
            design = _design_audit(
                result.model.exog,
                max_condition_number=max_condition_number,
            )
            valid = (
                bool(result.converged)
                and _finite_inference(all_coefficients)
                and bool(design["design_full_rank"])
                and bool(design["condition_number_acceptable"])
            )
            status = "ok" if valid else "model_invalid"
            error = (
                ""
                if valid
                else (
                    "model did not converge, returned non-finite inference, or "
                    "failed design rank/condition diagnostics"
                )
            )
            diagnostics.append(
                {
                    **common,
                    **design,
                    "status": status,
                    "fit_status": status,
                    "converged": bool(result.converged),
                    "robust_covariance": True,
                    "fit_warnings": warning_text,
                    "warning_count": len(records),
                    "error": error,
                }
            )
            if valid:
                output = all_coefficients[
                    all_coefficients.term.isin(
                        [f"{item}_z" for item in CONTINUOUS_COVARIATES]
                    )
                ].copy()
                output["term"] = output["term"].str.removesuffix("_z")
                output["term_scope"] = output["term"].map(_term_scope)
                output["input_standard_deviation"] = output["term"].map(
                    standard_deviations
                )
                output["scale"] = "per_input_standard_deviation"
                output["weighting"] = weighting
                output["fit_status"] = status
                output["fit_warnings"] = warning_text
                output["coefficient_scope"] = (
                    "common_slope_additive_standardized_sensitivity"
                )
                output["method_specific_coefficient"] = False
                outputs.append(output)
        except Exception as exc:
            diagnostics.append(
                {
                    **common,
                    "status": "error",
                    "fit_status": "error",
                    "converged": False,
                    "robust_covariance": True,
                    "design_rank": np.nan,
                    "design_columns": np.nan,
                    "design_full_rank": False,
                    "design_condition_number": np.nan,
                    "condition_number_limit": max_condition_number,
                    "condition_number_acceptable": False,
                    "fit_warnings": _warnings_text(records),
                    "warning_count": len(records),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    output_columns = (
        "model",
        "term",
        "estimate",
        "std_error",
        "ci_low",
        "ci_high",
        "p_value",
        "p_fdr_bh",
        "term_scope",
        "input_standard_deviation",
        "scale",
        "weighting",
        "fit_status",
        "fit_warnings",
        "coefficient_scope",
        "method_specific_coefficient",
    )
    combined = (
        pd.concat(outputs, ignore_index=True)
        if outputs
        else pd.DataFrame(columns=output_columns)
    )
    if not combined.empty:
        combined["p_fdr_bh"] = combined.groupby("model", sort=True)[
            "p_value"
        ].transform(lambda values: _bh_fdr(values.to_numpy(float)))
    return (
        combined.sort_values(["model", "term"], kind="stable").reset_index(drop=True),
        pd.DataFrame(diagnostics).sort_values("model").reset_index(drop=True),
    )


def _prediction_frame(
    result: Any,
    data: pd.DataFrame,
    *,
    model: str,
    grid: np.ndarray,
    alpha: float,
    difficulty_basis: str,
) -> pd.DataFrame:
    datasets = data.drop_duplicates("dataset_id").copy()
    design_info = result.model.data.design_info
    covariance = np.asarray(result.cov_params(), dtype=float)
    parameter_names = list(result.params.index)
    critical = float(norm.ppf(1 - alpha / 2))
    rows: list[dict[str, Any]] = []
    for method in sorted(data["method"].astype(str).unique()):
        for difficulty in grid:
            new = datasets.copy()
            new["method"] = method
            new["difficulty"] = float(difficulty)
            design = patsy.build_design_matrices(
                [design_info], new, return_type="dataframe"
            )[0].reindex(columns=parameter_names, fill_value=0.0)
            average = design.to_numpy(float).mean(axis=0)
            estimate = float(average @ result.params.to_numpy(float))
            variance = float(average @ covariance @ average)
            standard_error = float(np.sqrt(max(variance, 0.0)))
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "difficulty": float(difficulty),
                    "predicted_gap": estimate,
                    "std_error": standard_error,
                    "ci_low": estimate - critical * standard_error,
                    "ci_high": estimate + critical * standard_error,
                    "alpha": alpha,
                    "difficulty_basis": difficulty_basis,
                    "marginalization": "empirical_dataset_covariate_distribution",
                    "n_datasets": len(datasets),
                    "moderation_scope": "difficulty_by_method_only",
                    "method_specific_moderators": "difficulty",
                    "common_adjustment_terms": (
                        "log2_n|log10_p_over_n|log2_imbalance|platform_or_repository"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _diagnostic_reason(record: dict[str, Any]) -> str:
    parts = [str(record.get(key, "")).strip() for key in ("error", "fit_warnings")]
    return " | ".join(part for part in parts if part)


def _platform_marginal_frame(
    result: Any | None,
    data: pd.DataFrame,
    *,
    diagnostic: dict[str, Any],
    alpha: float,
    minimum_platform_datasets: int,
) -> pd.DataFrame:
    """Standardize every method-platform cell to one common dataset mix."""
    datasets = data.drop_duplicates("dataset_id").copy()
    methods = sorted(data["method"].astype(str).unique())
    platforms = sorted(data["platform"].astype(str).unique())
    reference = "|".join(CONTINUOUS_COVARIATES)
    source_status = str(diagnostic.get("status", "error"))
    source_reason = _diagnostic_reason(diagnostic)
    weighting = str(diagnostic.get("weighting", weighting_label(data)))
    formula = str(diagnostic.get("formula", ""))
    critical = float(norm.ppf(1 - alpha / 2))
    rows: list[dict[str, Any]] = []
    for method in methods:
        for platform in platforms:
            support = int((datasets["platform"].astype(str) == platform).sum())
            status, reason = source_status, source_reason
            estimate = standard_error = ci_low = ci_high = np.nan
            robust_ci = False
            if (
                result is not None
                and source_status == "ok"
                and support >= minimum_platform_datasets
            ):
                try:
                    new = datasets.copy()
                    new["method"] = method
                    new["platform"] = platform
                    design = patsy.build_design_matrices(
                        [result.model.data.design_info],
                        new,
                        return_type="dataframe",
                    )[0].reindex(columns=result.params.index, fill_value=0.0)
                    average = design.to_numpy(float).mean(axis=0)
                    covariance = np.asarray(result.cov_params(), dtype=float)
                    estimate = float(average @ result.params.to_numpy(float))
                    variance = float(average @ covariance @ average)
                    standard_error = float(np.sqrt(max(variance, 0.0)))
                    ci_low = estimate - critical * standard_error
                    ci_high = estimate + critical * standard_error
                    if np.isfinite([estimate, standard_error, ci_low, ci_high]).all():
                        robust_ci = True
                        status, reason = "ok", ""
                    else:
                        status = "model_invalid"
                        reason = "non-finite marginal estimate or robust CI"
                except Exception as exc:
                    status = "prediction_error"
                    reason = f"{type(exc).__name__}: {exc}"
            elif result is not None and source_status == "ok":
                status = "insufficient_platform_support"
                reason = (
                    f"fewer than {minimum_platform_datasets} datasets observed "
                    "for this platform"
                )
            elif not reason:
                reason = "valid platform-adjusted GEE result unavailable"
            rows.append(
                {
                    "model": "platform_adjusted",
                    "method": method,
                    "platform": platform,
                    "estimand": PLATFORM_MARGINAL_ESTIMAND,
                    "estimate": estimate,
                    "std_error": standard_error,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "alpha": alpha,
                    "robust_ci": robust_ci,
                    "n_dataset": len(datasets),
                    "n_platform_datasets": support,
                    "weighting": weighting,
                    "model_status": status,
                    "source_model_status": source_status,
                    "audit_reason": reason,
                    "reference_distribution": PLATFORM_REFERENCE_DISTRIBUTION,
                    "adjusted_covariates": reference,
                    "marginalization": "average_model_matrix_delta_method",
                    "ci_method": "delta_method_GEE_robust_sandwich",
                    "unadjusted_fallback_used": False,
                    "source_model_formula": formula,
                    "moderation_scope": "difficulty_by_method_only",
                    "method_specific_moderators": "difficulty",
                    "common_adjustment_terms": _common_adjustment_terms("platform"),
                }
            )
    return (
        pd.DataFrame(rows, columns=PLATFORM_MARGINAL_COLUMNS)
        .sort_values(["method", "platform"], kind="stable")
        .reset_index(drop=True)
    )


def marginal_predictions_prepared(
    data: pd.DataFrame,
    *,
    grid: np.ndarray,
    alpha: float,
    max_condition_number: float = GEE_MAX_CONDITION_NUMBER,
    difficulty_spline_min_unique_values: int = 6,
    difficulty_spline_min_datasets: int = 12,
    minimum_platform_datasets: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficients, diagnostics, fitted = fit_gee_prepared(
        data,
        alpha=alpha,
        max_condition_number=max_condition_number,
        difficulty_spline_min_unique_values=(difficulty_spline_min_unique_values),
        difficulty_spline_min_datasets=difficulty_spline_min_datasets,
    )
    model_data = _model_frame(data)
    frames = []
    for record in diagnostics.to_dict(orient="records"):
        result = fitted.get(record["model"])
        if result is not None:
            frames.append(
                _prediction_frame(
                    result,
                    model_data,
                    model=record["model"],
                    grid=grid,
                    alpha=alpha,
                    difficulty_basis=record["difficulty_basis"],
                )
            )
    predictions = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["model", "method", "difficulty"], kind="stable")
        .reset_index(drop=True)
        if frames
        else pd.DataFrame()
    )
    platform_record = (
        diagnostics.loc[diagnostics.model == "platform_adjusted"].iloc[0].to_dict()
    )
    platform_predictions = _platform_marginal_frame(
        fitted.get("platform_adjusted"),
        model_data,
        diagnostic=platform_record,
        alpha=alpha,
        minimum_platform_datasets=minimum_platform_datasets,
    )
    return predictions, platform_predictions, coefficients, diagnostics


def fit_mixedlm_prepared(
    data: pd.DataFrame,
    *,
    alpha: float,
    singular_variance_threshold: float = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dataset-random-intercept sensitivity with separate source fixed effects."""
    frame = _model_frame(data)
    coefficients: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    note = (
        "Repository/platform are fixed effects in separate models; repository "
        "is not a random effect because four repositories cannot support a "
        "stable repository-level variance estimate."
    )
    for source, name in (
        ("platform", "mixedlm_platform_fixed"),
        ("repository", "mixedlm_repository_fixed"),
    ):
        formula = _formula(source, "difficulty", interaction=False)
        records: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                result = smf.mixedlm(
                    formula,
                    data=frame,
                    groups=frame["dataset_id"],
                    re_formula="1",
                ).fit(reml=False, method="lbfgs", maxiter=500, disp=False)
                output = _coefficient_rows(
                    result, model=name, alpha=alpha, fixed_only=True
                )
            warning_text = _warnings_text(records)
            variance = float(np.asarray(result.cov_re)[0, 0])
            singular = (
                not np.isfinite(variance) or variance < singular_variance_threshold
            )
            finite = _finite_inference(output)
            if not bool(result.converged) or not finite:
                status = "model_invalid"
            else:
                status = "singular" if singular else "ok"
            output["fit_status"] = status
            output["fit_warnings"] = warning_text
            coefficients.append(output)
            diagnostics.append(
                {
                    "model": name,
                    "formula": formula,
                    "status": status,
                    "fit_status": status,
                    "converged": bool(result.converged),
                    "grouping_factor": "dataset_id",
                    "random_intercept": True,
                    "random_intercept_variance": variance,
                    "singular_random_effect": singular,
                    "repository_random_effect_estimated": False,
                    "singular_variance_threshold": singular_variance_threshold,
                    "moderation_scope": "additive_sensitivity_no_interactions",
                    "method_specific_moderators": "none_in_mixedlm_sensitivity",
                    "common_adjustment_terms": _common_adjustment_terms(source),
                    "n_rows": len(frame),
                    "n_datasets": frame["dataset_id"].nunique(),
                    "fit_warnings": warning_text,
                    "warning_count": len(records),
                    "warning": note,
                    "error": (
                        ""
                        if status in {"ok", "singular"}
                        else "model did not converge or returned non-finite inference"
                    ),
                }
            )
        except Exception as exc:
            warning_text = _warnings_text(records)
            diagnostics.append(
                {
                    "model": name,
                    "formula": formula,
                    "status": "error",
                    "fit_status": "error",
                    "converged": False,
                    "grouping_factor": "dataset_id",
                    "random_intercept": True,
                    "random_intercept_variance": np.nan,
                    "singular_random_effect": True,
                    "repository_random_effect_estimated": False,
                    "singular_variance_threshold": singular_variance_threshold,
                    "moderation_scope": "additive_sensitivity_no_interactions",
                    "method_specific_moderators": "none_in_mixedlm_sensitivity",
                    "common_adjustment_terms": _common_adjustment_terms(source),
                    "n_rows": len(frame),
                    "n_datasets": frame["dataset_id"].nunique(),
                    "fit_warnings": warning_text,
                    "warning_count": len(records),
                    "warning": note,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    columns = (
        "model",
        "term",
        "estimate",
        "std_error",
        "ci_low",
        "ci_high",
        "p_value",
        "p_fdr_bh",
        "fit_status",
        "fit_warnings",
    )
    coefficient_frame = (
        pd.concat(coefficients, ignore_index=True)
        if coefficients
        else pd.DataFrame(columns=columns)
    )
    return (
        coefficient_frame.sort_values(["model", "term"], kind="stable").reset_index(
            drop=True
        ),
        pd.DataFrame(diagnostics).sort_values("model").reset_index(drop=True),
    )


__all__ = [
    "fit_gee_prepared",
    "fit_mixedlm_prepared",
    "marginal_predictions_prepared",
    "PLATFORM_MARGINAL_ESTIMAND",
    "PLATFORM_REFERENCE_DISTRIBUTION",
    "standardized_coefficients_prepared",
    "weighting_label",
]
