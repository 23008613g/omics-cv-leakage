"""Shared formula, weighting, coefficient, and design helpers for meta models."""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd


CONTINUOUS_COVARIATES = (
    "difficulty",
    "log2_n",
    "log10_p_over_n",
    "log2_imbalance",
)
GEE_MAX_CONDITION_NUMBER = 1e12


def _model_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.sort_values(["dataset_id", "method"], kind="stable").copy()
    for column in ("method", "platform", "repository"):
        frame[column] = pd.Categorical(
            frame[column], categories=sorted(frame[column].unique())
        )
    return frame


def _difficulty_term(
    data: pd.DataFrame,
    *,
    minimum_unique_values: int,
    minimum_datasets: int,
) -> tuple[str, str, str]:
    unique = data["difficulty"].nunique()
    n_datasets = data["dataset_id"].nunique()
    if unique >= minimum_unique_values and n_datasets >= minimum_datasets:
        term = (
            "bs(difficulty, df=3, degree=2, include_intercept=False, "
            "lower_bound=0, upper_bound=1)"
        )
        return term, "restricted_quadratic_bs_df3", ""
    return (
        "difficulty",
        "linear_fallback_insufficient_support",
        f"only {unique} unique difficulty values and {n_datasets} datasets; "
        f"at least {minimum_unique_values} values and {minimum_datasets} "
        "datasets required",
    )


def _formula(source: str, difficulty_term: str, *, interaction: bool) -> str:
    method = (
        f"C(method) * {difficulty_term}"
        if interaction
        else f"C(method) + {difficulty_term}"
    )
    return (
        f"gap ~ {method} + log2_n + log10_p_over_n + "
        f"log2_imbalance + C({source})"
    )


def _bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not finite.size:
        return output
    order = finite[np.argsort(values[finite], kind="stable")]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    output[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    return output


def _coefficient_rows(
    result: Any,
    *,
    model: str,
    alpha: float,
    fixed_only: bool = False,
) -> pd.DataFrame:
    params = result.fe_params if fixed_only else result.params
    errors = result.bse_fe if fixed_only else result.bse
    pvalues = result.pvalues.reindex(params.index)
    confidence = result.conf_int(alpha=alpha).loc[params.index]
    frame = pd.DataFrame(
        {
            "model": model,
            "term": params.index.astype(str),
            "estimate": params.to_numpy(float),
            "std_error": errors.reindex(params.index).to_numpy(float),
            "ci_low": confidence.iloc[:, 0].to_numpy(float),
            "ci_high": confidence.iloc[:, 1].to_numpy(float),
            "p_value": pvalues.to_numpy(float),
        }
    )
    frame["p_fdr_bh"] = _bh_fdr(frame["p_value"])
    frame["term_scope"] = frame["term"].map(_term_scope)
    return frame


def _term_scope(term: str) -> str:
    if ":" in term:
        return (
            "method_specific_difficulty_moderation"
            if "difficulty" in term
            else "unexpected_interaction"
        )
    if term.startswith("C(method)"):
        return "method_main_effect"
    if term.startswith(("C(platform)", "C(repository)")):
        return "common_source_main_effect"
    if term in {"log2_n", "log10_p_over_n", "log2_imbalance"}:
        return "common_continuous_slope"
    if "difficulty" in term:
        return "difficulty_main_or_basis_component"
    if term == "Intercept":
        return "intercept"
    return "other_model_term"


def _weight_values(data: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    complete = (
        "gap_variance" in data
        and data["gap_variance"].notna().all()
        and (data["gap_variance"] > 0).all()
    )
    if complete:
        weights = 1 / data["gap_variance"].to_numpy(float)
        return weights / weights.mean(), "inverse_gap_variance"
    return None, "unweighted_no_complete_oof_variance"


def weighting_label(data: pd.DataFrame) -> str:
    return _weight_values(data)[1]


def _repository_determines_platform(data: pd.DataFrame) -> bool:
    return bool(
        data.groupby("repository", observed=True)["platform"].nunique().max() == 1
    )


def _warnings_text(records: Sequence[warnings.WarningMessage]) -> str:
    messages = {f"{record.category.__name__}: {record.message}" for record in records}
    return " | ".join(sorted(messages))


def _finite_inference(frame: pd.DataFrame) -> bool:
    columns = ("estimate", "std_error", "ci_low", "ci_high", "p_value")
    return bool(np.isfinite(frame[list(columns)].to_numpy(float)).all())


def _design_audit(
    exog: np.ndarray,
    *,
    max_condition_number: float,
) -> dict[str, Any]:
    design_rank = int(np.linalg.matrix_rank(exog))
    design_columns = int(exog.shape[1])
    condition_number = float(np.linalg.cond(exog))
    full_rank = design_rank == design_columns
    condition_acceptable = (
        np.isfinite(condition_number) and condition_number <= max_condition_number
    )
    return {
        "design_rank": design_rank,
        "design_columns": design_columns,
        "design_full_rank": full_rank,
        "design_condition_number": condition_number,
        "condition_number_limit": max_condition_number,
        "condition_number_acceptable": condition_acceptable,
    }


def _fixed_oof_uncertainty_available(data: pd.DataFrame) -> bool:
    return bool(
        "uncertainty_unit" in data
        and "gap_variance" in data
        and data["gap_variance"].notna().any()
    )


def _common_adjustment_terms(source: str) -> str:
    return "|".join(
        (
            "log2_n",
            "log10_p_over_n",
            "log2_imbalance",
            source,
        )
    )
