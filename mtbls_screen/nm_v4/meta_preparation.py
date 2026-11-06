"""Input validation and deterministic covariate preparation for meta-analysis."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


ALIASES = {
    "dataset_id": ("id",),
    "repository": ("source",),
    "platform": ("assay_platform", "plat"),
    "n_features": ("n_features_raw",),
    "baseline_l1_auc": ("baseline_l1", "baseline_mean_fold_auc"),
    "gap_variance": ("oof_gap_variance",),
    "gap_standard_error": ("oof_gap_standard_error",),
}
REQUIRED_COLUMNS = (
    "dataset_id",
    "method",
    "pre_split_auc",
    "fold_contained_auc",
    "gap",
    "baseline_l1_auc",
    "repository",
    "platform",
    "n_samples",
    "n_features",
    "n_pos",
    "n_neg",
)
INDEPENDENT_UNIT_COLUMNS = (
    "n_observations",
    "minority_observations",
    "n_independent_units",
    "minority_independent_units",
    "independent_unit_type",
    "minority_independent_units_reason",
)


def _copy_alias(frame: pd.DataFrame, target: str, aliases: Sequence[str]) -> None:
    if target in frame.columns:
        return
    source = next((name for name in aliases if name in frame.columns), None)
    if source is not None:
        frame[target] = frame[source]


def _require_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must be numeric") from exc
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite values")


def _validate_dataset_invariants(frame: pd.DataFrame) -> None:
    fields = (
        "repository",
        "platform",
        "n_samples",
        "n_features",
        "n_pos",
        "n_neg",
        "baseline_l1_auc",
    )
    for field in fields:
        counts = frame.groupby("dataset_id", sort=True)[field].nunique(dropna=False)
        if (counts > 1).any():
            bad = counts[counts > 1].index[0]
            raise ValueError(f"dataset-level field {field!r} varies within {bad!r}")


def _prepare_independent_unit_fields(frame: pd.DataFrame) -> None:
    present = [column in frame for column in INDEPENDENT_UNIT_COLUMNS]
    if any(present) and not all(present):
        raise ValueError(
            "independent-unit fields must be supplied together as one audited set"
        )
    if not any(present):
        frame["n_observations"] = frame["n_samples"]
        frame["minority_observations"] = frame[["n_pos", "n_neg"]].min(axis=1)
        frame["n_independent_units"] = np.nan
        frame["minority_independent_units"] = np.nan
        frame["independent_unit_type"] = "unavailable"
        frame["minority_independent_units_reason"] = (
            "independent_unit_metadata_not_supplied"
        )
        frame["independent_unit_metadata_available"] = False
        return

    if "independent_unit_metadata_available" in frame:
        availability = frame["independent_unit_metadata_available"]
        if not availability.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError("independent_unit_metadata_available must be boolean")
        if not bool(availability.any()):
            if (
                frame["n_independent_units"].notna().any()
                or frame["minority_independent_units"].notna().any()
                or not frame["independent_unit_type"].eq("unavailable").all()
                or not frame["minority_independent_units_reason"]
                .eq("independent_unit_metadata_not_supplied")
                .all()
            ):
                raise ValueError(
                    "unavailable independent-unit metadata has inconsistent values"
                )
            return
        if not bool(availability.all()):
            raise ValueError(
                "independent-unit metadata availability cannot vary within a table"
            )

    numeric = (
        "n_observations",
        "minority_observations",
        "n_independent_units",
    )
    _require_numeric(frame, numeric)
    frame["minority_independent_units"] = pd.to_numeric(
        frame["minority_independent_units"], errors="coerce"
    )
    count_columns = (
        "n_observations",
        "minority_observations",
        "n_independent_units",
        "minority_independent_units",
    )
    for column in count_columns:
        values = frame[column].dropna().to_numpy(float)
        if values.size and not np.equal(values, np.floor(values)).all():
            raise ValueError("independent-unit fields must contain integer counts")
    unit_type = frame["independent_unit_type"].astype(str).str.strip()
    reason = frame["minority_independent_units_reason"].astype(str).str.strip()
    if not unit_type.isin({"observation", "group"}).all():
        raise ValueError("independent_unit_type must be observation or group")
    if reason.eq("").any():
        raise ValueError("minority_independent_units_reason must be non-empty")
    if not (frame["n_observations"] == frame["n_samples"]).all():
        raise ValueError("n_observations must equal n_samples")
    expected_minority = frame[["n_pos", "n_neg"]].min(axis=1)
    if not (frame["minority_observations"] == expected_minority).all():
        raise ValueError("minority_observations must equal min(n_pos, n_neg)")
    if (frame["n_independent_units"] <= 0).any() or (
        frame["n_independent_units"] > frame["n_observations"]
    ).any():
        raise ValueError("n_independent_units must lie in [1, n_observations]")
    finite_minority = frame["minority_independent_units"].notna()
    if (frame.loc[finite_minority, "minority_independent_units"] <= 0).any() or (
        frame.loc[finite_minority, "minority_independent_units"]
        > frame.loc[finite_minority, "n_independent_units"]
    ).any():
        raise ValueError(
            "minority_independent_units must lie in [1, n_independent_units]"
        )
    observation = unit_type == "observation"
    if (
        not (
            frame.loc[observation, "n_independent_units"]
            == frame.loc[observation, "n_observations"]
        ).all()
        or not (
            frame.loc[observation, "minority_independent_units"]
            == frame.loc[observation, "minority_observations"]
        ).all()
    ):
        raise ValueError(
            "observation independent-unit counts must equal observation counts"
        )
    mixed = reason.eq("mixed_label_groups_no_unique_minority_unit_definition")
    expected_reasons = (
        unit_type.eq("observation") & reason.eq("row_level_no_groups")
    ) | (
        unit_type.eq("group")
        & reason.isin(
            {
                "constant_label_groups",
                "mixed_label_groups_no_unique_minority_unit_definition",
            }
        )
    )
    if not expected_reasons.all():
        raise ValueError(
            "independent-unit type and minority-unit audit reason disagree"
        )
    if not frame.loc[mixed, "minority_independent_units"].isna().all():
        raise ValueError(
            "mixed-label groups require missing minority_independent_units"
        )
    missing_minority = frame["minority_independent_units"].isna()
    if not (missing_minority == mixed).all():
        raise ValueError(
            "missing minority independent-unit counts require the audited "
            "mixed-label-group reason"
        )
    frame["independent_unit_type"] = unit_type
    frame["minority_independent_units_reason"] = reason
    frame["independent_unit_metadata_available"] = True


def prepare_tidy_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Validate and derive covariates without mutating the caller's table."""
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        raise ValueError("summary must be a non-empty pandas DataFrame")
    data = summary.copy(deep=True)
    for target, aliases in ALIASES.items():
        _copy_alias(data, target, aliases)
    if "gap" not in data and {"pre_split_auc", "fold_contained_auc"} <= set(data):
        data["gap"] = data["pre_split_auc"] - data["fold_contained_auc"]
    missing = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"summary is missing required columns: {missing}")

    for column in ("dataset_id", "method", "repository", "platform"):
        data[column] = data[column].astype(str).str.strip()
        if data[column].eq("").any():
            raise ValueError(f"{column} must be non-empty")
    if data.duplicated(["dataset_id", "method"]).any():
        raise ValueError("each (dataset_id, method) pair must be unique")

    numeric = (
        "pre_split_auc",
        "fold_contained_auc",
        "gap",
        "baseline_l1_auc",
        "n_samples",
        "n_features",
        "n_pos",
        "n_neg",
    )
    _require_numeric(data, numeric)
    for column in ("pre_split_auc", "fold_contained_auc", "baseline_l1_auc"):
        if not data[column].between(0, 1).all():
            raise ValueError(f"{column} must lie in [0, 1]")
    paired = data["pre_split_auc"] - data["fold_contained_auc"]
    if not np.allclose(data["gap"], paired, atol=1e-8, rtol=1e-8):
        raise ValueError("gap must equal pre_split_auc - fold_contained_auc")
    if (data[["n_samples", "n_features", "n_pos", "n_neg"]] <= 0).any().any():
        raise ValueError("sample, feature and class counts must be positive")
    if ((data["n_pos"] + data["n_neg"]) > data["n_samples"]).any():
        raise ValueError("n_pos + n_neg cannot exceed n_samples")
    _validate_dataset_invariants(data)
    _prepare_independent_unit_fields(data)
    for field in (*INDEPENDENT_UNIT_COLUMNS, "independent_unit_metadata_available"):
        counts = data.groupby("dataset_id", sort=True)[field].nunique(dropna=False)
        if (counts > 1).any():
            bad = counts[counts > 1].index[0]
            raise ValueError(
                f"dataset-level independent-unit field {field!r} varies within {bad!r}"
            )

    if "gap_variance" not in data and "gap_standard_error" in data:
        data["gap_variance"] = (
            pd.to_numeric(data["gap_standard_error"], errors="coerce") ** 2
        )
    if "gap_variance" in data:
        data["gap_variance"] = pd.to_numeric(data["gap_variance"], errors="coerce")
        finite = data["gap_variance"].notna()
        if (data.loc[finite, "gap_variance"] <= 0).any():
            raise ValueError("finite gap_variance values must be positive")

    data["minority_n"] = data[["n_pos", "n_neg"]].min(axis=1)
    data["imbalance_ratio"] = data[["n_pos", "n_neg"]].max(axis=1) / data["minority_n"]
    data["difficulty"] = 1 - 2 * np.abs(data["baseline_l1_auc"] - 0.5)
    data["log2_n"] = np.log2(data["n_samples"])
    data["log10_p_over_n"] = np.log10(data["n_features"] / data["n_samples"])
    data["log2_imbalance"] = np.log2(data["imbalance_ratio"])
    data["headroom"] = 1 - data["fold_contained_auc"]
    data["headroom_adjusted_gap"] = np.where(
        data["headroom"] > 0, data["gap"] / data["headroom"], np.nan
    )
    return data.sort_values(["dataset_id", "method"], kind="stable").reset_index(
        drop=True
    )


__all__ = [
    "ALIASES",
    "INDEPENDENT_UNIT_COLUMNS",
    "REQUIRED_COLUMNS",
    "prepare_tidy_summary",
]
