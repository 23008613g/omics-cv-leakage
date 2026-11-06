"""Checkpoint-derived null summaries used by the independent QC pass."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


NULL_SUMMARY_FIELDS = ("null_mean", "null_standard_deviation", "null_min", "null_max")


def checkpoint_null_summary(
    checkpoint: Mapping[str, Any], method: str, metric: str,
) -> tuple[list[float], dict[str, float | None]]:
    """Recompute descriptive null fields from validated checkpoint blocks."""
    values = [
        float(value)
        for block in checkpoint["state"].get("permutation_blocks", [])
        for value in block["result"]["methods"][method][metric]
    ]
    if not values:
        return values, dict.fromkeys(NULL_SUMMARY_FIELDS)
    mean = sum(values) / len(values)
    standard_deviation = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        if len(values) > 1 else 0.0
    )
    return values, {
        "null_mean": mean,
        "null_standard_deviation": standard_deviation,
        "null_min": min(values),
        "null_max": max(values),
    }


def phipson_smyth_plus_one(
    observed: float,
    null_values: list[float],
) -> tuple[int, int, float]:
    """Return valid B, right-tail exceedances, and the finite-sample +1 p-value."""
    if not math.isfinite(observed):
        raise ValueError("observed value must be finite")
    if any(not math.isfinite(value) for value in null_values):
        raise ValueError("null values must be finite")
    valid_b = len(null_values)
    exceedances = sum(value >= observed for value in null_values)
    return valid_b, exceedances, (1 + exceedances) / (valid_b + 1)


__all__ = [
    "NULL_SUMMARY_FIELDS",
    "checkpoint_null_summary",
    "phipson_smyth_plus_one",
]
