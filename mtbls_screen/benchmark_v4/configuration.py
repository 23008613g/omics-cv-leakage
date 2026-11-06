"""Immutable configuration and semantic hashing for the v4 protocol."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from .methods import METHODS
from .preprocessing import AUTO_NONNEGATIVE_RULE


STATISTIC_RULE = (
    "mean_fold_auc_and_repeat_averaged_raw_test_index_pooled_oof_auc_v1"
)


@dataclass(frozen=True)
class ProtocolConfig:
    """Specification shared by every dataset, method, and label vector."""

    seed: int = 42
    outer_splits: int = 5
    outer_repeats: int = 3
    inner_splits: int = 3
    n_permutations: int = 0
    completeness_threshold: float = 0.5
    variance_cap: int | None = 5_000
    classifier_c: float = 1.0
    l1_c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    rfe_svc_c: float = 0.1
    rfe_max_iter: int = 50_000
    max_iter: int = 5_000
    auto_nonnegative_rule: str = AUTO_NONNEGATIVE_RULE
    statistic_rule: str = STATISTIC_RULE
    methods: tuple[str, ...] = METHODS

    def __post_init__(self) -> None:
        if self.outer_splits < 2:
            raise ValueError("outer_splits must be at least 2")
        if self.outer_repeats < 1:
            raise ValueError("outer_repeats must be at least 1")
        if self.inner_splits < 2:
            raise ValueError("inner_splits must be at least 2")
        if self.n_permutations < 0:
            raise ValueError("n_permutations must be non-negative")
        if not 0.5 <= self.completeness_threshold <= 1.0:
            raise ValueError("completeness_threshold must be in [0.5, 1]")
        if self.variance_cap is not None and self.variance_cap < 1:
            raise ValueError("variance_cap must be positive or None")
        if self.classifier_c <= 0 or self.rfe_svc_c <= 0:
            raise ValueError("classifier and RFE C values must be positive")
        if self.max_iter < 1 or self.rfe_max_iter < 1:
            raise ValueError("max_iter and rfe_max_iter must be positive")
        if self.auto_nonnegative_rule != AUTO_NONNEGATIVE_RULE:
            raise ValueError(
                f"auto_nonnegative_rule must be {AUTO_NONNEGATIVE_RULE!r}"
            )
        if self.statistic_rule != STATISTIC_RULE:
            raise ValueError(f"statistic_rule must be {STATISTIC_RULE!r}")
        if len(self.l1_c_grid) != 4 or any(c <= 0 for c in self.l1_c_grid):
            raise ValueError("l1_c_grid must contain four positive values")
        if tuple(self.methods) != METHODS:
            raise ValueError(f"methods must be exactly {METHODS!r}")


def protocol_config_sha256(
    config: ProtocolConfig, *, include_permutation_count: bool = False
) -> str:
    """Hash semantics; default permits extending deterministic null blocks."""
    values = asdict(config)
    if not include_permutation_count:
        values = {
            key: value for key, value in values.items() if key != "n_permutations"
        }
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["ProtocolConfig", "STATISTIC_RULE", "protocol_config_sha256"]
