"""Leakage-safe, method-matched supervised benchmarking protocol."""
from __future__ import annotations

from hashlib import sha256
from inspect import signature
from typing import Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .configuration import ProtocolConfig, STATISTIC_RULE, protocol_config_sha256
from .evaluation_workspace import (
    GlobalPreprocessingWorkspace,
    PreparedInner as _PreparedInner,
    PreprocessingWorkspace,
    build_global_workspace as _build_global_workspace,
    build_workspace as _build_workspace,
    validate_workspace as _validate_workspace_cache,
)
from .methods import (
    METHODS,
    FittedRepresentation,
    candidate_grid,
    fit_candidate_bundle,
)
from .preprocessing import (
    TRANSFORM_KINDS,
    FoldPreprocessor,
    fit_fold_preprocessor,
)
from .protocol_estimands import (
    _fold_record as _fold_record,
    _paired_pooled_oof_metrics as _paired_pooled_oof_metrics,
    _repeat_averaged_oof_vectors as _repeat_averaged_oof_vectors,
    empirical_p_value as empirical_p_value,
)
from .splitting import (
    IndexSplit,
    OuterSplit,
    SplitPlan,
    binary_labels,
    derive_seed,
    materialize_plan,
    validate_groups,
)
from .stability import (
    feature_stability_summary,
    mean_pairwise_jaccard,
    mean_pairwise_kuncheva,
    nogueira_stability,
    summarize_feature_stability,
)


_USES_NEW_LOGISTIC_API = (
    signature(LogisticRegression).parameters["penalty"].default == "deprecated"
)
_METHOD_METRICS = (
    "pre_split_mean_fold_auc",
    "fold_contained_mean_fold_auc",
    "gap_mean_fold",
    "pre_split_oof_auc",
    "fold_contained_oof_auc",
    "gap_oof",
)


def fit_preprocessor(
    X_train: ArrayLike,
    config: ProtocolConfig,
    *,
    transform_kind: str = "already_scaled",
) -> FoldPreprocessor:
    """Public configuration-aware wrapper for fold-fitted preprocessing."""
    return fit_fold_preprocessor(
        X_train,
        completeness_threshold=config.completeness_threshold,
        variance_cap=config.variance_cap,
        transform_kind=transform_kind,
        auto_nonnegative_rule=config.auto_nonnegative_rule,
    )


def _fit_bundle(
    method: str,
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    config: ProtocolConfig,
    seed: int,
) -> tuple[FittedRepresentation, ...]:
    return fit_candidate_bundle(
        method,
        X,
        y,
        rfe_svc_c=config.rfe_svc_c,
        max_iter=(config.rfe_max_iter if method == "rfe" else config.max_iter),
        seed=seed,
    )


def _fit_logistic(
    model: LogisticRegression,
    X: NDArray,
    y: NDArray,
    *,
    phase: str,
) -> LogisticRegression:
    # sklearn 1.8 emits transition warnings for the still-portable penalty API.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=FutureWarning, module=r"^sklearn\."
            )
            warnings.filterwarnings("error", category=ConvergenceWarning)
            return model.fit(X, y)
    except ConvergenceWarning as exc:
        raise RuntimeError(
            f"LogisticRegression did not converge during {phase} at "
            f"max_iter={model.max_iter}"
        ) from exc


def _classifier(config: ProtocolConfig, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=config.classifier_c,
        solver="liblinear",
        max_iter=config.max_iter,
        random_state=seed,
    )


def _l1_classifier(
    *, c_value: float, max_iter: int, seed: int
) -> LogisticRegression:
    """Construct true L1 logistic regression across sklearn API generations."""
    common = {
        "C": c_value,
        "solver": "liblinear",
        "max_iter": max_iter,
        "random_state": seed,
    }
    if _USES_NEW_LOGISTIC_API:
        return LogisticRegression(l1_ratio=1.0, **common)
    return LogisticRegression(penalty="l1", **common)


def _fit_auc(
    train_x: NDArray[np.float64],
    train_y: NDArray[np.int_],
    test_x: NDArray[np.float64],
    test_y: NDArray[np.int_],
    config: ProtocolConfig,
    seed: int,
    *,
    phase: str,
) -> tuple[float, NDArray[np.float64], LogisticRegression]:
    model = _fit_logistic(
        _classifier(config, seed), train_x, train_y, phase=phase
    )
    scores = np.asarray(model.predict_proba(test_x)[:, 1], dtype=float)
    return float(roc_auc_score(test_y, scores)), scores, model


def select_best_candidate(mean_auc: Sequence[float]) -> int:
    """Maximize mean inner AUC; exact ties resolve to the earlier candidate."""
    scores = np.asarray(mean_auc, dtype=float)
    finite = np.flatnonzero(np.isfinite(scores))
    if finite.size == 0:
        raise ValueError("candidate selection requires at least one finite AUC")
    best = int(finite[0])
    for index in finite[1:]:
        if scores[index] > scores[best]:
            best = int(index)
    return best


def _candidate_params(method: str, index: int, rep: FittedRepresentation) -> dict:
    return {**candidate_grid(method)[index], **dict(rep.effective_params)}


def build_global_preprocessing_workspace(
    X: ArrayLike,
    config: ProtocolConfig,
    *,
    transform_kind: str = "already_scaled",
) -> GlobalPreprocessingWorkspace:
    """Fit label-free full-matrix preprocessing once for pre-split arms."""
    return _build_global_workspace(
        X,
        config,
        config_sha256=protocol_config_sha256(config),
        transform_kind=transform_kind,
        fit_preprocessor=fit_preprocessor,
    )


def build_preprocessing_workspace(
    X: NDArray[np.float64],
    plan: SplitPlan,
    config: ProtocolConfig,
    *,
    transform_kind: str = "already_scaled",
    global_workspace: GlobalPreprocessingWorkspace | None = None,
) -> PreprocessingWorkspace:
    """Materialize label-free outer/inner transforms for one split plan."""
    return _build_workspace(
        X,
        plan,
        config,
        config_sha256=protocol_config_sha256(config),
        transform_kind=transform_kind,
        fit_preprocessor=fit_preprocessor,
        global_workspace=global_workspace,
    )


def _tune_contained(
    method: str,
    prepared: tuple[_PreparedInner, ...],
    y: NDArray[np.int_],
    config: ProtocolConfig,
    seed_parts: tuple[object, ...],
) -> tuple[int, list[float]]:
    candidate_scores: list[list[float]] = [[], [], [], []]
    for inner_index, item in enumerate(prepared):
        train_y = y[np.asarray(item.train_indices, dtype=int)]
        test_y = y[np.asarray(item.test_indices, dtype=int)]
        bundle = _fit_bundle(
            method,
            item.train_x,
            train_y,
            config,
            derive_seed(config.seed, *seed_parts, inner_index, "representation"),
        )
        for candidate_index, rep in enumerate(bundle):
            auc, _, _ = _fit_auc(
                rep.transform(item.train_x),
                train_y,
                rep.transform(item.test_x),
                test_y,
                config,
                derive_seed(config.seed, *seed_parts, inner_index, candidate_index),
                phase=(
                    f"{method}:inner:{inner_index}:candidate:{candidate_index}"
                ),
            )
            candidate_scores[candidate_index].append(auc)
    means = [float(np.mean(scores)) for scores in candidate_scores]
    return select_best_candidate(means), means


def _evaluate_l1_fold(
    prepared: tuple[_PreparedInner, ...],
    y: NDArray[np.int_],
    outer_x: NDArray[np.float64],
    outer_y: NDArray[np.int_],
    test_x: NDArray[np.float64],
    test_y: NDArray[np.int_],
    config: ProtocolConfig,
    seed_parts: tuple[object, ...],
) -> tuple[float, NDArray[np.float64], LogisticRegression, int, list[float]]:
    candidate_scores: list[list[float]] = [[] for _ in config.l1_c_grid]
    for inner_index, item in enumerate(prepared):
        train_y = y[np.asarray(item.train_indices, dtype=int)]
        inner_test_y = y[np.asarray(item.test_indices, dtype=int)]
        for candidate_index, c_value in enumerate(config.l1_c_grid):
            model = _l1_classifier(
                c_value=c_value,
                max_iter=config.max_iter,
                seed=derive_seed(
                    config.seed, *seed_parts, inner_index, candidate_index
                ),
            )
            model = _fit_logistic(
                model,
                item.train_x,
                train_y,
                phase=f"baseline_l1:inner:{inner_index}:candidate:{candidate_index}",
            )
            prediction = model.predict_proba(item.test_x)[:, 1]
            candidate_scores[candidate_index].append(
                float(roc_auc_score(inner_test_y, prediction))
            )
    means = [float(np.mean(scores)) for scores in candidate_scores]
    chosen = select_best_candidate(means)
    final = _l1_classifier(
        c_value=config.l1_c_grid[chosen],
        max_iter=config.max_iter,
        seed=derive_seed(config.seed, *seed_parts, "final"),
    )
    final = _fit_logistic(
        final, outer_x, outer_y, phase="baseline_l1:outer-final"
    )
    prediction = np.asarray(final.predict_proba(test_x)[:, 1], dtype=float)
    return float(roc_auc_score(test_y, prediction)), prediction, final, chosen, means


def _evaluate_scenario(
    y: NDArray[np.int_],
    plan: SplitPlan,
    workspace: PreprocessingWorkspace,
    config: ProtocolConfig,
    dataset_id: str,
    tag: str,
    *,
    include_baseline: bool,
) -> dict:
    # ``tag`` is provenance only; model RNG schedules must be label-independent.
    del tag
    global_prep = workspace.global_workspace.preprocessor
    global_x = workspace.global_workspace.transformed_x
    global_bundles = {
        method: _fit_bundle(
            method, global_x, y, config,
            derive_seed(config.seed, dataset_id, method, "global"),
        )
        for method in METHODS
    }
    global_z = {
        method: tuple(rep.transform(global_x) for rep in bundle)
        for method, bundle in global_bundles.items()
    }
    pre_folds: dict[str, list[dict]] = {method: [] for method in METHODS}
    contained_folds: dict[str, list[dict]] = {method: [] for method in METHODS}
    baseline_folds: list[dict] = []
    preprocessing_records: list[dict] = []

    for outer_index, (split, prepared_outer) in enumerate(
        zip(plan.outer_splits, workspace.outer, strict=True)
    ):
        train = np.asarray(split.train_indices, dtype=int)
        test = np.asarray(split.test_indices, dtype=int)
        outer_x, test_x = prepared_outer.train_x, prepared_outer.test_x
        eligible = prepared_outer.eligible_features
        preprocessing_records.append(
            {
                "outer_index": outer_index,
                "repeat": split.repeat,
                "fold": split.fold,
                "p_after_preprocessing": len(eligible),
                "eligible_features": list(map(int, eligible)),
            }
        )
        for method in METHODS:
            seed_parts = (dataset_id, outer_index, method)
            chosen, contained_inner = _tune_contained(
                method,
                prepared_outer.inner,
                y,
                config,
                (*seed_parts, "fold_contained"),
            )
            pre_rep, pre_values = global_bundles[method][chosen], global_z[method][chosen]
            pre_auc, pre_scores, _ = _fit_auc(
                pre_values[train], y[train], pre_values[test], y[test], config,
                derive_seed(config.seed, *seed_parts, "pre_split_final"),
                phase=f"{method}:pre-split:outer-final:{outer_index}",
            )
            pre_features = [
                global_prep.original_feature_indices[index] for index in pre_rep.selected
            ]
            pre_folds[method].append(
                _fold_record(
                    outer_index, split, y[test], pre_scores, pre_auc,
                    _candidate_params(method, chosen, pre_rep), pre_features,
                    global_prep.n_features_out, contained_inner,
                )
            )
            outer_bundle = _fit_bundle(
                method,
                outer_x,
                y[train],
                config,
                derive_seed(config.seed, *seed_parts, "outer_representation"),
            )
            rep = outer_bundle[chosen]
            auc, scores, _ = _fit_auc(
                rep.transform(outer_x), y[train], rep.transform(test_x), y[test], config,
                derive_seed(config.seed, *seed_parts, "contained_final"),
                phase=f"{method}:fold-contained:outer-final:{outer_index}",
            )
            features = [eligible[index] for index in rep.selected]
            contained_folds[method].append(
                _fold_record(
                    outer_index, split, y[test], scores, auc,
                    _candidate_params(method, chosen, rep), features,
                    len(eligible), contained_inner,
                )
            )

        if include_baseline:
            auc, scores, model, chosen, inner_scores = _evaluate_l1_fold(
                prepared_outer.inner,
                y,
                outer_x,
                y[train],
                test_x,
                y[test],
                config,
                (dataset_id, outer_index, "baseline_l1"),
            )
            nonzero = np.flatnonzero(np.abs(model.coef_).reshape(-1) > 1e-12)
            features = [eligible[index] for index in nonzero]
            baseline_folds.append(
                _fold_record(
                    outer_index, split, y[test], scores, auc,
                    {"C": config.l1_c_grid[chosen], "n_nonzero": int(nonzero.size)},
                    features, len(eligible), inner_scores,
                )
            )

    method_results = {}
    for method in METHODS:
        pre_mean_fold_auc = float(
            np.mean([fold["auc"] for fold in pre_folds[method]])
        )
        contained_mean_fold_auc = float(
            np.mean([fold["auc"] for fold in contained_folds[method]])
        )
        pooled_oof = _paired_pooled_oof_metrics(
            pre_folds[method],
            contained_folds[method],
            n_samples=y.size,
            outer_repeats=config.outer_repeats,
            context=f"{dataset_id}.{method}",
        )
        sets = [fold["selected_features"] for fold in contained_folds[method]]
        stability = summarize_feature_stability(
            sets,
            eligible_feature_sets=[item.eligible_features for item in workspace.outer],
            raw_p=global_prep.n_features_in,
        )
        method_results[method] = {
            "pre_split_mean_fold_auc": pre_mean_fold_auc,
            "fold_contained_mean_fold_auc": contained_mean_fold_auc,
            "gap_mean_fold": pre_mean_fold_auc - contained_mean_fold_auc,
            **pooled_oof,
            "pre_split_folds": pre_folds[method],
            "fold_contained_folds": contained_folds[method],
            "feature_stability": stability,
        }
    baseline = None
    if include_baseline:
        baseline = {
            "auc": float(np.mean([fold["auc"] for fold in baseline_folds])),
            "folds": baseline_folds,
        }
    return {
        "methods": method_results,
        "baseline_l1": baseline,
        "outer_fold_preprocessing": preprocessing_records,
    }


def _validated_xy(X: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    values, labels = np.asarray(X, dtype=float), binary_labels(y)
    if values.ndim != 2 or values.shape[0] != labels.size:
        raise ValueError("X must be 2D with one row per y value")
    return values, labels


def _validate_workspace(
    values: NDArray[np.float64],
    plan: SplitPlan,
    config: ProtocolConfig,
    transform_kind: str,
    workspace: PreprocessingWorkspace,
) -> None:
    _validate_workspace_cache(
        values,
        plan,
        config_sha256=protocol_config_sha256(config),
        transform_kind=transform_kind,
        workspace=workspace,
    )


def evaluate_observed(
    X: ArrayLike,
    y: ArrayLike,
    *,
    dataset_id: str,
    config: ProtocolConfig = ProtocolConfig(),
    transform_kind: str = "already_scaled",
    groups: ArrayLike | None = None,
    workspace: PreprocessingWorkspace | None = None,
    global_workspace: GlobalPreprocessingWorkspace | None = None,
) -> dict:
    """Evaluate observed labels under pre-split and nested protocols."""
    values, labels = _validated_xy(X, y)
    plan = materialize_plan(labels, config, dataset_id, groups=groups)
    prepared = workspace or build_preprocessing_workspace(
        values,
        plan,
        config,
        transform_kind=transform_kind,
        global_workspace=global_workspace,
    )
    _validate_workspace(values, plan, config, transform_kind, prepared)
    scenario = _evaluate_scenario(
        labels,
        plan,
        prepared,
        config,
        dataset_id,
        "observed",
        include_baseline=True,
    )
    return {
        "dataset_id": str(dataset_id),
        "config_sha256": protocol_config_sha256(config),
        "statistic_rule": config.statistic_rule,
        "split_plan_sha256": plan.sha256,
        "split_strategy": plan.split_strategy,
        "requested_outer_splits": plan.requested_outer_splits,
        "effective_outer_splits": plan.effective_outer_splits,
        "group_adjustment_reason": plan.group_adjustment_reason,
        "group_fallback_reason": plan.group_fallback_reason,
        "preprocessing_workspace_sha256": prepared.workspace_sha256,
        "outer_fold_preprocessing": scenario["outer_fold_preprocessing"],
        "methods": {
            method: {"observed": scenario["methods"][method]} for method in METHODS
        },
        "baseline_l1": scenario["baseline_l1"],
    }


def _label_sha256(y: NDArray[np.int_]) -> str:
    return sha256(np.asarray(y, dtype="<i8").tobytes(order="C")).hexdigest()


def materialize_permuted_labels(
    y: ArrayLike,
    config: ProtocolConfig,
    dataset_id: str,
    index: int,
    *,
    groups: ArrayLike | None = None,
) -> tuple[NDArray[np.int_], str]:
    """Create a deterministic exchangeable or repeated-measures null vector."""
    if index < 0:
        raise ValueError("permutation indices must be non-negative")
    labels = binary_labels(y)
    rng = np.random.default_rng(
        derive_seed(config.seed, dataset_id, "label_permutation", index)
    )
    group_array = None if groups is None else np.asarray(groups, dtype=object)
    group_aware, _ = validate_groups(group_array, labels, config.outer_splits)
    if not group_aware:
        return np.asarray(labels[rng.permutation(labels.size)], dtype=int), "sample"

    assert group_array is not None
    unique_groups, inverse = np.unique(group_array, return_inverse=True)
    group_labels = []
    labels_are_constant = True
    for group_index in range(unique_groups.size):
        observed = np.unique(labels[inverse == group_index])
        labels_are_constant &= observed.size == 1
        group_labels.append(int(observed[0]))
    if labels_are_constant:
        shuffled = np.asarray(group_labels, dtype=int)[rng.permutation(unique_groups.size)]
        return np.asarray(shuffled[inverse], dtype=int), "group"

    candidate = labels.copy()
    for group_index in range(unique_groups.size):
        positions = np.flatnonzero(inverse == group_index)
        candidate[positions] = labels[positions][rng.permutation(positions.size)]
    return np.asarray(candidate, dtype=int), "within_group"


def _permutation_protocol(unit: str) -> str:
    if unit == "within_group":
        return "paired_repeated_measures_sensitivity"
    return "primary_exchangeable"


def _permutation_unit(
    labels: NDArray[np.int_],
    config: ProtocolConfig,
    groups: ArrayLike | None,
) -> str:
    group_array = None if groups is None else np.asarray(groups, dtype=object)
    group_aware, _ = validate_groups(group_array, labels, config.outer_splits)
    if not group_aware:
        return "sample"
    assert group_array is not None
    for group in np.unique(group_array):
        if np.unique(labels[group_array == group]).size != 1:
            return "within_group"
    return "group"


def evaluate_permutation_block(
    X: ArrayLike,
    y: ArrayLike,
    *,
    dataset_id: str,
    config: ProtocolConfig = ProtocolConfig(),
    permutation_indices: Sequence[int],
    transform_kind: str = "already_scaled",
    groups: ArrayLike | None = None,
    global_workspace: GlobalPreprocessingWorkspace | None = None,
) -> dict:
    """Rerun the full label-dependent split/tuning pipeline for each null vector."""
    values, labels = _validated_xy(X, y)
    indices = tuple(map(int, permutation_indices))
    if len(set(indices)) != len(indices):
        raise ValueError("permutation_indices must be unique")
    null = {
        method: {metric: [] for metric in _METHOD_METRICS}
        for method in METHODS
    }
    records = []
    global_cache = global_workspace or build_global_preprocessing_workspace(
        values, config, transform_kind=transform_kind
    )
    unit_for_design = _permutation_unit(labels, config, groups)
    protocol = _permutation_protocol(unit_for_design)
    for index in indices:
        permuted, unit = materialize_permuted_labels(
            labels, config, dataset_id, index, groups=groups
        )
        if unit != unit_for_design:
            raise RuntimeError("permutation unit changed within a dataset")
        plan = materialize_plan(permuted, config, dataset_id, groups=groups)
        prepared = build_preprocessing_workspace(
            values,
            plan,
            config,
            transform_kind=transform_kind,
            global_workspace=global_cache,
        )
        tag = f"permutation:{index}"
        scenario = _evaluate_scenario(
            permuted,
            plan,
            prepared,
            config,
            dataset_id,
            tag,
            include_baseline=False,
        )
        records.append(
            {
                "index": index,
                "label_sha256": _label_sha256(permuted),
                "split_plan_sha256": plan.sha256,
                "split_strategy": plan.split_strategy,
                "requested_outer_splits": plan.requested_outer_splits,
                "effective_outer_splits": plan.effective_outer_splits,
                "group_adjustment_reason": plan.group_adjustment_reason,
                "permutation_unit": unit,
                "permutation_protocol": protocol,
                "original_class_counts": list(map(int, np.bincount(labels, minlength=2))),
                "permuted_class_counts": list(map(int, np.bincount(permuted, minlength=2))),
            }
        )
        for method in METHODS:
            measured = scenario["methods"][method]
            for key in _METHOD_METRICS:
                null[method][key].append(measured[key])
    return {
        "dataset_id": str(dataset_id),
        "config_sha256": protocol_config_sha256(config),
        "statistic_rule": config.statistic_rule,
        "permutation_indices": list(indices),
        "permutations": records,
        "permutation_unit": unit_for_design,
        "permutation_protocol": protocol,
        "primary_null_eligible": protocol == "primary_exchangeable",
        "methods": null,
    }


def evaluate_supervised_protocol(
    X: ArrayLike,
    y: ArrayLike,
    *,
    dataset_id: str,
    config: ProtocolConfig = ProtocolConfig(),
    permutation_indices: Sequence[int] | None = None,
    transform_kind: str = "already_scaled",
    groups: ArrayLike | None = None,
) -> dict:
    """Run observed evaluation and its method-matched permutation null."""
    values, labels = _validated_xy(X, y)
    # Validate split feasibility before fitting any preprocessing state.
    materialize_plan(labels, config, dataset_id, groups=groups)
    global_cache = build_global_preprocessing_workspace(
        values, config, transform_kind=transform_kind
    )
    observed = evaluate_observed(
        values,
        labels,
        dataset_id=dataset_id,
        config=config,
        transform_kind=transform_kind,
        groups=groups,
        global_workspace=global_cache,
    )
    indices = (
        tuple(range(config.n_permutations))
        if permutation_indices is None
        else tuple(permutation_indices)
    )
    block = evaluate_permutation_block(
        values,
        labels,
        dataset_id=dataset_id,
        config=config,
        permutation_indices=indices,
        transform_kind=transform_kind,
        groups=groups,
        global_workspace=global_cache,
    )
    primary = bool(block["primary_null_eligible"])
    methods = {}
    for method in METHODS:
        measured, null = observed["methods"][method]["observed"], block["methods"][method]
        methods[method] = {
            "observed": measured,
            "permutation": {
                **null,
                "indices": list(indices),
                "permutation_unit": block["permutation_unit"],
                "permutation_protocol": block["permutation_protocol"],
                "primary_null_eligible": primary,
                "empirical_p_pre_split_mean_fold": (
                    empirical_p_value(
                        measured["pre_split_mean_fold_auc"],
                        null["pre_split_mean_fold_auc"],
                    )
                    if primary
                    else None
                ),
                "empirical_p_fold_contained_mean_fold": (
                    empirical_p_value(
                        measured["fold_contained_mean_fold_auc"],
                        null["fold_contained_mean_fold_auc"],
                    )
                    if primary
                    else None
                ),
                "empirical_p_gap_mean_fold": (
                    empirical_p_value(
                        measured["gap_mean_fold"], null["gap_mean_fold"]
                    )
                    if primary
                    else None
                ),
                "empirical_p_pre_split_oof": (
                    empirical_p_value(
                        measured["pre_split_oof_auc"], null["pre_split_oof_auc"]
                    )
                    if primary
                    else None
                ),
                "empirical_p_fold_contained_oof": (
                    empirical_p_value(
                        measured["fold_contained_oof_auc"],
                        null["fold_contained_oof_auc"],
                    )
                    if primary
                    else None
                ),
                "empirical_p_gap_oof": (
                    empirical_p_value(measured["gap_oof"], null["gap_oof"])
                    if primary
                    else None
                ),
            },
        }
    return {
        **{key: value for key, value in observed.items() if key != "methods"},
        "methods": methods,
        "permutations": block["permutations"],
        "permutation_protocol": block["permutation_protocol"],
        "primary_null_eligible": primary,
    }


__all__ = [
    "METHODS", "STATISTIC_RULE", "TRANSFORM_KINDS", "FoldPreprocessor", "IndexSplit", "OuterSplit",
    "GlobalPreprocessingWorkspace", "PreprocessingWorkspace", "ProtocolConfig",
    "SplitPlan", "build_global_preprocessing_workspace",
    "build_preprocessing_workspace", "candidate_grid", "derive_seed",
    "empirical_p_value", "evaluate_observed", "evaluate_permutation_block",
    "evaluate_supervised_protocol", "feature_stability_summary", "fit_preprocessor",
    "materialize_permuted_labels", "materialize_plan", "mean_pairwise_jaccard", "mean_pairwise_kuncheva",
    "nogueira_stability", "protocol_config_sha256", "select_best_candidate",
    "summarize_feature_stability",
]
