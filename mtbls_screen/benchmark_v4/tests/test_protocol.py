from __future__ import annotations

from dataclasses import FrozenInstanceError
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from benchmark_v4 import methods as supervised_methods
from benchmark_v4 import protocol as supervised_protocol

from benchmark_v4.protocol import (
    METHODS,
    ProtocolConfig,
    STATISTIC_RULE,
    build_preprocessing_workspace,
    candidate_grid,
    derive_seed,
    empirical_p_value,
    evaluate_observed,
    evaluate_permutation_block,
    evaluate_supervised_protocol,
    feature_stability_summary,
    fit_preprocessor,
    materialize_permuted_labels,
    materialize_plan,
    mean_pairwise_jaccard,
    mean_pairwise_kuncheva,
    nogueira_stability,
    protocol_config_sha256,
    select_best_candidate,
    summarize_feature_stability,
)


def test_protocol_config_is_frozen_and_has_required_default_design() -> None:
    config = ProtocolConfig()

    assert config.outer_splits == 5
    assert config.outer_repeats == 3
    assert config.inner_splits == 3
    assert config.max_iter == 5_000
    assert config.rfe_max_iter == 50_000
    assert (
        config.auto_nonnegative_rule
        == "train_finite_nonnegative_max_gt_50_v2"
    )
    assert config.methods == METHODS
    assert config.statistic_rule == STATISTIC_RULE
    assert STATISTIC_RULE == (
        "mean_fold_auc_and_repeat_averaged_raw_test_index_pooled_oof_auc_v1"
    )
    assert METHODS == ("top_k", "rfe", "spls", "pls_da", "opls_da")
    assert all(len(candidate_grid(method)) == 4 for method in METHODS)

    with pytest.raises(FrozenInstanceError):
        config.outer_splits = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"outer_splits": 1}, "outer_splits"),
        ({"inner_splits": 1}, "inner_splits"),
        ({"completeness_threshold": 0.49}, "completeness_threshold"),
        ({"variance_cap": 0}, "variance_cap"),
        ({"n_permutations": -1}, "n_permutations"),
        ({"max_iter": 0}, "max_iter"),
        ({"rfe_max_iter": 0}, "rfe_max_iter"),
        ({"auto_nonnegative_rule": "whole_dataset_v1"}, "auto_nonnegative_rule"),
        ({"statistic_rule": "mean_fold_only_v0"}, "statistic_rule"),
    ],
)
def test_protocol_config_rejects_invalid_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ProtocolConfig(**kwargs)


def test_sha256_seeds_are_deterministic_and_namespaced() -> None:
    first = derive_seed(42, "dataset-A", "outer")

    assert first == derive_seed(42, "dataset-A", "outer")
    assert first != derive_seed(42, "dataset-B", "outer")
    assert first != derive_seed(42, "dataset-A", "inner")
    assert 0 <= first < 2**32


def test_protocol_hash_is_invariant_to_permutation_block_size_by_default() -> None:
    short = ProtocolConfig(n_permutations=19)
    extended = ProtocolConfig(n_permutations=200)

    assert protocol_config_sha256(short) == protocol_config_sha256(extended)
    assert protocol_config_sha256(
        short, include_permutation_count=True
    ) != protocol_config_sha256(extended, include_permutation_count=True)
    assert protocol_config_sha256(short) != protocol_config_sha256(
        ProtocolConfig(n_permutations=19, rfe_max_iter=50_001)
    )


def test_rfe_has_a_dedicated_prespecified_iteration_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, int]] = []

    def record_bundle(  # noqa: ANN202
        method, X, y, *, rfe_svc_c, max_iter, seed  # noqa: ANN001
    ):
        captured.append((str(method), int(max_iter)))
        return ()

    monkeypatch.setattr(supervised_protocol, "fit_candidate_bundle", record_bundle)
    config = ProtocolConfig(max_iter=17, rfe_max_iter=50_000)
    X = np.arange(24, dtype=float).reshape(6, 4)
    y = np.array([0, 0, 0, 1, 1, 1])

    supervised_protocol._fit_bundle("rfe", X, y, config, seed=3)
    supervised_protocol._fit_bundle("pls_da", X, y, config, seed=3)

    assert captured == [("rfe", 50_000), ("pls_da", 17)]


def test_materialized_group_plan_is_shared_repeated_and_reproducible() -> None:
    y = np.tile([0, 1], 30)
    groups = np.repeat(np.arange(30), 2)
    config = ProtocolConfig(n_permutations=0)

    plan = materialize_plan(y, config, "shared-plan", groups=groups)
    again = materialize_plan(
        y, config, "shared-plan", groups=groups, analysis_tag="different-label-tag"
    )

    assert plan.sha256 == again.sha256
    assert plan.split_strategy == "repeated_stratified_group_kfold"
    assert len(plan.outer_splits) == 15
    assert all(len(item.inner_splits) == 3 for item in plan.outer_splits)

    test_counts = np.zeros(y.size, dtype=int)
    for item in plan.outer_splits:
        train = np.asarray(item.train_indices)
        test = np.asarray(item.test_indices)
        test_counts[test] += 1
        assert set(groups[train]).isdisjoint(groups[test])
        for inner in item.inner_splits:
            inner_train = train[np.asarray(inner.train_indices)]
            inner_test = train[np.asarray(inner.test_indices)]
            assert set(groups[inner_train]).isdisjoint(groups[inner_test])

    assert np.all(test_counts == config.outer_repeats)


def test_group_outer_folds_adapt_to_available_independent_groups() -> None:
    groups = np.repeat(np.arange(6), 4)
    y = np.repeat([0, 0, 0, 1, 1, 1], 4)
    config = ProtocolConfig(outer_splits=5, outer_repeats=2, inner_splits=3)

    plan = materialize_plan(y, config, "adaptive-group-outer", groups=groups)

    assert plan.split_strategy == "repeated_stratified_group_kfold"
    assert plan.requested_outer_splits == 5
    assert plan.effective_outer_splits == 3
    assert len(plan.outer_splits) == 6
    assert all(item.inner_split_strategy == "stratified_group_kfold" for item in plan.outer_splits)
    for item in plan.outer_splits:
        train, test = np.asarray(item.train_indices), np.asarray(item.test_indices)
        assert set(groups[train]).isdisjoint(groups[test])
        assert np.unique(y[train]).size == np.unique(y[test]).size == 2


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (np.array(["a", "b"]), "match y"),
        (np.array(["a", "b", None, "d"], dtype=object), "missing"),
    ],
)
def test_supplied_malformed_groups_fail_closed(groups: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        materialize_plan(
            np.array([0, 0, 1, 1]),
            ProtocolConfig(outer_splits=2, outer_repeats=1, inner_splits=2),
            "bad-groups",
            groups=groups,
        )


def test_supplied_groups_with_fewer_than_two_groups_per_class_fail_closed() -> None:
    with pytest.raises(ValueError, match="insufficient_groups"):
        materialize_plan(
            np.array([0, 0, 1, 1]),
            ProtocolConfig(outer_splits=3, outer_repeats=1, inner_splits=2),
            "insufficient-groups",
            groups=np.array(["control", "control", "case", "case"]),
        )


def test_group_aware_outer_cv_never_falls_back_to_row_level_inner_cv() -> None:
    groups = np.repeat(np.arange(6), 4)
    y = np.repeat([0, 0, 0, 1, 1, 1], 4)
    config = ProtocolConfig(outer_splits=3, outer_repeats=1, inner_splits=3)

    plan = materialize_plan(y, config, "adaptive-group-inner", groups=groups)

    assert plan.split_strategy == "repeated_stratified_group_kfold"
    assert all(item.inner_split_strategy == "stratified_group_kfold" for item in plan.outer_splits)
    assert all(item.effective_inner_splits == 2 for item in plan.outer_splits)
    for item in plan.outer_splits:
        outer_train = np.asarray(item.train_indices)
        for inner in item.inner_splits:
            inner_train = outer_train[np.asarray(inner.train_indices)]
            inner_test = outer_train[np.asarray(inner.test_indices)]
            assert set(groups[inner_train]).isdisjoint(groups[inner_test])


def test_fold_preprocessor_uses_training_only_filter_imputation_cap_and_scale() -> None:
    X_train = np.array(
        [
            [1.0, np.nan, 5.0, 0.0, 0.0],
            [2.0, np.nan, 5.0, 1.0, 100.0],
            [np.nan, 9.0, 5.0, 2.0, 200.0],
            [4.0, np.nan, 5.0, 3.0, 300.0],
        ]
    )
    config = ProtocolConfig(variance_cap=2, n_permutations=0)

    fitted = fit_preprocessor(X_train, config)
    transformed_train = fitted.transform(X_train)
    transformed_test = fitted.transform(
        np.array([[1_000.0, 999.0, 5.0, np.nan, 400.0]])
    )

    assert fitted.original_feature_indices == (3, 4)
    assert fitted.n_features_in == 5
    assert fitted.n_features_out == 2
    assert np.all(np.isfinite(transformed_train))
    assert np.allclose(transformed_train.mean(axis=0), 0.0)
    assert np.allclose(transformed_train.std(axis=0), 1.0)
    assert transformed_test.shape == (1, 2)
    assert np.isfinite(transformed_test).all()
    # The held-out NaN receives the training median (1.5), hence z=0.
    assert transformed_test[0, 0] == pytest.approx(0.0)


def test_fold_safe_raw_transform_parameters_are_learned_from_training_only() -> None:
    config = ProtocolConfig(variance_cap=None, n_permutations=0)
    intensity_train = np.array([[2.0, 8.0], [4.0, 16.0], [np.nan, 32.0]])
    intensity = fit_preprocessor(
        intensity_train, config, transform_kind="intensity"
    )

    assert intensity.positive_floor == pytest.approx(1.0)
    assert np.isfinite(intensity.transform(np.array([[0.0, 1_000_000.0]]))).all()

    counts = fit_preprocessor(
        np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]),
        config,
        transform_kind="counts",
    )
    assert np.isfinite(counts.transform(np.array([[4.0, 0.0]]))).all()

    microbiome = fit_preprocessor(
        np.array([[1.0, 9.0], [2.0, 8.0], [5.0, 5.0]]),
        config,
        transform_kind="microbiome_counts",
    )
    assert microbiome.pseudocount == pytest.approx(0.05)
    assert np.isfinite(microbiome.transform(np.array([[0.0, 10.0]]))).all()

    auto_log = fit_preprocessor(
        np.array([[2.0, 20.0], [4.0, 100.0], [8.0, 40.0]]),
        config,
        transform_kind="auto_nonnegative",
    )
    auto_linear = fit_preprocessor(
        np.array([[2.0, 20.0], [4.0, 30.0], [8.0, 40.0]]),
        config,
        transform_kind="auto_nonnegative",
    )
    assert auto_log.log_applied is True
    assert auto_log.positive_floor == pytest.approx(1.0)
    assert auto_linear.log_applied is False
    assert auto_linear.positive_floor is None
    assert np.isfinite(auto_log.transform(np.array([[0.0, 1_000_000.0]]))).all()


def test_auto_nonnegative_uses_identity_when_training_contains_negative_values() -> None:
    config = ProtocolConfig(variance_cap=None, n_permutations=0)
    training = np.array(
        [
            [-2.0, 60.0],
            [-1.0, 100.0],
            [0.0, 200.0],
        ]
    )

    fitted = fit_preprocessor(
        training, config, transform_kind="auto_nonnegative"
    )

    assert fitted.log_applied is False
    assert fitted.positive_floor is None
    assert np.allclose(fitted.means, training.mean(axis=0))
    assert np.allclose(fitted.scales, training.std(axis=0))
    assert np.isfinite(fitted.transform(np.array([[-100.0, 1_000.0]]))).all()


def test_auto_nonnegative_training_rule_has_strict_threshold_and_is_reused() -> None:
    config = ProtocolConfig(variance_cap=None, n_permutations=0)
    threshold_training = np.array(
        [
            [0.0, 2.0],
            [10.0, 20.0],
            [50.0, 40.0],
        ]
    )
    logarithmic_training = np.array(
        [
            [2.0, 20.0],
            [4.0, 100.0],
            [8.0, 40.0],
        ]
    )

    threshold = fit_preprocessor(
        threshold_training, config, transform_kind="auto_nonnegative"
    )
    logarithmic = fit_preprocessor(
        logarithmic_training, config, transform_kind="auto_nonnegative"
    )

    # A held-out maximum cannot retroactively trigger a training-fold log.
    assert threshold.log_applied is False
    assert threshold.positive_floor is None
    assert np.isfinite(
        threshold.transform(np.array([[-5.0, 1_000_000.0]]))
    ).all()

    # A held-out negative under a training-selected log is explicitly clipped
    # to the training-derived positive floor instead of producing NaN.
    assert logarithmic.log_applied is True
    assert logarithmic.positive_floor == pytest.approx(1.0)
    held_out_negative = logarithmic.transform(
        np.array([[-1_000.0, 1_000_000.0]])
    )
    held_out_floor = logarithmic.transform(
        np.array([[logarithmic.positive_floor, 1_000_000.0]])
    )
    assert np.isfinite(held_out_negative).all()
    assert held_out_negative[0, 0] == pytest.approx(held_out_floor[0, 0])


def test_signed_auto_nonnegative_is_identity_in_every_workspace_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = np.arange(24, dtype=float)
    X = np.column_stack(
        (
            -1.0 - rows / 100.0,
            60.0 + rows,
            np.sin(rows),
            rows % 5.0,
        )
    )
    y = np.repeat([0, 1], 12)
    config = ProtocolConfig(
        outer_splits=2,
        outer_repeats=1,
        inner_splits=2,
        variance_cap=None,
        n_permutations=0,
    )
    plan = materialize_plan(y, config, "signed-workspace")
    fitted_states: list[tuple[bool | None, float | None]] = []
    original = supervised_protocol.fit_preprocessor

    def recording_fit(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        fitted = original(*args, **kwargs)
        fitted_states.append((fitted.log_applied, fitted.positive_floor))
        return fitted

    monkeypatch.setattr(supervised_protocol, "fit_preprocessor", recording_fit)
    workspace = build_preprocessing_workspace(
        X, plan, config, transform_kind="auto_nonnegative"
    )

    expected_fits = 1 + config.outer_splits * (
        1 + config.inner_splits
    )
    assert fitted_states == [(False, None)] * expected_fits
    assert np.isfinite(workspace.global_workspace.transformed_x).all()
    for outer in workspace.outer:
        assert np.isfinite(outer.train_x).all()
        assert np.isfinite(outer.test_x).all()
        assert all(np.isfinite(inner.train_x).all() for inner in outer.inner)
        assert all(np.isfinite(inner.test_x).all() for inner in outer.inner)


def test_all_methods_use_the_same_candidate_selection_rule_and_first_tie_break() -> None:
    assert select_best_candidate([0.70, 0.80, 0.80, 0.75]) == 1
    assert select_best_candidate([np.nan, 0.60, 0.55, 0.50]) == 1
    with pytest.raises(ValueError, match="finite"):
        select_best_candidate([np.nan, np.nan, np.nan, np.nan])


def test_pls_and_opls_return_non_saturated_top_q_importance_sets() -> None:
    rng = np.random.default_rng(21)
    y = np.repeat([0, 1], 20)
    X = rng.normal(size=(40, 30))
    X[:, :4] += y[:, None]

    for method in ("pls_da", "opls_da"):
        bundle = supervised_methods.fit_candidate_bundle(
            method, X, y, rfe_svc_c=0.1, max_iter=5_000, seed=9
        )
        assert len(bundle) == 4
        assert all(len(candidate.selected) == 20 for candidate in bundle)
        assert all(dict(candidate.effective_params)["importance_q"] == 20 for candidate in bundle)


def test_pls_vip_and_opls_effective_importance_match_numeric_references() -> None:
    raw = np.array(
        [
            [-2.0, 0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 3.0],
            [0.5, 0.2, 0.7, 1.2],
            [-0.5, 0.8, 0.3, 1.8],
        ]
    )
    X = StandardScaler().fit_transform(raw)
    y = np.array([0, 0, 1, 1, 1, 0])

    pls_bundle = supervised_methods.fit_candidate_bundle(
        "pls_da", X, y, rfe_svc_c=0.1, max_iter=5_000, seed=1
    )
    pls = pls_bundle[0]
    opls = supervised_methods.fit_candidate_bundle(
        "opls_da", X, y, rfe_svc_c=0.1, max_iter=5_000, seed=1
    )[1]

    assert np.allclose(
        pls.importance_scores,
        [1.593469074429, 0.407427646725, 0.283915015854, 1.101930708028],
        atol=1e-10,
    )
    assert np.allclose(
        opls.importance_scores,
        [0.803877871723, 0.376685473403, 0.285577183206, 0.439677142837],
        atol=1e-10,
    )
    assert [
        dict(candidate.effective_params)["effective_n_components"]
        for candidate in pls_bundle
    ] == [1, 2, 3, 3]


def test_rfe_nonconvergence_is_a_structured_failure_not_a_silenced_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def emit_warning(self, X, y):  # noqa: ANN001, ANN202
        warnings.warn("forced nonconvergence", ConvergenceWarning)
        self.coef_ = np.ones((1, X.shape[1]))
        return self

    monkeypatch.setattr(supervised_methods.LinearSVC, "fit", emit_warning)
    X = np.arange(240, dtype=float).reshape(20, 12)
    y = np.repeat([0, 1], 10)

    with pytest.raises(RuntimeError, match="did not converge"):
        supervised_methods.fit_candidate_bundle(
            "rfe", X, y, rfe_svc_c=0.1, max_iter=5_000, seed=3
        )


def test_pls_nonconvergence_uses_configured_limit_and_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, int] = {}

    def emit_warning(self, X, y):  # noqa: ANN001, ANN202
        seen["max_iter"] = self.max_iter
        warnings.warn("forced PLS nonconvergence", ConvergenceWarning)
        return self

    monkeypatch.setattr(supervised_methods.PLSRegression, "fit", emit_warning)
    X = np.arange(240, dtype=float).reshape(20, 12)
    y = np.repeat([0, 1], 10)

    with pytest.raises(RuntimeError, match=r"PLS-DA.*max_iter=17"):
        supervised_methods.fit_candidate_bundle(
            "pls_da", X, y, rfe_svc_c=0.1, max_iter=17, seed=3
        )
    assert seen["max_iter"] == 17


def test_logistic_nonconvergence_fails_with_phase_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def emit_warning(self, X, y):  # noqa: ANN001, ANN202
        warnings.warn("forced logistic nonconvergence", ConvergenceWarning)
        return self

    monkeypatch.setattr(supervised_protocol.LogisticRegression, "fit", emit_warning)
    model = supervised_protocol.LogisticRegression(max_iter=19)

    with pytest.raises(RuntimeError, match=r"outer-final.*max_iter=19"):
        supervised_protocol._fit_logistic(
            model,
            np.arange(12, dtype=float).reshape(6, 2),
            np.array([0, 0, 0, 1, 1, 1]),
            phase="outer-final",
        )


def test_l1_logistic_factory_avoids_deprecated_penalty_api_warning() -> None:
    X = np.array(
        [[-2.0, 0.0], [-1.0, 0.5], [-0.5, 1.0], [0.5, 0.0], [1.0, 0.5], [2.0, 1.0]]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    model = supervised_protocol._l1_classifier(
        c_value=1.0, max_iter=500, seed=11
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        supervised_protocol._fit_logistic(
            model, X, y, phase="l1-api-compatibility-test"
        )

    assert not any("penalty is deprecated" in str(item.message) for item in caught)


def test_feature_stability_metrics_handle_equal_variable_and_saturated_sets() -> None:
    equal_sets = ({0, 1}, {0, 2}, {1, 2})

    assert mean_pairwise_jaccard(equal_sets) == pytest.approx(1 / 3)
    assert mean_pairwise_kuncheva(equal_sets, n_features=4) == pytest.approx(0.0)
    assert nogueira_stability(equal_sets, n_features=4) == pytest.approx(0.0)
    assert nogueira_stability(({0, 1}, {0, 1}, {0, 2}), n_features=5) == pytest.approx(
        4 / 9
    )

    variable_sets = ({0}, {0, 1}, {1, 2, 3})
    variable_summary = feature_stability_summary(variable_sets, n_features=4)
    assert np.isfinite(variable_summary["nogueira"])
    assert np.isnan(variable_summary["mean_kuncheva"])
    assert variable_summary["kuncheva_applicable"] is False

    saturated_summary = feature_stability_summary(
        ({0, 1, 2, 3}, {0, 1}), n_features=4
    )
    assert saturated_summary["q_equals_p_any"] is True


def test_primary_stability_uses_common_fold_eligible_universe() -> None:
    selected = ({0, 2}, {0, 3}, {0, 4})
    eligible = ({0, 1, 2}, {0, 1, 3}, {0, 1, 4})

    summary = summarize_feature_stability(
        selected, eligible_feature_sets=eligible, raw_p=5
    )

    assert summary["common_eligible_p"] == 2
    assert summary["raw_p"] == 5
    assert summary["common_eligible_coverage"] == pytest.approx(0.4)
    assert summary["mean_jaccard"] == pytest.approx(1 / 3)
    assert summary["nogueira"] == pytest.approx(1.0)
    assert summary["mean_kuncheva"] == pytest.approx(1.0)
    assert summary["raw_universe_sensitivity"]["nogueira"] < 1.0
    assert summary["q_equals_p_after_preprocessing_any"] is False


def test_stability_rejects_selected_features_ineligible_in_their_fold() -> None:
    with pytest.raises(ValueError, match="not eligible"):
        summarize_feature_stability(
            ({0, 3}, {0}), eligible_feature_sets=({0, 1}, {0, 1}), raw_p=4
        )


def test_empirical_p_value_uses_plus_one_correction() -> None:
    assert empirical_p_value(0.8, [0.7, 0.8, 0.9]) == pytest.approx(0.75)
    assert empirical_p_value(1.0, []) == pytest.approx(1.0)


def test_estimand_helpers_are_identical_protocol_module_aliases() -> None:
    from benchmark_v4 import protocol_estimands

    assert supervised_protocol._fold_record is protocol_estimands._fold_record
    assert (
        supervised_protocol._repeat_averaged_oof_vectors
        is protocol_estimands._repeat_averaged_oof_vectors
    )
    assert (
        supervised_protocol._paired_pooled_oof_metrics
        is protocol_estimands._paired_pooled_oof_metrics
    )
    assert supervised_protocol.empirical_p_value is protocol_estimands.empirical_p_value
    assert protocol_estimands.__all__ == ["empirical_p_value"]


def test_estimand_fold_record_preserves_the_frozen_checkpoint_schema() -> None:
    from benchmark_v4 import protocol_estimands

    split = supervised_protocol.OuterSplit(
        repeat=1,
        fold=2,
        train_indices=(0, 2, 4),
        test_indices=(1, 3),
        inner_splits=(),
        inner_split_strategy="stratified_kfold",
        effective_inner_splits=2,
    )

    record = protocol_estimands._fold_record(
        7,
        split,
        np.array([0, 1]),
        np.array([0.25, 0.75]),
        0.875,
        {"n_components": 2},
        (4, 4, 9),
        2,
        (0.6, 0.7, 0.8, 0.9),
    )

    assert record == {
        "outer_index": 7,
        "repeat": 1,
        "fold": 2,
        "train_size": 3,
        "test_size": 2,
        "p_after_preprocessing": 2,
        "saturated_after_preprocessing": True,
        "test_indices": [1, 3],
        "y_true": [0, 1],
        "y_score": [0.25, 0.75],
        "auc": 0.875,
        "chosen_params": {"n_components": 2},
        "selected_features": [4, 4, 9],
        "inner_candidate_scores": [0.6, 0.7, 0.8, 0.9],
    }


@pytest.mark.parametrize(
    ("fold", "message"),
    [
        (
            {
                "repeat": 0,
                "test_indices": [False, 1],
                "y_true": [0, 1],
                "y_score": [0.1, 0.9],
            },
            "raw test index",
        ),
        (
            {
                "repeat": 0,
                "test_indices": [0, 1],
                "y_true": [0, 1],
                "y_score": [0.1, np.inf],
            },
            "finite",
        ),
    ],
)
def test_estimand_oof_vectors_fail_closed_at_index_and_score_boundaries(
    fold: dict,
    message: str,
) -> None:
    from benchmark_v4 import protocol_estimands

    with pytest.raises(ValueError, match=message):
        protocol_estimands._repeat_averaged_oof_vectors(
            [fold], n_samples=2, outer_repeats=1, context="boundary"
        )


def test_pooled_oof_auc_averages_each_raw_test_index_across_repeats() -> None:
    def fold(
        repeat: int,
        fold_index: int,
        indices: list[int],
        labels: list[int],
        scores: list[float],
    ) -> dict:
        return {
            "repeat": repeat,
            "fold": fold_index,
            "test_indices": indices,
            "y_true": labels,
            "y_score": scores,
        }

    pre_split = [
        fold(0, 0, [0, 2], [0, 1], [0.1, 0.9]),
        fold(0, 1, [1, 3], [0, 1], [0.8, 0.7]),
        fold(1, 0, [0, 2], [0, 1], [0.8, 0.7]),
        fold(1, 1, [1, 3], [0, 1], [0.1, 0.9]),
    ]
    fold_contained = [
        fold(0, 0, [0, 2], [0, 1], [0.4, 0.6]),
        fold(0, 1, [1, 3], [0, 1], [0.7, 0.5]),
        fold(1, 0, [0, 2], [0, 1], [0.6, 0.4]),
        fold(1, 1, [1, 3], [0, 1], [0.3, 0.7]),
    ]

    metrics = supervised_protocol._paired_pooled_oof_metrics(
        pre_split,
        fold_contained,
        n_samples=4,
        outer_repeats=2,
        context="adaptive-group-two-fold",
    )

    assert metrics == {
        "pre_split_oof_auc": pytest.approx(1.0),
        "fold_contained_oof_auc": pytest.approx(0.75),
        "gap_oof": pytest.approx(0.25),
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda folds: [
                row for position, row in enumerate(folds) if position != 2
            ],
            "exactly outer_repeats",
        ),
        (
            lambda folds: [
                *folds[:2],
                {**folds[2], "y_true": [1, 1]},
                *folds[3:],
            ],
            "label changed",
        ),
    ],
)
def test_pooled_oof_auc_fails_closed_on_repeat_or_label_mismatch(
    mutation: object,
    message: str,
) -> None:
    folds = [
        {
            "repeat": repeat,
            "fold": fold_index,
            "test_indices": indices,
            "y_true": labels,
            "y_score": scores,
        }
        for repeat, fold_index, indices, labels, scores in (
            (0, 0, [0, 2], [0, 1], [0.1, 0.9]),
            (0, 1, [1, 3], [0, 1], [0.2, 0.8]),
            (1, 0, [0, 2], [0, 1], [0.3, 0.7]),
            (1, 1, [1, 3], [0, 1], [0.4, 0.6]),
        )
    ]

    with pytest.raises(ValueError, match=message):
        supervised_protocol._paired_pooled_oof_metrics(
            mutation(folds),  # type: ignore[operator]
            folds,
            n_samples=4,
            outer_repeats=2,
            context="invalid-oof",
        )


def test_constant_group_permutation_shuffles_group_labels_not_rows() -> None:
    sizes = np.array([2, 3, 4, 2, 3, 4])
    groups = np.repeat(np.arange(6), sizes)
    y = np.repeat([0, 0, 0, 1, 1, 1], sizes)
    config = ProtocolConfig(outer_splits=3, outer_repeats=1, inner_splits=2)

    permuted, unit = materialize_permuted_labels(
        y, config, "group-null", 4, groups=groups
    )
    repeated, repeated_unit = materialize_permuted_labels(
        y, config, "group-null", 4, groups=groups
    )

    assert unit == repeated_unit == "group"
    assert np.array_equal(permuted, repeated)
    assert all(np.unique(permuted[groups == group]).size == 1 for group in range(6))
    assert sum(permuted[groups == group][0] == 1 for group in range(6)) == 3



def test_mixed_group_permutation_shuffles_within_group_and_preserves_counts() -> None:
    groups = np.repeat(np.arange(6), 4)
    y = np.tile([0, 0, 1, 1], 6)
    config = ProtocolConfig(outer_splits=3, outer_repeats=1, inner_splits=2)

    permuted, unit = materialize_permuted_labels(
        y, config, "mixed-group-null", 2, groups=groups
    )
    repeated, repeated_unit = materialize_permuted_labels(
        y, config, "mixed-group-null", 2, groups=groups
    )

    assert unit == repeated_unit == "within_group"
    assert np.array_equal(permuted, repeated)
    for group in np.unique(groups):
        assert np.array_equal(
            np.bincount(permuted[groups == group], minlength=2),
            np.bincount(y[groups == group], minlength=2),
        )


def test_all_unique_groups_use_group_equivalent_primary_permutation() -> None:
    y = np.tile([0, 1], 12)
    groups = np.arange(y.size)
    config = ProtocolConfig(outer_splits=3, outer_repeats=1, inner_splits=2)

    permuted, unit = materialize_permuted_labels(
        y, config, "unique-group-null", 1, groups=groups
    )

    assert unit == "group"
    assert np.array_equal(np.bincount(permuted), np.bincount(y))


def test_mixed_group_permutations_are_flagged_as_sensitivity_not_primary_p() -> None:
    rng = np.random.default_rng(14)
    groups = np.repeat(np.arange(6), 4)
    y = np.tile([0, 0, 1, 1], 6)
    X = rng.normal(size=(y.size, 6))
    config = ProtocolConfig(
        outer_splits=3,
        outer_repeats=1,
        inner_splits=2,
        n_permutations=1,
    )

    result = evaluate_supervised_protocol(
        X, y, dataset_id="mixed-sensitivity", config=config, groups=groups
    )

    assert result["permutations"][0]["permutation_unit"] == "within_group"
    assert (
        result["permutations"][0]["permutation_protocol"]
        == "paired_repeated_measures_sensitivity"
    )
    for method in METHODS:
        null = result["methods"][method]["permutation"]
        assert null["primary_null_eligible"] is False
        assert null["empirical_p_fold_contained_mean_fold"] is None
        assert null["empirical_p_gap_oof"] is None


def test_end_to_end_evaluation_returns_paired_folds_features_oof_and_hashes() -> None:
    rng = np.random.default_rng(7)
    y = np.repeat([0, 1], 24)
    X = rng.normal(size=(48, 12))
    X[:, :3] += y[:, None] * 0.9
    X[::11, 5] = np.nan
    config = ProtocolConfig(
        outer_splits=3,
        outer_repeats=1,
        inner_splits=2,
        variance_cap=10,
        n_permutations=1,
        seed=13,
    )

    result = evaluate_supervised_protocol(
        X,
        y,
        dataset_id="synthetic",
        config=config,
        permutation_indices=(0,),
    )

    assert tuple(result["methods"]) == METHODS
    assert len(result["permutations"]) == 1
    assert len(result["permutations"][0]["label_sha256"]) == 64
    assert len(result["permutations"][0]["split_plan_sha256"]) == 64
    permuted_y, _ = materialize_permuted_labels(y, config, "synthetic", 0)
    expected_null_plan = materialize_plan(permuted_y, config, "synthetic")
    assert result["permutations"][0]["split_plan_sha256"] == expected_null_plan.sha256
    assert result["permutations"][0]["split_plan_sha256"] != result["split_plan_sha256"]
    assert len(result["baseline_l1"]["folds"]) == 3
    assert len(result["outer_fold_preprocessing"]) == 3
    assert all(
        len(record["eligible_features"]) == record["p_after_preprocessing"]
        for record in result["outer_fold_preprocessing"]
    )
    assert result["permutations"][0]["permutation_protocol"] == "primary_exchangeable"

    reference_tests = None
    for method in METHODS:
        observed = result["methods"][method]["observed"]
        folds = observed["fold_contained_folds"]
        pre_folds = observed["pre_split_folds"]
        stability = observed["feature_stability"]
        assert len(folds) == 3
        assert len(pre_folds) == 3
        assert set(observed).issuperset({
            "pre_split_mean_fold_auc",
            "fold_contained_mean_fold_auc",
            "gap_mean_fold",
            "pre_split_oof_auc",
            "fold_contained_oof_auc",
            "gap_oof",
        })
        assert len(
            result["methods"][method]["permutation"][
                "fold_contained_mean_fold_auc"
            ]
        ) == 1
        assert len(
            result["methods"][method]["permutation"]["gap_oof"]
        ) == 1
        assert result["methods"][method]["permutation"][
            "empirical_p_fold_contained_mean_fold"
        ] in {
            0.5,
            1.0,
        }
        assert result["methods"][method]["permutation"][
            "empirical_p_gap_oof"
        ] in {0.5, 1.0}
        current_tests = [tuple(fold["test_indices"]) for fold in folds]
        reference_tests = current_tests if reference_tests is None else reference_tests
        assert current_tests == reference_tests
        assert stability["common_eligible_p"] <= stability["raw_p"] == X.shape[1]
        assert "raw_universe_sensitivity" in stability
        assert [fold["chosen_params"] for fold in pre_folds] == [
            fold["chosen_params"] for fold in folds
        ]
        for fold in folds:
            assert len(fold["test_indices"]) == len(fold["y_true"])
            assert len(fold["y_true"]) == len(fold["y_score"])
            assert fold["p_after_preprocessing"] <= 10
            assert set(fold["selected_features"]).issubset(set(range(X.shape[1])))
            assert len(fold["inner_candidate_scores"]) == 4
            assert "eligible_features" not in fold

    baseline_tests = [tuple(fold["test_indices"]) for fold in result["baseline_l1"]["folds"]]
    assert baseline_tests == reference_tests


def test_preprocessing_workspace_is_label_free_reusable_and_result_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(31)
    y = np.repeat([0, 1], 12)
    X = rng.normal(size=(24, 8))
    X[::7, 3] = np.nan
    config = ProtocolConfig(
        outer_splits=2,
        outer_repeats=1,
        inner_splits=2,
        variance_cap=7,
        n_permutations=0,
    )
    plan = materialize_plan(y, config, "workspace")
    workspace = build_preprocessing_workspace(X, plan, config)

    cached = evaluate_observed(
        X, y, dataset_id="workspace", config=config, workspace=workspace
    )
    uncached = evaluate_observed(X, y, dataset_id="workspace", config=config)

    assert cached == uncached
    assert workspace.plan_sha256 == cached["split_plan_sha256"]
    assert not hasattr(workspace, "labels")

    calls = 0
    original = supervised_protocol.fit_preprocessor

    def counted(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(supervised_protocol, "fit_preprocessor", counted)
    block = evaluate_permutation_block(
        X,
        y,
        dataset_id="workspace-count",
        config=config,
        permutation_indices=(0, 1),
    )

    # One shared global fit, then outer + two inner fits for each scenario plan.
    assert calls == 1 + 2 * config.outer_splits * (1 + config.inner_splits)
    assert len(block["permutations"]) == 2


def test_permutation_index_does_not_change_model_seed_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(32)
    y = np.repeat([0, 1], 12)
    X = rng.normal(size=(24, 8))
    config = ProtocolConfig(
        outer_splits=2,
        outer_repeats=1,
        inner_splits=2,
        variance_cap=8,
    )
    captured: dict[str, dict[str, list[tuple[str, int]]]] = {}
    current_tag = ""
    original_scenario = supervised_protocol._evaluate_scenario
    original_bundle = supervised_protocol._fit_bundle
    original_auc = supervised_protocol._fit_auc

    def record_scenario(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal current_tag
        current_tag = str(args[5])
        captured[current_tag] = {"bundle": [], "auc": []}
        return original_scenario(*args, **kwargs)

    def record_bundle(method, X, y, config, seed):  # noqa: ANN001, ANN202
        captured[current_tag]["bundle"].append((method, int(seed)))
        return original_bundle(method, X, y, config, seed)

    def record_auc(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        captured[current_tag]["auc"].append((str(kwargs["phase"]), int(args[5])))
        return original_auc(*args, **kwargs)

    monkeypatch.setattr(supervised_protocol, "_evaluate_scenario", record_scenario)
    monkeypatch.setattr(supervised_protocol, "_fit_bundle", record_bundle)
    monkeypatch.setattr(supervised_protocol, "_fit_auc", record_auc)

    evaluate_permutation_block(
        X,
        y,
        dataset_id="fixed-model-seeds",
        config=config,
        permutation_indices=(0, 1),
    )

    assert captured["permutation:0"] == captured["permutation:1"]


def test_evaluation_rejects_nonbinary_or_too_small_input() -> None:
    config = ProtocolConfig(outer_splits=3, outer_repeats=1, inner_splits=2)

    with pytest.raises(ValueError, match="binary"):
        evaluate_supervised_protocol(
            np.ones((12, 3)),
            np.arange(12) % 3,
            dataset_id="bad-labels",
            config=config,
        )
    with pytest.raises(ValueError, match="at least"):
        evaluate_supervised_protocol(
            np.ones((6, 3)),
            np.array([0, 0, 1, 1, 1, 1]),
            dataset_id="too-small",
            config=config,
        )
