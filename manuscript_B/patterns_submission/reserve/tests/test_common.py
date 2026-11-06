"""Checks of the reserve-analysis code against the benchmark code it extends."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C                                         # noqa: E402

from presplit_audit import RFE, StandardPreprocessor, evaluate_protocols, make_splits   # noqa: E402

CFG = C.formal_config()


def binary_data(n=60, p=300, shift=1.0, seed=2):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(size=(n, p))
    X[:, :5] += shift * y[:, None]
    return X, y


@pytest.mark.parametrize("method", C.V.METHODS)
def test_generic_evaluator_reproduces_the_tool_on_binary_data(method):
    X, y = binary_data()
    folds = make_splits(y, n_splits=5, n_repeats=2, inner_splits=3, random_state=4)
    seeds = C.V.formal_seeds(CFG, "toy", method)
    tool = evaluate_protocols(X, y, folds, C.make_step("binary", method, CFG), preprocessor=StandardPreprocessor,
                              seeds=seeds)
    mine = C.evaluate_generic(X, y, folds, C.make_step("binary", method, CFG), preprocessor=StandardPreprocessor,
                              task="binary", seeds=seeds)
    assert mine["pre_split"] == tool["pre_split_auc"]
    assert mine["fold_contained"] == tool["fold_contained_auc"]
    assert mine["chosen_candidates"] == tool["chosen_candidates"]


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_preprocessing_cache_leaves_results_unchanged(task):
    X, y = binary_data()
    if task == "multiclass":
        y = np.arange(y.size) % 3
        X[:, :3] += y[:, None]
    elif task == "regression":
        y = X[:, 0] + np.random.default_rng(5).normal(size=y.size)
    folds = C.generic_plan(y, None, "toy", CFG, stratify=task != "regression")
    cache = {}
    for method in C.TASK_METHODS[task]:
        seeds = C.V.formal_seeds(CFG, "toy", method)
        plain = C.evaluate_generic(X, y, folds, C.make_step(task, method, CFG), preprocessor=StandardPreprocessor,
                                   task=task, seeds=seeds)
        cached = C.evaluate_generic(X, y, folds, C.make_step(task, method, CFG), preprocessor=StandardPreprocessor,
                                    task=task, seeds=seeds, cache=cache)
        assert plain == cached
    assert len(cache) == 1 + len(folds) * (1 + CFG.inner_splits)


def test_generic_evaluator_reproduces_the_formal_values_on_a_real_dataset():
    from nm_v4 import raw_loaders
    from nm_v4.splitting import materialize_plan
    (platform, entry), = raw_loaders.entries_for_whitelist(["crc_xiang"])
    cfg, ds, X, y = C.V._load("crc_xiang", platform, dict(entry))
    folds = C.V.formal_folds(materialize_plan(y, cfg, "crc_xiang", groups=ds.groups))
    formal = C.V.checkpoint("crc_xiang")["observed"]["methods"]
    for method in C.V.METHODS:
        out = C.evaluate_generic(X, y, folds, C.make_step("binary", method, cfg),
                                 preprocessor=C.preprocessor_for(cfg, ds.transform_kind), task="binary",
                                 seeds=C.V.formal_seeds(cfg, "crc_xiang", method))
        assert out["pre_split"] == formal[method]["observed"]["pre_split_oof_auc"]
        assert out["fold_contained"] == formal[method]["observed"]["fold_contained_oof_auc"]


def test_multiclass_rfe_with_two_classes_is_the_benchmark_rfe():
    X, y = binary_data(p=120)
    Z = StandardPreprocessor().fit(X).transform(X)
    ours = C.RFEMulticlass(svc_c=CFG.rfe_svc_c, max_iter=CFG.rfe_max_iter).fit_bundle(Z, y, seed=7)
    theirs = RFE(svc_c=CFG.rfe_svc_c, max_iter=CFG.rfe_max_iter).fit_bundle(Z, y, seed=7)
    probe = np.arange(Z.shape[1])[None, :].astype(float)
    for a, b in zip(ours, theirs):
        assert np.array_equal(a(probe), b(probe))


def test_permutation_is_the_formal_one_without_groups():
    from nm_v4.protocol import materialize_permuted_labels
    y = np.repeat([0, 1], [23, 31])
    for i in range(5):
        formal, unit = materialize_permuted_labels(y, CFG, "toy", i)
        assert unit == "sample" and np.array_equal(C.permuted(y, None, "toy", i, CFG), formal)


def test_permutation_moves_whole_groups_when_labels_are_constant_within_groups():
    groups = np.repeat(np.arange(12), 3).astype(object)
    y = np.repeat([0, 1] * 6, 3)
    yp = C.permuted(y, groups, "toy", 0, CFG)
    for g in np.unique(groups):
        assert np.unique(yp[groups == g]).size == 1
    assert sorted(yp) == sorted(y)


@pytest.mark.parametrize("stratify", [True, False])
def test_generic_plan_is_valid_stratified_and_group_aware(stratify):
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 30) if stratify else rng.normal(size=90)
    groups = np.repeat(np.arange(45), 2).astype(object)
    folds = C.generic_plan(y, groups, "toy", CFG, stratify=stratify)
    assert len(folds) == CFG.outer_splits * CFG.outer_repeats
    for f in folds:
        assert not set(groups[f.train]) & set(groups[f.test])
        if stratify:
            assert np.unique(y[f.test]).size == 3
            for tr, te in f.inner:
                assert np.unique(y[te]).size == 3


def test_final_model_uses_the_most_frequent_candidate_with_ties_to_the_earlier():
    X, y = binary_data()
    step = C.make_step("binary", "top_k", CFG)
    _, best = C.final_model_predictions(X, y, X[:5], step, [2, 1, 1, 2, 3], preprocessor=StandardPreprocessor,
                                        task="binary", rep_seed=0, clf_seed=0)
    assert best == 1


def test_genus_counts_sum_the_named_otus():
    s = C.load_study("crc_xiang")
    mask = np.ones(len(s.sample_ids), bool)
    g = C.genus_counts(s, mask)
    named = np.array([C.genus_lineage(t) is not None for t in s.taxonomy])
    assert np.allclose(g.to_numpy().sum(axis=1), s.counts[:, named].sum(axis=1))
    assert all(name.split(";")[-1].startswith("g__") and name.split(";")[-1] != "g__" for name in g.columns)


def test_generic_plan_reduces_folds_to_the_groups_per_class_and_retries_until_every_class_is_present():
    groups = np.repeat(np.arange(12), 2).astype(object)            # 12 participants, 4 per class
    y = np.repeat(np.arange(12) % 3, 2)
    folds = C.generic_plan(y, groups, "toy-groups", CFG, stratify=True)
    assert len(folds) == 4 * CFG.outer_repeats                      # 5 folds reduced to 4 groups per class
    for f in folds:
        assert np.unique(y[f.test]).size == 3 and not set(groups[f.train]) & set(groups[f.test])
        for tr, te in f.inner:
            assert np.unique(y[te]).size == 3


def test_first_seed_of_the_retry_is_the_plain_seed():
    y = np.repeat([0, 1, 2], 20)
    a = C.generic_plan(y, None, "toy", CFG, stratify=True)
    b = C._parts(y, None, CFG.outer_splits, C.derive_seed(CFG.seed, "toy", "outer", 0), True)
    assert all(np.array_equal(f.test, t) for f, (_, t) in zip(a[:CFG.outer_splits], b))


def test_null_labels_are_shared_within_participants():
    O = pytest.importorskip("other_tasks")           # not deposited with the cross-cohort analysis
    groups = np.repeat(np.arange(15), 2).astype(object)
    for task in ("multiclass", "regression"):
        y = O.null_labels(task, 30, groups, np.random.default_rng(0))
        for g in np.unique(groups):
            assert np.unique(y[groups == g]).size == 1
    y = O.null_labels("multiclass", 30, groups, np.random.default_rng(0))
    assert sorted(np.bincount(y[::2])) == [5, 5, 5]
