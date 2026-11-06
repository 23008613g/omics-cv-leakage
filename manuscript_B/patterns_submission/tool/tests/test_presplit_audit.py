import json

import numpy as np
import pytest

from presplit_audit import (HTSPLS, OPLSDA, PLSDA, RFE, CustomStep, StandardPreprocessor, TopQ, VIPSelect, audit,
                            empirical_p, evaluate_protocols, make_splits, permute_labels, validate_splits)
from presplit_audit.cli import main
from presplit_audit.risk import expected_presplit_null, main as risk_main
from presplit_audit.core import select_best


def null_data(n=40, p=500, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), np.repeat([0, 1], n // 2)


def signal_data(n=60, p=200, shift=2.0, seed=2):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(size=(n, p))
    X[:, :5] += shift * y[:, None]
    return X, y


def test_splits_partition_each_repeat_and_keep_groups_together():
    y = np.repeat([0, 1], 30)
    groups = np.repeat(np.arange(20), 3)
    folds = make_splits(y, groups, n_splits=4, n_repeats=2, inner_splits=3, random_state=3)
    validate_splits(folds, y.size, groups=groups)
    assert {f.repeat for f in folds} == {0, 1}
    for f in folds:
        assert not set(groups[f.train]) & set(groups[f.test])


def test_validate_splits_rejects_leaky_plan():
    y = np.repeat([0, 1], 10)
    folds = make_splits(y, n_splits=2, n_repeats=1)
    bad = folds[0].__class__(0, np.arange(20), folds[0].test, folds[0].inner)
    with pytest.raises(ValueError):
        validate_splits([bad, folds[1]], y.size)


def test_preprocessor_uses_training_rows_only():
    Xtr = np.array([[1.0, 5.0, np.nan], [3.0, 5.0, 2.0], [5.0, 5.0, np.nan]])
    prep = StandardPreprocessor(completeness=0.5).fit(Xtr)
    assert list(prep.features_) == [0]                     # column 1 constant, column 2 incomplete
    out = prep.transform(np.array([[3.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]))
    assert out[0, 0] == pytest.approx(0.0) and out[1, 0] == pytest.approx(0.0)   # mean 3, median 3


def test_tie_breaking_prefers_first_candidate():
    assert select_best([0.7, 0.9, 0.9]) == 1
    assert select_best([np.nan, 0.6, 0.6]) == 1


def test_presplit_selection_inflates_auc_on_noise():
    X, y = null_data()
    folds = make_splits(y, random_state=0)
    r = evaluate_protocols(X, y, folds, TopQ())
    assert r["pre_split_auc"] > 0.8
    assert abs(r["fold_contained_auc"] - 0.5) < 0.25
    assert r["gap"] == pytest.approx(r["pre_split_auc"] - r["fold_contained_auc"])


def test_real_signal_gives_high_auc_in_both_arms():
    X, y = signal_data()
    r = evaluate_protocols(X, y, make_splits(y, random_state=0), PLSDA())
    assert r["fold_contained_auc"] > 0.9 and r["pre_split_auc"] > 0.9


def test_vip_and_custom_steps_run():
    X, y = signal_data(p=50)
    folds = make_splits(y, n_repeats=1, random_state=0)
    assert evaluate_protocols(X, y, folds, VIPSelect())["fold_contained_auc"] > 0.8
    first = CustomStep("first_k", lambda X, y, k: (lambda Z: Z[:, :k]), [{"k": 3}, {"k": 10}])
    assert evaluate_protocols(X, y, folds, first)["fold_contained_auc"] > 0.8


def test_group_permutation_keeps_group_labels_constant():
    y = np.repeat([0, 1], 12)
    groups = np.repeat(np.arange(8), 3)
    yp = permute_labels(y, groups, np.random.default_rng(0))
    assert all(np.unique(yp[groups == g]).size == 1 for g in range(8))
    assert yp.sum() == y.sum()


def test_within_group_permutation_when_labels_vary_within_groups():
    y = np.tile([0, 1], 10)
    groups = np.repeat(np.arange(10), 2)
    yp = permute_labels(y, groups, np.random.default_rng(0))
    assert all(sorted(yp[groups == g]) == [0, 1] for g in range(10))


def test_empirical_p_counts_ties_against_detection():
    assert empirical_p(0.8, [0.8, 0.5, 0.9, 0.1]) == pytest.approx(3 / 5)
    assert empirical_p(1.0, [0.5] * 19) == pytest.approx(1 / 20)


def test_audit_permutation_null_shows_presplit_optimism():
    X, y = null_data(n=30, p=300)
    res = audit(X, y, "top_q", n_permutations=9, random_state=0)
    assert res.n_permutations == 9 and len(res.pre_split_null) == 9
    assert np.median(res.pre_split_null) > 0.75             # the pre-split null is itself inflated
    assert abs(np.median(res.fold_contained_null) - 0.5) < 0.2
    assert 0 < res.p_pre_split <= 1 and 0 < res.p_fold_contained <= 1
    assert "Fold-contained AUC" in res.summary()


def test_audit_accepts_string_labels_and_rejects_multiclass():
    X, y = signal_data(p=30)
    res = audit(X, np.where(y == 1, "case", "control"), "top_q", n_permutations=0)
    assert res.fold_contained_auc > 0.9
    with pytest.raises(ValueError):
        audit(X, np.arange(y.size) % 3, n_permutations=0)


def test_cli_writes_json(tmp_path):
    import pandas as pd
    X, y = signal_data(n=40, p=20)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)]).assign(label=y, sample=range(40))
    path, out = tmp_path / "d.csv", tmp_path / "r.json"
    frame.to_csv(path, index=False)
    assert main([str(path), "--label", "label", "--drop", "sample", "--permutations", "3",
                 "--json", str(out)]) == 0
    saved = json.loads(out.read_text())
    assert saved["n_permutations"] == 3 and saved["n_features"] == 20


def test_rfe_follows_one_nested_elimination_path():
    X, y = signal_data(n=40, p=120)
    bundle = RFE().fit_bundle(X, y, seed=3)
    cols = [set(np.flatnonzero(np.isin(np.arange(120), t(np.arange(120)[None, :].astype(float))[0]))) for t in bundle]
    sizes = [len(c) for c in cols]
    assert sizes == [5, 10, 20, 50]                          # candidate order of the formal grid
    assert cols[0] <= cols[1] <= cols[2] <= cols[3]          # pruned 50 -> 20 -> 10 -> 5 on one path
    again = RFE().fit_bundle(X, y, seed=3)
    assert all(np.array_equal(a(X), b(X)) for a, b in zip(bundle, again))


def test_ht_spls_and_opls_da_shapes_and_definitions():
    X, y = signal_data(n=40, p=60)
    outs = [t(X) for t in HTSPLS().fit_bundle(X, y)]
    assert [o.shape[1] for o in outs] == [1, 1, 2, 2]
    yc = y - y.mean()
    w = X.T @ yc
    w /= np.linalg.norm(w)
    opls = OPLSDA().fit_bundle(X, y)
    assert np.allclose(opls[0](X)[:, 0], X @ w)            # no orthogonal component: predictive score only
    assert all(t(X).shape == (40, 1) for t in opls)


def test_new_steps_inflate_presplit_auc_on_noise():
    X, y = null_data(n=40, p=300)
    folds = make_splits(y, n_repeats=1, random_state=0)
    for step in (RFE(), HTSPLS(), OPLSDA()):
        r = evaluate_protocols(X, y, folds, step)
        assert r["pre_split_auc"] > r["fold_contained_auc"] + 0.15, step.name


def test_seed_schedule_reaches_every_fit_and_default_is_constant():
    X, y = signal_data(n=40, p=80)
    folds = make_splits(y, n_repeats=1, random_state=1)
    seen = []

    def schedule(*context):
        seen.append(context[0])
        return 7

    a = evaluate_protocols(X, y, folds, RFE(), seed=7)
    b = evaluate_protocols(X, y, folds, RFE(), seeds=schedule)
    assert a == b                                           # a constant schedule equals the default
    assert set(seen) == {"global", "inner_representation", "inner_classifier", "pre_split_classifier",
                         "outer_representation", "fold_contained_classifier"}


def test_cli_offers_all_five_methods(tmp_path):
    import pandas as pd
    X, y = signal_data(n=40, p=20)
    path = tmp_path / "d.csv"
    pd.DataFrame(X, columns=[f"f{i}" for i in range(20)]).assign(label=y).to_csv(path, index=False)
    for step in ("rfe", "ht_spls", "opls_da"):
        assert main([str(path), "--label", "label", "--step", step, "--permutations", "0"]) == 0


def test_risk_estimate_falls_with_n_and_rises_with_p():
    for step in ("top_q", "rfe", "ht_spls", "pls_da", "opls_da"):
        small, large = expected_presplit_null(35, 3000, step), expected_presplit_null(300, 3000, step)
        few, many = expected_presplit_null(60, 150, step), expected_presplit_null(60, 20000, step)
        assert small["supported"] and large["supported"] and few["supported"] and many["supported"]
        assert small["auc"] >= large["auc"] and many["auc"] >= few["auc"], step
        lo, hi = small["range80"]
        assert lo <= small["auc"] <= hi


def test_risk_estimate_refuses_unsupported_designs():
    with pytest.warns(UserWarning):
        r = expected_presplit_null(10, 500, "pls_da")
    assert not r["supported"] and np.isnan(r["auc"])
    with pytest.raises(ValueError):
        expected_presplit_null(50, 500, "lasso")


def test_risk_cli_runs(capsys):
    assert risk_main(["--samples", "56", "--features", "1875"]) == 0
    assert "typical pre-split AUC" in capsys.readouterr().out
