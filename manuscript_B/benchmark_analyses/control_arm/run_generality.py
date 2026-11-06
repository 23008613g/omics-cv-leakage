"""Exploratory generality check: two common pipelines outside the formal method set.

Requested after v5 ("does the pre-split inflation extend beyond the five formal
methods and the L2-logistic classifier?"). Thresholds and settings below were
fixed before any result of this script was seen; they are conventions, not tuned.

Pipelines (no tuning, so both arms differ only in where steps are fitted)
  vip  PLS-DA with 2 components on the preprocessed features; keep features with
       VIP > 1 (the usual metabolomics rule); formal L2-logistic classifier.
  rf   univariate top-20 features by two-sample F statistic; random forest
       (300 trees, max_features = sqrt, otherwise sklearn defaults).

Arms (as in the formal design)
  pre_split       label-free preprocessing and the supervised step fitted once on
                  all samples; classifier fitted on each outer training partition.
  fold_contained  preprocessing, supervised step and classifier fitted on each
                  outer training partition only.

Other steps use the formal algorithms, with release source hashes checked as in
``run_control_arm.py``: data loading, label-dependent split plans (5 folds x 3
repeats), preprocessing, permutation label vectors and the repeat-averaged pooled
out-of-fold AUC.

Modes
  observed  observed labels, all 1,086 analysed datasets.
  null      the 100-dataset subset of the v2 null check, formal permutations 0-19.
"""
from __future__ import annotations

import argparse
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from run_control_arm import RESULTS, _pooled_auc, _write_atomic, verify_formal_code   # binds benchmark_v4
from run_v2_arms import NULL_INDICES, _load, select_ids

PIPELINES = ("vip", "rf")
ARMS = ("pre_split", "fold_contained")
VIP_COMPONENTS, VIP_THRESHOLD = 2, 1.0
TOP_Q, RF_TREES = 20, 300
SPEC = {"vip": f"PLS-DA ({VIP_COMPONENTS} components) VIP > {VIP_THRESHOLD}; formal L2-logistic classifier",
        "rf": f"univariate top-{TOP_Q} by F statistic; random forest, {RF_TREES} trees, max_features=sqrt"}


# ------------------------------------------------------------------ supervised steps
def vip_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Variable importance in projection of a PLS-DA model (standard definition).

    VIP_j = sqrt(p * sum_a SS_a * (w_ja / ||w_a||)^2 / sum_a SS_a), with
    SS_a = q_a^2 * t_a' t_a the response variance explained by component a.
    By construction sum_j VIP_j^2 = p, which is asserted.
    """
    from sklearn.cross_decomposition import PLSRegression

    p = X.shape[1]
    a = int(min(VIP_COMPONENTS, p, X.shape[0] - 1))
    pls = PLSRegression(n_components=a, scale=False).fit(X, y.astype(float))
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    ss = (q[0] ** 2) * np.sum(t ** 2, axis=0)
    wn = w / np.linalg.norm(w, axis=0, keepdims=True)
    vip = np.sqrt(p * ((wn ** 2) @ ss) / ss.sum())
    assert np.isclose(np.sum(vip ** 2), p, rtol=1e-6), "VIP identity sum(VIP^2) = p violated"
    return vip


def select_vip(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    vip = vip_scores(X, y)
    keep = np.flatnonzero(vip > VIP_THRESHOLD)
    return keep if keep.size else np.array([int(np.argmax(vip))])


def select_topq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.feature_selection import f_classif

    with np.errstate(divide="ignore", invalid="ignore"):
        f, _ = f_classif(X, y)
    f = np.where(np.isfinite(f), f, -np.inf)
    return np.sort(np.argsort(-f, kind="stable")[: min(TOP_Q, X.shape[1])])


SELECT = {"vip": select_vip, "rf": select_topq}


# ------------------------------------------------------------------ classifiers
def classify(pipe, train_x, train_y, test_x, test_y, config, seed, phase):
    from sklearn.metrics import roc_auc_score
    from benchmark_v4.protocol import _fit_auc

    if pipe == "vip":
        auc, scores, _ = _fit_auc(train_x, train_y, test_x, test_y, config, seed, phase=phase)
        return np.asarray(scores, dtype=float), float(auc)
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=RF_TREES, max_features="sqrt", random_state=seed, n_jobs=1)
    scores = model.fit(train_x, train_y).predict_proba(test_x)[:, 1]
    return np.asarray(scores, dtype=float), float(roc_auc_score(test_y, scores))


# ------------------------------------------------------------------ one label vector
def evaluate_labels(X, y, ds, config, dataset_id, global_x, tag) -> dict:
    from benchmark_v4.protocol import fit_preprocessor
    from benchmark_v4.protocol_estimands import _fold_record
    from benchmark_v4.splitting import derive_seed, materialize_plan

    plan = materialize_plan(y, config, dataset_id, groups=ds.groups)
    pre_sel = {pipe: SELECT[pipe](global_x, y) for pipe in PIPELINES}      # fitted on all samples
    folds = {(pipe, arm): [] for pipe in PIPELINES for arm in ARMS}
    n_con = {pipe: [] for pipe in PIPELINES}
    for k, split in enumerate(plan.outer_splits):
        train = np.asarray(split.train_indices, dtype=int)
        test = np.asarray(split.test_indices, dtype=int)
        prep = fit_preprocessor(X[train], config, transform_kind=ds.transform_kind)
        x_tr, x_te = prep.transform(X[train]), prep.transform(X[test])
        for pipe in PIPELINES:
            con_sel = SELECT[pipe](x_tr, y[train])
            n_con[pipe].append(int(con_sel.size))
            views = {"pre_split": (global_x[train], global_x[test], pre_sel[pipe]),
                     "fold_contained": (x_tr, x_te, con_sel)}
            for arm, (a, b, sel) in views.items():
                seed = derive_seed(config.seed, dataset_id, k, "generality", pipe, arm)
                scores, auc = classify(pipe, a[:, sel], y[train], b[:, sel], y[test], config, seed,
                                       phase=f"{pipe}:{arm}:{tag}:{k}")
                folds[(pipe, arm)].append(_fold_record(k, split, y[test], scores, auc, {}, sel, a.shape[1], []))
    out = {}
    for pipe in PIPELINES:
        out[pipe] = {f"{arm}_oof_auc": _pooled_auc(folds[(pipe, arm)], y.size, config.outer_repeats,
                                                   f"{dataset_id}.{tag}.{pipe}.{arm}") for arm in ARMS}
        out[pipe].update({"n_selected_pre_split": int(pre_sel[pipe].size),
                          "n_selected_fold_contained_median": float(np.median(n_con[pipe]))})
    return out


def mode_observed(dataset_id, platform, entry):
    from benchmark_v4.protocol import build_global_preprocessing_workspace

    config, ds, X, y = _load(dataset_id, platform, entry)
    gx = build_global_preprocessing_workspace(X, config, transform_kind=ds.transform_kind).transformed_x
    return {"dataset_id": dataset_id, "status": "ok", "n_samples": int(y.size), "spec": SPEC,
            "pipelines": evaluate_labels(X, y, ds, config, dataset_id, gx, "observed")}


def mode_null(dataset_id, platform, entry):
    from benchmark_v4.protocol import build_global_preprocessing_workspace, materialize_permuted_labels

    config, ds, X, y = _load(dataset_id, platform, entry)
    gx = build_global_preprocessing_workspace(X, config, transform_kind=ds.transform_kind).transformed_x
    perms = {}
    for index in NULL_INDICES:
        y_p, unit = materialize_permuted_labels(y, config, dataset_id, index, groups=ds.groups)
        perms[str(index)] = {"unit": unit,
                             "pipelines": evaluate_labels(X, y_p, ds, config, dataset_id, gx, f"perm{index}")}
    return {"dataset_id": dataset_id, "status": "ok", "n_samples": int(y.size), "spec": SPEC, "permutations": perms}


MODES = {"observed": mode_observed, "null": mode_null}


def _run(args):
    mode, dataset_id, platform, entry = args
    t0 = time.time()
    try:
        res = MODES[mode](dataset_id, platform, entry)
    except Exception as exc:                                  # recorded, never swallowed
        res = {"dataset_id": dataset_id, "status": "error", "error": repr(exc),
               "traceback": traceback.format_exc(limit=6)}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description="exploratory generality check (VIP and random forest)")
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default=None, help="results subdirectory (default v4_generality_<mode>)")
    args = parser.parse_args()

    from benchmark_v4 import raw_loaders

    code = verify_formal_code()
    if not all(code.values()):
        raise RuntimeError(f"formal code mismatch: {[k for k, v in code.items() if not v]}")
    ids = select_ids("pretuned" if args.mode == "observed" else "nullcheck")
    ids = ids[: args.limit] if args.limit else ids
    out_dir = RESULTS / (args.out or f"v4_generality_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (out_dir / f"{i}.json").exists()]
    resolved = raw_loaders.entries_for_whitelist(todo) if todo else []
    jobs = [(args.mode, i, p, dict(e)) for i, (p, e) in zip(todo, resolved, strict=True)]
    print(f"[{args.mode}] formal code 18/18 verified; {len(ids)} selected, {len(jobs)} to run "
          f"with {args.workers} workers -> {out_dir}", flush=True)
    t0, done, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_run, j) for j in jobs]):
            res = fut.result()
            _write_atomic(out_dir / f"{res['dataset_id']}.json", res)
            done += 1
            failed += res.get("status") != "ok"
            if done % 25 == 0 or done == len(jobs):
                eta = (time.time() - t0) / done * (len(jobs) - done) / 60
                print(f"  {done}/{len(jobs)}  not-ok {failed}  ~{eta:.0f} min left", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
