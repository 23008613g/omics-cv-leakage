"""Additional benchmark analyses for the paired protocol comparison.

Four modes using the formal algorithms in ``benchmark_v4`` (release sources bound and hash-checked
exactly as in ``run_control_arm.py``):

gradient   Within-dataset signal gradient. For each dataset eligible for
           exchangeable permutation, a fraction f in {0.25, 0.50, 0.75} of
           permutation units (samples, or participants when labels are constant
           within participant) have their labels permuted among themselves, and
           the complete formal pipeline (split plan, preprocessing, tuning, both
           arms) is rerun. f = 0 is the observed run and f = 1 the permutation
           null, both already in the deposit. Difficulty is therefore set by
           design, so the gap-difficulty relation cannot be produced by shared
           estimation noise.

holdout    External check of each arm's estimate. For datasets with n >= 100 and
           at least 30 minority observations, 25% of samples (or participants)
           are held out untouched. Both arms are cross-validated on the remaining
           75%; the final model (preprocessing, representation and classifier
           fitted on the 75%, setting = the candidate chosen most often across
           outer folds) is scored once on the held-out 25%. The two arms share
           that final model, so each arm's cross-validated AUC minus the held-out
           AUC is that arm's estimation error.

pretuned   Pre-split arm with tuning also done on the pre-split representation
           (inner cross-validation over the globally fitted candidates), the
           common practice the formal pre-split arm does not mimic. Observed
           labels, all analysed datasets.

nullcheck  For 100 randomly chosen exchangeable datasets and formal permutation
           indices 0-19: reproduce the formal pre-split and fold-contained null
           AUCs, and compute the swap, global-preprocessing and fully pre-split
           tuned arms under the same permuted labels.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from run_control_arm import (RESULTS, TABLES, _candidate_index, _pooled_auc,  # binds benchmark_v4
                             _write_atomic, verify_formal_code)

FRACTIONS = (0.25, 0.50, 0.75)
NULL_INDICES = tuple(range(20))
NULL_SUBSET = 100
HOLDOUT_SHARE = 0.25


# ------------------------------------------------------------------ shared
def _load(dataset_id: str, platform: str, entry: dict):
    from benchmark_v4 import raw_loaders
    from benchmark_v4.protocol import ProtocolConfig, _validated_xy

    config = ProtocolConfig(n_permutations=199)
    dataset, reason = raw_loaders.load_raw(platform, entry)
    if dataset is None or reason != "ok":
        raise RuntimeError(f"load failed: {reason}")
    X, y = _validated_xy(dataset.X, dataset.y)
    return config, dataset, X, y


def _scenario(X, y, config, dataset_id, dataset, global_cache, tag):
    from benchmark_v4.protocol import _evaluate_scenario, build_preprocessing_workspace
    from benchmark_v4.splitting import materialize_plan

    plan = materialize_plan(y, config, dataset_id, groups=dataset.groups)
    ws = build_preprocessing_workspace(X, plan, config, transform_kind=dataset.transform_kind,
                                       global_workspace=global_cache)
    return plan, ws, _evaluate_scenario(y, plan, ws, config, dataset_id, tag, include_baseline=False)


def _two_arm_summary(scenario, n, repeats, ctx) -> dict:
    out = {}
    for method, res in scenario["methods"].items():
        out[method] = {"pre_split_oof_auc": _pooled_auc(res["pre_split_folds"], n, repeats, ctx + method + ".pre"),
                       "fold_contained_oof_auc": _pooled_auc(res["fold_contained_folds"], n, repeats,
                                                             ctx + method + ".con")}
    return out


def extra_arms(y, plan, ws, scenario, config, dataset_id, seed_tag, arms) -> dict:
    """swap / global_prep / pre_tuned pooled OOF AUCs for any label vector."""
    from benchmark_v4.methods import METHODS
    from benchmark_v4.protocol import _fit_auc, _fit_bundle, _fold_record, select_best_candidate
    from benchmark_v4.splitting import derive_seed

    global_x = ws.global_workspace.transformed_x
    n, repeats = y.size, config.outer_repeats
    out = {}
    for method in METHODS:
        pre_folds = scenario["methods"][method]["pre_split_folds"]
        folds = {a: [] for a in arms}
        global_bundle = None
        if "pre_tuned" in arms:
            global_bundle = _fit_bundle(method, global_x, y, config,
                                        derive_seed(config.seed, dataset_id, method, "global"))
            global_z = [rep.transform(global_x) for rep in global_bundle]
        for split, pre_fold, prepared in zip(plan.outer_splits, pre_folds, ws.outer, strict=True):
            k = pre_fold["outer_index"]
            train = np.asarray(split.train_indices, dtype=int)
            test = np.asarray(split.test_indices, dtype=int)
            chosen = _candidate_index(method, pre_fold["chosen_params"])
            parts = (dataset_id, k, method)
            rep_seed = derive_seed(config.seed, *parts, "outer_representation")
            con_seed = derive_seed(config.seed, *parts, "contained_final")

            def rec(arm, scores, auc, params=pre_fold["chosen_params"]):
                if folds[arm] is None:
                    return
                folds[arm].append(_fold_record(k, split, y[test], scores, auc, params, [], 0, []))

            if "swap" in arms and folds.get("swap") is not None and any(
                    np.sum(y[test] == c) > np.sum(y[train] == c) for c in (0, 1)):
                folds["swap"] = None                          # not estimable, reported as such
            if "swap" in arms and folds.get("swap") is not None:
                rng = np.random.default_rng(derive_seed(config.seed, dataset_id, k, "swap_subset", seed_tag))
                drop = np.concatenate([rng.choice(np.flatnonzero(y[train] == c),
                                                  size=int(np.sum(y[test] == c)), replace=False) for c in (0, 1)])
                keep = np.setdiff1d(np.arange(train.size), drop)
                rep = _fit_bundle(method, np.vstack([prepared.train_x[keep], prepared.test_x]),
                                  np.concatenate([y[train][keep], y[test]]), config, rep_seed)[chosen]
                auc, sc, _ = _fit_auc(rep.transform(prepared.train_x), y[train], rep.transform(prepared.test_x),
                                      y[test], config, con_seed, phase=f"{method}:swap:{k}")
                rec("swap", sc, auc)
            if "global_prep" in arms:
                rep = _fit_bundle(method, global_x[train], y[train], config, rep_seed)[chosen]
                auc, sc, _ = _fit_auc(rep.transform(global_x[train]), y[train], rep.transform(global_x[test]),
                                      y[test], config, con_seed, phase=f"{method}:global-prep:{k}")
                rec("global_prep", sc, auc)
            if "pre_tuned" in arms:
                means = []
                for c, z in enumerate(global_z):
                    scores = []
                    for i, inner in enumerate(prepared.inner):
                        tr = np.asarray(inner.train_indices, dtype=int)
                        te = np.asarray(inner.test_indices, dtype=int)
                        a, _, _ = _fit_auc(z[tr], y[tr], z[te], y[te], config,
                                           derive_seed(config.seed, *parts, "pre_tuned_inner", i, c),
                                           phase=f"{method}:pre-tuned-inner:{k}:{i}:{c}")
                        scores.append(a)
                    means.append(float(np.mean(scores)))
                best = select_best_candidate(means)
                z = global_z[best]
                auc, sc, _ = _fit_auc(z[train], y[train], z[test], y[test], config,
                                      derive_seed(config.seed, *parts, "pre_split_final"),
                                      phase=f"{method}:pre-tuned:{k}")
                rec("pre_tuned", sc, auc, params={"candidate_index": best})
        out[method] = {f"{a}_oof_auc": (None if f is None else
                                        _pooled_auc(f, n, repeats, f"{dataset_id}.{method}.{a}"))
                       for a, f in folds.items()}
    return out


# ------------------------------------------------------------------ modes
def partial_labels(y, groups, config, dataset_id, frac):
    from benchmark_v4.protocol import _permutation_unit
    from benchmark_v4.splitting import derive_seed

    unit = _permutation_unit(y, config, groups)
    rng = np.random.default_rng(derive_seed(config.seed, dataset_id, "partial_permutation", f"{frac:.2f}"))
    if unit == "sample":
        k = int(round(frac * y.size))
        idx = rng.choice(y.size, size=k, replace=False)
        out = y.copy()
        out[idx] = y[idx][rng.permutation(k)]
        return out, unit
    if unit == "group":
        g = np.asarray(groups, dtype=object)
        uniq, inv = np.unique(g, return_inverse=True)
        lab = np.array([int(y[inv == j][0]) for j in range(uniq.size)])
        k = int(round(frac * uniq.size))
        idx = rng.choice(uniq.size, size=k, replace=False)
        new = lab.copy()
        new[idx] = lab[idx][rng.permutation(k)]
        return new[inv].astype(int), unit
    return None, unit                                        # within-participant designs are skipped


def mode_gradient(dataset_id, platform, entry, seed_tag):
    from benchmark_v4.protocol import build_global_preprocessing_workspace

    config, ds, X, y = _load(dataset_id, platform, entry)
    cache = build_global_preprocessing_workspace(X, config, transform_kind=ds.transform_kind)
    out = {"dataset_id": dataset_id, "status": "ok", "levels": {}}
    for f in FRACTIONS:
        y_f, unit = partial_labels(y, ds.groups, config, dataset_id, f)
        if y_f is None:
            return {"dataset_id": dataset_id, "status": f"skipped:{unit}"}
        _, _, sc = _scenario(X, y_f, config, dataset_id, ds, cache, f"partial:{f:.2f}")
        out["levels"][f"{f:.2f}"] = {"unit": unit, "share_labels_changed": float(np.mean(y_f != y)),
                                     "methods": _two_arm_summary(sc, y.size, config.outer_repeats,
                                                                 f"{dataset_id}.p{f}.")}
    return out


def _holdout_split(y, groups, config, dataset_id):
    from benchmark_v4.splitting import derive_seed, validate_groups

    rng = np.random.default_rng(derive_seed(config.seed, dataset_id, "holdout_split"))
    g = None if groups is None else np.asarray(groups, dtype=object)
    group_aware, _ = validate_groups(g, y, config.outer_splits)
    for _attempt in range(200):
        if group_aware:
            uniq = np.unique(g)
            order = rng.permutation(uniq.size)
            test_groups, count = [], 0
            for j in order:
                if count >= HOLDOUT_SHARE * y.size:
                    break
                test_groups.append(uniq[j])
                count += int(np.sum(g == uniq[j]))
            test = np.flatnonzero(np.isin(g, test_groups))
        else:
            test = np.concatenate([rng.choice(np.flatnonzero(y == c),
                                              size=int(round(HOLDOUT_SHARE * np.sum(y == c))), replace=False)
                                   for c in (0, 1)])
        dev = np.setdiff1d(np.arange(y.size), test)
        if np.unique(y[test]).size == 2 and min(np.bincount(y[dev], minlength=2)) >= 10:
            return np.sort(dev), np.sort(test), group_aware
    raise RuntimeError("no feasible holdout split")


def mode_holdout(dataset_id, platform, entry, seed_tag):
    from benchmark_v4.methods import METHODS
    from benchmark_v4.protocol import _fit_auc, _fit_bundle, fit_preprocessor
    from benchmark_v4.splitting import derive_seed

    config, ds, X, y = _load(dataset_id, platform, entry)
    dev, test, grouped = _holdout_split(y, ds.groups, config, dataset_id)
    sub_id = f"{dataset_id}|holdout-dev"

    class _Dev:
        groups = None if ds.groups is None else np.asarray(ds.groups, dtype=object)[dev]
        transform_kind = ds.transform_kind

    from benchmark_v4.protocol import build_global_preprocessing_workspace
    cache = build_global_preprocessing_workspace(X[dev], config, transform_kind=ds.transform_kind)
    _, _, sc = _scenario(X[dev], y[dev], config, sub_id, _Dev, cache, "holdout-dev")
    cv = _two_arm_summary(sc, dev.size, config.outer_repeats, f"{sub_id}.")
    prep = fit_preprocessor(X[dev], config, transform_kind=ds.transform_kind)
    x_dev, x_test = prep.transform(X[dev]), prep.transform(X[test])
    out = {"dataset_id": dataset_id, "status": "ok", "n_dev": int(dev.size), "n_test": int(test.size),
           "group_holdout": bool(grouped), "methods": {}}
    for method in METHODS:
        chosen_counts = Counter(_candidate_index(method, f["chosen_params"])
                                for f in sc["methods"][method]["pre_split_folds"])
        chosen = min(c for c, v in chosen_counts.items() if v == max(chosen_counts.values()))
        rep = _fit_bundle(method, x_dev, y[dev], config,
                          derive_seed(config.seed, dataset_id, method, "holdout_final"))[chosen]
        auc, _, _ = _fit_auc(rep.transform(x_dev), y[dev], rep.transform(x_test), y[test], config,
                             derive_seed(config.seed, dataset_id, method, "holdout_final_clf"),
                             phase=f"{method}:holdout-final")
        out["methods"][method] = {**cv[method], "heldout_auc": float(auc), "final_candidate": int(chosen)}
    return out


def mode_pretuned(dataset_id, platform, entry, seed_tag):
    config, ds, X, y = _load(dataset_id, platform, entry)
    plan, ws, sc = _scenario(X, y, config, dataset_id, ds, None, "observed")
    return {"dataset_id": dataset_id, "status": "ok",
            "methods": {m: {**v, **_two_arm_summary(sc, y.size, config.outer_repeats, f"{dataset_id}.")[m]}
                        for m, v in extra_arms(y, plan, ws, sc, config, dataset_id, seed_tag,
                                               ("pre_tuned",)).items()}}


CHECKPOINTS = TABLES.parent / "checkpoints"


def _formal_null_values(dataset_id):
    """Per-permutation pooled OOF AUCs recorded by the formal run, if present."""
    path = CHECKPOINTS / f"{dataset_id}.json.gz"
    if not path.exists():
        return {}
    d = json.load(gzip.open(path))
    values = {}
    for block in d.get("permutation_blocks") or []:          # list of 5-permutation blocks
        result = block.get("result") or {}
        idx = result.get("permutation_indices") or block.get("indices") or []
        for method, metrics in (result.get("methods") or {}).items():
            for metric in ("pre_split_oof_auc", "fold_contained_oof_auc"):
                for i, v in zip(idx, metrics.get(metric, []), strict=True):
                    values[(method, metric, int(i))] = float(v)
    return values


def mode_nullcheck(dataset_id, platform, entry, seed_tag):
    from benchmark_v4.protocol import build_global_preprocessing_workspace, materialize_permuted_labels

    config, ds, X, y = _load(dataset_id, platform, entry)
    cache = build_global_preprocessing_workspace(X, config, transform_kind=ds.transform_kind)
    formal = _formal_null_values(dataset_id)
    out = {"dataset_id": dataset_id, "status": "ok", "permutations": {}}
    for index in NULL_INDICES:
        y_p, unit = materialize_permuted_labels(y, config, dataset_id, index, groups=ds.groups)
        plan, ws, sc = _scenario(X, y_p, config, dataset_id, ds, cache, f"permutation:{index}")
        two = _two_arm_summary(sc, y.size, config.outer_repeats, f"{dataset_id}.perm{index}.")
        extra = extra_arms(y_p, plan, ws, sc, config, dataset_id, seed_tag, ("swap", "global_prep", "pre_tuned"))
        per = {}
        for m in two:
            match = {metric: (abs(two[m][metric] - formal[(m, metric, index)]) < 1e-12
                              if (m, metric, index) in formal else None)
                     for metric in ("pre_split_oof_auc", "fold_contained_oof_auc")}
            per[m] = {**two[m], **extra[m], "reproduces_formal": match}
        out["permutations"][str(index)] = {"unit": unit, "methods": per}
    return out


MODES = {"gradient": mode_gradient, "holdout": mode_holdout, "pretuned": mode_pretuned,
         "nullcheck": mode_nullcheck}


def _run(args):
    mode, dataset_id, platform, entry, seed_tag = args
    t0 = time.time()
    try:
        res = MODES[mode](dataset_id, platform, entry, seed_tag)
    except Exception as exc:                                  # recorded, never swallowed
        res = {"dataset_id": dataset_id, "status": "error", "error": repr(exc),
               "traceback": traceback.format_exc(limit=6)}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def select_ids(mode: str) -> list[str]:
    flow = pd.read_csv(TABLES / "dataset_eligibility_flow.csv", low_memory=False)
    analysed = flow[flow.analysis_eligible == True]           # noqa: E712
    primary = analysed[analysed.primary_null_eligible == True].dataset_id.astype(str).tolist()  # noqa: E712
    if mode == "pretuned":
        return analysed.dataset_id.astype(str).tolist()
    if mode == "gradient":
        return primary
    if mode == "nullcheck":
        rng = np.random.default_rng(20260928)
        return sorted(rng.choice(primary, size=NULL_SUBSET, replace=False).tolist())
    audit = pd.read_csv(TABLES / "subset_audit.csv", low_memory=False)
    keep = audit[(audit.n_observations >= 100) & (audit.minority_observations >= 30)].dataset_id.astype(str)
    return [i for i in analysed.dataset_id.astype(str) if i in set(keep)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Additional benchmark analyses")
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed-tag", default="primary")
    args = parser.parse_args()

    from benchmark_v4 import raw_loaders

    code = verify_formal_code()
    if not all(code.values()):
        raise RuntimeError(f"formal code mismatch: {[k for k, v in code.items() if not v]}")
    ids = select_ids(args.mode)[: args.limit] if args.limit else select_ids(args.mode)
    out_dir = RESULTS / f"v2_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (out_dir / f"{i}.json").exists()]
    resolved = raw_loaders.entries_for_whitelist(todo) if todo else []
    jobs = [(args.mode, i, p, dict(e), args.seed_tag) for i, (p, e) in zip(todo, resolved, strict=True)]
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
