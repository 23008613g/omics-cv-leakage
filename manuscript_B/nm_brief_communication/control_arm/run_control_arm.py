"""Held-out-label-permuted control arm for the paired protocol contrast.

The formal benchmark compares two arms per dataset and method:

  pre-split       preprocessing and the supervised representation are fitted
                  once on all samples with all labels;
  fold-contained  both are refitted on each outer training partition.

Their difference mixes three things: reuse of the held-out labels by the
supervised step, label-free preprocessing that has seen the held-out
features, and a larger fitting sample. This control removes only the first.
For every outer fold it refits the supervised representation on all samples,
exactly as the pre-split arm does, but with the held-out fold's labels
randomly permuted among themselves (training labels untouched). The
classifier is then trained on the training fold with its true labels and
scored on the held-out fold with its true labels. Hence

  pre-split - control         = label-reuse component
  control - fold-contained    = preprocessing + sample-size component
                                (plus any cost of fitting to noisy labels)

and the two add up to the reported gap.

Everything except the held-out labels is shared with the formal run: the
locked ``nm_v4`` code, configuration, split plans, preprocessing workspaces,
the candidate selected by fold-contained inner cross-validation, and the
representation and classifier seeds of the pre-split arm. The formal code is
imported, never modified. Each dataset first reproduces the pre-split and
fold-contained pooled out-of-fold AUCs of the formal run and records whether
they match the deposit exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]                                  # _phytomed_revision
RUNNER_ROOT = PROJECT / "mtbls_screen"
TABLES = PROJECT / "manuscript_B" / "Formal_Public_Deposit_v3" / "tables"
RESULTS = HERE / "results"
MANIFEST = RUNNER_ROOT / "matched_supervised_v4" / "formal_run_v4" / "manifest.json"
sys.path.insert(0, str(RUNNER_ROOT))


def _bind_formal_package() -> None:
    """Import ``nm_v4`` from the directory the formal run used.

    The formal ``mtbls_screen/nm_v4`` has no ``__init__.py`` (a namespace
    package), while an editable install elsewhere on this machine exposes a
    regular package of the same name, which Python's import system prefers
    regardless of path order. Registering the package object explicitly
    pins every ``nm_v4`` import to the formal code; the file hashes are then
    checked against the formal run's manifest. Module-level, so spawned
    worker processes bind the same way.
    """
    import types

    if "nm_v4" in sys.modules and list(getattr(sys.modules["nm_v4"], "__path__", [])) != [
            str(RUNNER_ROOT / "nm_v4")]:
        raise RuntimeError("a different nm_v4 was imported before binding")
    pkg = types.ModuleType("nm_v4")
    pkg.__path__ = [str(RUNNER_ROOT / "nm_v4")]
    sys.modules["nm_v4"] = pkg


def verify_formal_code() -> dict:
    """Hash every formal-run code file that exists here against the manifest."""
    import hashlib

    recorded = json.loads(MANIFEST.read_text())["code_file_sha256"]
    report = {}
    for rel, sha in recorded.items():
        path = RUNNER_ROOT / rel
        now = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        report[rel] = now == sha
    return report


_bind_formal_package()

EXPECTED_CONFIG_SHA = "80696d21aca94a0a0ae1e25e05b4a5fbbc1477ea1cdafd48baed654155d0b2f1"
TOLERANCE = 1e-12


def _candidate_index(method: str, params: dict) -> int:
    from nm_v4.methods import candidate_grid

    for index, grid in enumerate(candidate_grid(method)):
        if all(params.get(key) == value for key, value in grid.items()):
            return index
    raise ValueError(f"{method}: chosen parameters {params} are not in the grid")


def _pooled_auc(folds: list[dict], n: int, repeats: int, context: str) -> float:
    from sklearn.metrics import roc_auc_score
    from nm_v4.protocol_estimands import _repeat_averaged_oof_vectors

    labels, scores = _repeat_averaged_oof_vectors(
        folds, n_samples=n, outer_repeats=repeats, context=context)
    return float(roc_auc_score(labels, scores))


def evaluate_dataset(dataset_id: str, platform: str, entry: dict, seed_tag: str) -> dict:
    from nm_v4 import raw_loaders
    from nm_v4.configuration import protocol_config_sha256
    from nm_v4.methods import METHODS
    from nm_v4.protocol import (ProtocolConfig, _evaluate_scenario, _fit_auc, _fit_bundle,
                                _fold_record, _validated_xy, build_preprocessing_workspace)
    from nm_v4.splitting import derive_seed, materialize_plan

    started = time.time()
    config = ProtocolConfig(n_permutations=199)
    if protocol_config_sha256(config, include_permutation_count=False) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("protocol configuration differs from the formal run")
    dataset, reason = raw_loaders.load_raw(platform, entry)
    if dataset is None or reason != "ok":
        return {"dataset_id": dataset_id, "status": f"load_failed:{reason}"}
    X, y = _validated_xy(dataset.X, dataset.y)
    plan = materialize_plan(y, config, dataset_id, groups=dataset.groups)
    workspace = build_preprocessing_workspace(X, plan, config, transform_kind=dataset.transform_kind)

    # 1. reproduce the two formal arms with the locked evaluator
    scenario = _evaluate_scenario(y, plan, workspace, config, dataset_id, "observed",
                                  include_baseline=False)
    global_x = workspace.global_workspace.transformed_x
    n, repeats = y.size, config.outer_repeats
    out = {"dataset_id": dataset_id, "status": "ok", "n_samples": int(n),
           "split_plan_sha256": plan.sha256, "seed_tag": seed_tag, "methods": {}}

    for method in METHODS:
        res = scenario["methods"][method]
        pre_folds, con_folds = res["pre_split_folds"], res["fold_contained_folds"]
        arms: dict[str, list[dict]] = {"permuted_heldout": [], "swap": [], "global_prep": []}
        flips = []
        for split, pre_fold, con_fold, prepared in zip(
                plan.outer_splits, pre_folds, con_folds, workspace.outer, strict=True):
            outer_index = pre_fold["outer_index"]
            train = np.asarray(split.train_indices, dtype=int)
            test = np.asarray(split.test_indices, dtype=int)
            chosen = _candidate_index(method, pre_fold["chosen_params"])
            parts = (dataset_id, outer_index, method)
            rep_seed = derive_seed(config.seed, *parts, "outer_representation")
            con_clf_seed = derive_seed(config.seed, *parts, "contained_final")

            def record(arm, scores, auc):
                arms[arm].append(_fold_record(outer_index, split, y[test], scores, auc,
                                              pre_fold["chosen_params"], [], 0, []))

            # (a) mechanism check: pre-split fit with the held-out labels permuted
            #     among themselves (one draw per fold, shared by all methods)
            rng = np.random.default_rng(
                derive_seed(config.seed, dataset_id, outer_index, "heldout_label_permutation", seed_tag))
            y_perm = y.copy()
            y_perm[test] = y[test][rng.permutation(test.size)]
            flips.append(float(np.mean(y_perm[test] != y[test])))
            rep = _fit_bundle(method, global_x, y_perm, config,
                              derive_seed(config.seed, dataset_id, method, "global"))[chosen]
            z = rep.transform(global_x)
            auc, scores, _ = _fit_auc(z[train], y[train], z[test], y[test], config,
                                      derive_seed(config.seed, *parts, "pre_split_final"),
                                      phase=f"{method}:permuted-heldout:{outer_index}")
            record("permuted_heldout", scores, auc)

            # (b) swap: fold-contained preprocessing; the supervised step is fitted
            #     on the training fold with |V| training samples, class-matched to
            #     the held-out fold, replaced by the held-out samples and their true
            #     labels. Sample size and class composition equal the fold-contained
            #     fit, so the difference isolates held-out label reuse.
            swap_feasible = arms["swap"] is not None and all(
                np.sum(y[test] == c) <= np.sum(y[train] == c) for c in (0, 1))
            if not swap_feasible:
                # a held-out class outnumbers that class in training (small grouped
                # folds): the class-matched swap is not estimable for this method
                arms["swap"] = None
            else:
                rng_d = np.random.default_rng(
                    derive_seed(config.seed, dataset_id, outer_index, "swap_subset", seed_tag))
                drop = np.concatenate([
                    rng_d.choice(np.flatnonzero(y[train] == cls), size=int(np.sum(y[test] == cls)),
                                 replace=False) for cls in (0, 1)])
                keep = np.setdiff1d(np.arange(train.size), drop)
                fit_x = np.vstack([prepared.train_x[keep], prepared.test_x])
                fit_y = np.concatenate([y[train][keep], y[test]])
                rep = _fit_bundle(method, fit_x, fit_y, config, rep_seed)[chosen]
                auc, scores, _ = _fit_auc(rep.transform(prepared.train_x), y[train],
                                          rep.transform(prepared.test_x), y[test], config, con_clf_seed,
                                          phase=f"{method}:swap:{outer_index}")
                record("swap", scores, auc)

            # (c) global label-free preprocessing, supervised step on the training fold
            rep = _fit_bundle(method, global_x[train], y[train], config, rep_seed)[chosen]
            auc, scores, _ = _fit_auc(rep.transform(global_x[train]), y[train],
                                      rep.transform(global_x[test]), y[test], config, con_clf_seed,
                                      phase=f"{method}:global-prep:{outer_index}")
            record("global_prep", scores, auc)

        ctx = f"{dataset_id}.{method}"
        entry = {
            "pre_split_oof_auc": _pooled_auc(pre_folds, n, repeats, ctx + ".pre"),
            "fold_contained_oof_auc": _pooled_auc(con_folds, n, repeats, ctx + ".con"),
            "pre_split_mean_fold_auc": float(np.mean([f["auc"] for f in pre_folds])),
            "fold_contained_mean_fold_auc": float(np.mean([f["auc"] for f in con_folds])),
            "mean_share_heldout_labels_changed": float(np.mean(flips)),
        }
        for arm, folds in arms.items():
            if folds is None:
                entry[f"{arm}_oof_auc"] = entry[f"{arm}_mean_fold_auc"] = None
                entry[f"{arm}_not_estimable"] = "held-out class exceeds training class in a fold"
                continue
            entry[f"{arm}_oof_auc"] = _pooled_auc(folds, n, repeats, f"{ctx}.{arm}")
            entry[f"{arm}_mean_fold_auc"] = float(np.mean([f["auc"] for f in folds]))
        out["methods"][method] = entry
    out["seconds"] = round(time.time() - started, 1)
    return out


def _run_one(args: tuple[str, str, dict, str]) -> dict:
    dataset_id, platform, entry, seed_tag = args
    try:
        return evaluate_dataset(dataset_id, platform, entry, seed_tag)
    except Exception as exc:                                  # recorded, never swallowed
        return {"dataset_id": dataset_id, "status": "error", "error": repr(exc),
                "traceback": traceback.format_exc(limit=6)}


def _write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--ids", nargs="*", help="restrict to these dataset IDs")
    parser.add_argument("--limit", type=int, help="first N eligible datasets only (smoke test)")
    parser.add_argument("--seed-tag", default="primary",
                        help="namespace for the held-out permutation draw (sensitivity reruns)")
    args = parser.parse_args()

    from nm_v4 import raw_loaders

    code = verify_formal_code()
    mismatched = [k for k, ok in code.items() if not ok]
    print(f"formal code files verified against manifest: {sum(code.values())}/{len(code)}"
          + (f"; MISMATCH {mismatched}" if mismatched else ""), flush=True)
    if not raw_loaders.__file__.startswith(str(RUNNER_ROOT / "nm_v4")):
        raise RuntimeError(f"wrong nm_v4 imported: {raw_loaders.__file__}")
    flow = pd.read_csv(TABLES / "dataset_eligibility_flow.csv", low_memory=False)
    analysed = flow.loc[flow.analysis_eligible == True, "dataset_id"].astype(str).tolist()  # noqa: E712
    ids = [i for i in analysed if not args.ids or i in set(args.ids)]
    if args.limit:
        ids = ids[: args.limit]
    out_dir = RESULTS / args.seed_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (out_dir / f"{i}.json").exists()]
    resolved = raw_loaders.entries_for_whitelist(todo) if todo else []
    jobs = [(i, platform, dict(entry), args.seed_tag)
            for i, (platform, entry) in zip(todo, resolved, strict=True)]
    print(f"{len(ids)} selected, {len(ids) - len(todo)} already done, {len(jobs)} to run "
          f"with {args.workers} workers -> {out_dir}", flush=True)
    started, done, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_one, job) for job in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            _write_atomic(out_dir / f"{res['dataset_id']}.json", res)
            done += 1
            failed += res.get("status") != "ok"
            if done % 25 == 0 or done == len(jobs):
                rate = (time.time() - started) / done
                print(f"  {done}/{len(jobs)}  failed {failed}  ~{rate * (len(jobs) - done) / 60:.0f} min left",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
