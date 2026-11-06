"""Does presplit-audit reproduce the formal benchmark?

Three checks on a random sample of exchangeable datasets (seed and size fixed
below, before any result was seen):

  exact     the tool's own pipeline code, given the formal split plan, the
            formal fold preprocessor and the formal seed schedule, should
            reproduce the formal pooled out-of-fold AUCs of both arms for all
            five methods on observed labels (only RFE uses its seeds);
  null      on a subset, the same for formal permutations 0-4 (formal permuted
            label vectors and their split plans);
  own       the tool with its own random folds (default splitter, seed 0) and the
            formal fold preprocessor, to show how much the estimates move with
            fold assignment alone.

The formal evaluator code is hash-checked against the formal-run manifest before
anything runs. Results go to ``results/``; nothing in the formal deposit changes.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANUSCRIPT_B = HERE.parents[1]
CONTROL_ARM = MANUSCRIPT_B / "nm_brief_communication" / "control_arm"
DEPOSIT = Path(os.environ.get("PRESPLIT_FORMAL_DEPOSIT", MANUSCRIPT_B / "Formal_Public_Deposit_v3"))
sys.path.insert(0, str(CONTROL_ARM))
sys.path.insert(0, str(HERE.parent / "tool" / "src"))

from run_control_arm import verify_formal_code            # noqa: E402  (binds nm_v4)
from run_v2_arms import _load, select_ids                  # noqa: E402

from presplit_audit import (HTSPLS, OPLSDA, PLSDA, RFE, OuterFold, TopQ, default_splitter,  # noqa: E402
                            evaluate_protocols)

SAMPLE_SEED, SAMPLE_SIZE, NULL_SUBSET, NULL_INDICES = 20261018, 100, 20, range(5)
METHODS = ("top_k", "rfe", "spls", "pls_da", "opls_da")             # formal keys


def make_step(method: str, config):
    """The tool's built-in step for a formal method, with the formal settings."""
    return {"top_k": lambda: TopQ(), "rfe": lambda: RFE(svc_c=config.rfe_svc_c, max_iter=config.rfe_max_iter),
            "spls": lambda: HTSPLS(), "pls_da": lambda: PLSDA(max_iter=config.max_iter),
            "opls_da": lambda: OPLSDA()}[method]()


def formal_seeds(config, dataset_id: str, method: str):
    """The formal seed schedule (nm_v4.protocol._evaluate_scenario) behind the tool's fitting contexts."""
    from nm_v4.splitting import derive_seed
    names = {"inner_representation": lambda o, i: (o, method, "fold_contained", i, "representation"),
             "inner_classifier": lambda o, i, c: (o, method, "fold_contained", i, c),
             "pre_split_classifier": lambda o: (o, method, "pre_split_final"),
             "outer_representation": lambda o: (o, method, "outer_representation"),
             "fold_contained_classifier": lambda o: (o, method, "contained_final")}

    def schedule(kind, *idx):
        if kind == "global":
            return derive_seed(config.seed, dataset_id, method, "global")
        return derive_seed(config.seed, dataset_id, *names[kind](*idx))

    return schedule
OUT = HERE / "results"


class FormalPreprocessor:
    """The formal fold preprocessor behind the tool's fit/transform interface."""

    def __init__(self, config, transform_kind):
        self.config, self.transform_kind = config, transform_kind

    def fit(self, X):
        from nm_v4.protocol import fit_preprocessor
        self.fitted = fit_preprocessor(X, self.config, transform_kind=self.transform_kind)
        return self

    def transform(self, X):
        return self.fitted.transform(X)


def formal_folds(plan) -> list[OuterFold]:
    out = []
    for s in plan.outer_splits:
        train, test = np.asarray(s.train_indices, int), np.asarray(s.test_indices, int)
        inner = tuple((train[np.asarray(i.train_indices, int)], train[np.asarray(i.test_indices, int)])
                      for i in s.inner_splits)
        out.append(OuterFold(s.repeat, train, test, inner))
    return out


def checkpoint(dataset_id: str) -> dict:
    return json.load(gzip.open(DEPOSIT / "checkpoints" / f"{dataset_id}.json.gz"))


def run_one(job) -> dict:
    from nm_v4.protocol import materialize_permuted_labels
    from nm_v4.splitting import materialize_plan

    dataset_id, platform, entry, with_null = job
    t0 = time.time()
    config, ds, X, y = _load(dataset_id, platform, entry)
    prep = lambda: FormalPreprocessor(config, ds.transform_kind)       # noqa: E731
    ck = checkpoint(dataset_id)
    rec = {"dataset_id": dataset_id, "n": int(y.size), "p": int(X.shape[1]), "methods": {}}
    plan = materialize_plan(y, config, dataset_id, groups=ds.groups)
    own = default_splitter(random_state=0)(y, ds.groups)
    for m in METHODS:
        tm = time.time()
        seeds = formal_seeds(config, dataset_id, m)
        formal = ck["observed"]["methods"][m]["observed"]
        exact = evaluate_protocols(X, y, formal_folds(plan), make_step(m, config), preprocessor=prep, seeds=seeds)
        mine = evaluate_protocols(X, y, own, make_step(m, config), preprocessor=prep)
        r = {"formal_pre_split": formal["pre_split_oof_auc"], "formal_fold_contained": formal["fold_contained_oof_auc"],
             "exact_pre_split": exact["pre_split_auc"], "exact_fold_contained": exact["fold_contained_auc"],
             "own_pre_split": mine["pre_split_auc"], "own_fold_contained": mine["fold_contained_auc"]}
        if with_null:
            block = ck["permutation_blocks"][0]["result"]
            assert list(block["permutation_indices"])[: len(NULL_INDICES)] == list(NULL_INDICES)
            fm = block["methods"][m]
            nulls = []
            for i in NULL_INDICES:
                yp, _ = materialize_permuted_labels(y, config, dataset_id, i, groups=ds.groups)
                pp = materialize_plan(yp, config, dataset_id, groups=ds.groups)
                e = evaluate_protocols(X, yp, formal_folds(pp), make_step(m, config), preprocessor=prep, seeds=seeds)
                nulls.append({"index": i, "formal_pre_split": fm["pre_split_oof_auc"][i],
                              "formal_fold_contained": fm["fold_contained_oof_auc"][i],
                              "exact_pre_split": e["pre_split_auc"], "exact_fold_contained": e["fold_contained_auc"]})
            r["null"] = nulls
        r["seconds"] = round(time.time() - tm, 1)
        rec["methods"][m] = r
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--pilot", type=int, default=0, help="run only the first N sampled datasets (with null check)")
    args = ap.parse_args()
    code = verify_formal_code()
    if not all(code.values()):
        raise RuntimeError(f"formal code mismatch: {[k for k, v in code.items() if not v]}")
    from nm_v4 import raw_loaders

    primary = select_ids("gradient")                      # the exchangeable datasets
    ids = sorted(np.random.default_rng(SAMPLE_SEED).choice(primary, size=SAMPLE_SIZE, replace=False).tolist())
    null_ids = set(ids[:: SAMPLE_SIZE // NULL_SUBSET][:NULL_SUBSET])
    OUT.mkdir(exist_ok=True)
    (OUT / "sample.json").write_text(json.dumps({"seed": SAMPLE_SEED, "n_exchangeable": len(primary),
                                                 "ids": ids, "null_ids": sorted(null_ids)}, indent=1))
    resolved = raw_loaders.entries_for_whitelist(ids)
    jobs = [(i, p, dict(e), i in null_ids) for i, (p, e) in zip(ids, resolved, strict=True)]
    if args.pilot:
        jobs = [(i, p, e, True) for i, p, e, _ in jobs[: args.pilot]]
    print(f"formal code {sum(code.values())}/{len(code)} verified; {len(jobs)} datasets, "
          f"{len(null_ids)} with null check", flush=True)
    records, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, j): j[0] for j in jobs}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                records.append(fut.result())
            except Exception as exc:                          # recorded, never silently dropped
                records.append({"dataset_id": futures[fut], "error": repr(exc)})
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    records.sort(key=lambda r: r["dataset_id"])
    name = f"validation_pilot_{args.pilot}.json" if args.pilot else "validation_records.json"
    (OUT / name).write_text(json.dumps(records, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
