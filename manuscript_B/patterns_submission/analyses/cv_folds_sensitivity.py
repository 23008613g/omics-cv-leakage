"""Does the pre-split inflation depend on the number of outer folds? (Sensitivity analysis, exploratory.)

The formal benchmark used stratified 5-fold cross-validation repeated three times. Here the same 100-dataset
null subset (the one used for the arm checks, Table S3) is re-evaluated with 10 outer folds, keeping everything
else fixed: three repeats, three inner folds, the formal fold preprocessing, the formal seed schedule, the formal
candidate grids and classifier, and the formal permuted label vectors 0-19 (the permutation seeds do not depend on
the fold count, so both schemes see the same labels). The pipelines are run by presplit-audit, which reproduces the
formal benchmark exactly given the same inputs (Note S5); the 5-fold values are the formal ones.

For each dataset and method: observed pre-split and fold-contained AUCs, and the mean of each over the 20
permutations, under 10 outer folds. Results go to ``results/cv_folds_*``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "validation"))
import validate_against_formal as V                       # noqa: E402  (binds nm_v4, the tool and the loaders)

OUT = HERE / "results"
N_PERM, OUTER = 20, 10
METHODS = V.METHODS


def subset_ids() -> list[str]:
    return sorted(pd.read_csv(OUT / "nullcheck_per_dataset.csv").dataset_id.unique())


def run_one(job) -> dict:
    from nm_v4.protocol import materialize_permuted_labels
    from nm_v4.splitting import materialize_plan

    dataset_id, platform, entry = job
    t0 = time.time()
    config, ds, X, y = V._load(dataset_id, platform, entry)
    config10 = replace(config, outer_splits=OUTER)
    prep = lambda: V.FormalPreprocessor(config, ds.transform_kind)       # noqa: E731
    plan = materialize_plan(y, config10, dataset_id, groups=ds.groups)
    folds = V.formal_folds(plan)
    labels = [y] + [materialize_permuted_labels(y, config, dataset_id, i, groups=ds.groups)[0] for i in range(N_PERM)]
    plans = [folds] + [V.formal_folds(materialize_plan(yp, config10, dataset_id, groups=ds.groups))
                       for yp in labels[1:]]
    rec = {"dataset_id": dataset_id, "n": int(y.size), "p": int(X.shape[1]),
           "outer_folds": int(len(folds) // config10.outer_repeats), "methods": {}}
    for m in METHODS:
        seeds = V.formal_seeds(config, dataset_id, m)
        vals = [V.evaluate_protocols(X, yy, ff, V.make_step(m, config), preprocessor=prep, seeds=seeds)
                for yy, ff in zip(labels, plans)]
        rec["methods"][m] = {"pre_split": vals[0]["pre_split_auc"], "fold_contained": vals[0]["fold_contained_auc"],
                             "null_pre_split": float(np.mean([v["pre_split_auc"] for v in vals[1:]])),
                             "null_fold_contained": float(np.mean([v["fold_contained_auc"] for v in vals[1:]]))}
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--pilot", type=int, default=0)
    args = ap.parse_args()
    code = V.verify_formal_code()
    if not all(code.values()):
        raise RuntimeError("formal code mismatch")
    from nm_v4 import raw_loaders
    ids = subset_ids()
    assert len(ids) == 100
    if args.pilot:
        ids = ids[: args.pilot]
    resolved = raw_loaders.entries_for_whitelist(ids)
    jobs = [(i, p, dict(e)) for i, (p, e) in zip(ids, resolved, strict=True)]
    records, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, j): j[0] for j in jobs}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                records.append(fut.result())
            except Exception as exc:                       # recorded, never silently dropped
                records.append({"dataset_id": futures[fut], "error": repr(exc)})
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    records.sort(key=lambda r: r["dataset_id"])
    name = f"cv_folds_pilot_{args.pilot}.json" if args.pilot else "cv_folds_records.json"
    (OUT / name).write_text(json.dumps(records, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
