"""Sensitivity of the formal results to duplicated or near-identical samples (exploratory; duplicate_scan.py).

Duplicates inflate an estimate only when two copies with the same label fall on either side of a split. For every
dataset in which the scan found exact or near-identical pairs (r >= 0.99) with the same label that no recorded
participant group already joined, those pairs are merged into duplicate groups, joined with any recorded groups, and
the formal pipelines are rerun with these groups and nothing else changed:
- cross-validation on the observed labels (formal plan generator, fold preprocessing, candidates, classifier and
  seed schedule; the five methods share the label-free fold preprocessing, which leaves every value unchanged);
- the formal hold-out procedure (control_arm/run_v2_arms.mode_holdout) where the dataset is in the hold-out set.
Passing groups makes the formal generator use its group-aware path, which assigns folds (and the hold-out)
differently even when every sample is its own group. Each rerun is therefore paired with a control on the same path
in which only the recorded groups join samples (each sample its own group otherwise); the duplicate effect is the
difference between the two. Where groups were recorded, the control is the formal analysis itself and must
reproduce it. Permuted labels are not rerun: after permutation the two copies' labels are independent, so duplicates carry no label
information into the null. Pairs with different labels are reported by the scan but not grouped (they can only
lower an estimate). A dataset whose groups leave a class with fewer than two groups cannot be split and is listed.
Output: results/duplicate_sensitivity.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "reserve"))
import validate_against_formal as V                       # noqa: E402

OUT = HERE / "results"
CONTROL_ARM = ROOT.parent / "nm_brief_communication" / "control_arm"


def duplicate_groups(n: int, recorded, pairs: list[dict]) -> np.ndarray:
    """Recorded participant groups joined with the same-label exact or near-identical pairs."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    if recorded is not None:
        first: dict = {}
        for i, g in enumerate(recorded):
            union(i, first.setdefault(g, i))
    for p in pairs:
        if p["same_label"] and not p["handled"]:
            union(p["a"], p["b"])
    return np.array([f"g{find(i)}" for i in range(n)], dtype=object)


def control_groups(n: int, recorded) -> np.ndarray:
    """The same group-aware path without the duplicate groups: recorded groups, otherwise one group per sample."""
    return duplicate_groups(n, recorded, [])


def needs_rerun(record: dict) -> bool:
    return any(p["same_label"] and not p["handled"] for p in record["pairs"])


def cv_job(job) -> dict:
    import common as C
    from nm_v4.splitting import materialize_plan
    dataset_id, platform, entry, pairs, variant = job
    t0 = time.time()
    config, ds, X, y = V._load(dataset_id, platform, entry)
    groups = duplicate_groups(y.size, ds.groups, pairs) if variant == "duplicates" else control_groups(y.size, ds.groups)
    prep = lambda: V.FormalPreprocessor(config, ds.transform_kind)       # noqa: E731
    try:
        plan = materialize_plan(y, config, dataset_id, groups=groups)
    except ValueError as exc:                              # e.g. a class left with fewer than two groups
        return {"kind": "cv", "variant": variant, "dataset_id": dataset_id, "status": "not splittable",
                "reason": str(exc)}
    folds = V.formal_folds(plan)
    cache, methods = {}, {}
    for m in V.METHODS:
        r = C.evaluate_generic(X, y, folds, C.make_step("binary", m, config), preprocessor=prep, task="binary",
                               seeds=V.formal_seeds(config, dataset_id, m), cache=cache)
        methods[m] = {"pre_split": r["pre_split"], "fold_contained": r["fold_contained"]}
    return {"kind": "cv", "variant": variant, "dataset_id": dataset_id, "status": "ok", "n": int(y.size),
            "groups": int(np.unique(groups).size), "outer_folds": int(len(folds) // config.outer_repeats),
            "methods": methods, "seconds": round(time.time() - t0, 1)}


def holdout_job(job) -> dict:
    dataset_id, platform, entry, pairs, variant = job
    t0 = time.time()
    sys.path.insert(0, str(CONTROL_ARM))
    import run_v2_arms as A
    formal_load = A._load

    def load(*a):
        config, ds, X, y = formal_load(*a)
        g = duplicate_groups(y.size, ds.groups, pairs) if variant == "duplicates" else control_groups(y.size, ds.groups)
        return config, dataclasses.replace(ds, groups=g), X, y

    A._load = load
    try:
        out = A.mode_holdout(dataset_id, platform, entry, "duplicate_sensitivity")
    except Exception as exc:
        return {"kind": "holdout", "variant": variant, "dataset_id": dataset_id, "status": "failed",
                "reason": repr(exc)}
    finally:
        A._load = formal_load
    return {"kind": "holdout", "variant": variant, "dataset_id": dataset_id, "status": "ok", "result": out,
            "seconds": round(time.time() - t0, 1)}


def formal(dataset_id: str) -> dict:
    obs = V.checkpoint(dataset_id)["observed"]["methods"]
    out = {m: {"pre_split": obs[m]["observed"]["pre_split_oof_auc"],
               "fold_contained": obs[m]["observed"]["fold_contained_oof_auc"]} for m in V.METHODS}
    path = CONTROL_ARM / "results" / "v2_holdout" / f"{dataset_id}.json"
    if path.exists():
        h = json.loads(path.read_text())
        if h.get("status") == "ok":
            for m in V.METHODS:
                out[m]["holdout"] = {k: h["methods"][m][k] for k in
                                     ("heldout_auc", "pre_split_oof_auc", "fold_contained_oof_auc")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not all(V.verify_formal_code().values()):
        raise RuntimeError("formal code mismatch")
    from nm_v4 import raw_loaders
    scan = {r["dataset_id"]: r for r in json.loads((OUT / "duplicate_scan.json").read_text())["records"]}
    ids = sorted(d for d, r in scan.items() if needs_rerun(r))
    resolved = dict(zip(ids, raw_loaders.entries_for_whitelist(ids), strict=True))
    holdout_ids = [d for d in ids if (CONTROL_ARM / "results" / "v2_holdout" / f"{d}.json").exists()
                   and json.loads((CONTROL_ARM / "results" / "v2_holdout" / f"{d}.json").read_text()).get("status") == "ok"]
    jobs = [(fn, d, v) for v in ("duplicates", "control") for fn, pool_ids in ((cv_job, ids), (holdout_job, holdout_ids))
            for d in pool_ids]
    print(f"{len(ids)} datasets with same-label duplicates not already grouped; {len(holdout_ids)} in the hold-out set",
          flush=True)
    parts, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fn, (d, resolved[d][0], dict(resolved[d][1]), scan[d]["pairs"], v)): (fn.__name__, d, v)
                   for fn, d, v in jobs}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                parts.append(fut.result())
            except Exception as exc:                       # recorded, never silently dropped
                parts.append({"kind": "cv" if futures[fut][0] == "cv_job" else "holdout", "variant": futures[fut][2],
                              "dataset_id": futures[fut][1], "status": "failed", "reason": repr(exc)})
            if k % 20 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    result = {}
    for d in ids:
        def get(kind, variant):
            return next((p for p in parts if (p["kind"], p["variant"], p["dataset_id"]) == (kind, variant, d)), None)

        result[d] = {"scan": {k: scan[d][k] for k in ("n", "exact_pairs", "near_pairs", "median_nearest_r",
                                                        "groups_recorded")},
                     "same_label_pairs": sum(p["same_label"] and not p["handled"] for p in scan[d]["pairs"]),
                     "formal": formal(d), "cv": get("cv", "duplicates"), "cv_control": get("cv", "control"),
                     "holdout": get("holdout", "duplicates"), "holdout_control": get("holdout", "control")}
    (OUT / "duplicate_sensitivity.json").write_text(json.dumps(result, indent=1))
    bad = [d for d, r in result.items() if any(r[k] is not None and r[k]["status"] == "failed"
                                                for k in ("cv", "cv_control", "holdout", "holdout_control"))]
    print(f"written; {len(bad)} datasets with a failed job: {bad}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
