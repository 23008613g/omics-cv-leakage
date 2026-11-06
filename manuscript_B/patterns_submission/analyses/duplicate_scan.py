"""Scan of the 1,086 analysed datasets for duplicated or near-identical samples (quality check; exploratory).

Motivation: two twin studies turned out to contain related samples that the formal loader did not group
(family_sensitivity.py). Unrecorded technical replicates or repeated profiles would let a held-out sample's twin sit
in the training folds and inflate even fold-contained estimates.

Criteria, fixed before the scan:
- exact duplicates: identical rows of the analysed matrix;
- near-identical pairs: Pearson correlation of at least 0.99 between two samples' profiles after the formal
  label-free preprocessing fitted on all samples (log transform, filters, at most 5,000 features, z-scoring of each
  feature, so that the correlation compares deviations from the feature means rather than the shared profile);
  pairs at 0.95 or more are counted as a looser reference.
A pair is already handled if both samples belong to the same recorded participant group. For each dataset the
script records the pairs, whether they share the label and whether they were handled, and joins the unhandled
exact and near-identical pairs into duplicate groups (connected components). Output: results/duplicate_scan.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "validation"))
import validate_against_formal as V                       # noqa: E402
from cv_folds_all import analysed                         # noqa: E402

OUT = HERE / "results"
NEAR, LOOSE = 0.99, 0.95


def components(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in {i for p in pairs for i in p}:
        groups.setdefault(find(i), []).append(i)
    return sorted(sorted(g) for g in groups.values())


def scan(job) -> dict:
    dataset_id, platform, entry = job
    t0 = time.time()
    config, ds, X, y = V._load(dataset_id, platform, entry)
    X = np.asarray(X, dtype=float)
    # exact duplicates of the analysed matrix
    _, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    exact = [(int(a), int(b)) for k in np.flatnonzero(counts > 1)
             for idx in [np.flatnonzero(inverse == k)] for i, a in enumerate(idx) for b in idx[i + 1:]]
    # correlations of the preprocessed profiles
    Z = V.FormalPreprocessor(config, ds.transform_kind).fit(X).transform(X)
    Z = Z - Z.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Z, axis=1)
    norms[norms == 0] = np.inf
    R = (Z / norms[:, None]) @ (Z / norms[:, None]).T
    iu = np.triu_indices(y.size, 1)
    r = R[iu]
    groups = None if ds.groups is None else np.asarray(ds.groups, dtype=object)

    exact_set = set(exact)

    def describe(pairs):
        out = []
        for a, b in pairs:
            out.append({"a": a, "b": b, "r": float(R[a, b]), "exact": (a, b) in exact_set,
                        "same_label": bool(y[a] == y[b]),
                        "handled": bool(groups is not None and groups[a] == groups[b])})
        return out

    near = [(int(a), int(b)) for a, b, v in zip(*iu, r) if v >= NEAR]
    flagged = sorted(set(exact) | set(near))
    pairs = describe(flagged)
    unhandled = [(p["a"], p["b"]) for p in pairs if not p["handled"]]
    comps = components(y.size, unhandled)
    nn = np.where(np.eye(y.size, dtype=bool), -np.inf, R).max(axis=1)
    return {"dataset_id": dataset_id, "n": int(y.size), "p": int(X.shape[1]), "p_used": int(Z.shape[1]),
            "groups_recorded": groups is not None, "exact_pairs": len(exact), "near_pairs": len(near),
            "loose_pairs": int((r >= LOOSE).sum()), "max_r": float(r.max()) if r.size else None,
            "median_nearest_r": float(np.median(nn)), "pairs": pairs,
            "unhandled_pairs": len(unhandled), "unhandled_cross_label": sum(not p["same_label"] for p in pairs
                                                                            if not p["handled"]),
            "duplicate_groups": comps, "sample_ids": [str(ds.sample_ids[i]) for c in comps for i in c],
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    if not all(V.verify_formal_code().values()):
        raise RuntimeError("formal code mismatch")
    from nm_v4 import raw_loaders
    ids = list(analysed().dataset_id)
    assert len(ids) == 1086
    jobs = [(i, p, dict(e)) for i, (p, e) in zip(ids, raw_loaders.entries_for_whitelist(ids), strict=True)]
    records, errors, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan, j): j[0] for j in jobs}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                records.append(fut.result())
            except Exception as exc:                       # recorded, never silently dropped
                errors.append({"dataset_id": futures[fut], "error": repr(exc)})
            if k % 100 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  {(time.time() - t0) / 60:.1f} min  errors {len(errors)}", flush=True)
    records.sort(key=lambda r: r["dataset_id"])
    (OUT / "duplicate_scan.json").write_text(json.dumps({"near": NEAR, "loose": LOOSE, "records": records,
                                                         "errors": errors}, indent=1))
    flagged = [r for r in records if r["unhandled_pairs"]]
    print(f"datasets with unhandled exact or near-identical pairs: {len(flagged)}; errors: {len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
