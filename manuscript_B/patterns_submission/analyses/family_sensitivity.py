"""Family-aware sensitivity analysis of the two MicrobiomeHD obesity datasets with family records (exploratory).

ob_goodrich (786 analysed samples from 451 families) and ob_gordon_2008_v2 (257 samples from 54 families) contain
twins, their mothers and repeated samples, but the formal loader treated their samples as exchangeable. Here the
formal pipelines are rerun with splits that keep families together; nothing else changes.

- Cross-validation: the formal plan generator with the family codes as groups (5 outer folds x 3 repeats, 3 inner
  folds; fewer folds only if a class spans fewer families), the formal fold preprocessing, candidates, classifier and
  seed schedule, run by presplit-audit, which reproduces the formal pipelines exactly given the same inputs (Note S5).
  Labels: the observed ones and the formal permuted label vectors 0-19. These are the sample-level permutations of
  the formal run, so each value pairs with a formal value that differs only in the split. (The formal rule for groups
  whose labels vary, permutation within groups, would leave families with one label untouched and keep their signal.)
- Hold-out: the formal hold-out procedure (control_arm/run_v2_arms.mode_holdout) with the family codes as groups, so
  that the untouched 25% holds whole families. Its group-free run is repeated first and must reproduce the formal
  hold-out values.

Family codes come from the MicrobiomeHD metadata (reserve/common.load_study). Output: results/family_sensitivity.json.
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
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "reserve"))
import validate_against_formal as V                       # noqa: E402

OUT = HERE / "results"
CONTROL_ARM = ROOT.parent / "benchmark_analyses" / "control_arm"
DATASETS = ("ob_goodrich", "ob_gordon_2008_v2")
N_PERM = 20
BLOCKS = (("observed", 0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12, 13), (14, 15, 16, 17, 18, 19))


def families(dataset_id: str, ds) -> np.ndarray:
    import common as C
    study = C.load_study(dataset_id)
    codes = pd.Series(np.asarray(study.groups, dtype=object), index=study.sample_ids).reindex(list(ds.sample_ids))
    if codes.isna().any():
        raise ValueError(f"{dataset_id}: {int(codes.isna().sum())} samples without a family code")
    return codes.to_numpy(dtype=object)


def cv_block(job) -> dict:
    from benchmark_v4.protocol import materialize_permuted_labels
    from benchmark_v4.splitting import materialize_plan

    dataset_id, platform, entry, method, block = job
    t0 = time.time()
    config, ds, X, y = V._load(dataset_id, platform, entry)
    assert ds.groups is None, "the formal run treated these samples as exchangeable"
    fam = families(dataset_id, ds)
    prep = lambda: V.FormalPreprocessor(config, ds.transform_kind)       # noqa: E731
    seeds = V.formal_seeds(config, dataset_id, method)
    values, folds_used = [], set()
    for tag in BLOCKS[block]:
        yy = y if tag == "observed" else materialize_permuted_labels(y, config, dataset_id, tag, groups=None)[0]
        plan = materialize_plan(yy, config, dataset_id, groups=fam)
        folds = V.formal_folds(plan)
        for f in folds:
            assert not set(fam[f.train]) & set(fam[f.test]), "a family on both sides of a split"
        folds_used.add(len(folds) // config.outer_repeats)
        r = V.evaluate_protocols(X, yy, folds, V.make_step(method, config), preprocessor=prep, seeds=seeds)
        values.append({"labels": tag, "pre_split": r["pre_split_auc"], "fold_contained": r["fold_contained_auc"]})
    return {"kind": "cv", "dataset_id": dataset_id, "method": method, "block": block, "n": int(y.size),
            "families": int(np.unique(fam).size), "outer_folds": sorted(folds_used), "values": values,
            "seconds": round(time.time() - t0, 1)}


def holdout(job) -> dict:
    """mode_holdout of the formal run; with ``family`` the family codes replace the (absent) formal groups."""
    dataset_id, platform, entry, variant = job
    t0 = time.time()
    sys.path.insert(0, str(CONTROL_ARM))
    import run_v2_arms as A
    formal_load = A._load

    def load(*a):
        config, ds, X, y = formal_load(*a)
        assert ds.groups is None
        return config, dataclasses.replace(ds, groups=families(dataset_id, ds)), X, y

    if variant == "family":
        A._load = load
    try:
        out = A.mode_holdout(dataset_id, platform, entry, "family_sensitivity")
    finally:
        A._load = formal_load
    return {"kind": "holdout", "variant": variant, "dataset_id": dataset_id, "result": out,
            "seconds": round(time.time() - t0, 1)}


def formal_values(dataset_id: str) -> dict:
    ck = V.checkpoint(dataset_id)
    out = {}
    for m in V.METHODS:
        obs = ck["observed"]["methods"][m]["observed"]
        perm = {}
        for block in ck["permutation_blocks"]:
            res = block["result"]
            for i, a, b in zip(res["permutation_indices"], res["methods"][m]["pre_split_oof_auc"],
                               res["methods"][m]["fold_contained_oof_auc"], strict=True):
                perm[int(i)] = (a, b)
        out[m] = {"pre_split": obs["pre_split_oof_auc"], "fold_contained": obs["fold_contained_oof_auc"],
                  "permuted": [{"labels": i, "pre_split": perm[i][0], "fold_contained": perm[i][1]}
                               for i in range(N_PERM)]}
    return out


def summarize(parts: list[dict]) -> dict:
    result = {}
    for d in DATASETS:
        # formal hold-out record (Data S1 rounds the last digit of some values)
        fh = json.loads((CONTROL_ARM / "results" / "v2_holdout" / f"{d}.json").read_text())["methods"]
        cv = [p for p in parts if p["kind"] == "cv" and p["dataset_id"] == d]
        ho = {p["variant"]: p["result"] for p in parts if p["kind"] == "holdout" and p["dataset_id"] == d}
        formal = formal_values(d)
        row = {"n": cv[0]["n"], "families": cv[0]["families"],
               "outer_folds": sorted({f for p in cv for f in p["outer_folds"]}),
               "group_holdout": ho["family"]["group_holdout"], "n_test_family": ho["family"]["n_test"],
               "methods": {}}
        for m in V.METHODS:
            vals = {v["labels"]: v for p in cv if p["method"] == m for v in p["values"]}
            assert sorted(vals, key=str) == sorted(["observed"] + list(range(N_PERM)), key=str)
            fam_perm = [vals[i] for i in range(N_PERM)]
            fo = formal[m]
            hf, hx = ho["formal"]["methods"][m], ho["family"]["methods"][m]
            row["methods"][m] = {
                "family": {"pre_split": vals["observed"]["pre_split"],
                           "fold_contained": vals["observed"]["fold_contained"],
                           "null_pre_split": float(np.mean([v["pre_split"] for v in fam_perm])),
                           "null_fold_contained": float(np.mean([v["fold_contained"] for v in fam_perm])),
                           "holdout_heldout": hx["heldout_auc"]},
                "formal": {"pre_split": fo["pre_split"], "fold_contained": fo["fold_contained"],
                           "null_pre_split": float(np.mean([v["pre_split"] for v in fo["permuted"]])),
                           "null_fold_contained": float(np.mean([v["fold_contained"] for v in fo["permuted"]])),
                           "holdout_heldout": fh[m]["heldout_auc"],
                           "holdout_cv_pre_split": fh[m]["pre_split_oof_auc"],
                           "holdout_cv_fold_contained": fh[m]["fold_contained_oof_auc"]},
                "holdout_cv_family": {k: hx[k] for k in hx if k.endswith("_auc") and k != "heldout_auc"},
                "formal_holdout_reproduced": all(hf[k] == fh[m][k] for k in
                                                 ("heldout_auc", "pre_split_oof_auc", "fold_contained_oof_auc")),
                "family_permuted": fam_perm}
        result[d] = row
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    code = V.verify_formal_code()
    if not all(code.values()):
        raise RuntimeError("formal code mismatch")
    from benchmark_v4 import raw_loaders
    resolved = dict(zip(DATASETS, raw_loaders.entries_for_whitelist(list(DATASETS)), strict=True))
    partial = OUT / "family_sensitivity.partial.jsonl"
    parts = [json.loads(line) for line in partial.read_text().splitlines()] if partial.exists() else []
    done = {(p["kind"], p["dataset_id"], p.get("method"), p.get("block"), p.get("variant")) for p in parts}
    jobs = [(holdout, (d, resolved[d][0], dict(resolved[d][1]), v)) for d in DATASETS for v in ("formal", "family")]
    jobs += [(cv_block, (d, resolved[d][0], dict(resolved[d][1]), m, b)) for d in DATASETS for m in V.METHODS
             for b in range(len(BLOCKS))]

    def key(fn, a):
        return ("holdout", a[0], None, None, a[3]) if fn is holdout else ("cv", a[0], a[3], a[4], None)

    jobs = [(fn, a) for fn, a in jobs if key(fn, a) not in done]
    print(f"{len(jobs)} jobs to run, {len(done)} in the checkpoint", flush=True)
    errors, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, partial.open("a") as sink:
        futures = {pool.submit(fn, a): (fn.__name__, a[0], a[3:]) for fn, a in jobs}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
                parts.append(rec)
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
            except Exception as exc:                       # recorded, never silently dropped
                errors.append({"job": repr(futures[fut]), "error": repr(exc)})
            print(f"  {k}/{len(jobs)} {futures[fut]}  {(time.time() - t0) / 60:.1f} min", flush=True)
    if errors:
        (OUT / "family_sensitivity.errors.json").write_text(json.dumps(errors, indent=1))
        print(f"{len(errors)} failed jobs; rerun to retry them", flush=True)
        return 1
    summary = summarize(parts)
    (OUT / "family_sensitivity.json").write_text(json.dumps(summary, indent=1))
    for d, r in summary.items():
        print(d, "families", r["families"], "folds", r["outer_folds"], "family hold-out", r["group_holdout"])
        for m, v in r["methods"].items():
            f, o = v["family"], v["formal"]
            print(f"  {m:8s} fc {o['fold_contained']:.3f}->{f['fold_contained']:.3f}  pre {o['pre_split']:.3f}->"
                  f"{f['pre_split']:.3f}  null pre {o['null_pre_split']:.3f}->{f['null_pre_split']:.3f}  null fc "
                  f"{o['null_fold_contained']:.3f}->{f['null_fold_contained']:.3f}  heldout {o['holdout_heldout']:.3f}"
                  f"->{f['holdout_heldout']:.3f}  formal hold-out reproduced {v['formal_holdout_reproduced']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
