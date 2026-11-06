"""Reserve analyses B and C: multiclass classification and regression (exploratory; not in the manuscript).

B, multiclass. (i) Real labels: the 11 MicrobiomeHD studies with healthy controls and two disease groups of at
least 16 samples each, at OTU level with the formal microbiome preprocessing; observed labels and 100 permutations.
(ii) Null labels on real data: for each of the 100 datasets of the null subset (Table S3), 20 vectors of balanced
random three-class labels (each class one third of the samples, or of the participants where participants are
recorded, as in 20 of the datasets; a participant's samples then share one label). Methods: top-q (ANOVA F), RFE (one-vs-rest
linear SVM) and PLS-DA (one-hot response); HT-sPLS and OPLS-DA are defined for one response and are left out.
Classifier: multinomial L2 logistic regression (C = 1). Score: macro-averaged one-vs-rest AUC (chance 0.5).

C, regression. (i) Real outcomes: age and body-mass index recorded in MicrobiomeHD studies, for every outcome with
at least 60 valid values in a study (16 outcome-study pairs; age 0-120 years, BMI 10-80 kg/m2), all samples with
a valid value whatever their disease state, at OTU level; observed outcome and 100 permutations. (ii) Null
outcomes on real data: for each dataset of the null subset, 20 vectors of independent standard normal values (one
value per participant where participants are recorded).
Methods: top-q (univariate regression F), RFE (ridge, alpha = 1/(2C), C = 0.1), and PLS, HT-sPLS and OPLS
regression (the benchmark's PLS-DA, HT-sPLS and OPLS-DA steps given the continuous response). Regressor: ridge
regression (alpha = 1). Score: out-of-fold R2 (at most about 0 without signal).

Plans: 5 outer folds x 3 repeats with 3 inner folds (the formal sizes), stratified on the class for multiclass,
group-aware where a study records participants or families; with few groups the number of folds is reduced to the
number of groups (per class when stratified) and seeds are re-derived until every class is in every fold, as the
formal plan does for binary labels. Tuning, the two protocols, the seed contexts, the
formal fold preprocessing and the pooled out-of-fold score follow the binary benchmark (common.evaluate_generic).
Label vectors are evaluated in blocks so that one unit can use several workers; every vector's seed depends only
on the unit and its index, so the blocking does not affect any result.

Outputs: results/other_tasks_{part}.json for part in real_multiclass, real_regression, null_subset.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import common as C

MULTICLASS = {
    "cdi_schubert": ("H", "CDI", "nonCDI"),
    "crc_baxter": ("H", "CRC", "nonCRC"),
    "crc_zackular": ("H", "CRC", "nonCRC"),
    "ibd_alm": ("H", "CD", "UC"),
    "ibd_engstrand_maxee": ("H", "CD", "UC"),
    "ibd_huttenhower": ("H", "CD", "UC"),
    "mhe_zhang": ("H", "CIRR", "MHE"),
    "nash_ob_baker": ("H", "NASH", "nonNASH-OB"),
    "ob_goodrich": ("H", "OW", "OB"),
    "ob_gordon_2008_v2": ("H", "OW", "OB"),
    "ra_littman": ("H", "RA", "PSA"),
}
REGRESSION = (
    ("cdi_schubert", "age", "age"), ("crc_baxter", "Age_s", "age"), ("crc_baxter", "BMI_s", "bmi"),
    ("crc_zeller", "Age (years)", "age"), ("crc_zeller", "BMI (kg/m²)", "bmi"),
    ("hiv_noguerajulian", "host_age_s", "age"), ("hiv_noguerajulian", "host_body_mass_index_s", "bmi"),
    ("ibd_alm", "age", "age"), ("ibd_engstrand_maxee", "Age", "age"), ("ibd_huttenhower", "age", "age"),
    ("nash_ob_baker", "age", "age"), ("ob_goodrich", "age", "age"), ("ob_goodrich", "body_mass_index", "bmi"),
    ("ob_ross", "age_at_visit", "age"), ("ob_ross", "BMI", "bmi"), ("t1d_alkanani", "Age", "age"),
)
VALID = {"age": (0.0, 120.0), "bmi": (10.0, 80.0)}
N_PERM, N_NULL, MIN_CLASS, MIN_N = 100, 20, 16, 60
BLOCK = {"real_multiclass": 10, "real_regression": 25, "null_subset": 20}


def _evaluate(task, X, labels, plan_of, unit, cfg, prep, share_plan: bool) -> dict:
    """Both protocols for every method and label vector. The preprocessing cache is shared by all fits that use
    one split plan (all methods of a label vector; all label vectors when the plan does not depend on them)."""
    out = {m: [] for m in C.TASK_METHODS[task]}
    cache = {}
    for tag, yy in labels:
        if not share_plan:
            cache = {}
        folds = plan_of(yy)
        for m in C.TASK_METHODS[task]:
            r = C.evaluate_generic(X, yy, folds, C.make_step(task, m, cfg), preprocessor=prep, task=task,
                                   seeds=C.V.formal_seeds(cfg, unit, m), cache=cache)
            out[m].append({"labels": tag, "pre_split": r["pre_split"], "fold_contained": r["fold_contained"]})
    return out


def _vectors(y, groups, unit, cfg, indices):
    return [("observed", y) if i == "observed" else (i, C.permuted(y, groups, unit, i, cfg)) for i in indices]


def real_multiclass(study_name: str, indices: list) -> dict:
    cfg = C.formal_config()
    study = C.load_study(study_name)
    classes = MULTICLASS[study_name]
    mask = np.isin(study.state, classes)
    y = np.array([classes.index(s) for s in study.state[mask]])
    counts = np.bincount(y, minlength=len(classes))
    if counts.min() < MIN_CLASS:
        raise ValueError(f"{study_name}: a class has fewer than {MIN_CLASS} samples")
    X, groups = study.rows(mask)
    unit = f"multiclass|{study_name}"
    methods = _evaluate("multiclass", X, _vectors(y, groups, unit, cfg, indices),
                        lambda yy: C.generic_plan(yy, groups, unit, cfg, stratify=True), unit, cfg,
                        C.preprocessor_for(cfg, "microbiome_counts"), share_plan=False)
    return {"key": study_name, "study": study_name, "classes": list(classes), "class_counts": counts.tolist(),
            "n": int(y.size), "p": int(X.shape[1]), "groups": None if groups is None else int(np.unique(groups).size),
            "methods": methods}


def real_regression(study_name: str, column: str, kind: str, indices: list) -> dict:
    cfg = C.formal_config()
    study = C.load_study(study_name)
    value = pd.to_numeric(study.metadata[column], errors="coerce").to_numpy(dtype=float)
    low, high = VALID[kind]
    mask = np.isfinite(value) & (value >= low) & (value <= high)
    if mask.sum() < MIN_N:
        raise ValueError(f"{study_name}/{column}: fewer than {MIN_N} valid values")
    y = value[mask]
    X, groups = study.rows(mask)
    unit = f"regression|{study_name}|{column}"
    folds = C.generic_plan(y, groups, unit, cfg, stratify=False)          # depends on n and groups only
    methods = _evaluate("regression", X, _vectors(y, groups, unit, cfg, indices), lambda yy: folds, unit, cfg,
                        C.preprocessor_for(cfg, "microbiome_counts"), share_plan=True)
    return {"key": f"{study_name}/{column}", "study": study_name, "outcome": kind, "column": column, "n": int(y.size),
            "p": int(X.shape[1]), "mean": float(y.mean()), "sd": float(y.std(ddof=1)),
            "groups": None if groups is None else int(np.unique(groups).size), "methods": methods}


def null_labels(task: str, n: int, groups, rng) -> np.ndarray:
    """Balanced random three-class labels or standard normal outcomes, independent of the features. With
    participants recorded (20 of the 100 datasets), a participant's samples share one value, as the formal null
    permuted these datasets between participants."""
    if groups is None:
        return rng.permutation(np.arange(n) % 3) if task == "multiclass" else rng.standard_normal(n)
    unique, inverse = np.unique(groups, return_inverse=True)
    per_group = rng.permutation(np.arange(unique.size) % 3) if task == "multiclass" else rng.standard_normal(unique.size)
    return per_group[inverse]


def null_subset(dataset_id: str, platform: str, entry: dict, task: str, indices: list) -> dict:
    cfg, ds, X, y_binary = C.V._load(dataset_id, platform, entry)
    groups = None if ds.groups is None else np.asarray(ds.groups, dtype=object)
    if groups is not None and np.unique(groups).size == groups.size:
        groups = None
    n = y_binary.size
    unit = f"null_{task}|{dataset_id}"
    labels = [(i, null_labels(task, n, groups, np.random.default_rng(C.derive_seed(cfg.seed, unit, "null_labels", i))))
              for i in indices]
    if task == "multiclass":
        plan_of, share = (lambda yy: C.generic_plan(yy, groups, unit, cfg, stratify=True)), False
    else:
        folds = C.generic_plan(labels[0][1], groups, unit, cfg, stratify=False)
        plan_of, share = (lambda yy: folds), True
    methods = _evaluate(task, X, labels, plan_of, unit, cfg, C.preprocessor_for(cfg, ds.transform_kind), share)
    return {"key": f"{dataset_id}/{task}", "dataset_id": dataset_id, "task": task, "n": int(n), "p": int(X.shape[1]),
            "groups": None if groups is None else int(np.unique(groups).size), "methods": methods}


def _blocks(indices: list, size: int) -> list[list]:
    return [indices[i:i + size] for i in range(0, len(indices), size)]


def _jobs(part: str, args) -> list[tuple]:
    size = args.block or BLOCK[part]
    real = _blocks(["observed"] + list(range(args.permutations)), size)
    if part == "real_multiclass":
        return [(real_multiclass, (s, b), s) for b in real for s in MULTICLASS]     # blocks interleaved across units
    if part == "real_regression":
        return [(real_regression, (s, c, k, b), f"{s}/{c}") for b in real for s, c, k in REGRESSION]
    from benchmark_v4 import raw_loaders
    ids = sorted(pd.read_csv(C.ROOT / "analyses" / "results" / "nullcheck_per_dataset.csv").dataset_id.unique())
    assert len(ids) == 100
    resolved = raw_loaders.entries_for_whitelist(ids)
    null = _blocks(list(range(args.nulls)), size)
    return [(null_subset, (i, p, dict(e), task, b), f"{i}/{task}") for i, (p, e) in zip(ids, resolved, strict=True)
            for task in ("multiclass", "regression") for b in null]


def _call(job):
    fn, fn_args, name = job
    t0 = time.time()
    rec = fn(*fn_args)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def merge(parts: list[dict], expected: list) -> list[dict]:
    """Join the blocks of each unit; every expected label index must be present exactly once."""
    by_key = defaultdict(list)
    for p in parts:
        by_key[p["key"]].append(p)
    out = []
    order = {tag: k for k, tag in enumerate(expected)}
    for key, blocks in sorted(by_key.items()):
        rec = {k: v for k, v in blocks[0].items() if k not in ("methods", "seconds", "block_id")}
        rec["seconds"] = round(sum(b["seconds"] for b in blocks), 1)
        rec["methods"] = {}
        for m in blocks[0]["methods"]:
            entries = sorted((e for b in blocks for e in b["methods"][m]), key=lambda e: order[e["labels"]])
            if [e["labels"] for e in entries] != expected:
                raise RuntimeError(f"{key}/{m}: label vectors missing or duplicated")
            rec["methods"][m] = ({"observed": entries[0], "permuted": entries[1:]} if expected[0] == "observed"
                                 else {"null": entries})
        out.append(rec)
    return out


def _block_id(job) -> str:
    fn, fn_args, name = job
    block = fn_args[-1]
    return f"{name}|{block[0]}|{block[-1]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("real_multiclass", "real_regression", "null_subset"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--permutations", type=int, default=N_PERM)
    ap.add_argument("--nulls", type=int, default=N_NULL)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--pilot", nargs="*", default=None, help="unit names to run alone")
    args = ap.parse_args()
    C.check_formal_code()
    jobs = _jobs(args.part, args)
    if args.pilot is not None:
        jobs = [j for j in jobs if j[2] in set(args.pilot)]
    # finished blocks are appended to a checkpoint as they complete, and a rerun skips them
    name = f"other_tasks_{args.part}" + ("_pilot" if args.pilot is not None else "")
    checkpoint = C.RESULTS / f"{name}.partial.jsonl"
    parts = [json.loads(line) for line in checkpoint.read_text().splitlines()] if checkpoint.exists() else []
    done = {p["block_id"] for p in parts}
    todo = [j for j in jobs if _block_id(j) not in done]
    print(f"{args.part}: {len(jobs)} jobs, {len(jobs) - len(todo)} already in the checkpoint", flush=True)
    errors, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, checkpoint.open("a") as sink:
        futures = {pool.submit(_call, j): j for j in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            try:
                rec = fut.result()
                rec["block_id"] = _block_id(job)
                parts.append(rec)
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
            except Exception as exc:                       # recorded, never silently dropped
                errors.append({"job": job[2], "error": repr(exc)})
            print(f"  {k}/{len(todo)} {job[2]}  {(time.time() - t0) / 60:.1f} min", flush=True)
    expected = (list(range(args.nulls)) if args.part == "null_subset"
                else ["observed"] + list(range(args.permutations)))
    if errors:
        print(f"{len(errors)} failed blocks; rerun to retry them", flush=True)
        C.write_json(C.RESULTS / f"{name}.errors.json", errors)
        return 1
    C.write_json(C.RESULTS / f"{name}.json", merge(parts, expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
